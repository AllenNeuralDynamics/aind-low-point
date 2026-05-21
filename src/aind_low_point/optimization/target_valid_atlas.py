"""Target-valid visibility atlas (Stage A + Stage B).

Design: ``dev/target_valid_atlas_design.md``.

The atlas reports, per ``(probe, hole)``, the set of insertion poses
that thread the hole **and** place the recording bank usefully relative
to the probe's target. First implementation uses ``apex = T`` (probe
target); the data model carries ``source_apex_offset`` so a future
near-target apex grid plugs in without breaking the API.

Two-stage build:

  **Stage A** — Centerline feasibility region.
    Sample points inside the bottom section's oval (typically smallest
    cross-section, used as the rejection envelope). For each sample
    ``p``, the ray from ``T`` through ``p`` is back-projected to every
    other section plane; the resulting hit point must lie inside that
    section's oval. Accepted samples define centerline directions
    ``d = (p − T) / ‖p − T‖`` — convert to rig ``(AP, ML)``.

  **Stage B** — Spin × per-shank check.
    For each accepted Stage A direction and each spin in the spin grid,
    compute the K shank lines (parallel to ``d``, offsets ``R · δ_k``).
    Test every ``(shank, section)`` for ellipse membership. Accept the
    pose when all ``K × S`` tests pass.

  **K = 1 shortcut**: shank line ≡ centerline; spin doesn't affect
  threading; Stage B is skipped and each accepted Stage A sample
  becomes one atlas anchor at ``(AP, ML, spin=0.0)``.

The Stage B JIT kernel mirrors ``visibility_atlas._shank_in_section_jax``
but is invoked per accepted Stage A sample rather than per top-oval
grid point.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.atlas import Atlas, AtlasEntry, PoseAnchor
from aind_low_point.optimization.geometry import cap_basis
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.recording import get_recording_geometry
from aind_low_point.optimization.sdf_jax import arc_angles_to_rotation
from aind_low_point.optimization.visibility_atlas import (
    _sample_top_ellipse_points,
    _shank_in_section_jax,
    _vec_to_ap_ml_jax,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_ellipse_perimeter(section, n: int) -> NDArray:
    """Sample ``n`` points on the boundary of the section's ellipse, in 3D.

    Used to capture rays tangent to this section — the *boundary* of the
    centerline feasibility cone. Manual-quality plans often live at the
    threading boundary, so explicit perimeter coverage matters.
    """
    e1, e2 = cap_basis(np.asarray(section.axis, dtype=np.float64))
    a = float(section.a)
    b = float(section.b)
    theta = float(section.theta)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    center = np.asarray(section.center, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts: list[NDArray] = []
    for t in angles:
        u_local = a * float(np.cos(t))
        v_local = b * float(np.sin(t))
        u = c * u_local - s * v_local
        v = s * u_local + c * v_local
        pts.append(center + u * e1 + v * e2)
    return np.array(pts, dtype=np.float64)


def _pack_section_np(section) -> dict:
    """NumPy-only pack of a hole section (Stage A is CPU/NumPy)."""
    e1, e2 = cap_basis(np.asarray(section.axis, dtype=np.float64))
    return dict(
        center=np.asarray(section.center, dtype=np.float64),
        axis=np.asarray(section.axis, dtype=np.float64),
        e1=e1.astype(np.float64),
        e2=e2.astype(np.float64),
        a=float(section.a),
        b=float(section.b),
        theta=float(section.theta),
    )


def _probe_centroid_local(probe) -> NDArray:
    tips = np.asarray(probe.shank_tips_local, dtype=np.float64)
    if tips.shape[0] == 0:
        geom = get_recording_geometry(probe.kind)
        return np.array([0.0, 0.0, geom.active_center_mm], dtype=np.float64)
    geom = get_recording_geometry(probe.kind)
    return np.array(
        [
            float(tips[:, 0].mean()),
            float(tips[:, 1].mean()),
            float(geom.active_center_mm),
        ],
        dtype=np.float64,
    )


def _line_in_oval(
    line_anchor: NDArray,
    line_dir: NDArray,
    section: dict,
    *,
    g_max: float = 1.0,
) -> tuple[bool, float]:
    """Return ``(inside, g)`` for a line vs a section's ellipse.

    The line ``line_anchor + t * line_dir`` is intersected with the
    section plane; the hit point is tested against the section's oval.
    ``g = (u/a)^2 + (v/b)^2``; ``inside`` is ``g <= g_max``.

    Set ``g_max = 1 + ε`` to accept tangent points (the
    perimeter-sampling case where the source section's own ellipse
    is hit at g=1).
    """
    denom = float(np.dot(line_dir, section["axis"]))
    if abs(denom) < 1e-9:
        return False, np.inf
    rel = section["center"] - line_anchor
    t = float(np.dot(rel, section["axis"])) / denom
    P = line_anchor + t * line_dir
    diff = P - section["center"]
    u = float(np.dot(diff, section["e1"]))
    v = float(np.dot(diff, section["e2"]))
    c, s = float(np.cos(section["theta"])), float(np.sin(section["theta"]))
    u_l = c * u + s * v
    v_l = -s * u + c * v
    a, b = float(section["a"]), float(section["b"])
    g = (u_l / a) ** 2 + (v_l / b) ** 2
    return g <= g_max, g


def _pick_param_section(sections_packed: list[dict]) -> int:
    """Pick the smallest-area section as the parameterization plane.

    The smallest projected ellipse is the natural rejection envelope.
    Area scales as ``π * a * b``; we ignore the constant and just
    compare ``a * b`` across sections.
    """
    areas = [s["a"] * s["b"] for s in sections_packed]
    return int(np.argmin(areas))


def _vec_to_ap_ml(d_lps: NDArray) -> tuple[float, float]:
    """LPS direction → (AP, ML) in degrees. Matches the JAX helper."""
    d = np.asarray(d_lps, dtype=np.float64)
    n = d / np.linalg.norm(d)
    sign = -1.0 if n[2] < 0 else 1.0
    n = n * sign
    d_ras = np.array([-n[0], -n[1], n[2]])
    rx = float(np.arcsin(d_ras[1]))
    ry = float(np.arctan2(d_ras[0], d_ras[2]))
    return float(np.degrees(rx)), float(np.degrees(ry))


# ---------------------------------------------------------------------------
# Stage B JIT kernel: per (centerline direction, spin) -> per-shank pass/fail
# ---------------------------------------------------------------------------


def _build_stage_b_check(sections_packed_jnp: tuple[dict, ...]):
    """Closure that checks K shanks × N sections at a target-aligned pose."""
    def check(target, centerline_pt, spin_deg, tips_local, centroid_local):
        D_lps = centerline_pt - target
        D_lps_n = D_lps / jnp.linalg.norm(D_lps)
        D_ras = jnp.array([-D_lps_n[0], -D_lps_n[1], D_lps_n[2]])
        ap_deg, ml_deg = _vec_to_ap_ml_jax(D_ras)
        R = arc_angles_to_rotation(ap_deg, ml_deg, spin_deg)
        pose_tip = target - R @ centroid_local
        shaft_dir = R @ jnp.array([0.0, 0.0, 1.0])

        shanks_world = tips_local @ R.T + pose_tip

        valid = jnp.bool_(True)
        for sec in sections_packed_jnp:
            def per_shank(sh, sec=sec):
                return _shank_in_section_jax(
                    sh, shaft_dir,
                    sec["center"], sec["axis"], sec["e1"], sec["e2"],
                    sec["a"], sec["b"], sec["theta"],
                    oval_slack=0.0,   # strict; Stage A already filtered
                )
            masks = jax.vmap(per_shank)(shanks_world)
            valid = valid & jnp.all(masks)
        return valid, ap_deg, ml_deg

    return check


# ---------------------------------------------------------------------------
# Atlas builder
# ---------------------------------------------------------------------------


def build_target_valid_atlas(
    probes,
    holes: Sequence[Hole],
    *,
    n_interior_samples: int = 96,
    n_perimeter_samples: int = 48,
    n_spin: int = 72,
    spin_range_deg: tuple[float, float] = (-180.0, 180.0),
    apex_offsets_RA: Sequence[tuple[float, float]] = ((0.0, 0.0),),
    boundary_tol: float = 0.02,
    verbose: bool = False,
) -> Atlas:
    """Build the target-valid atlas for all (probe, hole) pairs.

    Parameters
    ----------
    probes : list[ProbeStaticInfo]
        Probes with ``.name``, ``.kind``, ``.target_LPS``, and
        ``.shank_tips_local``.
    holes : sequence of Hole
        Holes; each has ``.id`` and a sequence of sections.
    n_interior_samples : int
        Number of points sampled inside the smallest section's oval —
        covers the interior of the centerline feasibility region.
    n_perimeter_samples : int
        Number of perimeter points per section. Rays through these
        points are tangent to the source section (g=1) and accepted
        when they thread every other section. Captures the *boundary*
        of the feasibility cone where manual-quality plans live.
    n_spin : int
        Spin grid resolution for Stage B. Ignored when ``K == 1``.
    spin_range_deg : tuple
        Inclusive-exclusive spin range; default ``(-180, 180)``.
    apex_offsets_RA : sequence of (Δ_R, Δ_A) in mm
        Near-target apex offsets. Default ``((0, 0),)`` — apex = T.
        Each tuple produces its own Stage A region; anchors carry the
        offset for provenance.
    boundary_tol : float
        Tolerance on the source-section's own membership for perimeter
        rays (these are tangent so g≈1 by construction). Allows a small
        slack ``g ≤ 1 + boundary_tol`` for numerical safety on perimeter
        samples. Default 0.02.
    verbose : bool
        Print per-(probe, hole) progress.
    """
    probe_names = tuple(p.name for p in probes)
    hole_ids = tuple(h.id for h in holes)
    entries: dict[tuple[str, int], AtlasEntry] = {}

    probe_centroid_local = {p.name: _probe_centroid_local(p) for p in probes}
    probe_tips_local = {
        p.name: np.asarray(p.shank_tips_local, dtype=np.float32) for p in probes
    }
    probe_target = {
        p.name: np.asarray(p.target_LPS, dtype=np.float64) for p in probes
    }
    probe_n_shanks = {p.name: int(np.asarray(p.shank_tips_local).shape[0]) for p in probes}

    spins_np = np.linspace(
        spin_range_deg[0], spin_range_deg[1], n_spin, endpoint=False, dtype=np.float32
    )
    spins_jnp = jnp.asarray(spins_np)

    t0 = time.perf_counter()
    for hole in holes:
        sections_np = [_pack_section_np(s) for s in hole.sections]
        smallest_idx = _pick_param_section(sections_np)

        # JAX-packed sections for Stage B
        sections_jnp = tuple(
            {
                "center": jnp.asarray(s["center"].astype(np.float32)),
                "axis": jnp.asarray(s["axis"].astype(np.float32)),
                "e1": jnp.asarray(s["e1"].astype(np.float32)),
                "e2": jnp.asarray(s["e2"].astype(np.float32)),
                "a": jnp.float32(s["a"]),
                "b": jnp.float32(s["b"]),
                "theta": jnp.float32(s["theta"]),
            }
            for s in sections_np
        )

        # Build Stage A sample list: (3D point, source-section index,
        # is_perimeter). Interior samples come from the smallest section
        # only (provides general interior coverage). Perimeter samples
        # come from EVERY section — each captures tangent rays on the
        # boundary of the feasibility cone, where manual configs sit.
        param_samples: list[tuple[NDArray, int, bool]] = []

        # Interior of the smallest section
        class _Shim:
            axis = sections_np[smallest_idx]["axis"]
            a = float(sections_np[smallest_idx]["a"])
            b = float(sections_np[smallest_idx]["b"])
            theta = float(sections_np[smallest_idx]["theta"])
            center = sections_np[smallest_idx]["center"]
        for pt in _sample_top_ellipse_points(_Shim(), n_interior_samples):
            param_samples.append((pt, smallest_idx, False))

        # Perimeter of every section
        for sec_idx, _ in enumerate(sections_np):
            for pt in _sample_ellipse_perimeter(
                hole.sections[sec_idx], n_perimeter_samples
            ):
                param_samples.append((pt, sec_idx, True))

        if verbose:
            print(f"  [tv-atlas] hole {hole.id}: smallest section = {smallest_idx}, "
                  f"{len(param_samples)} stage-A candidate points "
                  f"(interior={n_interior_samples}, perimeter={n_perimeter_samples} "
                  f"per section × {len(sections_np)} sections)")

        # Build Stage B kernel once per hole (closure on sections)
        stage_b_check = _build_stage_b_check(sections_jnp)
        # vmap over centerline samples × spin
        stage_b_vmap = jax.jit(
            jax.vmap(
                jax.vmap(stage_b_check, in_axes=(None, 0, None, None, None)),
                in_axes=(None, None, 0, None, None),
            )
        )

        for probe in probes:
            name = probe.name
            target = probe_target[name]
            tips_local = probe_tips_local[name]
            centroid_local = probe_centroid_local[name].astype(np.float32)
            K = probe_n_shanks[name]

            all_anchors: list[PoseAnchor] = []

            for (off_R, off_A) in apex_offsets_RA:
                # Apex offset is currently 0, but the API carries it; for
                # future implementations the offset shifts T in the R/A
                # rig directions. Skipped here — apex = T only.
                if off_R != 0.0 or off_A != 0.0:
                    raise NotImplementedError(
                        "Near-target apex offsets not implemented yet; "
                        "leave apex_offsets_RA at the default ((0,0),)."
                    )
                apex = target

                # ----------------- Stage A -----------------
                accepted: list[tuple[NDArray, float, float]] = []
                seen_directions: set[tuple[int, int]] = set()  # dedup
                for p, source_idx, is_perimeter in param_samples:
                    d = p - apex
                    if np.linalg.norm(d) < 1e-9:
                        continue
                    d_n = d / np.linalg.norm(d)
                    # Test every section; the source section gets a slack
                    # for perimeter samples (tangent at g≈1).
                    centerline_passes = True
                    for sec_idx, sec in enumerate(sections_np):
                        if sec_idx == source_idx and is_perimeter:
                            # Source is hit at g=1 by construction
                            g_max = 1.0 + boundary_tol
                        else:
                            g_max = 1.0
                        inside, _g = _line_in_oval(apex, d_n, sec, g_max=g_max)
                        if not inside:
                            centerline_passes = False
                            break
                    if not centerline_passes:
                        continue
                    ap_deg, ml_deg = _vec_to_ap_ml(d_n)
                    # Dedup directions across multi-section perimeter sampling
                    dir_key = (int(round(ap_deg * 4)), int(round(ml_deg * 4)))
                    if dir_key in seen_directions:
                        continue
                    seen_directions.add(dir_key)
                    accepted.append((p.astype(np.float32), ap_deg, ml_deg))

                if not accepted:
                    continue

                if K <= 1:
                    # K = 1 shortcut: emit one anchor per accepted Stage A sample
                    for _, ap_deg, ml_deg in accepted:
                        all_anchors.append(
                            PoseAnchor(
                                ap_deg=ap_deg, ml_deg=ml_deg, spin_deg=0.0,
                                off_R_mm=off_R, off_A_mm=off_A,
                                depth_mm=0.0,
                                threading_max_g=-1.0, target_miss_mm=0.0,
                            )
                        )
                    continue

                # ----------------- Stage B -----------------
                centerline_pts = jnp.asarray(
                    np.stack([a[0] for a in accepted], axis=0)
                )
                target_jnp = jnp.asarray(apex.astype(np.float32))
                tips_jnp = jnp.asarray(tips_local)
                centroid_jnp = jnp.asarray(centroid_local)

                valid_grid, ap_grid, ml_grid = stage_b_vmap(
                    target_jnp, centerline_pts, spins_jnp, tips_jnp, centroid_jnp,
                )
                # valid_grid shape: (n_spin, n_centerlines)
                valid_np = np.asarray(valid_grid)
                ap_np = np.asarray(ap_grid)
                ml_np = np.asarray(ml_grid)
                spin_grid_b = np.broadcast_to(spins_np[:, None], valid_np.shape)

                valid_aps = ap_np[valid_np]
                valid_mls = ml_np[valid_np]
                valid_spins = spin_grid_b[valid_np]
                for ap_v, ml_v, sp_v in zip(valid_aps, valid_mls, valid_spins):
                    all_anchors.append(
                        PoseAnchor(
                            ap_deg=float(ap_v), ml_deg=float(ml_v),
                            spin_deg=float(sp_v),
                            off_R_mm=off_R, off_A_mm=off_A,
                            depth_mm=0.0,
                            threading_max_g=-1.0, target_miss_mm=0.0,
                        )
                    )

            if all_anchors:
                aps = [a.ap_deg for a in all_anchors]
                entries[(name, hole.id)] = AtlasEntry(
                    probe_name=name, hole_id=hole.id,
                    ap_min=float(min(aps)), ap_max=float(max(aps)),
                    anchors=tuple(all_anchors),
                )
            else:
                entries[(name, hole.id)] = AtlasEntry(
                    probe_name=name, hole_id=hole.id,
                    ap_min=None, ap_max=None, anchors=(),
                )

        if verbose:
            valid_count = sum(
                1 for pn in probe_names
                if entries[(pn, hole.id)].ap_min is not None
            )
            print(f"  [tv-atlas] hole {hole.id}: {valid_count}/{len(probes)} probes valid "
                  f"({time.perf_counter() - t0:.2f}s)")
    if verbose:
        for probe in probes:
            valid_hids = [
                hid for hid in hole_ids
                if entries[(probe.name, hid)].ap_min is not None
            ]
            n_anchors_total = sum(
                len(entries[(probe.name, hid)].anchors) for hid in valid_hids
            )
            print(f"  [tv-atlas] {probe.name:>5}: {len(valid_hids)}/{len(hole_ids)} holes "
                  f"({n_anchors_total} anchors total)")
    return Atlas(entries=entries, probe_names=probe_names, hole_ids=hole_ids)
