"""JAX-vmapped visibility atlas for the optimizer's Stage 1.

For each (probe, hole) pair, enumerate candidate top-ellipse
sample points × spin orientations. For each (sample, spin) we
construct the probe pose and test whether **all shanks** thread
**all hole sections**. The set of passing (sample, spin) configurations
defines the achievable AP/ML/spin cone for that (probe, hole).

This replaces the SLSQP-based atlas in :mod:`atlas.py`. Differences:
- **Visibility, not optimisation.** We don't optimise pose; we test
  closed-form geometry. Faster (sub-second build), no local minima.
- **Per-shank threading.** Each of the K shanks must pass through
  each section's ellipse. The earlier atlas only checked the
  shank-row centroid.
- **Spin-aware.** A 4-shank probe at different spins presents
  different patterns to the bore; spin matters for threading.
- **Mesh-target ready.** A target-region mesh adds one vmap axis
  (target_point); the kernel does not change.

The kernel is pure JAX, vmappable along (top_sample, spin) axes.
For our K=7 N=14 problem with n_top=64, n_spin=24 we evaluate
~1.5M plane intersections in one JIT'd batched call.
"""

from __future__ import annotations

import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.atlas import Atlas, AtlasEntry, PoseAnchor
from aind_low_point.optimization.geometry import cap_basis
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.recording import get_recording_geometry
from aind_low_point.optimization.sdf_jax import arc_angles_to_rotation


# ---------------------------------------------------------------------------
# NumPy helpers (pre-compute / pack)
# ---------------------------------------------------------------------------


def _sample_top_ellipse_points(section, n: int) -> NDArray:
    """Sample ~n points covering the **interior** of the section's oval.

    For 4-shank probes the centroid line must land inside a smaller
    "admissible sub-ellipse" so the shanks (offset by up to ~0.4 mm)
    still fit. Boundary-only sampling would put the chord at the edge
    of the bore — shanks offset from the chord would fall outside.

    Sampling pattern: concentric rings + center.
    For ``n=64``: ``n_rings=4`` (r=0.2, 0.5, 0.8, 1.0), ~16 points each.
    """
    e1, e2 = cap_basis(np.asarray(section.axis, dtype=np.float64))
    a = float(section.a)
    b = float(section.b)
    theta = float(section.theta)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    center = np.asarray(section.center, dtype=np.float64)

    n_rings = max(3, int(round(np.sqrt(n / 4))))
    n_theta = max(8, int(round(n / n_rings)))
    radii = np.linspace(0.0, 1.0, n_rings + 1)[1:]  # skip r=0 to avoid duplicates
    angles = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    pts = [center.copy()]  # include centre
    for r in radii:
        for t in angles:
            u_local = r * a * float(np.cos(t))
            v_local = r * b * float(np.sin(t))
            u = c * u_local - s * v_local
            v = s * u_local + c * v_local
            pts.append(center + u * e1 + v * e2)
    return np.array(pts, dtype=np.float64)


def _pack_section(section) -> dict[str, NDArray]:
    e1, e2 = cap_basis(np.asarray(section.axis, dtype=np.float64))
    return dict(
        center=np.asarray(section.center, dtype=np.float32),
        axis=np.asarray(section.axis, dtype=np.float32),
        e1=e1.astype(np.float32),
        e2=e2.astype(np.float32),
        a=float(section.a),
        b=float(section.b),
        theta=float(section.theta),
    )


def _probe_centroid_local(probe) -> NDArray:
    """Centroid of shank tips, in the probe's local frame."""
    tips = np.asarray(probe.shank_tips_local, dtype=np.float64)
    if tips.shape[0] == 0:
        # Degenerate — return active-center fallback
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


# ---------------------------------------------------------------------------
# JAX kernel: visibility check for one (top_sample, spin) candidate
# ---------------------------------------------------------------------------


def _vec_to_ap_ml_jax(v_ras: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX of ``vector_to_arc_angles(invert_AP=True)``. Direction-insensitive."""
    s = jnp.where(v_ras[2] < 0, -1.0, 1.0)
    v = v_ras * s
    n = v / jnp.linalg.norm(v)
    # rx = -arcsin(nv[1]); then invert_AP negates again → rx = +arcsin(nv[1])
    rx = jnp.arcsin(n[1])
    ry = jnp.arctan2(n[0], n[2])
    return jnp.rad2deg(rx), jnp.rad2deg(ry)


def _shank_in_section_jax(
    shank_pos: jnp.ndarray,
    shaft_dir: jnp.ndarray,
    sec_center: jnp.ndarray,
    sec_axis: jnp.ndarray,
    sec_e1: jnp.ndarray,
    sec_e2: jnp.ndarray,
    sec_a: float,
    sec_b: float,
    sec_theta: float,
    oval_slack: float = 0.2,
) -> jnp.ndarray:
    """True if the shank line crosses inside the section's ellipse, with
    a small ``oval_slack`` (default 20%). The slack *expands* the
    effective ellipse: ``g <= (1 + slack)^2``. It compensates for the
    DOFs the visibility atlas doesn't sample (offset_R/A, past_target):
    a config that's marginally outside the strict ellipse at the
    atlas's pinned pose may move inside after polish shifts the probe
    laterally or along the shaft. Without slack the atlas is more
    restrictive than the optimizer it feeds.

    The real cause of missing boundary configs (e.g. manual T12 VM→h7
    at ml=-8) is not slack but unsampled DOFs — the manual sits at
    past_target=-1.5325mm which the chord-anchored pose doesn't see.
    Top+bottom oval sampling (vs current top+target chord) is the
    proper fix; slack is a partial proxy."""
    rel = sec_center - shank_pos
    denom = jnp.dot(shaft_dir, sec_axis)
    safe = jnp.abs(denom) > 1e-9
    denom_safe = jnp.where(safe, denom, 1.0)
    t = jnp.dot(rel, sec_axis) / denom_safe
    P = shank_pos + t * shaft_dir
    diff = P - sec_center
    u = jnp.dot(diff, sec_e1)
    v = jnp.dot(diff, sec_e2)
    c, s = jnp.cos(sec_theta), jnp.sin(sec_theta)
    u_l = c * u + s * v
    v_l = -s * u + c * v
    g = (u_l / sec_a) ** 2 + (v_l / sec_b) ** 2
    return safe & (g <= (1.0 + oval_slack) ** 2)


def _build_check_for_hole(
    sections_packed: tuple[dict, ...],
):
    """Returns a closure check(target, top_sample, spin_deg, tips_local, centroid_local)
    → (valid, ap_deg, ml_deg). Sections are closure-captured to keep the JIT
    cache key clean (one compile per hole-section-count signature)."""
    n_sections = len(sections_packed)

    def check(target, top_sample, spin_deg, tips_local, centroid_local):
        D_lps = top_sample - target
        D_lps_n = D_lps / jnp.linalg.norm(D_lps)
        D_ras = jnp.array([-D_lps_n[0], -D_lps_n[1], D_lps_n[2]])
        ap_deg, ml_deg = _vec_to_ap_ml_jax(D_ras)
        R = arc_angles_to_rotation(ap_deg, ml_deg, spin_deg)
        pose_tip = target - R @ centroid_local
        shaft_dir = R @ jnp.array([0.0, 0.0, 1.0])

        # Per-shank world positions
        shanks_world = tips_local @ R.T + pose_tip  # (n_shanks, 3)

        # For each section, all shanks must pass through.
        valid = jnp.bool_(True)
        for sec in sections_packed:
            # vmap over shanks
            def per_shank(sh):
                return _shank_in_section_jax(
                    sh, shaft_dir,
                    sec["center"], sec["axis"], sec["e1"], sec["e2"],
                    sec["a"], sec["b"], sec["theta"],
                )
            masks = jax.vmap(per_shank)(shanks_world)
            valid = valid & jnp.all(masks)
        return valid, ap_deg, ml_deg

    return check


# ---------------------------------------------------------------------------
# Atlas builder
# ---------------------------------------------------------------------------


def build_visibility_atlas(
    probes,
    holes: Sequence[Hole],
    *,
    n_top: int = 64,
    n_spin: int = 24,
    spin_range_deg: tuple[float, float] = (-180.0, 180.0),
    verbose: bool = False,
) -> Atlas:
    """Build the visibility atlas for all (probe, hole) pairs.

    Each entry tests whether the probe's shanks all thread the hole's
    sections at any (top-ellipse-sample, spin) combination.
    """
    probe_names = tuple(p.name for p in probes)
    hole_ids = tuple(h.id for h in holes)
    entries: dict[tuple[str, int], AtlasEntry] = {}

    # Precompute per-probe centroid and shank tips
    probe_centroid = {p.name: _probe_centroid_local(p) for p in probes}
    probe_tips_local = {
        p.name: np.asarray(p.shank_tips_local, dtype=np.float32) for p in probes
    }
    probe_target = {
        p.name: np.asarray(p.target_LPS, dtype=np.float32) for p in probes
    }

    # Spin grid (closure-captured per JIT'd check)
    spins_np = np.linspace(
        spin_range_deg[0], spin_range_deg[1], n_spin, endpoint=False, dtype=np.float32
    )
    spins_jnp = jnp.asarray(spins_np)

    t0 = time.perf_counter()
    for h_idx, hole in enumerate(holes):
        top_pts_np = _sample_top_ellipse_points(hole.sections[0], n_top).astype(np.float32)
        top_pts_jnp = jnp.asarray(top_pts_np)
        sections_packed = tuple(
            {
                k: jnp.asarray(v) if isinstance(v, np.ndarray) else jnp.float32(v)
                for k, v in _pack_section(s).items()
            }
            for s in hole.sections
        )
        check = _build_check_for_hole(sections_packed)
        # vmap over (top_sample × spin)
        check_vmap = jax.jit(
            jax.vmap(
                jax.vmap(check, in_axes=(None, 0, None, None, None)),
                in_axes=(None, None, 0, None, None),
            )
        )

        for probe in probes:
            target = jnp.asarray(probe_target[probe.name])
            tips_local = jnp.asarray(probe_tips_local[probe.name])
            centroid_local = jnp.asarray(probe_centroid[probe.name], dtype=jnp.float32)
            if tips_local.shape[0] == 0:
                entries[(probe.name, hole.id)] = AtlasEntry(
                    probe_name=probe.name, hole_id=hole.id,
                    ap_min=None, ap_max=None, anchors=()
                )
                continue
            valid_grid, ap_grid, ml_grid = check_vmap(
                target, top_pts_jnp, spins_jnp, tips_local, centroid_local
            )
            valid_np = np.asarray(valid_grid)  # (n_spin, n_top)
            if not valid_np.any():
                entries[(probe.name, hole.id)] = AtlasEntry(
                    probe_name=probe.name, hole_id=hole.id,
                    ap_min=None, ap_max=None, anchors=()
                )
                continue
            ap_np = np.asarray(ap_grid)
            ml_np = np.asarray(ml_grid)
            # Build per-(spin, top) AtlasEntry anchors
            spin_grid = np.broadcast_to(spins_np[:, None], valid_np.shape)
            valid_aps = ap_np[valid_np]
            valid_mls = ml_np[valid_np]
            valid_spins = spin_grid[valid_np]
            anchors = tuple(
                PoseAnchor(
                    ap_deg=float(valid_aps[i]),
                    ml_deg=float(valid_mls[i]),
                    spin_deg=float(valid_spins[i]),
                    off_R_mm=0.0, off_A_mm=0.0, depth_mm=0.0,
                    threading_max_g=-1.0, target_miss_mm=0.0,
                )
                for i in range(int(valid_np.sum()))
            )
            entries[(probe.name, hole.id)] = AtlasEntry(
                probe_name=probe.name, hole_id=hole.id,
                ap_min=float(valid_aps.min()),
                ap_max=float(valid_aps.max()),
                anchors=anchors,
            )

        if verbose:
            valid_count = sum(
                1 for pn in probe_names
                if entries[(pn, hole.id)].ap_min is not None
            )
            print(f"  [vis-atlas] hole {hole.id}: {valid_count}/{len(probes)} probes valid "
                  f"({time.perf_counter() - t0:.2f}s)")

    if verbose:
        print(f"  [vis-atlas] total build: {time.perf_counter() - t0:.2f}s")
        for probe in probes:
            valid_hids = [
                hid for hid in hole_ids
                if entries[(probe.name, hid)].ap_min is not None
            ]
            n_anchors_total = sum(
                len(entries[(probe.name, hid)].anchors) for hid in valid_hids
            )
            print(f"  [vis-atlas] {probe.name:>5}: {len(valid_hids)}/{len(hole_ids)} holes "
                  f"({n_anchors_total} anchors total)")
    return Atlas(entries=entries, probe_names=probe_names, hole_ids=hole_ids)
