"""Structured spin-basin search for top-K Stage 2 candidates.

Applies the heuristics from ``dev/spin_search_heuristics.md`` to
candidates that survive the violation-fn cutoff. For each cand:

  1. Build coupling graph from target positions (H4: only close pairs
     get spin coordination; far probes decouple).
  2. Per probe, compute candidate spins:
     - 4-shank quadbase: {θ_slot, θ_slot + 180°}  (H1)
     - 1-shank NP 2.x:   {θ_slot, θ_slot ± 90°, θ_slot + 180°}
       optionally narrowed by H2's geometric optimum for close pairs.
  3. Beam-search assignments scored by H2 (perpendicular-to-gap-direction
     geometric optimum) and H3 (triangle term).
  4. Run the full chain (P1 + P2 + FCL validator) per kept assignment.
  5. Best-per-cand by (FCL-feasible, coverage).

Designed to run on top-K cands by violation_fn (NOT the whole 8908-cand
pool); per the spin_search_heuristics.md design, deeply infeasible cands
don't benefit from spin search.

Run::
    uv run --python 3.13 python -m scripts.spin_heuristic_search \\
        examples/836656-config-T12.yml scratch/0283-300-04.holes.yml \\
        --polish-pkl /tmp/full_polish_unitcircle.pkl \\
        --top-n 50 --out-dir examples/836656-config-T12_spin_search
"""

from __future__ import annotations

import argparse
import itertools
import os as _os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import yaml
from aind_mri_utils.arc_angles import arc_angles_to_affine
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    Phase1Weights,
    make_phase1_objective,
)
from aind_low_point.optimization.stage3_phase2_jax import (
    Phase2Weights,
    make_phase2,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config, save_plan_to_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    phase1_bounds,
)

# ---------------------------------------------------------------------------
# Per-probe-kind body asymmetry direction (LOCAL frame)
# ---------------------------------------------------------------------------
# These are the local-frame unit vectors along which the body is "wide"
# (i.e., perpendicular to this direction is the "narrow profile" that
# should face neighboring probes for max clearance). For all current
# Neuropixel-derived probes the wide axis is the +y axis of the local
# frame — matches the 4-shank row direction for quadbase and the
# dovetail/headstage flex direction for NP 2.x.
#
# If a future probe has a different geometry, override here.
BODY_LONG_AXIS_LOCAL = {
    "quadbase-alpha": np.array([0.0, 1.0, 0.0]),
    "quadbase-dovetail": np.array([0.0, 1.0, 0.0]),
    "2.1": np.array([0.0, 1.0, 0.0]),
    # default fallback below
}
_DEFAULT_LONG_AXIS = np.array([0.0, 1.0, 0.0])


def body_long_axis_local(kind: str) -> np.ndarray:
    return BODY_LONG_AXIS_LOCAL.get(kind, _DEFAULT_LONG_AXIS)


def is_four_shank(probe_static) -> bool:
    """Detect 4-shank probes dynamically from the static info.

    Covers quadbase-alpha, quadbase-dovetail, NP 2.4, and any future
    4-shank kinds. The threading constraint (H1) limits these to
    {slot, slot + 180°} regardless of name.
    """
    return len(probe_static.shank_tips_local) >= 4


# ---------------------------------------------------------------------------
# Optimal spin: rotate so local long-axis is perpendicular to gap_dir
# ---------------------------------------------------------------------------


def _orbit_basis(ap_deg: float, ml_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (a, b) such that the world-frame image of probe local +y
    under ``arc_angles_to_affine(ap, ml, spin)`` is::

        u(spin) = sin(spin) * a + cos(spin) * b

    Derivation: ``R_LPS = R_x(ap) R_y(-ml) R_z(-spin)`` and
    ``R_z(-spin) [0,1,0] = (sin spin, cos spin, 0)``. So::

        u(spin) = R_x(ap) R_y(-ml) (sin spin, cos spin, 0)
                = sin spin · R_x(ap) R_y(-ml) [1,0,0]
                + cos spin · R_x(ap) R_y(-ml) [0,1,0]

    Avoids calling ``arc_angles_to_affine`` twice per heuristic eval.
    """
    ap = np.deg2rad(ap_deg)
    ml = np.deg2rad(ml_deg)
    cap, sap = np.cos(ap), np.sin(ap)
    cml, sml = np.cos(ml), np.sin(ml)
    a = np.array([cml, -sap * sml, cap * sml])
    b = np.array([0.0, cap, sap])
    return a, b


def spin_to_align_y_with(
    target_dir_world: np.ndarray,
    ap_deg: float,
    ml_deg: float,
) -> float:
    """Closed-form spin (deg) that best aligns probe local +y with
    ``target_dir_world`` under ``arc_angles_to_affine(ap, ml, spin)``.

    ``u(spin) · sm`` is maximised at ``spin = atan2(a · sm, b · sm)``
    (see :func:`_orbit_basis`). When ``sm`` doesn't lie in the orbit
    plane (off bore-aligned (ap, ml)), the residual is the projection
    error — non-zero but typically <30° for our manual plans.
    """
    a, b = _orbit_basis(ap_deg, ml_deg)
    sm = np.asarray(target_dir_world, dtype=float)
    sm = sm / max(float(np.linalg.norm(sm)), 1e-12)
    return float(np.degrees(np.arctan2(float(a @ sm), float(b @ sm))))


def optimal_spin_for_gap(
    long_axis_local: np.ndarray,
    ap_deg: float,
    ml_deg: float,
    gap_dir_world: np.ndarray,
) -> tuple[float, float]:
    """Spin (deg) such that the probe's local long-axis is perpendicular
    to ``gap_dir_world``. Returns (θ, θ + 180°).

    For ``long_axis_local = (0, 1, 0)``, this means finding the spin where
    ``u(spin) · gap_dir = 0`` — i.e. local +y in world is perpendicular
    to the gap. From :func:`_orbit_basis`::

        u(spin) · g = sin spin · (a · g) + cos spin · (b · g) = 0
        → tan spin = -(b · g) / (a · g)
        → spin = atan2(-(b · g), (a · g))

    Equivalent to ``spin_to_align_y_with(gap × axis_z_world)``, but the
    direct atan2 form avoids needing the rotation-axis vector.
    """
    if not (
        abs(long_axis_local[0]) < 1e-6
        and abs(long_axis_local[2]) < 1e-6
        and abs(abs(long_axis_local[1]) - 1.0) < 1e-6
    ):
        raise NotImplementedError(
            f"optimal_spin_for_gap currently assumes long_axis_local "
            f"= (0, ±1, 0); got {long_axis_local}. Update derivation "
            f"before using a probe type with different body asymmetry."
        )
    a, b = _orbit_basis(ap_deg, ml_deg)
    g = np.asarray(gap_dir_world, dtype=float)
    ag = float(a @ g)
    bg = float(b @ g)
    # u · g = 0  →  sin spin · ag + cos spin · bg = 0
    # → tan spin = -bg / ag  →  spin = atan2(-bg, ag)
    theta_deg = float(np.degrees(np.arctan2(-bg, ag)))
    return (theta_deg, theta_deg + 180.0)


# ---------------------------------------------------------------------------
# Per-probe candidate spin generation (H1 + H2-informed)
# ---------------------------------------------------------------------------


def per_probe_spin_candidates(
    statics: list,
    coupling: dict[int, list[int]],
    target_LPS: np.ndarray,
    arc_aps: np.ndarray,
    ml_per_probe: np.ndarray,
    probe_kind_by_name: dict[str, str],
    seed_spins: dict[int, float] | None = None,
) -> dict[int, list[float]]:
    """For each probe, generate candidate spin angles (degrees).

    Combines H1 (threading) with H2 (geometric optimum given close
    neighbors). The returned list is deduplicated modulo 5°.
    """
    out: dict[int, list[float]] = {}
    for i, st in enumerate(statics):
        kind = probe_kind_by_name.get(st.name, "default")
        four_shank = is_four_shank(st)
        # H1: spin that aligns probe local +y (shank-row direction per
        # kinematics.py:120-121) with the slot's major axis under the
        # probe's actual (ap, ml). The closed form accounts for the
        # arc_angles_to_affine convention; the older ``π/2 − slot_theta``
        # warm-start in optimize.py is ~90° off because it assumed a
        # local-+x shank row (see scripts/diagnose_slot_major_formula.py).
        sm_world = st.assigned_hole.slot_major_dir()
        ap_i = float(arc_aps[st.arc_idx])
        ml_i = float(ml_per_probe[i])
        spin_align_y = spin_to_align_y_with(sm_world, ap_i, ml_i)

        # H1: threading-allowed spins
        if four_shank:
            # 4-shank (quadbase, NP 2.4, etc.): threading constrains to
            # ~0° or ~180° relative to slot major axis.
            h1 = [spin_align_y, spin_align_y + 180.0]
        else:
            # 1-shank: spin free for threading; offer 4 orientations.
            h1 = [
                spin_align_y,
                spin_align_y + 90.0,
                spin_align_y + 180.0,
                spin_align_y + 270.0,
            ]

        # H2: for each close neighbor, get the geometric optimum spin.
        # 4-shank: snap to nearest H1 entry. Otherwise add to candidate set.
        h2_candidates: list[float] = []
        neighbors = coupling.get(i, [])
        for j in neighbors:
            gap = target_LPS[j] - target_LPS[i]
            norm = float(np.linalg.norm(gap))
            if norm < 1e-9:
                continue
            gap_dir = gap / norm
            sp_a, sp_b = optimal_spin_for_gap(
                body_long_axis_local(kind),
                ap_i,
                ml_i,
                gap_dir,
            )
            h2_candidates.extend([sp_a, sp_b])

        if four_shank:
            # 4-shank must thread the slot — snap H2 optima to whichever
            # of the two H1 slot-aligned spins is closer.
            snapped = set()
            for sp in h2_candidates:
                d_slot = abs(_wrap_deg(sp - spin_align_y))
                d_flip = abs(_wrap_deg(sp - (spin_align_y + 180.0)))
                snapped.add(spin_align_y if d_slot < d_flip else spin_align_y + 180.0)
            cands = list(set(h1) | snapped)
        else:
            cands = list(h1) + h2_candidates

        # Always include the seed spin (the polished/augmented warm-start's
        # current spin for this probe). Acts as a safety net so the chain
        # at least retries the existing solution even if H1's geometric
        # derivation differs from the warm-start's basin.
        if seed_spins is not None and i in seed_spins:
            cands.append(float(seed_spins[i]))

        # Deduplicate modulo 5°
        cands = _dedup_angles(cands, tol_deg=5.0)
        # Wrap to [-180, 180] for stability
        cands = [_wrap_deg(c) for c in cands]
        out[i] = sorted(cands)
    return out


def _wrap_deg(x: float) -> float:
    return float(((x + 180.0) % 360.0) - 180.0)


def _dedup_angles(angles: list[float], tol_deg: float = 5.0) -> list[float]:
    out: list[float] = []
    for a in angles:
        wa = _wrap_deg(a)
        if not any(abs(_wrap_deg(wa - b)) < tol_deg for b in out):
            out.append(wa)
    return out


# ---------------------------------------------------------------------------
# Coupling graph — spin-swept-volume intersection
# ---------------------------------------------------------------------------
#
# Two probes' spins can interact iff the volumes their bodies sweep as spin
# varies over [0, 360) overlap. Spin rotates a probe about the local z-axis
# through the recording-array centre (pose_from_optimizer_vars: R_z(-spin)
# about that axis), so the swept volume is the REVOLUTION of the probe mesh
# about that axis. We represent each as the outer radius profile r_max(z) — the
# true shape (thin shank/neck, wide headstage), not a single bounding radius —
# and test overlap of the two solids placed at their (ap, ml, target) poses.
#
# Replaces the old target-distance heuristic (`d < D_interact_mm`), which both
# used the wrong distance (deep targets, not the bodies that actually interact
# above the brain) and a guessed threshold ≈ the whole mouse brain.

N_ZBINS = 120  # height bins for the r(z) revolution profile
N_THETA = 24  # angular samples when meshing the swept surface


def swept_profile(mesh_verts: np.ndarray, rec_center_local: np.ndarray) -> tuple:
    """Solid-of-revolution profile of a probe body about its insertion axis.

    Returns ``(z_centers, r_max, z_lo, z_hi)`` in the local revolved frame
    (``z`` = local z − recording-centre z; ``r`` = transverse distance from the
    recording-array spin axis). ``r_max(z)`` is the swept solid's outer radius.
    """
    q = np.asarray(mesh_verts, np.float64) - np.asarray(rec_center_local, np.float64)
    r = np.hypot(q[:, 0], q[:, 1])
    z = q[:, 2]
    z_lo, z_hi = float(z.min()), float(z.max())
    edges = np.linspace(z_lo, z_hi, N_ZBINS + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, N_ZBINS - 1)
    rmax = np.zeros(N_ZBINS)
    for b in range(N_ZBINS):
        m = idx == b
        if m.any():
            rmax[b] = r[m].max()
    zc = 0.5 * (edges[:-1] + edges[1:])
    return zc, rmax, z_lo, z_hi


def swept_surface_world(prof: tuple, R0: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Sample the swept solid's surface in world LPS.

    ``world = R0 @ q + target`` with ``R0 = arc_angles_to_affine(ap, ml, 0)``
    (spin-independent) and ``q`` over the revolved surface ``r_max(z)``.
    """
    zc, rmax, _, _ = prof
    th = np.linspace(0.0, 2 * np.pi, N_THETA, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    pts = []
    for z, rm in zip(zc, rmax):
        if rm <= 0:
            continue
        ring = np.stack([rm * ct, rm * st, np.full(N_THETA, z)], axis=1)
        pts.append(ring @ np.asarray(R0).T + np.asarray(target))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def inside_swept(
    world_pts: np.ndarray,
    prof: tuple,
    R0: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Boolean mask: which world points lie inside the swept solid."""
    zc, rmax, z_lo, z_hi = prof
    q = (np.asarray(world_pts) - np.asarray(target)) @ np.asarray(R0)
    z = q[:, 2]
    r = np.hypot(q[:, 0], q[:, 1])
    rm = np.interp(z, zc, rmax, left=0.0, right=0.0)
    return (z >= z_lo) & (z <= z_hi) & (r <= rm)


def swept_overlap(
    prof_i,
    R0_i,
    t_i,
    surf_i,
    prof_j,
    R0_j,
    t_j,
    surf_j,
) -> tuple[bool, float, np.ndarray | None]:
    """``(overlap, gap_mm, contact_world)`` between two swept solids.

    Overlap if either solid's sampled surface enters the other. For overlapping
    solids ``gap`` is ``-penetration`` (the deepest interpenetration — a
    meaningful tightness signal, unlike the surface nearest-distance which is
    ~0 for any intersection) and ``contact_world`` is the centroid of the
    overlap region (the world location where the two bodies actually conflict —
    the H2 "face the narrow profile toward the contact" direction). For disjoint
    solids ``gap`` is the positive nearest surface-surface distance and
    ``contact_world`` is ``None``.
    """
    from scipy.spatial import cKDTree

    tree_j = cKDTree(surf_j)
    in_j = inside_swept(surf_i, prof_j, R0_j, t_j)  # i-surf points inside j
    in_i = inside_swept(surf_j, prof_i, R0_i, t_i)  # j-surf points inside i
    if not (in_j.any() or in_i.any()):
        return False, float(tree_j.query(surf_i)[0].min()), None
    # Penetration ≈ deepest inside point's distance to the other boundary;
    # contact = centroid of the overlap region (the points of one body that lie
    # inside the other).
    pen = 0.0
    pts_in = []
    if in_j.any():
        pen = max(pen, float(tree_j.query(surf_i[in_j])[0].max()))
        pts_in.append(surf_i[in_j])
    if in_i.any():
        pen = max(pen, float(cKDTree(surf_i).query(surf_j[in_i])[0].max()))
        pts_in.append(surf_j[in_i])
    contact = np.concatenate(pts_in).mean(axis=0)
    return True, -pen, contact


def build_coupling_graph(
    statics: list,
    arc_aps: np.ndarray,
    ml_per_probe: np.ndarray,
    target_LPS: np.ndarray,
    mesh_verts_by_kind: dict[str, np.ndarray],
    probe_kind_by_name: dict[str, str],
) -> tuple[
    dict[int, list[int]],
    dict[tuple[int, int], float],
    dict[tuple[int, int], np.ndarray],
]:
    """``(coupling, tightness, contact)`` from spin-swept-volume intersection.

    ``i ↔ j`` iff the volumes the two probe bodies sweep over all spins overlap
    (so their spins interact). ``tightness[(i, j)]`` (i < j) is the penetration
    depth and ``contact[(i, j)]`` is the world centroid of the overlap region —
    the direction each probe should point its narrow profile toward. The H2
    "facing" terms are gated on this coupling and weighted by tightness
    (tightest partner dominates). Decoupled probes get no edge.

    ``mesh_verts_by_kind`` maps each probe kind to its canonical-local mesh
    vertices (``runtime.asset_catalog.get_geometry(f"probe:{kind}").raw.vertices``).
    """
    K = len(statics)
    prof_by_kind: dict[str, tuple] = {}
    profs, R0s, surfs = [], [], []
    for i, st in enumerate(statics):
        kind = probe_kind_by_name.get(st.name, "default")
        if kind not in prof_by_kind:
            prof_by_kind[kind] = swept_profile(mesh_verts_by_kind[kind], st.pivot_local)
        prof = prof_by_kind[kind]
        profs.append(prof)
        R0 = arc_angles_to_affine(
            float(arc_aps[st.arc_idx]), float(ml_per_probe[i]), 0.0
        )
        R0s.append(R0)
        surfs.append(swept_surface_world(prof, R0, target_LPS[i]))

    coupling: dict[int, list[int]] = {i: [] for i in range(K)}
    tightness: dict[tuple[int, int], float] = {}
    contact: dict[tuple[int, int], np.ndarray] = {}
    for i, j in itertools.combinations(range(K), 2):
        overlap, gap, ctr = swept_overlap(
            profs[i],
            R0s[i],
            target_LPS[i],
            surfs[i],
            profs[j],
            R0s[j],
            target_LPS[j],
            surfs[j],
        )
        if overlap:
            coupling[i].append(j)
            coupling[j].append(i)
            tightness[(i, j)] = -gap  # penetration depth (positive)
            contact[(i, j)] = ctr
    return coupling, tightness, contact


# ---------------------------------------------------------------------------
# Beam search over assignments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assignment:
    """A partial spin assignment plus its score."""

    spins: tuple[tuple[int, float], ...]  # ((probe_idx, spin_deg), ...)
    score: float


def beam_search_assignments(
    statics: list,
    candidates: dict[int, list[float]],
    coupling: dict[int, list[int]],
    target_LPS: np.ndarray,
    arc_aps: np.ndarray,
    ml_per_probe: np.ndarray,
    probe_kind_by_name: dict[str, str],
    D_far_mm: float = 25.0,
    beam_B: int = 64,
) -> list[Assignment]:
    """Beam-search over per-probe spin assignments using H2/H3 scoring.

    Returns the final beam (top ``beam_B`` complete assignments).
    """
    K = len(statics)
    # Process probes in order of coupling degree (most coupled first)
    order = sorted(range(K), key=lambda i: -len(coupling.get(i, [])))

    beam: list[Assignment] = [Assignment(spins=(), score=0.0)]
    for probe_idx in order:
        new_beam: list[Assignment] = []
        for partial in beam:
            for spin in candidates[probe_idx]:
                trial = partial.spins + ((probe_idx, spin),)
                s = _score_assignment(
                    trial,
                    coupling,
                    target_LPS,
                    arc_aps,
                    ml_per_probe,
                    statics,
                    probe_kind_by_name,
                    D_far_mm,
                )
                new_beam.append(Assignment(spins=trial, score=s))
        # Keep top beam_B by score
        new_beam.sort(key=lambda a: a.score)
        beam = new_beam[:beam_B]
    return beam


def _score_assignment(
    spins: tuple[tuple[int, float], ...],
    coupling: dict[int, list[int]],
    target_LPS: np.ndarray,
    arc_aps: np.ndarray,
    ml_per_probe: np.ndarray,
    statics: list,
    probe_kind_by_name: dict[str, str],
    D_far_mm: float,
) -> float:
    """H2 + H3 score: distance of each probe's spin from the geometric
    optimum for each of its close-pair neighbors, weighted by 1/d.
    Lower is better.
    """
    assignment = dict(spins)
    score = 0.0
    for i, spin_i in assignment.items():
        st_i = statics[i]
        kind_i = probe_kind_by_name.get(st_i.name, "default")
        ap_i = float(arc_aps[st_i.arc_idx])
        ml_i = float(ml_per_probe[i])
        for j in coupling.get(i, []):
            if j not in assignment:
                continue
            d = float(np.linalg.norm(target_LPS[j] - target_LPS[i]))
            if d > D_far_mm:
                continue
            gap_dir = (target_LPS[j] - target_LPS[i]) / d
            sp_opt_a, sp_opt_b = optimal_spin_for_gap(
                body_long_axis_local(kind_i),
                ap_i,
                ml_i,
                gap_dir,
            )
            # Distance to closer of the two optima
            d_a = abs(_wrap_deg(spin_i - sp_opt_a))
            d_b = abs(_wrap_deg(spin_i - sp_opt_b))
            err = min(d_a, d_b)
            # Closer pairs weighted more (H2 stronger when close)
            weight = max(0.0, 1.0 - d / D_far_mm)
            score += weight * (err / 90.0) ** 2  # normalized error squared
    # H3 (triangle term) — for each close-triple, add a small reward when
    # the third probe is roughly perpendicular to the pair's gap axis.
    # Omitted in this first draft; the H2 pairwise score covers most cases.
    return score


# ---------------------------------------------------------------------------
# Build phase1_x from a spin assignment, run chain
# ---------------------------------------------------------------------------


def x_with_spins(
    x_base: np.ndarray,
    statics: list,
    n_arcs: int,
    spin_overrides: dict[int, float],
) -> np.ndarray:
    """Take an existing phase1_x and override each probe's (sx, sy)
    with the new spin angle. Other DOF unchanged.
    """
    x = x_base.copy()
    for i, spin_deg in spin_overrides.items():
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        x[off + 1] = float(np.cos(np.deg2rad(spin_deg)))
        x[off + 2] = float(np.sin(np.deg2rad(spin_deg)))
    return x


def run_chain(
    x0: np.ndarray,
    statics: list,
    n_arcs: int,
    coverage_data,
    fixtures,
    validator,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Run Phase 1 + Phase 2 + FCL validator. Returns
    (x2, s_fcl, feas, cov_at_x2)."""
    n_probes = len(statics)
    bounds = phase1_bounds(n_arcs, n_probes)
    p1_fun, p1_jac = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=Phase1Weights(),
    )
    r1 = minimize(
        p1_fun,
        x0,
        jac=p1_jac,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=80, ftol=1e-5, gtol=1e-5),
    )
    x1 = np.asarray(r1.x, dtype=np.float64)
    p2 = make_phase2(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=Phase2Weights(min_clearance_mm=0.3),
    )
    r2 = minimize(
        p2["fun"],
        x1,
        jac=p2["jac"],
        method="trust-constr",
        bounds=bounds,
        constraints=p2["constraints_nlc"],
        options=dict(
            maxiter=80, xtol=1e-6, gtol=1e-5, initial_tr_radius=1.0, verbose=0
        ),
    )
    x2 = np.asarray(r2.x, dtype=np.float64)
    s_fcl = validator.slacks(x2)
    feas = bool(s_fcl.size == 0 or s_fcl.min() >= -1e-4)
    # Coverage at x2 — using a violation-free Phase1Weights to recover
    # coverage value (lambda_thread=lambda_clearance=... = 0, only
    # -coverage_total left).
    cov_w = Phase1Weights(
        lambda_thread=0.0,
        lambda_clearance=0.0,
        lambda_kinematic=0.0,
        lambda_bounds=0.0,
        lambda_clearance_fixture=0.0,
        lambda_margin_clear=0.0,
        lambda_margin_thread=0.0,
        lambda_margin_clear_fixture=0.0,
        lambda_unit_circle=0.0,
    )
    cov_fn, _ = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=cov_w,
    )
    cov = -float(cov_fn(x2))
    return x2, s_fcl, feas, cov


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument(
        "--polish-pkl", type=Path, default=Path("/tmp/full_polish_unitcircle.pkl")
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top N cands by violation_fn to spin-search",
    )
    p.add_argument(
        "--max-assignments-per-cand",
        type=int,
        default=8,
        help="After beam search, max # spin assignments to actually polish per cand.",
    )
    p.add_argument("--beam-b", type=int, default=64)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    print("Loading config / probes / SDFs / fixtures...", flush=True)
    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(args.holes)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    fixtures = build_fixture_sdf_data(runtime)
    bvh_cache = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fixture_bvhs = {
        f.name: make_fcl_bvh(runtime.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }

    with open(args.polish_pkl, "rb") as f:
        data = pickle.load(f)
    viol = np.asarray(data["violation_fn"])
    order = np.argsort(viol)
    cand_idxs = order[: args.top_n].tolist()
    print(
        f"Selected top-{args.top_n} cands by violation_fn (viol range "
        f"{viol[cand_idxs[0]]:.2f} to {viol[cand_idxs[-1]]:.2f})"
    )

    # _ProbeStatic doesn't carry the probe `kind`; build a lookup from
    # the runtime probes list. Needed for body-asymmetry vector +
    # threading-mode (4-shank vs 1-shank) decisions in H1/H2.
    probe_kind_by_name = {p.name: p.kind for p in probes}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for rank, cand_idx in enumerate(cand_idxs):
        cand = data["candidates"][int(cand_idx)]
        jc = data["results"][int(cand_idx)]
        statics = _build_probe_static(
            probes,
            holes,
            cand.ha,
            cand.aa,
            bvh_cache=bvh_cache,
            sdf_by_name=sdf_by_name,
        )
        n_arcs = jc.n_arcs
        n_probes = len(statics)
        coverage_data = build_coverage_data(probes, statics)
        validator = make_fcl_validator(
            statics,
            n_arcs,
            fixtures=fixtures,
            fixture_bvhs=fixture_bvhs,
        )

        x_aug = np.asarray(
            data["augmented_phase1_x"][int(cand_idx)],
            dtype=np.float64,
        )
        arc_aps = x_aug[:n_arcs]
        ml_per_probe = np.array(
            [x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)]
        )
        target_LPS = np.array([st.target_LPS for st in statics])

        # Build coupling graph (spin-swept-volume overlap) + candidates + beam.
        mesh_verts_by_kind = {
            k: np.asarray(runtime.asset_catalog.get_geometry(f"probe:{k}").raw.vertices)
            for k in set(probe_kind_by_name.values())
        }
        coupling, _tightness, _contact = build_coupling_graph(
            statics,
            arc_aps,
            ml_per_probe,
            target_LPS,
            mesh_verts_by_kind,
            probe_kind_by_name,
        )
        # Seed: current spin of each probe in the augmented warm-start
        # so the chain at least has the option of "keep current spin".
        seed_spins = {}
        for i in range(n_probes):
            off = n_arcs + PHASE1_PER_PROBE_VARS * i
            sx_w = float(x_aug[off + 1])
            sy_w = float(x_aug[off + 2])
            seed_spins[i] = float(np.degrees(np.arctan2(sy_w, sx_w)))
        spin_cands = per_probe_spin_candidates(
            statics,
            coupling,
            target_LPS,
            arc_aps,
            ml_per_probe,
            probe_kind_by_name,
            seed_spins=seed_spins,
        )
        beam = beam_search_assignments(
            statics,
            spin_cands,
            coupling,
            target_LPS,
            arc_aps,
            ml_per_probe,
            probe_kind_by_name,
            beam_B=args.beam_b,
        )

        # Take top-K assignments, run chain on each
        to_polish = beam[: args.max_assignments_per_cand]
        print(
            f"\n[rank {rank + 1}/{len(cand_idxs)}] cand#{int(cand_idx)} "
            f"viol_fn={viol[int(cand_idx)]:.2f}  "
            f"spin_cands_per_probe={[len(spin_cands[i]) for i in range(n_probes)]}  "
            f"assignments_to_polish={len(to_polish)}",
            flush=True,
        )

        best = None  # (feas, cov, x2)
        for asg_i, asg in enumerate(to_polish):
            overrides = dict(asg.spins)
            x0 = x_with_spins(x_aug, statics, n_arcs, overrides)
            t0 = time.time()
            x2, s_fcl, feas, cov = run_chain(
                x0,
                statics,
                n_arcs,
                coverage_data,
                fixtures,
                validator,
            )
            wall = time.time() - t0
            tag = "FEAS" if feas else "FAIL"
            print(
                f"    asg{asg_i}: score={asg.score:.2f} "
                f"fcl_min={s_fcl.min():+.4f} cov={cov:.2f} "
                f"wall={wall:.1f}s {tag}",
                flush=True,
            )
            if best is None or (feas, cov) > (best[0], best[1]):
                best = (feas, cov, x2, s_fcl)

        results.append((int(cand_idx), best))

    # Save best per cand
    print(f"\nSaving best chain output per cand to {args.out_dir}...")
    feas_count = sum(1 for _, b in results if b is not None and b[0])
    print(f"  {feas_count}/{len(results)} cands reached FCL feasible")
    for rank, (cand_idx, best) in enumerate(results, start=1):
        if best is None:
            continue
        feas, cov, x2, s_fcl = best
        # Save as config yml
        cfg_local = ConfigModel.from_yaml(args.config)
        rt_local = build_runtime_from_config(cfg_local)
        statics_local = _build_probe_static(
            probes,
            holes,
            data["candidates"][cand_idx].ha,
            data["candidates"][cand_idx].aa,
            bvh_cache=bvh_cache,
            sdf_by_name=sdf_by_name,
        )
        n_arcs = data["results"][cand_idx].n_arcs
        _apply_x_to_plan_state(rt_local.plan_state, x2, statics_local, n_arcs)
        candidate_cfg = save_plan_to_config(rt_local.plan_state, cfg_local)
        tag = "feas" if feas else "fail"
        fname = f"plan-{rank:03d}-{tag}-cand{cand_idx:05d}-cov{cov:05.2f}.yml"
        with open(args.out_dir / fname, "w") as f:
            yaml.safe_dump(
                candidate_cfg.model_dump(mode="json"),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
    return 0


def _apply_x_to_plan_state(plan_state, x, statics, n_arcs):
    """Same as save_chain_plans._apply_x_to_plan_state."""
    arc_aps = x[:n_arcs]
    arc_letters = [chr(ord("a") + i) for i in range(n_arcs)]
    plan_state.kinematics.arc_angles = {
        arc_letters[i]: float(arc_aps[i]) for i in range(n_arcs)
    }
    for i, st in enumerate(statics):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        ml = float(x[off + 0])
        sx = float(x[off + 1])
        sy = float(x[off + 2])
        off_R = float(x[off + 3])
        off_A = float(x[off + 4])
        depth = float(x[off + 5])
        spin = float(np.degrees(np.arctan2(sy, sx)))
        plan = plan_state.probes[st.name]
        plan.arc_id = arc_letters[st.arc_idx]
        plan.bind_ap_to_arc = True
        plan.ap_local = 0.0
        plan.ml_local = ml
        plan.spin = spin
        plan.offsets_RA = (off_R, off_A)
        plan.past_target_mm = depth


if __name__ == "__main__":
    raise SystemExit(main())
