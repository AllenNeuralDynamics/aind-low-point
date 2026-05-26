"""Joint (H, A) reranking layer.

A diagnostic on the 836656 / T12 problem (see
``dev/optimizer_review_response.md``) showed that the manual-feasible
plan's (hole assignment, arc assignment) is below rank 50 in both
discrete stages, because their costs are per-pair / per-probe and
don't see joint multi-probe feasibility (the AP coupling on shared
arcs, the ML separation that follows, the clearance between adjacent
probes).

This module adds a *reduced continuous reranking stage* between the
discrete layers and the full inner solve:

1. For every (H, A) candidate, solve a small SLSQP problem over
   ``(ap_arc_per_arc, ml_per_probe, spin_per_probe)`` (length
   ``n_arcs + 2K``) that captures threading, cross-arc AP separation,
   within-arc ML separation, and bounds. Offsets and depth are held at
   zero; coverage and clearance default to ``λ = 0``.
2. Score the SLSQP outcome with a feasibility-first lex key and keep
   the top ``k_joint`` candidates.
3. Run the existing full inner solve (CMA-ES + Stage A + Stage B +
   Stage C) on each survivor, warm-started from the reduced solution.

Multi-start: three reduced-SLSQP starts per (H, A):

1. Partitioner centroid + slot-major-axis spin.
2. AP-interval midpoint across on-arc probes + slot-major-axis spin.
3. Same as (2) but with alternating ``spin += 180°`` for odd-indexed
   on-arc probes — handles same-arc neighbours preferring opposite
   slot orientations.

(Future starts 4-8 could explore other multi-modal failure patterns,
e.g. swapping ml signs for narrow-gap probe pairs, mirroring across
the rig's centre plane. Skipped here until validation shows we need
them.)
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from dataclasses import dataclass, replace
from typing import Iterable

import fcl
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from aind_low_point.optimization.arc_assignment import (
    ArcAssignment,
    solve_top_k_arc_assignments,
)
from aind_low_point.optimization.geometry import cap_basis
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.hole_assignment import (
    AssignmentProbe,
    CostWeights,
    HoleAssignment,
    build_cost_matrix,
    solve_top_k_assignments,
)
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.kinematics import (
    pose_from_optimizer_vars,
)
from aind_low_point.optimization.objective import (
    ObjectiveWeights,
    VariableLayout,
)
from aind_low_point.optimization.optimize import (
    OptimizationResult,
    PlanCandidate,
    ProbeStaticInfo,
    _build_inner_context,
    _head_pitch_about_L_deg,
    _inner_solve_one,
)
from aind_low_point.optimization.pose_features import (
    PoseFeatures,
    precompute_pose_features,
)
from aind_low_point.optimization.recording import (
    RecordingGeometry,
    get_recording_geometry,
)

# ---------------------------------------------------------------------------
# Configuration + outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointWeights:
    """Penalty weights for the reduced-SLSQP scoring stage.

    Threading / AP-sep / ML-sep dominate. Bounds is a soft penalty
    that discourages the SLSQP from drifting outside a "comfortable"
    rig envelope without hard-clipping it (the hard rig limits are
    enforced by the box bounds on the reduced variable vector).

    ``λ_clearance`` is a placeholder for once per-kind headstage
    capsules land — until then the headstage capsule is a uniform
    placeholder, so the term would reject perfectly-fine joint poses.

    ``λ_coverage`` is a tie-breaker among feasible candidates only;
    leaving it at zero defers coverage scoring to the full inner solve
    where the cost matters most.
    """

    lambda_thread: float = 100.0
    lambda_arc_ap: float = 100.0
    lambda_ml: float = 100.0
    lambda_bounds: float = 1.0
    lambda_clearance: float = 100.0
    lambda_coverage: float = 0.0
    # Pulls (sx, sy) magnitude toward 1 (unit circle). spin_deg only
    # uses the direction of (sx, sy), so magnitude is a free DOF; this
    # penalty keeps it consistent across stages and away from the
    # gradient-undefined origin. See sdf_jax.unit_circle_penalty.
    # Reduced from 100 → 10 on 2026-05-26 after observing weight=100
    # over-dominated polish iter budget and pulled some cands into
    # worse local minima (cand 4195 manual rank #15 → #2866).
    lambda_unit_circle: float = 10.0
    comfortable_ap_deg: float = 50.0
    comfortable_ml_deg: float = 50.0
    min_arc_ap_sep_deg: float = 16.0
    min_intra_arc_ml_sep_deg: float = 16.0
    threading_oval_tolerance: float = 0.0
    # Minimum signed clearance (mm) between any pair of probe-mesh BVHs.
    # The clearance penalty is ``λ_clearance · Σ ReLU(min_clearance_mm − d)²``
    # per pair, using the same hybrid (fcl.distance for d>0, fcl.collide
    # for penetration depth in overlap) that the full inner solve uses.
    min_clearance_mm: float = 0.0


@dataclass(frozen=True)
class JointRerankMetrics:
    """Lex-ranking metrics for a :class:`JointCandidate`.

    ``max_violation`` is the max over the per-group max-violations
    (threading, cross-arc AP separation, within-arc ML separation).
    ``sum_violation_sq`` is ``Σ ReLU(violation)²`` across the same
    groups. Units mix (threading is dimensionless; AP/ML are deg) but
    the comparison is consistent across candidates.

    ``approximate_coverage`` is a cheap surrogate (sum of per-probe
    target hits at the reduced pose) — present so coverage can break
    ties among "feasible-enough" candidates without paying for the
    full Gaussian-density integral.

    ``original_lsap_cost`` is the LSAP score for the underlying hole
    assignment (``Σ_i C[i, H[i]]``); useful diagnostic.
    """

    max_violation: float
    sum_violation_sq: float
    max_violation_threading: float
    max_violation_arc_ap_sep: float
    max_violation_intra_arc_ml_sep: float
    max_violation_clearance: float
    # Diagnostic-only (Stage 2): max probe-vs-fixture penetration
    # (cone/well/headframe), measured via raw-mesh FCL. Zero when no
    # fixture BVHs were threaded into the metrics call. Not included in
    # ``max_violation`` to keep lex-ordering backward-compatible; use
    # ``lex_key_with_fixtures`` for fixture-aware ranking.
    max_violation_fixture: float = 0.0
    approximate_coverage: float = 0.0
    bounds_softpenalty: float = 0.0
    original_lsap_cost: float = 0.0

    def lex_key(self, feasibility_threshold: float = 0.0) -> tuple[float, float, float]:
        """Lex-rank tuple ``(eff_viol, sum_viol_sq, bounds_softpenalty)``.

        Under the reduced reranker's rotation-only pose convention,
        ``recording_center_local`` is constructed to land on the target
        regardless of ``(ap, ml, spin)`` — so any rigid distance-to-
        target metric (including the prior ``approximate_coverage``) is
        rotation-invariant. Coverage can't break ties at this stage.
        We use ``bounds_softpenalty`` (a soft preference for comfortable
        ap/ml angles) instead — lower = better rig comfort.

        Mirrors :meth:`PlanCandidate.lex_key`: an ``ε``-collapse on
        ``max_violation`` so any candidate at or below the slop budget
        ranks among "feasible enough".
        """
        eff = max(0.0, self.max_violation - feasibility_threshold)
        return (eff, self.sum_violation_sq, self.bounds_softpenalty)

    def lex_key_with_fixtures(
        self, feasibility_threshold: float = 0.0
    ) -> tuple[float, float, float, float]:
        """Lex key that includes the fixture max-violation as the
        primary tiebreak after probe-probe / threading / kinematic
        violations.

        Use this when the metrics were computed with fixture BVHs
        threaded in (otherwise ``max_violation_fixture == 0`` and it's
        a no-op vs :meth:`lex_key`).
        """
        eff = max(0.0, self.max_violation - feasibility_threshold)
        eff_fix = max(0.0, self.max_violation_fixture - feasibility_threshold)
        return (eff, eff_fix, self.sum_violation_sq, self.bounds_softpenalty)


@dataclass(frozen=True)
class JointCandidate:
    """One (hole, arc) candidate ranked by the reduced SLSQP outcome."""

    ha: HoleAssignment
    aa: ArcAssignment
    n_arcs: int
    reduced_y: NDArray[np.floating]
    metrics: JointRerankMetrics


# ---------------------------------------------------------------------------
# Reduced-vector layout helpers
# ---------------------------------------------------------------------------


def _reduced_layout(n_arcs: int, n_probes: int) -> dict[str, slice]:
    """Slice indices for the reduced variable vector ``y``.

    ``y = [ap_arc_0, ..., ap_arc_{n_arcs-1}, (ml_0, sx_0, sy_0), ...,
    (ml_{K-1}, sx_{K-1}, sy_{K-1})]``. Spin is parameterized as a 2D
    unit-circle vector ``(sx, sy) ∝ (cos θ, sin θ)`` to avoid the
    ±180° wraparound discontinuity that bound-clipped SLSQP on the
    scalar-angle layout (Patch B, 2026-05-21).
    """
    return {
        "arc_aps": slice(0, n_arcs),
        "probe_vars": slice(n_arcs, n_arcs + 3 * n_probes),
    }


def _ml_sxy_from_y(
    y: NDArray, n_arcs: int, probe_idx: int
) -> tuple[float, float, float]:
    """Return ``(ml, sx, sy)`` for probe ``probe_idx`` from the reduced
    vector. See :func:`_reduced_layout` for the layout."""
    offset = n_arcs + 3 * probe_idx
    return float(y[offset]), float(y[offset + 1]), float(y[offset + 2])


def _ml_spin_from_y(y: NDArray, n_arcs: int, probe_idx: int) -> tuple[float, float]:
    """Return ``(ml, spin_deg)`` for probe ``probe_idx``.

    Convenience wrapper for the (sx, sy) layout that recovers a
    scalar spin angle via ``atan2(sy, sx)``. Used at boundaries
    (Stage 3 handoff, plan export) where downstream code wants degrees.
    """
    ml, sx, sy = _ml_sxy_from_y(y, n_arcs, probe_idx)
    return ml, float(np.degrees(np.arctan2(sy, sx)))


# ---------------------------------------------------------------------------
# Per-probe static helpers used inside the surrogate objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProbeStatic:
    """Pre-built static info used inside the reduced objective.

    Carrying the asset hole reference + per-section ``cap_basis``
    arrays + per-section centres lets ``_max_g_threading`` skip
    repeated ``np.cross`` computations inside the SLSQP fn loop.
    """

    name: str
    target_LPS: NDArray
    shank_tips_local: NDArray
    pivot_local: NDArray
    assigned_hole: Hole
    arc_idx: int
    section_axes: NDArray  # shape (S, 3) — unit normals
    section_e1: NDArray  # shape (S, 3) — first cap basis vector
    section_e2: NDArray  # shape (S, 3) — second cap basis vector
    section_centers: NDArray  # shape (S, 3) — section centres
    section_cos_theta: NDArray  # shape (S,) — cos(theta)
    section_sin_theta: NDArray  # shape (S,) — sin(theta)
    section_a: NDArray  # shape (S,) — major half-extent
    section_b: NDArray  # shape (S,) — minor half-extent
    # FCL BVH collision object on the canonical-local probe mesh; the
    # reduced objective updates its transform each iteration to query
    # pairwise probe clearance. ``None`` for probes without a
    # ``collision_mesh`` (those pairs drop from the clearance check).
    bvh_obj: fcl.CollisionObject | None = None
    # Optional SDF data for this probe (grid, origin, spacing, surface
    # points — already converted to jnp arrays). When present, the
    # reduced objective queries the JAX SDF kernel for pairwise
    # clearance instead of FCL BVH+collide.
    sdf_data: dict | None = None


_SDF_JNP_CACHE: dict[tuple, dict] = {}


# Catastrophic-infeasibility shortcut for ``score_joint``: when SLSQP's
# final objective exceeds this, the cand's lex_key is already worst-bucket
# and the FCL pair-clearance sweep in metric_eval is wasted work.
# Threshold sized for λ_thread = λ_clearance = 100 and a 10-unit per-element
# violation → fn ≈ 100 × (10)² = 10 000. Below this the polish endpoint is
# interesting enough to evaluate fully.
_CATASTROPHIC_FN_THRESHOLD: float = 1.0e4
_CATASTROPHIC_MAX_VIOL_SENTINEL: float = 1.0e6


_STAGE2_TIMINGS: dict[str, float] = {
    "build_probe_static": 0.0,
    "build_starts": 0.0,
    "spin_restore": 0.0,
    "slsqp": 0.0,
    "metric_eval": 0.0,
}


def stage2_timings() -> dict[str, float]:
    """Return accumulated wall-time per Stage 2 component (for profiling)."""
    return dict(_STAGE2_TIMINGS)


def _sdf_jnp_payload(sdf) -> dict:
    """Return the jnp-array form of a ``ProbeSDF`` (body voxel grid +
    surface samples + shank OBBs), keyed on the object's id so repeated
    calls in the same process share the GPU-resident arrays.

    Stage 2 calls ``_build_probe_static`` once per (H, A) candidate —
    caching the conversion saves per-call ``jnp.asarray`` Python overhead
    times probe count times candidate count.
    """
    key = (id(sdf),)
    cached = _SDF_JNP_CACHE.get(key)
    if cached is not None:
        return cached
    import jax.numpy as jnp
    payload = dict(
        grid=jnp.asarray(sdf.grid, dtype=jnp.float32),
        origin=jnp.asarray(sdf.origin, dtype=jnp.float32),
        spacing=jnp.asarray(sdf.spacing, dtype=jnp.float32),
        surface=jnp.asarray(sdf.surface_points, dtype=jnp.float32),
        shank_centers=jnp.asarray(sdf.shank_centers, dtype=jnp.float32),
        shank_halves=jnp.asarray(sdf.shank_halves, dtype=jnp.float32),
    )
    _SDF_JNP_CACHE[key] = payload
    return payload


def _build_probe_static(
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    ha: HoleAssignment,
    aa: ArcAssignment,
    bvh_cache: dict[str, fcl.CollisionObject | None] | None = None,
    sdf_by_name: dict | None = None,
) -> list[_ProbeStatic]:
    """Build the per-probe static cache for the reduced objective.

    ``bvh_cache`` (optional) is a ``probe_name → CollisionObject`` map
    of pre-built BVHs. When provided, this avoids the ~100 ms / 46k-face
    cost of building a fresh FCL BVH per (H, A) candidate (and we
    score hundreds of candidates per run, so the waste compounds). The
    BVH is intrinsic to the probe mesh and reusable across all
    candidates — only its world transform changes per pose.

    ``sdf_by_name`` (optional) is a ``probe_name → ProbeSDF`` map.
    When provided, each probe's SDF arrays (body grid + surface samples
    + shank OBBs) are packed into a dict and attached as
    ``_ProbeStatic.sdf_data`` for the dual-rep JAX clearance backend.
    """
    holes_by_id = {h.id: h for h in holes}
    fallback_geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))
    out: list[_ProbeStatic] = []
    for p in probes:
        try:
            geom = get_recording_geometry(p.kind)
        except KeyError:
            geom = fallback_geom
        tips = np.asarray(p.shank_tips_local, dtype=np.float64)
        if tips.shape[0] > 0:
            pivot = np.array(
                [
                    float(tips[:, 0].mean()),
                    float(tips[:, 1].mean()),
                    float(geom.active_center_mm),
                ],
                dtype=np.float64,
            )
        else:
            pivot = np.array([0.0, 0.0, float(geom.active_center_mm)], dtype=np.float64)
        hole_id = ha.probe_to_hole[p.name]
        arc_idx = aa.probe_to_arc_idx[p.name]
        hole = holes_by_id[hole_id]
        # Precompute per-section basis vectors / centres / oval params.
        sections = hole.sections
        s_axes = np.array([np.asarray(s.axis, dtype=np.float64) for s in sections])
        s_e1 = np.empty_like(s_axes)
        s_e2 = np.empty_like(s_axes)
        for k, s in enumerate(sections):
            e1, e2 = cap_basis(s.axis)
            s_e1[k] = e1
            s_e2[k] = e2
        s_centers = np.array([np.asarray(s.center, dtype=np.float64) for s in sections])
        s_thetas = np.array([float(s.theta) for s in sections])
        s_a = np.array([float(s.a) for s in sections])
        s_b = np.array([float(s.b) for s in sections])
        if bvh_cache is not None and p.name in bvh_cache:
            bvh_obj = bvh_cache[p.name]
        else:
            bvh_obj = (
                make_fcl_bvh(p.collision_mesh) if p.collision_mesh is not None else None
            )
        sdf_payload = None
        if sdf_by_name is not None and p.name in sdf_by_name:
            sdf = sdf_by_name[p.name]
            sdf_payload = _sdf_jnp_payload(sdf)
        out.append(
            _ProbeStatic(
                name=p.name,
                target_LPS=np.asarray(p.target_LPS, dtype=np.float64),
                shank_tips_local=tips,
                pivot_local=pivot,
                assigned_hole=hole,
                arc_idx=int(arc_idx),
                section_axes=s_axes,
                section_e1=s_e1,
                section_e2=s_e2,
                section_centers=s_centers,
                section_cos_theta=np.cos(s_thetas),
                section_sin_theta=np.sin(s_thetas),
                section_a=s_a,
                section_b=s_b,
                bvh_obj=bvh_obj,
                sdf_data=sdf_payload,
            )
        )
    return out


def _max_g_threading(
    static: _ProbeStatic,
    *,
    ap_deg: float,
    ml_deg: float,
    spin_deg: float,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
) -> NDArray:
    """Per-(shank × section) threading ``g`` values at the given pose.

    Returns a flat 1-D array (shape ``(n_shanks * n_sections,)``).
    Vectorised across sections using the precomputed ``cap_basis``
    arrays on ``static`` — avoids the per-call ``np.cross`` cost
    inside the SLSQP fn loop.
    """
    R, pose_tip = pose_from_optimizer_vars(
        target_LPS=static.target_LPS,
        ap_deg=ap_deg,
        ml_deg=ml_deg,
        spin_deg=spin_deg,
        offset_R_mm=0.0,
        offset_A_mm=0.0,
        past_target_mm=0.0,
        recording_center_local=static.pivot_local,
    )
    tips_local = static.shank_tips_local
    if tips_local.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    pose_tip = np.asarray(pose_tip, dtype=np.float64)
    shaft_dir_world = R @ np.array([0.0, 0.0, 1.0])
    # tip world position per shank: tip_world = R @ tip_local + pose_tip
    tip_world = (R @ tips_local.T).T + pose_tip  # shape (n_shanks, 3)
    base_world = tip_world + shaft_length_mm * shaft_dir_world  # (n_shanks, 3)

    s_axes = static.section_axes  # (S, 3)
    s_centers = static.section_centers  # (S, 3)
    s_e1 = static.section_e1
    s_e2 = static.section_e2
    s_c = static.section_cos_theta  # (S,)
    s_s = static.section_sin_theta  # (S,)
    s_a = static.section_a
    s_b = static.section_b

    # For each (shank, section): line through (tip_world, base_world)
    # intersected with section plane (center, axis). t = dot(c - p0, n)
    # / dot(d, n). For each section, axis n is the plane normal.
    # Vectorise across sections by broadcasting.
    n_shanks = tip_world.shape[0]
    n_sections = s_axes.shape[0]
    out = np.empty((n_shanks, n_sections), dtype=np.float64)
    line_d = base_world - tip_world  # (n_shanks, 3)
    for k in range(n_sections):
        n = s_axes[k]
        c = s_centers[k]
        denom = line_d @ n  # (n_shanks,)
        # Where denom == 0 → parallel; set g = +inf (matches
        # shaft_section_oval_value behaviour).
        with np.errstate(divide="ignore", invalid="ignore"):
            t_arr = (c - tip_world) @ n / denom
        pts = tip_world + t_arr[:, None] * line_d  # (n_shanks, 3)
        rel = pts - c
        u_world = rel @ s_e1[k]
        v_world = rel @ s_e2[k]
        u_local = s_c[k] * u_world + s_s[k] * v_world
        v_local = -s_s[k] * u_world + s_c[k] * v_world
        g = (u_local / s_a[k]) ** 2 + (v_local / s_b[k]) ** 2 - 1.0
        g = np.where(np.abs(denom) < 1e-12, np.inf, g)
        out[:, k] = g
    return out.reshape(-1)


def _signed_pair_clearance(
    obj_a: fcl.CollisionObject, obj_b: fcl.CollisionObject
) -> float:
    """Signed clearance between two BVH probes.

    python-fcl's BVH-vs-BVH FCL has two limitations:

    1. ``fcl.distance(enable_signed_distance=True)`` returns ``0`` when
       the meshes either touch OR overlap. It does NOT distinguish:
       verified empirically that identical-pose probes give 0, and
       probes offset by 0.5 mm (still overlapping) also give 0; only
       fully separated meshes return positive.
    2. ``fcl.collide.penetration_depth`` is essentially garbage for
       thin BVH-vs-BVH meshes. Multi-mm depths for clearly-touching
       meshes; small or zero depths for severe overlap. Don't trust
       these values as actual penetration depth.

    Reliable signals: positive ``fcl.distance`` (true clearance) and
    boolean ``fcl.collide`` (has contacts ⇒ colliding).

    Return convention:
      - positive value: actual clearance in mm
      - ``-1.0``: colliding (sentinel; depth unknown)
      - ``0.0``: touching exactly (rare; fcl.distance=0, no contacts)
    """
    d_req = fcl.DistanceRequest(enable_signed_distance=True)
    d_res = fcl.DistanceResult()
    fcl.distance(obj_a, obj_b, d_req, d_res)
    d = float(d_res.min_distance)
    if d > 0.0:
        return d
    c_req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
    c_res = fcl.CollisionResult()
    fcl.collide(obj_a, obj_b, c_req, c_res)
    if c_res.contacts:
        return -1.0  # collision sentinel
    return 0.0


def _update_pose_and_pairwise_clearances(
    y: NDArray,
    statics: list[_ProbeStatic],
    n_arcs: int,
) -> list[float]:
    """Refresh every probe's BVH transform from the reduced vector ``y``
    and return the list of pairwise signed clearances (length
    ``K*(K-1)/2`` for ``K`` probes-with-BVH, in lex order).

    Probes without a ``bvh_obj`` are skipped from the pairwise list.

    When the statics' ``sdf_data`` is populated, uses the JAX SDF
    backend for clearance instead of FCL BVH+collide. Smooth + signed
    through overlap (FCL clamps at 0 on overlap, then needs a separate
    collide call for depth).
    """
    arc_aps = np.asarray(y[:n_arcs], dtype=np.float64)
    valid: list[int] = []
    # When SDF backend is active, store world poses for each probe so
    # the pair queries can reuse them without redundant pose math.
    poses_world: dict[int, tuple] = {}
    use_sdf = any(st.sdf_data is not None for st in statics)
    for i, st in enumerate(statics):
        if st.bvh_obj is None:
            continue
        ml_i, spin_i = _ml_spin_from_y(y, n_arcs, i)
        ap_i = float(arc_aps[st.arc_idx])
        R, pose_tip = pose_from_optimizer_vars(
            target_LPS=st.target_LPS,
            ap_deg=ap_i,
            ml_deg=ml_i,
            spin_deg=spin_i,
            offset_R_mm=0.0,
            offset_A_mm=0.0,
            past_target_mm=0.0,
            recording_center_local=st.pivot_local,
        )
        # FCL setTransform is only needed for the BVH fallback path
        # below. SDF backend (when active) consumes (R, pose_tip)
        # directly via poses_world, so the setTransform is wasted work.
        if not use_sdf:
            st.bvh_obj.setTransform(
                fcl.Transform(
                    np.ascontiguousarray(R, dtype=np.float64),
                    np.ascontiguousarray(pose_tip, dtype=np.float64),
                )
            )
        valid.append(i)
        if use_sdf:
            poses_world[i] = (R, pose_tip)
    out: list[float] = []
    if use_sdf:
        # SDF backend: query JAX kernel per pair. Uses the same probe
        # poses computed above; surface points + SDF grids come from
        # each probe's ``sdf_data``.
        import jax.numpy as jnp

        from aind_low_point.optimization.sdf_jax import (
            pairwise_signed_clearance_dual_hard_mins_jit,
        )

        for a in range(len(valid)):
            ia = valid[a]
            R_a, t_a = poses_world[ia]
            sa = statics[ia].sdf_data
            for b in range(a + 1, len(valid)):
                ib = valid[b]
                R_b, t_b = poses_world[ib]
                sb = statics[ib].sdf_data
                if sa is None or sb is None:
                    # Probe with no SDF falls back to BVH for that pair.
                    out.append(
                        _signed_pair_clearance(
                            statics[ia].bvh_obj, statics[ib].bvh_obj
                        )
                    )
                    continue
                hbb, hbs, hss = pairwise_signed_clearance_dual_hard_mins_jit(
                    jnp.asarray(R_a, dtype=jnp.float32),
                    jnp.asarray(t_a, dtype=jnp.float32),
                    jnp.asarray(R_b, dtype=jnp.float32),
                    jnp.asarray(t_b, dtype=jnp.float32),
                    sa["grid"], sa["origin"], sa["spacing"],
                    sb["grid"], sb["origin"], sb["spacing"],
                    sa["surface"], sb["surface"],
                    sa["shank_centers"], sa["shank_halves"],
                    sb["shank_centers"], sb["shank_halves"],
                )
                out.append(float(jnp.minimum(jnp.minimum(hbb, hbs), hss)))
        return out
    for a in range(len(valid)):
        ia = valid[a]
        for b in range(a + 1, len(valid)):
            ib = valid[b]
            out.append(_signed_pair_clearance(statics[ia].bvh_obj, statics[ib].bvh_obj))
    return out


def _softplus_squared(values: NDArray) -> float:
    """``Σ softplus(v)²`` — smooth positive penalty.

    ``softplus(v) = log(1 + exp(v))`` is positive, smooth, monotone,
    asymptotic to ``v`` for large ``v`` and to ``log(2)`` near
    ``v = 0``. Using its square gives quadratic-tail behaviour when
    the argument is well above zero, matching the ``ReLU(...)²`` shape
    used by the hard penalties but without the kink at zero. The
    near-origin penalty (``log(2)² ≈ 0.48``) is intentionally small;
    this term is a soft preference rather than a hard limit.
    """
    if values.size == 0:
        return 0.0
    # Numerically-stable softplus: max(0, v) + log(1 + exp(-|v|))
    v = np.asarray(values, dtype=np.float64)
    sp = np.maximum(0.0, v) + np.log1p(np.exp(-np.abs(v)))
    return float(np.sum(sp * sp))


# ---------------------------------------------------------------------------
# Reduced objective
# ---------------------------------------------------------------------------


def _reduced_objective(
    y: NDArray,
    statics: list[_ProbeStatic],
    n_arcs: int,
    weights: JointWeights,
) -> float:
    """Surrogate objective the joint reranker minimises.

    Returns a scalar ``J(y)``; see :class:`JointWeights` for the
    component weights. The reduced vector layout is documented in
    :func:`_reduced_layout`.
    """
    n_probes = len(statics)
    arc_aps = np.asarray(y[:n_arcs], dtype=np.float64)

    # Threading penalty (with tolerance).
    j_thread = 0.0
    for i, st in enumerate(statics):
        ml_i, spin_i = _ml_spin_from_y(y, n_arcs, i)
        ap_i = float(arc_aps[st.arc_idx])
        gs = _max_g_threading(st, ap_deg=ap_i, ml_deg=ml_i, spin_deg=spin_i)
        if gs.size > 0:
            excess = np.maximum(0.0, gs - weights.threading_oval_tolerance)
            j_thread += float(np.sum(excess * excess))

    # Cross-arc AP separation (between any two arcs).
    j_arc_ap = 0.0
    for a in range(n_arcs):
        for b in range(a + 1, n_arcs):
            diff = abs(float(arc_aps[a]) - float(arc_aps[b]))
            short = max(0.0, weights.min_arc_ap_sep_deg - diff)
            j_arc_ap += short * short

    # Within-arc ML separation.
    j_ml = 0.0
    for i in range(n_probes):
        for j in range(i + 1, n_probes):
            if statics[i].arc_idx != statics[j].arc_idx:
                continue
            ml_i, _ = _ml_spin_from_y(y, n_arcs, i)
            ml_j, _ = _ml_spin_from_y(y, n_arcs, j)
            diff = abs(float(ml_i) - float(ml_j))
            short = max(0.0, weights.min_intra_arc_ml_sep_deg - diff)
            j_ml += short * short

    # Soft bounds — softplus²(|value| - comfortable).
    arc_bound_vals = np.abs(arc_aps) - weights.comfortable_ap_deg
    j_bounds_ap = _softplus_squared(arc_bound_vals)
    ml_vals = np.array(
        [float(y[n_arcs + 3 * i]) for i in range(n_probes)], dtype=np.float64
    )
    j_bounds_ml = _softplus_squared(np.abs(ml_vals) - weights.comfortable_ml_deg)
    j_bounds = j_bounds_ap + j_bounds_ml

    # Pairwise clearance penalty (BVH-based hybrid). Skipped entirely
    # when λ_clearance = 0 to save the FCL traffic for runs that don't
    # want this signal.
    j_clearance = 0.0
    if weights.lambda_clearance > 0.0:
        clearances = _update_pose_and_pairwise_clearances(y, statics, n_arcs)
        for c in clearances:
            short = max(0.0, weights.min_clearance_mm - c)
            j_clearance += short * short

    # Coverage placeholder (λ defaults to 0); skip the work when unused.
    total = (
        weights.lambda_thread * j_thread
        + weights.lambda_arc_ap * j_arc_ap
        + weights.lambda_ml * j_ml
        + weights.lambda_bounds * j_bounds
        + weights.lambda_clearance * j_clearance
    )
    return float(total)


def compute_fixture_max_violation(
    y: NDArray,
    statics: list[_ProbeStatic],
    n_arcs: int,
    fixture_bvhs: list[fcl.CollisionObject],
) -> float:
    """Refresh probe BVH transforms at ``y``, then return the max
    probe-vs-fixture penetration via raw-mesh FCL signed distance.

    Diagnostic-only — Stage 2 doesn't use this as an optimization
    signal. Use post-polish to flag candidates whose polished pose
    happens to clash with cone / well / headframe geometry.
    """
    if not fixture_bvhs or len(statics) == 0:
        return 0.0
    _update_pose_and_pairwise_clearances(y, statics, n_arcs)
    max_viol = 0.0
    for st in statics:
        if st.bvh_obj is None:
            continue
        for fx in fixture_bvhs:
            d = _signed_pair_clearance(st.bvh_obj, fx)
            if d < 0.0 and -d > max_viol:
                max_viol = -d
    return float(max_viol)


def _evaluate_reduced_metrics(
    y: NDArray,
    statics: list[_ProbeStatic],
    n_arcs: int,
    weights: JointWeights,
    *,
    original_lsap_cost: float,
    fixture_bvhs: list[fcl.CollisionObject] | None = None,
) -> JointRerankMetrics:
    """Compute lex-ranking metrics for a reduced-vector outcome.

    Reports per-group maximum violations (in their native units) plus a
    sum-of-squares aggregate. ``approximate_coverage`` is a cheap
    surrogate: ``Σ_i exp(-||target_i - shaft_tip_i_after_rotation||²)``
    — proxies "shaft passes near the target". The full Gaussian-density
    integral is left to the inner solve.
    """
    n_probes = len(statics)
    arc_aps = np.asarray(y[:n_arcs], dtype=np.float64)

    max_thread = 0.0
    sum_thread_sq = 0.0
    approx_cov = 0.0
    for i, st in enumerate(statics):
        ml_i, spin_i = _ml_spin_from_y(y, n_arcs, i)
        ap_i = float(arc_aps[st.arc_idx])
        gs = _max_g_threading(st, ap_deg=ap_i, ml_deg=ml_i, spin_deg=spin_i)
        if gs.size > 0:
            excess = np.maximum(0.0, gs - weights.threading_oval_tolerance)
            sum_thread_sq += float(np.sum(excess * excess))
            max_thread = max(max_thread, float(excess.max(initial=0.0)))
        # Coverage was a per-probe Gaussian-density line integral here,
        # but the reduced reranker uses rotation-only pose construction
        # — ``pose_from_optimizer_vars`` with ``offset=0, past_target=0``
        # pins the recording center on target. Any rigid distance-to-
        # target metric is then rotation-invariant: every candidate
        # produces the same coverage. We leave ``approximate_coverage``
        # at 0.0 here and rank by ``bounds_softpenalty`` instead (see
        # ``JointRerankMetrics.lex_key``). A useful coverage surrogate
        # would need a region-based target (mixture density) or to
        # include offsets in the reduced problem — both are Stage 3
        # concerns.

    max_arc_ap = 0.0
    sum_arc_ap_sq = 0.0
    for a in range(n_arcs):
        for b in range(a + 1, n_arcs):
            diff = abs(float(arc_aps[a]) - float(arc_aps[b]))
            short = max(0.0, weights.min_arc_ap_sep_deg - diff)
            sum_arc_ap_sq += short * short
            if short > max_arc_ap:
                max_arc_ap = short

    max_ml = 0.0
    sum_ml_sq = 0.0
    for i in range(n_probes):
        for j in range(i + 1, n_probes):
            if statics[i].arc_idx != statics[j].arc_idx:
                continue
            ml_i, _ = _ml_spin_from_y(y, n_arcs, i)
            ml_j, _ = _ml_spin_from_y(y, n_arcs, j)
            diff = abs(float(ml_i) - float(ml_j))
            short = max(0.0, weights.min_intra_arc_ml_sep_deg - diff)
            sum_ml_sq += short * short
            if short > max_ml:
                max_ml = short

    bounds_pen = 0.0
    arc_bound_vals = np.abs(arc_aps) - weights.comfortable_ap_deg
    bounds_pen += _softplus_squared(arc_bound_vals)
    ml_vals = np.array(
        [float(y[n_arcs + 3 * i]) for i in range(n_probes)], dtype=np.float64
    )
    bounds_pen += _softplus_squared(np.abs(ml_vals) - weights.comfortable_ml_deg)

    # Pairwise clearance violations from BVH signed distance.
    max_clear = 0.0
    sum_clear_sq = 0.0
    if weights.lambda_clearance > 0.0:
        clearances = _update_pose_and_pairwise_clearances(y, statics, n_arcs)
        for c in clearances:
            short = max(0.0, weights.min_clearance_mm - c)
            sum_clear_sq += short * short
            if short > max_clear:
                max_clear = short

    # Diagnostic-only: probe-vs-fixture worst penetration via raw-mesh
    # FCL. Reports the per-cand max overlap with cone/well/headframe
    # at the polished pose. Stage 2 doesn't use this as an optimization
    # signal (the JAX kernel doesn't see fixtures), but ranking pipelines
    # downstream can lex-sort by it via ``lex_key_with_fixtures``.
    max_fixture_viol = 0.0
    if fixture_bvhs:
        # ``_update_pose_and_pairwise_clearances`` already set probe BVH
        # transforms while computing probe-probe clearances. If that
        # path was disabled (lambda_clearance == 0), refresh transforms
        # here so the FCL distance below is meaningful.
        if weights.lambda_clearance <= 0.0:
            # Recompute probe pose transforms without computing pair
            # clearances (we only need the transforms set).
            _update_pose_and_pairwise_clearances(y, statics, n_arcs)
        for st in statics:
            if st.bvh_obj is None:
                continue
            for fx_bvh in fixture_bvhs:
                d = _signed_pair_clearance(st.bvh_obj, fx_bvh)
                if d < 0.0 and -d > max_fixture_viol:
                    max_fixture_viol = float(-d)

    max_viol = max(max_thread, max_arc_ap, max_ml, max_clear)
    sum_viol_sq = sum_thread_sq + sum_arc_ap_sq + sum_ml_sq + sum_clear_sq
    return JointRerankMetrics(
        max_violation=float(max_viol),
        sum_violation_sq=float(sum_viol_sq),
        max_violation_threading=float(max_thread),
        max_violation_arc_ap_sep=float(max_arc_ap),
        max_violation_intra_arc_ml_sep=float(max_ml),
        max_violation_clearance=float(max_clear),
        max_violation_fixture=float(max_fixture_viol),
        approximate_coverage=float(approx_cov),
        bounds_softpenalty=float(bounds_pen),
        original_lsap_cost=float(original_lsap_cost),
    )


# ---------------------------------------------------------------------------
# Multi-start warm-start construction
# ---------------------------------------------------------------------------


def _build_starts(
    statics: list[_ProbeStatic],
    aa: ArcAssignment,
    pose_features: dict[tuple[str, int], PoseFeatures],
    n_arcs: int,
) -> list[NDArray]:
    """Three reduced-vector warm starts per (H, A); see module docstring.

    Each start is a length-``n_arcs + 2K`` vector laid out by
    :func:`_reduced_layout`.
    """
    n_probes = len(statics)
    n_vars = n_arcs + 3 * n_probes  # (ml, sx, sy) per probe under Patch B

    # Per-arc on-arc probe lists for the AP-interval intersection.
    on_arc: dict[int, list[int]] = {a: [] for a in range(n_arcs)}
    for i, st in enumerate(statics):
        on_arc[st.arc_idx].append(i)

    # Helper: slot-major-axis spin (deg) per probe, plus its (sx, sy)
    # form for the reduced y vector.
    spin_warm: list[float] = []
    spin_warm_xy: list[tuple[float, float]] = []
    for st in statics:
        spin_rad = float(np.pi / 2 - st.assigned_hole.slot_theta_rad)
        spin_warm.append(float(np.rad2deg(spin_rad)))
        spin_warm_xy.append(
            (float(np.cos(spin_rad)), float(np.sin(spin_rad)))
        )

    # Helper: required ML at a given ap_arc for probe i.
    def _ml_at_ap(probe_idx: int, ap: float) -> float:
        st = statics[probe_idx]
        feat = pose_features.get((st.name, int(st.assigned_hole.id)))
        if feat is None:
            return 0.0
        # Use the same closed form as in pose_features._ml_for_ap. To
        # avoid the import cycle of a private helper, recompute directly:
        b = st.target_LPS - np.asarray(
            st.assigned_hole.sections[-1].center, dtype=np.float64
        )
        norm = float(np.linalg.norm(b))
        if norm < 1e-12:
            return float(feat.required_ml_deg)
        b = b / norm
        ap_rad = float(np.deg2rad(ap))
        denom = float(np.sin(ap_rad) * float(b[1]) - np.cos(ap_rad) * float(b[2]))
        return float(np.rad2deg(np.arctan2(float(b[0]), denom)))

    # Start 1: partitioner centroid + slot-major spin + ml=0.
    # Matches ``_build_initial_x`` in ``optimize.py`` so the joint
    # reranker has a faithful baseline that re-creates the existing
    # warm-start. ``arc_centroids_deg`` from the partitioner uses
    # ``kinematics.required_ap_deg`` whose convention now matches
    # the rig-frame AP that aligns the shaft with the bore
    # (``atan2(-axis_y, axis_z)``) — same sign as the closed form in
    # :mod:`pose_features`.
    start1 = np.zeros(n_vars, dtype=np.float64)
    for a in range(n_arcs):
        start1[a] = float(aa.arc_centroids_deg[a])
    for i, st in enumerate(statics):
        start1[n_arcs + 3 * i] = 0.0          # ml
        start1[n_arcs + 3 * i + 1] = spin_warm_xy[i][0]  # sx
        start1[n_arcs + 3 * i + 2] = spin_warm_xy[i][1]  # sy

    # Start 2: AP-interval midpoint per arc (across on-arc probes) + slot spin.
    start2 = np.zeros(n_vars, dtype=np.float64)
    for a in range(n_arcs):
        members = on_arc.get(a, [])
        if not members:
            start2[a] = float(aa.arc_centroids_deg[a])
            continue
        intervals: list[tuple[float, float]] = []
        required_aps: list[float] = []
        for i in members:
            st = statics[i]
            feat = pose_features.get((st.name, int(st.assigned_hole.id)))
            if feat is None:
                continue
            intervals.append(feat.ap_interval_deg)
            required_aps.append(feat.required_ap_deg)
        if intervals:
            lo = max(iv[0] for iv in intervals)
            hi = min(iv[1] for iv in intervals)
            if hi >= lo:
                start2[a] = 0.5 * (lo + hi)
            elif required_aps:
                # No overlap; fall back to required-AP of probe closest
                # to the partitioner's centroid for this arc.
                target_ap = float(aa.arc_centroids_deg[a])
                idx = int(np.argmin([abs(rap - target_ap) for rap in required_aps]))
                start2[a] = float(required_aps[idx])
            else:
                start2[a] = float(aa.arc_centroids_deg[a])
        else:
            start2[a] = float(aa.arc_centroids_deg[a])
    for i, st in enumerate(statics):
        ap_i = float(start2[st.arc_idx])
        start2[n_arcs + 3 * i] = _ml_at_ap(i, ap_i)            # ml
        start2[n_arcs + 3 * i + 1] = spin_warm_xy[i][0]         # sx
        start2[n_arcs + 3 * i + 2] = spin_warm_xy[i][1]         # sy

    # Start 3: same APs as start 2, alternating spin offsets per on-arc
    # neighbour (odd-indexed gets spin + 180°, i.e. (sx, sy) negated).
    start3 = start2.copy()
    for a in range(n_arcs):
        members = on_arc.get(a, [])
        for k, i in enumerate(members):
            if k % 2 == 1:
                start3[n_arcs + 3 * i + 1] = -spin_warm_xy[i][0]
                start3[n_arcs + 3 * i + 2] = -spin_warm_xy[i][1]

    return [start1, start2, start3]


# ---------------------------------------------------------------------------
# Bounds + SLSQP wrapper
# ---------------------------------------------------------------------------


def _reduced_bounds(
    n_arcs: int,
    n_probes: int,
    head_pitch_deg: float,
) -> list[tuple[float, float]]:
    """Box bounds for the reduced ``y = (arc_aps, (ml, sx, sy) × P)``.

    AP bound shifted by head pitch; ML clamped to ±60°. Spin is
    parameterized as ``(sx, sy) ∝ (cos θ, sin θ)`` on the unit circle
    (Patch B): bounds are box-uniform ``[-1.5, +1.5]`` on each
    component, which excludes the ``(0, 0)`` ``atan2`` singularity and
    keeps SLSQP's internal step-size scaling well-conditioned (vs
    angle-space ±180° bounds which clip the wraparound and bound
    SLSQP's ability to flow across spin=±180°).
    """
    bounds: list[tuple[float, float]] = []
    for _ in range(n_arcs):
        bounds.append((-60.0 + head_pitch_deg, +60.0 + head_pitch_deg))
    for _ in range(n_probes):
        bounds.append((-60.0, +60.0))    # ml
        # (sx, sy) bounds at ±1.1 — the unit_circle_penalty pulls
        # magnitude toward 1; a small allowance keeps the optimizer
        # away from a rectangular constraint boundary while letting
        # the penalty do the work.
        bounds.append((-1.1, +1.1))       # sx
        bounds.append((-1.1, +1.1))       # sy
    return bounds


def _wrap_spin_to_bounds(y: NDArray, n_arcs: int, n_probes: int) -> NDArray:
    """No-op pass-through (kept for backward compat with callers).

    Under the (sx, sy) reduced layout (Patch B), the wraparound issue
    is gone — both components live in a continuous bounded box. This
    helper is retained so existing callers (``_slsqp_reduced``) don't
    need to change, but it just returns a copy of ``y``.
    """
    return np.asarray(y, dtype=np.float64).copy()


def _slsqp_reduced(
    y0: NDArray,
    statics: list[_ProbeStatic],
    n_arcs: int,
    weights: JointWeights,
    *,
    bounds: list[tuple[float, float]],
    max_iter: int,
) -> tuple[NDArray, float]:
    """Run a single SLSQP polish from ``y0`` and return ``(y_opt, fn_opt)``.

    ``fn_opt`` is the final objective value at ``y_opt`` (or ``inf`` if
    SLSQP raised). Callers use it as a cheap catastrophic-infeasibility
    signal — a huge ``fn_opt`` means the polish couldn't reduce
    threading/clearance/sep violations meaningfully, and downstream
    metric eval can be skipped.
    """
    y0 = _wrap_spin_to_bounds(y0, n_arcs, len(statics))

    use_jax = any(st.sdf_data is not None for st in statics)
    if use_jax:
        from aind_low_point.optimization.joint_rerank_jax import (
            make_jax_reduced_objective,
        )

        fn, jac = make_jax_reduced_objective(statics, n_arcs, weights)
        try:
            # Bounds-only soft-penalty objective → L-BFGS-B (proper
            # limited-memory BFGS with no wasted QP subproblem; SLSQP
            # was hitting maxiter on this without converging).
            result = minimize(
                fn,
                y0,
                method="L-BFGS-B",
                jac=jac,
                bounds=bounds,
                options={"maxiter": max_iter, "ftol": 1e-4, "gtol": 1e-5,
                         "disp": False},
            )
        except Exception:
            return y0, float("inf")
        return np.asarray(result.x, dtype=np.float64), float(result.fun)

    def fn(v):
        return _reduced_objective(
            np.asarray(v, dtype=np.float64), statics, n_arcs, weights
        )

    try:
        result = minimize(
            fn,
            y0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-6, "gtol": 1e-5,
                     "disp": False},
        )
    except Exception:
        return y0, float("inf")
    return np.asarray(result.x, dtype=np.float64), float(result.fun)


# ---------------------------------------------------------------------------
# Per-(H, A) scoring
# ---------------------------------------------------------------------------


def score_joint(
    ha: HoleAssignment,
    aa: ArcAssignment,
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    pose_features: dict[tuple[str, int], PoseFeatures],
    *,
    weights: JointWeights = JointWeights(),
    head_pitch_deg: float = 0.0,
    reduced_slsqp_max_iter: int = 50,
    original_lsap_cost: float = float("nan"),
    bvh_cache: dict[str, fcl.CollisionObject | None] | None = None,
    sdf_by_name: dict | None = None,
    fixture_bvhs: list[fcl.CollisionObject] | None = None,
    y0_override: NDArray | None = None,
    skip_spin_restore: bool = False,
) -> JointCandidate:
    """Run the reduced-SLSQP scoring for one (hole, arc) candidate.

    By default, tries three warm starts (see module docstring) and
    returns the candidate with the lex-best metrics.

    When ``y0_override`` is supplied, uses ONLY that single warm-start
    vector (skipping ``_build_starts``'s multi-seed logic). Useful when
    the caller has already produced a high-quality starting point —
    e.g., from a batched spin-restoration pass over all candidates.

    When ``skip_spin_restore=True``, the per-start spin-restoration
    step is bypassed entirely (the y0 is assumed already restored or
    intentionally not restored). Combine with ``y0_override`` to give
    each candidate a pre-spin-restored seed and skip the per-candidate
    ~500 ms spin-restore cost — a ~40% speedup on the full polish run.
    """
    n_probes = len(probes)
    n_arcs = max(aa.probe_to_arc_idx.values()) + 1 if aa.probe_to_arc_idx else 1
    _t0 = time.perf_counter()
    statics = _build_probe_static(
        probes, holes, ha, aa, bvh_cache=bvh_cache, sdf_by_name=sdf_by_name
    )
    _STAGE2_TIMINGS["build_probe_static"] += time.perf_counter() - _t0
    _t0 = time.perf_counter()
    if y0_override is not None:
        starts = [np.asarray(y0_override, dtype=np.float64)]
    else:
        starts = _build_starts(statics, aa, pose_features, n_arcs)
    _STAGE2_TIMINGS["build_starts"] += time.perf_counter() - _t0
    bounds = _reduced_bounds(n_arcs, n_probes, head_pitch_deg)

    best_y: NDArray | None = None
    best_metrics: JointRerankMetrics | None = None
    # Per-cand spin restore is intentionally NOT done here. Production
    # polish (``polish_all_with_batched_spin_restore``) does a GPU-
    # batched spin sweep upstream; direct ``score_joint`` callers rely
    # on the spin seed from ``_build_starts``. The legacy per-cand FCL
    # brute sweep + JAX 2D-angle sweep were both removed when Patch B
    # made the JAX path a no-op and the FCL fallback dead in production
    # (SDFs are always present). ``skip_spin_restore`` is kept as a
    # backward-compat kwarg only; it has no effect.
    for y0 in starts:
        _t0 = time.perf_counter()
        y_opt, fn_opt = _slsqp_reduced(
            y0, statics, n_arcs, weights, bounds=bounds, max_iter=reduced_slsqp_max_iter
        )
        _STAGE2_TIMINGS["slsqp"] += time.perf_counter() - _t0
        _t0 = time.perf_counter()
        # Catastrophic-infeasibility shortcut: when SLSQP exits with a
        # huge objective, the cand will sort to the bottom of any lex
        # ranking regardless of detailed metrics. Skip the full FCL
        # pair-clearance sweep in metric_eval; use a sentinel max_viol
        # so the cand still has a comparable lex_key.
        # Threshold: ``λ × max_viol²`` with λ ≈ 100 and max_viol = 10 gives
        # ``fn ≈ 10000``. Anything above this is genuinely uncoverable;
        # below this and the polish endpoint is interesting enough to
        # measure exactly.
        if fn_opt > _CATASTROPHIC_FN_THRESHOLD:
            m = JointRerankMetrics(
                max_violation=_CATASTROPHIC_MAX_VIOL_SENTINEL,
                sum_violation_sq=fn_opt,
                max_violation_threading=_CATASTROPHIC_MAX_VIOL_SENTINEL,
                max_violation_arc_ap_sep=0.0,
                max_violation_intra_arc_ml_sep=0.0,
                max_violation_clearance=0.0,
                original_lsap_cost=float(original_lsap_cost),
            )
        else:
            m = _evaluate_reduced_metrics(
                y_opt,
                statics,
                n_arcs,
                weights,
                original_lsap_cost=original_lsap_cost,
                fixture_bvhs=fixture_bvhs,
            )
        _STAGE2_TIMINGS["metric_eval"] += time.perf_counter() - _t0
        if best_metrics is None or m.lex_key() < best_metrics.lex_key():
            best_metrics = m
            best_y = y_opt

    assert best_y is not None and best_metrics is not None
    return JointCandidate(
        ha=ha, aa=aa, n_arcs=n_arcs, reduced_y=best_y, metrics=best_metrics
    )


# ---------------------------------------------------------------------------
# Reduced → full variable expansion
# ---------------------------------------------------------------------------


def expand_reduced_solution_to_full_x(
    jc: JointCandidate,
    layout: VariableLayout,
    probe_names_in_order: list[str],
) -> NDArray:
    """Map a reduced ``y`` to a full ``VariableLayout``-conformant ``x``.

    Layout: ``x = [ap_arc_0, ..., ap_arc_{A-1}, (ml, spin, off_R, off_A,
    depth)_0, ..., (ml, spin, off_R, off_A, depth)_{K-1}]``.

    Reduced ``y`` only contains ``(ap_arc, ml, spin)`` — offsets and
    depth start at zero in the full vector.

    ``probe_names_in_order`` is the order of probes the layout
    references; the reduced vector packs probes in this same order (see
    :func:`_build_probe_static`).
    """
    n_arcs = jc.n_arcs
    x = np.zeros(layout.n_vars, dtype=np.float64)
    x[:n_arcs] = np.asarray(jc.reduced_y[:n_arcs], dtype=np.float64)
    for i, _name in enumerate(probe_names_in_order):
        # Stage 2 reduced y stores spin as (sx, sy) (Patch B); convert
        # back to scalar spin_deg in [-180, 180] for the Stage 3 full
        # x layout, which still uses the angle parameterization.
        ml = float(jc.reduced_y[n_arcs + 3 * i])
        sx = float(jc.reduced_y[n_arcs + 3 * i + 1])
        sy = float(jc.reduced_y[n_arcs + 3 * i + 2])
        spin = float(np.degrees(np.arctan2(sy, sx)))
        off = layout.num_arcs + 5 * i
        x[off + 0] = ml
        x[off + 1] = spin
        x[off + 2] = 0.0
        x[off + 3] = 0.0
        x[off + 4] = 0.0
    return x


def _push_restore_full_x(  # noqa: C901
    x: NDArray,
    *,
    ctx,
    max_iter: int = 30,
    margin_mm: float = 0.02,
    off_bound_mm: float = 0.5,
    depth_bound_mm: float = 3.0,
) -> NDArray:
    """Bounded offset/depth push to resolve residual overlap after spin
    restoration. Operates on the *full* ``x`` (offsets and depth are
    not in the reduced ``y``).

    Same algorithm as the spin restoration but operating in translation
    space: find the worst pair, push each probe along the line between
    their pose tips by half the penetration (+ margin), realised as
    deltas in ``(off_R, off_A, past_target_mm)`` and clamped to bounds.

    The push slightly violates kinematic intent (offsets move the
    recording-array centre away from the target; depth re-aligns it
    along the shaft). Use only as a fallback when spin alone cannot
    fully clear all pairs.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    probes = ctx.probes
    headstage_objs = ctx.headstage_fcl_objs
    n_arcs = ctx.layout.num_arcs
    n_probes = len(probes)
    if n_probes < 2 or any(o is None for o in headstage_objs):
        return x

    def _project(R: NDArray, d_world: NDArray) -> tuple[float, float, float]:
        """``d_world`` in LPS → ``(δoff_R, δoff_A, δdepth)`` that
        translate pose_tip by ``d_world``."""
        shaft = R @ np.array([0.0, 0.0, 1.0])
        d_shaft = float(np.dot(d_world, shaft))
        # off_LPS = (-off_R, -off_A, 0). Cover xy via offsets; shaft via depth.
        return -float(d_world[0]), -float(d_world[1]), -d_shaft

    def _refresh_all() -> list[NDArray]:
        arc_aps = np.asarray(x[:n_arcs], dtype=np.float64)
        pose_tips: list[NDArray] = []
        for i, p in enumerate(probes):
            offset = n_arcs + 5 * i
            ml = float(x[offset + 0])
            spin = float(x[offset + 1])
            off_R = float(x[offset + 2])
            off_A = float(x[offset + 3])
            depth = float(x[offset + 4])
            ap = float(arc_aps[ctx.layout.arc_ids.index(p.arc_id)])
            tips = np.asarray(p.shank_tips_local, dtype=np.float64)
            geom = p.recording_geom
            if tips.shape[0] > 0:
                pivot = np.array(
                    [tips[:, 0].mean(), tips[:, 1].mean(), geom.active_center_mm],
                    dtype=np.float64,
                )
            else:
                pivot = np.array([0.0, 0.0, geom.active_center_mm], dtype=np.float64)
            R, pose_tip = pose_from_optimizer_vars(
                target_LPS=p.target_LPS,
                ap_deg=ap,
                ml_deg=ml,
                spin_deg=spin,
                offset_R_mm=off_R,
                offset_A_mm=off_A,
                past_target_mm=depth,
                recording_center_local=pivot,
            )
            R = np.asarray(R, dtype=np.float64)
            pose_tip = np.asarray(pose_tip, dtype=np.float64)
            obj = headstage_objs[i]
            if obj is not None:
                obj.setTransform(fcl.Transform(R, pose_tip))
            pose_tips.append(pose_tip)
        return pose_tips

    dist_req = fcl.DistanceRequest(enable_signed_distance=True)
    coll_req = fcl.CollisionRequest(num_max_contacts=4, enable_contact=True)

    def _signed_clear(a, b) -> float:
        oa, ob = headstage_objs[a], headstage_objs[b]
        if oa is None or ob is None:
            return float("inf")
        d_res = fcl.DistanceResult()
        fcl.distance(oa, ob, dist_req, d_res)
        d = float(d_res.min_distance)
        if d > 0.0:
            return d
        c_res = fcl.CollisionResult()
        fcl.collide(oa, ob, coll_req, c_res)
        if c_res.contacts:
            return -float(max(c.penetration_depth for c in c_res.contacts))
        return 0.0

    for _ in range(max_iter):
        pose_tips = _refresh_all()
        worst_a, worst_b, worst_d = -1, -1, 1e9
        for i in range(n_probes):
            if headstage_objs[i] is None:
                continue
            for j in range(i + 1, n_probes):
                if headstage_objs[j] is None:
                    continue
                d = _signed_clear(i, j)
                if d < worst_d:
                    worst_d, worst_a, worst_b = d, i, j
        if worst_d > margin_mm or worst_a < 0:
            return x
        push = pose_tips[worst_b] - pose_tips[worst_a]
        nrm = float(np.linalg.norm(push))
        push = push / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
        half_shift = (max(0.0, -worst_d) + margin_mm) / 2.0
        # Apply per-probe; compute R in-line from current x for chain rule.
        arc_aps = np.asarray(x[:n_arcs], dtype=np.float64)
        prev_x = x.copy()
        for sign, idx in ((-1.0, worst_a), (+1.0, worst_b)):
            p = probes[idx]
            offset = n_arcs + 5 * idx
            ap = float(arc_aps[ctx.layout.arc_ids.index(p.arc_id)])
            ml = float(x[offset + 0])
            spin = float(x[offset + 1])
            tips = np.asarray(p.shank_tips_local, dtype=np.float64)
            geom = p.recording_geom
            pivot = (
                np.array(
                    [tips[:, 0].mean(), tips[:, 1].mean(), geom.active_center_mm],
                    dtype=np.float64,
                )
                if tips.shape[0] > 0
                else np.array([0.0, 0.0, geom.active_center_mm], dtype=np.float64)
            )
            R, _ = pose_from_optimizer_vars(
                target_LPS=p.target_LPS,
                ap_deg=ap,
                ml_deg=ml,
                spin_deg=spin,
                offset_R_mm=float(x[offset + 2]),
                offset_A_mm=float(x[offset + 3]),
                past_target_mm=float(x[offset + 4]),
                recording_center_local=pivot,
            )
            R = np.asarray(R, dtype=np.float64)
            d_off_R, d_off_A, d_depth = _project(R, sign * half_shift * push)
            x[offset + 2] = float(
                np.clip(x[offset + 2] + d_off_R, -off_bound_mm, off_bound_mm)
            )
            x[offset + 3] = float(
                np.clip(x[offset + 3] + d_off_A, -off_bound_mm, off_bound_mm)
            )
            x[offset + 4] = float(
                np.clip(x[offset + 4] + d_depth, -depth_bound_mm, depth_bound_mm)
            )
        # Halt if clamping zeroed out all motion (otherwise we'd loop).
        if np.allclose(x, prev_x):
            return x
    return x

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _format_candidate_row(rank: int, jc: JointCandidate) -> str:
    return (
        f"  {rank:>3} | max_viol={jc.metrics.max_violation:7.4f} "
        f"| thread={jc.metrics.max_violation_threading:7.4f} "
        f"| ap_sep={jc.metrics.max_violation_arc_ap_sep:6.3f} "
        f"| ml_sep={jc.metrics.max_violation_intra_arc_ml_sep:6.3f} "
        f"| sum_viol²={jc.metrics.sum_violation_sq:8.3f} "
        f"| cov_proxy={jc.metrics.approximate_coverage:6.3f} "
        f"| LSAP={jc.metrics.original_lsap_cost:+8.3f}"
    )


def _lookup_seed_equivalent_rank(
    candidates: Iterable[JointCandidate],
    seed_to_hole: dict[str, int] | None,
    seed_to_arc_idx: dict[str, int] | None,
) -> int:
    """Find the 1-based rank of the seed-equivalent (H, A) in a sorted
    candidate list, or ``-1`` if not present.

    Arc indices are matched up to canonical permutation: two probe→arc
    dicts are equivalent if they partition the probes the same way,
    regardless of which integer labels each arc gets.
    """
    if seed_to_hole is None or seed_to_arc_idx is None:
        return -1

    def _signature(p2a: dict[str, int]) -> tuple[tuple[str, ...], ...]:
        groups: dict[int, list[str]] = {}
        for name, idx in p2a.items():
            groups.setdefault(idx, []).append(name)
        return tuple(sorted(tuple(sorted(g)) for g in groups.values()))

    seed_sig = _signature(seed_to_arc_idx)
    for rank, jc in enumerate(candidates, start=1):
        if dict(jc.ha.probe_to_hole) != seed_to_hole:
            continue
        if _signature(dict(jc.aa.probe_to_arc_idx)) != seed_sig:
            continue
        return rank
    return -1


# ---------------------------------------------------------------------------
# Inner-solve worker (for ProcessPoolExecutor)
# ---------------------------------------------------------------------------
#
# The inner SLSQP for each surviving JointCandidate is independent of the
# others — each builds its own ``OptimizerContext`` (FCL CollisionObjects
# can't cross process boundaries anyway) and runs its own multi-stage
# SLSQP. Run them in parallel via a ProcessPool to use all cores.

_INNER_WORKER_STATE: dict = {}


def _inner_solve_worker_init(
    probes,
    holes,
    weights,
    probe_names,
    threading_oval_tolerance,
    clearance_overlap_allowance_mm,
    subject_from_rig_rot,
    slsqp_max_iter,
    slsqp_constrained,
    two_stage_inner,
    feasibility_max_iter,
    final_feasibility_cleanup,
    polish_method,
    feasibility_threshold,
    verbose,
    sdf_by_name=None,
) -> None:
    """Set up per-worker shared state from the parent process. On Linux
    (fork start method) this is essentially free — the parent's memory
    is copy-on-write. On spawn-mode platforms (macOS/Windows) the args
    get pickled once per worker; trimesh meshes are the bulky bit but
    still tolerable (~few MB per probe kind)."""
    # Stage 3 is FCL-only and never imports JAX, but spawn workers
    # inherit the parent's ``JAX_PLATFORMS=cuda`` env and would each
    # try to acquire the GPU on first import — guaranteed OOM with 15
    # workers. Pin workers to CPU before any JAX import gets pulled in.
    import os as _os
    _os.environ["JAX_PLATFORMS"] = "cpu"
    global _INNER_WORKER_STATE
    # Build per-worker BVH cache ONCE here. Each worker may handle
    # multiple candidates; without this they'd each rebuild BVHs per
    # candidate (7 × ~100 ms × N_candidates_per_worker waste).
    bvh_cache = {
        p.name: (
            make_fcl_bvh(p.collision_mesh) if p.collision_mesh is not None else None
        )
        for p in probes
    }
    _INNER_WORKER_STATE = dict(
        probes=probes,
        holes=holes,
        weights=weights,
        probe_names=probe_names,
        threading_oval_tolerance=threading_oval_tolerance,
        clearance_overlap_allowance_mm=clearance_overlap_allowance_mm,
        subject_from_rig_rot=subject_from_rig_rot,
        slsqp_max_iter=slsqp_max_iter,
        slsqp_constrained=slsqp_constrained,
        two_stage_inner=two_stage_inner,
        feasibility_max_iter=feasibility_max_iter,
        final_feasibility_cleanup=final_feasibility_cleanup,
        polish_method=polish_method,
        feasibility_threshold=feasibility_threshold,
        verbose=verbose,
        bvh_cache=bvh_cache,
        sdf_by_name=sdf_by_name,
    )


def _inner_solve_worker(jc: JointCandidate) -> PlanCandidate:
    """Top-level worker function for ProcessPoolExecutor — must be
    pickleable, hence not a closure."""
    s = _INNER_WORKER_STATE
    ctx = _build_inner_context(
        s["probes"],
        s["holes"],
        jc.ha,
        jc.aa,
        s["weights"],
        threading_oval_tolerance=s["threading_oval_tolerance"],
        clearance_overlap_allowance_mm=s["clearance_overlap_allowance_mm"],
        subject_from_rig_rot=s["subject_from_rig_rot"],
        bvh_cache=s.get("bvh_cache"),
    )
    x0 = expand_reduced_solution_to_full_x(
        jc, ctx.layout, probe_names_in_order=s["probe_names"]
    )
    x0 = _push_restore_full_x(x0, ctx=ctx)

    # Optional SDF-based clearance constraint + analytic Jacobian.
    sdf_fun = None
    sdf_jac = None
    sdf_by_name = s.get("sdf_by_name")
    if sdf_by_name is not None:
        from aind_low_point.optimization.sdf_clearance import (
            build_probe_jax_data_for_context,
            build_sdf_clearance_callbacks,
        )

        jax_data = build_probe_jax_data_for_context(ctx, sdf_by_name)
        sdf_fun, sdf_jac = build_sdf_clearance_callbacks(
            n_arcs=ctx.layout.num_arcs,
            probe_data=jax_data,
            clearance_overlap_allowance_mm=s["clearance_overlap_allowance_mm"],
        )
    return _inner_solve_one(
        ctx,
        x0,
        ha=jc.ha,
        aa=jc.aa,
        n_arcs=jc.n_arcs,
        slsqp_max_iter=s["slsqp_max_iter"],
        slsqp_constrained=s["slsqp_constrained"],
        two_stage_inner=s["two_stage_inner"],
        feasibility_max_iter=s["feasibility_max_iter"],
        final_feasibility_cleanup=s["final_feasibility_cleanup"],
        polish_method=s["polish_method"],
        feasibility_threshold=s["feasibility_threshold"],
        verbose=s["verbose"],
        sdf_clearance_fun=sdf_fun,
        sdf_clearance_jac=sdf_jac,
    )


def optimize_joint(  # noqa: C901
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    *,
    # Forwarded to the existing full inner solve
    max_num_arcs: int = 4,
    min_num_arcs: int = 1,
    arc_count_penalty_deg2: float = 25.0,
    weights: ObjectiveWeights = ObjectiveWeights(),
    slsqp_max_iter: int = 100,
    slsqp_constrained: bool = True,
    two_stage_inner: bool = True,
    feasibility_max_iter: int = 80,
    min_arc_ap_sep_deg: float = 16.0,
    arc_sep_shortfall_weight: float = 10.0,
    threading_oval_tolerance: float = 0.0,
    clearance_overlap_allowance_mm: float = 0.0,
    final_feasibility_cleanup: bool = True,
    polish_method: str = "SLSQP",
    feasibility_threshold: float = 0.0,
    subject_from_rig_rot: NDArray | None = None,
    # Joint-reranker-specific kwargs
    k_holes_pool: int = 50,
    k_arcs_pool: int = 20,
    k_joint: int = 15,
    joint_weights: JointWeights = JointWeights(),
    reduced_slsqp_max_iter: int = 50,
    ap_sweep_half_deg: float = 25.0,
    ap_sweep_step_deg: float = 1.0,
    seed_to_hole: dict[str, int] | None = None,
    seed_to_arc_idx: dict[str, int] | None = None,
    n_workers: int | None = None,
    sdf_by_name: dict | None = None,
    use_atlas_stage1: bool = False,
    atlas_ap_step_deg: float = 2.0,
    atlas_max_excursion_deg: float = 15.0,
    atlas_max_target_miss_mm: float = 1.0,
    verbose: bool = False,
) -> OptimizationResult | None:
    """Three-level optimizer with a joint (H, A) reranking stage.

    Same outputs as :func:`optimize`; ``alternatives`` contains the
    full-inner-solve outcomes for the top-``k_joint`` (H, A) candidates
    surviving the joint reranker.

    Parameters
    ----------
    k_holes_pool
        Wide pool of LSAP hole assignments fed into the joint reranker.
        Default 50 (vs ``optimize``'s default 5) — the joint reranker
        is the gatekeeper now, so LSAP only needs to deliver enough
        variety for the reranker to find joint structures.
    k_arcs_pool
        Per-hole-assignment cap on arc partitions feeding into the joint
        reranker. Default 20.
    k_joint
        Number of joint candidates fed into the full inner solve.
        Default 15.
    joint_weights
        Reduced-SLSQP weights; see :class:`JointWeights`.
    reduced_slsqp_max_iter
        Max iterations for the reduced SLSQP. Default 50.
    ap_sweep_half_deg, ap_sweep_step_deg
        Resolution of the AP-interval sweep in :mod:`pose_features`.
    seed_to_hole, seed_to_arc_idx
        Optional ground-truth (H, A) for diagnostic ranking print.
    verbose
        Print per-stage timing, top-15 candidates, and seed-equivalent
        rank.
    """
    if not probes:
        return None

    t0 = time.time()

    # 0. Pre-compute per-(probe, hole) pose features.
    if verbose:
        print("[optimize_joint] Stage 0: precomputing pose features...")
    pose_features = precompute_pose_features(
        probes,
        holes,
        threading_oval_tolerance=threading_oval_tolerance,
        ap_sweep_half_deg=ap_sweep_half_deg,
        ap_sweep_step_deg=ap_sweep_step_deg,
    )
    t_pose = time.time()
    if verbose:
        print(
            f"[optimize_joint] precompute_pose_features: {t_pose - t0:.2f}s "
            f"({len(pose_features)} pairs)"
        )

    # 1. Wide LSAP pool.
    assignment_probes = [
        AssignmentProbe(
            name=p.name,
            target_LPS=np.asarray(p.target_LPS, dtype=np.float64),
            shank_tips_local=np.asarray(p.shank_tips_local, dtype=np.float64),
            kind=p.kind,
            density_sigma_mm=p.density_sigma_mm,
        )
        for p in probes
    ]
    cost_weights = CostWeights()
    cost_matrix = build_cost_matrix(assignment_probes, holes, weights=cost_weights)
    probe_names = [p.name for p in probes]
    holes_id_to_col = {h.id: j for j, h in enumerate(holes)}
    probe_name_to_row = {p.name: i for i, p in enumerate(probes)}

    # Build BVH per probe ONCE — its geometry doesn't depend on (H, A),
    # only its world transform does. Avoids ~7k redundant BVH builds
    # across the score_joint loop (Stage 2) for typical pool sizes.
    bvh_cache: dict[str, fcl.CollisionObject | None] = {
        p.name: (
            make_fcl_bvh(p.collision_mesh) if p.collision_mesh is not None else None
        )
        for p in probes
    }

    if use_atlas_stage1:
        if verbose:
            print(f"[optimize_joint] Stage 1: building target-aligned atlas "
                  f"(step={atlas_ap_step_deg}°, excursion=±{atlas_max_excursion_deg}°)...")
        from aind_low_point.optimization.atlas import atlas_stage1 as _atlas_stage1
        from aind_low_point.optimization.hole_assignment import (
            build_cost_matrix as _build_cost_matrix,
        )
        # Per-cell viol matrix to apply LSAP's hard-reject in addition
        # to atlas non-empty check.
        cost_mat_for_atlas = _build_cost_matrix(
            assignment_probes, holes, weights=cost_weights
        )
        # Per-cell viol — extract from multi_pose_evaluate. Cheap to redo.
        from aind_low_point.optimization.hole_assignment import multi_pose_evaluate
        K_p = len(assignment_probes)
        N_h = len(holes)
        viol_for_atlas = np.zeros((K_p, N_h))
        for ii, pp in enumerate(assignment_probes):
            for jj, hh in enumerate(holes):
                viol_for_atlas[ii, jj] = multi_pose_evaluate(pp, hh).min_violation_sq
        _atlas, hole_assignments = _atlas_stage1(
            probes, holes,
            ap_step_deg=atlas_ap_step_deg,
            ap_max_excursion_deg=atlas_max_excursion_deg,
            max_target_miss_mm=atlas_max_target_miss_mm,
            viol_mat=viol_for_atlas,
            cost_for_ordering=cost_mat_for_atlas,
            cap_hole_assignments=k_holes_pool,
            min_arc_sep_deg=min_arc_ap_sep_deg,
            max_arcs=max_num_arcs,
            verbose=verbose,
        )
        t_lsap = time.time()
        if verbose:
            print(f"[optimize_joint] atlas Stage 1: {t_lsap - t_pose:.2f}s "
                  f"({len(hole_assignments)} HAs)")
    else:
        if verbose:
            print(f"[optimize_joint] Stage 1: solving top-{k_holes_pool} LSAP...")
        hole_assignments = solve_top_k_assignments(
            assignment_probes, holes, k=k_holes_pool, weights=cost_weights
        )
        t_lsap = time.time()
        if verbose:
            print(
                f"[optimize_joint] solve_top_k_assignments: {t_lsap - t_pose:.2f}s "
                f"({len(hole_assignments)} HAs)"
            )

    if not hole_assignments:
        if verbose:
            print("[optimize_joint] No feasible hole assignment.")
        return None

    # 2. For each HA, enumerate top-k_arcs_pool arc partitions; for each
    #    (HA, AA) run the reduced SLSQP scoring.
    head_pitch_deg = _head_pitch_about_L_deg(subject_from_rig_rot)
    joint_candidates: list[JointCandidate] = []
    if verbose:
        print(
            f"[optimize_joint] Stage 2: arc enumeration + reduced SLSQP "
            f"(k_arcs_pool={k_arcs_pool})..."
        )
    eff_weights = replace(
        joint_weights,
        threading_oval_tolerance=threading_oval_tolerance,
        min_arc_ap_sep_deg=min_arc_ap_sep_deg,
    )

    for ha in hole_assignments:
        arc_assignments = solve_top_k_arc_assignments(
            ha.probe_to_hole,
            holes,
            max_num_arcs=max_num_arcs,
            min_num_arcs=min_num_arcs,
            k=k_arcs_pool,
            min_arc_ap_sep_deg=min_arc_ap_sep_deg,
            arc_sep_shortfall_weight=arc_sep_shortfall_weight,
            arc_count_penalty_deg2=arc_count_penalty_deg2,
        )
        if not arc_assignments:
            continue
        lsap_cost = 0.0
        for name, hole_id in ha.probe_to_hole.items():
            lsap_cost += float(
                cost_matrix[probe_name_to_row[name], holes_id_to_col[hole_id]]
            )
        for aa in arc_assignments:
            jc = score_joint(
                ha, aa, probes, holes, pose_features,
                weights=eff_weights,
                head_pitch_deg=head_pitch_deg,
                reduced_slsqp_max_iter=reduced_slsqp_max_iter,
                original_lsap_cost=lsap_cost,
                bvh_cache=bvh_cache,
                sdf_by_name=sdf_by_name,
            )
            joint_candidates.append(jc)

    t_joint = time.time()
    if verbose:
        print(
            f"[optimize_joint] reduced SLSQP scoring: {t_joint - t_lsap:.2f}s "
            f"({len(joint_candidates)} (H,A) candidates)"
        )

    if not joint_candidates:
        if verbose:
            print("[optimize_joint] No joint candidates produced.")
        return None

    # 3. Rank joint candidates and truncate.
    joint_candidates.sort(key=lambda c: c.metrics.lex_key(feasibility_threshold))

    if verbose:
        seed_rank = _lookup_seed_equivalent_rank(
            joint_candidates, seed_to_hole, seed_to_arc_idx
        )
        if seed_rank > 0:
            in_top = "in" if seed_rank <= k_joint else "OUTSIDE"
            print(
                f"[optimize_joint] seed-equivalent (H, A) rank in joint pool: "
                f"{seed_rank}/{len(joint_candidates)} ({in_top} k_joint={k_joint})"
            )
        elif seed_to_hole is not None and seed_to_arc_idx is not None:
            print(
                f"[optimize_joint] seed-equivalent (H, A) NOT FOUND in "
                f"joint pool ({len(joint_candidates)} candidates)"
            )
        print("[optimize_joint] top-15 JointCandidates:")
        for rank, jc in enumerate(joint_candidates[:15], start=1):
            print(_format_candidate_row(rank, jc))

    survivors = joint_candidates[:k_joint]

    # 4. Run the full inner solve on each survivor, warm-started from
    #    the reduced solution. Parallelised via ProcessPoolExecutor —
    #    each candidate's SLSQP is independent (its own OptimizerContext,
    #    FCL BVHs, scipy state). Initargs piggy-back on Linux's fork
    #    so the bulky mesh data isn't re-pickled per worker.
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 2)
    n_workers = max(1, min(n_workers, len(survivors)))
    # Process-pool startup is ~1–2 s per worker (fork + module load).
    # For tiny survivor lists the overhead dominates; run sequential.
    if len(survivors) <= 2:
        n_workers = 1
    if verbose:
        print(
            f"[optimize_joint] Stage 3: full inner solve on "
            f"{len(survivors)} survivors using {n_workers} workers..."
        )
    init_args = (
        probes,
        holes,
        weights,
        probe_names,
        threading_oval_tolerance,
        clearance_overlap_allowance_mm,
        subject_from_rig_rot,
        slsqp_max_iter,
        slsqp_constrained,
        two_stage_inner,
        feasibility_max_iter,
        final_feasibility_cleanup,
        polish_method,
        feasibility_threshold,
        verbose,
        sdf_by_name,
    )
    if n_workers == 1:
        # Sequential path — useful for debugging and avoids the worker
        # startup overhead for tiny survivor lists.
        _inner_solve_worker_init(*init_args)
        plan_candidates: list[PlanCandidate] = [
            _inner_solve_worker(jc) for jc in survivors
        ]
    else:
        # 'spawn' avoids inheriting CUDA contexts from JAX-GPU used in
        # Stage 2 — fork() leaves CUDA contexts in an unusable state.
        import multiprocessing as _mp
        _spawn_ctx = _mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_inner_solve_worker_init,
            initargs=init_args,
            mp_context=_spawn_ctx,
        ) as pool:
            plan_candidates = list(pool.map(_inner_solve_worker, survivors))

    t_inner = time.time()
    if verbose:
        print(
            f"[optimize_joint] full inner solve: {t_inner - t_joint:.2f}s "
            f"({len(plan_candidates)} plans)"
        )

    if not plan_candidates:
        return None

    plan_candidates.sort(key=lambda c: c.lex_key(feasibility_threshold))
    best = plan_candidates[0]
    if verbose:
        print(
            f"[optimize_joint] best plan: feasible={best.feasible} "
            f"max_viol={best.max_violation:.4g} coverage={best.coverage:.3f} "
            f"cost={best.cost:.3f}"
        )

    return OptimizationResult(
        probe_to_hole=best.probe_to_hole,
        probe_to_arc_idx=best.probe_to_arc_idx,
        arc_centroids_deg=best.arc_centroids_deg,
        n_arcs=best.n_arcs,
        x=best.x,
        cost=best.cost,
        breakdown=best.breakdown,
        alternatives=tuple(plan_candidates),
    )


__all__ = [
    "JointWeights",
    "JointRerankMetrics",
    "JointCandidate",
    "expand_reduced_solution_to_full_x",
    "optimize_joint",
    "score_joint",
]
