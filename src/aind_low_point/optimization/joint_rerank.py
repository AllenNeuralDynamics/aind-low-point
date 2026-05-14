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

import time
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from aind_low_point.optimization.arc_assignment import (
    ArcAssignment,
    solve_top_k_arc_assignments,
)
from aind_low_point.optimization.geometry import cap_basis, shaft_section_oval_value
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
    shank_capsules_from_pose,
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
    lambda_clearance: float = 0.0
    lambda_coverage: float = 0.0
    comfortable_ap_deg: float = 50.0
    comfortable_ml_deg: float = 50.0
    min_arc_ap_sep_deg: float = 16.0
    min_intra_arc_ml_sep_deg: float = 16.0
    threading_oval_tolerance: float = 0.0


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
    approximate_coverage: float
    bounds_softpenalty: float
    original_lsap_cost: float

    def lex_key(self, feasibility_threshold: float = 0.0) -> tuple[float, float, float]:
        """Lex-rank tuple ``(eff_viol, sum_viol_sq, -coverage)``.

        Mirrors :meth:`PlanCandidate.lex_key`: an ``ε``-collapse on
        ``max_violation`` so any candidate at or below the slop budget
        ranks among "feasible enough"; coverage breaks ties.
        """
        eff = max(0.0, self.max_violation - feasibility_threshold)
        return (eff, self.sum_violation_sq, -self.approximate_coverage)


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

    ``y = [ap_arc_0, ..., ap_arc_{n_arcs-1}, (ml_0, spin_0), ...,
    (ml_{K-1}, spin_{K-1})]``. Returns a small dict of slices for
    each named block.
    """
    return {
        "arc_aps": slice(0, n_arcs),
        "probe_vars": slice(n_arcs, n_arcs + 2 * n_probes),
    }


def _ml_spin_from_y(y: NDArray, n_arcs: int, probe_idx: int) -> tuple[float, float]:
    """Return ``(ml, spin)`` for probe ``probe_idx`` from the reduced vector."""
    offset = n_arcs + 2 * probe_idx
    return float(y[offset]), float(y[offset + 1])


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


def _build_probe_static(
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    ha: HoleAssignment,
    aa: ArcAssignment,
) -> list[_ProbeStatic]:
    """Build the per-probe static cache for the reduced objective."""
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
            pivot = np.array(
                [0.0, 0.0, float(geom.active_center_mm)], dtype=np.float64
            )
        hole_id = ha.probe_to_hole[p.name]
        arc_idx = aa.probe_to_arc_idx[p.name]
        hole = holes_by_id[hole_id]
        # Precompute per-section basis vectors / centres / oval params.
        sections = hole.sections
        s_axes = np.array(
            [np.asarray(s.axis, dtype=np.float64) for s in sections]
        )
        s_e1 = np.empty_like(s_axes)
        s_e2 = np.empty_like(s_axes)
        for k, s in enumerate(sections):
            e1, e2 = cap_basis(s.axis)
            s_e1[k] = e1
            s_e2[k] = e2
        s_centers = np.array(
            [np.asarray(s.center, dtype=np.float64) for s in sections]
        )
        s_thetas = np.array([float(s.theta) for s in sections])
        s_a = np.array([float(s.a) for s in sections])
        s_b = np.array([float(s.b) for s in sections])
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
        [float(y[n_arcs + 2 * i]) for i in range(n_probes)], dtype=np.float64
    )
    j_bounds_ml = _softplus_squared(np.abs(ml_vals) - weights.comfortable_ml_deg)
    j_bounds = j_bounds_ap + j_bounds_ml

    # Clearance / coverage terms are placeholders (λ defaults to 0); skip
    # the work entirely when unused.
    total = (
        weights.lambda_thread * j_thread
        + weights.lambda_arc_ap * j_arc_ap
        + weights.lambda_ml * j_ml
        + weights.lambda_bounds * j_bounds
    )
    return float(total)


def _evaluate_reduced_metrics(
    y: NDArray,
    statics: list[_ProbeStatic],
    n_arcs: int,
    weights: JointWeights,
    *,
    original_lsap_cost: float,
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
        # Approximate coverage proxy: gaussian distance from target to the
        # shaft tip after pose. Higher = closer.
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
        if st.shank_tips_local.shape[0] > 0:
            # Centre of mass of shanks lands at target by construction
            # (pivot_local subtraction); use the distance the centre
            # actually achieves as the coverage proxy.
            shank_centroid_world = (
                R @ np.asarray(st.shank_tips_local, dtype=np.float64).mean(axis=0)
                + pose_tip
            )
            d = float(np.linalg.norm(shank_centroid_world - st.target_LPS))
            approx_cov += float(np.exp(-(d**2)))

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
        [float(y[n_arcs + 2 * i]) for i in range(n_probes)], dtype=np.float64
    )
    bounds_pen += _softplus_squared(np.abs(ml_vals) - weights.comfortable_ml_deg)

    max_viol = max(max_thread, max_arc_ap, max_ml)
    sum_viol_sq = sum_thread_sq + sum_arc_ap_sq + sum_ml_sq
    return JointRerankMetrics(
        max_violation=float(max_viol),
        sum_violation_sq=float(sum_viol_sq),
        max_violation_threading=float(max_thread),
        max_violation_arc_ap_sep=float(max_arc_ap),
        max_violation_intra_arc_ml_sep=float(max_ml),
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
    n_vars = n_arcs + 2 * n_probes

    # Per-arc on-arc probe lists for the AP-interval intersection.
    on_arc: dict[int, list[int]] = {a: [] for a in range(n_arcs)}
    for i, st in enumerate(statics):
        on_arc[st.arc_idx].append(i)

    # Helper: slot-major-axis spin (deg) per probe.
    spin_warm: list[float] = []
    for st in statics:
        spin_rad = float(np.pi / 2 - st.assigned_hole.slot_theta_rad)
        spin_warm.append(float(np.rad2deg(spin_rad)))

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
    # warm-start. The partitioner's centroid uses
    # ``required_ap_deg(hole.axis)``, which is sign-flipped relative
    # to the strict rig formula in :mod:`pose_features` but matches
    # the existing optimizer's convention.
    start1 = np.zeros(n_vars, dtype=np.float64)
    for a in range(n_arcs):
        start1[a] = float(aa.arc_centroids_deg[a])
    for i, st in enumerate(statics):
        start1[n_arcs + 2 * i] = 0.0  # ml
        start1[n_arcs + 2 * i + 1] = spin_warm[i]

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
                idx = int(
                    np.argmin([abs(rap - target_ap) for rap in required_aps])
                )
                start2[a] = float(required_aps[idx])
            else:
                start2[a] = float(aa.arc_centroids_deg[a])
        else:
            start2[a] = float(aa.arc_centroids_deg[a])
    for i, st in enumerate(statics):
        ap_i = float(start2[st.arc_idx])
        start2[n_arcs + 2 * i] = _ml_at_ap(i, ap_i)
        start2[n_arcs + 2 * i + 1] = spin_warm[i]

    # Start 3: same APs as start 2, alternating spin offsets per on-arc
    # neighbour (odd-indexed gets spin + 180°).
    start3 = start2.copy()
    for a in range(n_arcs):
        members = on_arc.get(a, [])
        for k, i in enumerate(members):
            if k % 2 == 1:
                start3[n_arcs + 2 * i + 1] = spin_warm[i] + 180.0

    return [start1, start2, start3]


# ---------------------------------------------------------------------------
# Bounds + SLSQP wrapper
# ---------------------------------------------------------------------------


def _reduced_bounds(
    n_arcs: int,
    n_probes: int,
    head_pitch_deg: float,
) -> list[tuple[float, float]]:
    """Box bounds matching :func:`_default_bounds` for the reduced vector.

    AP bound shifted by head pitch; ML clamped to ±60°; spin to ±180°.
    """
    bounds: list[tuple[float, float]] = []
    for _ in range(n_arcs):
        bounds.append((-60.0 + head_pitch_deg, +60.0 + head_pitch_deg))
    for _ in range(n_probes):
        bounds.append((-60.0, +60.0))  # ml
        bounds.append((-180.0, +180.0))  # spin
    return bounds


def _wrap_spin_to_bounds(y: NDArray, n_arcs: int, n_probes: int) -> NDArray:
    """Wrap spin angles into ``[-180, 180]`` before passing to SLSQP.

    SLSQP enforces box bounds but starts may have spin = +(slot+180°)
    which can land outside ``[-180, 180]``. Wrap rather than clip so the
    geometric rotation is preserved.
    """
    out = np.asarray(y, dtype=np.float64).copy()
    for i in range(n_probes):
        idx = n_arcs + 2 * i + 1
        out[idx] = ((out[idx] + 180.0) % 360.0) - 180.0
    return out


def _slsqp_reduced(
    y0: NDArray,
    statics: list[_ProbeStatic],
    n_arcs: int,
    weights: JointWeights,
    *,
    bounds: list[tuple[float, float]],
    max_iter: int,
) -> NDArray:
    """Run a single SLSQP polish from ``y0`` and return the optimum.

    Returns ``y0`` when SLSQP fails to make progress (e.g. zero
    gradient at the start). Failure here doesn't raise — the caller
    decides whether to retry from a different start.
    """
    y0 = _wrap_spin_to_bounds(y0, n_arcs, len(statics))

    def fn(v):
        return _reduced_objective(
            np.asarray(v, dtype=np.float64), statics, n_arcs, weights
        )

    try:
        result = minimize(
            fn,
            y0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-6, "disp": False},
        )
    except Exception:
        return y0
    return np.asarray(result.x, dtype=np.float64)


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
) -> JointCandidate:
    """Run the reduced-SLSQP scoring for one (hole, arc) candidate.

    Tries three warm starts (see module docstring) and returns the
    candidate with the lex-best metrics.
    """
    n_probes = len(probes)
    n_arcs = max(aa.probe_to_arc_idx.values()) + 1 if aa.probe_to_arc_idx else 1
    statics = _build_probe_static(probes, holes, ha, aa)
    starts = _build_starts(statics, aa, pose_features, n_arcs)
    bounds = _reduced_bounds(n_arcs, n_probes, head_pitch_deg)

    best_y: NDArray | None = None
    best_metrics: JointRerankMetrics | None = None
    for y0 in starts:
        y_opt = _slsqp_reduced(
            y0, statics, n_arcs, weights, bounds=bounds, max_iter=reduced_slsqp_max_iter
        )
        m = _evaluate_reduced_metrics(
            y_opt,
            statics,
            n_arcs,
            weights,
            original_lsap_cost=original_lsap_cost,
        )
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
        ml = float(jc.reduced_y[n_arcs + 2 * i])
        spin = float(jc.reduced_y[n_arcs + 2 * i + 1])
        # Wrap spin into [-180, 180] for the full SLSQP bounds.
        spin = ((spin + 180.0) % 360.0) - 180.0
        off = layout.num_arcs + 5 * i
        x[off + 0] = ml
        x[off + 1] = spin
        x[off + 2] = 0.0
        x[off + 3] = 0.0
        x[off + 4] = 0.0
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


def optimize_joint(  # noqa: C901
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    *,
    # Forwarded to the existing full inner solve
    max_num_arcs: int = 4,
    min_num_arcs: int = 1,
    arc_count_penalty_deg2: float = 25.0,
    weights: ObjectiveWeights = ObjectiveWeights(),
    use_cma: bool = True,
    cma_population: int = 30,
    cma_generations: int = 100,
    cma_sigma: float = 5.0,
    cma_stage_multipliers: tuple[float, ...] = (0.1, 1.0, 10.0),
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
        # LSAP cost for this HA (Σ matrix cells along the assignment).
        lsap_cost = 0.0
        for name, hole_id in ha.probe_to_hole.items():
            lsap_cost += float(
                cost_matrix[probe_name_to_row[name], holes_id_to_col[hole_id]]
            )

        for aa in arc_assignments:
            jc = score_joint(
                ha,
                aa,
                probes,
                holes,
                pose_features,
                weights=replace(
                    joint_weights,
                    threading_oval_tolerance=threading_oval_tolerance,
                    min_arc_ap_sep_deg=min_arc_ap_sep_deg,
                ),
                head_pitch_deg=head_pitch_deg,
                reduced_slsqp_max_iter=reduced_slsqp_max_iter,
                original_lsap_cost=lsap_cost,
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
        print(f"[optimize_joint] top-15 JointCandidates:")
        for rank, jc in enumerate(joint_candidates[:15], start=1):
            print(_format_candidate_row(rank, jc))

    survivors = joint_candidates[:k_joint]

    # 4. Run the full inner solve on each survivor, warm-started from
    #    the reduced solution.
    if verbose:
        print(
            f"[optimize_joint] Stage 3: full inner solve on "
            f"{len(survivors)} survivors..."
        )
    plan_candidates: list[PlanCandidate] = []
    for jc in survivors:
        ctx = _build_inner_context(
            probes,
            holes,
            jc.ha,
            jc.aa,
            weights,
            threading_oval_tolerance=threading_oval_tolerance,
            clearance_overlap_allowance_mm=clearance_overlap_allowance_mm,
            subject_from_rig_rot=subject_from_rig_rot,
        )
        # The layout's probe order is taken from the probes list (see
        # _build_inner_context), so the reduced y aligns 1:1 with x.
        x0 = expand_reduced_solution_to_full_x(
            jc,
            ctx.layout,
            probe_names_in_order=probe_names,
        )
        cand = _inner_solve_one(
            ctx,
            x0,
            ha=jc.ha,
            aa=jc.aa,
            n_arcs=jc.n_arcs,
            use_cma=use_cma,
            cma_population=cma_population,
            cma_generations=cma_generations,
            cma_sigma=cma_sigma,
            cma_stage_multipliers=cma_stage_multipliers,
            slsqp_max_iter=slsqp_max_iter,
            slsqp_constrained=slsqp_constrained,
            two_stage_inner=two_stage_inner,
            feasibility_max_iter=feasibility_max_iter,
            final_feasibility_cleanup=final_feasibility_cleanup,
            polish_method=polish_method,
            feasibility_threshold=feasibility_threshold,
            verbose=verbose,
        )
        plan_candidates.append(cand)

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
