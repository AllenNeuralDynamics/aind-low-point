"""Optimizer objective and supporting structures.

Assembles the inner-loop scalar objective ``J(x)`` from:
- coverage (per-probe line integral of target density across active
  recording ranges, summed over shanks per :mod:`recording`)
- threading (per (probe, shank, section) oval inequality)
- probe-probe headstage clearance (capsule-capsule signed distance)
- kinematic separation (chained pairwise convexified
  ``ap_arc`` and ``ml_local`` constraints)

Variable vector layout
----------------------
``x`` is a flat 1-D array. Slicing is handled by
:class:`VariableLayout`:

::

    x = [ap_arc_0, ap_arc_1, ..., ap_arc_{A-1},
         ml_0, spin_0, off_R_0, off_A_0, depth_0,
         ml_1, spin_1, off_R_1, off_A_1, depth_1,
         ...
         ml_{K-1}, spin_{K-1}, off_R_{K-1}, off_A_{K-1}, depth_{K-1}]

so ``len(x) = num_arcs + 5 * num_probes``. The order of arcs and
probes is fixed at layout-build time and reused for every evaluation.

Numpy-only for v1. Operations are JAX-traceable as-is once we wire
``jax.numpy`` (the only places we'd need to swap are the per-probe
loops over shanks and sections; everything underneath is already
elementwise / linear-algebra).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.density import (
    DensityFn,
    coverage,
)
from aind_low_point.optimization.geometry import (
    Capsule,
    capsule_capsule_dist,
    shaft_section_oval_value,
)
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.kinematics import (
    pose_from_optimizer_vars,
    shank_capsules_from_pose,
)
from aind_low_point.optimization.recording import (
    RecordingGeometry,
    recording_center_local_for_kind,
)


# ---------------------------------------------------------------------------
# Static context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariableLayout:
    """Defines how the flat variable vector ``x`` is sliced."""

    arc_ids: tuple[str, ...]
    probe_names: tuple[str, ...]

    @property
    def num_arcs(self) -> int:
        return len(self.arc_ids)

    @property
    def num_probes(self) -> int:
        return len(self.probe_names)

    @property
    def n_vars(self) -> int:
        return self.num_arcs + 5 * self.num_probes

    def arc_ap(self, x: NDArray, arc_id: str) -> float:
        idx = self.arc_ids.index(arc_id)
        return float(x[idx])

    def arc_aps(self, x: NDArray) -> NDArray:
        return np.asarray(x[: self.num_arcs], dtype=np.float64)

    def probe_vars(self, x: NDArray, probe_idx: int) -> NDArray:
        """Returns ``(ml, spin, off_R, off_A, depth)`` for probe ``probe_idx``."""
        offset = self.num_arcs + 5 * probe_idx
        return np.asarray(x[offset : offset + 5], dtype=np.float64)


@dataclass(frozen=True)
class ProbeContext:
    """Static per-probe info baked at optimizer-build time."""

    name: str
    target_LPS: NDArray[np.floating]
    kind: str
    arc_id: str
    shank_tips_local: NDArray[np.floating]
    assigned_hole: Hole
    density_fn: DensityFn
    recording_geom: RecordingGeometry


@dataclass(frozen=True)
class ObjectiveWeights:
    """Penalty / margin weights used to combine the objective.

    ``λ_feas`` ramps up in CMA-ES via a homotopy schedule; SLSQP either
    uses its native inequality constraints (preferred) or these
    penalties at a fixed final-stage value.

    ``lambda_margin`` defaults to 0 in v1: the softmin margin reward
    can diverge to ``-∞`` when clearances are very negative, which
    perversely rewards huge headstage overlaps. Re-enable by setting
    it positive *only* once the inner loop also clips clearances or
    SLSQP gets proper inequality constraints. Coverage + the three
    feasibility penalties suffice for the v1 driver.
    """

    lambda_threading: float = 1.0e3
    lambda_clearance: float = 1.0e3
    lambda_kinematic: float = 1.0e4   # large; rig limits are inviolable
    lambda_margin: float = 0.0        # disabled in v1 — see class docstring
    margin_softmin_beta: float = 0.5  # mm
    safety_clearance_mm: float = 0.0  # min headstage-to-headstage gap


@dataclass(frozen=True)
class OptimizerContext:
    """Bundle of static info the objective closure needs."""

    layout: VariableLayout
    probes: tuple[ProbeContext, ...]
    arc_for_probe: dict[str, str] = field(default_factory=dict)
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    shaft_length_mm: float = 10.0
    shank_radius_mm: float = 0.05
    headstage_base_along_shaft_mm: float = 10.0
    headstage_length_mm: float = 5.0
    headstage_radius_mm: float = 2.0
    min_arc_ap_sep_deg: float = 16.0
    min_within_arc_ml_sep_deg: float = 16.0
    coverage_n_samples: int = 41

    def probe_index(self, name: str) -> int:
        return self.layout.probe_names.index(name)


# ---------------------------------------------------------------------------
# Per-probe evaluation
# ---------------------------------------------------------------------------


def headstage_capsule(
    R: NDArray, pose_tip: NDArray, ctx: OptimizerContext
) -> Capsule:
    """Coarse capsule above the probe's local origin.

    Default (configurable on ``ctx``): 10 mm up the shaft from the
    probe's local origin, 5 mm long, 2 mm radius. This is a *placeholder*
    until per-probe-kind headstage geometry is registered (``recording.py``
    is the right home for that table once we have the dimensions).
    """
    shaft_dir = R @ np.array([0.0, 0.0, 1.0])
    base = np.asarray(pose_tip, dtype=np.float64) + (
        ctx.headstage_base_along_shaft_mm * shaft_dir
    )
    top = base + ctx.headstage_length_mm * shaft_dir
    return Capsule(p0=base, p1=top, radius=ctx.headstage_radius_mm)


@dataclass(frozen=True)
class ProbeEvaluation:
    """Per-probe outputs of an objective evaluation."""

    R: NDArray[np.floating]
    pose_tip: NDArray[np.floating]
    shanks: list[Capsule]
    headstage: Capsule
    coverage: float
    threading_gs: NDArray[np.floating]  # one g per (shank × section)


def evaluate_probe(
    probe: ProbeContext,
    ap_deg: float,
    ml_deg: float,
    spin_deg: float,
    off_R_mm: float,
    off_A_mm: float,
    past_target_mm: float,
    *,
    ctx: OptimizerContext,
) -> ProbeEvaluation:
    """Compute pose, capsules, coverage, and threading values for one probe.

    Pivot is computed from the probe's actual ``shank_tips_local`` —
    ``(centroid_x, centroid_y, active_center_mm)`` — rather than from
    a hardcoded direction in :mod:`recording`. This matches whatever
    canonicalization the upstream mesh used.
    """
    tips = np.asarray(probe.shank_tips_local, dtype=np.float64)
    if tips.shape[0] > 0:
        pivot_local = np.array(
            [
                float(tips[:, 0].mean()),
                float(tips[:, 1].mean()),
                float(probe.recording_geom.active_center_mm),
            ],
            dtype=np.float64,
        )
    else:
        pivot_local = recording_center_local_for_kind(probe.kind)
    R, pose_tip = pose_from_optimizer_vars(
        target_LPS=probe.target_LPS,
        ap_deg=ap_deg, ml_deg=ml_deg, spin_deg=spin_deg,
        offset_R_mm=off_R_mm, offset_A_mm=off_A_mm,
        past_target_mm=past_target_mm,
        recording_center_local=pivot_local,
    )
    shanks = shank_capsules_from_pose(
        R, pose_tip, probe.shank_tips_local,
        shaft_length_mm=ctx.shaft_length_mm,
        shank_radius_mm=ctx.shank_radius_mm,
    )
    cov = coverage(
        probe.density_fn, shanks, probe.recording_geom,
        n_samples=ctx.coverage_n_samples,
    )
    threading_gs = np.array(
        [
            shaft_section_oval_value(sh, sec)
            for sh in shanks
            for sec in probe.assigned_hole.sections
        ],
        dtype=np.float64,
    )
    return ProbeEvaluation(
        R=R, pose_tip=pose_tip, shanks=shanks,
        headstage=headstage_capsule(R, pose_tip, ctx),
        coverage=float(cov),
        threading_gs=threading_gs,
    )


# ---------------------------------------------------------------------------
# Pairwise constraints
# ---------------------------------------------------------------------------


def pairwise_headstage_clearances(
    evals: list[ProbeEvaluation],
) -> NDArray[np.floating]:
    """Signed clearances between every pair of headstage capsules.

    Positive = clearance, 0 = touching, negative = penetration. Empty
    array if there's only one probe.
    """
    n = len(evals)
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(n * (n - 1) // 2, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            out[k] = capsule_capsule_dist(evals[i].headstage, evals[j].headstage)
            k += 1
    return out


def kinematic_separations(
    arc_aps_deg: NDArray,
    probe_mls_deg: NDArray,
    probe_arc_indices: NDArray,
) -> tuple[NDArray, NDArray]:
    """Return ``(ap_pair_seps, ml_pair_seps_within_arc)`` in degrees.

    ``ap_pair_seps[k]`` is ``|ap_i − ap_j|`` for the k-th pair of arcs
    (lex order). ``ml_pair_seps_within_arc[k]`` is ``|ml_i − ml_j|`` for
    the k-th pair of probes that share an arc.

    These are the *non-convex absolute-difference* form. The middle
    layer's pre-ordering produces signed chained constraints
    (``ap_{σ(i+1)} − ap_{σ(i)} ≥ 16``) which are convex; this function
    is the symmetric form used for soft penalty calculation in CMA-ES.
    """
    n_arcs = len(arc_aps_deg)
    if n_arcs < 2:
        ap_seps = np.zeros(0, dtype=np.float64)
    else:
        idxs = np.array([(i, j) for i in range(n_arcs) for j in range(i + 1, n_arcs)])
        ap_seps = np.abs(arc_aps_deg[idxs[:, 0]] - arc_aps_deg[idxs[:, 1]])

    n_probes = len(probe_mls_deg)
    ml_pairs: list[float] = []
    for i in range(n_probes):
        for j in range(i + 1, n_probes):
            if probe_arc_indices[i] == probe_arc_indices[j]:
                ml_pairs.append(
                    abs(float(probe_mls_deg[i]) - float(probe_mls_deg[j]))
                )
    return ap_seps, np.asarray(ml_pairs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Penalty assembly + objective
# ---------------------------------------------------------------------------


def _quadratic_violation_penalty(
    values: NDArray, *, threshold: float = 0.0
) -> float:
    """``Σ max(0, value − threshold)²``. Smooth on the violating side,
    zero on the satisfied side."""
    if values.size == 0:
        return 0.0
    excess = np.maximum(0.0, values - threshold)
    return float(np.sum(excess * excess))


def _softmin(values: NDArray, beta: float) -> float:
    """Smooth approximation of ``min(values)``: ``-β · log(Σ exp(-v/β))``.

    Returns ``min(values)`` in the limit ``β → 0``. Used for the margin
    reward term — encourages the optimizer to maximise the *minimum*
    clearance across pairs, not just the average.
    """
    if values.size == 0:
        return 0.0
    return float(-beta * np.log(np.sum(np.exp(-values / beta))))


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """Component-wise breakdown of the scalar objective for diagnostics."""

    total: float
    coverage_total: float
    threading_penalty: float
    clearance_penalty: float
    kinematic_penalty: float
    margin_reward: float
    per_probe_evals: list[ProbeEvaluation]


def evaluate_objective(
    x: NDArray, ctx: OptimizerContext
) -> ObjectiveBreakdown:
    """Evaluate the objective at ``x``, returning the scalar plus diagnostics.

    Use :func:`scalar_objective` (a thin wrapper that returns just the
    scalar) for the optimizer's call.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (ctx.layout.n_vars,):
        raise ValueError(
            f"x has shape {x.shape}; expected ({ctx.layout.n_vars},)"
        )

    # Per-probe pose, shanks, coverage, threading
    arc_aps = ctx.layout.arc_aps(x)
    arc_id_to_idx = {a: i for i, a in enumerate(ctx.layout.arc_ids)}
    evals: list[ProbeEvaluation] = []
    probe_mls = np.empty(ctx.layout.num_probes, dtype=np.float64)
    probe_arc_idxs = np.empty(ctx.layout.num_probes, dtype=np.int64)
    for i, probe in enumerate(ctx.probes):
        ml, spin, off_R, off_A, depth = ctx.layout.probe_vars(x, i)
        ap = float(arc_aps[arc_id_to_idx[probe.arc_id]])
        evals.append(
            evaluate_probe(
                probe, ap, ml, spin, off_R, off_A, depth, ctx=ctx
            )
        )
        probe_mls[i] = ml
        probe_arc_idxs[i] = arc_id_to_idx[probe.arc_id]

    # Coverage (sum across probes)
    coverage_total = float(sum(ev.coverage for ev in evals))

    # Threading penalty (one quadratic term per (probe, shank, section))
    all_threading = np.concatenate(
        [ev.threading_gs for ev in evals], axis=0
    )
    threading_penalty = ctx.weights.lambda_threading * _quadratic_violation_penalty(
        all_threading, threshold=0.0
    )

    # Headstage-headstage clearance (negative clearance = penetration)
    pair_clearances = pairwise_headstage_clearances(evals)
    clearance_penalty = ctx.weights.lambda_clearance * _quadratic_violation_penalty(
        -pair_clearances,
        threshold=-ctx.weights.safety_clearance_mm,
    )

    # Kinematic separations (penalty if below threshold)
    ap_seps, ml_seps = kinematic_separations(
        arc_aps, probe_mls, probe_arc_idxs
    )
    ap_kin_pen = _quadratic_violation_penalty(
        ctx.min_arc_ap_sep_deg - ap_seps
    )
    ml_kin_pen = _quadratic_violation_penalty(
        ctx.min_within_arc_ml_sep_deg - ml_seps
    )
    kinematic_penalty = ctx.weights.lambda_kinematic * (ap_kin_pen + ml_kin_pen)

    # Margin reward (softmin over clearances; encourages bigger gaps)
    if pair_clearances.size > 0:
        margin_reward = ctx.weights.lambda_margin * _softmin(
            pair_clearances, beta=ctx.weights.margin_softmin_beta
        )
    else:
        margin_reward = 0.0

    total = (
        -coverage_total
        + threading_penalty
        + clearance_penalty
        + kinematic_penalty
        - margin_reward
    )
    return ObjectiveBreakdown(
        total=float(total),
        coverage_total=coverage_total,
        threading_penalty=threading_penalty,
        clearance_penalty=clearance_penalty,
        kinematic_penalty=kinematic_penalty,
        margin_reward=margin_reward,
        per_probe_evals=evals,
    )


def scalar_objective(x: NDArray, ctx: OptimizerContext) -> float:
    """Just the scalar — for ``cma`` / ``scipy.optimize`` consumption."""
    return evaluate_objective(x, ctx).total


@dataclass(frozen=True)
class ConstraintVectors:
    """Raw constraint slack vectors at ``x``.

    All entries are ``slack_i = limit - violation_i`` so that ``slack_i
    >= 0`` indicates feasibility. Pass straight to scipy's SLSQP
    ``constraints=[{"type": "ineq", "fun": ...}]`` form.
    """

    threading: NDArray[np.floating]      # one entry per (probe, shank, section)
    clearance: NDArray[np.floating]      # one entry per probe pair
    arc_ap_separation: NDArray[np.floating]   # one entry per arc pair
    intra_arc_ml_separation: NDArray[np.floating]  # one entry per intra-arc pair
    coverage_total: float                # objective term (to maximise)


def evaluate_constraints(x: NDArray, ctx: OptimizerContext) -> ConstraintVectors:
    """Compute the raw, ReLU-free constraint slack vectors at ``x`` and
    the (negated-for-minimisation-friendliness) coverage total.

    Used by :func:`_slsqp_polish` when running with native inequality
    constraints instead of soft penalties — scipy expects ``g(x) >= 0``
    for feasibility, which is what each slack array provides.
    """
    x = np.asarray(x, dtype=np.float64)
    arc_aps = ctx.layout.arc_aps(x)
    arc_id_to_idx = {a: i for i, a in enumerate(ctx.layout.arc_ids)}
    evals: list[ProbeEvaluation] = []
    probe_mls = np.empty(ctx.layout.num_probes, dtype=np.float64)
    probe_arc_idxs = np.empty(ctx.layout.num_probes, dtype=np.int64)
    for i, probe in enumerate(ctx.probes):
        ml, spin, off_R, off_A, depth = ctx.layout.probe_vars(x, i)
        ap = float(arc_aps[arc_id_to_idx[probe.arc_id]])
        evals.append(
            evaluate_probe(probe, ap, ml, spin, off_R, off_A, depth, ctx=ctx)
        )
        probe_mls[i] = ml
        probe_arc_idxs[i] = arc_id_to_idx[probe.arc_id]

    threading_gs = (
        np.concatenate([ev.threading_gs for ev in evals], axis=0)
        if evals else np.zeros(0, dtype=np.float64)
    )
    pair_clearances = pairwise_headstage_clearances(evals)
    ap_seps, ml_seps = kinematic_separations(arc_aps, probe_mls, probe_arc_idxs)

    return ConstraintVectors(
        # threading_gs > 0 means infeasible; slack = -g.
        threading=-threading_gs,
        # pair_clearances < safety means infeasible; slack = clearance - safety.
        clearance=pair_clearances - ctx.weights.safety_clearance_mm,
        # ap_seps < min means infeasible; slack = ap_sep - min.
        arc_ap_separation=ap_seps - ctx.min_arc_ap_sep_deg,
        intra_arc_ml_separation=ml_seps - ctx.min_within_arc_ml_sep_deg,
        coverage_total=float(sum(ev.coverage for ev in evals)),
    )


def coverage_objective(x: NDArray, ctx: OptimizerContext) -> float:
    """``-coverage_total`` for use as the SLSQP objective when the
    feasibility terms are expressed as inequality constraints."""
    return -evaluate_constraints(x, ctx).coverage_total


def feasibility_violation_squared(x: NDArray, ctx: OptimizerContext) -> float:
    """``Σ max(0, -slack)²`` across all constraint groups at ``x``.

    Treats threading, clearance, arc-AP separation, and intra-arc ML
    separation slacks symmetrically — zero at feasibility, smooth on
    the violating side. Used as the Stage-A scalar in the two-stage
    inner solve: minimising it drives the optimizer to (or close to)
    the feasibility tube before Stage B optimises coverage subject to
    hard constraints.
    """
    cv = evaluate_constraints(x, ctx)
    total = 0.0
    for arr in (
        cv.threading,
        cv.clearance,
        cv.arc_ap_separation,
        cv.intra_arc_ml_separation,
    ):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size > 0:
            excess = np.maximum(0.0, -arr)
            total += float(np.sum(excess * excess))
    return total


def make_objective(ctx: OptimizerContext) -> Callable[[NDArray], float]:
    """Bind ``ctx`` and return ``J(x) -> float`` for the optimizer."""
    def J(x: NDArray) -> float:
        return scalar_objective(x, ctx)
    return J
