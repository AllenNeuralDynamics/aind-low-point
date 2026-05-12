"""Three-level optimizer driver.

Glues the outer layer (probe→hole, ``hole_assignment``), the middle
layer (probe→arc, ``arc_assignment``), and the inner continuous
optimization (CMA-ES warm-start → SLSQP polish on the
:func:`evaluate_objective` scalar) into a single ``optimize()`` entry
point.

::

    enumerate top-K_h probe→hole assignments
    for each:
      enumerate top-K_a probe→arc assignments
      for each:
        warm-start variable vector x0
        CMA-ES global → SLSQP polish (or SLSQP-only if cma missing)
        record (cost, x, ObjectiveBreakdown)
    return the (hole, arc, x) combination with the best inner cost

CMA-ES via the ``cma`` PyPI package is optional at v1: when not
installed the driver runs SLSQP from the warm start only and prints a
note. Install ``cma`` to enable the global stage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from aind_low_point.optimization.arc_assignment import (
    ArcAssignment,
    solve_top_k_arc_assignments,
)
from aind_low_point.optimization.density import gaussian_density
from aind_low_point.optimization.hole_assignment import (
    AssignmentProbe,
    HoleAssignment,
    solve_top_k_assignments,
)
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.objective import (
    ObjectiveBreakdown,
    ObjectiveWeights,
    OptimizerContext,
    ProbeContext,
    VariableLayout,
    coverage_objective,
    evaluate_constraints,
    evaluate_objective,
    feasibility_violation_squared,
    scalar_objective,
)
from aind_low_point.optimization.recording import (
    RECORDING_GEOMETRY,
    RecordingGeometry,
    get_recording_geometry,
)


# ---------------------------------------------------------------------------
# Per-probe static info input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeStaticInfo:
    """Per-probe info the optimizer needs from the caller.

    Combines what the outer + inner layers each need: target, kind,
    detected shank tips. ``density_sigma_mm`` controls the coverage
    objective's Gaussian width.
    """

    name: str
    target_LPS: NDArray[np.floating]
    kind: str
    shank_tips_local: NDArray[np.floating]
    density_sigma_mm: float = 0.5


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanCandidate:
    """One (hole, arc, x*) attempt + the metrics used for lex-ranking.

    Feasibility metrics (``max_violation``, ``sum_violation_sq``) come
    from :func:`evaluate_constraints`'s slack vectors: ``slack ≥ 0`` is
    feasible, so ``max_violation = max(0, max_j -slack_j)`` and
    ``sum_violation_sq = Σ_j ReLU(-slack_j)²``. Units across groups are
    mixed (threading is dimensionless oval value, clearance is mm, AP/ML
    separations are deg), but we compare candidates on the same key so
    the mix is consistent.

    A candidate is feasible iff ``max_violation == 0``.
    """

    probe_to_hole: dict[str, int]
    probe_to_arc_idx: dict[str, int]
    arc_centroids_deg: tuple[float, ...]
    n_arcs: int
    x: NDArray[np.floating]
    cost: float
    breakdown: ObjectiveBreakdown
    max_violation: float
    sum_violation_sq: float
    coverage: float
    min_headstage_clearance_mm: float
    min_arc_ap_sep_deg: float
    min_intra_arc_ml_sep_deg: float
    # Per-group max-violation breakdown — helps diagnose *which* constraint
    # is forcing a candidate infeasible. Units differ per group: threading
    # is dimensionless oval value, clearance is mm, AP/ML are deg. Zero on
    # a group means that group is fully satisfied at this candidate.
    max_violation_threading: float = 0.0
    max_violation_clearance: float = 0.0
    max_violation_arc_ap_sep: float = 0.0
    max_violation_intra_arc_ml_sep: float = 0.0

    @property
    def feasible(self) -> bool:
        return self.max_violation <= 0.0

    @property
    def dominant_violation_group(self) -> str:
        """Name of the group contributing the largest violation, or
        ``"feasible"`` if the candidate satisfies every constraint."""
        if self.max_violation <= 0.0:
            return "feasible"
        groups = {
            "threading": self.max_violation_threading,
            "clearance": self.max_violation_clearance,
            "arc_ap_sep": self.max_violation_arc_ap_sep,
            "intra_arc_ml_sep": self.max_violation_intra_arc_ml_sep,
        }
        return max(groups, key=lambda k: groups[k])

    def lex_key(
        self, feasibility_threshold: float = 0.0
    ) -> tuple[float, float, float]:
        """Sort key for lexicographic ranking with a "feasible enough"
        threshold.

        ``feasibility_threshold = 0`` (default) gives strict
        feasibility-first: ``(max_violation, sum_violation_sq,
        -coverage)``. Strictly correct when the literal model and the
        physical constraints agree; aggressive when the model is more
        conservative than reality (as on 836656 with the build5 implant).

        ``feasibility_threshold = ε > 0`` collapses any plan with
        ``max_violation ≤ ε`` to the same first-tier rank (zero), so
        coverage breaks ties among "feasible enough" plans. Plans
        violating by more than ε are ranked by their excess (max_viol
        − ε). This matches the practitioner's preference: a plan with
        max_viol 0.4 / coverage 4.4 beats max_viol 0.1 / coverage 0.0
        at any ε ≥ 0.4. Set ε to the physical "slop budget" the
        manual workflow tolerates — e.g. ε = 0.5 if 0.5 mm of
        clearance overlap or 0.5° of AP/ML-sep margin is "fine".
        """
        eff_viol = max(0.0, self.max_violation - feasibility_threshold)
        return (eff_viol, self.sum_violation_sq, -self.coverage)


@dataclass(frozen=True)
class OptimizationResult:
    """Final optimizer output.

    ``probe_to_hole`` and ``probe_to_arc_idx`` give the discrete
    decisions; ``arc_centroids_deg`` gives the AP cluster centroids
    (post-optimization, the actual ``ap_arc`` values are read from
    ``x[:n_arcs]``). ``cost`` is the inner-loop scalar; ``breakdown``
    has the component-wise terms for diagnostics.

    ``best`` is the lex-best plan candidate; ``alternatives`` is the
    full ranked list of attempted (hole, arc) inner solves (best
    first), useful for surfacing top-K plans to the user. Empty when
    no inner solve completed.
    """

    probe_to_hole: dict[str, int]
    probe_to_arc_idx: dict[str, int]
    arc_centroids_deg: tuple[float, ...]
    n_arcs: int
    x: NDArray[np.floating]
    cost: float
    breakdown: ObjectiveBreakdown
    alternatives: tuple[PlanCandidate, ...] = ()


# ---------------------------------------------------------------------------
# Inner-loop helpers
# ---------------------------------------------------------------------------


def _build_inner_context(
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    hole_assignment: HoleAssignment,
    arc_assignment: ArcAssignment,
    weights: ObjectiveWeights,
    *,
    threading_oval_tolerance: float = 0.0,
    clearance_overlap_allowance_mm: float = 0.0,
) -> OptimizerContext:
    """Build :class:`OptimizerContext` from the discrete assignments."""
    n_arcs = max(arc_assignment.probe_to_arc_idx.values()) + 1
    arc_ids = tuple(f"arc_{i}" for i in range(n_arcs))
    probe_names = tuple(p.name for p in probes)
    layout = VariableLayout(arc_ids=arc_ids, probe_names=probe_names)

    holes_by_id = {h.id: h for h in holes}
    fallback_geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))

    probe_contexts: list[ProbeContext] = []
    for p in probes:
        hole_id = hole_assignment.probe_to_hole[p.name]
        arc_idx = arc_assignment.probe_to_arc_idx[p.name]
        if p.kind in RECORDING_GEOMETRY:
            geom = get_recording_geometry(p.kind)
        else:
            geom = fallback_geom
        probe_contexts.append(
            ProbeContext(
                name=p.name,
                target_LPS=np.asarray(p.target_LPS, dtype=np.float64),
                kind=p.kind,
                arc_id=f"arc_{arc_idx}",
                shank_tips_local=np.asarray(p.shank_tips_local, dtype=np.float64),
                assigned_hole=holes_by_id[hole_id],
                density_fn=gaussian_density(p.target_LPS, p.density_sigma_mm),
                recording_geom=geom,
            )
        )

    return OptimizerContext(
        layout=layout,
        probes=tuple(probe_contexts),
        weights=weights,
        threading_oval_tolerance=threading_oval_tolerance,
        clearance_overlap_allowance_mm=clearance_overlap_allowance_mm,
    )


def _build_initial_x(
    ctx: OptimizerContext,
    holes: list[Hole],
    hole_assignment: HoleAssignment,
    arc_assignment: ArcAssignment,
) -> NDArray[np.floating]:
    """Warm-start variable vector.

    - Per-arc ``ap_arc_i`` ← arc centroid from the middle layer.
    - Per-probe ``spin`` ← assigned hole's slot major-axis angle (so
      the shank row is pre-aligned to the slot — CMA-ES doesn't have
      to find the ±15° basin from random init).
    - Per-probe ``ml``, ``off_R``, ``off_A``, ``depth`` ← 0 (the
      kinematic chain auto-centers the recording array on the target).
    """
    holes_by_id = {h.id: h for h in holes}
    x = np.zeros(ctx.layout.n_vars, dtype=np.float64)
    for i, ap in enumerate(arc_assignment.arc_centroids_deg):
        x[i] = float(ap)
    for p_idx, probe in enumerate(ctx.probes):
        hole_id = hole_assignment.probe_to_hole[probe.name]
        slot_theta_rad = holes_by_id[hole_id].slot_theta_rad
        # Derive the spin that rotates the canonical shank row (along
        # local +x) onto the slot's major axis under
        # ``arc_angles_to_affine(0, 0, spin)``. With that rotation,
        # ``R @ [1, 0, 0] = (-cos(spin), sin(spin), 0)``; matching
        # ``slot_major = (-sin(theta), cos(theta), 0)`` gives
        # ``spin = π/2 − theta`` (modulo π by slot symmetry).
        spin_deg = float(np.rad2deg(np.pi / 2 - slot_theta_rad))
        off = ctx.layout.num_arcs + 5 * p_idx
        # Order is (ml, spin, off_R, off_A, depth) — see VariableLayout.
        x[off + 0] = 0.0
        x[off + 1] = spin_deg
        x[off + 2] = 0.0
        x[off + 3] = 0.0
        x[off + 4] = 0.0
    return x


def _try_cma_es(
    ctx: OptimizerContext,
    x0: NDArray,
    *,
    population: int,
    generations: int,
    sigma: float,
) -> NDArray | None:
    """Run CMA-ES if the ``cma`` package is available; return the best
    ``x`` or ``None`` if the package is missing."""
    try:
        import cma  # type: ignore[import-not-found]
    except ImportError:
        return None
    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        sigma,
        {
            "popsize": population,
            "maxiter": generations,
            "verbose": -9,
        },
    )
    es.optimize(lambda v: scalar_objective(np.asarray(v, dtype=np.float64), ctx))
    return np.asarray(es.result.xbest, dtype=np.float64)


def _scale_weights(w: ObjectiveWeights, mult: float) -> ObjectiveWeights:
    """Return a new :class:`ObjectiveWeights` with feasibility-penalty
    lambdas multiplied by ``mult``. Coverage / margin terms unchanged."""
    return ObjectiveWeights(
        lambda_threading=w.lambda_threading * mult,
        lambda_clearance=w.lambda_clearance * mult,
        lambda_kinematic=w.lambda_kinematic * mult,
        lambda_margin=w.lambda_margin,
        margin_softmin_beta=w.margin_softmin_beta,
        safety_clearance_mm=w.safety_clearance_mm,
    )


def _ctx_with_weights(ctx: OptimizerContext, w: ObjectiveWeights) -> OptimizerContext:
    """Return a copy of ``ctx`` with replaced ``weights``."""
    return OptimizerContext(
        layout=ctx.layout,
        probes=ctx.probes,
        arc_for_probe=ctx.arc_for_probe,
        weights=w,
        shaft_length_mm=ctx.shaft_length_mm,
        shank_radius_mm=ctx.shank_radius_mm,
        headstage_base_along_shaft_mm=ctx.headstage_base_along_shaft_mm,
        headstage_length_mm=ctx.headstage_length_mm,
        headstage_radius_mm=ctx.headstage_radius_mm,
        min_arc_ap_sep_deg=ctx.min_arc_ap_sep_deg,
        min_within_arc_ml_sep_deg=ctx.min_within_arc_ml_sep_deg,
        coverage_n_samples=ctx.coverage_n_samples,
        threading_oval_tolerance=ctx.threading_oval_tolerance,
        clearance_overlap_allowance_mm=ctx.clearance_overlap_allowance_mm,
    )


def _multistage_cma_es(
    ctx: OptimizerContext,
    x0: NDArray,
    *,
    population: int,
    total_generations: int,
    sigma: float,
    stage_multipliers: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> NDArray | None:
    """Run CMA-ES in stages with a homotopy schedule on the feasibility
    penalties. Each stage scales ``lambda_threading``, ``lambda_clearance``,
    and ``lambda_kinematic`` by the corresponding multiplier, warm-starting
    from the previous stage's best ``x``.

    Early stages (low multiplier) let CMA-ES explore — the search space
    is smoother and coverage / global geometry can dominate. Late stages
    (high multiplier) clamp down on threading / kinematic violations,
    forcing the optimizer to find genuinely feasible solutions.

    Generations are split evenly across stages.
    """
    try:
        import cma  # type: ignore[import-not-found]
    except ImportError:
        return None
    n_stages = len(stage_multipliers)
    gens_per_stage = max(1, total_generations // n_stages)
    sigma_per_stage = sigma
    x = np.asarray(x0, dtype=np.float64).copy()
    for mult in stage_multipliers:
        stage_ctx = _ctx_with_weights(ctx, _scale_weights(ctx.weights, mult))
        es = cma.CMAEvolutionStrategy(
            x.tolist(),
            sigma_per_stage,
            {
                "popsize": population,
                "maxiter": gens_per_stage,
                "verbose": -9,
            },
        )
        es.optimize(
            lambda v: scalar_objective(np.asarray(v, dtype=np.float64), stage_ctx)
        )
        x = np.asarray(es.result.xbest, dtype=np.float64)
        # Tighten sigma between stages so later stages refine rather
        # than re-explore. Keep enough room for non-trivial moves
        # since the penalty landscape is also changing.
        sigma_per_stage = max(sigma_per_stage * 0.5, 0.5)
    return x


def _default_bounds(ctx: OptimizerContext) -> list[tuple[float, float]]:
    """Conservative box bounds for SLSQP, by variable role.

    These keep the optimizer from wandering off into absurd regions
    while still leaving room for any reasonable physical configuration.
    Order matches :class:`VariableLayout`: arc-APs first, then per-probe
    ``(ml, spin, off_R, off_A, depth)`` blocks.
    """
    bounds: list[tuple[float, float]] = []
    for _ in range(ctx.layout.num_arcs):
        bounds.append((-60.0, +60.0))   # ap_arc deg
    for _ in range(ctx.layout.num_probes):
        bounds.append((-30.0, +30.0))   # ml_local deg
        bounds.append((-180.0, +180.0)) # spin deg
        # Lateral offsets must stay within the slot's half-extent so
        # the probe physically fits through the bore. Bounding to
        # ±0.5 mm keeps the threading constraint inside the optimizer's
        # feasible basin.
        bounds.append((-0.5, +0.5))     # off_R mm
        bounds.append((-0.5, +0.5))     # off_A mm
        # past_target_mm: typical probe insertion is 2-5 mm into brain;
        # ±3 mm covers a centered recording bank with margin.
        bounds.append((-3.0, +3.0))     # past_target_mm
    return bounds


def _slsqp_polish(
    ctx: OptimizerContext, x0: NDArray, *, max_iter: int
) -> tuple[NDArray, ObjectiveBreakdown]:
    """SLSQP local minimization with box bounds.

    Constraints are folded into the objective via
    :class:`ObjectiveWeights` penalties (no separate ``constraints=...``
    argument) — this keeps the inner-loop API simple at the cost of
    slower convergence near tight feasibility boundaries.

    Box bounds prevent runaway behaviour (without them SLSQP can walk
    variables to ~10⁷ when the gradient is malformed near the
    constraint boundary).
    """

    def fn(v):
        return scalar_objective(np.asarray(v, dtype=np.float64), ctx)

    result = minimize(
        fn,
        np.asarray(x0, dtype=np.float64),
        method="SLSQP",
        bounds=_default_bounds(ctx),
        options={"maxiter": max_iter, "ftol": 1e-6, "disp": False},
    )
    x_opt = np.asarray(result.x, dtype=np.float64)
    return x_opt, evaluate_objective(x_opt, ctx)


def _feasibility_solve(
    ctx: OptimizerContext, x0: NDArray, *, max_iter: int
) -> tuple[NDArray, float]:
    """Stage A of the two-stage inner solve.

    Minimises ``feasibility_violation_squared`` (sum of squared
    constraint violations across threading, clearance, AP sep, and
    intra-arc ML sep) starting from ``x0``. SLSQP with bounds only —
    no ``ineq`` constraints, since the *objective* is the violation.

    Returns ``(x, violation²)``. Even if the result isn't strictly
    feasible (``violation² > 0``), it's the point that minimises
    distance from the feasibility tube — a much better warm start for
    Stage B's coverage polish than the raw CMA-ES output, which can
    sit deep inside an infeasible region.
    """
    bounds = _default_bounds(ctx)

    def fn(v):
        return feasibility_violation_squared(np.asarray(v, dtype=np.float64), ctx)

    result = minimize(
        fn,
        np.asarray(x0, dtype=np.float64),
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-9, "disp": False},
    )
    return np.asarray(result.x, dtype=np.float64), float(result.fun)


def _slsqp_polish_constrained(
    ctx: OptimizerContext, x0: NDArray, *, max_iter: int,
    method: str = "SLSQP",
) -> tuple[NDArray, ObjectiveBreakdown]:
    """SLSQP (or trust-constr) polish with native inequality constraints.

    Objective is just ``-coverage_total``; threading, clearance, and
    kinematic separations are passed as ``ineq`` constraints.

    ``method="SLSQP"`` (default) uses scipy's vector-valued ineq
    constraint dicts. SLSQP doesn't strictly maintain feasibility
    between iterations — after an active-set step, finite-diff Jacobian
    noise can let the iterate drift outside the feasibility tube. For a
    coverage objective whose gradient pulls probes off-bore, this drift
    accumulates.

    ``method="trust-constr"`` uses interior-point with a barrier; the
    iterate is kept inside the feasibility tube (or the strictly-positive
    side of slack) by construction. Slower per iteration but the
    coverage polish stays on the constraint manifold.
    """
    bounds = _default_bounds(ctx)

    def obj(v):
        return coverage_objective(np.asarray(v, dtype=np.float64), ctx)

    def make_constraint(field_name):
        def fn(v):
            cv = evaluate_constraints(np.asarray(v, dtype=np.float64), ctx)
            arr = np.asarray(getattr(cv, field_name), dtype=np.float64)
            # SciPy SLSQP can't handle empty arrays; substitute a single
            # always-feasible scalar.
            return arr if arr.size > 0 else np.array([1.0])
        return fn

    field_names = (
        "threading", "clearance",
        "arc_ap_separation", "intra_arc_ml_separation",
    )
    if method == "SLSQP":
        constraints = [
            {"type": "ineq", "fun": make_constraint(name)}
            for name in field_names
        ]
        options = {"maxiter": max_iter, "ftol": 1e-6, "disp": False}
    elif method == "trust-constr":
        from scipy.optimize import NonlinearConstraint

        constraints = [
            NonlinearConstraint(
                make_constraint(name), 0.0, np.inf,
            )
            for name in field_names
        ]
        options = {
            "maxiter": max_iter,
            "xtol": 1e-7, "gtol": 1e-6,
            "verbose": 0,
            "initial_constr_penalty": 1.0,
        }
    else:
        raise ValueError(f"unknown method {method!r}")

    result = minimize(
        obj,
        np.asarray(x0, dtype=np.float64),
        method=method,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    x_opt = np.asarray(result.x, dtype=np.float64)
    return x_opt, evaluate_objective(x_opt, ctx)


# ---------------------------------------------------------------------------
# Plan candidate construction + reporting
# ---------------------------------------------------------------------------


def _build_plan_candidate(
    x_opt: NDArray,
    ctx: OptimizerContext,
    breakdown: ObjectiveBreakdown,
    *,
    ha: HoleAssignment,
    aa: ArcAssignment,
    n_arcs: int,
) -> PlanCandidate:
    """Wrap an inner-solve outcome with its lex-ranking metrics.

    Pulls feasibility from the slack vectors (any slack < 0 ⇒ infeasible)
    and the actually-realised minimums (clearance in mm; AP/ML separations
    in deg) so the report can show physical numbers, not penalty terms.
    """
    cv = evaluate_constraints(x_opt, ctx)
    max_viol = 0.0
    sum_viol_sq = 0.0
    per_group_max: dict[str, float] = {}
    for name, arr in (
        ("threading", cv.threading),
        ("clearance", cv.clearance),
        ("arc_ap_separation", cv.arc_ap_separation),
        ("intra_arc_ml_separation", cv.intra_arc_ml_separation),
    ):
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            per_group_max[name] = 0.0
            continue
        excess = np.maximum(0.0, -a)
        group_max = float(excess.max()) if excess.size > 0 else 0.0
        per_group_max[name] = group_max
        if group_max > max_viol:
            max_viol = group_max
        sum_viol_sq += float(np.sum(excess * excess))

    # Physical mins (in their native units). Subtract the tolerance/
    # allowance shifts so reporting reflects raw geometry, independent
    # of what slack we're tolerating.
    min_clear = (
        float(cv.clearance.min())
        + ctx.weights.safety_clearance_mm
        - ctx.clearance_overlap_allowance_mm
        if cv.clearance.size else float("inf")
    )
    min_ap_sep = (
        float(cv.arc_ap_separation.min()) + ctx.min_arc_ap_sep_deg
        if cv.arc_ap_separation.size else float("inf")
    )
    min_ml_sep = (
        float(cv.intra_arc_ml_separation.min()) + ctx.min_within_arc_ml_sep_deg
        if cv.intra_arc_ml_separation.size else float("inf")
    )

    return PlanCandidate(
        probe_to_hole=dict(ha.probe_to_hole),
        probe_to_arc_idx=dict(aa.probe_to_arc_idx),
        arc_centroids_deg=aa.arc_centroids_deg,
        n_arcs=n_arcs,
        x=np.asarray(x_opt, dtype=np.float64),
        cost=float(breakdown.total),
        breakdown=breakdown,
        max_violation=max_viol,
        sum_violation_sq=sum_viol_sq,
        coverage=float(cv.coverage_total),
        min_headstage_clearance_mm=min_clear,
        min_arc_ap_sep_deg=min_ap_sep,
        min_intra_arc_ml_sep_deg=min_ml_sep,
        max_violation_threading=per_group_max["threading"],
        max_violation_clearance=per_group_max["clearance"],
        max_violation_arc_ap_sep=per_group_max["arc_ap_separation"],
        max_violation_intra_arc_ml_sep=per_group_max["intra_arc_ml_separation"],
    )


def format_plan_table(candidates: tuple[PlanCandidate, ...]) -> str:
    """Render the lex-ranked candidates as a Markdown table.

    Columns: rank, feasible, max violation (worst constraint slack — unitless
    for threading; mm for clearance; deg for AP/ML sep), min headstage
    clearance (mm), min arc-AP / intra-arc-ML separation (deg), coverage,
    inner cost. Use sparingly — meant for end-of-run summary, not per-iter.
    """
    if not candidates:
        return "(no candidates)"
    rows = [
        "| # | feas? | max viol | dominant group | thread | clear (mm) | AP sep (°) | ML sep (°) | coverage | cost |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, c in enumerate(candidates, start=1):
        rows.append(
            f"| {i} | {'yes' if c.feasible else 'no'} | "
            f"{c.max_violation:.4g} | "
            f"{c.dominant_violation_group} | "
            f"{c.max_violation_threading:.3g} | "
            f"{c.min_headstage_clearance_mm:.3f} | "
            f"{c.min_arc_ap_sep_deg:.2f} | "
            f"{c.min_intra_arc_ml_sep_deg:.2f} | "
            f"{c.coverage:.3f} | "
            f"{c.cost:.3g} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def optimize(
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    *,
    max_num_arcs: int,
    min_num_arcs: int = 1,
    k_holes: int = 5,
    k_arcs: int = 3,
    weights: ObjectiveWeights = ObjectiveWeights(),
    arc_count_penalty_deg2: float = 25.0,
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
    verbose: bool = False,
) -> OptimizationResult | None:
    """Run the three-level optimizer.

    Returns ``None`` if no feasible (hole, arc) combination is found.
    See :class:`OptimizationResult` for the output shape.
    """
    if not probes:
        return None

    # 1. Outer: probe→hole.
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
    hole_assignments = solve_top_k_assignments(
        assignment_probes, holes, k=k_holes
    )
    if not hole_assignments:
        if verbose:
            print("[optimize] No feasible hole assignment.")
        return None

    if use_cma:
        try:
            import cma  # noqa: F401
        except ImportError:
            warnings.warn(
                "cma package not installed; running SLSQP-only inner loop. "
                "Install via `uv add cma` to enable the global CMA-ES stage.",
                stacklevel=2,
            )
            use_cma = False

    candidates: list[PlanCandidate] = []
    for ha_idx, ha in enumerate(hole_assignments):
        if verbose:
            print(
                f"[optimize] hole assignment #{ha_idx} "
                f"(cost={ha.cost:.3f}): {ha.probe_to_hole}"
            )
        # 2. Middle: probe→arc.
        arc_assignments = solve_top_k_arc_assignments(
            ha.probe_to_hole, holes,
            max_num_arcs=max_num_arcs,
            min_num_arcs=min_num_arcs,
            k=k_arcs,
            min_arc_ap_sep_deg=min_arc_ap_sep_deg,
            arc_sep_shortfall_weight=arc_sep_shortfall_weight,
            arc_count_penalty_deg2=arc_count_penalty_deg2,
        )
        if not arc_assignments:
            if verbose:
                print("  no feasible arc assignment, skipping.")
            continue

        for aa_idx, aa in enumerate(arc_assignments):
            n_arcs = max(aa.probe_to_arc_idx.values()) + 1
            if verbose:
                print(
                    f"  arc assignment #{aa_idx} "
                    f"(cost={aa.cost:.3f}, n_arcs={n_arcs}): "
                    f"{aa.probe_to_arc_idx}"
                )

            ctx = _build_inner_context(
                probes, holes, ha, aa, weights,
                threading_oval_tolerance=threading_oval_tolerance,
                clearance_overlap_allowance_mm=clearance_overlap_allowance_mm,
            )
            x0 = _build_initial_x(ctx, holes, ha, aa)

            x = x0
            if use_cma:
                if cma_stage_multipliers:
                    x_cma = _multistage_cma_es(
                        ctx, x,
                        population=cma_population,
                        total_generations=cma_generations,
                        sigma=cma_sigma,
                        stage_multipliers=cma_stage_multipliers,
                    )
                else:
                    x_cma = _try_cma_es(
                        ctx, x,
                        population=cma_population,
                        generations=cma_generations,
                        sigma=cma_sigma,
                    )
                if x_cma is not None:
                    x = x_cma
            if slsqp_constrained and two_stage_inner:
                v_pre = feasibility_violation_squared(x, ctx)
                x_feas, v_feas = _feasibility_solve(
                    ctx, x, max_iter=feasibility_max_iter
                )
                if v_feas <= v_pre:
                    x = x_feas
                if verbose:
                    print(
                        f"    feasibility stage: violation² "
                        f"{v_pre:.4g} → {v_feas:.4g}"
                    )
            if slsqp_constrained:
                x_opt, breakdown_opt = _slsqp_polish_constrained(
                    ctx, x, max_iter=slsqp_max_iter,
                    method=polish_method,
                )
            else:
                x_opt, breakdown_opt = _slsqp_polish(
                    ctx, x, max_iter=slsqp_max_iter
                )

            cand = _build_plan_candidate(
                x_opt, ctx, breakdown_opt,
                ha=ha, aa=aa, n_arcs=n_arcs,
            )
            # Stage C: feasibility cleanup. Stage B's coverage polish can
            # drift off the feasibility tube because the coverage gradient
            # pulls probes off-bore; re-solve from x_opt and keep
            # whichever result has the better lex key. Cheap (one extra
            # bounded SLSQP call) and never produces a worse candidate.
            if (
                slsqp_constrained
                and final_feasibility_cleanup
                and not cand.feasible
            ):
                x_clean, v_clean = _feasibility_solve(
                    ctx, x_opt, max_iter=feasibility_max_iter
                )
                breakdown_clean = evaluate_objective(x_clean, ctx)
                cand_clean = _build_plan_candidate(
                    x_clean, ctx, breakdown_clean,
                    ha=ha, aa=aa, n_arcs=n_arcs,
                )
                if (
                    cand_clean.lex_key(feasibility_threshold)
                    < cand.lex_key(feasibility_threshold)
                ):
                    if verbose:
                        print(
                            f"    Stage C: max_viol "
                            f"{cand.max_violation:.4g} → "
                            f"{cand_clean.max_violation:.4g}, "
                            f"coverage {cand.coverage:.3f} → "
                            f"{cand_clean.coverage:.3f}"
                        )
                    cand = cand_clean
            candidates.append(cand)
            if verbose:
                print(
                    f"    inner cost={breakdown_opt.total:.3f}  "
                    f"(coverage={breakdown_opt.coverage_total:.3f}, "
                    f"thread={breakdown_opt.threading_penalty:.3f}, "
                    f"clear={breakdown_opt.clearance_penalty:.3f}, "
                    f"kin={breakdown_opt.kinematic_penalty:.3f})"
                    f"  feas={'Y' if cand.feasible else 'N'}"
                    f" max_viol={cand.max_violation:.4g}"
                )

    if not candidates:
        return None
    candidates.sort(key=lambda c: c.lex_key(feasibility_threshold))
    best_cand = candidates[0]
    if verbose:
        print(
            f"[optimize] best plan: feasible={best_cand.feasible} "
            f"max_viol={best_cand.max_violation:.4g} "
            f"coverage={best_cand.coverage:.3f} "
            f"cost={best_cand.cost:.3f}"
        )
    return OptimizationResult(
        probe_to_hole=best_cand.probe_to_hole,
        probe_to_arc_idx=best_cand.probe_to_arc_idx,
        arc_centroids_deg=best_cand.arc_centroids_deg,
        n_arcs=best_cand.n_arcs,
        x=best_cand.x,
        cost=best_cand.cost,
        breakdown=best_cand.breakdown,
        alternatives=tuple(candidates),
    )
