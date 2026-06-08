"""Three-level optimizer driver.

Glues the outer layer (probe→hole, ``hole_assignment``), the middle
layer (probe→arc, ``arc_assignment``), and the inner SLSQP polish on
the :func:`evaluate_objective` scalar into a single ``optimize()``
entry point.

::

    enumerate top-K_h probe→hole assignments
    for each:
      enumerate top-K_a probe→arc assignments
      for each:
        warm-start variable vector x0
        SLSQP polish
        record (cost, x, ObjectiveBreakdown)
    return the (hole, arc, x) combination with the best inner cost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.optimize import minimize

from aind_low_point.optimization.arc_assignment import (
    ArcAssignment,
    solve_top_k_arc_assignments,
)
from aind_low_point.optimization.density import (
    gaussian_density,
    voxel_kde_density,
)
from aind_low_point.optimization.geometry import shaft_section_oval_value
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.hole_assignment import (
    AssignmentProbe,
    HoleAssignment,
    solve_top_k_assignments,
)
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.kinematics import (
    pose_from_optimizer_vars,
    shank_capsules_from_pose,
)
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
    objective's Gaussian width / mixture bandwidth. ``collision_mesh``
    (optional) is the probe's canonical-local mesh used for the
    inner-loop pairwise clearance constraint via FCL BVH / GJK; pass
    the full probe mesh (not just the headstage region) so the silicon
    body and connector regions are included.

    ``target_points`` (optional) holds an ``(N, 3)`` point cloud (in
    world LPS mm) that, when set, switches the coverage density from a
    single-point Gaussian on ``target_LPS`` to an equally-weighted
    Gaussian mixture over the cloud. ``target_LPS`` is still used for
    LSAP target-anchored pose-bank construction and should be the
    cloud's centroid in that case.
    """

    name: str
    target_LPS: NDArray[np.floating]
    kind: str
    shank_tips_local: NDArray[np.floating]
    density_sigma_mm: float = 0.5
    collision_mesh: trimesh.Trimesh | None = field(default=None, compare=False)
    target_points: NDArray[np.floating] | None = field(default=None, compare=False)
    # Per-target priority weight applied (after normalization) to this probe's
    # coverage in the normalized objective's weighted SUM (the fairness floor
    # stays unweighted). 1.0 ⇒ no preference.
    coverage_weight: float = 1.0


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

    # Numerical tolerance for declaring strict feasibility. SLSQP and
    # trust-constr both leave sub-micron residuals on active constraints
    # at convergence; ``1e-6`` is comfortably below any physically
    # meaningful violation (mm, deg, dimensionless oval value all use
    # this scale).
    _FEASIBLE_EPSILON: ClassVar[float] = 1e-6

    @property
    def feasible(self) -> bool:
        return self.max_violation <= self._FEASIBLE_EPSILON

    @property
    def dominant_violation_group(self) -> str:
        """Name of the group contributing the largest violation, or
        ``"feasible"`` if the candidate satisfies every constraint."""
        if self.max_violation <= self._FEASIBLE_EPSILON:
            return "feasible"
        groups = {
            "threading": self.max_violation_threading,
            "clearance": self.max_violation_clearance,
            "arc_ap_sep": self.max_violation_arc_ap_sep,
            "intra_arc_ml_sep": self.max_violation_intra_arc_ml_sep,
        }
        return max(groups, key=lambda k: groups[k])

    def lex_key(self, feasibility_threshold: float = 0.0) -> tuple[float, float, float]:
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
    subject_from_rig_rot: NDArray | None = None,
    bvh_cache=None,  # dict[probe_name, fcl.CollisionObject | None] | None
) -> OptimizerContext:
    """Build :class:`OptimizerContext` from the discrete assignments.

    ``bvh_cache`` (optional) lets the caller pre-build the per-probe
    FCL BVH objects once and reuse them across multiple inner-solve
    runs — saves ~100 ms / probe / call on 46k-face meshes.
    """
    n_arcs = max(arc_assignment.probe_to_arc_idx.values()) + 1
    arc_ids = tuple(f"arc_{i}" for i in range(n_arcs))
    probe_names = tuple(p.name for p in probes)
    layout = VariableLayout(arc_ids=arc_ids, probe_names=probe_names)

    holes_by_id = {h.id: h for h in holes}
    fallback_geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))

    probe_contexts: list[ProbeContext] = []
    headstage_objs: list[object] = []  # fcl.CollisionObject | None
    for p in probes:
        hole_id = hole_assignment.probe_to_hole[p.name]
        arc_idx = arc_assignment.probe_to_arc_idx[p.name]
        if p.kind in RECORDING_GEOMETRY:
            geom = get_recording_geometry(p.kind)
        else:
            geom = fallback_geom
        if p.target_points is not None:
            density_fn = voxel_kde_density(p.target_points, sigma_mm=p.density_sigma_mm)
        else:
            density_fn = gaussian_density(p.target_LPS, p.density_sigma_mm)
        probe_contexts.append(
            ProbeContext(
                name=p.name,
                target_LPS=np.asarray(p.target_LPS, dtype=np.float64),
                kind=p.kind,
                arc_id=f"arc_{arc_idx}",
                shank_tips_local=np.asarray(p.shank_tips_local, dtype=np.float64),
                assigned_hole=holes_by_id[hole_id],
                density_fn=density_fn,
                recording_geom=geom,
            )
        )
        # Build a fresh FCL BVH CollisionObject per probe. The mesh
        # vertices live in the canonical-local frame; per-iteration the
        # objective updates each object's transform from the optimizer's
        # current (R, pose_tip). BVH (full mesh) rather than Convex hull
        # — the convex hull of a probe body fills concavities and the
        # body-region hull misses the silicon-body / connector stretch
        # between shanks and PCB, both of which break the pairwise
        # clearance check on real plans.
        if bvh_cache is not None and p.name in bvh_cache:
            headstage_objs.append(bvh_cache[p.name])
        elif p.collision_mesh is not None:
            headstage_objs.append(make_fcl_bvh(p.collision_mesh))
        else:
            headstage_objs.append(None)

    return OptimizerContext(
        layout=layout,
        probes=tuple(probe_contexts),
        weights=weights,
        threading_oval_tolerance=threading_oval_tolerance,
        clearance_overlap_allowance_mm=clearance_overlap_allowance_mm,
        subject_from_rig_rot=subject_from_rig_rot,
        headstage_fcl_objs=tuple(headstage_objs),
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


def _head_pitch_about_L_deg(subject_from_rig_rot: NDArray | None) -> float:
    """Extract the X-axis (= LPS L axis) Euler component of the head tilt.

    For the typical headframe (mouse mounted with nose pitched down),
    ``subject_from_rig`` is a pure rotation about LPS +x; this helper
    returns that rotation angle in degrees. For non-pure-X rotations,
    returns the X-component of the XYZ Euler decomposition — good enough
    for shifting the AP bound. Returns 0 when the rotation is identity
    or ``None``.
    """
    if subject_from_rig_rot is None:
        return 0.0
    R = np.asarray(subject_from_rig_rot, dtype=np.float64)
    # Standard X-component of rotation: atan2(R[2,1], R[1,1]) for a
    # pure-X rotation matrix; matches the X Euler angle of an XYZ
    # decomposition when the Y component is near zero.
    return float(np.rad2deg(np.arctan2(R[2, 1], R[1, 1])))


def _default_bounds(ctx: OptimizerContext) -> list[tuple[float, float]]:
    """Box bounds for SLSQP, matching the rig's mechanical joint limits.

    Order matches :class:`VariableLayout`: arc-APs first, then per-probe
    ``(ml, spin, off_R, off_A, depth)`` blocks. AP and ML bounds match
    ``PoseLimits`` defaults so the optimizer is allowed to explore the
    full rig envelope (±60° AP/ML) — but expressed in *subject-frame*
    angles, the rig's AP envelope is shifted by the head tilt.

    Concretely: with ``rig_ap = subject_ap − head_pitch_about_L``,
    the rig's ``|rig_ap| ≤ 60°`` becomes
    ``subject_ap ∈ [−60 + head_pitch, 60 + head_pitch]``. With the
    headframe's nominal +14° pitch about L (nose down), the
    subject-frame AP range is ``[−46°, +74°]`` — asymmetric, biased
    toward positive (mouse pitch-down) values.

    Head tilt is assumed to be purely about the L axis (R-axis in RAS).
    ML and spin ranges are unaffected.
    """
    head_pitch = _head_pitch_about_L_deg(ctx.subject_from_rig_rot)
    ap_lo = -60.0 + head_pitch
    ap_hi = +60.0 + head_pitch

    bounds: list[tuple[float, float]] = []
    for _ in range(ctx.layout.num_arcs):
        bounds.append((ap_lo, ap_hi))  # ap_arc deg — rig limit, shifted by head pitch
    for _ in range(ctx.layout.num_probes):
        bounds.append((-60.0, +60.0))  # ml_local deg — rig mechanical limit
        bounds.append((-180.0, +180.0))  # spin deg
        # NB: scipy SLSQP uses bounds for internal step-size scaling; do not
        # widen these (verified 2026-05-19: widening to ±720° caused
        # immediate stagnation). The batched Adam path uses loose bounds
        # in its own static; only scipy stays at ±180°.
        # Lateral offsets must stay within the slot's half-extent so
        # the probe physically fits through the bore. Bounding to
        # ±0.5 mm keeps the threading constraint inside the optimizer's
        # feasible basin.
        bounds.append((-0.5, +0.5))  # off_R mm
        bounds.append((-0.5, +0.5))  # off_A mm
        # past_target_mm: typical probe insertion is 2-5 mm into brain;
        # ±3 mm covers a centered recording bank with margin.
        bounds.append((-3.0, +3.0))  # past_target_mm
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
    ctx: OptimizerContext,
    x0: NDArray,
    *,
    max_iter: int,
    method: str = "SLSQP",
    sdf_clearance_fun=None,
    sdf_clearance_jac=None,
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
        "threading",
        "clearance",
        "arc_ap_separation",
        "intra_arc_ml_separation",
    )
    # When the SDF/JAX backend is engaged for clearance, also use the
    # JAX threading + AP-sep + intra-ML-sep with analytic Jacobians.
    # The three groups share their JIT cache via stage3_jax — Stage 3
    # polishes the survivors with stable hole/arc assignments, so cache
    # hits dominate after the first call.
    jax_stage3: dict | None = None
    if sdf_clearance_fun is not None:
        from aind_low_point.optimization import stage3_jax as _s3

        jax_stage3 = _s3.make_stage3_constraints(ctx)
    if method == "SLSQP":
        constraints = []
        for name in field_names:
            if name == "clearance" and sdf_clearance_fun is not None:
                # Use SDF-backed clearance with analytic Jacobian.
                # Smooth gradient through overlap; SLSQP gets a reliable
                # constraint Jacobian instead of finite-diff on noisy
                # FCL distance.
                d = {"type": "ineq", "fun": sdf_clearance_fun}
                if sdf_clearance_jac is not None:
                    d["jac"] = sdf_clearance_jac
                constraints.append(d)
            elif jax_stage3 is not None and name in jax_stage3:
                d = {
                    "type": "ineq",
                    "fun": jax_stage3[name]["fun"],
                    "jac": jax_stage3[name]["jac"],
                }
                constraints.append(d)
            else:
                constraints.append({"type": "ineq", "fun": make_constraint(name)})
        options = {"maxiter": max_iter, "ftol": 1e-6, "disp": False}
    elif method == "trust-constr":
        from scipy.optimize import NonlinearConstraint

        constraints = [
            NonlinearConstraint(
                make_constraint(name),
                0.0,
                np.inf,
            )
            for name in field_names
        ]
        options = {
            "maxiter": max_iter,
            "xtol": 1e-7,
            "gtol": 1e-6,
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
        if cv.clearance.size
        else float("inf")
    )
    min_ap_sep = (
        float(cv.arc_ap_separation.min()) + ctx.min_arc_ap_sep_deg
        if cv.arc_ap_separation.size
        else float("inf")
    )
    min_ml_sep = (
        float(cv.intra_arc_ml_separation.min()) + ctx.min_within_arc_ml_sep_deg
        if cv.intra_arc_ml_separation.size
        else float("inf")
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
        "| # | feas? | max viol | dominant group | thread | clear (mm) | "
        "AP sep (°) | ML sep (°) | coverage | cost |",
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
# Best-fit hole detection (for seeded inner solves)
# ---------------------------------------------------------------------------


def best_fit_hole_id_at_pose(
    probe: ProbeStaticInfo,
    holes: list[Hole],
    *,
    ap_deg: float,
    ml_deg: float,
    spin_deg: float,
    off_R_mm: float,
    off_A_mm: float,
    past_target_mm: float,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
    recording_geom: RecordingGeometry | None = None,
) -> tuple[int, float]:
    """Pick the hole whose max(g_thread) at the given pose is smallest.

    Returns ``(hole_id, max_g)``. ``max_g <= 0`` means the probe physically
    threads through that bore; ``> 0`` means the shank row would graze the
    slot wall. Used by ``polish_seed`` and ``scripts/score_manual_plan.py``
    to infer the bore implied by a manually-authored plan.
    """
    if recording_geom is None:
        recording_geom = (
            get_recording_geometry(probe.kind)
            if probe.kind in RECORDING_GEOMETRY
            else RecordingGeometry(active_ranges_mm=((0.2, 1.2),))
        )
    tips = np.asarray(probe.shank_tips_local, dtype=np.float64)
    if tips.shape[0] > 0:
        pivot_local = np.array(
            [
                float(tips[:, 0].mean()),
                float(tips[:, 1].mean()),
                float(recording_geom.active_center_mm),
            ],
            dtype=np.float64,
        )
    else:
        from aind_low_point.optimization.recording import (
            recording_center_local_for_kind,
        )

        pivot_local = recording_center_local_for_kind(probe.kind)
    R, pose_tip = pose_from_optimizer_vars(
        target_LPS=probe.target_LPS,
        ap_deg=ap_deg,
        ml_deg=ml_deg,
        spin_deg=spin_deg,
        offset_R_mm=off_R_mm,
        offset_A_mm=off_A_mm,
        past_target_mm=past_target_mm,
        recording_center_local=pivot_local,
    )
    shanks = shank_capsules_from_pose(
        R,
        pose_tip,
        probe.shank_tips_local,
        shaft_length_mm=shaft_length_mm,
        shank_radius_mm=shank_radius_mm,
    )
    best_id = -1
    best_max_g = float("inf")
    for h in holes:
        max_g = max(
            shaft_section_oval_value(sh, sec) for sh in shanks for sec in h.sections
        )
        if max_g < best_max_g:
            best_max_g = max_g
            best_id = int(h.id)
    return best_id, float(best_max_g)


# ---------------------------------------------------------------------------
# Inner-loop body (shared by `optimize` and `polish_seed`)
# ---------------------------------------------------------------------------


def _inner_solve_one(
    ctx: OptimizerContext,
    x0: NDArray,
    *,
    ha: HoleAssignment,
    aa: ArcAssignment,
    n_arcs: int,
    slsqp_max_iter: int,
    slsqp_constrained: bool,
    two_stage_inner: bool,
    feasibility_max_iter: int,
    final_feasibility_cleanup: bool,
    polish_method: str,
    feasibility_threshold: float,
    verbose: bool,
    sdf_clearance_fun=None,
    sdf_clearance_jac=None,
) -> PlanCandidate:
    """Run the inner solve for one (hole, arc) combination from a warm start.

    Encapsulates Stage A feasibility solve (optional) → Stage B coverage
    polish → Stage C feasibility cleanup (optional). Used by both
    :func:`optimize` and :func:`polish_seed`; the only difference between
    callers is how ``x0`` is chosen.
    """
    x = np.asarray(x0, dtype=np.float64)
    if slsqp_constrained and two_stage_inner:
        v_pre = feasibility_violation_squared(x, ctx)
        x_feas, v_feas = _feasibility_solve(ctx, x, max_iter=feasibility_max_iter)
        if v_feas <= v_pre:
            x = x_feas
        if verbose:
            print(f"    feasibility stage: violation² {v_pre:.4g} → {v_feas:.4g}")
    if slsqp_constrained:
        x_opt, breakdown_opt = _slsqp_polish_constrained(
            ctx,
            x,
            max_iter=slsqp_max_iter,
            method=polish_method,
            sdf_clearance_fun=sdf_clearance_fun,
            sdf_clearance_jac=sdf_clearance_jac,
        )
    else:
        x_opt, breakdown_opt = _slsqp_polish(ctx, x, max_iter=slsqp_max_iter)

    cand = _build_plan_candidate(x_opt, ctx, breakdown_opt, ha=ha, aa=aa, n_arcs=n_arcs)
    if slsqp_constrained and final_feasibility_cleanup and not cand.feasible:
        x_clean, _v_clean = _feasibility_solve(
            ctx, x_opt, max_iter=feasibility_max_iter
        )
        breakdown_clean = evaluate_objective(x_clean, ctx)
        cand_clean = _build_plan_candidate(
            x_clean, ctx, breakdown_clean, ha=ha, aa=aa, n_arcs=n_arcs
        )
        if cand_clean.lex_key(feasibility_threshold) < cand.lex_key(
            feasibility_threshold
        ):
            if verbose:
                print(
                    f"    Stage C: max_viol "
                    f"{cand.max_violation:.4g} → {cand_clean.max_violation:.4g}, "
                    f"coverage {cand.coverage:.3f} → {cand_clean.coverage:.3f}"
                )
            cand = cand_clean
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
    return cand


# ---------------------------------------------------------------------------
# Seed-plan polish (skip outer + middle layers)
# ---------------------------------------------------------------------------


def polish_seed(
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    *,
    probe_to_hole: dict[str, int],
    probe_to_arc_idx: dict[str, int],
    arc_centroids_deg: tuple[float, ...],
    x0: NDArray,
    weights: ObjectiveWeights = ObjectiveWeights(),
    slsqp_max_iter: int = 100,
    slsqp_constrained: bool = True,
    two_stage_inner: bool = True,
    feasibility_max_iter: int = 80,
    final_feasibility_cleanup: bool = True,
    polish_method: str = "SLSQP",
    feasibility_threshold: float = 0.0,
    threading_oval_tolerance: float = 0.0,
    clearance_overlap_allowance_mm: float = 0.0,
    min_arc_ap_sep_deg: float = 16.0,
    subject_from_rig_rot: NDArray | None = None,
    sdf_clearance_fun=None,
    sdf_clearance_jac=None,
    verbose: bool = False,
) -> PlanCandidate:
    """Run the inner solve from a caller-supplied seed.

    Bypasses the LSAP and arc-partition layers — feeds the given
    ``(probe_to_hole, probe_to_arc_idx, arc_centroids_deg, x0)`` straight
    into the same inner solve ``optimize`` runs for each (hole, arc) pair.

    Used to diagnose where the optimizer falls short of a known plan:
    if the polish from a manual plan stays near the manual's metrics,
    the inner solve is healthy and the gap is in upstream search; if
    the polish drifts away, the gap is the inner solve itself.
    """
    n_arcs = max(probe_to_arc_idx.values()) + 1 if probe_to_arc_idx else 0
    ha = HoleAssignment(probe_to_hole=dict(probe_to_hole), cost=0.0)
    aa = ArcAssignment(
        probe_to_arc_idx=dict(probe_to_arc_idx),
        arc_centroids_deg=arc_centroids_deg,
        cost=0.0,
    )
    ctx = _build_inner_context(
        probes,
        holes,
        ha,
        aa,
        weights,
        threading_oval_tolerance=threading_oval_tolerance,
        clearance_overlap_allowance_mm=clearance_overlap_allowance_mm,
        subject_from_rig_rot=subject_from_rig_rot,
    )
    # The middle layer normally also sets ``min_arc_ap_sep_deg`` inside the
    # ctx; ``_build_inner_context`` uses ``OptimizerContext`` defaults, so
    # override here if the caller passed a non-default value.
    if min_arc_ap_sep_deg != ctx.min_arc_ap_sep_deg:
        ctx = OptimizerContext(
            layout=ctx.layout,
            probes=ctx.probes,
            arc_for_probe=ctx.arc_for_probe,
            weights=ctx.weights,
            shaft_length_mm=ctx.shaft_length_mm,
            shank_radius_mm=ctx.shank_radius_mm,
            headstage_fcl_objs=ctx.headstage_fcl_objs,
            headstage_base_along_shaft_mm=ctx.headstage_base_along_shaft_mm,
            headstage_length_mm=ctx.headstage_length_mm,
            headstage_radius_mm=ctx.headstage_radius_mm,
            min_arc_ap_sep_deg=min_arc_ap_sep_deg,
            min_within_arc_ml_sep_deg=ctx.min_within_arc_ml_sep_deg,
            coverage_n_samples=ctx.coverage_n_samples,
            threading_oval_tolerance=ctx.threading_oval_tolerance,
            clearance_overlap_allowance_mm=ctx.clearance_overlap_allowance_mm,
            subject_from_rig_rot=ctx.subject_from_rig_rot,
        )
    return _inner_solve_one(
        ctx,
        x0,
        ha=ha,
        aa=aa,
        n_arcs=n_arcs,
        slsqp_max_iter=slsqp_max_iter,
        slsqp_constrained=slsqp_constrained,
        two_stage_inner=two_stage_inner,
        feasibility_max_iter=feasibility_max_iter,
        final_feasibility_cleanup=final_feasibility_cleanup,
        polish_method=polish_method,
        feasibility_threshold=feasibility_threshold,
        verbose=verbose,
        sdf_clearance_fun=sdf_clearance_fun,
        sdf_clearance_jac=sdf_clearance_jac,
    )


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
    hole_assignments = solve_top_k_assignments(assignment_probes, holes, k=k_holes)
    if not hole_assignments:
        if verbose:
            print("[optimize] No feasible hole assignment.")
        return None

    candidates: list[PlanCandidate] = []
    for ha_idx, ha in enumerate(hole_assignments):
        if verbose:
            print(
                f"[optimize] hole assignment #{ha_idx} "
                f"(cost={ha.cost:.3f}): {ha.probe_to_hole}"
            )
        # 2. Middle: probe→arc.
        arc_assignments = solve_top_k_arc_assignments(
            ha.probe_to_hole,
            holes,
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
                probes,
                holes,
                ha,
                aa,
                weights,
                threading_oval_tolerance=threading_oval_tolerance,
                clearance_overlap_allowance_mm=clearance_overlap_allowance_mm,
                subject_from_rig_rot=subject_from_rig_rot,
            )
            x0 = _build_initial_x(ctx, holes, ha, aa)
            cand = _inner_solve_one(
                ctx,
                x0,
                ha=ha,
                aa=aa,
                n_arcs=n_arcs,
                slsqp_max_iter=slsqp_max_iter,
                slsqp_constrained=slsqp_constrained,
                two_stage_inner=two_stage_inner,
                feasibility_max_iter=feasibility_max_iter,
                final_feasibility_cleanup=final_feasibility_cleanup,
                polish_method=polish_method,
                feasibility_threshold=feasibility_threshold,
                verbose=verbose,
            )
            candidates.append(cand)

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
