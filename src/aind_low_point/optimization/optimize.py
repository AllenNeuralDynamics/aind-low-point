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
    evaluate_objective,
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
class OptimizationResult:
    """Final optimizer output.

    ``probe_to_hole`` and ``probe_to_arc_idx`` give the discrete
    decisions; ``arc_centroids_deg`` gives the AP cluster centroids
    (post-optimization, the actual ``ap_arc`` values are read from
    ``x[:n_arcs]``). ``cost`` is the inner-loop scalar; ``breakdown``
    has the component-wise terms for diagnostics.
    """

    probe_to_hole: dict[str, int]
    probe_to_arc_idx: dict[str, int]
    arc_centroids_deg: tuple[float, ...]
    n_arcs: int
    x: NDArray[np.floating]
    cost: float
    breakdown: ObjectiveBreakdown


# ---------------------------------------------------------------------------
# Inner-loop helpers
# ---------------------------------------------------------------------------


def _build_inner_context(
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    hole_assignment: HoleAssignment,
    arc_assignment: ArcAssignment,
    weights: ObjectiveWeights,
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
    slower convergence near tight feasibility boundaries. Future
    variant: pass kinematic / threading violations as proper SLSQP
    inequality constraints once we have JAX-autodiff Jacobians wired
    in.

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
    slsqp_max_iter: int = 100,
    min_arc_ap_sep_deg: float = 16.0,
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

    best: OptimizationResult | None = None
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

            ctx = _build_inner_context(probes, holes, ha, aa, weights)
            x0 = _build_initial_x(ctx, holes, ha, aa)

            x = x0
            if use_cma:
                x_cma = _try_cma_es(
                    ctx, x,
                    population=cma_population,
                    generations=cma_generations,
                    sigma=cma_sigma,
                )
                if x_cma is not None:
                    x = x_cma
            x_opt, breakdown_opt = _slsqp_polish(
                ctx, x, max_iter=slsqp_max_iter
            )

            if verbose:
                print(
                    f"    inner cost={breakdown_opt.total:.3f}  "
                    f"(coverage={breakdown_opt.coverage_total:.3f}, "
                    f"thread={breakdown_opt.threading_penalty:.3f}, "
                    f"clear={breakdown_opt.clearance_penalty:.3f}, "
                    f"kin={breakdown_opt.kinematic_penalty:.3f})"
                )

            if best is None or breakdown_opt.total < best.cost:
                best = OptimizationResult(
                    probe_to_hole=dict(ha.probe_to_hole),
                    probe_to_arc_idx=dict(aa.probe_to_arc_idx),
                    arc_centroids_deg=aa.arc_centroids_deg,
                    n_arcs=n_arcs,
                    x=x_opt,
                    cost=float(breakdown_opt.total),
                    breakdown=breakdown_opt,
                )

    if verbose and best is not None:
        print(f"[optimize] best total cost: {best.cost:.3f}")
    return best
