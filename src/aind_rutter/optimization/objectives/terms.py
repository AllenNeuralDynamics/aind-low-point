"""Terms both phases compute the same way.

:func:`objective_terms` is the objective itself: Phase 2 minimizes it under
constraints and Phase 1 minimizes it plus penalties on their shortfall. The
rest are the pieces both phases read identically: the variable vector, the
intra-arc ML separation and the normalized coverage.

Every function here is traced. Each has a parity test asserting it emits the
same jaxpr as the arithmetic it replaced, because "same answer" is not enough
when the caller is jitted.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from aind_rutter.optimization.clearance.poses import unit_circle_penalty
from aind_rutter.optimization.clearance.smooth import smooth_abs
from aind_rutter.optimization.objectives.coverage import (
    coverage_per_probe_over_probes,
    normalized_coverage_objective,
)
from aind_rutter.optimization.objectives.layout import (
    PHASE1_PER_PROBE_VARS,
    SPIN_COS,
    SPIN_SIN,
)
from aind_rutter.optimization.objectives.objective_spec import ObjectiveSpec
from aind_rutter.optimization.objectives.rewards import (
    saturating_reward_mean,
    saturating_reward_worst,
)
from aind_rutter.optimization.objectives.threading import softplus_squared


def arc_angles_of(x: Array, n_arcs: int) -> Array:
    """The ``(n_arcs,)`` block of arc AP angles that opens ``x``."""
    return x[:n_arcs]


def ml_of(x: Array, n_arcs: int, n_probes: int) -> Array:
    """Each probe's ML angle, ``(n_probes,)``.

    A gather rather than a stride, because the ML entries sit
    ``PHASE1_PER_PROBE_VARS`` apart and the trailing block may be padded.
    """
    return jnp.stack([x[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)])


def spin_components_of(x: Array, n_arcs: int, n_probes: int) -> tuple[Array, Array]:
    """Each probe's ``(cos spin, sin spin)`` pair, as two ``(n_probes,)`` arrays.

    Spin is carried as a point on the unit circle rather than an angle so the
    objective stays smooth across the wrap; :func:`unit_circle_penalty` is what
    holds the pair on the circle.
    """
    return (
        x[n_arcs + SPIN_COS :: PHASE1_PER_PROBE_VARS][:n_probes],
        x[n_arcs + SPIN_SIN :: PHASE1_PER_PROBE_VARS][:n_probes],
    )


def comfort_bound_penalty(
    arc_aps: Array, ml_vals: Array, ap_cap_deg: float, ml_cap_deg: float
) -> Array:
    """Squared-softplus pull-back toward the comfortable angular range.

    Zero inside the caps and growing smoothly outside, so a pose the rig can
    reach but would rather not is discouraged without being forbidden. The
    hard joint limits are a separate constraint.
    """
    penalty = softplus_squared(smooth_abs(arc_aps) - ap_cap_deg)
    return penalty + softplus_squared(smooth_abs(ml_vals) - ml_cap_deg)


def ml_pair_separation(ml_vals: Array) -> Array:
    """``(P, P)`` smooth absolute ML angle difference for every probe pair.

    Only same-arc entries mean anything; the caller masks the rest.
    """
    return smooth_abs(ml_vals[:, None] - ml_vals[None, :])


def normalized_coverage(
    Rs: Array,
    ts: Array,
    tips_local: Array,
    shank_mask: Array,
    coverage_data,
    cov_ceilings,
    *,
    n_samples: int,
    alpha: float,
    softmin_beta: float,
    weights,
) -> Array:
    """Coverage as a fraction of what each region could achieve.

    Dividing by the per-region ceiling is what stops a four-shank probe over a
    large structure from outweighing a one-shank probe over a small one; the
    soft-min floor then protects the worst-covered region.
    """
    cov_pp = coverage_per_probe_over_probes(
        Rs,
        ts,
        tips_local,
        shank_mask,
        coverage_data,
        n_samples=n_samples,
    )
    return normalized_coverage_objective(
        cov_pp,
        cov_ceilings,
        alpha=alpha,
        softmin_beta=softmin_beta,
        weights=weights,
    )


def objective_terms(
    objective: ObjectiveSpec,
    coverage_total: Array,
    arc_aps: Array,
    ml_vals: Array,
    sx: Array,
    sy: Array,
    clearances: tuple[tuple[Array, Array], ...],
    thread_slacks: Array,
    thread_masks: Array,
    coverage_scale=1.0,
) -> Array:
    """The objective every stage minimizes, feasibility aside.

    ``clearances`` holds each clearance query's ``(hard, soft)`` distances with
    categories on the last axis; the reward reads each pair's worst category.
    ``thread_slacks`` and ``thread_masks`` are ``(P, K)``. ``coverage_scale``
    multiplies the coverage gain at run time, so one compiled kernel also
    serves a stage that ignores coverage.
    """
    j_bounds = comfort_bound_penalty(
        arc_aps,
        ml_vals,
        float(objective.comfortable_ap_deg),
        float(objective.comfortable_ml_deg),
    )
    worst_pairs = [
        jnp.min(soft if objective.smooth_clearance_reward else hard, axis=-1).reshape(
            -1
        )
        for hard, soft in clearances
    ]
    reward_clear = (
        saturating_reward_mean(
            jnp.concatenate(worst_pairs), float(objective.tau_clear_mm)
        )
        if worst_pairs
        else jnp.float32(0.0)
    )
    reward_thread = saturating_reward_worst(
        thread_slacks,
        float(objective.tau_thread_gunits),
        thread_masks,
        beta=(
            float(objective.softmin_beta_thread)
            if objective.smooth_thread_reward
            else None
        ),
    )
    j_unit_circle = unit_circle_penalty(sx, sy)
    gain = coverage_scale * float(objective.coverage_gain)
    return (
        -gain * coverage_total
        + float(objective.lambda_bounds) * j_bounds
        + float(objective.lambda_unit_circle) * j_unit_circle
        - float(objective.lambda_margin_clear) * reward_clear
        - float(objective.lambda_margin_thread) * reward_thread
    )
