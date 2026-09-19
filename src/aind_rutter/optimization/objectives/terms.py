"""Terms both phases compute the same way.

Phase 1 minimizes these as soft penalties and Phase 2 carries most of them as
hard constraints, but a handful are identical in both: reading the variable
vector, the comfort-range pull-back, the intra-arc ML separation and the
normalized coverage. Each was written out twice, so a change to one reached
only one phase.

Every function here is traced. Each has a parity test asserting it emits the
same jaxpr as the arithmetic it replaced, because "same answer" is not enough
when the caller is jitted.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

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
