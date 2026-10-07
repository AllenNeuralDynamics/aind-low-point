"""The feasibility conditions as slacks, one function each; ≥ 0 is feasible.

Phase 2 constrains ``gain · slack ≥ 0``. Phase 1 and the spin restore penalise
``(gain · shortfall(slack))²``, so a pose Phase 1 charges nothing for is one Phase
2 accepts under the same spec. The gains (``clearance.aggregate``) scale the
solver's view of a row and never change its sign.
"""

from __future__ import annotations

import jax.numpy as jnp

from aind_rutter.optimization.objectives.feasibility import (
    FIXTURE_PAIR_SOFTMIN_BIAS_MM,
    PROBE_PAIR_SOFTMIN_BIAS_MM,
    FeasibilitySpec,
)


def shortfall(slack):
    """How far a slack falls short of feasible; zero where it is met."""
    return jnp.maximum(0.0, -slack)


def _clearance_slack(
    soft, bias_mm: tuple[float, ...], margin_mm, spec: FeasibilitySpec
):
    margin = jnp.asarray(margin_mm, jnp.float32)
    if spec.softmin_correction:
        return soft + jnp.asarray(bias_mm, jnp.float32) - margin
    return soft - margin


def pair_clearance_slack(soft, spec: FeasibilitySpec):
    """Probe-pair clearance beyond the margin (mm); ``soft`` holds the four
    categories on its last axis."""
    return _clearance_slack(
        soft, PROBE_PAIR_SOFTMIN_BIAS_MM, spec.pair_margins_mm(), spec
    )


def fixture_clearance_slack(soft, spec: FeasibilitySpec):
    """Probe-fixture clearance beyond the margin (mm); ``soft`` holds the two
    categories on its last axis."""
    return _clearance_slack(
        soft, FIXTURE_PAIR_SOFTMIN_BIAS_MM, spec.fixture_margins_mm(), spec
    )


def threading_slack(g, spec: FeasibilitySpec):
    """Bore fit left before the tolerance, in g-units."""
    return float(spec.threading_tolerance) - g


def brain_slack(d, spec: FeasibilitySpec):
    """How far past the margin a tip sits inside the brain (mm), from the brain
    SDF ``d`` (negative inside)."""
    return -(d + float(spec.brain_margin_mm))


def arc_separation_slack(ap_separation, spec: FeasibilitySpec):
    """Arc AP separation beyond the minimum (deg)."""
    return ap_separation - float(spec.min_arc_ap_sep_deg)


def ml_separation_slack(ml_separation, spec: FeasibilitySpec):
    """Same-arc ML separation beyond the minimum (deg)."""
    return ml_separation - float(spec.min_intra_arc_ml_sep_deg)


def coverage_floor_slack(per_probe, threaded_ceilings, spec: FeasibilitySpec):
    """Each probe's coverage past the floor, as a fraction of its threaded best."""
    return per_probe / threaded_ceilings - float(spec.min_coverage_fraction)
