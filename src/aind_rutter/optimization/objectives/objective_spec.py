"""What every stage that scores a plan maximizes.

Phase 2 minimizes :func:`~aind_rutter.optimization.objectives.terms.objective_terms`
under its constraints; Phase 1 minimizes the same function plus penalties on
those constraints' shortfall, so the plan Phase 1 prefers is the one Phase 2
would. Jax-free, so settings can import it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveSpec:
    """Weights and shapes of the objective's terms.

    The defaults are the pipeline's. ``coverage_gain`` is the coverage-vs-margin
    balance; ``cov_alpha`` blends mean and worst region, and only reads when
    per-probe coverage ceilings are supplied.
    """

    coverage_gain: float = 7.0
    cov_alpha: float = 0.2
    softmin_beta_cov: float = 20.0
    # Squared-softplus pull-back outside the comfortable angular range.
    lambda_bounds: float = 1.0
    comfortable_ap_deg: float = 50.0
    comfortable_ml_deg: float = 50.0
    # Holds (cos spin, sin spin) on the unit circle; zero in the angle layout.
    lambda_unit_circle: float = 10.0
    # Saturating reward on clearance margin. Off by default: it duplicates the
    # clearance constraint and its gradient comes from the one closest surface
    # sample, which flips between near-tied samples and makes solves
    # irreproducible.
    lambda_margin_clear: float = 0.0
    tau_clear_mm: float = 0.8
    # The soft minimum over surface samples instead of the hard one.
    smooth_clearance_reward: bool = False
    # Saturating reward on each probe's tightest shank staying off its bore wall;
    # the threading constraint alone lets a shank touch it.
    lambda_margin_thread: float = 1.0
    tau_thread_gunits: float = 0.5
    # Soft minimum over shanks and softplus zero gate, of this sharpness per
    # g-unit; the hard forms are kinked where shanks tie and at zero slack.
    smooth_thread_reward: bool = False
    softmin_beta_thread: float = 20.0
