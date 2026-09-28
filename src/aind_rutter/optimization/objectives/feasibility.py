"""What a stage counts as a feasible plan.

A plan is feasible under a spec when every slack in
:mod:`~aind_rutter.optimization.objectives.slacks` is ≥ 0. Phase 2 constrains
those slacks and Phase 1 and the spin restore penalise their shortfall, so a
stage holding a spec tests what the spec states. Jax-free, so settings and the
handoff can import it.
"""

from __future__ import annotations

from dataclasses import dataclass

# How far each category's soft minimum typically reads below the exact minimum of
# the same samples (mm), in category order: body-body, body-shank corners,
# body-shank OBB, shank-shank; fixture body, fixture OBB. Near-tied samples
# approach the ln(k)/beta bound and isolated ones read none, so a single pair can
# land up to its offset short.
PROBE_PAIR_SOFTMIN_BIAS_MM = (0.10, 0.09, 0.0, 0.0)
FIXTURE_PAIR_SOFTMIN_BIAS_MM = (0.05, 0.06)

# The soft minimum the offsets were measured at; any other stales them.
SOFTMIN_BIAS_MEASURED_AT = {
    "softmin_beta": 20.0,
    "top_k_body_body": 16,
    "top_k_body_shank": 8,
    "top_k_shank_shank": 8,
}


@dataclass(frozen=True)
class FeasibilitySpec:
    """The thresholds a plan must meet, and the soft minimum clearance is read with.

    ``softmin_correction`` adds each category's typical soft-min shortfall back
    before comparing with ``min_clearance_mm``, so the margin reads as a distance
    between surfaces. The offsets hold only at the soft minimum they were measured
    at, so a spec with the correction refuses any other.
    """

    min_clearance_mm: float
    softmin_correction: bool
    threading_tolerance: float = 0.0
    # Length of the shaft segment whose crossing of each bore section is tested.
    shaft_length_mm: float = 10.0
    min_arc_ap_sep_deg: float = 16.0
    min_intra_arc_ml_sep_deg: float = 16.0
    brain_margin_mm: float = 0.2
    softmin_beta: float = 20.0
    top_k_body_body: int = 16
    top_k_body_shank: int = 8
    top_k_shank_shank: int = 8

    def __post_init__(self) -> None:
        if not self.softmin_correction:
            return
        stale = {
            name: getattr(self, name)
            for name, measured in SOFTMIN_BIAS_MEASURED_AT.items()
            if getattr(self, name) != measured
        }
        if stale:
            raise ValueError(
                f"the soft-min correction was measured at {SOFTMIN_BIAS_MEASURED_AT}, "
                f"not {stale}; re-measure the offsets or turn the correction off"
            )

    def pair_softmin_kwargs(self) -> dict:
        """Soft-min keywords of the probe-pair clearance queries."""
        return {
            "beta": float(self.softmin_beta),
            "top_k_body_body": int(self.top_k_body_body),
            "top_k_body_shank": int(self.top_k_body_shank),
            "top_k_shank_shank": int(self.top_k_shank_shank),
        }

    def fixture_softmin_kwargs(self) -> dict:
        """Soft-min keywords of the probe-fixture queries, whose body and OBB
        categories read the body-body and body-shank top-k."""
        return {
            "beta": float(self.softmin_beta),
            "top_k_body": int(self.top_k_body_body),
            "top_k_obb": int(self.top_k_body_shank),
        }
