"""One clearance number per pair, from the per-representation terms.

A pair is scored under every representation that applies to it, each scaled
by its category gain, and the worst governs. In a KKT solver the Lagrange
multiplier tracks the constraint magnitude, so a thickness-bounded OBB term
(~0.024 mm) would be under-prioritised beside a millimetre-scale voxel term
without the gain; a shank crossing is physically the more severe of the two.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

from aind_rutter.optimization.clearance.pairs import (
    pairwise_signed_clearance_probe_fixture_body_world,
    pairwise_signed_clearance_probe_obb_fixture_world,
)

# Per-category gains on the clearance slacks, applied to the signed slack (a
# constraint) or to its shortfall (a penalty) before any squaring. A solver's
# multiplier tracks its constraint's magnitude, so a thickness-bounded OBB term
# (about 0.024 mm) would be under-weighted beside a millimetre-scale voxel term
# without one; a shank crossing is also the more severe failure. Voxel-SDF
# categories are mm-native and take 1. The gains scale a row and never flip it,
# so they do not change which plans are feasible.
SLACK_GAIN_BODY_BODY = 1.0
SLACK_GAIN_BODY_SHANK_CORNERS = 1.0
SLACK_GAIN_FIXTURE_BODY = 1.0
OBB_SLACK_GAIN = 100.0


def probe_pair_slack_gains(
    obb_gain: float = OBB_SLACK_GAIN,
) -> tuple[float, float, float, float]:
    """Per-category probe-pair gains, in :class:`PairClearance` order."""
    return (SLACK_GAIN_BODY_BODY, SLACK_GAIN_BODY_SHANK_CORNERS, obb_gain, obb_gain)


def fixture_slack_gains(obb_gain: float = OBB_SLACK_GAIN) -> tuple[float, float]:
    """Per-category probe-fixture gains, in :class:`FixtureClearance` order."""
    return (SLACK_GAIN_FIXTURE_BODY, obb_gain)


# ---------------------------------------------------------------------------
# Dual-rep aggregates over the per-representation queries
# ---------------------------------------------------------------------------


class PairClearance(NamedTuple):
    """The probe-pair clearance categories in slack order, as
    :func:`.sweep.pair_clearance` fills them.

    Each field is ``(hard_min, soft_min_topk)`` of that category's
    signed-distance pool. Positive ⇒ clear by that distance; negative
    ⇒ penetration. Soft is the smooth surrogate used in objective /
    constraint slack form; hard is the exact min used for the
    saturating-margin reward.
    """

    body_body: tuple[Array, Array]  # voxel-SDF, mm-native
    body_shank_corners: tuple[Array, Array]  # dense shank-box samples → voxel SDF
    body_shank_obb: tuple[Array, Array]  # unmeasured; the empty sentinel
    shank_shank: tuple[Array, Array]  # OBB-OBB exact SAT


class FixtureClearance(NamedTuple):
    """Dual-rep clearance between one probe and one fixture.

    Each field is ``(hard_min, soft_min_topk)``.
    """

    body: tuple[Array, Array]  # body voxel-SDF vs fixture surface samples
    obb: tuple[Array, Array]  # probe OBBs SDF vs fixture surface samples


def dual_rep_fixture_clearance(
    R_p: Array,
    t_p: Array,
    sdf_p_grid: Array,
    sdf_p_origin: Array,
    sdf_p_spacing: Array,
    fx_grid: Array,
    fx_origin: Array,
    fx_spacing: Array,
    world_surface_p: Array,
    fx_surface: Array,
    shank_centers: Array,
    shank_halves: Array,
    *,
    beta: float,
    top_k_body: int,
    top_k_obb: int,
    n_real_p: Array | None = None,
    n_real_f: Array | None = None,
    shank_mask: Array | None = None,
) -> FixtureClearance:
    """Both probe-fixture clearance categories in one call.

    Categories: probe body voxel-SDF vs fixture surface samples; probe
    OBB analytic SDF vs fixture surface samples. The OBB direction
    catches probe shank / transition-zone contact with fixture
    surfaces (well-bore-rim grazing) that the body voxel direction
    misses at the α-wrap closing cap.

    ``n_real_p`` / ``shank_mask`` carry the probe grid's real extent and the OBB
    validity mask when the probe grid/OBB come from a padded per-kind table.
    ``n_real_f`` carries the fixture grid's real extent when the fixtures are
    stacked into a padded table to share a vmap axis (see clearance_sweep). All
    ``None`` ⇒ unchanged behaviour.
    """
    body = pairwise_signed_clearance_probe_fixture_body_world(
        R_p,
        t_p,
        sdf_p_grid,
        sdf_p_origin,
        sdf_p_spacing,
        fx_grid,
        fx_origin,
        fx_spacing,
        world_surface_p,
        fx_surface,
        beta=beta,
        top_k=top_k_body,
        n_real_p=n_real_p,
        n_real_f=n_real_f,
    )
    obb = pairwise_signed_clearance_probe_obb_fixture_world(
        R_p,
        t_p,
        fx_surface,
        shank_centers,
        shank_halves,
        beta=beta,
        shank_mask=shank_mask,
        top_k=top_k_obb,
    )
    return FixtureClearance(body=body, obb=obb)
