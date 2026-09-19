"""One clearance number per pair, from the per-representation terms.

A pair is scored under every representation that applies to it, each scaled
by its category gain, and the worst governs. In a KKT solver the Lagrange
multiplier tracks the constraint magnitude, so a thickness-bounded OBB term
(~0.024 mm) would be under-prioritised beside a millimetre-scale voxel term
without the gain; a shank crossing is physically the more severe of the two.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from aind_rutter.optimization.clearance.boxes import (
    obb_obb_signed_distance,
    obb_sdf_world_to_local,
    shank_world_samples,
)
from aind_rutter.optimization.clearance.grids import trilinear_sdf
from aind_rutter.optimization.clearance.pairs import (
    body_body_pair_clearance,
    body_shank_corners_pair_clearance,
    hard_soft,
    pairwise_signed_clearance_probe_fixture_body_world,
    pairwise_signed_clearance_probe_obb_fixture_world,
    shank_only_pair_clearance,
)

# Per-category gains on clearance violation magnitudes. After the
# 2026-05-24 cleanup, every category that uses analytic OBB SDF /
# OBB-OBB SAT shares one bounded-magnitude scaling. Voxel-SDF
# categories stay at 1.0 because their magnitudes are already mm-scale.
#
# Categories in use:
#   probe-probe
#     - body-body voxel-SDF top-k        (mm-native)
#     - body-shank OBB direction (SAT)   (thickness-bounded ≈ 0.024 mm)
#     - shank-shank OBB SAT              (thickness-bounded)
#   probe-fixture
#     - body voxel-SDF vs fixture        (mm-native)
#     - probe OBB vs fixture surface     (thickness-bounded)
#
# Why the gain: in a KKT solver Lagrange multipliers track constraint
# magnitudes; a small-magnitude constraint gets a small multiplier and
# the optimiser under-prioritises it. Shank crossings / OBB violations
# are physically more severe than equally-deep body-body grazing
# (probes can't be inserted) so we boost their KKT weight to keep them
# competitive with body-body in the merit function.
#
# Apply the gain to the violation magnitude (penalty form) or directly
# to the signed slack (constraint form), before any squaring — so
# first-order gradient contributions scale linearly with the gain.
SLACK_GAIN_BODY_BODY = 1.0


SLACK_GAIN_BODY_SHANK_CORNERS = 1.0  # voxel-SDF lookup → mm-native magnitude


SLACK_GAIN_BODY_SHANK_OBB = 100.0


SLACK_GAIN_SHANK_SHANK = 100.0


SLACK_GAIN_FIXTURE_BODY = 1.0


SLACK_GAIN_FIXTURE_OBB = 100.0


# ---------------------------------------------------------------------------
# Dual-rep aggregator helpers (reduced / Phase 1 / Phase 2 / validation)
# ---------------------------------------------------------------------------


class PairClearance(NamedTuple):
    """Dual-rep clearance between two probes, all four categories.

    Each field is ``(hard_min, soft_min_topk)`` of that category's
    signed-distance pool. Positive ⇒ clear by that distance; negative
    ⇒ penetration. Soft is the smooth surrogate used in objective /
    constraint slack form; hard is the exact min used for the
    saturating-margin reward.
    """

    body_body: tuple[Array, Array]  # voxel-SDF, mm-native
    body_shank_corners: tuple[Array, Array]  # shank OBB corners → voxel SDF
    body_shank_obb: tuple[Array, Array]  # body samples → shank OBB SDF
    shank_shank: tuple[Array, Array]  # OBB-OBB exact SAT


class FixtureClearance(NamedTuple):
    """Dual-rep clearance between one probe and one fixture.

    Each field is ``(hard_min, soft_min_topk)``.
    """

    body: tuple[Array, Array]  # body voxel-SDF vs fixture surface samples
    obb: tuple[Array, Array]  # probe OBBs SDF vs fixture surface samples


# Per-category gains, ordered to match the NamedTuple fields. Sites
# that scale slacks by per-category importance (reduced penalty,
# Phase 1 penalty, Phase 2 constraint, etc.) can ``zip`` over
# ``pc.softs`` and the appropriate gain tuple without re-inlining the
# constant list.
PROBE_PAIR_SLACK_GAINS = (
    SLACK_GAIN_BODY_BODY,
    SLACK_GAIN_BODY_SHANK_CORNERS,
    SLACK_GAIN_BODY_SHANK_OBB,
    SLACK_GAIN_SHANK_SHANK,
)


FIXTURE_PAIR_SLACK_GAINS = (
    SLACK_GAIN_FIXTURE_BODY,
    SLACK_GAIN_FIXTURE_OBB,
)


def dual_rep_pair_clearance(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    sdf_a_grid: Array,
    sdf_a_origin: Array,
    sdf_a_spacing: Array,
    sdf_b_grid: Array,
    sdf_b_origin: Array,
    sdf_b_spacing: Array,
    world_surface_a: Array,
    world_surface_b: Array,
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k_body_body: int = 16,
    top_k_body_shank: int = 8,
    top_k_shank_shank: int = 8,
) -> PairClearance:
    """All four dual-rep probe-pair clearance categories in one call.

    Equivalent to invoking :func:`body_body_pair_clearance`,
    :func:`body_shank_corners_pair_clearance`, and
    :func:`shank_only_pair_clearance` separately and packing the
    results. JAX-traceable, no behavioural change vs the inline
    pattern shared across reduced / Phase 1 / Phase 2 /
    Phase 3 — purely a packaging convenience that gives every site one
    source of truth for "what counts as dual-rep probe-pair clearance".
    """
    body_body = body_body_pair_clearance(
        R_a,
        t_a,
        R_b,
        t_b,
        sdf_a_grid,
        sdf_a_origin,
        sdf_a_spacing,
        sdf_b_grid,
        sdf_b_origin,
        sdf_b_spacing,
        world_surface_a,
        world_surface_b,
        beta=beta,
        top_k=top_k_body_body,
    )
    body_shank_corners = body_shank_corners_pair_clearance(
        R_a,
        t_a,
        R_b,
        t_b,
        sdf_a_grid,
        sdf_a_origin,
        sdf_a_spacing,
        sdf_b_grid,
        sdf_b_origin,
        sdf_b_spacing,
        shank_centers_a,
        shank_halves_a,
        shank_centers_b,
        shank_halves_b,
        beta=beta,
        top_k=top_k_body_shank,
    )
    body_shank_obb, shank_shank = shank_only_pair_clearance(
        R_a,
        t_a,
        R_b,
        t_b,
        world_surface_a,
        world_surface_b,
        shank_centers_a,
        shank_halves_a,
        shank_centers_b,
        shank_halves_b,
        beta=beta,
        top_k_body_shank=top_k_body_shank,
        top_k_shank_shank=top_k_shank_shank,
    )
    return PairClearance(
        body_body=body_body,
        body_shank_corners=body_shank_corners,
        body_shank_obb=body_shank_obb,
        shank_shank=shank_shank,
    )


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
    beta: float = 20.0,
    top_k_body: int = 16,
    top_k_obb: int = 8,
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


def pairwise_signed_clearance_dual(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    sdf_a_grid: Array,
    sdf_a_origin: Array,
    sdf_a_spacing: Array,
    sdf_b_grid: Array,
    sdf_b_origin: Array,
    sdf_b_spacing: Array,
    surface_a: Array,  # (Nbody, 3) body envelope samples (a-local)
    surface_b: Array,  # (Nbody, 3) body envelope samples (b-local)
    shank_centers_a: Array,  # (Sa, 3) shank centres in a-local
    shank_halves_a: Array,  # (Sa, 3) shank half-extents
    shank_centers_b: Array,  # (Sb, 3)
    shank_halves_b: Array,  # (Sb, 3)
    *,
    beta: float = 20.0,
    top_k_body_body: int = 16,
    top_k_body_shank: int = 8,
    top_k_shank_shank: int = 8,
) -> tuple[
    tuple[Array, Array],
    tuple[Array, Array],
    tuple[Array, Array],
]:
    """Dual-rep pair clearance: body voxel SDF + analytic shank OBBs.

    Returns three ``(hard_min, soft_min)`` tuples, one per category:

      - ``body_body``  : envelope-sample SDF lookups (both directions)
      - ``body_shank`` : shank corners vs other body's SDF (both ways)
      - ``shank_shank``: shank corners vs other-probe shank OBBs

    The caller should apply ReLU-squared penalties to each soft min and
    sum, so the optimizer gets independent gradient signal per category
    rather than letting sample-count imbalance bias the gradient toward
    body samples in a pooled softmin.

    Hard mins are exposed for diagnostics and feasibility checks (use
    these in lex keys and hard constraints).

    The body SDF is looked up trilinearly. Soft-min top-k aggregation
    across 16 samples absorbs trilinear's C⁰ voxel-edge gradient
    discontinuities, so a C¹ tricubic lookup bought no convergence at
    ~5× the cost per pair. OBB SDFs are closed-form regardless.

    β=20/mm default → 50 µm smoothing window. Per-category top_k caps
    the bias at ``log(top_k)/β`` (~0.14 mm body-body, ~0.10 mm shank).
    """
    # Hoist the local→world surface transforms to ONE per probe by
    # computing them and delegating to the world-frame variant. Callers
    # in tight loops over many pairs should call that variant directly
    # to skip the K(K-1) per-pair redundancy XLA's CSE pass leaves in
    # (verified 2026-05-23 via HLO dump).
    world_surface_a = surface_a @ R_a.T + t_a
    world_surface_b = surface_b @ R_b.T + t_b
    return pairwise_signed_clearance_dual_world(
        R_a,
        t_a,
        R_b,
        t_b,
        sdf_a_grid,
        sdf_a_origin,
        sdf_a_spacing,
        sdf_b_grid,
        sdf_b_origin,
        sdf_b_spacing,
        world_surface_a,
        world_surface_b,
        shank_centers_a,
        shank_halves_a,
        shank_centers_b,
        shank_halves_b,
        beta=beta,
        top_k_body_body=top_k_body_body,
        top_k_body_shank=top_k_body_shank,
        top_k_shank_shank=top_k_shank_shank,
    )


def pairwise_signed_clearance_dual_world(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    sdf_a_grid: Array,
    sdf_a_origin: Array,
    sdf_a_spacing: Array,
    sdf_b_grid: Array,
    sdf_b_origin: Array,
    sdf_b_spacing: Array,
    world_surface_a: Array,  # (Nbody, 3) body envelope samples in WORLD
    world_surface_b: Array,  # (Nbody, 3) body envelope samples in WORLD
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k_body_body: int = 16,
    top_k_body_shank: int = 8,
    top_k_shank_shank: int = 8,
) -> tuple[
    tuple[Array, Array],
    tuple[Array, Array],
    tuple[Array, Array],
]:
    """Same as :func:`pairwise_signed_clearance_dual` but takes
    pre-transformed world-frame body surface samples. Caller is
    responsible for computing ``world_surface[i] = surface[i] @ R[i].T
    + t[i]`` once per probe outside the pair loop. Avoids O(K(K-1))
    redundant transforms across a probe set with K probes' pairs
    (CSE doesn't catch the duplication — HLO inspection 2026-05-23).
    """
    # 1+2: body-body (per-sample, not min-reduced).
    local_in_a = (world_surface_b - t_a) @ R_a
    d_body_b_in_a = trilinear_sdf(sdf_a_grid, sdf_a_origin, sdf_a_spacing, local_in_a)
    local_in_b = (world_surface_a - t_b) @ R_b
    d_body_a_in_b = trilinear_sdf(sdf_b_grid, sdf_b_origin, sdf_b_spacing, local_in_b)
    body_body_pool = jnp.concatenate(
        [d_body_b_in_a.reshape(-1), d_body_a_in_b.reshape(-1)], axis=0
    )

    Sa = shank_centers_a.shape[0]
    Sb = shank_centers_b.shape[0]
    corners_a_world = (
        shank_world_samples(R_a, t_a, shank_centers_a, shank_halves_a)
        if Sa > 0
        else jnp.zeros((0, 3), dtype=world_surface_a.dtype)
    )
    corners_b_world = (
        shank_world_samples(R_b, t_b, shank_centers_b, shank_halves_b)
        if Sb > 0
        else jnp.zeros((0, 3), dtype=world_surface_b.dtype)
    )

    # 3+4: body-shank — both directions (see pairwise_signed_clearance_dual).
    body_shank_chunks = []
    if Sa > 0:
        ca_in_b = (corners_a_world - t_b) @ R_b
        d_corners_a_in_b = trilinear_sdf(
            sdf_b_grid, sdf_b_origin, sdf_b_spacing, ca_in_b
        )
        body_shank_chunks.append(d_corners_a_in_b.reshape(-1))
        d_body_b_vs_a_obbs = jax.vmap(
            lambda c, h: obb_sdf_world_to_local(world_surface_b, R_a, t_a, c, h)
        )(shank_centers_a, shank_halves_a)
        body_shank_chunks.append(d_body_b_vs_a_obbs.reshape(-1))
    if Sb > 0:
        cb_in_a = (corners_b_world - t_a) @ R_a
        d_corners_b_in_a = trilinear_sdf(
            sdf_a_grid, sdf_a_origin, sdf_a_spacing, cb_in_a
        )
        body_shank_chunks.append(d_corners_b_in_a.reshape(-1))
        d_body_a_vs_b_obbs = jax.vmap(
            lambda c, h: obb_sdf_world_to_local(world_surface_a, R_b, t_b, c, h)
        )(shank_centers_b, shank_halves_b)
        body_shank_chunks.append(d_body_a_vs_b_obbs.reshape(-1))
    body_shank_pool = (
        jnp.concatenate(body_shank_chunks, axis=0)
        if body_shank_chunks
        else jnp.zeros((0,), dtype=world_surface_a.dtype)
    )

    # 5+6: shank-shank via exact OBB-OBB SAT (no sampling). Returns one
    # signed-distance scalar per (Sa, Sb) OBB pair — closed-form. Catches
    # interior crossings that surface-sample-based queries miss for thin
    # OBBs (the 24/70 µm silicon shanks).
    shank_shank_chunks = []
    if Sa > 0 and Sb > 0:
        # vmap outer over A's OBBs, inner over B's OBBs.
        def _pair_distance(ca, ha, cb, hb):
            return obb_obb_signed_distance(R_a, t_a, ca, ha, R_b, t_b, cb, hb)

        d_sa_vs_sb = jax.vmap(
            lambda ca, ha: jax.vmap(lambda cb, hb: _pair_distance(ca, ha, cb, hb))(
                shank_centers_b, shank_halves_b
            )
        )(shank_centers_a, shank_halves_a)  # (Sa, Sb)
        shank_shank_chunks.append(d_sa_vs_sb.reshape(-1))
    shank_shank_pool = (
        jnp.concatenate(shank_shank_chunks, axis=0)
        if shank_shank_chunks
        else jnp.zeros((0,), dtype=world_surface_a.dtype)
    )

    return (
        hard_soft(body_body_pool, beta=beta, top_k=top_k_body_body),
        hard_soft(body_shank_pool, beta=beta, top_k=top_k_body_shank),
        hard_soft(shank_shank_pool, beta=beta, top_k=top_k_shank_shank),
    )
