"""Signed clearance for one pair of colliders, per representation.

Each function answers the same question for a different pair of shapes: at
these two world poses, how far apart are they, negative when overlapping and
smooth through the overlap, where FCL's BVH distance clamps at zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from aind_rutter.optimization.clearance.boxes import (
    EMPTY_CLEARANCE_SENTINEL,
    SHANK_SAMPLES_PER_BOX,
    boxes_sdf,
    obb_obb_signed_distance,
    obb_sdf_world_to_local,
    shank_world_samples,
)
from aind_rutter.optimization.clearance.grids import (
    trilinear_sdf,
    trilinear_sdf_stacked,
)
from aind_rutter.optimization.clearance.smooth import soft_min_topk


def pairwise_signed_clearance_probe_obb_fixture_world(
    R_p: Array,
    t_p: Array,
    fixture_surface_world: Array,  # (Nf, 3) fixture envelope samples in world LPS
    shank_centers: Array,  # (S, 3) probe OBB centers in probe local
    shank_halves: Array,  # (S, 3) probe OBB half-extents
    *,
    beta: float = 20.0,
    top_k: int = 8,
    shank_mask: Array | None = None,
) -> tuple[Array, Array]:
    """Probe-OBB vs fixture clearance via fixture surface samples →
    probe OBB analytic SDF.

    Symmetric counterpart to the body-shank-OBB direction in
    :func:`shank_only_pair_clearance`: dense surface samples on one
    side + analytic OBB SDF on the other. Catches probe shank /
    transition-zone OBB vs fixture geometry (e.g. probe body-bottom
    grazing the well bore rim — invisible to the body-vs-fixture-body
    SDF check because the body α-wrap closing cap under-inflates at
    the shank-strip boundary).

    ``shank_mask`` (``(S,)`` bool/0-1, optional) marks which OBB rows are real vs
    padding when the per-kind OBB table is padded to a uniform ``max_S``; invalid
    rows are set to the no-collision sentinel so they never win the min/soft-min.
    ``None`` ⇒ no padding (legacy), identical to before.

    Returns ``(hard_min, soft_min)`` over the pool of all
    (fixture_point, probe_OBB) signed distances. Negative ⇒ at least
    one fixture surface point lies inside the probe OBB.
    """
    S = shank_centers.shape[0]
    if S == 0:
        return (
            jnp.float32(1e3),
            jnp.float32(1e3),
        )
    # For each probe OBB, signed distance of each fixture surface point
    # to that OBB. Shape: (S, Nf).
    d_obbs = jax.vmap(
        lambda c, h: obb_sdf_world_to_local(fixture_surface_world, R_p, t_p, c, h)
    )(shank_centers, shank_halves)
    if shank_mask is not None:
        d_obbs = jnp.where(
            shank_mask.astype(bool)[:, None], d_obbs, EMPTY_CLEARANCE_SENTINEL
        )
    pool = d_obbs.reshape(-1)
    return hard_soft(pool, beta=beta, top_k=top_k)


def pairwise_signed_clearance_probe_fixture_body_world(
    R_p: Array,
    t_p: Array,
    sdf_p_grid: Array,
    sdf_p_origin: Array,
    sdf_p_spacing: Array,
    sdf_f_grid: Array,
    sdf_f_origin: Array,
    sdf_f_spacing: Array,
    world_surface_p: Array,  # (Np, 3) pre-transformed probe envelope samples in world
    surface_f: Array,  # (Nf, 3) fixture envelope samples in world LPS
    *,
    beta: float = 20.0,
    top_k: int = 16,
    n_real_p: Array | None = None,
    n_real_f: Array | None = None,
) -> tuple[Array, Array]:
    """Same as :func:`pairwise_signed_clearance_probe_fixture_body` but
    takes pre-transformed world-frame probe surface samples. Callers
    iterating over ``n_fixtures × n_probes`` pairs should hoist
    ``world_surface[i] = surface[i] @ R[i].T + t[i]`` once per probe
    (per Phase 1's body-body hoist).

    ``n_real_p`` is the probe grid's real (unpadded) extent, for the fixture-
    surface-vs-probe-grid query when ``sdf_p_grid`` is a padded slot of a
    per-kind table (see clearance_sweep). ``n_real_f`` is the same for the
    fixture grid, for the probe-surface-vs-fixture-grid query when the fixtures
    are stacked into a padded table to share a vmap axis. Both ``None`` ⇒
    unchanged (un-padded constant grid).
    """
    kw_p = {} if n_real_p is None else {"n_real": n_real_p}
    kw_f = {} if n_real_f is None else {"n_real": n_real_f}

    d_p_in_f = trilinear_sdf(
        sdf_f_grid,
        sdf_f_origin,
        sdf_f_spacing,
        world_surface_p,
        **kw_f,
    )

    local_f_in_p = (surface_f - t_p) @ R_p
    d_f_in_p = trilinear_sdf(
        sdf_p_grid, sdf_p_origin, sdf_p_spacing, local_f_in_p, **kw_p
    )

    distances = jnp.concatenate([d_p_in_f.reshape(-1), d_f_in_p.reshape(-1)])
    hard_min = jnp.min(distances)
    soft = soft_min_topk(distances, beta=beta, top_k=top_k)
    return hard_min, soft


def hard_soft(values: Array, *, beta: float, top_k: int) -> tuple[Array, Array]:
    """Return ``(hard_min, soft_min_topk)`` for a (possibly empty) 1-D
    sample vector. Empty pools return ``(sentinel, sentinel)`` so the
    caller's ReLU penalty is silent.
    """
    if values.size == 0:
        s = jnp.asarray(EMPTY_CLEARANCE_SENTINEL, dtype=jnp.float32)
        return s, s
    return jnp.min(values), soft_min_topk(values, beta=beta, top_k=top_k)


def shank_only_pair_clearance(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    world_surface_a: Array,
    world_surface_b: Array,
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k_body_shank: int = 8,
    top_k_shank_shank: int = 8,
    shank_mask_a: Array | None = None,
    shank_mask_b: Array | None = None,
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """Shank-related dual-rep categories that DON'T use trilinear SDF.

    Computes:
    - **body-shank, OBB direction only**: ``body_samples @ each-OBB-SDF``
      (analytic). The trilinear "shank corners → other body's SDF"
      direction is omitted — caller computes that per-pair if needed.
    - **shank-shank SAT**: exact OBB-vs-OBB distance via 15 candidate
      axes. No SDF, no gather, fully analytic.

    Returns ``((hbs_obb_only, sbs_obb_only), (hss, sss))``. Both pools
    are exact analytic distances (no soft-min bias from gather discont-
    inuities).

    Vmap-friendly: assumes uniform shank shapes ``(S, 3)`` across pairs
    (post-2026-05-23 OBB-union, S=2 for every probe). The internal
    inner vmaps over (Sa, Sb) compose with an outer vmap over pairs to
    give a single XLA launch.
    """
    Sa = shank_centers_a.shape[0]
    Sb = shank_centers_b.shape[0]

    body_shank_chunks = []
    if Sa > 0:
        d_body_b_vs_a_obbs = jax.vmap(
            lambda c, h: obb_sdf_world_to_local(world_surface_b, R_a, t_a, c, h)
        )(shank_centers_a, shank_halves_a)  # (Sa, Nbody)
        if shank_mask_a is not None:
            d_body_b_vs_a_obbs = jnp.where(
                shank_mask_a.astype(bool)[:, None],
                d_body_b_vs_a_obbs,
                EMPTY_CLEARANCE_SENTINEL,
            )
        body_shank_chunks.append(d_body_b_vs_a_obbs.reshape(-1))
    if Sb > 0:
        d_body_a_vs_b_obbs = jax.vmap(
            lambda c, h: obb_sdf_world_to_local(world_surface_a, R_b, t_b, c, h)
        )(shank_centers_b, shank_halves_b)
        if shank_mask_b is not None:
            d_body_a_vs_b_obbs = jnp.where(
                shank_mask_b.astype(bool)[:, None],
                d_body_a_vs_b_obbs,
                EMPTY_CLEARANCE_SENTINEL,
            )
        body_shank_chunks.append(d_body_a_vs_b_obbs.reshape(-1))
    body_shank_pool = (
        jnp.concatenate(body_shank_chunks, axis=0)
        if body_shank_chunks
        else jnp.zeros((0,), dtype=world_surface_a.dtype)
    )

    return (
        hard_soft(body_shank_pool, beta=beta, top_k=top_k_body_shank),
        shank_shank_pair_clearance(
            R_a,
            t_a,
            R_b,
            t_b,
            shank_centers_a,
            shank_halves_a,
            shank_centers_b,
            shank_halves_b,
            beta=beta,
            top_k=top_k_shank_shank,
            shank_mask_a=shank_mask_a,
            shank_mask_b=shank_mask_b,
            dtype=world_surface_a.dtype,
        ),
    )


def shank_shank_pair_clearance(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k: int = 8,
    shank_mask_a: Array | None = None,
    shank_mask_b: Array | None = None,
    dtype=jnp.float32,
) -> tuple[Array, Array]:
    """Shank-shank category: exact OBB-vs-OBB distance over 15 separating axes for
    every shank pair, masked rows reading the no-collision sentinel."""
    Sa, Sb = shank_centers_a.shape[0], shank_centers_b.shape[0]
    if Sa == 0 or Sb == 0:
        return hard_soft(jnp.zeros((0,), dtype=dtype), beta=beta, top_k=top_k)

    d_sa_vs_sb = jax.vmap(
        lambda ca, ha: jax.vmap(
            lambda cb, hb: obb_obb_signed_distance(R_a, t_a, ca, ha, R_b, t_b, cb, hb)
        )(shank_centers_b, shank_halves_b)
    )(shank_centers_a, shank_halves_a)  # (Sa, Sb)
    if shank_mask_a is not None or shank_mask_b is not None:
        ma = (
            shank_mask_a.astype(bool)
            if shank_mask_a is not None
            else jnp.ones((Sa,), bool)
        )
        mb = (
            shank_mask_b.astype(bool)
            if shank_mask_b is not None
            else jnp.ones((Sb,), bool)
        )
        d_sa_vs_sb = jnp.where(
            ma[:, None] & mb[None, :], d_sa_vs_sb, EMPTY_CLEARANCE_SENTINEL
        )
    return hard_soft(d_sa_vs_sb.reshape(-1), beta=beta, top_k=top_k)


def body_shank_corners_pair_clearance(
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
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k: int = 8,
    shank_mask_a: Array | None = None,
    shank_mask_b: Array | None = None,
    n_real_a: Array | None = None,
    n_real_b: Array | None = None,
) -> tuple[Array, Array]:
    """Body-shank "shank corners → other body's SDF" direction ONLY.

    Trilinear-gather path; on CPU its gradient suffers the scatter-
    contention slowdown when vmap'd across pairs (see
    [[vmap-cpu-gpu-polish-arch]]). Kept as its own helper so callers
    can decide whether to per-pair-loop it (CPU) or vmap it (GPU).

    ``shank_mask_{a,b}`` (``(S,)`` bool/0-1, optional) mark which OBB rows
    are real vs padding when the per-kind OBB tables are padded to a uniform
    ``max_S``: each invalid OBB's pooled distances are set to the
    "no-collision" sentinel so they never win the min/soft-min. ``None`` ⇒
    no padding (legacy ragged tables), identical to before.
    """
    kw_a = {} if n_real_a is None else {"n_real": n_real_a}
    kw_b = {} if n_real_b is None else {"n_real": n_real_b}
    Sa = shank_centers_a.shape[0]
    Sb = shank_centers_b.shape[0]
    chunks = []
    if Sa > 0:
        corners_a_world = shank_world_samples(R_a, t_a, shank_centers_a, shank_halves_a)
        ca_in_b = (corners_a_world - t_b) @ R_b
        d_corners_a_in_b = trilinear_sdf(
            sdf_b_grid, sdf_b_origin, sdf_b_spacing, ca_in_b, **kw_b
        ).reshape(-1)
        if shank_mask_a is not None:
            m = jnp.repeat(shank_mask_a.astype(bool), SHANK_SAMPLES_PER_BOX)
            d_corners_a_in_b = jnp.where(m, d_corners_a_in_b, EMPTY_CLEARANCE_SENTINEL)
        chunks.append(d_corners_a_in_b)
    if Sb > 0:
        corners_b_world = shank_world_samples(R_b, t_b, shank_centers_b, shank_halves_b)
        cb_in_a = (corners_b_world - t_a) @ R_a
        d_corners_b_in_a = trilinear_sdf(
            sdf_a_grid, sdf_a_origin, sdf_a_spacing, cb_in_a, **kw_a
        ).reshape(-1)
        if shank_mask_b is not None:
            m = jnp.repeat(shank_mask_b.astype(bool), SHANK_SAMPLES_PER_BOX)
            d_corners_b_in_a = jnp.where(m, d_corners_b_in_a, EMPTY_CLEARANCE_SENTINEL)
        chunks.append(d_corners_b_in_a)
    pool = (
        jnp.concatenate(chunks, axis=0) if chunks else jnp.zeros((0,), dtype=R_a.dtype)
    )
    return hard_soft(pool, beta=beta, top_k=top_k)


def body_body_pair_clearance(
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
    *,
    beta: float = 20.0,
    top_k: int = 16,
    n_real_a: Array | None = None,
    n_real_b: Array | None = None,
) -> tuple[Array, Array]:
    """Body-body category only of the dual-rep clearance. Returns
    ``(hard_min, soft_min_topk)``. Vmap-friendly: shape-static (no
    Python ``if`` branches), uniform per-probe inputs, suitable for
    ``jax.vmap`` across a pair axis.

    Hoisting this out of ``pairwise_signed_clearance_dual_world`` lets
    the caller batch all P probe-pairs into ONE XLA kernel launch
    instead of P separate launches per ``jit_obj`` call — the unrolled
    Python pair loop was responsible for ~50% of the objective wall (per
    2026-05-23 jax.profiler trace).
    """
    kw_a = {} if n_real_a is None else {"n_real": n_real_a}
    kw_b = {} if n_real_b is None else {"n_real": n_real_b}
    local_b_in_a = (world_surface_b - t_a) @ R_a
    d_b_in_a = trilinear_sdf(
        sdf_a_grid, sdf_a_origin, sdf_a_spacing, local_b_in_a, **kw_a
    )
    local_a_in_b = (world_surface_a - t_b) @ R_b
    d_a_in_b = trilinear_sdf(
        sdf_b_grid, sdf_b_origin, sdf_b_spacing, local_a_in_b, **kw_b
    )
    pool = jnp.concatenate([d_b_in_a.reshape(-1), d_a_in_b.reshape(-1)])
    return jnp.min(pool), soft_min_topk(pool, beta=beta, top_k=top_k)


# Clearance read for padded cell slots; large enough never to be the minimum.
_PAD_CLEARANCE_MM = 1e3


def body_body_pair_clearance_c2f(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    kind_a: Array,
    kind_b: Array,
    grids: Array,
    origins: Array,
    spacings: Array,
    n_reals: Array,
    outside_min: Array,
    coarse: Array,
    fine: Array,
    cells: Array,
    radius: Array,
    *,
    beta: float = 20.0,
    top_k: int = 16,
    n_cells: int = 16,
) -> tuple[Array, Array]:
    """Body-body clearance from coarse-to-fine surface samples, returning
    ``(hard_min, soft_min_topk)`` like :func:`body_body_pair_clearance`.

    Looks up each probe's coarse points in the other probe's SDF, ranks cells by
    ``coarse value − cell radius`` (no fine point in a cell reads lower, since a
    distance field changes no faster than its query point moves), and looks up the
    fine points of the ``n_cells`` best cells. Per-kind tables in each kind's local
    frame: ``coarse`` (K, C, 3), ``fine`` (K, F, 3), ``cells`` (K, C, W) listing fine
    indices padded with F, ``radius`` (K, C); grids as for
    :func:`trilinear_sdf_stacked`. ``outside_min`` (K,) is the lowest value on each
    grid's boundary: a point off the grid is at least that far from the body, which
    stands in for the out-of-grid sentinel when ranking, so a cell whose coarse point
    is off the grid but whose members are close still ranks by a valid bound. See
    :mod:`aind_rutter.optimization.clearance.samples`.
    """
    n_coarse, n_fine = coarse.shape[1], fine.shape[1]

    def into(points: Array, R_src: Array, t_src: Array, R_dst: Array, t_dst: Array):
        return (points @ R_src.T + t_src - t_dst) @ R_dst

    def lookup(kind: Array, points: Array) -> Array:
        return trilinear_sdf_stacked(grids, origins, spacings, n_reals, kind, points)

    coarse_values = jnp.concatenate(
        [
            lookup(kind_a, into(coarse[kind_b], R_b, t_b, R_a, t_a)),
            lookup(kind_b, into(coarse[kind_a], R_a, t_a, R_b, t_b)),
        ]
    )
    off_grid = coarse_values >= _PAD_CLEARANCE_MM
    floor = jnp.concatenate(
        [
            jnp.broadcast_to(outside_min[kind_a], (n_coarse,)),
            jnp.broadcast_to(outside_min[kind_b], (n_coarse,)),
        ]
    )
    bound = jnp.where(off_grid, floor, coarse_values) - jnp.concatenate(
        [radius[kind_b], radius[kind_a]]
    )
    _, chosen = jax.lax.top_k(-bound, n_cells)
    of_a = chosen >= n_coarse  # the cell holds probe a's points, read in b's SDF
    owner = jnp.where(of_a, kind_a, kind_b)
    members = cells[owner, jnp.where(of_a, chosen - n_coarse, chosen)]  # (n_cells, W)
    points = fine[owner[:, None], jnp.minimum(members, n_fine - 1)]
    R_src = jnp.where(of_a[:, None, None], R_a, R_b)
    t_src = jnp.where(of_a[:, None], t_a, t_b)
    R_dst = jnp.where(of_a[:, None, None], R_b, R_a)
    t_dst = jnp.where(of_a[:, None], t_b, t_a)
    world = jnp.einsum("cwj,ckj->cwk", points, R_src) + t_src[:, None, :]
    local = jnp.einsum("cwk,ckj->cwj", world - t_dst[:, None, :], R_dst)
    fine_values = lookup(jnp.where(of_a, kind_b, kind_a)[:, None], local)
    fine_values = jnp.where(members < n_fine, fine_values, _PAD_CLEARANCE_MM)
    pool = jnp.concatenate([coarse_values, fine_values.reshape(-1)])
    return jnp.min(pool), soft_min_topk(pool, beta=beta, top_k=top_k)


def body_shank_box_clearance_c2f(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    kind_a: Array,
    kind_b: Array,
    coarse: Array,
    fine: Array,
    cells: Array,
    radius: Array,
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k: int = 8,
    n_cells: int = 16,
    shank_mask_a: Array | None = None,
    shank_mask_b: Array | None = None,
) -> tuple[Array, Array]:
    """Body-vs-shank-box category from the coarse-to-fine body samples, returning
    ``(hard_min, soft_min_topk)`` like the body-vs-OBB pool of
    :func:`shank_only_pair_clearance`.

    Each probe's coarse body points are measured against the other probe's shank
    boxes, cells are ranked by ``coarse value − cell radius`` (the box distance is
    also 1-Lipschitz, so no cell member reads lower), and the fine points of the
    ``n_cells`` best cells are measured. Sample tables are those of
    :func:`body_body_pair_clearance_c2f`; masked-out boxes and padded cell slots
    read the no-collision sentinel.
    """
    n_coarse, n_fine = coarse.shape[1], fine.shape[1]

    def to_world(points, R, t):
        return points @ R.T + t

    coarse_values = jnp.concatenate(
        [
            boxes_sdf(
                to_world(coarse[kind_b], R_b, t_b),
                R_a,
                t_a,
                shank_centers_a,
                shank_halves_a,
                shank_mask_a,
            ),
            boxes_sdf(
                to_world(coarse[kind_a], R_a, t_a),
                R_b,
                t_b,
                shank_centers_b,
                shank_halves_b,
                shank_mask_b,
            ),
        ]
    )
    bound = coarse_values - jnp.concatenate([radius[kind_b], radius[kind_a]])
    _, chosen = jax.lax.top_k(-bound, n_cells)
    of_a = chosen >= n_coarse  # the cell holds probe a's points, read against b
    owner = jnp.where(of_a, kind_a, kind_b)
    members = cells[owner, jnp.where(of_a, chosen - n_coarse, chosen)]  # (n_cells, W)
    points = fine[owner[:, None], jnp.minimum(members, n_fine - 1)]
    source_R = jnp.where(of_a[:, None, None], R_a, R_b)
    source_t = jnp.where(of_a[:, None], t_a, t_b)
    world = jnp.einsum("cwj,ckj->cwk", points, source_R) + source_t[:, None, :]
    flat = world.reshape(-1, 3)
    against_a = boxes_sdf(flat, R_a, t_a, shank_centers_a, shank_halves_a, shank_mask_a)
    against_b = boxes_sdf(flat, R_b, t_b, shank_centers_b, shank_halves_b, shank_mask_b)
    fine_values = jnp.where(
        jnp.repeat(of_a, members.shape[1]), against_b, against_a
    ).reshape(members.shape)
    fine_values = jnp.where(members < n_fine, fine_values, EMPTY_CLEARANCE_SENTINEL)
    pool = jnp.concatenate([coarse_values, fine_values.reshape(-1)])
    return hard_soft(pool, beta=beta, top_k=top_k)
