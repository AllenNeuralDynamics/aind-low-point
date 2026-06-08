"""Vmapped probe-pair clearance sweep over a padded per-probe SDF/OBB table.

The Stage-3 Phase-1/2 objectives compute dual-rep clearance for every probe
pair with a Python-unrolled ``for ia, ib in sdf_pair_list`` loop. Reverse-mode
autodiff through ~C(P,2) unrolled copies of the dual-rep subgraph dominates the
XLA compile (measured: the pair loop owns ~90% of Phase-2's ~115s grad /
constr_jac compile; fixtures ~10%). This module collapses that loop to a single
``jax.vmap`` over the static pair-index list, gathering each probe's grid / OBB
from a padded ``(P, ...)`` table — one dual-rep subgraph copy instead of C(P,2).

Grids/OBBs are padded to a common shape so they can be a single stacked operand;
the padding is **bit-exact** for the clearance:

  - SDF grids are edge-padded (replicate the boundary) to the common shape, and
    each probe's real extent is carried in ``real_shapes`` and passed to
    ``trilinear_sdf`` via ``n_real`` so its in-bounds / out-of-bounds behaviour
    uses the true extent, not the padded shape. body_body queries the OTHER
    probe's surface against this grid (often out-of-extent); this makes the
    padded grid bit-exact with the unpadded grid for in- AND out-of-extent
    queries (zero-padding would return a phantom ~0 collision instead).
  - Shank-OBB tables are padded to ``max_Sa`` rows and the padded rows are masked
    out via the mask-aware helpers (``shank_mask_a/b``) ⇒ inert.
  - Body-surface point clouds are uniform across real probes (no padding needed);
    placeholder/no-SDF probes (never referenced by a pair) pad to the max.

Only the **pair** loop is vmapped here. The fixture loop stays unrolled in the
caller: it is ~10% of the compile and ``dual_rep_fixture_clearance`` has no mask
parameter, so a padded-OBB fixture path would not be bit-exact.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aind_low_point.optimization.sdf_jax import (
    body_body_pair_clearance,
    body_shank_corners_pair_clearance,
    shank_only_pair_clearance,
)

# Category order MUST match PROBE_PAIR_SLACK_GAINS and the unrolled loop's
# (body_body, body_shank_corners, body_shank_obb, shank_shank) tuple order.
N_PAIR_CATEGORIES = 4


def build_padded_probe_tables(
    sdf_grids: tuple,
    sdf_origins: tuple,
    sdf_spacings: tuple,
    sdf_surfaces: tuple,
    shank_obb_centers: tuple,
    shank_obb_halves: tuple,
) -> dict:
    """Pad the heterogeneous per-probe SDF/OBB tuples into uniform stacked
    tables, so a dynamic pair index can gather them inside ``jax.vmap``.

    All inputs are length-P tuples of jnp arrays (as produced by
    ``stage3_phase1_jax._pack_statics``). Shapes are static at trace time, so the
    pad widths below are concrete Python ints. Returns a dict of stacked tables
    plus the per-probe ``obb_mask`` (1.0 for real OBB rows, 0.0 for padding).
    """
    P = len(sdf_grids)
    gx = max(int(g.shape[0]) for g in sdf_grids)
    gy = max(int(g.shape[1]) for g in sdf_grids)
    gz = max(int(g.shape[2]) for g in sdf_grids)
    nsurf = max(int(s.shape[0]) for s in sdf_surfaces)
    max_sa = max(int(c.shape[0]) for c in shank_obb_centers)
    max_sa = max(max_sa, 1)  # avoid a 0-width OBB axis

    # Edge-pad (replicate boundary), NOT zero-pad: body_body queries the OTHER
    # probe's surface against this grid, and those points fall outside this
    # probe's SDF extent. The unpadded grid clamps such out-of-extent queries to
    # the real edge value (large = far); zero-padding would instead return ~0
    # (a phantom collision). Edge-padding makes the clamped value equal the real
    # edge ⇒ bit-exact with the unpadded grid for in- and out-of-extent queries.
    grids = jnp.stack(
        [
            jnp.pad(
                g,
                ((0, gx - g.shape[0]), (0, gy - g.shape[1]), (0, gz - g.shape[2])),
                mode="edge",
            )
            for g in sdf_grids
        ]
    )
    origins = jnp.stack([jnp.asarray(o, jnp.float32) for o in sdf_origins])
    spacings = jnp.asarray([jnp.asarray(s, jnp.float32) for s in sdf_spacings])
    surfaces = jnp.stack(
        [jnp.pad(s, ((0, nsurf - s.shape[0]), (0, 0))) for s in sdf_surfaces]
    )

    obb_c = jnp.zeros((P, max_sa, 3), jnp.float32)
    obb_h = jnp.zeros((P, max_sa, 3), jnp.float32)
    obb_m = jnp.zeros((P, max_sa), jnp.float32)
    for i in range(P):
        n = int(shank_obb_centers[i].shape[0])
        if n > 0:
            obb_c = obb_c.at[i, :n].set(shank_obb_centers[i])
            obb_h = obb_h.at[i, :n].set(shank_obb_halves[i])
            obb_m = obb_m.at[i, :n].set(1.0)

    # Real (unpadded) extent per probe, so trilinear's in-bounds test uses the
    # true extent (not the padded shape) — out-of-extent queries then return the
    # out-of-bounds sentinel exactly as for the unpadded grid.
    real_shapes = jnp.asarray(
        [[int(g.shape[0]), int(g.shape[1]), int(g.shape[2])] for g in sdf_grids],
        dtype=jnp.int32,
    )

    return dict(
        grids=grids,
        origins=origins,
        spacings=spacings,
        surfaces=surfaces,
        obb_centers=obb_c,
        obb_halves=obb_h,
        obb_mask=obb_m,
        real_shapes=real_shapes,
    )


def swept_pair_clearances(
    Rs,
    ts,
    tables: dict,
    pair_a,
    pair_b,
    *,
    beta: float,
    top_k_body_body: int,
    top_k_body_shank: int,
    top_k_shank_shank: int,
):
    """Per-pair dual-rep clearance, ``vmap``'d over the static pair index list.

    Parameters
    ----------
    Rs, ts : (P,3,3), (P,3) world poses (stacked).
    tables : output of :func:`build_padded_probe_tables`.
    pair_a, pair_b : (n_pairs,) int32 — probe indices of each pair (i<j), static.

    Returns
    -------
    hard, soft : (n_pairs, 4) — per-pair clearance in category order
        (body_body, body_shank_corners, body_shank_obb, shank_shank), matching
        ``PROBE_PAIR_SLACK_GAINS`` and the unrolled loop.
    """
    grids = tables["grids"]
    origins = tables["origins"]
    spacings = tables["spacings"]
    surfaces = tables["surfaces"]
    obb_c = tables["obb_centers"]
    obb_h = tables["obb_halves"]
    obb_m = tables["obb_mask"]
    real_shapes = tables["real_shapes"]
    # World-frame body surfaces, once per probe (stacked).
    world_surf = jnp.matmul(surfaces, jnp.transpose(Rs, (0, 2, 1))) + ts[:, None, :]

    def _pair(ia, ib):
        Ra, ta, Rb, tb = Rs[ia], ts[ia], Rs[ib], ts[ib]
        ga, oa, sa = grids[ia], origins[ia], spacings[ia]
        gb, ob, sb = grids[ib], origins[ib], spacings[ib]
        nra, nrb = real_shapes[ia], real_shapes[ib]
        sfa, sfb = world_surf[ia], world_surf[ib]
        oca, oha, oma = obb_c[ia], obb_h[ia], obb_m[ia]
        ocb, ohb, omb = obb_c[ib], obb_h[ib], obb_m[ib]
        bb = body_body_pair_clearance(
            Ra,
            ta,
            Rb,
            tb,
            ga,
            oa,
            sa,
            gb,
            ob,
            sb,
            sfa,
            sfb,
            beta=beta,
            top_k=top_k_body_body,
            n_real_a=nra,
            n_real_b=nrb,
        )
        bsc = body_shank_corners_pair_clearance(
            Ra,
            ta,
            Rb,
            tb,
            ga,
            oa,
            sa,
            gb,
            ob,
            sb,
            oca,
            oha,
            ocb,
            ohb,
            beta=beta,
            top_k=top_k_body_shank,
            shank_mask_a=oma,
            shank_mask_b=omb,
            n_real_a=nra,
            n_real_b=nrb,
        )
        bso, sss = shank_only_pair_clearance(
            Ra,
            ta,
            Rb,
            tb,
            sfa,
            sfb,
            oca,
            oha,
            ocb,
            ohb,
            beta=beta,
            top_k_body_shank=top_k_body_shank,
            top_k_shank_shank=top_k_shank_shank,
            shank_mask_a=oma,
            shank_mask_b=omb,
        )
        hard = jnp.stack([bb[0], bsc[0], bso[0], sss[0]])
        soft = jnp.stack([bb[1], bsc[1], bso[1], sss[1]])
        return hard, soft

    return jax.vmap(_pair)(pair_a, pair_b)
