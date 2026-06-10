"""Vmapped probe-pair clearance sweep over a padded per-probe SDF/OBB table.

The Stage-3 Phase-1/2 objectives compute dual-rep clearance for every probe
pair with a Python-unrolled ``for ia, ib in sdf_pair_list`` loop. Reverse-mode
autodiff through ~C(P,2) unrolled copies of the dual-rep subgraph dominates the
XLA compile (measured: the pair loop owns ~90% of Phase-2's ~115s grad /
constr_jac compile; fixtures ~10%). This module collapses that loop to a single
``jax.vmap`` over the static pair-index list, gathering each probe's grid / OBB
from a padded per-KIND ``(N_kinds, ...)`` table by ``kind_id`` (the same
representation the spin-restore uses) — one dual-rep subgraph copy instead of
C(P,2), and only N_kinds distinct grids stored rather than P.

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

Both the **pair** loop (:func:`swept_pair_clearances`) and the **fixture** loop
(:func:`swept_fixture_clearances`, a double vmap over fixture × probe) are
collapsed here. Fixtures are stacked into a padded table
(:func:`build_padded_fixture_table`) the same way: edge-padded grids + per-fixture
``real_shapes`` → ``n_real_f``, with uniform surface counts (the cone-crop keeps
the surface in full). The cone is cropped to the well neighbourhood upstream so
the padded fixture stack stays small.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from aind_low_point.optimization.sdf.kernels import (
    body_body_pair_clearance,
    body_shank_corners_pair_clearance,
    dual_rep_fixture_clearance,
    shank_only_pair_clearance,
)

# Category order MUST match PROBE_PAIR_SLACK_GAINS and the unrolled loop's
# (body_body, body_shank_corners, body_shank_obb, shank_shank) tuple order.
N_PAIR_CATEGORIES = 4


# ---------------------------------------------------------------------------
# bf16 grid-storage policy (shared by Phase-1 and Phase-2)
# ---------------------------------------------------------------------------
#
# Which SDFs get bf16 storage lives here, in ONE place: every *collision* grid —
# the per-probe probe grids, the swept-pair ``sdf_table`` grids, and the fixture
# (well/cone/headframe) grids. ``trilinear_sdf`` is dtype-polymorphic (bf16
# gather/blend, fp32 reduction — rank-safe for the 5000-sample SDF pools). The
# brain SDF is deliberately NOT cast: it's a few-point shank-tip query, the
# few-element regime where bf16 erodes rank. ``grid_dtype == float32`` ⇒ no-op.


def cast_fixture_grids(fixtures, grid_dtype):
    """Return ``fixtures`` with each fixture's SDF grid stored at ``grid_dtype``.

    Cast fixtures BEFORE they are closure-captured by the JIT builder. No-op for
    float32.
    """
    if grid_dtype == jnp.float32:
        return fixtures
    return tuple(replace(fx, grid=jnp.asarray(fx.grid, grid_dtype)) for fx in fixtures)


def cast_packed_grids(packed: dict, grid_dtype) -> dict:
    """Cast the collision grids in a packed/shared statics dict (from
    ``_pack_statics``) to ``grid_dtype`` IN PLACE: the per-probe ``sdf_grids``
    tuple (fixture loop) and the ``sdf_table`` grids (pair sweep). Brain is left
    fp32. No-op for float32. Returns ``packed`` for chaining.
    """
    if grid_dtype == jnp.float32:
        return packed
    packed["sdf_grids"] = tuple(jnp.asarray(g, grid_dtype) for g in packed["sdf_grids"])
    packed["sdf_table"] = {
        **packed["sdf_table"],
        "grids": jnp.asarray(packed["sdf_table"]["grids"], grid_dtype),
    }
    return packed


def build_padded_probe_tables(
    sdf_grids: tuple,
    sdf_origins: tuple,
    sdf_spacings: tuple,
    sdf_surfaces: tuple,
    shank_obb_centers: tuple,
    shank_obb_halves: tuple,
) -> dict:
    """Dedup the per-probe SDF/OBB tuples into a per-KIND stacked table plus a
    per-probe ``kind_id``, so a dynamic index gathers ``grids[kind_id[i]]`` inside
    ``jax.vmap`` — the same representation the spin-restore uses.

    Same-kind probes share the *identical* SDF grid object (one ``ProbeSDF`` per
    kind), so deduping by grid identity is bit-exact with a per-probe table while
    storing only ``N_kinds`` distinct grids (e.g. 3 vs 7) — less padding, less
    HBM. Inputs are length-P tuples of jnp arrays (from ``_pack_statics``); pad
    widths are concrete Python ints at trace time. Returns the per-kind tables +
    per-probe ``kind_id``, with ``obb_mask`` (1.0 real OBB rows, 0.0 padding).
    """
    P = len(sdf_grids)
    # Dedup by grid object identity → per-kind lists + per-probe kind index.
    kind_of: dict[int, int] = {}
    kind_id_list: list[int] = []
    kg, ko, ks, ksurf, kobc, kobh = [], [], [], [], [], []
    for i in range(P):
        gid = id(sdf_grids[i])
        if gid not in kind_of:
            kind_of[gid] = len(kg)
            kg.append(sdf_grids[i])
            ko.append(sdf_origins[i])
            ks.append(sdf_spacings[i])
            ksurf.append(sdf_surfaces[i])
            kobc.append(shank_obb_centers[i])
            kobh.append(shank_obb_halves[i])
        kind_id_list.append(kind_of[gid])
    nk = len(kg)
    gx = max(int(g.shape[0]) for g in kg)
    gy = max(int(g.shape[1]) for g in kg)
    gz = max(int(g.shape[2]) for g in kg)
    nsurf = max(int(s.shape[0]) for s in ksurf)
    max_sa = max(max(int(c.shape[0]) for c in kobc), 1)  # avoid 0-width OBB axis

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
            for g in kg
        ]
    )
    origins = jnp.stack([jnp.asarray(o, jnp.float32) for o in ko])
    spacings = jnp.asarray([jnp.asarray(s, jnp.float32) for s in ks])
    surfaces = jnp.stack([jnp.pad(s, ((0, nsurf - s.shape[0]), (0, 0))) for s in ksurf])

    obb_c = jnp.zeros((nk, max_sa, 3), jnp.float32)
    obb_h = jnp.zeros((nk, max_sa, 3), jnp.float32)
    obb_m = jnp.zeros((nk, max_sa), jnp.float32)
    for k in range(nk):
        n = int(kobc[k].shape[0])
        if n > 0:
            obb_c = obb_c.at[k, :n].set(kobc[k])
            obb_h = obb_h.at[k, :n].set(kobh[k])
            obb_m = obb_m.at[k, :n].set(1.0)

    # Real (unpadded) extent per kind, so trilinear's in-bounds test uses the
    # true extent (not the padded shape) — out-of-extent queries then return the
    # out-of-bounds sentinel exactly as for the unpadded grid.
    real_shapes = jnp.asarray(
        [[int(g.shape[0]), int(g.shape[1]), int(g.shape[2])] for g in kg],
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
        kind_id=jnp.asarray(kind_id_list, jnp.int32),
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
    kind_id = tables["kind_id"]  # (P,) probe → kind
    # World-frame body surfaces, once per PROBE: gather the per-kind local
    # surface by kind, then transform by the probe's pose.
    world_surf = (
        jnp.matmul(surfaces[kind_id], jnp.transpose(Rs, (0, 2, 1))) + ts[:, None, :]
    )

    def _pair(ia, ib):
        Ra, ta, Rb, tb = Rs[ia], ts[ia], Rs[ib], ts[ib]
        ka, kb = kind_id[ia], kind_id[ib]  # gather grids/OBB by KIND, not probe
        ga, oa, sa = grids[ka], origins[ka], spacings[ka]
        gb, ob, sb = grids[kb], origins[kb], spacings[kb]
        nra, nrb = real_shapes[ka], real_shapes[kb]
        sfa, sfb = world_surf[ia], world_surf[ib]
        oca, oha, oma = obb_c[ka], obb_h[ka], obb_m[ka]
        ocb, ohb, omb = obb_c[kb], obb_h[kb], obb_m[kb]
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


# Category order MUST match FIXTURE_PAIR_SLACK_GAINS and the unrolled loop's
# (body, obb) tuple order.
N_FIXTURE_CATEGORIES = 2


def build_padded_fixture_table(fixtures) -> dict:
    """Stack heterogeneous-shape fixture SDFs into one padded table so the
    fixture loop collapses to a vmap axis (fixture × probe in one fused kernel).

    Grids are edge-padded to a common shape with each fixture's real extent in
    ``real_shapes`` → ``trilinear_sdf(n_real=...)`` (the ``d_p_in_f`` direction:
    probe surface points vs the fixture grid), bit-exact with the unpadded grid
    for in- and out-of-extent queries. Surface counts are uniform across fixtures
    (the cone crop keeps the surface in full), so surfaces stack directly with no
    mask. Origins / spacings stack per fixture. Built ONCE at objective-build
    time and closure-captured. Empty ⇒ ``{}`` (no fixtures).
    """
    if not fixtures:
        return {}
    grids = [jnp.asarray(f.grid) for f in fixtures]
    gx = max(int(g.shape[0]) for g in grids)
    gy = max(int(g.shape[1]) for g in grids)
    gz = max(int(g.shape[2]) for g in grids)
    padded = jnp.stack(
        [
            jnp.pad(
                g,
                ((0, gx - g.shape[0]), (0, gy - g.shape[1]), (0, gz - g.shape[2])),
                mode="edge",
            )
            for g in grids
        ]
    )
    return dict(
        grids=padded,
        origins=jnp.stack([jnp.asarray(f.origin, jnp.float32) for f in fixtures]),
        spacings=jnp.stack([jnp.asarray(f.spacing, jnp.float32) for f in fixtures]),
        surfaces=jnp.stack([jnp.asarray(f.surface, jnp.float32) for f in fixtures]),
        real_shapes=jnp.asarray(
            [[int(g.shape[0]), int(g.shape[1]), int(g.shape[2])] for g in grids],
            dtype=jnp.int32,
        ),
    )


def swept_fixture_clearances(
    Rs,
    ts,
    tables: dict,
    fix_table: dict,
    probe_idx,
    *,
    beta: float,
    top_k_body: int,
    top_k_obb: int,
):
    """Probe-fixture clearance, double-``vmap``'d over (fixture × probe).

    Both loops collapse to vmap axes: the fixtures are gathered from a padded
    ``fix_table`` (edge-padded grids + per-fixture ``real_shapes`` → ``n_real_f``;
    uniform surfaces), the probes gather grid / OBB / real extent from the
    per-kind ``tables`` by ``kind_id`` (``n_real_p`` + masked OBB rows). One fused
    kernel over n_fixtures × n_probes — better GPU occupancy than n_fixtures
    sequential probe-vmaps, and one dual-rep subgraph traced instead of
    n_fixtures copies.

    Parameters
    ----------
    Rs, ts : (P,3,3), (P,3) world poses.
    tables : output of :func:`build_padded_probe_tables`.
    fix_table : output of :func:`build_padded_fixture_table`.
    probe_idx : (n_sdf_probes,) int32 — probe indices with an SDF (static).

    Returns
    -------
    hard, soft : (n_fixtures, n_sdf_probes, 2) — per (fixture, probe) clearance in
        category order (body, obb), matching ``FIXTURE_PAIR_SLACK_GAINS`` and the
        unrolled ``for fx: for i:`` loop.
    """
    grids = tables["grids"]
    origins = tables["origins"]
    spacings = tables["spacings"]
    surfaces = tables["surfaces"]
    obb_c = tables["obb_centers"]
    obb_h = tables["obb_halves"]
    obb_m = tables["obb_mask"]
    real_shapes = tables["real_shapes"]
    kind_id = tables["kind_id"]
    world_surf = (
        jnp.matmul(surfaces[kind_id], jnp.transpose(Rs, (0, 2, 1))) + ts[:, None, :]
    )
    idx = jnp.asarray(probe_idx, jnp.int32)

    def _per_fixture(fg, fo, fs, fsurf, fnr):
        def _probe(i):
            k = kind_id[i]
            fc = dual_rep_fixture_clearance(
                Rs[i],
                ts[i],
                grids[k],
                origins[k],
                spacings[k],
                fg,
                fo,
                fs,
                world_surf[i],
                fsurf,
                obb_c[k],
                obb_h[k],
                beta=beta,
                top_k_body=top_k_body,
                top_k_obb=top_k_obb,
                n_real_p=real_shapes[k],
                n_real_f=fnr,
                shank_mask=obb_m[k],
            )
            return jnp.stack([fc.body[0], fc.obb[0]]), jnp.stack(
                [fc.body[1], fc.obb[1]]
            )

        return jax.vmap(_probe)(idx)  # (n_sdf_probes, 2) each

    return jax.vmap(_per_fixture)(
        fix_table["grids"],
        fix_table["origins"],
        fix_table["spacings"],
        fix_table["surfaces"],
        fix_table["real_shapes"],
    )
