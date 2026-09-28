"""Vmappable per-candidate clearance score.

``make_min_clear_one`` builds a closure over the same per-candidate arglist as
the batched Phase-1 objective, so it vmaps across candidates.
"""

from __future__ import annotations

import jax.numpy as jnp

from aind_rutter.optimization.clearance.poses import (
    pose_from_optimizer_vars,
    spin_deg_from_sxy,
)
from aind_rutter.optimization.clearance.sweep import (
    build_padded_fixture_table,
    swept_fixture_clearances,
    swept_pair_clearances,
)
from aind_rutter.optimization.objectives.feasibility import FeasibilitySpec
from aind_rutter.optimization.objectives.slacks import (
    fixture_clearance_slack,
    pair_clearance_slack,
)
from aind_rutter.optimization.objectives.soft import PHASE1_PER_PROBE_VARS

PPV = PHASE1_PER_PROBE_VARS


def make_min_clear_one(n_arcs, n_probes, fixtures, spec: FeasibilitySpec):
    """Per-candidate worst clearance slack under ``spec`` (mm; < 0 infeasible),
    over every probe pair and every probe against ``fixtures``.

    Reads the rows Phase 2 constrains, before their gains, so a candidate
    scoring ≥ 0 meets Phase 2's clearance rows on these fixtures. Every probe
    must have an SDF. Same args as the batched objective so it vmaps over the
    shared arglist.
    """
    pairs = [(i, j) for i in range(n_probes) for j in range(i + 1, n_probes)]
    pair_a = jnp.asarray([a for a, _ in pairs], jnp.int32)
    pair_b = jnp.asarray([b for _, b in pairs], jnp.int32)
    fix_table = build_padded_fixture_table(fixtures)

    def min_clear_one(
        x,
        target_LPS,
        pivot_local,
        arc_idx,
        tips_local,
        shank_mask,
        s_axes,
        s_centers,
        s_e1,
        s_e2,
        s_cos,
        s_sin,
        s_a,
        s_b,
        section_mask,
        same_arc_mask,
        sdf_grids,
        sdf_origins,
        sdf_spacings,
        sdf_surfaces,
        shank_obb_centers,
        shank_obb_halves,
        sdf_table,
        w_normals=None,  # shared ARG_ORDER args; these only affect threading
        w_offsets=None,
        shank_half_xy=None,
    ):
        arc_aps = x[:n_arcs]
        Rs, ts = [], []
        for i in range(n_probes):
            off = n_arcs + PPV * i
            R, t = pose_from_optimizer_vars(
                target_LPS=target_LPS[i],
                ap_deg=arc_aps[arc_idx[i]],
                ml_deg=x[off + 0],
                spin_deg=spin_deg_from_sxy(x[off + 1], x[off + 2]),
                offset_R_mm=x[off + 3],
                offset_A_mm=x[off + 4],
                past_target_mm=x[off + 5],
                recording_center_local=pivot_local[i],
            )
            Rs.append(R)
            ts.append(t)
        Rs, ts = jnp.stack(Rs), jnp.stack(ts)
        slacks = []
        if pairs:
            _, soft = swept_pair_clearances(
                Rs, ts, sdf_table, pair_a, pair_b, **spec.pair_softmin_kwargs()
            )
            slacks.append(pair_clearance_slack(soft, spec).reshape(-1))
        if fix_table:
            _, soft = swept_fixture_clearances(
                Rs,
                ts,
                sdf_table,
                fix_table,
                list(range(n_probes)),
                **spec.fixture_softmin_kwargs(),
            )
            slacks.append(fixture_clearance_slack(soft, spec).reshape(-1))
        if not slacks:
            return jnp.float32(jnp.inf)
        return jnp.min(jnp.concatenate(slacks))

    return min_clear_one
