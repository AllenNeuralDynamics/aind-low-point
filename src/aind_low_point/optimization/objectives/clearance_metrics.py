"""Vmappable per-candidate minimum-clearance metric (dual-rep, JAX).

``make_min_clear_one`` builds a closure over the same per-candidate arglist as
the batched Phase-1 objective, so it vmaps across candidates to report the hard
(non-softened) minimum dual-representation clearance in millimetres (<0 = overlap).
"""

from __future__ import annotations

import jax.numpy as jnp

from aind_low_point.optimization.objectives.phase1 import PHASE1_PER_PROBE_VARS
from aind_low_point.optimization.sdf.kernels import (
    dual_rep_fixture_clearance,
    dual_rep_pair_clearance,
    pose_from_optimizer_vars,
    spin_deg_from_sxy,
)

PPV = PHASE1_PER_PROBE_VARS


def make_min_clear_one(n_arcs, n_probes, fixtures, w):
    """Per-candidate min hard dual-rep clearance (mm; <0 = overlap). Same args
    as the batched objective so it vmaps over the shared arglist."""
    pairs = [(i, j) for i in range(n_probes) for j in range(i + 1, n_probes)]
    beta = float(w.softmin_beta)
    tk_bb, tk_bs, tk_ss = (
        int(w.top_k_body_body),
        int(w.top_k_body_shank),
        int(w.top_k_shank_shank),
    )

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
        sdf_table=None,  # shared ARG_ORDER arg (swept-pair table); unused here
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
        ws = [sdf_surfaces[i] @ Rs[i].T + ts[i] for i in range(n_probes)]
        clears = []
        for ia, ib in pairs:
            pc = dual_rep_pair_clearance(
                Rs[ia],
                ts[ia],
                Rs[ib],
                ts[ib],
                sdf_grids[ia],
                sdf_origins[ia],
                sdf_spacings[ia],
                sdf_grids[ib],
                sdf_origins[ib],
                sdf_spacings[ib],
                ws[ia],
                ws[ib],
                shank_obb_centers[ia],
                shank_obb_halves[ia],
                shank_obb_centers[ib],
                shank_obb_halves[ib],
                beta=beta,
                top_k_body_body=tk_bb,
                top_k_body_shank=tk_bs,
                top_k_shank_shank=tk_ss,
            )
            clears.append(
                jnp.min(
                    jnp.stack(
                        [
                            pc.body_body[0],
                            pc.body_shank_corners[0],
                            pc.body_shank_obb[0],
                            pc.shank_shank[0],
                        ]
                    )
                )
            )
        for fx in fixtures:
            for i in range(n_probes):
                fc = dual_rep_fixture_clearance(
                    Rs[i],
                    ts[i],
                    sdf_grids[i],
                    sdf_origins[i],
                    sdf_spacings[i],
                    fx.grid,
                    fx.origin,
                    fx.spacing,
                    ws[i],
                    fx.surface,
                    shank_obb_centers[i],
                    shank_obb_halves[i],
                    beta=beta,
                    top_k_body=tk_bb,
                    top_k_obb=tk_bs,
                )
                clears.append(jnp.minimum(fc.body[0], fc.obb[0]))
        return jnp.min(jnp.stack(clears))

    return min_clear_one
