"""Read a solved ``x`` back out as world poses and threading margins.

Reporting and validation rather than optimization: both map over probes with
``jax.vmap`` but neither is differentiated, and the FCL gate and the threading
diagnostics go through them.
"""

from __future__ import annotations

import jax

from aind_rutter.optimization.clearance.poses import (
    pose_from_optimizer_vars,
    spin_deg_from_sxy,
)
from aind_rutter.optimization.objectives.layout import PHASE1_PER_PROBE_VARS
from aind_rutter.optimization.objectives.threading import threading_g_matrix


def poses_from_x(x, n_arcs, n_probes, target_LPS, pivot_local, arc_idx):
    """Stacked (Rs, ts) ``(P,3,3)``/``(P,3)`` per probe from a Phase 2 x vector,
    vmapped over probes (one pose subgraph instead of P unrolled copies). x after
    the arcs is (ml, sx, sy, off_R, off_A, depth) × P."""
    arc_aps = x[:n_arcs]
    xp = x[n_arcs : n_arcs + PHASE1_PER_PROBE_VARS * n_probes].reshape(
        n_probes, PHASE1_PER_PROBE_VARS
    )
    aps = arc_aps[arc_idx]

    def _one(xp6, ap, target, pivot):
        return pose_from_optimizer_vars(
            target_LPS=target,
            ap_deg=ap,
            ml_deg=xp6[0],
            spin_deg=spin_deg_from_sxy(xp6[1], xp6[2]),
            offset_R_mm=xp6[3],
            offset_A_mm=xp6[4],
            past_target_mm=xp6[5],
            recording_center_local=pivot,
        )

    return jax.vmap(_one)(xp, aps, target_LPS, pivot_local)


def threading_g_per_probe(
    Rs,
    ts,
    tips_local,
    s_axes,
    s_centers,
    s_e1,
    s_e2,
    s_cos,
    s_sin,
    s_a,
    s_b,
    section_mask,
    shank_mask,
    w_normals=None,
    w_offsets=None,
    *,
    shaft_len,
):
    """vmap ``threading_g_matrix`` over probes → ``(g, valid)``, each
    ``(P, S, SH)`` — one threading subgraph instead of P unrolled copies. Both
    the objective (reward) and the constraint vector consume this; flatten with
    ``.reshape(-1)`` for the probe-major order the old per-probe loop produced."""

    def _one(R, t, tips, sax, scen, se1, se2, scos, ssin, sa, sb, sec_m, sh_m, wn, wo):
        g = threading_g_matrix(
            R,
            t,
            tips,
            sax,
            scen,
            se1,
            se2,
            scos,
            ssin,
            sa,
            sb,
            shaft_length_mm=shaft_len,
            w_normals=wn,
            w_offsets=wo,
        )
        return g, sec_m[:, None] * sh_m[None, :]

    return jax.vmap(_one)(
        Rs,
        ts,
        tips_local,
        s_axes,
        s_centers,
        s_e1,
        s_e2,
        s_cos,
        s_sin,
        s_a,
        s_b,
        section_mask,
        shank_mask,
        w_normals,
        w_offsets,
    )
