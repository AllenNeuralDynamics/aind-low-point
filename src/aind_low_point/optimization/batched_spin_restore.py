"""Batched spin restoration for Stage 2 warm-start.

Adam needs a good initial spin per probe — the loss landscape is
multi-modal in spin (often with a 180° gap between basins, since
many configurations have "flip the probe" as a degenerate-feasible
neighbour). Starting from spin=0 for all probes leaves Adam stuck
in whatever basin is local to that point.

This module provides a JAX-batched round-robin spin sweep:

  for round in 1..n_rounds:
      for probe i in 1..K:
          for each candidate spin s in spin_grid:
              y_test = y.at[probe_i_spin].set(s)         # all-B same
              loss[s, b] = obj(y_test[b], static[b])
          best_spin[b] = argmin_s loss[s, b]
          y[probe_i_spin] = best_spin

K * n_rounds * n_spins objective evaluations per outer step, fully
vmapped across the batch. Much simpler than the existing pairwise
greedy sweep and good enough as a warm-start for Adam.

Phase 4 of the batched-Stage-2 refactor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.batched_objective import (
    _threading_g_for_probe,
    make_batched_reduced_objective,
)
from aind_low_point.optimization.batched_static import BatchedProbeStatic
from aind_low_point.optimization.joint_rerank import JointWeights
from aind_low_point.optimization.pipeline.contracts import (
    SpinRestoreFn,
    SpinRestoreWithLosses,
)
from aind_low_point.optimization.sdf_jax import (
    body_body_pair_clearance,
    body_shank_corners_pair_clearance,
    dual_rep_fixture_clearance,
    pose_from_optimizer_vars,
    shank_only_pair_clearance,
    spin_deg_from_sxy,
)


def make_batched_spin_restore_chunked(
    probe_set_static: BatchedProbeStatic,
    weights: JointWeights,
    *,
    n_spins: int = 8,
    n_rounds: int = 2,
    fixtures: tuple = (),
) -> SpinRestoreFn:
    """Build a chunkable spin-restore function.

    Returns ``restore(y, *varying_arrays) -> y`` where ``varying_arrays``
    is the tuple returned by ``obj_batched.extract_arrays(bs_chunk)``.
    Same-shape chunks reuse one JIT compile (vs. a closure-capture variant
    that bakes bs into the trace, forcing a full ~75s recompile per chunk).

    ``probe_set_static`` provides the probe-set constants (probe
    targets, pivots, shank tips, SDF tables) — these are identical
    across all candidates and are closure-captured. The per-candidate
    arrays (arc_idx, section geometry) flow as JIT runtime args.
    """
    obj_batched, _ = make_batched_reduced_objective(probe_set_static, weights, fixtures)
    obj_jit = obj_batched.from_arrays  # type: ignore[attr-defined]
    n_arcs = probe_set_static.n_arcs
    K = probe_set_static.K

    # Patch B: sweep (sx, sy) on the unit circle instead of scalar spin.
    spin_angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_spins, endpoint=False).astype(
        jnp.float32
    )
    spin_xy_grid = jnp.stack(
        [jnp.cos(spin_angles), jnp.sin(spin_angles)], axis=-1
    )  # (n_spins, 2)

    def _round_for_probe(y, i, varying):
        sx_idx = n_arcs + 3 * i + 1
        sy_idx = n_arcs + 3 * i + 2

        def eval_at_sxy(sxy):
            y_new = y.at[:, sx_idx].set(sxy[0]).at[:, sy_idx].set(sxy[1])
            return obj_jit(y_new, *varying)

        losses = jax.vmap(eval_at_sxy)(spin_xy_grid)
        best_idx = jnp.argmin(losses, axis=0)
        best_sxy = spin_xy_grid[best_idx]
        return y.at[:, sx_idx].set(best_sxy[:, 0]).at[:, sy_idx].set(best_sxy[:, 1])

    def restore(y, *varying):
        # Nested lax.fori_loop (both rounds AND probes) so the traced graph is
        # ONE copy of the per-probe body regardless of n_rounds*K — vs the old
        # Python double-loop which unrolled K*n_rounds copies (n_rounds=64 was
        # pathological). Round count is now a runtime trip count: free to raise.
        # ``i`` is traced, so _round_for_probe's column indices (n_arcs+3i+1/2)
        # are dynamic — a no-op vs static indices for in-bounds columns.
        def probe_body(i, yc):
            return _round_for_probe(yc, i, varying)

        def round_body(_r, yc):
            return jax.lax.fori_loop(0, K, probe_body, yc)

        return jax.lax.fori_loop(0, n_rounds, round_body, y)

    return jax.jit(restore)


def make_batched_spin_restore_partial(
    static: BatchedProbeStatic,
    weights: JointWeights,
    *,
    n_spins: int = 8,
    n_rounds: int = 4,
    fixtures: tuple = (),
) -> SpinRestoreWithLosses:
    """Partial/incremental spin restore: same round-robin coordinate descent,
    but each probe-``i`` spin sweep evaluates ONLY the terms that depend on
    probe ``i``'s spin (its threading + the K-1 clearance pairs incident to it
    + its fixture clearance) instead of the full O(K^2) objective. Everything
    else is additive-constant across the sweep, so the argmin is identical — see
    the ``spin_losses`` parity hook.

    Hoisted: the other probes' fixed poses/world-surfaces are built once per
    sweep; only probe ``i``'s pose/surface vary across the ``n_spins`` grid.
    Both loops are ``lax.fori_loop`` (dynamic probe index gathers the now-uniform
    per-kind SDF/OBB tables; the ``j==i`` self-pair is masked out).

    Same ``restore(y, *varying)`` calling convention as
    :func:`make_batched_spin_restore_chunked` (``varying`` from the reduced
    objective's ``extract_arrays``). Exposes ``.spin_losses(y, i, *varying)`` →
    ``(n_spins,)`` for a single candidate, for argmin-parity testing.
    """
    K = static.K
    n_arcs = static.n_arcs
    target_LPS = static.probe_target_lps[0]  # (K, 3)
    pivot_local = static.probe_pivot_local[0]  # (K, 3)
    shank_tips = static.probe_shank_tips[0]  # (K, SH, 3)
    shank_mask = static.probe_shank_mask[0].astype(jnp.float32)  # (K, SH)
    kind_np = np.asarray(static.sdf_kind_id[0])  # (K,) static
    kind_id = jnp.asarray(kind_np, jnp.int32)  # (K,) dyn gather
    sdf_grids = static.sdf_grids
    sdf_origins = static.sdf_origins
    sdf_spacings = static.sdf_spacings
    sdf_surface_points = static.sdf_surface_points
    obb_cen = static.sdf_shank_centers_padded  # (Nk, max_Sa, 3)
    obb_hlv = static.sdf_shank_halves_padded
    obb_msk = static.sdf_shank_obb_mask  # (Nk, max_Sa)

    lt = float(weights.lambda_thread)
    lc = float(weights.lambda_clearance)
    lf = float(getattr(weights, "lambda_clearance_fixture", weights.lambda_clearance))
    thr_tol = float(weights.threading_oval_tolerance)
    min_clear = float(weights.min_clearance_mm)
    fixture_data = [
        (
            jnp.asarray(fx.grid),
            jnp.asarray(fx.origin),
            jnp.asarray(fx.spacing),
            jnp.asarray(fx.surface),
        )
        for fx in fixtures
    ]
    spin_angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_spins, endpoint=False).astype(
        jnp.float32
    )
    spin_xy_grid = jnp.stack(
        [jnp.cos(spin_angles), jnp.sin(spin_angles)], axis=-1
    )  # (n_spins, 2)
    idxK = jnp.arange(K)

    def _pose_k(y, arc_aps, arc_idx, k):
        off = n_arcs + 3 * k
        spin = spin_deg_from_sxy(y[off + 1], y[off + 2])
        return pose_from_optimizer_vars(
            target_LPS=target_LPS[k],
            ap_deg=arc_aps[arc_idx[k]],
            ml_deg=y[off],
            spin_deg=spin,
            offset_R_mm=jnp.float32(0.0),
            offset_A_mm=jnp.float32(0.0),
            past_target_mm=jnp.float32(0.0),
            recording_center_local=pivot_local[k],
        )

    def _spin_losses(y, i, arc_idx, sections):
        arc_aps = y[:n_arcs]
        # Fixed state of ALL probes (hoisted; recomputed once per probe sweep).
        Rs, ts = [], []
        for k in range(K):
            R, t = _pose_k(y, arc_aps, arc_idx, k)
            Rs.append(R)
            ts.append(t)
        Rs = jnp.stack(Rs)
        ts = jnp.stack(ts)
        surfs = jnp.stack(
            [sdf_surface_points[int(kind_np[k])] @ Rs[k].T + ts[k] for k in range(K)]
        )  # (K, Nsurf, 3)
        arc_i = arc_aps[arc_idx[i]]
        ml_i = y[n_arcs + 3 * i]
        tgt_i, piv_i = target_LPS[i], pivot_local[i]
        ki = kind_id[i]
        grid_i, org_i, sp_i = sdf_grids[ki], sdf_origins[ki], sdf_spacings[ki]
        oc_i, oh_i, om_i = obb_cen[ki], obb_hlv[ki], obb_msk[ki]
        tips_i, smask_i = shank_tips[i], shank_mask[i]
        sec_i = tuple(s[i] for s in sections)

        def eval_spin(sxy):
            spin_i = spin_deg_from_sxy(sxy[0], sxy[1])
            R_i, t_i = pose_from_optimizer_vars(
                target_LPS=tgt_i,
                ap_deg=arc_i,
                ml_deg=ml_i,
                spin_deg=spin_i,
                offset_R_mm=jnp.float32(0.0),
                offset_A_mm=jnp.float32(0.0),
                past_target_mm=jnp.float32(0.0),
                recording_center_local=piv_i,
            )
            surf_i = sdf_surface_points[ki] @ R_i.T + t_i
            thr = _threading_g_for_probe(R_i, t_i, tips_i, smask_i, *sec_i, thr_tol)

            def cj(j):
                R_j, t_j, surf_j = Rs[j], ts[j], surfs[j]
                kj = kind_id[j]
                g_j, o_j, s_j = sdf_grids[kj], sdf_origins[kj], sdf_spacings[kj]
                oc_j, oh_j, om_j = obb_cen[kj], obb_hlv[kj], obb_msk[kj]
                sbb = body_body_pair_clearance(
                    R_i,
                    t_i,
                    R_j,
                    t_j,
                    grid_i,
                    org_i,
                    sp_i,
                    g_j,
                    o_j,
                    s_j,
                    surf_i,
                    surf_j,
                )[1]
                sbsc = body_shank_corners_pair_clearance(
                    R_i,
                    t_i,
                    R_j,
                    t_j,
                    grid_i,
                    org_i,
                    sp_i,
                    g_j,
                    o_j,
                    s_j,
                    oc_i,
                    oh_i,
                    oc_j,
                    oh_j,
                    shank_mask_a=om_i,
                    shank_mask_b=om_j,
                )[1]
                (_h1, sbso), (_h2, sss) = shank_only_pair_clearance(
                    R_i,
                    t_i,
                    R_j,
                    t_j,
                    surf_i,
                    surf_j,
                    oc_i,
                    oh_i,
                    oc_j,
                    oh_j,
                    shank_mask_a=om_i,
                    shank_mask_b=om_j,
                )
                cl = jnp.float32(0.0)
                for d in (sbb, sbsc, sbso, sss):
                    short = jnp.maximum(0.0, min_clear - d)
                    cl = cl + short * short
                return jnp.where(j != i, cl, jnp.float32(0.0))

            clall = jnp.sum(jax.vmap(cj)(idxK))
            fixv = jnp.float32(0.0)
            for fg, fo, fsp, fsurf in fixture_data:
                fc = dual_rep_fixture_clearance(
                    R_i,
                    t_i,
                    grid_i,
                    org_i,
                    sp_i,
                    fg,
                    fo,
                    fsp,
                    surf_i,
                    fsurf,
                    oc_i,
                    oh_i,
                )
                for d in (fc.body[1], fc.obb[1]):
                    short = jnp.maximum(0.0, min_clear - d)
                    fixv = fixv + short * short
            return lt * thr + lc * clall + lf * fixv

        return jax.vmap(eval_spin)(spin_xy_grid)  # (n_spins,)

    def _probe_body(y, i, arc_idx, sections):
        losses = _spin_losses(y, i, arc_idx, sections)
        best = spin_xy_grid[jnp.argmin(losses)]
        return y.at[n_arcs + 3 * i + 1].set(best[0]).at[n_arcs + 3 * i + 2].set(best[1])

    def restore_one(y, *varying):
        arc_idx, sections = varying[0], varying[1:]

        def round_body(_r, yc):
            return jax.lax.fori_loop(
                0, K, lambda i, yy: _probe_body(yy, i, arc_idx, sections), yc
            )

        return jax.lax.fori_loop(0, n_rounds, round_body, y)

    restore_v = jax.jit(jax.vmap(restore_one, in_axes=(0,) * 11))

    def restore_call(y, *varying):
        return restore_v(y, *varying)

    def spin_losses(y, i, *varying):
        """Single-candidate (n_spins,) partial losses for probe i — for the
        argmin-parity test against the full objective."""
        return _spin_losses(y, i, varying[0], varying[1:])

    restore_call.spin_losses = spin_losses  # type: ignore[attr-defined]
    return restore_call
