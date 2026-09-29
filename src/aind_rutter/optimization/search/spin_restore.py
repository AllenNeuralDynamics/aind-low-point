"""Batched spin restoration for reduced warm-starts.

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
vmapped across the batch. The loss is Phase 1's threading and clearance
penalty: the same spec, clearance query and gains, so the spin it picks is the
one Phase 1 would charge least for.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from aind_rutter.optimization.clearance.aggregate import (
    fixture_slack_gains,
    probe_pair_slack_gains,
)
from aind_rutter.optimization.clearance.poses import (
    pose_from_optimizer_vars,
    spin_deg_from_sxy,
)
from aind_rutter.optimization.clearance.sweep import (
    build_padded_fixture_table,
    pair_clearance,
    probe_fixture_clearance,
)
from aind_rutter.optimization.clearance.vec3 import matmul3
from aind_rutter.optimization.objectives.feasibility import FeasibilitySpec
from aind_rutter.optimization.objectives.layout import (
    ML,
    SPIN_COS,
    SPIN_SIN,
    reduced_block,
    reduced_var,
)
from aind_rutter.optimization.objectives.packing import BatchedProbeStatic
from aind_rutter.optimization.objectives.slacks import (
    fixture_clearance_slack,
    pair_clearance_slack,
    shortfall,
    threading_slack,
)
from aind_rutter.optimization.objectives.threading import threading_g_matrix
from aind_rutter.optimization.pipeline.records import (
    SpinRestoreWithLosses,
)


@dataclass(frozen=True)
class RestoreWeights:
    """What the spin restore charges: Phase 1's threading and clearance terms
    under ``spec``, at Phase 1's weights."""

    spec: FeasibilitySpec
    lambda_thread: float = 100.0
    lambda_clearance: float = 100.0
    lambda_clearance_fixture: float = 100.0


def _threading_penalty(
    R: jnp.ndarray,
    pose_tip: jnp.ndarray,
    tips_local: jnp.ndarray,  # (SH, 3)
    shank_mask: jnp.ndarray,  # (SH,)
    s_axes: jnp.ndarray,  # (S, 3)
    s_centers: jnp.ndarray,  # (S, 3)
    s_e1: jnp.ndarray,  # (S, 3)
    s_e2: jnp.ndarray,  # (S, 3)
    s_cos: jnp.ndarray,  # (S,)
    s_sin: jnp.ndarray,  # (S,)
    s_a: jnp.ndarray,  # (S,)
    s_b: jnp.ndarray,  # (S,)
    section_mask: jnp.ndarray,  # (S,)
    w_normals: jnp.ndarray,  # (W, 3)
    w_offsets: jnp.ndarray,  # (W,)
    spec: FeasibilitySpec,
    shank_half_xy: jnp.ndarray | None,  # (2,)
) -> jnp.ndarray:
    """Sum of squared threading shortfall over one probe's real (section, shank)
    entries."""
    g = threading_g_matrix(
        R,
        pose_tip,
        tips_local,
        s_axes,
        s_centers,
        s_e1,
        s_e2,
        s_cos,
        s_sin,
        s_a,
        s_b,
        float(spec.shaft_length_mm),
        w_normals=w_normals,
        w_offsets=w_offsets,
        shank_half_xy=shank_half_xy,
    )  # (S, SH)
    valid = section_mask[:, None] * shank_mask[None, :]
    excess = shortfall(threading_slack(g, spec))
    return jnp.sum(valid * excess * excess)


def _pair_penalty(probe_i, others, tables, spec, gains):
    """Sum of squared gained clearance shortfall over probe ``i``'s pairs, each
    queried in (lower, higher) probe order as Phase 1 does."""
    i, R_i, t_i, surf_i, kind_i = probe_i
    Rs, ts, surfs, kind_id = others

    def one(j):
        first = i < j
        _, soft = pair_clearance(
            jnp.where(first, R_i, Rs[j]),
            jnp.where(first, t_i, ts[j]),
            jnp.where(first, surf_i, surfs[j]),
            jnp.where(first, kind_i, kind_id[j]),
            jnp.where(first, Rs[j], R_i),
            jnp.where(first, ts[j], t_i),
            jnp.where(first, surfs[j], surf_i),
            jnp.where(first, kind_id[j], kind_i),
            tables,
            **spec.pair_softmin_kwargs(),
        )
        short = shortfall(pair_clearance_slack(soft, spec)) * gains
        return jnp.where(j != i, jnp.sum(short * short), jnp.float32(0.0))

    return jnp.sum(jax.vmap(one)(jnp.arange(Rs.shape[0])))


def _fixture_penalty(probe_i, tables, fix_table, spec, gains):
    """Sum of squared gained clearance shortfall of probe ``i`` against every
    fixture."""
    _i, R_i, t_i, surf_i, kind_i = probe_i

    def one(fg, fo, fs, fsurf, fnr):
        _, soft = probe_fixture_clearance(
            R_i,
            t_i,
            surf_i,
            kind_i,
            tables,
            fg,
            fo,
            fs,
            fsurf,
            fnr,
            **spec.fixture_softmin_kwargs(),
        )
        short = shortfall(fixture_clearance_slack(soft, spec)) * gains
        return jnp.sum(short * short)

    return jnp.sum(
        jax.vmap(one)(
            fix_table["grids"],
            fix_table["origins"],
            fix_table["spacings"],
            fix_table["surfaces"],
            fix_table["real_shapes"],
        )
    )


def make_batched_spin_restore_partial(
    static: BatchedProbeStatic,
    weights: RestoreWeights,
    *,
    n_spins: int = 8,
    n_rounds: int = 4,
    tables: dict | None = None,
    fixtures: tuple = (),
) -> SpinRestoreWithLosses:
    """Partial/incremental spin restore: same round-robin coordinate descent,
    but each probe-``i`` spin sweep evaluates ONLY the terms that depend on
    probe ``i``'s spin (its threading + the K-1 clearance pairs incident to it
    + its fixture clearance) instead of the full O(K^2) objective. Everything
    else is additive-constant across the sweep, so the argmin is identical — see
    the ``spin_losses`` parity hook.

    ``tables`` is Phase 1's swept-pair table
    (:func:`~aind_rutter.optimization.clearance.sweep.build_padded_probe_tables`),
    at Phase 1's grid dtype; without it the loss is threading alone, and
    ``fixtures`` need it. Each pair is queried in Phase 1's (lower, higher)
    probe order, so ties in the query's cell selection break as they do there.

    Hoisted: the other probes' fixed poses/world-surfaces are built once per
    sweep; only probe ``i``'s pose/surface vary across the ``n_spins`` grid.
    Both loops are ``lax.fori_loop`` (dynamic probe index gathers the per-kind
    tables; the ``j==i`` self-pair is masked out).

    ``varying`` is :func:`~aind_rutter.optimization.objectives.packing.restore_arrays`.
    Exposes ``.spin_losses(y, i, *varying)`` →
    ``(n_spins,)`` for a single candidate, for argmin-parity testing.
    """
    if fixtures and tables is None:
        raise ValueError("fixture clearance needs the probe tables")
    K = static.K
    n_arcs = static.n_arcs
    target_LPS = static.probe_target_lps[0]  # (K, 3)
    pivot_local = static.probe_pivot_local[0]  # (K, 3)
    shank_tips = static.probe_shank_tips[0]  # (K, SH, 3)
    shank_mask = static.probe_shank_mask[0].astype(jnp.float32)  # (K, SH)
    half_xy = static.probe_shank_half_xy
    shank_half_xy = None if half_xy is None else half_xy[0]  # (K, 2)

    spec = weights.spec
    lt = float(weights.lambda_thread)
    lc = float(weights.lambda_clearance)
    lcf = float(weights.lambda_clearance_fixture)
    pair_gains = jnp.asarray(probe_pair_slack_gains(), jnp.float32)
    fixture_gains = jnp.asarray(fixture_slack_gains(), jnp.float32)
    fix_table = build_padded_fixture_table(fixtures)
    spin_angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_spins, endpoint=False).astype(
        jnp.float32
    )
    spin_xy_grid = jnp.stack(
        [jnp.cos(spin_angles), jnp.sin(spin_angles)], axis=-1
    )  # (n_spins, 2)

    def _pose_k(y, arc_aps, arc_idx, k):
        off = reduced_block(n_arcs, k)
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

    def _world_surface(kind, R, t):
        return matmul3(tables["surfaces"][kind], R.T) + t

    def _spin_losses(y, i, arc_idx, sections):
        arc_aps = y[:n_arcs]
        arc_i = arc_aps[arc_idx[i]]
        ml_i = y[reduced_var(n_arcs, i, ML)]
        tgt_i, piv_i = target_LPS[i], pivot_local[i]
        tips_i, smask_i = shank_tips[i], shank_mask[i]
        sec_i = tuple(s[i] for s in sections)
        if tables is not None:
            # Fixed state of ALL probes (hoisted; recomputed once per probe sweep).
            poses = [_pose_k(y, arc_aps, arc_idx, k) for k in range(K)]
            Rs = jnp.stack([R for R, _ in poses])
            ts = jnp.stack([t for _, t in poses])
            kind_id = tables["kind_id"]
            surfs = jax.vmap(_world_surface)(kind_id, Rs, ts)  # (K, Nsurf, 3)
            kind_i = kind_id[i]

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
            loss = lt * _threading_penalty(
                R_i,
                t_i,
                tips_i,
                smask_i,
                *sec_i,
                spec,
                None if shank_half_xy is None else shank_half_xy[i],
            )
            if tables is None:
                return loss
            surf_i = _world_surface(kind_i, R_i, t_i)
            probe_i = (i, R_i, t_i, surf_i, kind_i)
            loss = loss + lc * _pair_penalty(
                probe_i, (Rs, ts, surfs, kind_id), tables, spec, pair_gains
            )
            if fix_table:
                loss = loss + lcf * _fixture_penalty(
                    probe_i, tables, fix_table, spec, fixture_gains
                )
            return loss

        return jax.vmap(eval_spin)(spin_xy_grid)  # (n_spins,)

    def _probe_body(y, i, arc_idx, sections):
        losses = _spin_losses(y, i, arc_idx, sections)
        best = spin_xy_grid[jnp.argmin(losses)]
        off = reduced_block(n_arcs, i)
        return y.at[off + SPIN_COS].set(best[0]).at[off + SPIN_SIN].set(best[1])

    def restore_one(y, *varying):
        arc_idx, sections = varying[0], varying[1:]

        def round_body(_r, yc):
            return jax.lax.fori_loop(
                0, K, lambda i, yy: _probe_body(yy, i, arc_idx, sections), yc
            )

        return jax.lax.fori_loop(0, n_rounds, round_body, y)

    restore_v = jax.jit(jax.vmap(restore_one))

    def restore_call(y, *varying):
        return restore_v(y, *varying)

    def spin_losses(y, i, *varying):
        """Single-candidate (n_spins,) partial losses for probe i — for the
        argmin-parity test against the full objective."""
        return _spin_losses(y, i, varying[0], varying[1:])

    restore_call.spin_losses = spin_losses  # type: ignore[attr-defined]
    return restore_call
