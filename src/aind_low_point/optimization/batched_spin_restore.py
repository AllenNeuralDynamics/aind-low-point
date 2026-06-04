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

from typing import Callable

import jax
import jax.numpy as jnp

from aind_low_point.optimization.batched_objective import (
    make_batched_reduced_objective,
)
from aind_low_point.optimization.batched_static import BatchedProbeStatic
from aind_low_point.optimization.joint_rerank import JointWeights


def make_batched_spin_restore(
    static: BatchedProbeStatic,
    weights: JointWeights,
    *,
    n_spins: int = 8,
    n_rounds: int = 2,
    fixtures: tuple = (),
) -> Callable:
    """Build a batched spin-restore function.

    Returns ``restore(y) -> y`` where ``y: (B, n_vars)``. Each probe
    has its spin set to the value (out of ``n_spins`` linearly spaced
    over ``[-180, 180)``) that minimizes the batched objective with
    other probes' spins held fixed.

    ``n_rounds`` repeats the K-probe round-robin sweep so later probes
    can react to early probes' updated spins.

    ``fixtures`` (e.g. the well-bore SDF) are folded into the objective so
    the spin-basin argmin accounts for probe-vs-fixture clearance — without
    it the basin ranking is uncorrelated with FCL feasibility.
    """
    obj_fn, _ = make_batched_reduced_objective(static, weights, fixtures)
    n_arcs = static.n_arcs
    K = static.K

    # Patch B: spin is parameterized as (sx, sy) on the unit circle. The
    # spin sweep evaluates n_spins points around the circle, replacing
    # (sx, sy) with (cos θ_k, sin θ_k).
    spin_angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_spins, endpoint=False).astype(
        jnp.float32
    )
    spin_xy_grid = jnp.stack(
        [jnp.cos(spin_angles), jnp.sin(spin_angles)], axis=-1
    )  # (n_spins, 2)

    def _round_for_probe(y, i):
        """Sweep probe ``i``'s (sx, sy) over the unit-circle grid; update to argmin."""
        sx_idx = n_arcs + 3 * i + 1
        sy_idx = n_arcs + 3 * i + 2

        def eval_at_sxy(sxy):
            y_new = y.at[:, sx_idx].set(sxy[0]).at[:, sy_idx].set(sxy[1])
            return obj_fn(y_new, static)  # (B,)

        losses = jax.vmap(eval_at_sxy)(spin_xy_grid)  # (n_spins, B)
        best_idx = jnp.argmin(losses, axis=0)          # (B,)
        best_sxy = spin_xy_grid[best_idx]              # (B, 2)
        return (
            y.at[:, sx_idx].set(best_sxy[:, 0])
            .at[:, sy_idx].set(best_sxy[:, 1])
        )

    def restore(y):
        for _ in range(n_rounds):
            for i in range(K):
                y = _round_for_probe(y, i)
        return y

    return jax.jit(restore)


def make_batched_spin_restore_chunked(
    probe_set_static: BatchedProbeStatic,
    weights: JointWeights,
    *,
    n_spins: int = 8,
    n_rounds: int = 2,
    fixtures: tuple = (),
) -> Callable:
    """Build a chunkable spin-restore function.

    Returns ``restore(y, *varying_arrays) -> y`` where ``varying_arrays``
    is the tuple returned by ``obj_batched.extract_arrays(bs_chunk)``.
    Same-shape chunks reuse one JIT compile (vs. the closure-capture
    variant in :func:`make_batched_spin_restore` which bakes bs into
    the trace, forcing a full ~75s recompile per chunk).

    ``probe_set_static`` provides the probe-set constants (probe
    targets, pivots, shank tips, SDF tables) — these are identical
    across all candidates and are closure-captured. The per-candidate
    arrays (arc_idx, section geometry) flow as JIT runtime args.
    """
    obj_batched, _ = make_batched_reduced_objective(
        probe_set_static, weights, fixtures
    )
    obj_jit = obj_batched.from_arrays  # type: ignore[attr-defined]
    n_arcs = probe_set_static.n_arcs
    K = probe_set_static.K

    # Patch B: sweep (sx, sy) on the unit circle instead of scalar spin.
    spin_angles = jnp.linspace(
        0.0, 2.0 * jnp.pi, n_spins, endpoint=False
    ).astype(jnp.float32)
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
        return (
            y.at[:, sx_idx].set(best_sxy[:, 0])
            .at[:, sy_idx].set(best_sxy[:, 1])
        )

    def restore(y, *varying):
        for _ in range(n_rounds):
            for i in range(K):
                y = _round_for_probe(y, i, varying)
        return y

    return jax.jit(restore)
