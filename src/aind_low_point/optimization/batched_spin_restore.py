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
) -> Callable:
    """Build a batched spin-restore function.

    Returns ``restore(y) -> y`` where ``y: (B, n_vars)``. Each probe
    has its spin set to the value (out of ``n_spins`` linearly spaced
    over ``[-180, 180)``) that minimizes the batched objective with
    other probes' spins held fixed.

    ``n_rounds`` repeats the K-probe round-robin sweep so later probes
    can react to early probes' updated spins.
    """
    obj_fn, _ = make_batched_reduced_objective(static, weights)
    n_arcs = static.n_arcs
    K = static.K

    spin_grid = jnp.linspace(-180.0, 180.0, n_spins, endpoint=False).astype(jnp.float32)

    def _round_for_probe(y, i):
        """Sweep probe ``i``'s spin over the grid; update to argmin."""
        spin_idx = n_arcs + 2 * i + 1

        def eval_at_spin(s):
            y_new = y.at[:, spin_idx].set(s)
            return obj_fn(y_new, static)  # (B,)

        losses = jax.vmap(eval_at_spin)(spin_grid)  # (n_spins, B)
        best_idx = jnp.argmin(losses, axis=0)        # (B,)
        best_spin = spin_grid[best_idx]              # (B,)
        return y.at[:, spin_idx].set(best_spin)

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
    obj_batched, _ = make_batched_reduced_objective(probe_set_static, weights)
    obj_jit = obj_batched.from_arrays  # type: ignore[attr-defined]
    n_arcs = probe_set_static.n_arcs
    K = probe_set_static.K

    spin_grid = jnp.linspace(
        -180.0, 180.0, n_spins, endpoint=False
    ).astype(jnp.float32)

    def _round_for_probe(y, i, varying):
        spin_idx = n_arcs + 2 * i + 1

        def eval_at_spin(s):
            y_new = y.at[:, spin_idx].set(s)
            return obj_jit(y_new, *varying)

        losses = jax.vmap(eval_at_spin)(spin_grid)
        best_idx = jnp.argmin(losses, axis=0)
        best_spin = spin_grid[best_idx]
        return y.at[:, spin_idx].set(best_spin)

    def restore(y, *varying):
        for _ in range(n_rounds):
            for i in range(K):
                y = _round_for_probe(y, i, varying)
        return y

    return jax.jit(restore)
