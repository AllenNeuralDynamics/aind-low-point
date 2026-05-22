"""JAX-SDF spin-only feasibility restoration.

Replaces the FCL-BVH ``_spin_restore_starts`` brute sweep
(48% of Stage 2 wall in the strict-clearance profile run) with a
vmapped pairwise-SDF kernel:

  - Per outer iteration: find worst pair by computing all 21 pairwise
    signed clearances in one batched JAX call.
  - For the worst pair (a, b): coarse sweep on a ``G×G`` spin grid via
    ``jax.vmap`` over (spin_a, spin_b) — all G² × n_pairs SDF lookups
    run in one GPU launch. Refine 4×4 around the coarse winner.

Same module-level JIT cache as Stage 2: signature keys on probe-set,
per-probe SDF grid shapes, and grid size. The probe-set is fixed for
a given ``optimize_joint`` run, so the second-onwards (H, A) candidate
reuses the same compiled XLA program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.sdf_jax import (
    pairwise_signed_clearance,
    pose_from_optimizer_vars,
)

_JIT_CACHE: dict[Hashable, "_SpinRestoreFns"] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def cache_stats() -> dict:
    return {**_CACHE_STATS, "entries": len(_JIT_CACHE)}


@dataclass(frozen=True)
class _SpinRestoreFns:
    all_clearances: Callable  # (y) -> (n_pairs,) signed clearances
    spin_sweep: Callable  # (y, idx_a, idx_b, spins_a, spins_b) -> (G_a, G_b)


def _signature(statics, n_arcs: int) -> tuple:
    """Cache key: probe set, per-probe SDF grid shape, surface count."""
    per_probe_sdf_shapes = []
    for s in statics:
        if s.sdf_data is None:
            per_probe_sdf_shapes.append(None)
        else:
            per_probe_sdf_shapes.append(
                tuple(int(x) for x in np.asarray(s.sdf_data["grid"]).shape)
            )
    n_surf = 0
    for s in statics:
        if s.sdf_data is not None:
            n_surf = int(np.asarray(s.sdf_data["surface"]).shape[0])
            break
    return (
        len(statics),
        int(n_arcs),
        tuple(per_probe_sdf_shapes),
        n_surf,
    )


def _build_jit(sig: tuple, n_probes: int, n_arcs: int) -> _SpinRestoreFns:
    """Construct the JIT'd ``all_clearances`` + ``spin_sweep`` kernels."""
    # Pair list (Python-static; only pairs where both probes have SDF).
    _, _, per_probe_sdf_shapes, _ = sig
    pair_list: list[tuple[int, int]] = []
    for i in range(n_probes):
        if per_probe_sdf_shapes[i] is None:
            continue
        for j in range(i + 1, n_probes):
            if per_probe_sdf_shapes[j] is None:
                continue
            pair_list.append((i, j))

    def _poses(y, target_LPS, pivot_local, arc_idx):
        """Compute (R, t) for every probe at reduced ``y``."""
        arc_aps = y[:n_arcs]
        Rs, ts = [], []
        for i in range(n_probes):
            off = n_arcs + 2 * i
            ml = y[off + 0]
            spin = y[off + 1]
            ap = arc_aps[arc_idx[i]]
            R, t = pose_from_optimizer_vars(
                target_LPS=target_LPS[i],
                ap_deg=ap, ml_deg=ml, spin_deg=spin,
                offset_R_mm=jnp.float32(0.0),
                offset_A_mm=jnp.float32(0.0),
                past_target_mm=jnp.float32(0.0),
                recording_center_local=pivot_local[i],
            )
            Rs.append(R)
            ts.append(t)
        return Rs, ts

    def all_clearances(
        y,
        target_LPS, pivot_local, arc_idx,
        sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces,
    ):
        Rs, ts = _poses(y, target_LPS, pivot_local, arc_idx)
        clears = []
        for ia, ib in pair_list:
            d = pairwise_signed_clearance(
                Rs[ia], ts[ia], Rs[ib], ts[ib],
                sdf_grids[ia], sdf_origins[ia], sdf_spacings[ia],
                sdf_grids[ib], sdf_origins[ib], sdf_spacings[ib],
                sdf_surfaces[ia], sdf_surfaces[ib],
            )
            clears.append(d)
        return jnp.stack(clears)  # (n_pairs,)

    def _min_clearance_at_spin(
        y, idx_a, idx_b, sa, sb,
        target_LPS, pivot_local, arc_idx,
        sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces,
    ):
        """Set y[idx_a] = sa, y[idx_b] = sb, return min signed clearance
        over all pairs. ``idx_a, idx_b`` are y-indices of the spin
        components."""
        y = y.at[idx_a].set(sa).at[idx_b].set(sb)
        clears = all_clearances(
            y, target_LPS, pivot_local, arc_idx,
            sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces,
        )
        return jnp.min(clears)

    def spin_sweep(
        y, idx_a, idx_b, spins_a, spins_b,
        target_LPS, pivot_local, arc_idx,
        sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces,
    ):
        """Vectorise ``_min_clearance_at_spin`` over the spin grid.

        Returns ``(G_a, G_b)`` matrix of min clearances; argmax of this
        is the best spin combination."""
        # vmap inner (over spins_b) then outer (over spins_a)
        def per_a(sa):
            def per_b(sb):
                return _min_clearance_at_spin(
                    y, idx_a, idx_b, sa, sb,
                    target_LPS, pivot_local, arc_idx,
                    sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces,
                )
            return jax.vmap(per_b)(spins_b)
        return jax.vmap(per_a)(spins_a)

    return _SpinRestoreFns(
        all_clearances=jax.jit(all_clearances),
        spin_sweep=jax.jit(spin_sweep),
    )


def _pack(statics, n_arcs: int) -> dict:
    """Pack runtime args (per-probe arrays) from statics."""
    P = len(statics)
    target_LPS = jnp.stack(
        [jnp.asarray(s.target_LPS, dtype=jnp.float32) for s in statics]
    )
    pivot_local = jnp.stack(
        [jnp.asarray(s.pivot_local, dtype=jnp.float32) for s in statics]
    )
    arc_idx = jnp.asarray([s.arc_idx for s in statics], dtype=jnp.int32)
    sdf_grids = tuple(s.sdf_data["grid"] for s in statics)
    sdf_origins = tuple(s.sdf_data["origin"] for s in statics)
    sdf_spacings = tuple(s.sdf_data["spacing"] for s in statics)
    sdf_surfaces = tuple(s.sdf_data["surface"] for s in statics)
    return dict(
        target_LPS=target_LPS, pivot_local=pivot_local, arc_idx=arc_idx,
        sdf_grids=sdf_grids, sdf_origins=sdf_origins,
        sdf_spacings=sdf_spacings, sdf_surfaces=sdf_surfaces,
    )


def spin_restore_jax(
    y: NDArray,
    statics,
    n_arcs: int,
    *,
    coarse_grid: int = 4,
    fine_grid: int = 4,
    fine_window_deg: float = 45.0,
    max_outer_iter: int = 3,
    margin_mm: float = 0.02,
) -> NDArray:
    """JAX-SDF spin restoration.

    Under Patch B (sx, sy) layout this is currently a no-op
    pass-through. The batched spin-restore path
    (:func:`make_batched_spin_restore_chunked` in
    :mod:`batched_spin_restore`) handles spin initialisation under the
    new layout. The per-cand 2D sweep here would need a 4D sweep
    (sx_a, sy_a, sx_b, sy_b) which is not yet implemented.

    Probes without SDF data fall back to the FCL caller (the reduced
    SLSQP path requires SDF anyway, so this is the only case)."""
    has_sdf = all(s.sdf_data is not None for s in statics)
    if not has_sdf or len(statics) < 2:
        return np.asarray(y, dtype=np.float64)
    # No-op under (sx, sy) layout — see docstring.
    return np.asarray(y, dtype=np.float64)
    # pylint: disable=unreachable

    sig = _signature(statics, n_arcs)
    if sig not in _JIT_CACHE:
        _JIT_CACHE[sig] = _build_jit(sig, len(statics), n_arcs)
        _CACHE_STATS["misses"] += 1
    else:
        _CACHE_STATS["hits"] += 1
    fns = _JIT_CACHE[sig]
    packed = _pack(statics, n_arcs)
    pair_list = [(i, j) for i in range(len(statics)) for j in range(i + 1, len(statics))]

    y = np.asarray(y, dtype=np.float64).copy()
    coarse = jnp.linspace(-180.0, 180.0, coarse_grid, endpoint=False)

    for _ in range(max_outer_iter):
        y_j = jnp.asarray(y, dtype=jnp.float32)
        clears = fns.all_clearances(y_j, **packed)
        clears_np = np.asarray(clears)
        worst_idx = int(np.argmin(clears_np))
        worst_d = float(clears_np[worst_idx])
        if worst_d > margin_mm:
            return y
        worst_a, worst_b = pair_list[worst_idx]
        idx_a_y = n_arcs + 2 * worst_a + 1
        idx_b_y = n_arcs + 2 * worst_b + 1

        # Coarse sweep
        scores = fns.spin_sweep(
            y_j, idx_a_y, idx_b_y, coarse, coarse, **packed,
        )
        scores_np = np.asarray(scores)
        ia_c, ib_c = np.unravel_index(int(np.argmax(scores_np)), scores_np.shape)
        coarse_a = float(coarse[ia_c])
        coarse_b = float(coarse[ib_c])
        coarse_best = float(scores_np[ia_c, ib_c])

        # Fine sweep in a window around the coarse winner
        fine_a = jnp.linspace(
            coarse_a - fine_window_deg, coarse_a + fine_window_deg, fine_grid
        )
        fine_b = jnp.linspace(
            coarse_b - fine_window_deg, coarse_b + fine_window_deg, fine_grid
        )
        scores = fns.spin_sweep(
            y_j, idx_a_y, idx_b_y, fine_a, fine_b, **packed,
        )
        scores_np = np.asarray(scores)
        ia_f, ib_f = np.unravel_index(int(np.argmax(scores_np)), scores_np.shape)
        fine_best = float(scores_np[ia_f, ib_f])

        if fine_best >= coarse_best:
            best_sa = float(fine_a[ia_f])
            best_sb = float(fine_b[ib_f])
        else:
            best_sa, best_sb = coarse_a, coarse_b

        y[idx_a_y] = ((best_sa + 180.0) % 360.0) - 180.0
        y[idx_b_y] = ((best_sb + 180.0) % 360.0) - 180.0

    return y
