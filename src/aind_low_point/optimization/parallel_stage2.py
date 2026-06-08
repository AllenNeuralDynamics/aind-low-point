"""Multiprocess scipy SLSQP wrapper for Stage 2 polish.

Mirrors the Stage 3 ``_inner_solve_worker`` ProcessPool pattern used in
``joint_rerank.py``. Each worker polishes one ``(HA, AA)`` candidate
through scipy SLSQP via ``score_joint`` and returns a ``JointCandidate``.

Workers share the same probes / holes / SDFs / pose_features (passed at
init time and cached in module-global state). Each worker rebuilds its
own BVH cache (FCL CollisionObjects can't cross process boundaries).

Designed in 2026-05-20 conversation as the load-bearing path for the
new architecture: arc-first emits ~10-20K deduped candidates; this
module polishes them in parallel. Stage 2 polish is the trusted quality
evaluator (cell-static scoring was empirically falsified).

Usage::

    from aind_low_point.optimization.parallel_stage2 import polish_all

    joint_cands = polish_all(
        candidates,                # list of (HA, AA) from arc-first
        probes, holes_list, pose_features,
        sdf_by_name=sdf_by_name,
        weights=JointWeights(min_arc_ap_sep_deg=16.0),
        n_workers=8,
    )
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator

import numpy as np

from aind_low_point.optimization.arc_assignment import ArcAssignment
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.joint_rerank import (
    JointCandidate,
    JointWeights,
    score_joint,
)
from aind_low_point.optimization.optimize import ProbeStaticInfo
from aind_low_point.optimization.pose_features import PoseFeatures

# Module-global state populated by worker init. Workers reuse this
# across multiple candidates; pickling cost is paid once per worker.
_W: dict = {}


def _worker_init(
    probes,
    holes,
    pose_features,
    sdf_by_name,
    weights,
    head_pitch_deg,
    reduced_slsqp_max_iter,
    skip_spin_restore,
) -> None:
    """Worker process init. Pins JAX to CPU (matches the Stage 3 pattern;
    prevents spawn-mode CUDA contention if any kernel imports JAX).

    Builds a per-worker BVH cache so ``_build_probe_static`` doesn't
    rebuild ~100ms FCL BVH per probe per candidate (was ~600ms/cand
    waste on a 7-probe rig in the 2026-05-20 profile)."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    bvh_cache = {
        p.name: (
            make_fcl_bvh(p.collision_mesh) if p.collision_mesh is not None else None
        )
        for p in probes
    }
    global _W
    _W = dict(
        probes=probes,
        holes=holes,
        pose_features=pose_features,
        sdf_by_name=sdf_by_name,
        weights=weights,
        head_pitch_deg=head_pitch_deg,
        reduced_slsqp_max_iter=reduced_slsqp_max_iter,
        skip_spin_restore=skip_spin_restore,
        bvh_cache=bvh_cache,
    )


def _worker_polish(
    payload: tuple,
) -> tuple[int, JointCandidate]:
    """Polish one (HA, AA) candidate. Returns ``(idx, jc)``.

    Payload is either ``(idx, ha, aa, lsap_cost)`` or
    ``(idx, ha, aa, lsap_cost, y0_override_bytes)`` where the last
    element is a numpy-pickled override of the warm-start y vector.
    """
    if len(payload) == 5:
        idx, ha, aa, lsap_cost, y0_override = payload
    else:
        idx, ha, aa, lsap_cost = payload
        y0_override = None
    jc = score_joint(
        ha,
        aa,
        _W["probes"],
        _W["holes"],
        _W["pose_features"],
        weights=_W["weights"],
        head_pitch_deg=_W["head_pitch_deg"],
        reduced_slsqp_max_iter=_W["reduced_slsqp_max_iter"],
        original_lsap_cost=lsap_cost,
        sdf_by_name=_W["sdf_by_name"],
        bvh_cache=_W["bvh_cache"],
        y0_override=y0_override,
        skip_spin_restore=_W["skip_spin_restore"],
    )
    return idx, jc


@contextlib.contextmanager
def polish_worker_pool(
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    pose_features: dict[tuple[str, int], PoseFeatures],
    *,
    weights: JointWeights = JointWeights(),
    head_pitch_deg: float = 0.0,
    reduced_slsqp_max_iter: int = 50,
    sdf_by_name: dict | None = None,
    n_workers: int | None = None,
    spawn: bool = True,
    skip_spin_restore: bool = False,
) -> Iterator[ProcessPoolExecutor]:
    """Build a polish worker pool once and yield it.

    Reusing one pool across multiple polish phases avoids re-paying
    spawn cost + JIT cache load + BVH cache rebuild per phase. The init
    args here MUST match what every call to ``polish_all`` would otherwise
    pass — same probes/holes/SDFs/weights/skip_spin_restore. Pin
    JAX_PLATFORMS=cpu in the parent before spawn so workers inherit a
    CPU-only JAX (avoids contending with the parent's GPU runtime).
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 2)
    ctx = mp.get_context("spawn" if spawn else "fork")
    init_args = (
        probes,
        holes,
        pose_features,
        sdf_by_name,
        weights,
        head_pitch_deg,
        reduced_slsqp_max_iter,
        skip_spin_restore,
    )
    prev_jax_platforms = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            yield pool
    finally:
        if prev_jax_platforms is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = prev_jax_platforms


def polish_all(
    candidates: list[tuple[HoleAssignment, ArcAssignment, float]],
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    pose_features: dict[tuple[str, int], PoseFeatures],
    *,
    weights: JointWeights = JointWeights(),
    head_pitch_deg: float = 0.0,
    reduced_slsqp_max_iter: int = 50,
    sdf_by_name: dict | None = None,
    n_workers: int | None = None,
    spawn: bool = True,
    progress_every: int = 50,
    y0_per_candidate=None,
    skip_spin_restore: bool = False,
    executor: ProcessPoolExecutor | None = None,
    verbose: bool = False,
) -> list[JointCandidate]:
    """Polish ``candidates`` in parallel via scipy SLSQP.

    Parameters
    ----------
    candidates : list of (ha, aa, lsap_cost)
        Each entry is one Stage 2 input. ``lsap_cost`` is stored in
        the returned ``JointRerankMetrics.original_lsap_cost`` for
        diagnostic continuity (use ``float("nan")`` if not applicable).
    probes, holes, pose_features, sdf_by_name : as in ``score_joint``
    weights, head_pitch_deg, reduced_slsqp_max_iter : as in
        ``score_joint``.
    n_workers : int or None
        Default = ``cpu_count() - 2``, clamped to ``[1, n_candidates]``.
    spawn : bool
        True (default) uses ``mp.get_context("spawn")``; safer in
        repeated invocations from scripts that may have imported JAX.
        Fork is faster on Linux but inherits parent state (CUDA
        contexts in particular).
    progress_every : int
        Print one progress line per ``progress_every`` polished
        candidates when ``verbose=True``.
    """
    n = len(candidates)
    if n == 0:
        return []

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 2)
    n_workers = max(1, min(n_workers, n))

    if verbose:
        print(
            f"[parallel_stage2] polishing {n} candidates on {n_workers} workers "
            f"({'spawn' if spawn else 'fork'} mode)"
        )

    init_args = (
        probes,
        holes,
        pose_features,
        sdf_by_name,
        weights,
        head_pitch_deg,
        reduced_slsqp_max_iter,
        skip_spin_restore,
    )

    # Payloads: (idx, ha, aa, lsap_cost[, y0_override])
    if y0_per_candidate is not None:
        if len(y0_per_candidate) != n:
            raise ValueError("y0_per_candidate length must match candidates")
        payloads = [
            (i, ha, aa, lsap_cost, y0_per_candidate[i])
            for i, (ha, aa, lsap_cost) in enumerate(candidates)
        ]
    else:
        payloads = [
            (i, ha, aa, lsap_cost) for i, (ha, aa, lsap_cost) in enumerate(candidates)
        ]

    results: list[JointCandidate | None] = [None] * n
    t0 = time.perf_counter()

    if n_workers == 1 and executor is None:
        _worker_init(*init_args)
        for k, payload in enumerate(payloads):
            idx, jc = _worker_polish(payload)
            results[idx] = jc
            if verbose and (k + 1) % progress_every == 0:
                elapsed = time.perf_counter() - t0
                rate = (k + 1) / elapsed
                eta = (n - k - 1) / rate
                print(
                    f"  {k + 1:>5}/{n}  ({rate:.1f} cands/s, ETA {eta:.0f}s)",
                    flush=True,
                )
    else:
        # If a pre-built executor is supplied, reuse it (the caller is
        # responsible for the JAX_PLATFORMS pin via ``polish_worker_pool``).
        # Otherwise build a one-shot pool for this call.
        if executor is not None:
            pool_ctx = contextlib.nullcontext(executor)
            jax_pin_ctx = contextlib.nullcontext()
        else:
            ctx = mp.get_context("spawn" if spawn else "fork")

            # Workers inherit parent env at spawn time. Force them into CPU JAX
            # so they don't fight over the GPU when importing joint_rerank.
            @contextlib.contextmanager
            def _pin_jax_cpu() -> Iterator[None]:
                prev = os.environ.get("JAX_PLATFORMS")
                os.environ["JAX_PLATFORMS"] = "cpu"
                try:
                    yield
                finally:
                    if prev is None:
                        os.environ.pop("JAX_PLATFORMS", None)
                    else:
                        os.environ["JAX_PLATFORMS"] = prev

            jax_pin_ctx = _pin_jax_cpu()
            pool_ctx = ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=init_args,
            )
        with jax_pin_ctx, pool_ctx as pool:
            done = 0
            # chunksize=16: tasks have fairly uniform runtime (~100ms each),
            # so batching reduces IPC overhead (~few ms / task on chunksize=1)
            # without imbalancing workers. At 8908 cands × 16 chunksize = 557
            # dispatches across 8 workers = 70 chunks/worker, plenty fine-grained.
            for idx, jc in pool.map(_worker_polish, payloads, chunksize=16):
                results[idx] = jc
                done += 1
                if verbose and done % progress_every == 0:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed
                    eta = (n - done) / rate if rate > 0 else float("inf")
                    print(
                        f"  {done:>5}/{n}  ({rate:.1f} cands/s, ETA {eta:.0f}s)",
                        flush=True,
                    )

    elapsed = time.perf_counter() - t0
    if verbose:
        print(
            f"[parallel_stage2] done: {elapsed:.1f}s "
            f"({elapsed / n * 1000:.0f}ms/cand, "
            f"{n * n_workers / elapsed:.1f}x speedup vs ideal)"
        )

    return [r for r in results if r is not None]  # type: ignore


# ---------------------------------------------------------------------------
# Orchestrator: batched-spin-restore + multiproc-SLSQP
# ---------------------------------------------------------------------------


def estimate_spin_restore_chunk(
    *,
    n_surf: int = 5000,
    K: int = 7,
    n_spins: int = 8,
    total_B: int,
    safety_margin: float = 0.2,
    fallback: int = 100,
) -> int:
    """Estimate a safe per-call chunk size from free GPU memory.

    The peak allocation during batched spin restore scales with
    ``B × K × n_surf × n_spins`` (transformed surface points + grad
    intermediates). Empirically ~70 MB/cand at K=7, n_surf=20000,
    n_spins=8. We scale linearly in (n_surf × K × n_spins).

    Budget = ``free_VRAM − safety_margin × total_VRAM``. Reserving a
    fixed fraction of *total* VRAM (rather than scaling with free)
    keeps the headroom stable when other processes grow/shrink during
    the run.

    Clamps the resulting chunk to the actual batch size ``total_B``.
    Falls back to ``fallback`` if pynvml is unavailable (CPU-only
    runs, or driverless containers) or the budget comes out negative.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_bytes = int(info.free)
        total_bytes = int(info.total)
        pynvml.nvmlShutdown()
    except Exception:
        return min(fallback, total_B)
    budget = free_bytes - safety_margin * total_bytes
    if budget <= 0:
        return min(fallback, total_B)
    # Empirical bytes/cand, scaled by the four dimensions that drive
    # the peak allocation. Re-calibrated 2026-05-22 after batched path
    # picked up dual-rep clearance + shank-shank SAT (task #58): a
    # chunk=196 run at n_surf=5000 OOM'd needing another 4 GB → actual
    # ~50 MB/cand. Base bumped 140e6 → 360e6 to cover SAT cross-axis
    # intermediates + dual-direction body-shank sampling, plus 60%
    # headroom.
    mem_per_cand = 360e6 * (n_surf / 20000.0) * (K / 7.0) * (n_spins / 8.0)
    chunk = int(budget / max(mem_per_cand, 1.0))
    return max(1, min(chunk, total_B))


def polish_all_with_batched_spin_restore(
    candidates,  # list of ArcFirstCandidate
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    pose_features: dict[tuple[str, int], PoseFeatures],
    *,
    weights: JointWeights = JointWeights(),
    head_pitch_deg: float = 0.0,
    reduced_slsqp_max_iter: int = 50,
    sdf_by_name: dict | None = None,
    n_workers: int | None = None,
    n_arcs: int = 3,
    n_spins_restore: int = 8,
    spin_restore_rounds: int = 2,
    spin_restore_chunk: int = 100,
    spawn: bool = True,
    progress_every: int = 50,
    executor: ProcessPoolExecutor | None = None,
    verbose: bool = False,
) -> list[JointCandidate]:
    """Polish a list of ``ArcFirstCandidate`` with the spin-restore step
    batched over GPU.

    Pipeline:
      1. Build :class:`BatchedProbeStatic` for the whole candidate list.
      2. Build initial y0 batch from each candidate's atlas-anchor ml/spin
         seeds + arc-AP centroids.
      3. Run batched spin restore on GPU → y0_restored batch.
      4. ``polish_all(...)`` with each candidate's restored y0 as
         ``y0_override`` and ``skip_spin_restore=True``. Workers polish
         on CPU without paying the ~500 ms per-candidate spin restore.

    Saves ~40 % wall time on full polish runs (the per-cand spin restore
    is the second-biggest scipy-SLSQP cost component after SLSQP itself).

    ``spin_restore_chunk`` bounds the per-call GPU batch size for the
    spin-restore step. The vmap'd kernel allocates ~22 MB / candidate
    of intermediate buffers (n_spins × K × n_rounds outer-product of
    section/shank tensors), so a chunk of 500 needs ~11 GB and OOMs
    on consumer GPUs. The default of 100 fits in ~2 GB; bump up if
    your GPU has headroom.
    """
    # Local imports — keep batched JAX out of worker-process module-level scope
    from aind_low_point.optimization.batched_objective import (
        make_batched_reduced_objective,
    )
    from aind_low_point.optimization.batched_spin_restore import (
        make_batched_spin_restore_partial,
    )
    from aind_low_point.optimization.batched_static import (
        build_batched_probe_static,
    )

    if verbose:
        print(f"[batched_orchestrator] {len(candidates)} candidates")

    B = len(candidates)
    K = len(probes)
    n_vars = n_arcs + 3 * K  # (ml, sx, sy) per probe under Patch B
    probe_names_order = [p.name for p in probes]

    # Pre-build full y0 batch (cheap numpy work — no JAX state). Spin
    # seed in degrees is converted to (sx, sy) on the unit circle.
    y0_np = np.zeros((B, n_vars), dtype=np.float32)
    for b, cand in enumerate(candidates):
        for arc_idx in range(min(n_arcs, len(cand.aa.arc_centroids_deg))):
            y0_np[b, arc_idx] = float(cand.aa.arc_centroids_deg[arc_idx])
        for k, name in enumerate(probe_names_order):
            spin_deg = float(cand.spin_seed.get(name, 0.0))
            spin_rad = np.deg2rad(spin_deg)
            y0_np[b, n_arcs + 3 * k] = float(cand.ml_seed.get(name, 0.0))
            y0_np[b, n_arcs + 3 * k + 1] = float(np.cos(spin_rad))  # sx
            y0_np[b, n_arcs + 3 * k + 2] = float(np.sin(spin_rad))  # sy

    import jax
    import jax.numpy as jnp

    # Build probe_set_static (a small bs_chunk[0] is enough — only the
    # probe-set constants are read) once, then construct the chunked
    # spin-restore JIT one time. Per-chunk bs flows as runtime args so
    # one JIT compile serves every same-shape chunk.
    t_sr0 = time.perf_counter()
    first_lo = 0
    first_hi = min(spin_restore_chunk, B)
    first_pairs = [(c.ha, c.aa) for c in candidates[first_lo:first_hi]]
    probe_set_bs = build_batched_probe_static(
        first_pairs,
        probes,
        holes,
        n_arcs=n_arcs,
        sdf_by_name=sdf_by_name,
        head_pitch_deg=head_pitch_deg,
    )
    spin_restore = make_batched_spin_restore_partial(
        probe_set_bs,
        weights,
        n_spins=n_spins_restore,
        n_rounds=spin_restore_rounds,
    )
    obj_batched, _ = make_batched_reduced_objective(probe_set_bs, weights)
    extract_arrays = obj_batched.extract_arrays  # type: ignore[attr-defined]

    y0_restored_chunks: list[np.ndarray] = []
    n_chunks = (B + spin_restore_chunk - 1) // spin_restore_chunk
    for chunk_idx in range(n_chunks):
        lo = chunk_idx * spin_restore_chunk
        hi = min(lo + spin_restore_chunk, B)
        chunk_pairs = [(c.ha, c.aa) for c in candidates[lo:hi]]

        t_chunk0 = time.perf_counter()
        if chunk_idx == 0:
            bs_chunk = probe_set_bs
        else:
            bs_chunk = build_batched_probe_static(
                chunk_pairs,
                probes,
                holes,
                n_arcs=n_arcs,
                sdf_by_name=sdf_by_name,
                head_pitch_deg=head_pitch_deg,
            )
        y0_chunk = jnp.asarray(y0_np[lo:hi])
        varying = extract_arrays(bs_chunk)
        y0_restored_chunk = spin_restore(y0_chunk, *varying)
        y0_restored_chunk.block_until_ready()
        y0_restored_chunks.append(np.asarray(y0_restored_chunk, dtype=np.float64))

        if verbose:
            t_chunk = time.perf_counter() - t_chunk0
            print(
                f"  chunk {chunk_idx + 1}/{n_chunks} "
                f"[{lo}:{hi}] (n={hi - lo}): {t_chunk:.1f}s"
            )

        del bs_chunk, y0_chunk, y0_restored_chunk, varying

    y0_restored_np = np.concatenate(y0_restored_chunks, axis=0)
    t_sr = time.perf_counter() - t_sr0
    if verbose:
        print(
            f"  batched spin-restore total ({B} cands, "
            f"{n_chunks} chunks): {t_sr:.2f}s "
            f"({t_sr / B * 1000:.1f} ms/cand)"
        )

    # Clear JAX in-process caches before spawning CPU workers so we
    # don't hold GPU memory while subprocesses fight for resources.
    del spin_restore, obj_batched, probe_set_bs
    jax.clear_caches()

    # Now polish via scipy SLSQP per candidate, skipping spin-restore
    polish_inputs = [(cand.ha, cand.aa, float("nan")) for cand in candidates]
    y0_list = [y0_restored_np[i] for i in range(B)]

    if verbose:
        print(
            f"  starting multiproc SLSQP polish ({n_workers or 'auto'} workers, "
            f"skip_spin_restore=True)"
        )
    return polish_all(
        polish_inputs,
        probes,
        holes,
        pose_features,
        weights=weights,
        head_pitch_deg=head_pitch_deg,
        reduced_slsqp_max_iter=reduced_slsqp_max_iter,
        sdf_by_name=sdf_by_name,
        n_workers=n_workers,
        spawn=spawn,
        progress_every=progress_every,
        y0_per_candidate=y0_list,
        skip_spin_restore=True,
        executor=executor,
        verbose=verbose,
    )


def _per_arc_envelopes(cand, atlas) -> list[tuple[float, float] | None]:
    """For each arc of the candidate, compute the per-probe envelope
    intersection ``(lo, hi)``. Returns ``None`` for an arc whose
    intersection is empty or whose atlas entries are missing.
    """
    n_arcs = len(cand.aa.arc_centroids_deg)
    arc_probes: list[list[str]] = [[] for _ in range(n_arcs)]
    for name, arc_idx in cand.aa.probe_to_arc_idx.items():
        arc_probes[arc_idx].append(name)
    envelopes: list[tuple[float, float] | None] = []
    for arc_idx in range(n_arcs):
        per_probe = []
        ok = True
        for name in arc_probes[arc_idx]:
            hid = cand.ha.probe_to_hole[name]
            e = atlas.entries.get((name, hid))
            if e is None or e.ap_min is None or e.ap_max is None:
                ok = False
                break
            per_probe.append((float(e.ap_min), float(e.ap_max)))
        if not ok or not per_probe:
            envelopes.append(None)
            continue
        lo = max(p[0] for p in per_probe)
        hi = min(p[1] for p in per_probe)
        envelopes.append((lo, hi) if lo <= hi else None)
    return envelopes


def _variant_at_ap(cand, arc_idx: int, new_ap: float):
    """Return a copy of ``cand`` (ArcFirstCandidate) with one arc's AP
    overridden. HA, ml/spin seeds, signals all preserved."""
    new_aps = list(cand.aa.arc_centroids_deg)
    new_aps[arc_idx] = new_ap
    new_aa = ArcAssignment(
        probe_to_arc_idx=dict(cand.aa.probe_to_arc_idx),
        arc_centroids_deg=tuple(new_aps),
        cost=cand.aa.cost,
    )
    return type(cand)(
        ha=cand.ha,
        aa=new_aa,
        ml_seed=dict(cand.ml_seed),
        spin_seed=dict(cand.spin_seed),
        ap_intersection_min_width_deg=cand.ap_intersection_min_width_deg,
        min_intra_arc_ml_slack_deg=cand.min_intra_arc_ml_slack_deg,
        total_atlas_anchors=cand.total_atlas_anchors,
        ap_centeredness_sum=cand.ap_centeredness_sum,
        arc_ap_pairwise_min_sep_deg=cand.arc_ap_pairwise_min_sep_deg,
        composite_order_score=cand.composite_order_score,
        components=dict(cand.components),
    )


def polish_all_adaptive(
    candidates,
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    pose_features: dict[tuple[str, int], PoseFeatures],
    atlas,  # for per-arc envelope reconstruction
    *,
    weights: JointWeights = JointWeights(),
    head_pitch_deg: float = 0.0,
    reduced_slsqp_max_iter: int = 50,
    sdf_by_name: dict | None = None,
    n_workers: int | None = None,
    boundary_lo: float = 0.05,
    boundary_hi: float = 2.0,
    extra_quantiles: tuple[float, ...] = (0.25, 0.75),
    n_arcs: int = 3,
    spin_restore_chunk: int = 100,
    verbose: bool = False,
) -> list[JointCandidate]:
    """Single-seed full polish + adaptive retry for boundary candidates.

    Pipeline:
      1. Run ``polish_all_with_batched_spin_restore`` on all candidates
         with the midpoint AP seed (default in ArcFirstCandidate.aa).
      2. Identify "boundary" candidates with polish max_viol in
         ``[boundary_lo, boundary_hi]`` — near-feasible but not at zero.
         Catastrophic candidates (max_viol > boundary_hi) are skipped
         because multi-seed never rescues them in our data.
      3. For each boundary candidate, build extra AP-seed variants by
         perturbing the WIDEST arc to the supplied ``extra_quantiles``
         of its per-probe envelope intersection.
      4. Polish all retry variants via the same batched-spin-restore +
         multiproc-SLSQP path.
      5. Return, per candidate, the JointCandidate with the lex-best
         metrics across {initial, retry variants}.

    Captures the basin diversity that single midpoint seeding misses
    in ~5-10% of candidates without paying 3× cost on all.
    """
    if not candidates:
        return []

    # One persistent worker pool across phase 1 + phase 3. Saves the
    # respawn (~10-15s spawn + ~5-10s per-worker disk-JIT load) the
    # adaptive retry would otherwise pay. Phase 1 init args (probes,
    # holes, SDFs, weights, skip_spin_restore=True) match phase 3 exactly,
    # so one pool is correct.
    with polish_worker_pool(
        probes,
        holes,
        pose_features,
        weights=weights,
        head_pitch_deg=head_pitch_deg,
        reduced_slsqp_max_iter=reduced_slsqp_max_iter,
        sdf_by_name=sdf_by_name,
        n_workers=n_workers,
        skip_spin_restore=True,
    ) as pool:
        if verbose:
            print(f"[adaptive] phase 1: single-seed polish ({len(candidates)} cands)")
        initial = polish_all_with_batched_spin_restore(
            candidates,
            probes,
            holes,
            pose_features,
            weights=weights,
            head_pitch_deg=head_pitch_deg,
            reduced_slsqp_max_iter=reduced_slsqp_max_iter,
            sdf_by_name=sdf_by_name,
            n_workers=n_workers,
            spin_restore_chunk=spin_restore_chunk,
            n_arcs=n_arcs,
            executor=pool,
            verbose=verbose,
        )

        # Identify boundary candidates
        boundary_idxs = []
        for i, jc in enumerate(initial):
            mv = float(jc.metrics.max_violation)
            if boundary_lo <= mv <= boundary_hi:
                boundary_idxs.append(i)
        if verbose:
            print(
                f"[adaptive] phase 2: {len(boundary_idxs)} boundary candidates "
                f"({len(boundary_idxs) / max(1, len(candidates)) * 100:.1f}%)"
            )

        if not boundary_idxs:
            return initial

        # Build retry variants
        retry_variants = []
        retry_owner: list[int] = []  # parallel: which boundary_idxs[i] each came from
        for i in boundary_idxs:
            cand = candidates[i]
            envs = _per_arc_envelopes(cand, atlas)
            widths = [(e[1] - e[0]) if e is not None else 0.0 for e in envs]
            arc_idx = int(np.argmax(widths))
            env = envs[arc_idx]
            if env is None:
                continue
            lo, hi = env
            for q in extra_quantiles:
                new_ap = lo + q * (hi - lo)
                retry_variants.append(_variant_at_ap(cand, arc_idx, new_ap))
                retry_owner.append(i)
        if verbose:
            print(f"[adaptive] phase 3: polishing {len(retry_variants)} retry variants")

        if not retry_variants:
            return initial

        retry_polished = polish_all_with_batched_spin_restore(
            retry_variants,
            probes,
            holes,
            pose_features,
            weights=weights,
            head_pitch_deg=head_pitch_deg,
            reduced_slsqp_max_iter=reduced_slsqp_max_iter,
            sdf_by_name=sdf_by_name,
            n_workers=n_workers,
            spin_restore_chunk=spin_restore_chunk,
            n_arcs=n_arcs,
            executor=pool,
            verbose=verbose,
        )

    # Merge: for each original candidate, take lex-best across {initial, retries}
    best_per_owner: dict[int, JointCandidate] = {
        i: initial[i] for i in range(len(candidates))
    }
    n_improved = 0
    for retry_idx, owner_i in enumerate(retry_owner):
        retry_jc = retry_polished[retry_idx]
        current = best_per_owner[owner_i]
        if retry_jc.metrics.lex_key() < current.metrics.lex_key():
            best_per_owner[owner_i] = retry_jc
            n_improved += 1
    if verbose:
        print(f"[adaptive] phase 4: {n_improved} boundary candidates improved")

    return [best_per_owner[i] for i in range(len(candidates))]
