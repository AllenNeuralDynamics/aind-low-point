"""Run the FULL MRV pool through restore → reduced 500 → full 500 (TUNED).

Enumerates the MRV pool (3 arcs, <=4 probes/arc; ~19k candidates), seeds each
from the joint ``emit_seed`` (arc/ml/spin) via ``Enumerator.seed``, spin-restores
(N_SPINS=16), then runs each candidate through a two-stage (reduced→full)
optimization and a per-candidate FCL gate. Output → OUT (then cull/select/
trust-constr from there).

The optimizer is TUNED (see dev/POOL_RUN_CONFIGS.md for the measurements):
  - WELL=thick   — solidified well SDF (fixes the thin-skin false-negative);
                   FCL still uses the true thin mesh (honest gate).
  - iRprop− — sign-based; immune to the ADAM v-freeze that stalls long runs.
  - coarse→fine surf — reduced/full each run (STAGE-REDUCED_FINE/FULL_FINE) steps
                   @COARSE_N surf then the FINE_* finish @5000 (the homotopy win).

Two documented presets (copy-paste commands in dev/POOL_RUN_CONFIGS.md):
  THROUGHPUT (default): COARSE_N=1000, REDUCED_FINE=FULL_FINE=50   (~2.16x; 545:105/20)
  YIELD:                COARSE_N=3000, REDUCED_FINE=FULL_FINE=100  (~1.31x; 545:123/21)

Per candidate saves the final + reduced-checkpoint pose, min dual-rep clearance
(the cull metric), coverage, the discrete decision, and the MRV seed gap.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 rutter-phase1
Env:  WELL=thick COARSE_N=1000 REDUCED_FINE=50 FULL_FINE=50
      STAGE1=500 STAGE2=500 N_SPINS=16 CHUNK=256 RESTORE_CHUNK=128
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import gc
import multiprocessing as _mp
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from aind_rutter.optimization.jax_env import configure_compile_cache
from aind_rutter.optimization.objectives.batched_reduced import (
    make_batched_reduced_objective,
)
from aind_rutter.optimization.objectives.batched_static import (
    build_batched_probe_static,
)
from aind_rutter.optimization.objectives.clearance_metrics import make_min_clear_one
from aind_rutter.optimization.objectives.phase1 import Phase1Weights
from aind_rutter.optimization.objectives.probe_static import (
    JointWeights,
    _build_probe_static,
)
from aind_rutter.optimization.objectives.spin_restore import (
    make_batched_spin_restore_partial,
)
from aind_rutter.optimization.objectives.variables import build_y
from aind_rutter.optimization.pipeline.contracts import (
    ArglistBuilder,
    EnumeratorCandidate,
    MRVArcAssignment,
    MRVHoleAssignment,
    Partition,
    Phase1PoolPayload,
    Phase1PoolRecord,
    ProbeToHole,
    SeedMap,
)
from aind_rutter.optimization.pipeline.enumeration import (
    Enumerator,
    build_or_load_atlas,
)
from aind_rutter.optimization.pipeline.payloads import (
    read_pool,
    read_seed_cache,
    write_pool,
    write_seed_cache,
)
from aind_rutter.optimization.pipeline.phase1_build import (
    ARG_ORDER,
    PER_CAND,
    build_cw_fns,
    make_batched_phase1_chunked,
    make_staged_rprop,
)
from aind_rutter.optimization.pipeline.phase1_geometry import (
    build_coverage_data,
    phase1_bounds,
)
from aind_rutter.optimization.pipeline.restore import (
    PPV,
    setup,
    setup_runtime,
    spins_deg_from_reduced,
)
from aind_rutter.optimization.pipeline.settings import Phase1Settings
from aind_rutter.planning import AP_LIMIT_DEG

# Print the normalization summary once (not once per arc-group).
_group_log_once = [True]


@dataclass
class MRVCand:
    ha: MRVHoleAssignment
    aa: MRVArcAssignment
    ml_seed: SeedMap
    spin_seed: SeedMap
    min_ml_gap: float
    n_arcs: int
    probe_to_hole: ProbeToHole
    partition: Partition


def wrap(cand_dict: EnumeratorCandidate, enum: Enumerator) -> MRVCand | None:
    """MRV dict → candidate with .ha/.aa stand-ins (interface-compatible with
    _build_probe_static + the optimizer). Returns None if emit_seed fails."""
    res = enum.seed(cand_dict)
    if res is None:
        return None
    arc_aps, ml_seed, spin_seed, gap = res
    # Canonical arc order matching Enumerator.seed (sorted by member names) so
    # probe_to_arc_idx aligns with arc_aps regardless of the hash seed.
    groups = sorted(cand_dict["partition"], key=sorted)
    p2a = {name: ai for ai, grp in enumerate(groups) for name in grp}
    ha = MRVHoleAssignment(probe_to_hole=cand_dict["probe_to_hole"])
    aa = MRVArcAssignment(probe_to_arc_idx=p2a, arc_centroids_deg=list(arc_aps))
    return MRVCand(
        ha,
        aa,
        dict(ml_seed),
        dict(spin_seed),
        float(gap),
        len(groups),
        cand_dict["probe_to_hole"],
        cand_dict["partition"],
    )


def _seed_record(c: MRVCand) -> dict[str, object]:
    if c.ha.probe_to_hole != c.probe_to_hole:
        raise ValueError("candidate hole assignment disagrees with probe_to_hole")
    return {
        "n_arcs": c.n_arcs,
        "probe_to_hole": c.probe_to_hole,
        "partition": c.partition,
        "probe_to_arc_idx": c.aa.probe_to_arc_idx,
        "arc_centroids_deg": c.aa.arc_centroids_deg,
        "ml_seed": c.ml_seed,
        "spin_seed": c.spin_seed,
        "min_ml_gap": c.min_ml_gap,
    }


def _cand_from_record(r: dict[str, Any]) -> MRVCand:
    # Shares one probe_to_hole dict between .ha and the candidate, as wrap does.
    return MRVCand(
        MRVHoleAssignment(probe_to_hole=r["probe_to_hole"]),
        MRVArcAssignment(
            probe_to_arc_idx=r["probe_to_arc_idx"],
            arc_centroids_deg=r["arc_centroids_deg"],
        ),
        r["ml_seed"],
        r["spin_seed"],
        r["min_ml_gap"],
        r["n_arcs"],
        r["probe_to_hole"],
        r["partition"],
    )


# Per-worker enumerator for the spawn-based seed pool. The pool uses ``spawn``
# (not ``fork``): this module imports JAX, which is multithreaded, and forking a
# multithreaded process risks a deadlock. ``_seed_init`` receives the (pickled,
# read-only) enumerator once per worker and stashes it here.
_SEED_ENUM: "Enumerator | None" = None


def _seed_init(enum: "Enumerator") -> None:
    """Pool initializer: stash the enumerator for this worker's tasks."""
    global _SEED_ENUM
    _SEED_ENUM = enum


def _seed_one(cand_dict: EnumeratorCandidate) -> "MRVCand | None":
    """Seed one candidate in a worker, using the initializer-provided enumerator."""
    assert _SEED_ENUM is not None  # set by _seed_init
    return wrap(cand_dict, _SEED_ENUM)


def reduced_lohi(
    lo: np.ndarray, hi: np.ndarray, n_arcs: int, K: int
) -> tuple[np.ndarray, np.ndarray]:
    lo_r, hi_r = lo.copy(), hi.copy()
    for k in range(K):
        for off in (3, 4, 5):
            lo_r[n_arcs + PPV * k + off] = 0.0
            hi_r[n_arcs + PPV * k + off] = 0.0
    return lo_r, hi_r


def restore_group(
    n_arcs,
    cands: list[MRVCand],
    *,
    settings: Phase1Settings,
    probes,
    holes,
    sdf_by_name,
    well,
    head_pitch_deg=0.0,
):
    """Batched spin restore seeded from each cand's MRV ml/spin. Returns per-cand
    restored spin-degrees (parallel to cands)."""
    K = len(probes)
    names = [p.name for p in probes]
    weights = JointWeights()
    fixtures = (well,)
    seed_rows: list[np.ndarray] = []
    for c in cands:
        y0 = np.zeros(n_arcs + 3 * K, np.float32)
        for a in range(min(n_arcs, len(c.aa.arc_centroids_deg))):
            y0[a] = float(c.aa.arc_centroids_deg[a])
        for k, p in enumerate(probes):
            sp = np.deg2rad(float(c.spin_seed.get(p.name, 0.0)))
            y0[n_arcs + 3 * k] = float(c.ml_seed.get(p.name, 0.0))
            y0[n_arcs + 3 * k + 1] = float(np.cos(sp))
            y0[n_arcs + 3 * k + 2] = float(np.sin(sp))
        seed_rows.append(y0)
    seeds = np.stack(seed_rows)
    B = len(cands)
    initial_pairs = cast(
        Any, [(c.ha, c.aa) for c in cands[: min(settings.restore_chunk, B)]]
    )
    bs0 = build_batched_probe_static(
        initial_pairs,
        probes,
        holes,
        n_arcs=n_arcs,
        sdf_by_name=sdf_by_name,
        head_pitch_deg=head_pitch_deg,
    )
    restore = make_batched_spin_restore_partial(
        bs0,
        weights,
        n_spins=settings.n_spins,
        n_rounds=settings.restore_rounds,
        fixtures=fixtures,
    )
    obj_b, _ = make_batched_reduced_objective(bs0, weights, fixtures)
    obj_b = cast(Any, obj_b)
    out: list[np.ndarray] = []
    for lo in range(0, B, settings.restore_chunk):
        hi = min(lo + settings.restore_chunk, B)
        bs = (
            bs0
            if lo == 0
            else build_batched_probe_static(
                cast(Any, [(c.ha, c.aa) for c in cands[lo:hi]]),
                probes,
                holes,
                n_arcs=n_arcs,
                sdf_by_name=sdf_by_name,
                head_pitch_deg=head_pitch_deg,
            )
        )
        y_r = restore(jnp.asarray(seeds[lo:hi]), *obj_b.extract_arrays(bs))
        y_r.block_until_ready()
        out.extend(
            spins_deg_from_reduced(np.asarray(y_r[b], np.float64), n_arcs, K)
            for b in range(hi - lo)
        )
    _ = names
    return out


def make_runner(vgrad_cw) -> Callable:
    """run(x0, arglist, lo, hi, cov_weight, n_steps)."""
    return make_staged_rprop(vgrad_cw, eta0_frac=0.02, etamax_frac=0.5)


def _kernel(
    st0,
    n_arcs,
    cov,
    well_soft,
    brain_sdf,
    grid_dtype,
    *,
    settings: Phase1Settings,
    ceilings=None,
    cov_weights=None,
) -> tuple[Callable, ArglistBuilder, Callable]:
    """Build (vobj, build_arglist, run) for one fidelity's template statics."""
    weights = (
        Phase1Weights(cov_alpha=settings.cov_alpha)
        if settings.cov_norm
        else Phase1Weights()
    )
    vobj, _vg, barg = make_batched_phase1_chunked(
        st0,
        n_arcs,
        weights,
        (well_soft,),
        coverage_data=cov,
        grid_dtype=grid_dtype,
        brain_sdf=brain_sdf,
        coverage_ceilings=ceilings,
        coverage_weights=cov_weights,
    )
    vg = build_cw_fns(
        st0,
        n_arcs,
        cov,
        well_soft,
        brain_sdf,
        weights=weights,
        coverage_ceilings=ceilings,
        coverage_weights=cov_weights,
    )[1]
    return vobj, barg, make_runner(vg)


def run_group(  # noqa: C901
    n_arcs: int,
    cands: list[MRVCand],
    *,
    settings: Phase1Settings,
    probes,
    holes,
    sdf_fine,
    sdf_coarse,
    bvh,
    well_soft,
    brain_sdf,
    head_pitch_deg=0.0,
):
    K = len(probes)
    names = [p.name for p in probes]
    bounds = phase1_bounds(n_arcs, K, head_pitch_deg)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    lo_r, hi_r = reduced_lohi(lo, hi, n_arcs, K)

    t0 = time.time()
    spins = restore_group(
        n_arcs,
        cands,
        settings=settings,
        probes=probes,
        holes=holes,
        sdf_by_name=sdf_fine,
        well=well_soft,
        head_pitch_deg=head_pitch_deg,
    )
    print(f"  restore {time.time() - t0:.1f}s", flush=True)

    st_f, st_c, x0_rows = [], [], []
    for c, sp in zip(cands, spins):
        st_f.append(
            _build_probe_static(
                probes,
                holes,
                cast(Any, c.ha),
                cast(Any, c.aa),
                bvh_cache=bvh,
                sdf_by_name=sdf_fine,
            )
        )
        if settings.two_fidelity:
            st_c.append(
                _build_probe_static(
                    probes,
                    holes,
                    cast(Any, c.ha),
                    cast(Any, c.aa),
                    bvh_cache=bvh,
                    sdf_by_name=sdf_coarse,
                )
            )
        arc_aps = np.zeros(n_arcs)
        for a in range(min(n_arcs, len(c.aa.arc_centroids_deg))):
            arc_aps[a] = float(c.aa.arc_centroids_deg[a])
        mls = np.array([float(c.ml_seed.get(n, 0.0)) for n in names])
        z = np.zeros(K)
        x0_rows.append(build_y(arc_aps, n_arcs, mls, np.asarray(sp), z, z, z))
    x0 = np.stack(x0_rows).astype(np.float32)

    cov = build_coverage_data(probes, st_f[0])
    # Per-region normalization: ceilings (achievable per-probe coverage) +
    # per-target priority weights are per-probe-fixed (target/σ/density/geometry),
    # so compute once per group from any candidate's statics. Only active when
    # settings.cov_norm is set; otherwise pass None ⇒ legacy plain-sum coverage.
    ceilings, cov_weights = None, None
    if settings.cov_norm:
        from aind_rutter.optimization.objectives.coverage import (
            coverage_ceiling_per_probe,
        )

        ceilings = tuple(float(c) for c in coverage_ceiling_per_probe(st_f[0], cov))
        cov_weights = tuple(float(p.coverage_weight) for p in probes)
        if _group_log_once[0]:
            alpha, gain = settings.cov_alpha, settings.cov_weight
            print(
                f"  coverage NORMALIZED; ceilings={[round(c, 3) for c in ceilings]}, "
                f"weights={[round(w, 3) for w in cov_weights]}, α={alpha}, "
                f"gain λ_cov={gain}",
                flush=True,
            )
            _group_log_once[0] = False
    grid_dtype = jnp.bfloat16 if settings.bf16_store else jnp.float32
    vobj, barg_f, run_f = _kernel(
        st_f[0],
        n_arcs,
        cov,
        well_soft,
        brain_sdf,
        grid_dtype,
        settings=settings,
        ceilings=ceilings,
        cov_weights=cov_weights,
    )
    if settings.two_fidelity:
        _vo, barg_c, run_c = _kernel(
            st_c[0],
            n_arcs,
            cov,
            well_soft,
            brain_sdf,
            grid_dtype,
            settings=settings,
            ceilings=ceilings,
            cov_weights=cov_weights,
        )
    else:
        barg_c, run_c, st_c = barg_f, run_f, st_f

    n_rows = x0.shape[0]
    n_pad = (-n_rows) % settings.chunk
    if n_pad:
        st_f = st_f + [st_f[-1]] * n_pad
        st_c = st_c + [st_c[-1]] * n_pad
        x0 = np.concatenate([x0, np.repeat(x0[-1:], n_pad, 0)], 0)
    n_tot = x0.shape[0]
    x0_dev = jnp.asarray(x0, jnp.float32)

    rc, rf = settings.stage1 - settings.reduced_fine, settings.reduced_fine
    fc, ff = settings.stage2 - settings.full_fine, settings.full_fine
    n_chunks = n_tot // settings.chunk
    fid = f"coarse{settings.coarse_n}→fine" if settings.two_fidelity else "fine"
    print(
        f"  rprop {fid} (chunk={settings.chunk}, reduced {rc}c+{rf}f → "
        f"full {fc}c+{ff}f, {n_chunks} chunks)...",
        flush=True,
    )
    t0 = time.time()
    x_out = np.zeros_like(x0)  # full@end pose
    x_red = np.zeros_like(x0)  # reduced@end pose (cull checkpoint)
    for ci, s in enumerate(range(0, n_tot, settings.chunk)):
        cargs_f = barg_f(st_f[s : s + settings.chunk])
        cargs_c = (
            barg_c(st_c[s : s + settings.chunk]) if settings.two_fidelity else cargs_f
        )
        x = x0_dev[s : s + settings.chunk]
        if rc > 0:
            x = run_c(x, cargs_c, lo_r, hi_r, 0.0, rc)  # reduced coarse
        if rf > 0:
            x = run_f(x, cargs_f, lo_r, hi_r, 0.0, rf)  # reduced fine finish
        x_red[s : s + settings.chunk] = np.asarray(x)
        if fc > 0:
            x = run_c(x, cargs_c, lo, hi, settings.cov_weight, fc)  # full coarse
        if ff > 0:
            x = run_f(x, cargs_f, lo, hi, settings.cov_weight, ff)  # full fine finish
        x_out[s : s + settings.chunk] = np.asarray(x)
        if (ci + 1) % settings.progress_every == 0 or ci + 1 == n_chunks:
            el = time.time() - t0
            print(
                f"    chunk {ci + 1}/{n_chunks}  {el:.0f}s  "
                f"ETA {el / (ci + 1) * (n_chunks - ci - 1):.0f}s",
                flush=True,
            )
    print(f"  {time.time() - t0:.1f}s optimize", flush=True)

    # Final scores at fine fidelity (soft clearance uses the thick well too).
    in_axes = (0,) + tuple(0 if k in PER_CAND else None for k in ARG_ORDER)
    clear_b = jax.jit(
        jax.vmap(make_min_clear_one(n_arcs, K, (well_soft,), Phase1Weights()), in_axes)
    )
    obj = np.empty(n_tot, np.float32)
    clr = np.empty(n_tot, np.float32)  # full@end clearance (cull metric)
    clr_red = np.empty(n_tot, np.float32)  # reduced@end clearance (checkpoint)
    x_dev = jnp.asarray(x_out, jnp.float32)
    xr_dev = jnp.asarray(x_red, jnp.float32)
    for s in range(0, n_tot, settings.chunk):
        cargs = barg_f(st_f[s : s + settings.chunk])
        obj[s : s + settings.chunk] = np.asarray(
            vobj(x_dev[s : s + settings.chunk], *cargs)
        )
        clr[s : s + settings.chunk] = np.asarray(
            clear_b(x_dev[s : s + settings.chunk], *cargs)
        )
        clr_red[s : s + settings.chunk] = np.asarray(
            clear_b(xr_dev[s : s + settings.chunk], *cargs)
        )
    return (
        st_f[:n_rows],
        x_out[:n_rows],
        x_red[:n_rows],
        obj[:n_rows],
        clr[:n_rows],
        clr_red[:n_rows],
    )


def make_phase1_pool_record(
    c: MRVCand,
    n_arcs: int,
    x: np.ndarray,
    x_reduced: np.ndarray,
    objective: float,
    min_clear: float,
    min_clear_reduced: float,
) -> Phase1PoolRecord:
    return dict(
        n_arcs=n_arcs,
        probe_to_hole=c.probe_to_hole,
        partition=c.partition,
        # Arc assignment saved EXPLICITLY (not reconstructed from the
        # frozenset partition, whose iteration order is hash-random
        # across processes) so Phase 2 rebuilds the identical `aa`
        # the pose `x` was optimized against. See phase2_ipopt.
        probe_to_arc_idx=dict(c.aa.probe_to_arc_idx),
        arc_centroids_deg=list(c.aa.arc_centroids_deg),
        min_ml_gap=c.min_ml_gap,
        x=x.astype(np.float32),
        x_reduced=x_reduced.astype(np.float32),
        objective=float(objective),
        min_clear=float(min_clear),
        min_clear_reduced=float(min_clear_reduced),
    )


def save_results(records: list[Phase1PoolRecord], settings: Phase1Settings) -> None:
    payload: Phase1PoolPayload = dict(
        records=records,
        stage1=settings.stage1,
        stage2=settings.stage2,
        n_spins=settings.n_spins,
        max_arcs=settings.max_arcs,
        max_ppa=settings.max_probes_per_arc,
        minimizer="rprop",
        well=settings.well,
        coarse_n=settings.coarse_n,
        reduced_fine=settings.reduced_fine,
        full_fine=settings.full_fine,
    )
    write_pool(settings.out, payload)


def load_or_seed_groups(
    enum_factory: Callable[[], Enumerator],
    settings: Phase1Settings,
) -> dict[int, list[MRVCand]]:
    """Enumerate and seed the MRV pool once, cached grouped by arc count.

    On a restart with the cache present and no candidate cap, this just reloads:
    the seed CSP takes about 14 minutes.
    """
    if (
        settings.seed_cache
        and not settings.limit
        and _os.path.exists(settings.seed_cache)
    ):
        try:
            cached_by_arcs = {
                n_arcs: [_cand_from_record(r) for r in records]
                for n_arcs, records in read_seed_cache(settings.seed_cache).items()
            }
        except ValueError as e:
            print(f"seed cache unusable, re-seeding: {e}", flush=True)
        else:
            print(
                f"loaded seeds from {settings.seed_cache}: groups "
                + ", ".join(f"{k}:{len(v)}" for k, v in sorted(cached_by_arcs.items())),
                flush=True,
            )
            return cached_by_arcs

    arcs, per_arc = settings.max_arcs, settings.max_probes_per_arc
    print(f"enumerating MRV pool (arcs<={arcs}, probes/arc<={per_arc})...", flush=True)
    enum = enum_factory()
    t0 = time.time()
    raw = enum.enumerate()
    print(f"  {len(raw)} discrete candidates in {time.time() - t0:.1f}s", flush=True)
    if settings.limit:
        raw = raw[: settings.limit]
    t0 = time.time()
    seed_workers = settings.seed_workers
    if seed_workers > 1 and len(raw) > seed_workers:
        # Seeding is independent per candidate and CPU-bound (numpy CSP backtrack
        # over tiny arrays — GIL-held, so threads don't help) but parallelises
        # cleanly across processes. Use spawn, NOT fork: this module imports JAX,
        # which is multithreaded, and forking a multithreaded process risks a
        # deadlock. Spawn workers re-import the module (importing JAX is lazy — no
        # GPU is grabbed until an op runs, and seeding runs only numpy) and receive
        # the read-only enumerator once via the initializer. ex.map preserves input
        # order; seed values are process-independent, so the result equals the
        # serial pool modulo cosmetic dict/arc-label ordering.
        chunk = max(1, len(raw) // (seed_workers * 8))
        with ProcessPoolExecutor(
            max_workers=seed_workers,
            mp_context=_mp.get_context("spawn"),
            initializer=_seed_init,
            initargs=(enum,),
        ) as ex:
            results = list(ex.map(_seed_one, raw, chunksize=chunk))
    else:
        results = [wrap(d, enum) for d in raw]
    cands = [c for c in results if c is not None]
    n_fail = sum(1 for c in results if c is None)
    print(
        f"  seeded {len(cands)} cands ({n_fail} dropped: no anchors) "
        f"in {time.time() - t0:.1f}s [{seed_workers}w]",
        flush=True,
    )
    by_arcs: dict[int, list[MRVCand]] = {}
    for c in cands:
        by_arcs.setdefault(c.n_arcs, []).append(c)
    print("  groups " + ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_arcs.items())))
    if settings.seed_cache and not settings.limit:
        write_seed_cache(
            settings.seed_cache,
            {n_arcs: [_seed_record(c) for c in g] for n_arcs, g in by_arcs.items()},
        )
        print(f"  cached seeds → {settings.seed_cache}", flush=True)
    return by_arcs


def run(settings: Phase1Settings) -> int:
    """Build the Phase-1 pool described by ``settings``.

    Importable, so a caller in Python constructs the settings directly rather
    than through the environment. Resumable: ``settings.out`` is reloaded and
    the arc-count groups already in it are skipped.
    """
    # Explicit, because no import configures the compile cache any more.
    configure_compile_cache()
    opt = setup_runtime()
    _cfg, _rt, probes, holes, sdf_fine, bvh, fixtures, well_thin, _fbvh = setup(opt)
    brain = opt.brain_sdf()

    # Tuned optimizer: thick well (soft side only; FCL uses true mesh) + coarse SDF.
    well_soft = (
        opt.thick_well_fixture(well_thin) if settings.well == "thick" else well_thin
    )
    sdf_coarse = (
        opt.probe_sdfs(settings.coarse_n) if settings.two_fidelity else sdf_fine
    )
    fidelity = (
        f"coarse{settings.coarse_n}→fine" if settings.two_fidelity else "fine-only"
    )
    reduced_coarse = settings.stage1 - settings.reduced_fine
    full_coarse = settings.stage2 - settings.full_fine
    print(
        f"config: well={settings.well} {fidelity} "
        f"reduced {reduced_coarse}c+{settings.reduced_fine}f → "
        f"full {full_coarse}c+{settings.full_fine}f → {settings.out}",
        flush=True,
    )

    def _enum_factory() -> Enumerator:
        atlas_payload = build_or_load_atlas()
        # rig AP = subject AP + head_pitch (head nose-down) → rig-reachable subject
        # window = rig[±AP_LIMIT] − head_pitch (mirrors phase1_bounds /
        # _ap_bounds_deg). See dev memory rig_ap_sign_convention. ML is invariant.
        ap_range = (
            -AP_LIMIT_DEG - atlas_payload.head_pitch_deg,
            AP_LIMIT_DEG - atlas_payload.head_pitch_deg,
        )
        return Enumerator(
            atlas_payload.atlas,
            atlas_payload.probe_names,
            ml_margin_deg=0.0,
            max_arcs=settings.max_arcs,
            max_probes_per_arc=settings.max_probes_per_arc,
            ap_range=ap_range,
        )

    by_arcs = load_or_seed_groups(_enum_factory, settings)

    if settings.only_narcs:
        by_arcs = {k: v for k, v in by_arcs.items() if k == settings.only_narcs}
        print(
            f"  settings.only_narcs={settings.only_narcs}: "
            + (f"{len(next(iter(by_arcs.values())))} cands" if by_arcs else "no cands")
        )

    # Resume: reload any previously-saved records and skip those n_arcs groups.
    records: list[Phase1PoolRecord] = []
    done_narcs: set = set()
    if _os.path.exists(settings.out):
        # An unreadable pool aborts the run: save_results rewrites the file with
        # every group, so carrying on would discard the groups it already holds.
        records = list(read_pool(settings.out)["records"])
        done_narcs = {r["n_arcs"] for r in records}
        if done_narcs:
            print(
                f"resuming from {settings.out}: {len(records)} records, "
                f"done groups {sorted(done_narcs)}",
                flush=True,
            )

    # Largest group first: the hungriest spin-restore runs on the cleanest GPU.
    for n_arcs in sorted(by_arcs, key=lambda k: -len(by_arcs[k])):
        if n_arcs in done_narcs:
            print(
                f"\n[n_arcs={n_arcs}] already in {settings.out}, skipping", flush=True
            )
            continue
        g = by_arcs[n_arcs]
        print(f"\n[n_arcs={n_arcs}] {len(g)} cands", flush=True)
        statics_flat, x_out, x_red, obj, clr, clr_red = run_group(
            n_arcs,
            g,
            settings=settings,
            probes=probes,
            holes=holes,
            sdf_fine=sdf_fine,
            sdf_coarse=sdf_coarse,
            bvh=bvh,
            well_soft=well_soft,
            brain_sdf=brain,
            head_pitch_deg=opt.head_pitch_deg,
        )
        for i, c in enumerate(g):
            records.append(
                make_phase1_pool_record(
                    c,
                    n_arcs,
                    x_out[i],
                    x_red[i],
                    float(obj[i]),
                    float(clr[i]),
                    float(clr_red[i]),
                )
            )
        # Incremental save + free this group's GPU buffers before the next group
        # compiles its own kernels (cross-group accumulation caused the OOM).
        save_results(records, settings)
        print(f"  saved {len(records)} records → {settings.out}", flush=True)
        del statics_flat, x_out, x_red, obj, clr, clr_red
        gc.collect()
        jax.clear_caches()

    save_results(records, settings)
    print(f"\nsaved {len(records)} records → {settings.out}")
    return 0


def main() -> int:
    """Console entry point: settings from the environment, then `run`."""
    return run(Phase1Settings())


if __name__ == "__main__":
    raise SystemExit(main())
