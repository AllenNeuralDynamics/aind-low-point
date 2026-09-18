"""Parallel Phase 2 across candidates + diversity-aware (MMR) handoff ranking.

trust-constr is single-threaded, but candidates are independent — so we run
Phase 2 in a process pool (pinned to 1 BLAS thread/worker to avoid 24x24
oversubscription). Each worker builds the heavy setup once (SDFs load from disk
cache), then polishes its candidates with the balanced soft-bonus config.

Then: keep FCL >= -TOL (human-fixable), and rank by MMR — greedily pick highest
post-Phase-2 coverage, penalizing similarity (shared probe->hole fraction) to
already-picked plans, so the ranked handoff is high-coverage AND diverse.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 rutter-phase2
      or call ``run(recs, settings)`` from Python.
Env:  every ``pipeline.settings.Phase2Settings`` field, under its alias. POOL,
      PLATFORM, THREADS and GPU_MEM_FRACTION are read below at import instead,
      because they must be set before jax loads.
"""

from __future__ import annotations

import os as _os

# Set BLAS/OMP threads-per-worker BEFORE jax/numpy import (THREADS=1 pins each
# worker single-thread to avoid Nworkers x Ncore oversubscription; raise it to
# test whether some oversubscription fills memory-bound stalls). Spawned workers
# inherit this env and re-run this block.
_THREADS = _os.environ.get("THREADS", "1")
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_v, _THREADS)
# POOL=thread (default): ONE process/GPU context, N threads sharing it —
# trust-constr/IPOPT's JAX evals release the GIL, so threads overlap and fill
# the GPU's idle gaps (the scipy subproblem is CPU-bound and leaves the GPU
# idle), with NO extra GPU memory. POOL=process: N worker processes (N GPU
# contexts → memory-limited). Read here (before the platform block) so the GPU
# memory fraction can depend on it.
POOL = _os.environ.get("POOL", "thread")
# Platform: PLATFORM=gpu (default) or cpu. On GPU, never preallocate. A THREAD
# pool is one context → give it most of the card; a PROCESS pool is N contexts →
# a small fraction each. Spawned workers inherit this env.
_PLATFORM = _os.environ.get("PLATFORM", "gpu")
if _PLATFORM in ("gpu", "cuda"):
    _PLATFORM = "cuda"  # JAX backend name
    # Never preallocate on GPU: grow on demand. Measured on a CLEAN card, one
    # Phase-2 context's true working set is ~1.1 GB (platform allocator) / ~2.4 GB
    # resident with BFC pooling — so N process workers fit easily (3 × ~2.4 GB +
    # desktop < 10 GB). Earlier "~7.6 GB greedy" readings were a contaminated
    # bench: orphan/zombie contexts left by killed workers, NOT real footprint.
    # A THREAD pool is one context → give it most of the card; a PROCESS pool is
    # N contexts → a modest fraction each. Spawned workers inherit this env.
    # IMPORTANT: preallocate=false MUST be in the environment BEFORE this process
    # starts python — setting it here at import is too late to win against jax's
    # backend init, so each context falls back to XLA's default preallocate=true
    # and grabs ~0.75 of the card (~7.6 GB). Then only ONE process context fits
    # and the rest die ("no supported devices"). With it set in the launcher env,
    # contexts grow on demand to ~2.4 GB and 3 workers coexist (measured 2.7x).
    # The setdefault below is a best-effort fallback for the thread pool / direct
    # runs; PROCESS-pool drivers MUST `export XLA_PYTHON_CLIENT_PREALLOCATE=false`
    # (the _require_gpu_headroom check below aborts loudly if they didn't).
    _os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _default_frac = "0.9" if POOL == "thread" else "0.18"
    _os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        _os.environ.get("GPU_MEM_FRACTION", _default_frac),
    )
_os.environ.setdefault("JAX_PLATFORMS", _PLATFORM)

import itertools  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from typing import Any, cast  # noqa: E402

import numpy as np  # noqa: E402

from aind_rutter.optimization.pipeline.contracts import (  # noqa: E402
    MRVArcAssignment,
    MRVHoleAssignment,
    Phase2HandoffPayload,
    Phase2InputRecord,
    Phase2Problem,
    Phase2ResultRecord,
    ProbeToHole,
)
from aind_rutter.optimization.pipeline.handoff import (  # noqa: E402
    classify_results,
    handoff_config,
)
from aind_rutter.optimization.pipeline.payloads import (  # noqa: E402
    read_pool,
    write_handoff,
)
from aind_rutter.optimization.pipeline.selection import (  # noqa: E402
    select_records,
)
from aind_rutter.optimization.pipeline.settings import Phase2Settings  # noqa: E402

_G: dict = {}


def _setup_compile_cache():
    """Enable JAX's persistent compile cache for this worker.

    A spawned worker cannot see the parent's in-memory cache, so without the
    on-disk one every worker repays the whole compile. One warmup writes the
    executables and the rest load them.
    """
    from aind_rutter.optimization.jax_env import configure_compile_cache

    configure_compile_cache()


def _init(settings: Phase2Settings | None = None) -> None:
    """Per-worker heavy setup (SDFs load from disk cache, so this is cheap).

    Spawned workers cannot see the parent's state, so the settings arrive as an
    initializer argument rather than through inherited module globals, and every
    worker task reads them back from ``_G``. Omitting them resolves from the
    environment, which is what a caller outside the pipeline gets.
    """
    settings = settings or Phase2Settings()
    # A second run in one process must inherit nothing from the first: _G holds
    # the previous subject's geometry and its derived coverage ceilings, and the
    # compiled kernels are keyed on shapes that a different subject can match.
    _G.clear()
    _G["settings"] = settings
    _setup_compile_cache()
    from aind_rutter.optimization.objectives.phase2 import clear_jit_cache
    from aind_rutter.optimization.objectives.probe_static import _build_probe_static
    from aind_rutter.optimization.pipeline.phase1_geometry import (
        build_coverage_data,
    )
    from aind_rutter.optimization.pipeline.runtime_adapter import (
        OptimizationRuntime,
    )

    clear_jit_cache()
    opt = OptimizationRuntime.from_config_path(settings.config, settings.holes)
    assets = opt.build_problem_assets(well_mode=settings.well, include_brain=True)
    # The FCL gate uses the same fixture names plus the world-frame implant BVH.
    # The implant remains excluded from soft SDF constraints because probes pass
    # through its bored holes; FCL is the ground-truth threading/body gate.
    fcl = opt.fcl_fixture_set(
        assets.fixtures, fixture_bvhs=assets.fixture_bvhs, include_implant=True
    )
    _G.update(
        probes=list(opt.probes),
        holes=list(opt.holes),
        sdf=assets.probe_sdfs,
        bvh=assets.probe_bvhs,
        fx=assets.fixtures,
        fbvh=assets.fixture_bvhs,
        fcl_fixtures=fcl.fixtures,
        fcl_fbvh=fcl.bvhs,
        brain_sdf=assets.brain_sdf,
        cov_data=None,
        n_probes=len(opt.probes),
        head_pitch_deg=opt.head_pitch_deg,
        build_static=_build_probe_static,
        build_cov=build_coverage_data,
    )


def _st_for_rec(rec: Phase2InputRecord):
    """Rebuild the per-probe static `st` for a Phase-1 pool record. Uses the
    arc assignment SAVED by Phase 1 (probe_to_hole / probe_to_arc_idx /
    arc_centroids_deg) so the geometry matches exactly what the pose `x` was
    optimized against — no re-seed, no frozenset-order ambiguity."""
    ha = MRVHoleAssignment(probe_to_hole=rec["probe_to_hole"])
    aa = MRVArcAssignment(
        probe_to_arc_idx=rec["probe_to_arc_idx"],
        arc_centroids_deg=list(rec["arc_centroids_deg"]),
    )
    return _G["build_static"](
        _G["probes"], _G["holes"], ha, aa, bvh_cache=_G["bvh"], sdf_by_name=_G["sdf"]
    )


def _cov_norm_kwargs(st) -> dict[str, object]:
    """make_phase2 kwargs for coverage normalization (ceilings + per-target
    weights), or empty when it is off. Ceilings/weights are per-probe-fixed so
    compute once per worker and cache in ``_G``."""
    if not _G["settings"].cov_norm:
        return {}
    if _G.get("cov_norm") is None:
        from aind_rutter.optimization.objectives.coverage import (
            coverage_ceiling_per_probe,
        )

        ceilings = tuple(
            float(c) for c in coverage_ceiling_per_probe(st, _G["cov_data"])
        )
        weights = tuple(float(p.coverage_weight) for p in _G["probes"])
        _G["cov_norm"] = (ceilings, weights)
    ceilings, weights = _G["cov_norm"]
    return {"coverage_ceilings": ceilings, "coverage_weights": weights}


def _live_group_sizes(slack_start: dict, live_rows) -> list[int]:
    """Per-group row counts as the solver sees them: the whole group when no rows
    were dropped, otherwise the live ones."""
    sizes = [a.size for a in slack_start.values()]
    live = np.asarray(live_rows)
    if live.all():
        return sizes
    edges = np.cumsum([0] + sizes)
    return [int(live[a:b].sum()) for a, b in itertools.pairwise(edges)]


def _phase2_one(rec: Phase2InputRecord) -> Phase2ResultRecord:
    from scipy.optimize import minimize

    from aind_rutter.optimization.objectives.coverage import (
        coverage_total_over_probes,
    )
    from aind_rutter.optimization.objectives.fcl_validator import make_fcl_validator
    from aind_rutter.optimization.objectives.variables import (
        _poses,
        worst_threading_g,
    )
    from aind_rutter.optimization.pipeline.phase1_geometry import phase1_bounds
    from aind_rutter.optimization.pipeline.phase2_diagnostics import (
        minimize_ipopt_logged,
        perturb_pose,
    )

    s = _G["settings"]
    idx, n_arcs, pose = rec["idx"], rec["n_arcs"], np.asarray(rec["pose"], float)
    rank = rec.get("rank", -1)
    st = _st_for_rec(rec)
    p2 = _build_problem(st, n_arcs)
    cov_data = _G["cov_data"]
    bounds = phase1_bounds(n_arcs, _G["n_probes"], _G["head_pitch_deg"])
    pose_start = perturb_pose(
        pose,
        n_arcs,
        _G["n_probes"],
        bounds,
        s.p2_perturb,
        (s.p2_perturb_seed, int(idx)),
    )
    v = make_fcl_validator(
        st, n_arcs, fixtures=tuple(_G["fcl_fixtures"]), fixture_bvhs=_G["fcl_fbvh"]
    )
    diag: dict[str, Any] = {}
    if s.p2_diag:
        diag["slack_start"] = {
            k: a.astype(np.float32) for k, a in p2["slack_parts"](pose_start).items()
        }
        diag["fcl_start"] = float(np.asarray(v.slacks(pose_start)).min())
        diag["fcl_pairs_start"] = v.violating_pairs(pose_start)
    t0 = time.perf_counter()
    if s.solver == "ipopt":
        from cyipopt import minimize_ipopt

        # phase1_bounds may be a scipy Bounds or a list of (lo, hi) tuples;
        # minimize_ipopt wants the latter.
        bnds: list[tuple[float, float]] = (
            list(zip(np.asarray(bounds.lb, float), np.asarray(bounds.ub, float)))
            if hasattr(bounds, "lb")
            else [(float(lo), float(hi)) for lo, hi in bounds]
        )
        ipopt_options = dict(
            hessian_approximation="limited-memory",
            limited_memory_max_history=s.ip_hist,
            mu_strategy=s.ip_mu,
            max_iter=s.p2_iter,
            tol=s.ip_tol,
            constr_viol_tol=s.ip_cvtol,
            acceptable_iter=s.ip_acc_iter,
            acceptable_tol=s.ip_acc_tol,
            acceptable_constr_viol_tol=s.ip_cvtol,
            print_level=0,
            sb="yes",
        )
        if s.ip_acc_tol > 1e-6:
            # Prefixed lookups fall back to the unprefixed value, so a loosened
            # acceptable_tol would also relax the restoration subproblem's exit —
            # the branch that reports local infeasibility.
            ipopt_options["resto.acceptable_iter"] = 0
        if s.p2_diag:
            from aind_rutter.optimization.objectives.phase2 import PADDED_SLACK

            res, diag["diag_hist"] = minimize_ipopt_logged(
                p2["fun"],
                pose_start,
                jac=p2["jac"],
                bounds=bnds,
                constraints=p2["constraints"],
                options=ipopt_options,
                group_sizes=_live_group_sizes(diag["slack_start"], p2["live_rows"]),
                padded_slack=PADDED_SLACK,
            )
        else:
            res = minimize_ipopt(
                p2["fun"],
                pose_start,
                jac=p2["jac"],
                bounds=bnds,
                constraints=p2["constraints"],  # dict ineq: g(x) >= 0
                options=ipopt_options,
            )
    else:
        # exactly one of hess / hessp is non-None per HESS mode (None ⇒ BFGS).
        mkw: dict[str, object] = {}
        if p2["hess"] is not None:
            mkw["hess"] = p2["hess"]
        if p2["hessp"] is not None:
            mkw["hessp"] = p2["hessp"]
        res = minimize(
            p2["fun"],
            pose_start,
            jac=p2["jac"],
            method="trust-constr",
            bounds=bounds,
            constraints=p2["constraints_nlc"],
            options=dict(
                maxiter=s.p2_iter,
                xtol=1e-6,
                gtol=1e-5,
                initial_tr_radius=1.0,
                verbose=0,
            ),
            **mkw,
        )
    dt = time.perf_counter() - t0
    fcl = float(np.asarray(v.slacks(res.x)).min())
    if s.p2_diag:
        diag["slack_end"] = {
            k: a.astype(np.float32) for k, a in p2["slack_parts"](res.x).items()
        }
        diag["fcl_pairs_end"] = v.violating_pairs(res.x)
        diag["slack_labels"] = p2["slack_labels"]
        diag["fcl_pair_names"] = list(v.pair_names)
    # Threading-feasibility of the SOLVED pose. IPOPT can return a pose that
    # satisfies probe↔probe FCL but failed its threading constraint (g >> 0,
    # shank nowhere near the bore). Record the worst g so main() can gate it.
    max_g_thread = worst_threading_g(st, np.asarray(res.x, float), n_arcs)
    Rs, ts, tp, mk = _poses(st, np.asarray(res.x, float), n_arcs)
    cov = float(coverage_total_over_probes(Rs, ts, tp, mk, cov_data, 41))
    return dict(
        idx=idx,
        rank=rank,
        n_arcs=n_arcs,
        fcl=fcl,
        max_g_thread=max_g_thread,
        coverage=cov,
        pose=res.x,
        pose_in=pose,  # Phase-1 input pose (full@end), persisted for inspection
        objective_p1=rec.get("objective"),  # the Phase-1 objective it was culled by
        nit=int(res.nit),
        solver_status=int(res.status),
        solver_message=str(res.message),
        secs=dt,
        hole=dict(rec["probe_to_hole"]),
        partition=rec["partition"],
        probe_to_arc_idx=rec["probe_to_arc_idx"],
        arc_centroids_deg=list(rec["arc_centroids_deg"]),
        min_clear=rec.get("min_clear"),
        pose_start=pose_start,
        perturb=(
            {"scale": s.p2_perturb, "seed": [s.p2_perturb_seed, int(idx)]}
            if s.p2_perturb > 0
            else None
        ),
        **diag,
    )


def _similarity(a: ProbeToHole, b: ProbeToHole) -> float:
    keys = set(a) | set(b)
    same = sum(1 for k in keys if a.get(k) == b.get(k))
    return same / max(len(keys), 1)


def _mmr_rank(rows: list[Phase2ResultRecord], lam: float) -> list[Phase2ResultRecord]:
    """Greedy MMR: coverage primary, penalize similarity to picked."""
    pool = list(rows)
    if not pool:
        return []
    cmax = max(r["coverage"] for r in pool)
    cmin = min(r["coverage"] for r in pool)
    span = max(cmax - cmin, 1e-9)
    picked: list[Phase2ResultRecord] = []
    while pool:
        if not picked:
            best = max(pool, key=lambda r: r["coverage"])
        else:

            def score(r):
                cn = (r["coverage"] - cmin) / span
                sim = max(_similarity(r["hole"], p["hole"]) for p in picked)
                return cn - lam * sim

            best = max(pool, key=score)
        picked.append(best)
        pool.remove(best)
    return picked


def _fmt(v: object, spec: str = "+.3f") -> str:
    """Format a possibly-None / NaN numeric for a progress line."""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    return "nan" if f != f else format(f, spec)


def _log_cand(k: int, n: int, r: "Phase2ResultRecord") -> None:
    """Emit one per-candidate progress line as it completes (completion order)."""
    print(
        f"  [{k:>4}/{n}] rank {str(r.get('rank')):<5} {r.get('n_arcs')}arc "
        f"fcl {_fmt(r.get('fcl'))} g {_fmt(r.get('max_g_thread'))} "
        f"cov {_fmt(r.get('coverage'), '.3f')} {_fmt(r.get('secs'), '.0f')}s",
        flush=True,
    )


def _build_problem(st, n_arcs: int) -> Phase2Problem:
    """The Phase-2 problem for one candidate under the worker's settings.

    ``_warmup`` and ``_phase2_one`` both build through here, so the warmup compiles
    the functions the solve calls.
    """
    from aind_rutter.optimization.objectives.phase2 import (
        Phase2Weights,
        make_phase2,
    )

    s = _G["settings"]
    if _G["cov_data"] is None:
        _G["cov_data"] = _G["build_cov"](_G["probes"], st)
    return make_phase2(
        st,
        n_arcs,
        coverage_data=_G["cov_data"],
        fixtures=tuple(_G["fx"]),
        weights=Phase2Weights(
            min_clearance_mm=s.minclear,
            lambda_margin_clear=s.lam_clear,
            tau_clear_mm=s.tau_clear,
            lambda_cov=s.cov_weight,
            cov_alpha=s.cov_alpha if s.cov_norm else 0.0,
            obb_slack_gain=s.obb_gain,
            smooth_clearance_reward=s.smooth_reward,
        ),
        brain_sdf=_G.get("brain_sdf"),
        hessian=s.hess,
        drop_padded_rows=s.drop_dead_rows,
        **cast(Any, _cov_norm_kwargs(st)),
    )


def _warmup(recs: list[Phase2InputRecord]) -> None:
    """Compile Phase 2 once per distinct n_arcs before any solve is timed.

    A thread pool warms in the parent; a process pool warms in one worker, which
    writes the disk compile cache the others load instead of all compiling at
    once (the OOM cause). Evaluates each function the configured solve calls
    without running the minimize.
    """
    s = _G["settings"]
    done = set()
    for r in recs:
        na = r["n_arcs"]
        if na in done:
            continue
        done.add(na)
        p2 = _build_problem(_st_for_rec(r), na)
        x = np.asarray(r["pose"], float)
        t0 = time.time()
        p2["fun"](x)
        p2["jac"](x)
        con = p2["constraints"][0]
        g = np.asarray(con["fun"](x))
        con["jac"](x)
        if s.solver == "trust-constr":
            # IPOPT runs limited-memory and never calls the exact Hessians.
            nlc_hess = p2["constraints_nlc"][0].hess
            if p2["hess"] is not None:
                p2["hess"](x)
                nlc_hess(x, np.ones_like(g))
            if p2["hessp"] is not None:
                p2["hessp"](x, x)
                nlc_hess(x, np.ones_like(g)).matvec(x)
        if s.p2_diag:
            p2["slack_parts"](x)
        print(f"  warmed n_arcs={na} in {time.time() - t0:.0f}s", flush=True)


def _require_gpu_headroom(n_workers: int) -> None:
    """Fail fast if free VRAM can't hold a ``POOL=process`` GPU run.

    One Phase-2 context's measured footprint is ~2.4 GB resident (clean card).
    A stale CUDA context, a leftover MPS server, or another GPU user silently
    eats that budget — and the process pool then spins up workers that can't
    init CUDA and respawns them in a "no supported devices" loop. Check up front
    and abort with a diagnostic instead. Tunable via env; skipped on CPU/thread.
    """
    if _PLATFORM != "cuda" or POOL == "thread":
        return
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if not smi:
        return  # can't check — proceed rather than block
    # Read the JAX GPU's (index 0) free memory a few times and take the MAX. A
    # single spurious/transient low reading (observed under MPS) must NOT
    # false-abort, while a real orphan/leftover context holds memory steadily
    # across all reads. (`-i 0` assumes the JAX device is index 0 — true here.)
    frees: list[int] = []
    for _ in range(3):
        try:
            out = subprocess.run(
                [
                    smi,
                    "-i",
                    "0",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            frees.append(int(out.stdout.split()[0]))
        except Exception:
            pass
        time.sleep(0.3)
    if not frees:
        return  # probe failed — don't block on a flaky reading
    free_mb = max(frees)
    # ~2.4 GB measured resident per context (clean card, incl. cold compile);
    # 0.4 GB slack. So 3 workers need ~7.6 GB (fits a ~8 GB-free desktop card),
    # and the check fires only when something has eaten the budget.
    per_worker_gb = float(_os.environ.get("P2_PER_WORKER_GB", "2.4"))
    headroom_gb = float(_os.environ.get("P2_HEADROOM_GB", "0.4"))
    need_mb = int((n_workers * per_worker_gb + headroom_gb) * 1024)
    if free_mb < need_mb:
        raise SystemExit(
            f"Phase-2 ABORT: {free_mb} MiB free GPU < ~{need_mb} MiB needed for "
            f"{n_workers} process workers (~{per_worker_gb} GB each + "
            f"{headroom_gb} GB headroom). MOST COMMON CAUSE: "
            f"XLA_PYTHON_CLIENT_PREALLOCATE not 'false' in the LAUNCH env — each "
            f"context then preallocates ~0.75 of the card (~7.6 GB) and only one "
            f"fits. Fix: `export XLA_PYTHON_CLIENT_PREALLOCATE=false` before "
            f"launching (must precede python; setting it in code is too late). "
            f"Else: a stale CUDA context / leftover nvidia-cuda-mps-server / "
            f"another GPU user — free the card (`nvidia-smi`) or lower WORKERS. "
            f"Tune the estimate with P2_PER_WORKER_GB / P2_HEADROOM_GB."
        )
    print(
        f"[gpu] {free_mb} MiB free ≥ ~{need_mb} MiB for {n_workers} workers — ok",
        flush=True,
    )


def solve_candidates(
    recs: list[Phase2InputRecord],
    settings: Phase2Settings,
    *,
    on_result: Callable[[int, int, Phase2ResultRecord], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Phase2ResultRecord]:
    """Solve every candidate, returning the results in rank order.

    ``on_result`` sees each result as it completes, which is not rank order, and
    ``log`` receives coarse progress. Both default to silent: an imported caller
    gets results rather than stdout.
    """
    note = log or (lambda _msg: None)
    # Pay the one-time compile ONCE. Threads share the PARENT's context, so warm
    # there. Spawned workers each have their own context and can't see the
    # parent's in-memory cache — so for a PROCESS pool we warm inside ONE worker,
    # which writes the persistent disk cache; the rest LOAD from it. This keeps
    # the parent GPU-free (no idle ~2.4GiB context), leaving HBM for an extra
    # worker. Largest n_arcs first so the heaviest compile is the one warmed.
    nstr = sorted({r["n_arcs"] for r in recs})
    if POOL == "thread":
        # Threads share the parent's state, so the parent holds the settings the
        # worker tasks read back from _G. The setup itself is not optional —
        # only the compile that warmup pays up front, which WARMUP=0 moves onto
        # the first candidate instead.
        _init(settings)
        if settings.warmup:
            tw = time.time()
            note(f"warming (in-parent, single-threaded) for n_arcs {nstr}...")
            _warmup(recs)
            note(f"  warmup {time.time() - tw:.0f}s")
        t0 = time.time()
        # One GPU context shared across threads (no per-worker memory).
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[Phase2ResultRecord] = []
        with ThreadPoolExecutor(settings.workers) as ex:
            futs = [ex.submit(_phase2_one, r) for r in recs]
            for k, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                if on_result is not None:
                    on_result(k, len(recs), r)
        wall = time.time() - t0  # processing only (excludes warmup)
    else:
        _require_gpu_headroom(settings.workers)
        ctx = get_context("spawn")
        with ctx.Pool(
            settings.workers, initializer=_init, initargs=(settings,)
        ) as pool:
            if settings.warmup:
                tw = time.time()
                note(
                    f"warming (one worker → disk cache; parent stays GPU-free) "
                    f"for n_arcs {nstr}..."
                )
                pool.apply(_warmup, (recs,))
                note(f"  warmup {time.time() - tw:.0f}s")
            t0 = time.time()
            results = []
            for k, r in enumerate(pool.imap_unordered(_phase2_one, recs), 1):
                results.append(r)
                if on_result is not None:
                    on_result(k, len(recs), r)
            wall = time.time() - t0  # processing only (excludes warmup)
    # Results arrive in completion order; restore rank order so all downstream
    # selection and reporting is independent of how the pool interleaved.
    results.sort(key=lambda r: r.get("rank", 0))
    n = max(len(results), 1)
    compute = float(sum(r["secs"] for r in results))
    note(
        f"  {wall / 60:.2f} min wall; {compute:.0f}s total compute; "
        f"{compute / max(wall, 1e-9):.1f}x effective parallelism "
        f"({compute / n:.0f}s/cand); "
        f"throughput {len(results) / max(wall, 1e-9):.2f} cand/s"
    )
    note(
        f"  per-cand secs (1st incl compile if WARMUP off): "
        f"{[round(r['secs']) for r in results]}"
    )
    return results


def run(
    recs: list[Phase2InputRecord],
    settings: Phase2Settings,
    *,
    on_result: Callable[[int, int, Phase2ResultRecord], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> Phase2HandoffPayload:
    """Solve, classify and rank a selection of candidates.

    The returned payload is what ``pipeline.emit`` consumes; writing it to disk
    is the caller's business. Each record carries pose, probe_to_hole,
    probe_to_arc_idx, arc_centroids_deg and n_arcs — enough to rebuild the plan
    and emit a config per candidate.
    """
    note = log or (lambda _msg: None)
    note(
        f"Parallel Phase 2 [{_PLATFORM}]: {len(recs)} cands, {settings.workers}"
        f" workers x {_THREADS} thr, maxiter={settings.p2_iter},"
        f" lam={settings.lam_clear} tau={settings.tau_clear},"
        f" diag={settings.p2_diag}, perturb={settings.p2_perturb}"
    )
    results = solve_candidates(recs, settings, on_result=on_result, log=log)
    feasible = classify_results(results, settings)
    payload: Phase2HandoffPayload = dict(
        ranked=_mmr_rank(feasible, settings.mmr_lambda),
        all=results,
        config=handoff_config(settings),
    )
    if settings.p2_diag:
        from aind_rutter.optimization.objectives.phase2 import SLACK_GROUPS

        # Labels are identical across candidates of one probe/fixture set: store once.
        labels = [r.pop("slack_labels", None) for r in results]
        names = [r.pop("fcl_pair_names", None) for r in results]
        payload["config"].update(
            slack_groups=list(SLACK_GROUPS),
            slack_labels=next((x for x in labels if x), None),
            fcl_pair_names=next((x for x in names if x), None),
        )
    return payload


def _print_report(payload: Phase2HandoffPayload, settings: Phase2Settings) -> None:
    """Print the keep-band summary and the two per-candidate tables."""
    results = payload["all"]
    ranked = payload["ranked"]
    feasible = [r for r in results if r["kept"]]
    print(
        f"  KEEP band FCL>=-{settings.fcl_tol} AND g<=+{settings.g_tol}: "
        f"{len(feasible)}/{len(results)}  (STRICT FCL>=-1e-4 AND g<=0: "
        f"{sum(1 for r in results if r['strict_feasible'])})"
    )
    g_kept = [r.get("max_g_thread", float("nan")) for r in feasible]
    if g_kept:
        print(
            f"  kept-plan threading g: max={np.nanmax(g_kept):+.3f} "
            f"median={np.nanmedian(g_kept):+.3f} (g<=0 ⇒ shank inside inset bore)"
        )

    # Stratification view: feasibility + post-Phase-2 coverage vs rerank rank,
    # so we can see where in the distribution good feasibles stop appearing.
    print("\n=== per-candidate (rank-ordered) ===")
    print(
        f"{'rank':>5} {'cand':>6} {'fcl':>8} {'max_g':>7} "
        f"{'keep':>4} {'strict':>6} {'coverage':>9}"
    )
    for r in sorted(results, key=lambda r: r["rank"]):
        print(
            f"{r['rank']:>5} {r['idx']:>6} {r['fcl']:>+8.4f} "
            f"{r.get('max_g_thread', float('nan')):>+7.3f} "
            f"{'Y' if r['kept'] else 'n':>4} "
            f"{'Y' if r['strict_feasible'] else 'n':>6} {r['coverage']:>9.3f}"
        )

    print(
        f"\n=== handoff ranking (MMR lam={settings.mmr_lambda}, "
        f"coverage + diversity) ==="
    )
    print(f"{'#':>3} {'cand':>6} {'coverage':>9} {'fcl':>8} {'maxsim_prev':>11}")
    for i, r in enumerate(ranked):
        sim = max(_similarity(r["hole"], p["hole"]) for p in ranked[:i]) if i else 0.0
        print(
            f"{i + 1:>3} {r['idx']:>6} {r['coverage']:>9.3f} {r['fcl']:>+8.4f} "
            f"{sim:>11.2f}"
        )


def _flush_print(msg: str) -> None:
    """Progress sink for ``run``; flushed so a piped log stays live."""
    print(msg, flush=True)


def main() -> int:
    settings = Phase2Settings()
    recs = select_records(read_pool(settings.poses)["records"], settings)
    payload = run(recs, settings, on_result=_log_cand, log=_flush_print)
    _print_report(payload, settings)
    write_handoff(settings.out, payload)
    print(
        f"\nsaved → {settings.out}  ({len(payload['ranked'])} feasible ranked, "
        f"{len(payload['all'])} total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
