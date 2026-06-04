"""Parallel Phase 2 across candidates + diversity-aware (MMR) handoff ranking.

trust-constr is single-threaded, but candidates are independent — so we run
Phase 2 in a process pool (pinned to 1 BLAS thread/worker to avoid 24x24
oversubscription). Each worker builds the heavy setup once (SDFs load from disk
cache), then polishes its candidates with the balanced soft-bonus config.

Then: keep FCL >= -TOL (human-fixable), and rank by MMR — greedily pick highest
post-Phase-2 coverage, penalizing similarity (shared probe->hole fraction) to
already-picked plans, so the ranked handoff is high-coverage AND diverse.

Run:  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.phase2_parallel
Env:  TOPK (default 80), WORKERS (default 16), P2_ITER (200), MINCLEAR (0.2),
      LAM_CLEAR (5.0), TAU_CLEAR (0.8), FCL_TOL (0.2), MMR_LAMBDA (0.5)
"""

from __future__ import annotations

import os as _os

# Set BLAS/OMP threads-per-worker BEFORE jax/numpy import (THREADS=1 pins each
# worker single-thread to avoid Nworkers x Ncore oversubscription; raise it to
# test whether some oversubscription fills memory-bound stalls). Spawned workers
# inherit this env and re-run this block.
_THREADS = _os.environ.get("THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, _THREADS)
# Platform: PLATFORM=cpu (default) or gpu. On GPU, never preallocate (each
# worker would grab the whole 10GB) and cap each worker's fraction so a few
# workers share the GPU without OOM. Spawned workers inherit this env.
_PLATFORM = _os.environ.get("PLATFORM", "cpu")
if _PLATFORM in ("gpu", "cuda"):
    _PLATFORM = "cuda"  # JAX backend name
    _os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",
                           _os.environ.get("GPU_MEM_FRACTION", "0.18"))
_os.environ.setdefault("JAX_PLATFORMS", _PLATFORM)

import pickle
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

TOPK = int(_os.environ.get("TOPK", "80"))
WORKERS = int(_os.environ.get("WORKERS", "16"))
P2_ITER = int(_os.environ.get("P2_ITER", "200"))
MINCLEAR = float(_os.environ.get("MINCLEAR", "0.2"))
LAM_CLEAR = float(_os.environ.get("LAM_CLEAR", "5.0"))
TAU_CLEAR = float(_os.environ.get("TAU_CLEAR", "0.8"))
FCL_TOL = float(_os.environ.get("FCL_TOL", "0.2"))
MMR_LAMBDA = float(_os.environ.get("MMR_LAMBDA", "0.5"))
WARMUP = _os.environ.get("WARMUP", "1") == "1"
# POOL=process: N worker processes (N GPU contexts → memory-limited on GPU).
# POOL=thread: ONE process/GPU context, N threads sharing it — trust-constr's
# JAX evals release the GIL, so threads overlap and fill the GPU's idle gaps
# (the scipy subproblem is CPU-bound and leaves the GPU idle). No extra memory.
POOL = _os.environ.get("POOL", "process")

_G: dict = {}


def _init():
    """Per-worker heavy setup (SDFs load from disk cache, so this is cheap)."""
    from aind_low_point.config import ConfigModel
    from aind_low_point.optimization.headstages import make_fcl_bvh
    from aind_low_point.optimization.holes import load_holes
    from aind_low_point.optimization.joint_rerank import _build_probe_static
    from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
    from aind_low_point.runtime import build_runtime_from_config
    from aind_low_point.runtime.transforms import compile_all_transforms
    from scripts.run_optimizer import _probe_static_info, _transform_holes
    from scripts.run_phase1_sample import (
        build_coverage_data,
        build_fixture_sdf_data,
    )

    cfg = ConfigModel.from_yaml("examples/836656-config-T12.yml")
    rt = build_runtime_from_config(cfg)
    probes = [_probe_static_info(rt.plan_state, rt, n)
              for n in rt.plan_state.probes]
    holes = load_holes(Path("scratch/0283-300-04.holes.yml"))
    comp = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in comp:
        R, t = comp["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf = {p.name: build_probe_sdf_from_alpha_wrap(
        rt.asset_catalog.get_geometry(f"probe:{p.kind}").raw) for p in probes}
    bvh = {p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
           for p in probes}
    fx = build_fixture_sdf_data(rt)
    fbvh = {f.name: make_fcl_bvh(rt.asset_catalog.get_geometry(f.name).raw)
            for f in fx}
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    _G.update(probes=probes, holes=holes, sdf=sdf, bvh=bvh, fx=fx, fbvh=fbvh,
              pool=pool, cov_data=None, n_probes=len(probes),
              build_static=_build_probe_static, build_cov=build_coverage_data)


def _phase2_one(rec):
    from scipy.optimize import minimize

    from aind_low_point.optimization.coverage_jax import (
        coverage_total_over_probes,
    )
    from aind_low_point.optimization.stage3_phase2_jax import (
        Phase2Weights,
        make_phase2,
    )
    from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
    from scripts.ingest_analysis import _poses
    from scripts.run_phase1_sample import phase1_bounds

    idx, n_arcs, pose = rec["idx"], rec["n_arcs"], np.asarray(rec["pose"], float)
    rank = rec.get("rank", -1)
    c = _G["pool"]["candidates"][idx]
    st = _G["build_static"](_G["probes"], _G["holes"], c.ha, c.aa,
                            bvh_cache=_G["bvh"], sdf_by_name=_G["sdf"])
    if _G["cov_data"] is None:
        _G["cov_data"] = _G["build_cov"](_G["probes"], st)
    cov_data = _G["cov_data"]
    bounds = phase1_bounds(n_arcs, _G["n_probes"])
    p2 = make_phase2(st, n_arcs, coverage_data=cov_data, fixtures=tuple(_G["fx"]),
                     weights=Phase2Weights(min_clearance_mm=MINCLEAR,
                                           lambda_margin_clear=LAM_CLEAR,
                                           tau_clear_mm=TAU_CLEAR))
    t0 = time.perf_counter()
    res = minimize(p2["fun"], pose, jac=p2["jac"], method="trust-constr",
                   bounds=bounds, constraints=p2["constraints_nlc"],
                   options=dict(maxiter=P2_ITER, xtol=1e-6, gtol=1e-5,
                                initial_tr_radius=1.0, verbose=0))
    dt = time.perf_counter() - t0
    v = make_fcl_validator(st, n_arcs, fixtures=tuple(_G["fx"]),
                           fixture_bvhs=_G["fbvh"])
    fcl = float(np.asarray(v.slacks(res.x)).min())
    Rs, ts, tp, mk = _poses(st, np.asarray(res.x, float), n_arcs)
    cov = float(coverage_total_over_probes(Rs, ts, tp, mk, cov_data, 41))
    hole = dict(c.ha.probe_to_hole)
    return dict(idx=idx, rank=rank, n_arcs=n_arcs, fcl=fcl, coverage=cov,
                pose=res.x, nit=int(res.nit), secs=dt, hole=hole)


def _similarity(a, b):
    keys = set(a) | set(b)
    same = sum(1 for k in keys if a.get(k) == b.get(k))
    return same / max(len(keys), 1)


def _mmr_rank(rows, lam):
    """Greedy MMR: coverage primary, penalize similarity to picked."""
    pool = list(rows)
    cmax = max(r["coverage"] for r in pool)
    cmin = min(r["coverage"] for r in pool)
    span = max(cmax - cmin, 1e-9)
    picked = []
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


def _warmup(recs):
    """Compile Phase 2 once per distinct n_arcs in the PARENT so the disk
    compile cache is warm; spawned workers then LOAD instead of all compiling
    simultaneously (the OOM cause). Evals the jit callables (triggers compile)
    without running the full minimize."""
    from aind_low_point.optimization.stage3_phase2_jax import (
        Phase2Weights,
        make_phase2,
    )
    _init()
    done = set()
    for r in recs:
        na = r["n_arcs"]
        if na in done:
            continue
        done.add(na)
        c = _G["pool"]["candidates"][r["idx"]]
        st = _G["build_static"](_G["probes"], _G["holes"], c.ha, c.aa,
                                bvh_cache=_G["bvh"], sdf_by_name=_G["sdf"])
        if _G["cov_data"] is None:
            _G["cov_data"] = _G["build_cov"](_G["probes"], st)
        p2 = make_phase2(st, na, coverage_data=_G["cov_data"],
                         fixtures=tuple(_G["fx"]),
                         weights=Phase2Weights(min_clearance_mm=MINCLEAR,
                                               lambda_margin_clear=LAM_CLEAR,
                                               tau_clear_mm=TAU_CLEAR))
        x = np.asarray(r["pose"], float)
        t0 = time.time()
        p2["fun"](x)
        p2["jac"](x)
        nlc = p2["constraints_nlc"][0]
        nlc.fun(x)
        nlc.jac(x)
        print(f"  warmed n_arcs={na} in {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    rer = pickle.load(open("scratch/full_rerank_0283.pkl", "rb"))
    all_recs = rer["records"]
    # RANKS overrides top-TOPK: an explicit (possibly stratified) rank list, so
    # we can probe where in the rank distribution good feasibles stop appearing
    # rather than guessing a fixed cutoff. Each rec carries its rank for output.
    ranks_env = _os.environ.get("RANKS", "")
    if ranks_env:
        sel = [int(x) for x in ranks_env.split(",") if x.strip()]
        sel = [i for i in sel if i < len(all_recs)]
    else:
        sel = list(range(min(TOPK, len(all_recs))))
    recs = [dict(idx=all_recs[i]["idx"], n_arcs=all_recs[i]["n_arcs"],
                 pose=all_recs[i]["pose"], rank=i) for i in sel]
    print(f"Parallel Phase 2 [{_PLATFORM}]: {len(recs)} cands, {WORKERS} workers"
          f" x {_THREADS} thr, maxiter={P2_ITER}, lam={LAM_CLEAR} tau={TAU_CLEAR}")
    # Pay the one-time startup ONCE, single-threaded: compiles/loads the graph
    # into the in-process _JIT_CACHE + onto the GPU, and builds cov_data — so
    # threads (POOL=thread) all hit the warm cache (no compile race), and
    # spawned workers (POOL=process) load from the disk compile cache.
    tw = time.time()
    if POOL == "thread" or WARMUP:
        print(f"warming (single-threaded) for n_arcs "
              f"{sorted({r['n_arcs'] for r in recs})}...", flush=True)
        _warmup(recs)
        print(f"  warmup {time.time()-tw:.0f}s", flush=True)
    t0 = time.time()
    if POOL == "thread":
        # One GPU context shared across threads (no per-worker memory); the
        # cache is already warm so each thread starts at steady speed.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(WORKERS) as ex:
            results = list(ex.map(_phase2_one, recs))
    else:
        ctx = get_context("spawn")
        with ctx.Pool(WORKERS, initializer=_init) as pool:
            results = pool.map(_phase2_one, recs)
    wall = time.time() - t0  # processing only (excludes warmup)
    compute = float(sum(r["secs"] for r in results))
    print(f"  {wall/60:.2f} min wall; {compute:.0f}s total compute; "
          f"{compute/max(wall,1e-9):.1f}x effective parallelism "
          f"({compute/len(results):.0f}s/cand); throughput {len(results)/wall:.2f} cand/s")
    print(f"  per-cand secs (1st incl compile if WARMUP off): "
          f"{[round(r['secs']) for r in results]}")

    feas = [r for r in results if r["fcl"] >= -FCL_TOL]
    print(f"  FCL >= -{FCL_TOL}: {len(feas)}/{len(results)} "
          f"(strictly feasible: {sum(1 for r in results if r['fcl'] >= -1e-4)})")

    # Stratification view: feasibility + post-Phase-2 coverage vs rerank rank,
    # so we can see where in the distribution good feasibles stop appearing.
    by_rank = sorted(results, key=lambda r: r["rank"])
    print(f"\n=== per-candidate (rank-ordered) ===")
    print(f"{'rank':>5} {'cand':>6} {'fcl':>8} {'≥-.2':>4} {'coverage':>9}")
    for r in by_rank:
        print(f"{r['rank']:>5} {r['idx']:>6} {r['fcl']:>+8.4f} "
              f"{'Y' if r['fcl'] >= -FCL_TOL else 'n':>4} {r['coverage']:>9.3f}")
    ranked = _mmr_rank(feas, MMR_LAMBDA)

    print(f"\n=== handoff ranking (MMR lam={MMR_LAMBDA}, coverage + diversity) ===")
    print(f"{'#':>3} {'cand':>6} {'coverage':>9} {'fcl':>8} {'maxsim_prev':>11}")
    for i, r in enumerate(ranked):
        sim = (max(_similarity(r["hole"], p["hole"]) for p in ranked[:i])
               if i else 0.0)
        print(f"{i+1:>3} {r['idx']:>6} {r['coverage']:>9.3f} {r['fcl']:>+8.4f} "
              f"{sim:>11.2f}")

    out = Path("scratch/phase2_handoff.pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(ranked=ranked, all=results,
                         config=dict(topk=TOPK, minclear=MINCLEAR,
                                     lam_clear=LAM_CLEAR, tau_clear=TAU_CLEAR,
                                     p2_iter=P2_ITER, fcl_tol=FCL_TOL,
                                     mmr_lambda=MMR_LAMBDA)), f)
    print(f"\nsaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
