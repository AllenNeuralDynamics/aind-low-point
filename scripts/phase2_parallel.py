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
    _os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _default_frac = "0.9" if POOL == "thread" else "0.18"
    _os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        _os.environ.get("GPU_MEM_FRACTION", _default_frac),
    )
_os.environ.setdefault("JAX_PLATFORMS", _PLATFORM)

import pickle  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

TOPK = int(_os.environ.get("TOPK", "80"))
# Default 8: the GPU-thread-shared sweet spot from the bandwidth bake-off (W=8 ≈
# 0.067 cand/s on the shared HBM; the knee is past 4 threads). POOL=thread means
# all 8 share one GPU context.
WORKERS = int(_os.environ.get("WORKERS", "4"))
P2_ITER = int(_os.environ.get("P2_ITER", "200"))
MINCLEAR = float(_os.environ.get("MINCLEAR", "0.2"))
LAM_CLEAR = float(_os.environ.get("LAM_CLEAR", "5.0"))
TAU_CLEAR = float(_os.environ.get("TAU_CLEAR", "0.8"))
FCL_TOL = float(_os.environ.get("FCL_TOL", "0.2"))
MMR_LAMBDA = float(_os.environ.get("MMR_LAMBDA", "0.5"))
# Second-order mode for trust-constr: "none" (BFGS approx, default/fast),
# "dense" (exact n×n Hessian — ~44x slower), or "hessp" (exact Hessian-VECTOR
# products via JAX — ~2x a gradient, the affordable exact second-order).
HESS = _os.environ.get("HESS", "none").lower()
# Solver: "ipopt" (cyipopt interior-point, default) or "trust-constr" (scipy).
# IPOPT's restoration phase reaches feasibility from infeasible starts far
# better than trust-constr — on the tuned-545 stalled set it rescues 9/11 vs
# trust-constr-BFGS 0/11. We run it LIMITED-MEMORY (L-BFGS): no second-order, so
# (a) it evaluates only the first-order obj/grad/constraint/Jacobian on the GPU
# — the same surface trust-constr-BFGS uses, NO exposure to the `_slacks_hessp`
# GPU autotuner crash — and (b) it BEATS exact-Hessian IPOPT here (more rescues,
# doesn't drive feasible cands into collision, 2-5x faster) because this NLP is
# nonconvex and the exact Lagrangian Hessian is indefinite away from the optimum.
SOLVER = _os.environ.get("SOLVER", "ipopt").lower()
# IPOPT knobs (only consulted when SOLVER=ipopt). max_iter reuses P2_ITER.
IP_HIST = int(_os.environ.get("IP_HIST", "6"))  # limited_memory_max_history
IP_MU = _os.environ.get("IP_MU", "adaptive")  # mu_strategy
# Feasibility tolerances are on the UNSCALED (gain-carrying) constraint: the
# body/voxel rows are mm-native (gain 1) but the OBB rows carry gain 100 (0.01mm
# units), so a threshold X bounds the mm rows at X mm. 1e-4 mm is inside the FCL
# −1e-4 gate; the default acceptable_constr_viol_tol of 1e-2 (=0.01mm slack on
# the mm rows) is a footgun → tightened to match.
IP_CVTOL = float(_os.environ.get("IP_CVTOL", "1e-4"))  # constr_viol_tol (mm)
IP_ACC_ITER = int(_os.environ.get("IP_ACC_ITER", "25"))  # 0 disables early stop
# Subject is config-driven (generalizes across subjects): CONFIG selects the
# YAML, HOLES the implant-bore file (placed by the config's implant_to_lps).
CONFIG = _os.environ.get("CONFIG", "examples/836656-config-T12.yml")
HOLES = _os.environ.get("HOLES", "scratch/0283-300-04.holes.yml")
# Phase-1 pool (mrv_pool_run output): records carry probe_to_hole, partition,
# probe_to_arc_idx, arc_centroids_deg, x (Phase-1 pose), min_clear. Phase 2
# rebuilds `st` from the saved arc assignment — NO full_polish_0283 dependency.
POSES_PKL = _os.environ.get("POSES", "scratch/mrv_pool_results.pkl")
# Selection metric for the top-TOPK handed to Phase 2. Default min_clear (soft
# clearance) — there is NO FCL cull between Phase 1 and 2; FCL runs once at the
# end as the ground-truth gate.
SELECT_BY = _os.environ.get("SELECT_BY", "min_clear")
OUT_PKL = _os.environ.get("OUT", "scratch/phase2_handoff.pkl")
WELL = _os.environ.get("WELL", "thick").lower()  # thin | thick (thick = tuned)
WARMUP = _os.environ.get("WARMUP", "1") == "1"
# Coverage normalization (mirror of the Phase-1 driver): divide each probe's
# coverage by its achievable ceiling, blend average vs worst region by COV_ALPHA
# in [0,1], and apply the target spec's per-target priority weights. COV_WEIGHT
# is the overall coverage-vs-clearance gain (coverage is a [0,1] scalar). Must
# match the Phase-1 settings for the objective to be consistent across stages.
COV_NORM = _os.environ.get("COV_NORM", "0") == "1"
COV_ALPHA = float(_os.environ.get("COV_ALPHA", "0.2"))
COV_WEIGHT = float(_os.environ.get("COV_WEIGHT", "1.0"))
# POOL is read at the top (before the platform block) so the GPU memory fraction
# can depend on it.

_G: dict = {}


def _setup_compile_cache():
    """Enable JAX's PERSISTENT (on-disk) compile cache, shared by every process.

    Without this, each spawned worker recompiles the Phase-2 graph from scratch
    (~20s) — the parent warmup only ever warmed the parent's in-memory cache,
    which a spawn()ed worker can't see. With a disk cache, a single warmup (in
    ONE worker) writes the compiled executables to disk and all other workers
    LOAD them — so the parent never needs a GPU context."""
    import jax

    cache_dir = _os.environ.get("JAX_CACHE_DIR", "scratch/jax_p2_cache")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


def _init():
    """Per-worker heavy setup (SDFs load from disk cache, so this is cheap)."""
    _setup_compile_cache()
    from aind_low_point.config import ConfigModel
    from aind_low_point.optimization.headstages import make_fcl_bvh
    from aind_low_point.optimization.holes import load_holes
    from aind_low_point.optimization.joint_rerank import _build_probe_static
    from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
    from aind_low_point.runtime import build_runtime_from_config
    from aind_low_point.runtime.transforms import compile_all_transforms
    from scripts.run_optimizer import (
        _probe_static_info,
        _transform_holes,
        retro_opts_from_env,
    )
    from scripts.run_phase1_sample import (
        build_coverage_data,
        build_fixture_sdf_data,
        maybe_build_brain_sdf,
    )

    cfg = ConfigModel.from_yaml(CONFIG)
    rt = build_runtime_from_config(cfg)
    _ro = retro_opts_from_env(rt)
    probes = [
        _probe_static_info(rt.plan_state, rt, n, _ro) for n in rt.plan_state.probes
    ]
    holes = load_holes(Path(HOLES))
    comp = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in comp:
        R, t = comp["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf = {
        p.name: build_probe_sdf_from_alpha_wrap(
            rt.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    bvh = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fx = build_fixture_sdf_data(rt)
    if WELL == "thick":
        from scripts.thick_well_sdf import fit_well_cone, make_thick_well_sdf

        mesh = rt.asset_catalog.get_geometry("well").raw
        well_thin = next(f for f in fx if f.name == "well")
        well_thick = make_thick_well_sdf(mesh, well_thin, cone=fit_well_cone(mesh))
        fx = tuple(well_thick if f.name == "well" else f for f in fx)
    fbvh = {f.name: make_fcl_bvh(rt.asset_catalog.get_geometry(f.name).raw) for f in fx}
    brain_sdf = maybe_build_brain_sdf(rt, comp)
    _G.update(
        probes=probes,
        holes=holes,
        sdf=sdf,
        bvh=bvh,
        fx=fx,
        fbvh=fbvh,
        brain_sdf=brain_sdf,
        cov_data=None,
        n_probes=len(probes),
        build_static=_build_probe_static,
        build_cov=build_coverage_data,
    )


def _st_for_rec(rec):
    """Rebuild the per-probe static `st` for a Phase-1 pool record. Uses the
    arc assignment SAVED by Phase 1 (probe_to_hole / probe_to_arc_idx /
    arc_centroids_deg) so the geometry matches exactly what the pose `x` was
    optimized against — no re-seed, no frozenset-order ambiguity."""
    from types import SimpleNamespace

    ha = SimpleNamespace(probe_to_hole=rec["probe_to_hole"])
    aa = SimpleNamespace(
        probe_to_arc_idx=rec["probe_to_arc_idx"],
        arc_centroids_deg=list(rec["arc_centroids_deg"]),
    )
    return _G["build_static"](
        _G["probes"], _G["holes"], ha, aa, bvh_cache=_G["bvh"], sdf_by_name=_G["sdf"]
    )


def _cov_norm_kwargs(st):
    """make_phase2 kwargs for coverage normalization (ceilings + per-target
    weights), or empty when COV_NORM is off. Ceilings/weights are per-probe-fixed
    so compute once per worker and cache in ``_G``."""
    if not COV_NORM:
        return {}
    if _G.get("cov_norm") is None:
        from aind_low_point.optimization.coverage_jax import (
            coverage_ceiling_per_probe,
        )

        ceilings = tuple(
            float(c) for c in coverage_ceiling_per_probe(st, _G["cov_data"])
        )
        weights = tuple(float(p.coverage_weight) for p in _G["probes"])
        _G["cov_norm"] = (ceilings, weights)
    ceilings, weights = _G["cov_norm"]
    return {"coverage_ceilings": ceilings, "coverage_weights": weights}


def _phase2_one(rec):
    from scipy.optimize import minimize

    from aind_low_point.optimization.coverage_jax import (
        coverage_total_over_probes,
    )
    from aind_low_point.optimization.optimizer_vars import _poses
    from aind_low_point.optimization.stage3_phase2_jax import (
        Phase2Weights,
        make_phase2,
    )
    from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
    from scripts.run_phase1_sample import phase1_bounds

    idx, n_arcs, pose = rec["idx"], rec["n_arcs"], np.asarray(rec["pose"], float)
    rank = rec.get("rank", -1)
    st = _st_for_rec(rec)
    if _G["cov_data"] is None:
        _G["cov_data"] = _G["build_cov"](_G["probes"], st)
    cov_data = _G["cov_data"]
    bounds = phase1_bounds(n_arcs, _G["n_probes"])
    p2 = make_phase2(
        st,
        n_arcs,
        coverage_data=cov_data,
        fixtures=tuple(_G["fx"]),
        weights=Phase2Weights(
            min_clearance_mm=MINCLEAR,
            lambda_margin_clear=LAM_CLEAR,
            tau_clear_mm=TAU_CLEAR,
            lambda_cov=COV_WEIGHT,
            cov_alpha=COV_ALPHA if COV_NORM else 0.0,
        ),
        brain_sdf=_G.get("brain_sdf"),
        hessian=HESS,
        **_cov_norm_kwargs(st),
    )
    t0 = time.perf_counter()
    if SOLVER == "ipopt":
        from cyipopt import minimize_ipopt

        # phase1_bounds may be a scipy Bounds or a list of (lo, hi) tuples;
        # minimize_ipopt wants the latter.
        bnds = (
            list(zip(np.asarray(bounds.lb, float), np.asarray(bounds.ub, float)))
            if hasattr(bounds, "lb")
            else [tuple(map(float, t)) for t in bounds]
        )
        res = minimize_ipopt(
            p2["fun"],
            pose,
            jac=p2["jac"],
            bounds=bnds,
            constraints=p2["constraints"],  # dict ineq: g(x) >= 0
            options=dict(
                hessian_approximation="limited-memory",
                limited_memory_max_history=IP_HIST,
                mu_strategy=IP_MU,
                max_iter=P2_ITER,
                tol=1e-6,
                constr_viol_tol=IP_CVTOL,
                acceptable_iter=IP_ACC_ITER,
                acceptable_constr_viol_tol=IP_CVTOL,
                print_level=0,
                sb="yes",
            ),
        )
    else:
        # exactly one of hess / hessp is non-None per HESS mode (None ⇒ BFGS).
        mkw = {}
        if p2["hess"] is not None:
            mkw["hess"] = p2["hess"]
        if p2["hessp"] is not None:
            mkw["hessp"] = p2["hessp"]
        res = minimize(
            p2["fun"],
            pose,
            jac=p2["jac"],
            method="trust-constr",
            bounds=bounds,
            constraints=p2["constraints_nlc"],
            options=dict(
                maxiter=P2_ITER, xtol=1e-6, gtol=1e-5, initial_tr_radius=1.0, verbose=0
            ),
            **mkw,
        )
    dt = time.perf_counter() - t0
    v = make_fcl_validator(
        st, n_arcs, fixtures=tuple(_G["fx"]), fixture_bvhs=_G["fbvh"]
    )
    fcl = float(np.asarray(v.slacks(res.x)).min())
    Rs, ts, tp, mk = _poses(st, np.asarray(res.x, float), n_arcs)
    cov = float(coverage_total_over_probes(Rs, ts, tp, mk, cov_data, 41))
    return dict(
        idx=idx,
        rank=rank,
        n_arcs=n_arcs,
        fcl=fcl,
        coverage=cov,
        pose=res.x,
        nit=int(res.nit),
        secs=dt,
        hole=dict(rec["probe_to_hole"]),
        partition=rec["partition"],
        probe_to_arc_idx=rec["probe_to_arc_idx"],
        arc_centroids_deg=list(rec["arc_centroids_deg"]),
        min_clear=rec.get("min_clear"),
    )


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
        st = _st_for_rec(r)
        if _G["cov_data"] is None:
            _G["cov_data"] = _G["build_cov"](_G["probes"], st)
        p2 = make_phase2(
            st,
            na,
            coverage_data=_G["cov_data"],
            fixtures=tuple(_G["fx"]),
            weights=Phase2Weights(
                min_clearance_mm=MINCLEAR,
                lambda_margin_clear=LAM_CLEAR,
                tau_clear_mm=TAU_CLEAR,
                lambda_cov=COV_WEIGHT,
                cov_alpha=COV_ALPHA if COV_NORM else 0.0,
            ),
            brain_sdf=_G.get("brain_sdf"),
            **_cov_norm_kwargs(st),
        )
        x = np.asarray(r["pose"], float)
        t0 = time.time()
        p2["fun"](x)
        p2["jac"](x)
        nlc = p2["constraints_nlc"][0]
        nlc.fun(x)
        nlc.jac(x)
        print(f"  warmed n_arcs={na} in {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    rer = pickle.load(open(POSES_PKL, "rb"))
    all_recs = rer["records"]
    # NO FCL cull between Phase 1 and 2 — rank the Phase-1 pool by soft min_clear
    # (SELECT_BY), best first, and hand the top-TOPK to Phase 2. FCL runs once at
    # the end as the ground-truth gate.
    order = sorted(
        range(len(all_recs)), key=lambda i: -float(all_recs[i].get(SELECT_BY, -1e9))
    )
    # RANKS overrides top-TOPK: an explicit rank list into the sorted order, to
    # probe where good feasibles stop appearing rather than guessing a cutoff.
    ranks_env = _os.environ.get("RANKS", "")
    if ranks_env:
        sel_ranks = [int(x) for x in ranks_env.split(",") if x.strip()]
        sel_ranks = [r for r in sel_ranks if r < len(order)]
    else:
        sel_ranks = list(range(min(TOPK, len(order))))

    def _norm(src, rank):
        r = all_recs[src]
        return dict(
            idx=r.get("idx", src),
            n_arcs=r["n_arcs"],
            pose=r.get("pose", r.get("x")),
            probe_to_hole=r["probe_to_hole"],
            partition=r["partition"],
            probe_to_arc_idx=r["probe_to_arc_idx"],
            arc_centroids_deg=r["arc_centroids_deg"],
            min_clear=r.get("min_clear"),
            rank=rank,
        )

    recs = [_norm(order[rk], rk) for rk in sel_ranks]
    print(
        f"Parallel Phase 2 [{_PLATFORM}]: {len(recs)} cands, {WORKERS} workers"
        f" x {_THREADS} thr, maxiter={P2_ITER}, lam={LAM_CLEAR} tau={TAU_CLEAR}"
    )
    # Pay the one-time compile ONCE. Threads share the PARENT's context, so warm
    # there. Spawned workers each have their own context and can't see the
    # parent's in-memory cache — so for a PROCESS pool we warm inside ONE worker,
    # which writes the persistent disk cache; the rest LOAD from it. This keeps
    # the parent GPU-free (no idle ~2.4GiB context), leaving HBM for an extra
    # worker. Largest n_arcs first so the heaviest compile is the one warmed.
    nstr = sorted({r["n_arcs"] for r in recs})
    if POOL == "thread":
        tw = time.time()
        print(f"warming (in-parent, single-threaded) for n_arcs {nstr}...", flush=True)
        _warmup(recs)
        print(f"  warmup {time.time() - tw:.0f}s", flush=True)
        t0 = time.time()
        # One GPU context shared across threads (no per-worker memory).
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(WORKERS) as ex:
            results = list(ex.map(_phase2_one, recs))
        wall = time.time() - t0  # processing only (excludes warmup)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(WORKERS, initializer=_init) as pool:
            if WARMUP:
                tw = time.time()
                print(
                    f"warming (one worker → disk cache; parent stays GPU-free) "
                    f"for n_arcs {nstr}...",
                    flush=True,
                )
                pool.apply(_warmup, (recs,))
                print(f"  warmup {time.time() - tw:.0f}s", flush=True)
            t0 = time.time()
            results = pool.map(_phase2_one, recs)
            wall = time.time() - t0  # processing only (excludes warmup)
    compute = float(sum(r["secs"] for r in results))
    print(
        f"  {wall / 60:.2f} min wall; {compute:.0f}s total compute; "
        f"{compute / max(wall, 1e-9):.1f}x effective parallelism "
        f"({compute / len(results):.0f}s/cand); "
        f"throughput {len(results) / wall:.2f} cand/s"
    )
    print(
        f"  per-cand secs (1st incl compile if WARMUP off): "
        f"{[round(r['secs']) for r in results]}"
    )

    feas = [r for r in results if r["fcl"] >= -FCL_TOL]
    print(
        f"  FCL >= -{FCL_TOL}: {len(feas)}/{len(results)} "
        f"(strictly feasible: {sum(1 for r in results if r['fcl'] >= -1e-4)})"
    )

    # Stratification view: feasibility + post-Phase-2 coverage vs rerank rank,
    # so we can see where in the distribution good feasibles stop appearing.
    by_rank = sorted(results, key=lambda r: r["rank"])
    print("\n=== per-candidate (rank-ordered) ===")
    print(f"{'rank':>5} {'cand':>6} {'fcl':>8} {'≥-.2':>4} {'coverage':>9}")
    for r in by_rank:
        print(
            f"{r['rank']:>5} {r['idx']:>6} {r['fcl']:>+8.4f} "
            f"{'Y' if r['fcl'] >= -FCL_TOL else 'n':>4} {r['coverage']:>9.3f}"
        )
    ranked = _mmr_rank(feas, MMR_LAMBDA)

    print(f"\n=== handoff ranking (MMR lam={MMR_LAMBDA}, coverage + diversity) ===")
    print(f"{'#':>3} {'cand':>6} {'coverage':>9} {'fcl':>8} {'maxsim_prev':>11}")
    for i, r in enumerate(ranked):
        sim = max(_similarity(r["hole"], p["hole"]) for p in ranked[:i]) if i else 0.0
        print(
            f"{i + 1:>3} {r['idx']:>6} {r['coverage']:>9.3f} {r['fcl']:>+8.4f} "
            f"{sim:>11.2f}"
        )

    out = Path(OUT_PKL)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            dict(
                ranked=ranked,
                all=results,
                config=dict(
                    subject_config=CONFIG,
                    topk=TOPK,
                    select_by=SELECT_BY,
                    minclear=MINCLEAR,
                    lam_clear=LAM_CLEAR,
                    tau_clear=TAU_CLEAR,
                    p2_iter=P2_ITER,
                    fcl_tol=FCL_TOL,
                    mmr_lambda=MMR_LAMBDA,
                    well=WELL,
                    solver=SOLVER,
                ),
            ),
            f,
        )
    # Each ranked/all record carries pose + probe_to_hole + probe_to_arc_idx +
    # arc_centroids_deg + n_arcs — everything needed to rebuild the plan and emit
    # a trame config per feasible candidate.
    print(f"\nsaved → {out}  ({len(ranked)} feasible ranked, {len(results)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
