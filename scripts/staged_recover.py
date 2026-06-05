"""Recover production-like feasibility from the staged (reduced→full) stack.

Flexible driver for the seed/objective ablations. One experiment per run,
env-controlled; prints feasible /45 and overlap with production's durable
feasible set, saves a pkl.

Motivation: staged ADAM (4/45) and staged L-BFGS (2/45) both leave Stage 1 at
the -1.000 FCL sentinel for ~every candidate, vs the durable production path's
20/45. The objectives are nearly identical (the Phase-1 reduced stage is if
anything STRICTER), so the suspect is the SEED. Discovery: the spin-restore
already seeds at the candidate's PRODUCTION ml (`cand.ml_seed`, via
`enum_seed_y0`), but the staged Stage-1 x0 used a NEW-MRV ml that differs by
8-10deg — so the restored spins were optimized for a different ml than Stage-1
started from. ML_SEED=prod fixes that inconsistency.

Ablation knobs (run these in sequence; each adds to the previous):
  ML_SEED=prod    use cand.ml_seed (production) instead of new-MRV greedy ml
  RESTORE_WELL=0  drop the well from the spin restore
  S1_WELL=0       drop the well from the reduced Stage-1 objective
  OFFSET_ITERS=N  insert an offset/depth-only full-objective stage (N iters)
                  between Stage 1 and Stage 2 (mirrors production's augment)
  OPT=adam        use batched ADAM instead of per-cand L-BFGS-B (final check)

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.staged_recover
Env:  ML_SEED=prod OPT=lbfgs STAGE1=200 STAGE2=200 OFFSET_ITERS=0
      S1_WELL=1 RESTORE_WELL=1 NTHREADS=2 ADAM_STEPS=1000 IDXS=<override>
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import minimize

from aind_low_point.optimization.coverage_jax import coverage_total_over_probes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.stage3_phase1_jax import (
    Phase1Weights,
    make_phase1_objective,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from scripts.arc_first_mrv import Enumerator, build_or_load_atlas
from scripts.ingest_analysis import _poses
from scripts.restore_well_adam_manual import build_y, setup
from scripts.run_phase1_sample import build_coverage_data, phase1_bounds
from scripts.staged_adam import (
    PPV,
    adam_pass,
    emit_ml_seed,
    reduced_bounds,
    restore_spins_group,
)

ML_SEED = _os.environ.get("ML_SEED", "prod")  # prod | mrv
OPT = _os.environ.get("OPT", "lbfgs")  # lbfgs | adam
STAGE1 = int(_os.environ.get("STAGE1", "200"))
STAGE2 = int(_os.environ.get("STAGE2", "200"))
OFFSET_ITERS = int(_os.environ.get("OFFSET_ITERS", "0"))
ADAM_STEPS = int(_os.environ.get("ADAM_STEPS", "1000"))
S1_WELL = _os.environ.get("S1_WELL", "1") == "1"
RESTORE_WELL = _os.environ.get("RESTORE_WELL", "1") == "1"
NTHREADS = int(_os.environ.get("NTHREADS", "2"))
FCL_TOL = -1e-4
_LBFGS_OPTS = {"ftol": 1e-4, "gtol": 1e-5}


def seed_ml(cand, names, enum):
    """Per-probe ml seed array in probe order."""
    if ML_SEED == "prod":
        return np.array([float(cand.ml_seed[n]) for n in names])
    m = emit_ml_seed(
        enum, cand.ha.probe_to_hole, cand.aa.probe_to_arc_idx, cand.aa.arc_centroids_deg
    )
    return np.array([m[n] for n in names])


def offset_bounds(x_row, n_arcs, K):
    """phase1 bounds with EVERYTHING except (off_R, off_A, depth) pinned to
    x_row — i.e. an offsets/depth-only polish stage (mirrors production's
    augment_polish_with_offsets)."""
    b = list(phase1_bounds(n_arcs, K))
    free = {n_arcs + PPV * k + o for k in range(K) for o in (3, 4, 5)}
    for i in range(len(b)):
        if i not in free:
            b[i] = (float(x_row[i]), float(x_row[i]))
    return b


def lbfgs_pass(
    statics_flat, x0_rows, n_arcs, *, coverage_data, fixtures, bounds_list, max_iter
):
    """One L-BFGS-B polish per row (NTHREADS pool). ``bounds_list`` is one
    bounds spec per row. Objectives built single-threaded first (warms the
    shared JIT cache) then minimize runs concurrently."""
    objs = [
        make_phase1_objective(
            st,
            n_arcs,
            coverage_data=coverage_data,
            fixtures=fixtures,
            weights=Phase1Weights(),
        )
        for st in statics_flat
    ]
    opts = {**_LBFGS_OPTS, "maxiter": max_iter}

    def run_one(args):
        (fun, jac), x0, bnds = args
        try:
            res = minimize(
                fun,
                np.asarray(x0, np.float64),
                method="L-BFGS-B",
                jac=jac,
                bounds=bnds,
                options=opts,
            )
            return np.asarray(res.x, np.float64)
        except Exception:
            return np.asarray(x0, np.float64)

    with ThreadPoolExecutor(max_workers=NTHREADS) as ex:
        out = list(ex.map(run_one, zip(objs, x0_rows, bounds_list)))
    return out


def run_stage(
    statics_flat,
    x0_rows,
    n_arcs,
    K,
    *,
    coverage_data,
    fixtures,
    bounds_list,
    iters,
    well_obj,
):
    """Dispatch a stage to L-BFGS (per-row bounds) or batched ADAM (shared
    bounds only — used for the no-offset final check)."""
    if OPT == "lbfgs":
        return lbfgs_pass(
            statics_flat,
            x0_rows,
            n_arcs,
            coverage_data=coverage_data,
            fixtures=fixtures,
            bounds_list=bounds_list,
            max_iter=iters,
        )
    # ADAM: requires uniform bounds across rows.
    b0 = bounds_list[0]
    if any(bb != b0 for bb in bounds_list):
        raise ValueError("ADAM path needs uniform bounds (no per-row pinning)")
    x = np.stack([np.asarray(r, np.float32) for r in x0_rows])
    out = adam_pass(
        statics_flat,
        x,
        n_arcs,
        coverage_data=coverage_data,
        well_obj=well_obj,
        bounds=b0,
        steps=iters,
    )
    return [out[i] for i in range(out.shape[0])]


def main() -> int:
    _cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    K = len(probes)
    names = [p.name for p in probes]
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    h2 = pickle.load(open("scratch/phase2_handoff.pkl", "rb"))
    rer = pickle.load(open("scratch/full_rerank_0283.pkl", "rb"))
    stored = {r["idx"]: r for r in rer["records"]}

    env_idxs = _os.environ.get("IDXS")
    dense = int(_os.environ.get("RERANK_DENSE", "0"))
    sparse = int(_os.environ.get("RERANK_SPARSE", "0"))
    if env_idxs:
        idxs = [int(x) for x in env_idxs.split(",")]
    elif dense or sparse:
        # Dense top-of-rerank (where ranking matters for handoff) + a sparse
        # even sweep across the tail (full-range spread for the rank corr).
        ranked = sorted(rer["records"], key=lambda r: r["rank"])
        idxs = [r["idx"] for r in ranked[:dense]]
        rest = ranked[dense:]
        if sparse > 0 and rest:
            step = max(1, len(rest) // sparse)
            idxs += [r["idx"] for r in rest[::step][:sparse]]
    else:
        idxs = [r["idx"] for r in h2["all"] if r["fcl"] >= -0.2]

    atlas, atlas_names = build_or_load_atlas()
    enum = Enumerator(atlas, atlas_names, ml_margin_deg=0.0, ml_mode="greedy")

    by_arcs: dict[int, list[int]] = {}
    for idx in idxs:
        by_arcs.setdefault(int(pool["results"][idx].n_arcs), []).append(idx)

    tag = (
        f"{OPT}_ml-{ML_SEED}_s1well-{int(S1_WELL)}_rwell-{int(RESTORE_WELL)}"
        f"_off-{OFFSET_ITERS}_n{len(idxs)}"
    )
    print(f"=== staged recover [{tag}] ===")
    print(
        f"opt={OPT} stage1={STAGE1} stage2={STAGE2} offset_iters={OFFSET_ITERS} "
        f"(adam_steps={ADAM_STEPS}); ml_seed={ML_SEED}; "
        f"S1_WELL={S1_WELL} RESTORE_WELL={RESTORE_WELL}"
    )
    groups_str = ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_arcs.items()))
    print(f"grouped by n_arcs {{ {groups_str} }}\n", flush=True)

    s1_iters = ADAM_STEPS if OPT == "adam" else STAGE1
    s2_iters = ADAM_STEPS if OPT == "adam" else STAGE2

    rows = []
    for n_arcs in sorted(by_arcs):
        g = by_arcs[n_arcs]
        print(f"[n_arcs={n_arcs}] {len(g)} cands ...", flush=True)
        spins = restore_spins_group(
            n_arcs,
            g,
            probes=probes,
            holes=holes,
            pool=pool,
            sdf_by_name=sdf_by_name,
            well=well,
            with_well=RESTORE_WELL,
        )
        statics_flat, x0_rows = [], []
        for idx, sp in zip(g, spins):
            cand = pool["candidates"][idx]
            st = _build_probe_static(
                probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
            )
            mls = seed_ml(cand, names, enum)
            arc_aps = np.zeros(n_arcs)
            for a in range(min(n_arcs, len(cand.aa.arc_centroids_deg))):
                arc_aps[a] = float(cand.aa.arc_centroids_deg[a])
            zero = np.zeros(K)
            statics_flat.append(st)
            x0_rows.append(build_y(arc_aps, n_arcs, mls, sp, zero, zero, zero))
        cov_data = build_coverage_data(probes, statics_flat[0])

        # Stage 1: reduced (coverage off, offsets pinned).
        red_b = reduced_bounds(n_arcs, K)
        s1_fix = (well,) if S1_WELL else ()
        x1 = run_stage(
            statics_flat,
            x0_rows,
            n_arcs,
            K,
            coverage_data=None,
            fixtures=s1_fix,
            bounds_list=[red_b] * len(g),
            iters=s1_iters,
            well_obj=well if S1_WELL else None,
        )

        # Optional offset/depth-only full-objective stage (L-BFGS only).
        if OFFSET_ITERS > 0:
            if OPT != "lbfgs":
                raise ValueError("offset stage is L-BFGS only for now")
            off_b = [offset_bounds(x1[i], n_arcs, K) for i in range(len(g))]
            x1 = lbfgs_pass(
                statics_flat,
                x1,
                n_arcs,
                coverage_data=cov_data,
                fixtures=(well,),
                bounds_list=off_b,
                max_iter=OFFSET_ITERS,
            )

        # Stage 2: full, coverage on, all DOF free.
        full_b = phase1_bounds(n_arcs, K)
        x2 = run_stage(
            statics_flat,
            x1,
            n_arcs,
            K,
            coverage_data=cov_data,
            fixtures=(well,),
            bounds_list=[full_b] * len(g),
            iters=s2_iters,
            well_obj=well,
        )

        for ci, idx in enumerate(g):
            st = statics_flat[ci]
            v = make_fcl_validator(
                st, n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fixture_bvhs
            )

            def fcl(x, v=v):
                return float(np.asarray(v.slacks(np.asarray(x, np.float64))).min())

            def cov(x, st=st, n_arcs=n_arcs):
                Rs, ts, tips, mask = _poses(st, x, n_arcs)
                return float(
                    coverage_total_over_probes(Rs, ts, tips, mask, cov_data, 41)
                )

            # Phase-1 full objective (coverage on + well) = production's rerank
            # `viol` ranking key; evaluate at our staged pose and at the rerank
            # pose (the latter for a parity check against stored `viol`).
            ofun, _ = make_phase1_objective(
                st,
                n_arcs,
                coverage_data=cov_data,
                fixtures=(well,),
                weights=Phase1Weights(),
            )

            f1, c1 = fcl(x1[ci]), cov(x1[ci])
            f2, c2 = fcl(x2[ci]), cov(x2[ci])
            sp_pose = np.asarray(stored[idx]["pose"], np.float64)
            rows.append(
                dict(
                    idx=idx,
                    n_arcs=n_arcs,
                    s1_fcl=f1,
                    s1_cov=c1,
                    s2_fcl=f2,
                    s2_cov=c2,
                    feas=bool(f2 >= FCL_TOL),
                    dur_fcl=fcl(sp_pose),
                    dur_cov=cov(sp_pose),
                    x2=np.asarray(x2[ci], np.float64),
                    dur_pose=sp_pose,
                    staged_obj=float(ofun(x2[ci])),
                    prod_obj=float(ofun(sp_pose)),
                    prod_viol=float(stored[idx]["viol"]),
                    prod_rank=int(stored[idx]["rank"]),
                )
            )

    rows.sort(key=lambda r: idxs.index(r["idx"]))
    print(
        f"\n{'idx':>5} | {'S1: fcl   cov':>14} | {'S2: fcl   cov  feas':>20} | "
        f"{'durable: fcl  cov':>18}"
    )
    for r in rows:
        flag = " <-DUR" if (r["feas"] and r["dur_fcl"] >= FCL_TOL) else ""
        print(
            f"{r['idx']:>5} | {r['s1_fcl']:>+7.3f} {r['s1_cov']:>6.2f} | "
            f"{r['s2_fcl']:>+7.3f} {r['s2_cov']:>6.2f} "
            f"{'FEAS' if r['feas'] else 'infes':>5} | "
            f"{r['dur_fcl']:>+7.3f} {r['dur_cov']:>6.2f}{flag}"
        )

    n = len(rows)
    feas = {r["idx"] for r in rows if r["feas"]}
    dur_feas = {r["idx"] for r in rows if r["dur_fcl"] >= FCL_TOL}
    overlap = feas & dur_feas
    win = sum(r["feas"] and r["s2_cov"] > r["dur_cov"] + 0.5 for r in rows)
    print(f"\n=== [{tag}] feasible: {len(feas)}/{n} ===")
    print(f"    durable feasible:           {len(dur_feas)}/{n}")
    print(
        f"    overlap (both feasible):    {len(overlap)}/{len(dur_feas)} "
        f"of production's feasibles recovered"
    )
    print(f"    staged feas & higher cov:   {win}/{n}")
    with open(f"scratch/staged_recover_{tag}.pkl", "wb") as f:
        pickle.dump(
            {
                "rows": rows,
                "tag": tag,
                "ml_seed": ML_SEED,
                "opt": OPT,
                "stage1": STAGE1,
                "stage2": STAGE2,
                "offset_iters": OFFSET_ITERS,
                "s1_well": S1_WELL,
                "restore_well": RESTORE_WELL,
            },
            f,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
