"""Two-stage L-BFGS-B on the 45 stage-3-feasible candidates.

IDENTICAL stack to ``scripts/staged_adam`` — same seeds, same objectives, same
bounds/DOF — but the minimizer is scipy ``L-BFGS-B`` instead of projected ADAM.
This isolates the optimizer: the staged-ADAM run was only 4/45 FCL-feasible with
Stage-1 stuck at the -1.000 collision sentinel, suggesting the reduced
clearance-first *minimizer* (not just the staging) is load-bearing. This run
answers that directly.

Pipeline per candidate, seeded from the NEW MRV greedy-stab ml + restore-with-well
spins (atlas arc AP centroids):

  Stage 1  reduced DOF, clearance-first   : coverage OFF, offsets/depth PINNED to 0
                                            (arc/ml/spin free), STAGE1 max_iter.
  Stage 2  full DOF, coverage-aware       : coverage ON, all DOFs free, STAGE2 iter.

The objective is the SAME full Phase-1 ``_build_jit`` kernel used by the ADAM
path (via ``make_phase1_objective`` → scipy ``(fun, jac)``), so this is a pure
minimizer swap. L-BFGS-B options mirror production ``_slsqp_reduced``
(ftol=1e-4, gtol=1e-5). The spin-restore SEED is the batched restore, identical
to the ADAM run.

Per-candidate scipy calls are serial CPU control flow over a shared JIT'd kernel;
a small thread pool (NTHREADS, default 2) overlaps two candidates' L-BFGS loops
so one's GPU dispatch hides the other's host-side bookkeeping. The JIT cache is
warmed single-threaded (first build per group compiles) before the pool runs.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.staged_lbfgs
Env:  STAGE1=50  STAGE2=50  S1_WELL=1  NTHREADS=2  IDXS=<override the 45>
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
from scripts.staged_adam import emit_ml_seed, reduced_bounds, restore_spins_group

STAGE1 = int(_os.environ.get("STAGE1", "50"))
STAGE2 = int(_os.environ.get("STAGE2", "50"))
S1_WELL = _os.environ.get("S1_WELL", "1") == "1"
NTHREADS = int(_os.environ.get("NTHREADS", "2"))
FCL_TOL = -1e-4
_LBFGS_OPTS = {"ftol": 1e-4, "gtol": 1e-5}


def lbfgs_pass(
    statics_flat, x0_rows, n_arcs, *, coverage_data, fixtures, bounds, max_iter
):
    """Run one L-BFGS-B polish per row, 2-threaded. Returns stacked x.

    Objectives are built single-threaded first (warms the shared JIT cache on
    the first cand of the group), then the scipy ``minimize`` calls run in a
    ``NTHREADS`` pool — each ``fun``/``jac`` dispatches into the already-compiled
    kernel, so concurrent calls only overlap host control flow + GPU queueing.
    """
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
        (fun, jac), x0 = args
        try:
            res = minimize(
                fun,
                np.asarray(x0, np.float64),
                method="L-BFGS-B",
                jac=jac,
                bounds=bounds,
                options=opts,
            )
            return np.asarray(res.x, np.float64)
        except Exception:
            return np.asarray(x0, np.float64)

    with ThreadPoolExecutor(max_workers=NTHREADS) as ex:
        out = list(ex.map(run_one, zip(objs, x0_rows)))
    return out


def main() -> int:
    _cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    K = len(probes)
    names = [p.name for p in probes]
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    h2 = pickle.load(open("scratch/phase2_handoff.pkl", "rb"))
    rer = pickle.load(open("scratch/full_rerank_0283.pkl", "rb"))
    stored = {r["idx"]: r for r in rer["records"]}

    env_idxs = _os.environ.get("IDXS")
    if env_idxs:
        idxs = [int(x) for x in env_idxs.split(",")]
    else:
        idxs = [r["idx"] for r in h2["all"] if r["fcl"] >= -0.2]

    atlas, atlas_names = build_or_load_atlas()
    enum = Enumerator(atlas, atlas_names, ml_margin_deg=0.0, ml_mode="greedy")

    by_arcs: dict[int, list[int]] = {}
    for idx in idxs:
        by_arcs.setdefault(int(pool["results"][idx].n_arcs), []).append(idx)

    groups_str = ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_arcs.items()))
    print(
        f"staged L-BFGS-B (reduced {STAGE1} → full {STAGE2}); "
        f"grouped by n_arcs {{ {groups_str} }}; {NTHREADS} threads"
    )
    print(
        f"seed = MRV greedy ml + restore-with-well spins; well-in-reduced={S1_WELL}\n"
    )

    rows = []
    for n_arcs in sorted(by_arcs):
        g = by_arcs[n_arcs]
        print(
            f"[n_arcs={n_arcs}] {len(g)} cands: restore → reduced L-BFGS "
            f"→ full L-BFGS ...",
            flush=True,
        )
        spins = restore_spins_group(
            n_arcs,
            g,
            probes=probes,
            holes=holes,
            pool=pool,
            sdf_by_name=sdf_by_name,
            well=well,
            with_well=True,
        )
        statics_flat, x0_rows = [], []
        for idx, sp in zip(g, spins):
            cand = pool["candidates"][idx]
            st = _build_probe_static(
                probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
            )
            ml_map = emit_ml_seed(
                enum,
                cand.ha.probe_to_hole,
                cand.aa.probe_to_arc_idx,
                cand.aa.arc_centroids_deg,
            )
            mls = np.array([ml_map[n] for n in names])
            arc_aps = np.zeros(n_arcs)
            for a in range(min(n_arcs, len(cand.aa.arc_centroids_deg))):
                arc_aps[a] = float(cand.aa.arc_centroids_deg[a])
            zero = np.zeros(K)
            statics_flat.append(st)
            x0_rows.append(build_y(arc_aps, n_arcs, mls, sp, zero, zero, zero))
        cov_data = build_coverage_data(probes, statics_flat[0])

        s1_fix = (well,) if S1_WELL else ()
        x1 = lbfgs_pass(
            statics_flat,
            x0_rows,
            n_arcs,
            coverage_data=None,
            fixtures=s1_fix,
            bounds=reduced_bounds(n_arcs, K),
            max_iter=STAGE1,
        )
        x2 = lbfgs_pass(
            statics_flat,
            x1,
            n_arcs,
            coverage_data=cov_data,
            fixtures=(well,),
            bounds=phase1_bounds(n_arcs, K),
            max_iter=STAGE2,
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
                )
            )

    rows.sort(key=lambda r: idxs.index(r["idx"]))
    print(
        f"\n{'idx':>5} | {'S1: fcl   cov':>14} | {'S2: fcl   cov  feas':>20} | "
        f"{'durable: fcl  cov':>18}"
    )
    for r in rows:
        print(
            f"{r['idx']:>5} | {r['s1_fcl']:>+7.3f} {r['s1_cov']:>6.2f} | "
            f"{r['s2_fcl']:>+7.3f} {r['s2_cov']:>6.2f} "
            f"{'FEAS' if r['feas'] else 'infes':>5} | "
            f"{r['dur_fcl']:>+7.3f} {r['dur_cov']:>6.2f}"
        )

    n = len(rows)
    n_feas = sum(r["feas"] for r in rows)
    dur_feas = sum(r["dur_fcl"] >= FCL_TOL for r in rows)
    win = sum(r["feas"] and r["s2_cov"] > r["dur_cov"] + 0.5 for r in rows)
    print(f"\n=== staged L-BFGS feasible: {n_feas}/{n} (well-in-reduced={S1_WELL}) ===")
    print(f"    durable stored feasible:   {dur_feas}/{n}")
    print(f"    staged feas & higher cov:  {win}/{n}")
    tag = "well" if S1_WELL else "nowell"
    with open(f"scratch/staged_lbfgs_{tag}.pkl", "wb") as f:
        pickle.dump(
            {"rows": rows, "stage1": STAGE1, "stage2": STAGE2, "s1_well": S1_WELL}, f
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
