"""(A) Does the CONVERGED restore seed beat the 2-round (production) seed?

For the manual candidate (4195): restore to 2 rounds and to convergence, then
ADAM (brain on) from each, and compare the final plans on FCL clearance,
coverage, and brain-tip containment. The 2-round restore is an under-converged
transient (see restore_convergence); this asks whether running the reduced
objective to convergence helps or hurts the final ADAM plan.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.restore_rounds_adam
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle

import numpy as np

from aind_low_point.optimization.coverage_jax import coverage_total_over_probes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.ingest_analysis import _poses
from scripts.restore_well_adam_manual import (
    build_adam_kernel,
    make_basin_sets,
    run_restore,
    setup,
    spins_deg_from_phase1,
    spins_deg_from_reduced,
)
from scripts.run_phase1_sample import build_coverage_data, maybe_build_brain_sdf
from scripts.test_h1_chain_cand4195 import build_y

IDX = 4195


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def converged_restore(cand, probes, holes, sdf_by_name, n_arcs, well, K,
                      max_rounds=8):
    """Iterate single rounds (feeding spins back) until no probe moves ≥0.5°."""
    seed = None
    prev = None
    nr = 0
    for nr in range(1, max_rounds + 1):
        y = run_restore(cand, probes, holes, sdf_by_name, n_arcs, well,
                        with_well=True, n_rounds=1, seed_spins_deg=seed)
        sp = spins_deg_from_reduced(y, n_arcs, K)
        if prev is not None and float(np.abs(_wrap(sp - prev)).max()) < 0.5:
            return y, nr
        prev = sp
        seed = sp
    return y, nr


def main() -> int:
    cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    K = len(probes)
    names = [p.name for p in probes]
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    cand = pool["candidates"][IDX]
    n_arcs = int(pool["results"][IDX].n_arcs)
    st = _build_probe_static(probes, holes, cand.ha, cand.aa,
                             bvh_cache=bvh, sdf_by_name=sdf_by_name)
    comp = compile_all_transforms(cfg.transforms)
    brain_sdf = maybe_build_brain_sdf(rt, comp)
    cov_data = build_coverage_data(probes, st)
    adam_eval = build_adam_kernel(st, n_arcs, K, well, cov_data, brain_sdf=brain_sdf)
    v = make_fcl_validator(st, n_arcs, fixtures=tuple(fixtures),
                           fixture_bvhs=fixture_bvhs)

    def coverage(x):
        Rs, ts, tips, mask = _poses(st, x, n_arcs)
        return float(coverage_total_over_probes(Rs, ts, tips, mask, cov_data,
                                                n_samples=41))

    y2 = run_restore(cand, probes, holes, sdf_by_name, n_arcs, well,
                     with_well=True, n_rounds=2)
    yc, nconv = converged_restore(cand, probes, holes, sdf_by_name, n_arcs,
                                  well, K)
    print(f"cand {IDX}  probes={names}  (converged at round {nconv})\n")

    for label, y_red in (("2-round", y2), (f"converged({nconv})", yc)):
        rest_sp = spins_deg_from_reduced(y_red, n_arcs, K)
        arc_aps, mls, sets = make_basin_sets(y_red, st, n_arcs, K)
        zero = np.zeros(K)
        x0 = [build_y(arc_aps, n_arcs, mls, sp, zero, zero, zero)
              for sp in sets["A_restore1"]]
        viol, xa = adam_eval(x0)
        br = int(np.argmin(viol))
        fcl = float(np.asarray(v.slacks(xa[br])).min())
        asp = np.round(spins_deg_from_phase1(xa[br], n_arcs, K), 0).astype(int)
        print(f"[{label:>12}] restore {np.round(rest_sp, 0).astype(int).tolist()}")
        print(f"{'':>14} ADAM    {asp.tolist()}")
        print(f"{'':>14} viol {viol[br]:+.3f}  fcl {fcl:+.3f}  "
              f"coverage {coverage(xa[br]):.3f}  "
              f"{'FEAS' if fcl >= -1e-4 else 'infeas'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
