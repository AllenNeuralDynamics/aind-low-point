"""Phase 2 (trust-constr) throughput + early predictor signal.

Warm-starts Phase 2 from each candidate's ADAM soft pose (the realistic init)
and measures, per candidate: minimize wall, iterations, per-iteration cost,
convergence, and post-Phase-2 FCL feasibility. Runs a varied sample (top-cov
feasibles → mid/low ranks) so we also see how intermediate metrics relate to
Phase-2 outcome. Throughput = iters x per-iter; per-iter is fixed, iters varies
with init/tuning — both reported.

Run:  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.phase2_throughput
Env:  P2_ITER (default 100), RANKS (comma list of rerank ranks to sample)
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.coverage_jax import coverage_total_over_probes
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase2_jax import Phase2Weights, make_phase2
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.ingest_analysis import _poses
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    phase1_bounds,
)

P2_ITER = int(_os.environ.get("P2_ITER", "100"))
# Phase 2 clearance constraint is d_soft >= min_clear, and d_soft over-reports
# real clearance by ~0.7-1.2 mm (envelope gap). So min_clear must be set above
# that for Phase-2 output to be FCL-feasible. Tune via P2_MINCLEAR.
P2_MINCLEAR = float(_os.environ.get("P2_MINCLEAR", "0.1"))
# Soft clearance-margin bonus knobs: lambda weights it vs coverage; tau is the
# saturation scale (reward = 1-exp(-clear/tau)). Default tau=0.2 saturates by
# ~0.5mm so it never pulls past the ~0.7mm envelope gap; raise both to pull
# Phase 2 toward feasible-with-headroom WITHOUT a hard wall.
P2_LAM_CLEAR = float(_os.environ.get("P2_LAM_CLEAR", "1.0"))
P2_TAU_CLEAR = float(_os.environ.get("P2_TAU_CLEAR", "0.2"))
RANKS = [int(x) for x in _os.environ.get(
    "RANKS", "0,1,4,7,12,30,100,500,2000").split(",")]


def main() -> int:
    cfg = ConfigModel.from_yaml("examples/836656-config-T12.yml")
    rt = build_runtime_from_config(cfg)
    probes = [_probe_static_info(rt.plan_state, rt, n)
              for n in rt.plan_state.probes]
    n_probes = len(probes)
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

    rer = pickle.load(open("scratch/full_rerank_0283.pkl", "rb"))
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    recs = rer["records"]
    cov_data = None

    print(f"Phase 2 trust-constr maxiter={P2_ITER} min_clear={P2_MINCLEAR} "
          f"lam_clear={P2_LAM_CLEAR} tau_clear={P2_TAU_CLEAR}")
    print(f"{'rank':>5} {'cand':>6} {'min':>6} {'it':>4} {'s/it':>6} "
          f"{'fcl':>8} {'≥-.2':>4} {'coverage':>14}")
    for ri in RANKS:
        if ri >= len(recs):
            continue
        r = recs[ri]
        c = pool["candidates"][r["idx"]]
        st = _build_probe_static(probes, holes, c.ha, c.aa, bvh_cache=bvh,
                                 sdf_by_name=sdf)
        if cov_data is None:
            cov_data = build_coverage_data(probes, st)
        bounds = phase1_bounds(r["n_arcs"], n_probes)
        x0 = np.asarray(r["pose"], np.float64)

        t0 = time.perf_counter()
        p2 = make_phase2(st, r["n_arcs"], coverage_data=cov_data,
                         fixtures=tuple(fx),
                         weights=Phase2Weights(min_clearance_mm=P2_MINCLEAR,
                                               lambda_margin_clear=P2_LAM_CLEAR,
                                               tau_clear_mm=P2_TAU_CLEAR))
        # one eval to trigger any lazy compile in this signature
        _ = p2["fun"](x0)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = minimize(p2["fun"], x0, jac=p2["jac"], method="trust-constr",
                       bounds=bounds, constraints=p2["constraints_nlc"],
                       options=dict(maxiter=P2_ITER, xtol=1e-6, gtol=1e-5,
                                    initial_tr_radius=1.0, verbose=0))
        t_min = time.perf_counter() - t0

        v = make_fcl_validator(st, r["n_arcs"], fixtures=tuple(fx),
                               fixture_bvhs=fbvh)
        fcl = float(np.asarray(v.slacks(res.x)).min())
        near = fcl >= -0.2  # "good plan, human-fixable" bar
        Rs, ts, tp, mk = _poses(st, np.asarray(res.x, float), r["n_arcs"])
        cov_out = float(coverage_total_over_probes(Rs, ts, tp, mk, cov_data, 41))
        Rs0, ts0, tp0, mk0 = _poses(st, x0, r["n_arcs"])
        cov_in = float(coverage_total_over_probes(Rs0, ts0, tp0, mk0, cov_data, 41))
        spi = t_min / max(res.nit, 1)
        print(f"{ri:>5} {r['idx']:>6} {t_min:>6.0f}s {res.nit:>4} "
              f"{spi:>6.3f} {fcl:>+8.4f} {'Y' if near else 'n':>4} "
              f"cov {cov_in:>6.2f}->{cov_out:>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
