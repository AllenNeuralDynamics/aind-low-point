"""Stratified Phase 1 → Phase 2 → Phase 3 validation.

Samples Stage 2 polish output across ``max_violation`` bins and runs
the full Phase 1 → Phase 2 → Phase 3 chain on each. Reports per-bin
success rate to identify where Phase 1/2/3 stops being able to recover
from Stage 2 violations.

Bins:
  strict       : max_viol ≤ 0.001
  mild         : 0.001 < max_viol ≤ 0.1
  moderate     : 0.1 < max_viol ≤ 1.0
  hard         : 1.0 < max_viol ≤ 10.0
  catastrophic : max_viol > 10.0

For each cand:
  Phase 1: soft-penalty SLSQP (warm-up with offsets+depth+coverage)
  Phase 2: hard-constraint JAX SDF SLSQP
  Phase 3: hard-constraint FCL raw-mesh SLSQP (ground truth)

Final verdict: FCL-feasible iff Phase 3 ends with all FCL slacks ≥ −1e-4.

Run::
    uv run --python 3.13 python -m scripts.stratified_p1_p2_p3 \\
        examples/836656-config-T12.yml /tmp/836656-holes.yml \\
        --polish-pkl /tmp/full_polish_post_sat.pkl --n-per-bin 3
"""

from __future__ import annotations

import argparse
import os as _os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    Phase1Weights, make_phase1_objective, phase1_n_vars, reduced_to_phase1,
)
from aind_low_point.optimization.stage3_phase2_jax import (
    Phase2Weights, make_phase2,
)
from aind_low_point.optimization.stage3_phase3_fcl import (
    Phase3Weights, make_phase3,
)
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data, build_fixture_sdf_data, phase1_bounds,
)


BINS = [
    ("strict",       0.0,     0.001),
    ("mild",         0.001,   0.1),
    ("moderate",     0.1,     1.0),
    ("hard",         1.0,     10.0),
    ("catastrophic", 10.0,    float("inf")),
]


@dataclass
class CandResult:
    cand_idx: int
    bin_name: str
    mv_stage2: float
    p1_fn_end: float
    p1_nit: int
    p1_wall: float
    p2_min_slack: float
    p2_n_violating: int
    p2_nit: int
    p2_wall: float
    p3_min_slack_analytic: float
    p3_min_slack_fcl: float
    p3_n_violating_fcl: int
    p3_nit: int
    p3_wall: float
    fcl_feasible: bool


def stratified_sample(results, rng, n_per_bin):
    out = {}
    for name, lo, hi in BINS:
        bucket = [
            i for i, r in enumerate(results)
            if lo < r.metrics.max_violation <= hi
            or (lo == 0.0 and r.metrics.max_violation <= hi)
        ]
        if not bucket:
            out[name] = []
            continue
        k = min(n_per_bin, len(bucket))
        out[name] = rng.choice(bucket, size=k, replace=False).tolist()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--polish-pkl", type=Path,
                   default=Path("/tmp/full_polish_post_sat.pkl"))
    p.add_argument("--n-per-bin", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p1-iter", type=int, default=80)
    p.add_argument("--p2-iter", type=int, default=80)
    p.add_argument("--p3-iter", type=int, default=40)
    args = p.parse_args()

    print("Loading config + building probes / SDFs / fixtures...", flush=True)
    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(args.holes)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes = _transform_holes(holes, R, t)

    t_setup = time.time()
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    fixtures = build_fixture_sdf_data(runtime)
    bvh_cache = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fixture_bvhs = {
        f.name: make_fcl_bvh(runtime.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }
    print(f"  setup: {time.time()-t_setup:.1f}s", flush=True)

    with open(args.polish_pkl, "rb") as f:
        data = pickle.load(f)

    rng = np.random.default_rng(args.seed)
    picks = stratified_sample(data["results"], rng, args.n_per_bin)
    total = sum(len(idxs) for idxs in picks.values())
    print(f"Sampled {total} cands across {len(BINS)} bins:")
    for name, idxs in picks.items():
        print(f"  {name:<12}: {len(idxs):>2}  {[int(i) for i in idxs]}")
    print()

    results: list[CandResult] = []

    for bin_name, idxs in picks.items():
        for cand_idx in idxs:
            cand_idx = int(cand_idx)
            cand = data["candidates"][cand_idx]
            jc = data["results"][cand_idx]
            mv = float(jc.metrics.max_violation)

            statics = _build_probe_static(
                probes, holes, cand.ha, cand.aa,
                bvh_cache=bvh_cache, sdf_by_name=sdf_by_name,
            )
            n_arcs = jc.n_arcs
            n_vars = phase1_n_vars(n_arcs, len(statics))
            coverage_data = build_coverage_data(probes, statics)
            bounds = phase1_bounds(n_arcs, len(statics))
            x0 = reduced_to_phase1(jc.reduced_y, n_arcs, len(statics))

            # ---- Phase 1 ----
            p1_fun, p1_jac = make_phase1_objective(
                statics, n_arcs, coverage_data=coverage_data,
                fixtures=fixtures, weights=Phase1Weights(),
            )
            t0 = time.time()
            r1 = minimize(p1_fun, x0, jac=p1_jac, method="SLSQP",
                          bounds=bounds,
                          options=dict(maxiter=args.p1_iter, ftol=1e-5))
            p1_wall = time.time() - t0
            x1 = np.asarray(r1.x, dtype=np.float64)

            # ---- Phase 2 ----
            p2 = make_phase2(
                statics, n_arcs, coverage_data=coverage_data,
                fixtures=fixtures, weights=Phase2Weights(),
            )
            t0 = time.time()
            r2 = minimize(p2["fun"], x1, jac=p2["jac"], method="SLSQP",
                          bounds=bounds, constraints=p2["constraints"],
                          options=dict(maxiter=args.p2_iter, ftol=1e-6))
            p2_wall = time.time() - t0
            x2 = np.asarray(r2.x, dtype=np.float64)
            s_p2 = p2["constraints"][0]["fun"](x2)
            p2_min = float(np.min(s_p2))
            p2_violating = int(np.sum(s_p2 < -1e-4))

            # ---- Phase 3 ----
            p3 = make_phase3(
                statics, n_arcs, coverage_data=coverage_data,
                fixtures=fixtures, fixture_bvhs=fixture_bvhs,
                weights=Phase3Weights(),
            )
            t0 = time.time()
            r3 = minimize(p3["fun"], x2, jac=p3["jac"], method="SLSQP",
                          bounds=bounds, constraints=p3["constraints"],
                          options=dict(maxiter=args.p3_iter, ftol=1e-6))
            p3_wall = time.time() - t0
            x3 = np.asarray(r3.x, dtype=np.float64)
            s_an = p3["constraints"][0]["fun"](x3)
            s_fcl = (
                p3["constraints"][1]["fun"](x3)
                if len(p3["constraints"]) > 1 else np.zeros(0)
            )
            p3_an = float(np.min(s_an)) if s_an.size else 0.0
            p3_fcl = float(np.min(s_fcl)) if s_fcl.size else 0.0
            p3_fcl_viol = int(np.sum(s_fcl < -1e-4)) if s_fcl.size else 0
            fcl_feas = (p3_fcl >= -1e-4) and (p3_an >= -1e-4)

            cr = CandResult(
                cand_idx=cand_idx, bin_name=bin_name, mv_stage2=mv,
                p1_fn_end=float(r1.fun), p1_nit=int(r1.nit), p1_wall=p1_wall,
                p2_min_slack=p2_min, p2_n_violating=p2_violating,
                p2_nit=int(r2.nit), p2_wall=p2_wall,
                p3_min_slack_analytic=p3_an,
                p3_min_slack_fcl=p3_fcl,
                p3_n_violating_fcl=p3_fcl_viol,
                p3_nit=int(r3.nit), p3_wall=p3_wall,
                fcl_feasible=fcl_feas,
            )
            results.append(cr)
            tag = "FEAS" if fcl_feas else "FAIL"
            print(
                f"  {bin_name:<12} cand#{cand_idx:<5} mv={mv:>9.4f} "
                f"P1[fn={r1.fun:+8.2f} nit={r1.nit:>3}] "
                f"P2[min_s={p2_min:+.4f} nv={p2_violating}] "
                f"P3[fcl={p3_fcl:+.4f} an={p3_an:+.4f}] "
                f"wall={p1_wall+p2_wall+p3_wall:.0f}s {tag}",
                flush=True,
            )

    # ---- Per-bin summary ----
    print("\n" + "=" * 78)
    print("Per-bin success rate (Phase 3 FCL-feasible / n_sampled)")
    print("=" * 78)
    print(f"{'bin':<12} {'n':>3} {'FEAS':>5} {'pct':>6}  "
          f"{'P1 wall':>8} {'P2 wall':>8} {'P3 wall':>8}")
    for name, _lo, _hi in BINS:
        bin_results = [r for r in results if r.bin_name == name]
        if not bin_results:
            continue
        n = len(bin_results)
        feas = sum(r.fcl_feasible for r in bin_results)
        wall_p1 = np.mean([r.p1_wall for r in bin_results])
        wall_p2 = np.mean([r.p2_wall for r in bin_results])
        wall_p3 = np.mean([r.p3_wall for r in bin_results])
        print(f"{name:<12} {n:>3} {feas:>5} {feas/n*100:>5.0f}%  "
              f"{wall_p1:>7.1f}s {wall_p2:>7.1f}s {wall_p3:>7.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
