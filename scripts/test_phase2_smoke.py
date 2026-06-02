"""Smoke test for Stage 3 Phase 1 → Phase 2 → Phase 3 chain.

Pipeline:
  1. Load Stage 2 polish pkl; pick a cand.
  2. Phase 1 SLSQP (soft-penalty): warm-starts with offsets + depth +
     coverage active.
  3. Phase 2 SLSQP (hard-constrained on JAX SDF): refines toward
     JAX-feasibility.
  4. Phase 3 SLSQP (hard-constrained on FCL raw mesh): final polish
     against true geometry, FD Jacobian for FCL constraints.
  5. Report per-phase: SLSQP status, min slack, coverage delta,
     wall time, and final FCL feasibility check.

Run::

    uv run --python 3.13 python -m scripts.test_phase2_smoke \\
        examples/836656-config-T12.yml scratch/0283-300-04.holes.yml \\
        --polish-pkl /tmp/full_polish_patchAB.pkl --cand 1405
"""

from __future__ import annotations

import argparse
import os as _os
import pickle
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    make_phase1_objective,
    reduced_to_phase1,
    Phase1Weights,
    PHASE1_PER_PROBE_VARS,
    phase1_n_vars,
)
from aind_low_point.optimization.stage3_phase2_jax import (
    make_phase2,
    Phase2Weights,
)
from aind_low_point.optimization.stage3_phase3_fcl import (
    make_fcl_validator,
)
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    phase1_bounds,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--polish-pkl", type=Path,
                   default=Path("/tmp/full_polish_patchAB.pkl"))
    p.add_argument("--cand", type=int, default=1405)
    p.add_argument("--phase1-iter", type=int, default=80)
    p.add_argument("--phase2-iter", type=int, default=80)
    p.add_argument("--phase3-iter", type=int, default=40)
    args = p.parse_args()

    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes_list = load_holes(args.holes)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes_list = _transform_holes(holes_list, R, t)

    print(f"Building SDFs (α-wrap envelope + shank OBBs)...")
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    fixtures = build_fixture_sdf_data(runtime)
    print(f"  {len(fixtures)} fixture SDFs: {[f.name for f in fixtures]}")

    with open(args.polish_pkl, "rb") as f:
        data = pickle.load(f)
    cand = data["candidates"][args.cand]
    jc = data["results"][args.cand]
    statics = _build_probe_static(
        probes, holes_list, cand.ha, cand.aa, sdf_by_name=sdf_by_name,
    )
    n_arcs = jc.n_arcs
    P = len(statics)

    # ---- Phase 1 polish: get a feasible-ish warm start ----
    coverage_data = build_coverage_data(probes, statics)
    p1_fun, p1_jac = make_phase1_objective(
        statics, n_arcs, coverage_data=coverage_data, fixtures=fixtures,
        weights=Phase1Weights(),
    )
    bounds = phase1_bounds(n_arcs, P)
    x0 = reduced_to_phase1(jc.reduced_y, n_arcs, P)
    print(f"\n[Phase 1] fn0={float(p1_fun(x0)):.2f}")
    res1 = minimize(
        p1_fun, x0, jac=p1_jac, method="L-BFGS-B", bounds=bounds,
        options=dict(maxiter=args.phase1_iter, ftol=1e-5, gtol=1e-5),
    )
    x1 = np.asarray(res1.x, dtype=np.float64)
    print(f"[Phase 1] fn={res1.fun:.2f}, iter={res1.nit}, success={res1.success}")

    # ---- Phase 2: hard-constrained SLSQP from x1 ----
    p2 = make_phase2(
        statics, n_arcs, coverage_data=coverage_data, fixtures=fixtures,
        weights=Phase2Weights(),
    )
    n_vars = phase1_n_vars(n_arcs, P)
    print(f"\n[Phase 2] n_vars={n_vars}")
    obj0 = float(p2["fun"](x1))
    slacks0 = p2["constraints"][0]["fun"](x1)
    print(f"[Phase 2] x1 (Phase 1 endpoint): obj={obj0:.4f}, "
          f"slacks: min={slacks0.min():+.4f}, mean={slacks0.mean():+.4f}, "
          f"n={len(slacks0)}, n_violating={int(np.sum(slacks0 < 0))}")

    res2 = minimize(
        p2["fun"], x1, jac=p2["jac"], method="SLSQP",
        bounds=bounds, constraints=p2["constraints"],
        options=dict(maxiter=args.phase2_iter, ftol=1e-6),
    )
    x2 = np.asarray(res2.x, dtype=np.float64)
    obj_end = float(p2["fun"](x2))
    slacks_end = p2["constraints"][0]["fun"](x2)
    print(f"[Phase 2] fn={obj_end:.4f}, iter={res2.nit}, "
          f"success={res2.success}, status={res2.status}")
    print(f"          message: {res2.message}")
    print(f"[Phase 2] x2 slacks: min={slacks_end.min():+.4f}, "
          f"mean={slacks_end.mean():+.4f}, n_violating={int(np.sum(slacks_end < 0))}")

    # Coverage delta (Phase 2 objective uses -coverage; compare manually)
    delta = obj_end - obj0
    print(f"\n[Summary] obj: {obj0:.4f} → {obj_end:.4f} (Δ={delta:+.4f}; "
          f"more negative = more coverage)")

    # Compare x2 vs x1 magnitude
    dx_norm = float(np.linalg.norm(x2 - x1))
    print(f"          ||x2 − x1|| = {dx_norm:.4f}")

    # Are all slacks ≥ 0 ?
    if slacks_end.min() >= -1e-4:
        print("[Phase 2 verdict] feasible point (slacks ≥ ~0).")
    else:
        print(f"[Phase 2 verdict] left {int(np.sum(slacks_end < 0))} "
              f"constraints violated (min slack {slacks_end.min():+.4f}).")

    # ---- Phase 3: FCL hard-constrained polish from x2 ----
    print("\n[Phase 3] Building FCL BVHs for fixtures...")
    fixture_bvhs = {}
    for f in fixtures:
        mesh = runtime.asset_catalog.get_geometry(f.name).raw
        fixture_bvhs[f.name] = make_fcl_bvh(mesh)

    # Phase 3 retired 2026-05-24: FCL is a validator, not an optimiser.
    validator = make_fcl_validator(
        statics, n_arcs,
        fixtures=fixtures, fixture_bvhs=fixture_bvhs,
    )
    print(f"[FCL validator] n_pairs={len(validator.pair_names)}")
    fcl_s = validator.slacks(x2)
    print(f"          FCL slacks at x2: min={fcl_s.min():+.4f}, "
          f"n_violating={int(np.sum(fcl_s < 0))}/{len(fcl_s)}")
    if fcl_s.min() >= -1e-4:
        print("[FCL verdict] FCL-feasible at Phase 2 exit.")
    else:
        print(f"[FCL verdict] {int(np.sum(fcl_s < 0))} FCL violations "
              f"(min {fcl_s.min():+.4f}).")
        for name, slack in validator.violating_pairs(x2):
            print(f"            {name}: {slack:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
