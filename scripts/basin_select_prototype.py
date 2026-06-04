"""Cheap fixture-aware spin-basin SELECTION prototype.

For a sample of candidates (top Stage-2 ranks + the manual #4195), replace
the frozen grid-sweep spin basin with: propose N_BASINS joint spin
assignments (cheap beam), Phase-1-polish each with the FULL offset/
fixture/coverage objective, keep the one with the lowest violation_fn,
then Phase-2 + FCL on that winner.

Compares, per candidate:
  * prod_viol : production frozen-basin violation_fn (from the pkl)
  * sel_viol  : basin-selected violation_fn (this prototype)
  * fcl_min   : FCL ground-truth min slack of the basin-selected winner
  * feasible  : fcl_min >= -1e-4

Headline: how many of the sample flip to FCL-feasible, and where the
manual lands by sel_viol vs prod_viol.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS, Phase1Weights, make_phase1_objective,
)
from aind_low_point.optimization.stage3_phase2_jax import Phase2Weights, make_phase2
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data, build_fixture_sdf_data, phase1_bounds,
)
from scripts.spin_heuristic_search import (
    beam_search_assignments, build_coupling_graph, per_probe_spin_candidates,
)
from scripts.test_h1_chain_cand4195 import build_y, extract_spins

N_SAMPLE = 40
N_BASINS = 4
MANUAL_IDX = 4195


def main() -> int:
    cfg = ConfigModel.from_yaml("examples/836656-config-T12.yml")
    runtime = build_runtime_from_config(cfg)
    probes = [_probe_static_info(runtime.plan_state, runtime, n)
              for n in runtime.plan_state.probes]
    holes = load_holes(Path("scratch/0283-300-04.holes.yml"))
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        R, t = compiled["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf_by_name = {p.name: build_probe_sdf_from_alpha_wrap(
        runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw) for p in probes}
    bvh_cache = {p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh
                 else None for p in probes}
    fixtures = build_fixture_sdf_data(runtime)
    fixture_bvhs = {f.name: make_fcl_bvh(
        runtime.asset_catalog.get_geometry(f.name).raw) for f in fixtures}
    probe_kind_by_name = {p.name: p.kind for p in probes}

    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    results = data["results"]
    prod_viol = np.asarray(data["violation_fn"], float)
    s2_mv = np.array([float(r.metrics.max_violation) for r in results])
    order = list(np.argsort(s2_mv))
    sample = order[:N_SAMPLE]
    if MANUAL_IDX not in sample:
        sample = list(sample) + [MANUAL_IDX]
    print(f"sample: top-{N_SAMPLE} by Stage-2 max_viol + manual "
          f"(#{MANUAL_IDX}); {len(sample)} cands, {N_BASINS} basins each\n")

    viol_w = Phase1Weights(lambda_margin_clear=0.0, lambda_margin_thread=0.0,
                           lambda_margin_clear_fixture=0.0)

    rows = []
    for n_done, idx in enumerate(sample):
        t0 = time.time()
        cand = data["candidates"][idx]
        jc = results[idx]
        statics = _build_probe_static(probes, holes, cand.ha, cand.aa,
                                      bvh_cache=bvh_cache, sdf_by_name=sdf_by_name)
        n_arcs = jc.n_arcs
        n_probes = len(statics)
        coverage_data = build_coverage_data(probes, statics)
        validator = make_fcl_validator(statics, n_arcs, fixtures=fixtures,
                                       fixture_bvhs=fixture_bvhs)
        x_aug = np.asarray(data["augmented_phase1_x"][idx], float)
        arc_aps = x_aug[:n_arcs]
        mls = np.array([x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i]
                        for i in range(n_probes)])
        target_LPS = np.array([st.target_LPS for st in statics])
        coupling = build_coupling_graph(target_LPS)
        spin_aug = extract_spins(x_aug, n_arcs, n_probes)
        seed = {i: float(spin_aug[i]) for i in range(n_probes)}
        spin_cands = per_probe_spin_candidates(
            statics, coupling, target_LPS, arc_aps, mls, probe_kind_by_name,
            seed_spins=seed)
        beam = beam_search_assignments(
            statics, spin_cands, coupling, target_LPS, arc_aps, mls,
            probe_kind_by_name, beam_B=16)
        basins = beam[:N_BASINS]

        bounds = phase1_bounds(n_arcs, n_probes)
        p1_fun, p1_jac = make_phase1_objective(
            statics, n_arcs, coverage_data=coverage_data, fixtures=fixtures,
            weights=Phase1Weights())
        p1_viol, _ = make_phase1_objective(
            statics, n_arcs, coverage_data=None, fixtures=fixtures,
            weights=viol_w)
        zero = np.zeros(n_probes)
        best = None
        for asg in basins:
            ov = dict(asg.spins)
            spins = np.array([ov[i] for i in range(n_probes)])
            y0 = build_y(arc_aps, n_arcs, mls, spins, zero, zero, zero)
            r1 = minimize(p1_fun, y0, jac=p1_jac, method="L-BFGS-B",
                          bounds=bounds,
                          options=dict(maxiter=40, ftol=1e-5, gtol=1e-5))
            v = float(p1_viol(np.asarray(r1.x)))
            if best is None or v < best[0]:
                best = (v, np.asarray(r1.x))
        sel_viol, bx = best
        # Phase 2 + FCL on the winner only.
        p2 = make_phase2(statics, n_arcs, coverage_data=coverage_data,
                         fixtures=fixtures,
                         weights=Phase2Weights(min_clearance_mm=0.3))
        r2 = minimize(p2["fun"], bx, jac=p2["jac"], method="trust-constr",
                      bounds=bounds, constraints=p2["constraints_nlc"],
                      options=dict(maxiter=80, xtol=1e-6, gtol=1e-5,
                                   initial_tr_radius=1.0, verbose=0))
        s = np.asarray(validator.slacks(np.asarray(r2.x)))
        feas = bool(s.min() >= -1e-4)
        rows.append((idx, float(prod_viol[idx]), sel_viol, float(s.min()), feas))
        tag = " <-- MANUAL" if idx == MANUAL_IDX else ""
        print(f"[{n_done+1}/{len(sample)}] cand {idx:5d}  "
              f"prod_viol={prod_viol[idx]:10.2f}  sel_viol={sel_viol:9.2f}  "
              f"fcl={s.min():+.3f}  {'FEAS' if feas else 'fail'}  "
              f"({time.time()-t0:.0f}s){tag}", flush=True)

    rows.sort(key=lambda r: r[2])
    n_feas = sum(1 for r in rows if r[4])
    n_prod_feas = sum(1 for r in rows if r[1] < 10)
    print(f"\n=== summary ({len(rows)} cands) ===")
    print(f"FCL-feasible after basin-select: {n_feas}/{len(rows)}")
    print(f"(production frozen-basin violation_fn<10: {n_prod_feas}/{len(rows)})")
    man = next(r for r in rows if r[0] == MANUAL_IDX)
    man_rank = [r[0] for r in rows].index(MANUAL_IDX)
    print(f"\nManual #{MANUAL_IDX}: prod_viol={man[1]:.1f} -> sel_viol={man[2]:.2f}, "
          f"fcl={man[3]:+.3f} {'FEAS' if man[4] else 'fail'}; "
          f"sel_viol rank {man_rank+1}/{len(rows)} in sample")
    print(f"\nTop-10 by basin-selected violation_fn:")
    print(f"{'cand':>6} {'prod_viol':>11} {'sel_viol':>9} {'fcl':>8} feas")
    for r in rows[:10]:
        tag = " <-- MANUAL" if r[0] == MANUAL_IDX else ""
        print(f"{r[0]:>6} {r[1]:>11.1f} {r[2]:>9.2f} {r[3]:>+8.3f} "
              f"{'Y' if r[4] else 'n'}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
