"""Full-DOF L-BFGS-B Phase-1 basin behavior on cand 4195.

Concern: with offsets/depth unfrozen, does the full-DOF solve stay in a
sensible basin, or do the extra DOF let it wander / pin offsets to the
bounds to fake-satisfy the soft penalty?

For each seed (manual spins + a few heuristic spin basins), run full-DOF
Phase-1 (offsets + depth + fixtures + coverage, L-BFGS-B) and report:
  seed spins -> final spins (|drift|), final offset/depth magnitudes
  (and whether any pin to the ±3 / ±2 bounds), FCL min slack, coverage.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    Phase1Weights,
    make_phase1_objective,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    phase1_bounds,
)
from scripts.spin_heuristic_search import (
    beam_search_assignments,
    build_coupling_graph,
    per_probe_spin_candidates,
)
from scripts.test_h1_chain_cand4195 import build_y, extract_spins

MANUAL = {
    "MD": -34.0,
    "BLA": 0.0,
    "PL": 131.0,
    "VM": -180.0,
    "RSP": 4.0,
    "CA1": 87.0,
    "CLA": 171.0,
}


def main() -> int:
    cfg = ConfigModel.from_yaml("examples/836656-config-T12.yml")
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(Path("scratch/0283-300-04.holes.yml"))
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        R, t = compiled["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    bvh_cache = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fixtures = build_fixture_sdf_data(runtime)
    fixture_bvhs = {
        f.name: make_fcl_bvh(runtime.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }
    probe_kind_by_name = {p.name: p.kind for p in probes}

    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    cand = data["candidates"][4195]
    jc = data["results"][4195]
    statics = _build_probe_static(
        probes, holes, cand.ha, cand.aa, bvh_cache=bvh_cache, sdf_by_name=sdf_by_name
    )
    n_arcs = jc.n_arcs
    n_probes = len(statics)
    coverage_data = build_coverage_data(probes, statics)
    validator = make_fcl_validator(
        statics, n_arcs, fixtures=fixtures, fixture_bvhs=fixture_bvhs
    )
    x_aug = np.asarray(data["augmented_phase1_x"][4195], float)
    arc_aps = x_aug[:n_arcs]
    mls = np.array([x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)])

    # Heuristic basins.
    target_LPS = np.array([st.target_LPS for st in statics])
    coupling = build_coupling_graph(target_LPS)
    spin_aug = extract_spins(x_aug, n_arcs, n_probes)
    seed = {i: float(spin_aug[i]) for i in range(n_probes)}
    spin_cands = per_probe_spin_candidates(
        statics, coupling, target_LPS, arc_aps, mls, probe_kind_by_name, seed_spins=seed
    )
    beam = beam_search_assignments(
        statics,
        spin_cands,
        coupling,
        target_LPS,
        arc_aps,
        mls,
        probe_kind_by_name,
        beam_B=16,
    )

    seeds = [("manual", np.array([MANUAL[st.name] for st in statics]))]
    for k, asg in enumerate(beam[:4]):
        ov = dict(asg.spins)
        seeds.append((f"beam{k}", np.array([ov[i] for i in range(n_probes)])))

    bounds = phase1_bounds(n_arcs, n_probes)
    p1_fun, p1_jac = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=Phase1Weights(),
    )
    cov_w = Phase1Weights(
        lambda_thread=0.0,
        lambda_clearance=0.0,
        lambda_kinematic=0.0,
        lambda_bounds=0.0,
        lambda_clearance_fixture=0.0,
        lambda_margin_clear=0.0,
        lambda_margin_thread=0.0,
        lambda_margin_clear_fixture=0.0,
        lambda_unit_circle=0.0,
    )
    cov_fun, _ = make_phase1_objective(
        statics, n_arcs, coverage_data=coverage_data, fixtures=fixtures, weights=cov_w
    )
    zero = np.zeros(n_probes)

    def ang(a, b):
        return abs((a - b + 180) % 360 - 180)

    for name, spins in seeds:
        y0 = build_y(arc_aps, n_arcs, mls, spins, zero, zero, zero)
        r1 = minimize(
            p1_fun,
            y0,
            jac=p1_jac,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=80, ftol=1e-5, gtol=1e-5),
        )
        x1 = np.asarray(r1.x)
        fin = extract_spins(x1, n_arcs, n_probes)
        offs = np.array(
            [
                [
                    x1[n_arcs + PHASE1_PER_PROBE_VARS * i + 3],
                    x1[n_arcs + PHASE1_PER_PROBE_VARS * i + 4],
                    x1[n_arcs + PHASE1_PER_PROBE_VARS * i + 5],
                ]
                for i in range(n_probes)
            ]
        )
        s = np.asarray(validator.slacks(x1))
        cov = -float(cov_fun(x1))
        drift = np.array([ang(fin[i], spins[i]) for i in range(n_probes)])
        # offset pinning: R/A bound ±3, depth ±2
        pin_off = int((np.abs(offs[:, :2]) > 2.99).sum())
        pin_dep = int((np.abs(offs[:, 2]) > 1.99).sum())
        print(f"\n=== seed: {name} ===")
        print(
            f"  FCL min {s.min():+.3f}  ({'FEAS' if s.min() >= -1e-4 else 'infeas'})"
            f"  coverage {cov:.2f}  |  spin drift: max {drift.max():.0f} "
            f"mean {drift.mean():.0f} deg"
        )
        print(
            f"  offsets: max|R/A| {np.abs(offs[:, :2]).max():.2f}mm  "
            f"max|depth| {np.abs(offs[:, 2]).max():.2f}mm  "
            f"(pinned to bound: {pin_off} R/A, {pin_dep} depth)"
        )
        print(
            f"  {'probe':<5} {'seed':>6} {'final':>7} {'drift':>6} "
            f"{'offR':>6} {'offA':>6} {'dep':>6}"
        )
        for i, st in enumerate(statics):
            print(
                f"  {st.name:<5} {spins[i]:>+6.0f} {fin[i]:>+7.1f} "
                f"{drift[i]:>6.0f} {offs[i, 0]:>+6.2f} {offs[i, 1]:>+6.2f} "
                f"{offs[i, 2]:>+6.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
