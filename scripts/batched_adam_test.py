"""Batched ADAM on cand 4195's spin basins, head-to-head vs scipy L-BFGS-B.

Step 2 of the build. Builds the batched Phase-1 objective (vmap of the
per-cand _objective, well-only fixture) over N spin-basin seeds of cand
4195, runs a vectorized ADAM, and compares the final feasibility (FCL)
to per-basin scipy L-BFGS-B from the same seeds. Question: does ADAM
reach the same feasible basins L-BFGS-B does?
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.optimizer_vars import build_y, extract_spins
from aind_low_point.optimization.pipeline.phase1_build import (
    make_batched_phase1_objective,
)
from aind_low_point.optimization.pipeline.phase1_geometry import (
    build_fixture_sdf_data,
    phase1_bounds,
)
from aind_low_point.optimization.pipeline.probe_setup import (
    _probe_static_info,
    _transform_holes,
)
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    Phase1Weights,
    make_phase1_objective,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.spin_heuristic_search import (
    beam_search_assignments,
    build_coupling_graph,
    per_probe_spin_candidates,
)

MANUAL = {
    "MD": -34.0,
    "BLA": 0.0,
    "PL": 131.0,
    "VM": -180.0,
    "RSP": 4.0,
    "CA1": 87.0,
    "CLA": 171.0,
}
PHASE1_PER_PROBE_VARS = 6


def adam(x0, grad_fn, lo, hi, *, steps=600, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
    """Vectorized projected ADAM over a batch x0:(B,nv). grad_fn(x)->(B,nv)."""
    x = jnp.asarray(x0, jnp.float32)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    lo = jnp.asarray(lo, jnp.float32)
    hi = jnp.asarray(hi, jnp.float32)
    for t in range(1, steps + 1):
        g = grad_fn(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        x = x - lr * mh / (jnp.sqrt(vh) + eps)
        x = jnp.clip(x, lo, hi)
    return np.asarray(x)


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
    well = next(f for f in fixtures if "well" in f.name.lower())
    fixture_bvhs = {
        f.name: make_fcl_bvh(runtime.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }

    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    cand = data["candidates"][4195]
    jc = data["results"][4195]
    statics = _build_probe_static(
        probes, holes, cand.ha, cand.aa, bvh_cache=bvh_cache, sdf_by_name=sdf_by_name
    )
    n_arcs = jc.n_arcs
    n_probes = len(statics)
    validator = make_fcl_validator(
        statics,
        n_arcs,
        fixtures=(well,),
        fixture_bvhs={well.name: fixture_bvhs[well.name]},
    )
    x_aug = np.asarray(data["augmented_phase1_x"][4195], float)
    arc_aps = x_aug[:n_arcs]
    mls = np.array([x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)])

    # Propose basins: manual + beam top-4.
    target_LPS = np.array([st.target_LPS for st in statics])
    coupling = build_coupling_graph(target_LPS)
    spin_aug = extract_spins(x_aug, n_arcs, n_probes)
    spin_cands = per_probe_spin_candidates(
        statics,
        coupling,
        target_LPS,
        arc_aps,
        mls,
        {p.name: p.kind for p in probes},
        seed_spins={i: float(spin_aug[i]) for i in range(n_probes)},
    )
    beam = beam_search_assignments(
        statics,
        spin_cands,
        coupling,
        target_LPS,
        arc_aps,
        mls,
        {p.name: p.kind for p in probes},
        beam_B=16,
    )
    basins = [("manual", np.array([MANUAL[st.name] for st in statics]))]
    for k, asg in enumerate(beam[:4]):
        ov = dict(asg.spins)
        basins.append((f"beam{k}", np.array([ov[i] for i in range(n_probes)])))

    zero = np.zeros(n_probes)
    x0 = np.stack(
        [build_y(arc_aps, n_arcs, mls, sp, zero, zero, zero) for _, sp in basins]
    )  # (B, nv)
    B = x0.shape[0]

    bounds = phase1_bounds(n_arcs, n_probes)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)

    weights = Phase1Weights()
    bobj, bgrad = make_batched_phase1_objective(
        [statics] * B, n_arcs, weights, (well,), coverage_data=None
    )

    print(f"Batched ADAM on {B} basins of cand 4195 (well-only fixture)...")
    t0 = time.time()
    x_adam = adam(x0, bgrad, lo, hi, steps=600, lr=0.02)
    print(f"  ADAM 600 steps x {B} basins: {time.time() - t0:.1f}s")

    # scipy L-BFGS-B per basin (same objective) for comparison.
    fun, jac = make_phase1_objective(
        statics, n_arcs, coverage_data=None, fixtures=(well,), weights=weights
    )
    a_viols = np.asarray(bobj(x_adam))  # (B,) — call on the full B=5 batch
    print(
        f"\n{'basin':<8} {'ADAM viol':>10} {'ADAM fcl':>9} {'feas':>5}  "
        f"{'LBFGS viol':>11} {'LBFGS fcl':>10} {'feas':>5}"
    )
    print("-" * 66)
    for i, (name, _) in enumerate(basins):
        a_viol = float(a_viols[i])
        a_fcl = float(np.asarray(validator.slacks(x_adam[i])).min())
        r = minimize(
            fun,
            x0[i],
            jac=jac,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=200, ftol=1e-6, gtol=1e-6),
        )
        l_viol = float(r.fun)
        l_fcl = float(np.asarray(validator.slacks(np.asarray(r.x))).min())
        print(
            f"{name:<8} {a_viol:>10.3f} {a_fcl:>+9.3f} "
            f"{'Y' if a_fcl >= -1e-4 else 'n':>5}  "
            f"{l_viol:>11.3f} {l_fcl:>+10.3f} "
            f"{'Y' if l_fcl >= -1e-4 else 'n':>5}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
