"""Rank-correlation gate: batched ADAM vs scipy L-BFGS-B over a pool chunk.

For a sample spanning the quality range, seed both optimizers from the
same start (augmented basin, offsets zeroed so both must descend),
optimize the SAME soft objective (well-only fixture, coverage off), and
compare the final soft-violation scores. Spearman ~1 ⇒ ADAM ranks the
pool like scipy ⇒ safe drop-in for the Stage-2 ranking.

A one-time FCL pass on the top of each ranking confirms the feasible
plans sit at the top.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.pipeline.phase1_build import (
    make_batched_phase1_objective,
)
from aind_low_point.optimization.pipeline.phase1_geometry import (
    phase1_bounds,
)
from aind_low_point.optimization.pipeline.runtime_adapter import (
    OptimizationRuntime,
)
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    Phase1Weights,
    make_phase1_objective,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from scripts.batched_adam_test import adam

N_SAMPLE = 48
ADAM_STEPS = 600
ADAM_LR = 0.02


def zero_offsets(x, n_arcs, n_probes):
    x = x.copy()
    for i in range(n_probes):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        x[off + 3] = x[off + 4] = x[off + 5] = 0.0
    return x


def main() -> int:
    opt = OptimizationRuntime.from_config_path(
        "examples/836656-config-T12.yml", "scratch/0283-300-04.holes.yml"
    )
    _cfg, _rt, probes, holes, sdf_by_name, bvh_cache, fixtures, well, fixture_bvhs = (
        opt.as_legacy_setup()
    )

    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    vf = np.asarray(data["violation_fn"], float)
    results = data["results"]
    # Batching requires uniform n_arcs (same x-dim + objective signature).
    # Filter to n_arcs==3 (common case, includes the manual). Production
    # would group-by-n_arcs or pad to max_arcs; immaterial for the gate.
    n_arcs = 3
    n_probes = len(probes)
    eligible = [i for i in np.argsort(vf) if results[i].n_arcs == n_arcs]
    idxs = list(
        np.asarray(eligible)[np.linspace(0, len(eligible) - 1, N_SAMPLE).astype(int)]
    )
    if 4195 not in idxs and results[4195].n_arcs == n_arcs:
        idxs[-1] = 4195

    statics_list, y0_list = [], []
    for idx in idxs:
        cand = data["candidates"][idx]
        st = _build_probe_static(
            probes,
            holes,
            cand.ha,
            cand.aa,
            bvh_cache=bvh_cache,
            sdf_by_name=sdf_by_name,
        )
        statics_list.append(st)
        x_aug = np.asarray(data["augmented_phase1_x"][idx], np.float64)
        y0_list.append(zero_offsets(x_aug, n_arcs, n_probes))
    y0 = np.stack(y0_list).astype(np.float32)

    weights = Phase1Weights()
    bounds = phase1_bounds(n_arcs, n_probes)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)

    print(f"Building batched objective over {len(idxs)} candidates...")
    bobj, bgrad = make_batched_phase1_objective(
        statics_list, n_arcs, weights, (well,), coverage_data=None
    )

    print(f"Batched ADAM ({ADAM_STEPS} steps)...")
    t0 = time.time()
    x_adam = adam(y0, bgrad, lo, hi, steps=ADAM_STEPS, lr=ADAM_LR)
    adam_viol = np.asarray(bobj(x_adam))
    t_adam = time.time() - t0
    print(
        f"  {t_adam:.1f}s total ({t_adam / len(idxs) * 1000:.0f} ms/cand incl compile)"
    )

    print("scipy L-BFGS-B per candidate...")
    scipy_viol = np.zeros(len(idxs))
    scipy_x = []
    t0 = time.time()
    for i, st in enumerate(statics_list):
        fun, jac = make_phase1_objective(
            st, n_arcs, coverage_data=None, fixtures=(well,), weights=weights
        )
        r = minimize(
            fun,
            y0[i],
            jac=jac,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=200, ftol=1e-6, gtol=1e-6),
        )
        scipy_viol[i] = r.fun
        scipy_x.append(np.asarray(r.x))
    print(f"  {time.time() - t0:.1f}s total")

    rho, _ = spearmanr(adam_viol, scipy_viol)
    print(f"\n=== Spearman(ADAM viol, scipy viol) = {rho:.4f} ===")
    # rank agreement on the top-10 (the part that feeds Stage 3)
    adam_rank = np.argsort(adam_viol)
    scipy_rank = np.argsort(scipy_viol)
    top_a = set(adam_rank[:10].tolist())
    top_s = set(scipy_rank[:10].tolist())
    print(f"top-10 overlap (ADAM vs scipy): {len(top_a & top_s)}/10")

    # FCL confirm: is the top of each ranking actually feasible?
    _validator = make_fcl_validator(
        statics_list[0], n_arcs, fixtures=(well,), fixture_bvhs=fixture_bvhs
    )
    print("\nTop-8 by ADAM violation (FCL on each):")
    print(f"{'cand':>6} {'adam_viol':>10} {'scipy_viol':>11} {'adam_fcl':>9}")
    for j in adam_rank[:8]:
        idx = idxs[j]
        v = make_fcl_validator(
            statics_list[j], n_arcs, fixtures=(well,), fixture_bvhs=fixture_bvhs
        )
        fcl = float(np.asarray(v.slacks(x_adam[j])).min())
        tag = " <-- MANUAL" if idx == 4195 else ""
        print(
            f"{idx:>6} {adam_viol[j]:>10.3f} {scipy_viol[j]:>11.3f} {fcl:>+9.3f}{tag}"
        )
    # worst rank disagreements
    ar = np.argsort(np.argsort(adam_viol))
    sr = np.argsort(np.argsort(scipy_viol))
    dis = np.argsort(-np.abs(ar - sr))[:5]
    print("\nWorst rank disagreements (|adam_rank - scipy_rank|):")
    for j in dis:
        print(
            f"  cand {idxs[j]:>5}: adam_rank {ar[j]:>3} vs scipy_rank {sr[j]:>3}"
            f"  (adam_viol {adam_viol[j]:.2f}, scipy_viol {scipy_viol[j]:.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
