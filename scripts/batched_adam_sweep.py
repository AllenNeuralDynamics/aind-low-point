"""ADAM hyperparameter sweep on the rank-corr sample.

Goal: cheapest (steps, lr) that keeps the FCL-feasible plans on top.
Metric is top-N recall of the known-feasible set {4195, 1035} + the
manual's rank, NOT global Spearman. One ADAM trajectory per lr (to 600
steps) snapshots the violation at intermediate step counts, so `steps`
is swept for free off a single compiled gradient.
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

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    Phase1Weights,
)
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.batched_phase1_build import make_batched_phase1_objective
from scripts.rank_corr_gate import zero_offsets
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import build_fixture_sdf_data, phase1_bounds

N_SAMPLE = 48
FEASIBLE = {4195, 1035}  # known FCL-feasible in this sample (from the gate)
LRS = [0.01, 0.02, 0.05, 0.1]
SNAPS = [100, 200, 300, 400, 600]


def adam_snap(x0, grad_fn, obj_fn, lo, hi, lr, snaps, b1=0.9, b2=0.999, eps=1e-8):
    x = jnp.asarray(x0, jnp.float32)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    lo = jnp.asarray(lo, jnp.float32)
    hi = jnp.asarray(hi, jnp.float32)
    out = {}
    for t in range(1, max(snaps) + 1):
        g = grad_fn(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + eps)
        x = jnp.clip(x, lo, hi)
        if t in snaps:
            out[t] = np.asarray(obj_fn(x))
    return out


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

    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    vf = np.asarray(data["violation_fn"], float)
    results = data["results"]
    n_arcs = 3
    n_probes = len(probes)
    eligible = [i for i in np.argsort(vf) if results[i].n_arcs == n_arcs]
    idxs = list(
        np.asarray(eligible)[np.linspace(0, len(eligible) - 1, N_SAMPLE).astype(int)]
    )
    for f in FEASIBLE:
        if f not in idxs:
            idxs[-1] = f

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

    bounds = phase1_bounds(n_arcs, n_probes)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    bobj, bgrad = make_batched_phase1_objective(
        statics_list, n_arcs, Phase1Weights(), (well,), coverage_data=None
    )

    feas_pos = [idxs.index(f) for f in FEASIBLE]
    man_pos = idxs.index(4195)

    print(
        f"{'lr':>6} {'steps':>6} {'man_rank':>9} {'feas_in_top3':>13} "
        f"{'n_soft_neg':>11} {'time':>6}"
    )
    print("-" * 60)
    for lr in LRS:
        t0 = time.time()
        snaps = adam_snap(y0, bgrad, bobj, lo, hi, lr, SNAPS)
        dt = time.time() - t0
        for s in SNAPS:
            viol = snaps[s]
            rank = np.argsort(viol)  # ascending
            order = {int(j): r for r, j in enumerate(rank)}
            man_rank = order[man_pos] + 1
            feas_top3 = sum(1 for p in feas_pos if order[p] < 3)
            n_neg = int((viol < 0).sum())
            print(
                f"{lr:>6.3f} {s:>6} {man_rank:>9} "
                f"{feas_top3:>10}/{len(feas_pos)} {n_neg:>11} "
                f"{dt if s == SNAPS[-1] else 0:>6.0f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
