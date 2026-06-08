"""Profile a single scipy SLSQP polish wrapped in jax.profiler.

Captures where time actually goes during an end-to-end SLSQP iter:
- JIT kernel calls (forward + grad)
- scipy's line-search and KKT bookkeeping
- numpy↔jax boundary overhead

Saves perfetto trace; also reports wall summary.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time
from pathlib import Path

import jax
import numpy as np
import yaml
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.arc_assignment import ArcAssignment
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import (
    JointWeights,
    _build_probe_static,
)
from aind_low_point.optimization.joint_rerank_jax import (
    make_jax_reduced_objective,
)
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes

MANUAL_H = {"MD": 3, "BLA": 4, "PL": 1, "VM": 7, "RSP": 5, "CA1": 10, "CLA": 12}


def main():
    cfg = ConfigModel.from_yaml(Path("examples/836656-config-T12.yml"))
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(Path("scratch/0283-300-04.holes.yml"))
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    with open("examples/836656-config-T12.plan.yml") as f:
        mp = yaml.safe_load(f)
    arcs = mp["arcs"]
    arc_letters = sorted(arcs.keys(), key=lambda k: arcs[k])
    letter_to_idx = {k: i for i, k in enumerate(arc_letters)}
    arc_centroids = tuple(arcs[k] for k in arc_letters)
    ptoarc, ptohole = {}, {}
    for p in probes:
        spec = mp["probes"][p.name]
        ptoarc[p.name] = letter_to_idx[spec["arc"]]
        ptohole[p.name] = MANUAL_H[p.name]
    ha = HoleAssignment(probe_to_hole=ptohole, cost=0.0)
    aa = ArcAssignment(
        probe_to_arc_idx=ptoarc, arc_centroids_deg=arc_centroids, cost=0.0
    )
    statics = _build_probe_static(probes, holes, ha, aa, sdf_by_name=sdf_by_name)
    n_arcs = 3
    K = len(statics)

    weights = JointWeights()
    fun, jac = make_jax_reduced_objective(statics, n_arcs, weights)

    # Use a polished y from a near-feasible cand so SLSQP actually runs
    # more iters (manual y0 is too far out of bounds and bails after 5).
    import pickle

    polish_pkl = Path("/tmp/full_polish_post_sat.pkl")
    if polish_pkl.exists():
        with open(polish_pkl, "rb") as f:
            data = pickle.load(f)
        sample_idx = next(
            (
                i
                for i, r in enumerate(data["results"])
                if 0.05 <= r.metrics.max_violation <= 0.5
            ),
            0,
        )
        sample_cand = data["candidates"][sample_idx]
        sample_jc = data["results"][sample_idx]
        statics = _build_probe_static(
            probes,
            holes,
            sample_cand.ha,
            sample_cand.aa,
            sdf_by_name=sdf_by_name,
        )
        n_arcs = sample_jc.n_arcs
        K = len(statics)
        y0 = np.asarray(sample_jc.reduced_y, dtype=np.float64)
        # Perturb so SLSQP actually has work to do; ~0.5σ of typical range.
        y0 = y0 + np.random.default_rng(0).normal(0, 0.5, size=y0.shape)
        print(
            f"  using cand {sample_idx} polished y "
            f"(max_viol {sample_jc.metrics.max_violation:.3f}) + 0.5σ noise"
        )
        fun, jac = make_jax_reduced_objective(statics, n_arcs, weights)
    else:
        y0 = np.zeros(n_arcs + 3 * K, dtype=np.float64)
        for a in range(min(n_arcs, len(aa.arc_centroids_deg))):
            y0[a] = float(aa.arc_centroids_deg[a])
        for k in range(K):
            y0[n_arcs + 3 * k + 1] = 1.0  # sx

    bounds = [(-90.0, 90.0)] * n_arcs
    for _ in range(K):
        bounds.extend([(-55.0, 55.0), (-1.5, 1.5), (-1.5, 1.5)])

    # Warm up JIT.
    print("Warmup (JIT compile)...")
    t0 = time.perf_counter()
    _ = fun(y0)
    _ = jac(y0)
    print(f"  warmup: {time.perf_counter() - t0:.2f}s")

    # Counter-wrapped fun/jac to count calls.
    n_fun_calls = [0]
    n_jac_calls = [0]
    t_fun = [0.0]
    t_jac = [0.0]

    def fun_wrapped(y):
        n_fun_calls[0] += 1
        t0 = time.perf_counter()
        v = fun(y)
        t_fun[0] += time.perf_counter() - t0
        return v

    def jac_wrapped(y):
        n_jac_calls[0] += 1
        t0 = time.perf_counter()
        g = jac(y)
        t_jac[0] += time.perf_counter() - t0
        return g

    # Run actual SLSQP polish.
    print("\nRunning SLSQP polish (maxiter=50, ftol=1e-4)...")
    t0 = time.perf_counter()
    result = minimize(
        fun_wrapped,
        y0,
        jac=jac_wrapped,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-4, "disp": False},
    )
    t_total = time.perf_counter() - t0

    print("\nSLSQP result:")
    print(f"  success: {result.success}, nit: {result.nit}, fn: {result.fun:.4g}")
    print(f"  total wall: {t_total:.3f}s")
    print(
        f"  fun calls: {n_fun_calls[0]}, total: {t_fun[0]:.3f}s, "
        f"avg: {t_fun[0] / max(n_fun_calls[0], 1) * 1000:.2f} ms"
    )
    print(
        f"  jac calls: {n_jac_calls[0]}, total: {t_jac[0]:.3f}s, "
        f"avg: {t_jac[0] / max(n_jac_calls[0], 1) * 1000:.2f} ms"
    )
    print(
        f"  scipy overhead: {t_total - t_fun[0] - t_jac[0]:.3f}s "
        f"({(t_total - t_fun[0] - t_jac[0]) / t_total * 100:.1f}%)"
    )
    print(f"  kernel-time fraction: {(t_fun[0] + t_jac[0]) / t_total * 100:.1f}%")

    # Capture a jax trace of the same polish (re-runs from warm).
    trace_dir = Path("/tmp/jax_slsqp_trace")
    trace_dir.mkdir(exist_ok=True)
    print(f"\nCapturing jax.profiler trace to {trace_dir}...")
    with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
        minimize(
            fun_wrapped,
            y0,
            jac=jac_wrapped,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-4, "disp": False},
        )
    print(
        "  trace saved (open .json.gz under {trace_dir}/plugins/profile/ at ui.perfetto.dev)"
    )


if __name__ == "__main__":
    main()
