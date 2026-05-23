"""Validation: batched reduced objective vs existing scalar JAX objective.

Builds a B=1 batched objective from the 836656/T12 manual candidate and
compares its scalar output (and gradient) against
``make_jax_reduced_objective`` for the same single candidate. Should
match within float32 tolerance.

Run::

    uv run --python 3.13 python -m scripts.test_batched_objective
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.arc_assignment import ArcAssignment
from aind_low_point.optimization.batched_objective import (
    make_batched_reduced_objective,
)
from aind_low_point.optimization.batched_static import (
    build_batched_probe_static,
    initial_y_from_aa,
)
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import (
    JointWeights,
    _build_probe_static,
)
from aind_low_point.optimization.joint_rerank_jax import (
    make_jax_reduced_objective,
)
from aind_low_point.optimization.sdf import build_probe_sdf
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes


MANUAL_H = {"MD": 3, "BLA": 4, "PL": 1, "VM": 7, "RSP": 5, "CA1": 10, "CLA": 12}


def parse_manual_plan(plan_path: Path, probe_names: list[str]):
    with open(plan_path) as f:
        mp = yaml.safe_load(f)
    arcs = mp["arcs"]
    arc_letters_sorted = sorted(arcs.keys(), key=lambda k: arcs[k])
    letter_to_idx = {k: i for i, k in enumerate(arc_letters_sorted)}
    arc_centroids = tuple(arcs[k] for k in arc_letters_sorted)
    probe_to_arc_idx = {}
    probe_to_hole = {}
    for name in probe_names:
        spec = mp["probes"][name]
        probe_to_arc_idx[name] = letter_to_idx[spec["arc"]]
        probe_to_hole[name] = MANUAL_H[name]
    return (
        HoleAssignment(probe_to_hole=probe_to_hole, cost=0.0),
        ArcAssignment(
            probe_to_arc_idx=probe_to_arc_idx,
            arc_centroids_deg=arc_centroids,
            cost=0.0,
        ),
    )


def main() -> int:
    cfg_path = Path("examples/836656-config-T12.yml")
    holes_path = Path("/tmp/836656-holes.yml")
    plan_path = Path("examples/836656-config-T12.plan.yml")

    print("Setup...")
    cfg = ConfigModel.from_yaml(cfg_path)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, name)
        for name in runtime.plan_state.probes
    ]
    holes_list = load_holes(holes_path)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes_list = _transform_holes(holes_list, R, t)

    probe_names = [p.name for p in probes]
    ha, aa = parse_manual_plan(plan_path, probe_names)

    # Build SDFs (this matches the optimizer's --sdf-clearance path)
    print("Building SDFs...")
    sdf_by_name = {}
    for p in probes:
        mesh = runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        sdf_by_name[p.name] = build_probe_sdf(mesh)

    # ---- Build batched static (B=1) ----
    batched = build_batched_probe_static(
        [(ha, aa)],
        probes,
        holes_list,
        sdf_by_name=sdf_by_name,
    )

    # ---- Build the two objectives ----
    weights = JointWeights()
    print("Building batched objective...")
    obj_batched, grad_batched = make_batched_reduced_objective(batched, weights)

    print("Building reference objective...")
    statics_ref = _build_probe_static(
        probes, holes_list, ha, aa, sdf_by_name=sdf_by_name
    )
    obj_ref, grad_ref = make_jax_reduced_objective(statics_ref, batched.n_arcs, weights)

    # ---- Initial y from manual ----
    y0_np = initial_y_from_aa([(ha, aa)], probes, n_arcs=batched.n_arcs)
    print(f"y0 shape: {y0_np.shape}")
    print(f"y0[0] arc APs: {y0_np[0, :batched.n_arcs]}")
    y_batch = jnp.asarray(y0_np)

    # ---- Forward eval ----
    print()
    print("=" * 60)
    print("Forward eval (B=1) — batched vs reference")
    print("=" * 60)
    t0 = time.perf_counter()
    val_batched = obj_batched(y_batch, batched)
    val_batched.block_until_ready()
    t1 = time.perf_counter()
    val_ref = obj_ref(y0_np[0])
    if hasattr(val_ref, "block_until_ready"):
        val_ref.block_until_ready()
    t2 = time.perf_counter()
    print(f"  batched val: {float(val_batched[0]):+.6f}  (compile+run {t1-t0:.2f}s)")
    print(f"  reference   : {float(val_ref):+.6f}  (compile+run {t2-t1:.2f}s)")
    rel = abs(float(val_batched[0]) - float(val_ref)) / max(abs(float(val_ref)), 1e-6)
    print(f"  rel error  : {rel:.2e}")

    # ---- Gradient eval ----
    print()
    print("=" * 60)
    print("Gradient eval (B=1)")
    print("=" * 60)
    g_batched = grad_batched(y_batch, batched)
    g_batched.block_until_ready()
    g_ref = grad_ref(y0_np[0])
    if hasattr(g_ref, "block_until_ready"):
        g_ref.block_until_ready()
    g_batched_np = np.asarray(g_batched[0])
    g_ref_np = np.asarray(g_ref)
    max_abs_err = float(np.max(np.abs(g_batched_np - g_ref_np)))
    rel_err = max_abs_err / max(float(np.max(np.abs(g_ref_np))), 1e-6)
    print(f"  max |Δgrad|: {max_abs_err:.4e}  rel: {rel_err:.4e}")
    if rel_err > 1e-3:
        print(f"  ⚠ Gradient mismatch — check per-probe pose path")
        print(f"  ref grad : {g_ref_np[:8]} ...")
        print(f"  batched  : {g_batched_np[:8]} ...")

    # ---- Verdict ----
    print()
    ok = rel < 1e-4 and rel_err < 1e-3
    print("PASS" if ok else "FAIL", "— batched objective matches reference"
          if ok else "— mismatch above tolerance")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
