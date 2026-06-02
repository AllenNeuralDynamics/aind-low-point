"""Dump HLO for the Stage 2 reduced objective and count world-transform matmuls.

XLA's CSE pass either deduplicates ``surface[i] @ R[i].T + t[i]`` across
the 21 pair iterations or it doesn't. The HLO directly tells us:
  - If ~K=7 dot ops for surface×R: CSE works, hoisting won't help.
  - If ~K(K-1)/2*2 = 42 dot ops for surface×R: CSE off, hoisting wins.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path
import numpy as np
import yaml

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.arc_assignment import ArcAssignment
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import (
    JointWeights, _build_probe_static,
)
from aind_low_point.optimization.joint_rerank_jax import (
    _JIT_CACHE, _signature, _pack_statics, MAX_SHANKS_PAD,
    MAX_SECTIONS_PAD, make_jax_reduced_objective,
)
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes


MANUAL_H = {"MD": 3, "BLA": 4, "PL": 1, "VM": 7, "RSP": 5, "CA1": 10, "CLA": 12}


def main():
    cfg = ConfigModel.from_yaml(Path("examples/836656-config-T12.yml"))
    runtime = build_runtime_from_config(cfg)
    probes = [_probe_static_info(runtime.plan_state, runtime, n)
              for n in runtime.plan_state.probes]
    holes = load_holes(Path("scratch/0283-300-04.holes.yml"))
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        ) for p in probes
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
    aa = ArcAssignment(probe_to_arc_idx=ptoarc, arc_centroids_deg=arc_centroids, cost=0.0)
    statics = _build_probe_static(probes, holes, ha, aa, sdf_by_name=sdf_by_name)
    n_arcs = 3
    K = len(statics)
    n_pairs = K * (K - 1) // 2
    print(f"K={K} probes, n_pairs={n_pairs}")

    weights = JointWeights()
    obj_fn, grad_fn = make_jax_reduced_objective(statics, n_arcs, weights)

    sig = _signature(statics, n_arcs, weights)
    jit_obj, _ = _JIT_CACHE[sig]

    n_surf = int(np.asarray(statics[0].sdf_data["surface"]).shape[0])
    sdf_grid_shape = tuple(int(x) for x in np.asarray(statics[0].sdf_data["grid"]).shape)
    packed = _pack_statics(
        statics, n_arcs, MAX_SHANKS_PAD, MAX_SECTIONS_PAD, True, sdf_grid_shape, n_surf,
    )

    import jax.numpy as jnp
    y0 = jnp.zeros(n_arcs + 3 * K, dtype=jnp.float32)
    y0 = y0.at[0:n_arcs].set(jnp.array(aa.arc_centroids_deg[:n_arcs]))
    for k in range(K):
        y0 = y0.at[n_arcs + 3 * k + 1].set(1.0)  # sx

    print("Lowering + getting HLO...")
    lowered = jit_obj.lower(y0, **packed)
    hlo_text = lowered.compile().as_text()
    print(f"HLO size: {len(hlo_text)} chars, {hlo_text.count(chr(10))} lines")

    # Count distinct dot ops (matmuls). Each "dot(" in HLO is a matmul.
    n_dot = hlo_text.count(" dot(")
    n_dot_general = hlo_text.count("dot_general")
    n_fusion = hlo_text.count("fusion")
    n_convolution = hlo_text.count("convolution")
    print(f"\nHLO op counts:")
    print(f"  dot ops:          {n_dot}")
    print(f"  dot_general ops:  {n_dot_general}")
    print(f"  fusion ops:       {n_fusion}")
    print(f"  convolution ops:  {n_convolution}")

    # Look for the body-body shape signature: surface_a @ R_a.T where
    # surface_a is (N_surf, 3) and R_a is (3, 3).
    # In HLO this is dot/dot_general with operand shapes (N_surf, 3) and (3, 3).
    # The N_surf is in packed; we know it.
    print(f"\nLooking for (N_surf={n_surf}, 3) × (3, 3) matmul shapes...")
    for line in hlo_text.split("\n"):
        if "dot" in line and f"{n_surf}" in line and "f32" in line:
            print(f"  {line.strip()[:200]}")


if __name__ == "__main__":
    main()
