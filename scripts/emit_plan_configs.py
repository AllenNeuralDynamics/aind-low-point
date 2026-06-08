"""Emit trame-app config YAMLs from a Phase-2 handoff (the overnight pipeline
output, ``phase2_parallel``). For each of the top-N MMR-ranked *feasible* plans,
rebuild the plan_state from the saved pose + arc assignment and round-trip it
through ``save_plan_to_config`` — one ready-to-open config per candidate.

Config-driven (works on any subject): CONFIG/HOLES select the subject; HANDOFF is
that subject's Phase-2 output. Each handoff record already carries everything
needed (pose, probe_to_hole, probe_to_arc_idx, arc_centroids_deg, n_arcs), so no
re-optimization happens here — pure reconstruction.

Run:
  CONFIG=examples/837229-config.yml HOLES=scratch/0283-300-04.holes.yml \\
  HANDOFF=scratch/837229_phase2_handoff.pkl N=15 OUTDIR=examples/837229_plans \\
  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.emit_plan_configs
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.runtime import (
    build_runtime_from_config,
    planning_state_to_plan_model,
)
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import (
    _probe_static_info,
    _transform_holes,
    retro_opts_from_env,
)
from scripts.save_chain_plans import _apply_x_to_plan_state

CONFIG = os.environ.get("CONFIG", "examples/836656-config-T12.yml")
HOLES = os.environ.get("HOLES", "scratch/0283-300-04.holes.yml")
HANDOFF = os.environ.get("HANDOFF", "scratch/phase2_handoff.pkl")
N = int(os.environ.get("N", "15"))
OUTDIR = os.environ.get("OUTDIR", "scratch/plans")


def main() -> int:
    cfg = ConfigModel.from_yaml(CONFIG)
    rt = build_runtime_from_config(cfg)
    _ro = retro_opts_from_env(rt)
    probes = [
        _probe_static_info(rt.plan_state, rt, n, _ro) for n in rt.plan_state.probes
    ]
    holes = load_holes(Path(HOLES))
    comp = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in comp:
        R, t = comp["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf = {
        p.name: build_probe_sdf_from_alpha_wrap(
            rt.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    bvh = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }

    H = pickle.load(open(HANDOFF, "rb"))
    ranked = H.get("ranked", [])  # MMR-ranked feasible plans
    out = Path(OUTDIR)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(ranked)} feasible plans in handoff; emitting top {min(N, len(ranked))}"
    )

    for i, r in enumerate(ranked[:N]):
        n_arcs = r["n_arcs"]
        ha = SimpleNamespace(probe_to_hole=r["hole"])
        aa = SimpleNamespace(
            probe_to_arc_idx=r["probe_to_arc_idx"],
            arc_centroids_deg=list(r["arc_centroids_deg"]),
        )
        statics = _build_probe_static(
            probes, holes, ha, aa, bvh_cache=bvh, sdf_by_name=sdf
        )
        # Fresh runtime per plan so plan_state mutations don't bleed across plans.
        cfg_local = ConfigModel.from_yaml(CONFIG)
        rt_local = build_runtime_from_config(cfg_local)
        _apply_x_to_plan_state(
            rt_local.plan_state, np.asarray(r["pose"], float), statics, n_arcs
        )
        # Emit a PLAN-ONLY file (just the plan section, a PlanningModel) so it
        # pairs with the base config via ``--plan`` — not a full standalone config.
        plan_model = planning_state_to_plan_model(rt_local.plan_state, cfg_local.plan)
        fname = f"plan-{i:03d}-cov{r['coverage']:05.2f}-fcl{r['fcl']:+.3f}.plan.yml"
        path = out / fname
        with open(path, "w") as f:
            yaml.safe_dump(
                plan_model.model_dump(mode="json"),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        print(f"  [{i:>2}] cov={r['coverage']:.2f} fcl={r['fcl']:+.3f} → {path}")
    print(f"\nwrote {min(N, len(ranked))} configs → {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
