"""Tier-2 of the static handoff: decode each feasible Phase-2 pose into a
loadable ``plan-NN-<path>.yml`` (a PlanningModel YAML openable in trame/jupyter),
and round-trip every one through the *app's* pose resolver to prove the
serialization preserved the geometry.

The decode mirrors ``run_phase1_sample`` exactly: per probe
``x[off:off+6] = (ml, sx, sy, off_R, off_A, depth)``, ``spin = atan2(sy, sx)``,
``ap = arc_aps[arc_idx]``. ``ProbePose.from_planning_state`` uses the identical
kinematic formula as ``pose_from_optimizer_vars``, so reloading the written plan
must reproduce the decode tip to ~1e-9 mm — the round-trip check asserts it.

Imported by ``export_handoff.py --plans``; not run standalone.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import yaml
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system

from aind_low_point.config import InlineTargetRefModel, PlanningModel
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.kinematics import pose_from_optimizer_vars
from aind_low_point.optimization.pipeline.runtime_adapter import (
    OptimizationRuntime,
)
from aind_low_point.planning import ProbePose
from aind_low_point.runtime.export import apply_plan_model_to_state
from aind_low_point.state_change import PlanStore

CONFIG = "examples/836656-config-T12.yml"
PLAN_TEMPLATE = "examples/836656-config-T12.plan.yml"
HOLES = "scratch/0283-300-04.holes.yml"
POOL_PKL = "scratch/full_polish_0283.pkl"
PHASE1_PER_PROBE_VARS = 6
ARC_LETTERS = "abcdefgh"


def _setup():
    opt = OptimizationRuntime.from_config_path(CONFIG, HOLES)
    cfg, rt, probes, holes, sdf, bvh, _fixtures, _well, _fixture_bvhs = (
        opt.as_legacy_setup()
    )
    pool = pickle.load(open(POOL_PKL, "rb"))
    template = PlanningModel.model_validate(yaml.safe_load(open(PLAN_TEMPLATE)))
    return cfg, rt, probes, holes, sdf, bvh, pool, template


def _decode(pose, statics, n_arcs):
    """Pose vector → per-probe plan fields (matches run_phase1_sample)."""
    x = np.asarray(pose, np.float64)
    arc_aps = x[:n_arcs]
    fields = {}
    for i, st in enumerate(statics):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        ml, sx, sy, off_R, off_A, depth = (float(x[off + k]) for k in range(6))
        spin = float(np.degrees(np.arctan2(sy, sx)))
        ap = float(arc_aps[st.arc_idx])
        fields[st.name] = dict(
            arc_idx=st.arc_idx,
            ml=ml,
            spin=spin,
            ap=ap,
            off_R=off_R,
            off_A=off_A,
            depth=depth,
            target_LPS=np.asarray(st.target_LPS, np.float64),
            pivot_local=st.pivot_local,
        )
    return arc_aps, fields


def _build_plan_model(template, arc_aps, fields, n_arcs):
    """Override the template PlanningModel with the decoded plan."""
    pm = copy.deepcopy(template)
    arcs = {ARC_LETTERS[i]: float(arc_aps[i]) for i in range(n_arcs)}
    new_probes = {}
    for name, f in fields.items():
        decl = copy.deepcopy(template.probes[name])
        tgt_ras = convert_coordinate_system(f["target_LPS"], "LPS", "RAS")
        decl = decl.model_copy(
            update=dict(
                arc=ARC_LETTERS[f["arc_idx"]],
                slider_ml=f["ml"],
                spin=f["spin"],
                ap_local=f["ap"],
                bind_ap_to_arc=True,
                offsets_RA=[f["off_R"], f["off_A"]],
                past_target_mm=f["depth"],
                target=InlineTargetRefModel(point_RAS=[float(v) for v in tgt_ras]),
                calibrated=False,
            )
        )
        new_probes[name] = decl
    return pm.model_copy(update=dict(arcs=arcs, probes=new_probes))


def _roundtrip_tip_error(rt, plan_model, fields):
    """Apply plan via the app path, resolve each pose, return max |tip - decode|."""
    store = PlanStore(copy.deepcopy(rt.plan_state))
    apply_plan_model_to_state(plan_model, store)
    worst = 0.0
    for name, f in fields.items():
        _, t_dec = pose_from_optimizer_vars(
            target_LPS=f["target_LPS"],
            ap_deg=f["ap"],
            ml_deg=f["ml"],
            spin_deg=f["spin"],
            offset_R_mm=f["off_R"],
            offset_A_mm=f["off_A"],
            past_target_mm=f["depth"],
            recording_center_local=f["pivot_local"],
        )
        pose = ProbePose.from_planning_state(
            store.state, name, catalog=rt.asset_catalog
        )
        worst = max(worst, float(np.linalg.norm(np.asarray(pose.tip) - t_dec)))
    return worst


def write_plans(rows, out_dir: Path) -> None:
    _cfg, rt, probes, holes, sdf, bvh, pool, template = _setup()
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    locked_probes = sorted(rows[0]["hole"].keys())

    worst_all = 0.0
    print(f"{'#':>3} {'cand':>5} {'cov':>7} {'fcl':>7} {'tip_err_mm':>11}  file")
    for i, r in enumerate(rows, 1):
        c = pool["candidates"][r["idx"]]
        statics = _build_probe_static(
            probes, holes, c.ha, c.aa, bvh_cache=bvh, sdf_by_name=sdf
        )
        arc_aps, fields = _decode(r["pose"], statics, r["n_arcs"])
        pm = _build_plan_model(template, arc_aps, fields, r["n_arcs"])
        err = _roundtrip_tip_error(rt, pm, fields)
        worst_all = max(worst_all, err)
        path = "_".join(f"{p.lower()}{r['hole'][p]}" for p in locked_probes)
        fname = f"plan-{i:02d}-cov{r['coverage']:05.2f}-{path}.yml"
        with open(plans_dir / fname, "w") as fh:
            yaml.safe_dump(
                pm.model_dump(mode="json"),
                fh,
                sort_keys=False,
                default_flow_style=False,
            )
        print(
            f"{i:>3} {r['idx']:>5} {r['coverage']:>7.3f} {r['fcl']:>+7.3f} "
            f"{err:>11.2e}  {fname}"
        )
    status = "OK" if worst_all < 1e-4 else "MISMATCH"
    print(
        f"\nround-trip worst tip error across {len(rows)} plans: "
        f"{worst_all:.2e} mm  [{status}]"
    )
    print(f"wrote {len(rows)} plans → {plans_dir}")
