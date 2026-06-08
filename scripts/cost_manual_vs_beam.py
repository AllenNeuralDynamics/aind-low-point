"""Evaluate Phase 1 cost (and component breakdown) for the manual plan
vs the 3 beam-found feasible plans for cand 4195.

Builds x vectors from each plan file, evaluates make_phase1_objective at
each, and reports FCL ground-truth slack alongside.

Notes:
- All plans share cand 4195's HA/AA (hole + arc) assignment.
- Arc index ordering in the x vector follows cand 4195 (arc_idx 0 = arc
  with most-negative centroid).
- Cost is the Phase 1 objective with default :class:`Phase1Weights`.
- Component costs are the same objective with everything but one term
  zeroed (so they sum to ~total minus the unit-circle / coverage offset).
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

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
from scripts.run_phase1_sample import build_coverage_data, build_fixture_sdf_data


def plan_to_x(
    plan_data: dict, statics: list, n_arcs: int, probe_to_arc_idx: dict[str, int]
) -> np.ndarray:
    """Construct the 45-dim Phase 1 x vector from a plan dict.

    Resolves the arc-letter → arc-idx mapping from the plan itself: each
    probe in the plan names its arc letter, and ``probe_to_arc_idx``
    tells us which arc_idx that probe occupies in the cand 4195 layout.
    """
    n_probes = len(statics)
    arc_aps_dict = plan_data["arcs"]
    probes_pd = plan_data["probes"]

    # Build arc_idx → letter map from this plan
    arc_idx_to_letter: dict[int, str] = {}
    for name, info in probes_pd.items():
        if name not in probe_to_arc_idx:
            continue
        idx = probe_to_arc_idx[name]
        letter = info["arc"]
        if idx in arc_idx_to_letter and arc_idx_to_letter[idx] != letter:
            raise ValueError(
                f"plan inconsistent: arc_idx {idx} has two letters "
                f"{arc_idx_to_letter[idx]!r} and {letter!r}"
            )
        arc_idx_to_letter[idx] = letter

    arc_aps = np.array(
        [float(arc_aps_dict[arc_idx_to_letter[i]]) for i in range(n_arcs)]
    )

    x = np.zeros(n_arcs + PHASE1_PER_PROBE_VARS * n_probes, dtype=np.float64)
    x[:n_arcs] = arc_aps

    for i, st in enumerate(statics):
        probe = probes_pd[st.name]
        ml = float(probe["slider_ml"])
        spin = float(probe["spin"])
        rad = np.deg2rad(spin)
        offsets = probe.get("offsets_RA", [0.0, 0.0])
        off_R = float(offsets[0])
        off_A = float(offsets[1])
        depth = float(probe.get("past_target_mm", 0.0))
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        x[off + 0] = ml
        x[off + 1] = np.cos(rad)
        x[off + 2] = np.sin(rad)
        x[off + 3] = off_R
        x[off + 4] = off_A
        x[off + 5] = depth
    return x


def yml_to_plan_data(path: Path) -> dict:
    """Extract a ``{arcs, probes}`` view (matching plan.yml shape) from a
    beam-saved ConfigModel YAML.

    The saved files have ``plan: {arcs: ..., probes: {name: {arc, slider_ml,
    spin, offsets_RA, past_target_mm, ...}}}``."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["plan"]


def cost_with_field_zeroed(
    x: np.ndarray,
    statics: list,
    n_arcs: int,
    coverage_data,
    fixtures,
    field: str | None,
) -> float:
    """Evaluate Phase 1 cost at ``x`` with one lambda set to 0 (others
    default). ``field`` is e.g. ``lambda_clearance_fixture`` or ``None``
    for full default weights.

    The delta vs the full-default cost tells you exactly how much that
    one term contributed to the total.
    """
    w = Phase1Weights() if field is None else replace(Phase1Weights(), **{field: 0.0})
    fn, _ = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=w,
    )
    return float(fn(x))


def main() -> int:
    print("Loading config / probes / SDFs / fixtures...", flush=True)
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
    fixtures = build_fixture_sdf_data(runtime)
    bvh_cache = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fixture_bvhs = {
        f.name: make_fcl_bvh(runtime.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }

    with open("/tmp/full_polish_unitcircle.pkl", "rb") as f:
        data = pickle.load(f)
    cand_idx = 4195
    cand = data["candidates"][cand_idx]
    jc = data["results"][cand_idx]
    statics = _build_probe_static(
        probes,
        holes,
        cand.ha,
        cand.aa,
        bvh_cache=bvh_cache,
        sdf_by_name=sdf_by_name,
    )
    n_arcs = jc.n_arcs
    coverage_data = build_coverage_data(probes, statics)
    validator = make_fcl_validator(
        statics,
        n_arcs,
        fixtures=fixtures,
        fixture_bvhs=fixture_bvhs,
    )

    # cand 4195's probe → arc_idx map
    probe_to_arc_idx = dict(cand.aa.probe_to_arc_idx)

    # Manual plan
    with open("examples/836656-config-T12.plan.yml") as f:
        manual_plan = yaml.safe_load(f)
    plans = {"manual": ("examples/836656-config-T12.plan.yml", manual_plan)}
    beam_dir = Path("examples/836656-config-T12_4195_beam")
    for f in sorted(beam_dir.glob("plan-*.yml")):
        plans[f.stem] = (str(f), yml_to_plan_data(f))

    # For each plan: full cost, then -delta when each lambda is zeroed.
    # The delta IS that term's contribution to the total.
    fields = [
        ("thread", "lambda_thread"),
        ("clear", "lambda_clearance"),
        ("fixture", "lambda_clearance_fixture"),
        ("kine", "lambda_kinematic"),
        ("bounds", "lambda_bounds"),
        ("unit_c", "lambda_unit_circle"),
        ("m_clr", "lambda_margin_clear"),
        ("m_thd", "lambda_margin_thread"),
    ]
    rows = []
    for label, (path, plan_data) in plans.items():
        x = plan_to_x(plan_data, statics, n_arcs, probe_to_arc_idx)
        s_fcl = validator.slacks(x)
        n_viol = int((s_fcl < -1e-4).sum()) if s_fcl.size else 0
        total = cost_with_field_zeroed(
            x,
            statics,
            n_arcs,
            coverage_data,
            fixtures,
            field=None,
        )
        deltas: dict[str, float] = {}
        for short, attr in fields:
            v = cost_with_field_zeroed(
                x,
                statics,
                n_arcs,
                coverage_data,
                fixtures,
                field=attr,
            )
            deltas[short] = total - v
        rows.append((label, total, deltas, n_viol, float(s_fcl.min())))

    # Pretty print
    print("\nPhase 1 cost (total) and per-term contribution to total\n")
    short_names = [s for s, _ in fields]
    header = (
        f"{'plan':<36} {'total':>10}  "
        + " ".join(f"{s:>9}" for s in short_names)
        + f"  {'fcl':>9} {'viol':>4}"
    )
    print(header)
    print("-" * len(header))
    for label, total, deltas, n_viol, fcl_min in rows:
        line = f"{label:<36} {total:>+10.3f}  "
        line += " ".join(f"{deltas[s]:>+9.3f}" for s in short_names)
        line += f"  {fcl_min:>+9.3f} {n_viol:>4d}"
        print(line)

    print()
    print("Notes:")
    print("- 'total' is the full Phase 1 objective with default Phase1Weights.")
    print("- Each per-term column is the CONTRIBUTION of that lambda to the")
    print("  total (computed as total − cost_with_lambda_zeroed).")
    print("- Positive values are penalties; negative values (m_clr, m_thd)")
    print("  are saturating margin rewards (good).")
    print("- 'fcl' is the FCL validator min slack (mm). Positive = clear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
