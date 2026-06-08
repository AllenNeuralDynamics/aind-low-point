"""Discriminating experiment: does Stage 2 with the manual spin
(but no extra DOF) reach FCL feasibility?

If YES → Stage 2's DOF (arc_ap + ml + spin) is sufficient; the
multi-modality is purely in spin and a spin-only fix suffices.

If NO → Stage 2 lacks DOF (depth or offsets are required for FCL
feasibility); we need a structured full-DOF preview (the agent's
proposal).

Poses evaluated against FCL validator for cand #4195:

  1. Stage 2 polished (current): wrong spin, off=depth=0
  2. **Manual spin grafted onto Stage 2's polished ml + arc_ap**,
     off=depth=0
  3. **Manual spin + manual ml + manual arc_ap**, off=depth=0
     (i.e., the manual minus its depths and offsets)
  4. Full manual from plan.yml (already known: FCL-feasible)

If 2 or 3 are FCL-feasible while 1 is not, the basin is the issue.
If neither 2 nor 3 is feasible, we need depth/offset DOF.
"""

from __future__ import annotations

import argparse
import os as _os
import pickle
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import yaml

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    reduced_to_phase1,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import build_fixture_sdf_data


def _manual_pose_parts(plan_path, statics, n_arcs):
    with open(plan_path) as f:
        plan = yaml.safe_load(f)
    arcs_by_letter = plan["arcs"]
    probe_poses = plan["probes"]

    letter_for_arc_idx = {}
    for st in statics:
        letter = probe_poses[st.name]["arc"]
        letter_for_arc_idx[st.arc_idx] = letter

    arc_aps = np.zeros(n_arcs)
    for ai, letter in letter_for_arc_idx.items():
        arc_aps[ai] = float(arcs_by_letter[letter])

    per_probe = []
    for st in statics:
        pp = probe_poses[st.name]
        per_probe.append(
            {
                "ml": float(pp["slider_ml"]),
                "spin": float(pp["spin"]),
                "off_R": float(pp["offsets_RA"][0]),
                "off_A": float(pp["offsets_RA"][1]),
                "depth": float(pp["past_target_mm"]),
            }
        )
    return arc_aps, per_probe


def _build_x(arc_aps, per_probe, statics, n_arcs):
    """Build phase1_x (45-dim) from arc_aps + per-probe param dicts."""
    n_probes = len(statics)
    x = np.zeros(n_arcs + PHASE1_PER_PROBE_VARS * n_probes)
    x[:n_arcs] = arc_aps
    for i, p in enumerate(per_probe):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        x[off + 0] = p["ml"]
        spin_rad = np.deg2rad(p["spin"])
        x[off + 1] = float(np.cos(spin_rad))  # sx
        x[off + 2] = float(np.sin(spin_rad))  # sy
        x[off + 3] = p["off_R"]
        x[off + 4] = p["off_A"]
        x[off + 5] = p["depth"]
    return x


def _report(label, x, validator):
    s = validator.slacks(x)
    n_viol = int((s < -1e-4).sum())
    feas = n_viol == 0
    tag = "FEAS" if feas else "FAIL"
    print(f"\n[{tag}] {label}")
    print(f"   n_viol={n_viol}/{len(s)}  min_slack={s.min():+.4f} mm")
    if n_viol:
        for name, slack in validator.violating_pairs(x):
            print(f"     {name}: {slack:+.4f} mm")
    return feas


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument(
        "--plan", type=Path, default=Path("examples/836656-config-T12.plan.yml")
    )
    p.add_argument(
        "--polish-pkl", type=Path, default=Path("/tmp/full_polish_lbfgsb_augmented.pkl")
    )
    p.add_argument("--cand", type=int, default=4195)
    args = p.parse_args()

    print("Loading config / probes / SDFs / fixtures...", flush=True)
    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(args.holes)
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
    fixtures = build_fixture_sdf_data(runtime)
    bvh_cache = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fixture_bvhs = {
        f.name: make_fcl_bvh(runtime.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }

    with open(args.polish_pkl, "rb") as f:
        data = pickle.load(f)
    cand = data["candidates"][args.cand]
    jc = data["results"][args.cand]
    statics = _build_probe_static(
        probes,
        holes,
        cand.ha,
        cand.aa,
        bvh_cache=bvh_cache,
        sdf_by_name=sdf_by_name,
    )
    n_arcs = jc.n_arcs
    n_probes = len(statics)

    validator = make_fcl_validator(
        statics,
        n_arcs,
        fixtures=fixtures,
        fixture_bvhs=fixture_bvhs,
    )

    # Manual pose decomposed
    manual_arc_aps, manual_per_probe = _manual_pose_parts(args.plan, statics, n_arcs)

    # Stage 2 polished pose
    x_s2 = reduced_to_phase1(jc.reduced_y, n_arcs, n_probes)
    s2_arc_aps = np.asarray(x_s2[:n_arcs])
    s2_per_probe = []
    for i in range(n_probes):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        sx = float(x_s2[off + 1])
        sy = float(x_s2[off + 2])
        s2_per_probe.append(
            {
                "ml": float(x_s2[off + 0]),
                "spin": float(np.degrees(np.arctan2(sy, sx))),
                "off_R": 0.0,
                "off_A": 0.0,
                "depth": 0.0,
            }
        )

    print(f"\nCand #{args.cand}: n_probes={n_probes}, n_arcs={n_arcs}")
    print(f"  manual arc_aps:  {[float(v) for v in manual_arc_aps]}")
    print(f"  S2     arc_aps:  {[float(v) for v in s2_arc_aps]}")

    # ===== Pose 1: Stage 2 polished (control — known FAIL) =====
    _report("POSE 1: Stage 2 polished (off=depth=0)", x_s2, validator)

    # ===== Pose 2: Stage 2 arc_ap + ml, but MANUAL spin (and zero off/depth) =====
    pose2 = [
        {
            "ml": s2_per_probe[i]["ml"],
            "spin": manual_per_probe[i]["spin"],
            "off_R": 0.0,
            "off_A": 0.0,
            "depth": 0.0,
        }
        for i in range(n_probes)
    ]
    x_pose2 = _build_x(s2_arc_aps, pose2, statics, n_arcs)
    _report("POSE 2: Manual spin + S2 ml/arc_ap (off=depth=0)", x_pose2, validator)

    # ===== Pose 3: Manual spin + manual ml + manual arc_ap, off=depth=0 =====
    pose3 = [
        {
            "ml": manual_per_probe[i]["ml"],
            "spin": manual_per_probe[i]["spin"],
            "off_R": 0.0,
            "off_A": 0.0,
            "depth": 0.0,
        }
        for i in range(n_probes)
    ]
    x_pose3 = _build_x(manual_arc_aps, pose3, statics, n_arcs)
    _report(
        "POSE 3: Manual spin + manual ml + manual arc_ap (off=depth=0)",
        x_pose3,
        validator,
    )

    # ===== Pose 4: Full manual (sanity check — should be FEAS) =====
    x_pose4 = _build_x(manual_arc_aps, manual_per_probe, statics, n_arcs)
    _report("POSE 4: Full manual (incl. depths)", x_pose4, validator)

    # ===== Pose 5: Manual spin + manual ml + manual arc_ap + manual depth, off=0 =====
    pose5 = [
        {
            "ml": manual_per_probe[i]["ml"],
            "spin": manual_per_probe[i]["spin"],
            "off_R": 0.0,
            "off_A": 0.0,
            "depth": manual_per_probe[i]["depth"],
        }
        for i in range(n_probes)
    ]
    x_pose5 = _build_x(manual_arc_aps, pose5, statics, n_arcs)
    _report(
        "POSE 5: Manual spin + manual ml + manual arc_ap + "
        "manual depth (off=0; should match pose 4 since manual "
        "offsets are 0)",
        x_pose5,
        validator,
    )

    # ===== Pose 6: S2 polish + manual depth (no spin change) =====
    pose6 = [
        {
            "ml": s2_per_probe[i]["ml"],
            "spin": s2_per_probe[i]["spin"],
            "off_R": 0.0,
            "off_A": 0.0,
            "depth": manual_per_probe[i]["depth"],
        }
        for i in range(n_probes)
    ]
    x_pose6 = _build_x(s2_arc_aps, pose6, statics, n_arcs)
    _report("POSE 6: S2 polish + manual depth (NO spin change)", x_pose6, validator)

    # ===== Pose 7: Manual spin + manual depth + S2 ml/arc =====
    pose7 = [
        {
            "ml": s2_per_probe[i]["ml"],
            "spin": manual_per_probe[i]["spin"],
            "off_R": 0.0,
            "off_A": 0.0,
            "depth": manual_per_probe[i]["depth"],
        }
        for i in range(n_probes)
    ]
    x_pose7 = _build_x(s2_arc_aps, pose7, statics, n_arcs)
    _report("POSE 7: Manual spin + manual depth + S2 ml/arc", x_pose7, validator)

    print("\n" + "=" * 72)
    print("Now testing: do Phase 1 + Phase 2 chain recover feasibility")
    print("from a good-spin warm-start with zero offsets/depth?")
    print("=" * 72)

    # Inline the chain on each warm-start of interest
    from scipy.optimize import minimize as _scipy_minimize

    from aind_low_point.optimization.stage3_phase1_jax import (
        Phase1Weights,
        make_phase1_objective,
    )
    from aind_low_point.optimization.stage3_phase2_jax import (
        Phase2Weights,
        make_phase2,
    )
    from scripts.run_phase1_sample import (
        build_coverage_data as _build_cov,
    )
    from scripts.run_phase1_sample import (
        phase1_bounds,
    )

    coverage_data = _build_cov(probes, statics)
    bounds = phase1_bounds(n_arcs, n_probes)
    p1_fun, p1_jac = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=Phase1Weights(),
    )

    def _run_chain(x0, label):
        r1 = _scipy_minimize(
            p1_fun,
            x0,
            jac=p1_jac,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=80, ftol=1e-5, gtol=1e-5),
        )
        x1 = np.asarray(r1.x, dtype=np.float64)
        p2 = make_phase2(
            statics,
            n_arcs,
            coverage_data=coverage_data,
            fixtures=fixtures,
            weights=Phase2Weights(min_clearance_mm=0.3),
        )
        r2 = _scipy_minimize(
            p2["fun"],
            x1,
            jac=p2["jac"],
            method="trust-constr",
            bounds=bounds,
            constraints=p2["constraints_nlc"],
            options=dict(
                maxiter=80, xtol=1e-6, gtol=1e-5, initial_tr_radius=1.0, verbose=0
            ),
        )
        x2 = np.asarray(r2.x, dtype=np.float64)
        _report(label + " (chain exit)", x2, validator)

    print("\nChain from POSE 2 (manual spin + S2 ml/arc + off=depth=0):")
    _run_chain(x_pose2, "POSE 2")

    print("\nChain from POSE 3 (manual spin + manual ml/arc + off=depth=0):")
    _run_chain(x_pose3, "POSE 3")

    print("\nChain from POSE 1 (Stage 2 polish + off=depth=0; current behaviour):")
    _run_chain(x_s2, "POSE 1")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
