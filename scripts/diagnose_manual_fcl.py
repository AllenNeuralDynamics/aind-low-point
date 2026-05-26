"""Diagnose why the manual plan (config + plan.yml) fails the FCL chain.

Loads the manual pose DIRECTLY from the plan.yml file and compares 4
poses against the FCL validator for cand #4195:

  1. **Manual plan.yml**: arc_ap + per-probe (ml, spin, off, depth)
     read from the plan file. This is what the human drew on the rig.
  2. **Stage 2 polished**: ``reduced_y`` from the polish pkl, lifted
     with off=depth=0.
  3. **Augmented**: ``augmented_phase1_x`` (offset polish exit).
  4. **Phase 2 chain exit**: re-runs P1 + P2 from the augmented warm
     start, prints the resulting pose.

For each pose: per-probe params + FCL slacks (which pairs violate, by
how much). Reveals whether the chain drifts AWAY from a manual-FCL-
feasible pose, or whether the manual itself is FCL-infeasible per the
raw collision meshes.

Run::
    uv run --python 3.13 python -m scripts.diagnose_manual_fcl \\
        examples/836656-config-T12.yml /tmp/836656-holes.yml \\
        --plan examples/836656-config-T12.plan.yml \\
        --polish-pkl /tmp/full_polish_lbfgsb_augmented.pkl
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
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    Phase1Weights,
    make_phase1_objective,
    reduced_to_phase1,
)
from aind_low_point.optimization.stage3_phase2_jax import (
    Phase2Weights,
    make_phase2,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    phase1_bounds,
)


def _manual_x_from_plan(plan_path, statics, n_arcs):
    """Build phase1_x (45-dim) from a manual plan.yml.

    Reads ``arcs`` (letter → deg) + per-probe ``slider_ml``, ``spin``,
    ``past_target_mm``, ``offsets_RA`` from the plan file. Arc letter
    ↔ arc_idx mapping is inferred via each probe's ``arc`` field.
    """
    with open(plan_path) as f:
        plan = yaml.safe_load(f)
    arcs_by_letter = plan["arcs"]
    probe_poses = plan["probes"]

    letter_for_arc_idx: dict[int, str] = {}
    for st in statics:
        letter = probe_poses[st.name]["arc"]
        if st.arc_idx in letter_for_arc_idx:
            if letter_for_arc_idx[st.arc_idx] != letter:
                raise RuntimeError(
                    f"Inconsistent arc-letter mapping for arc_idx="
                    f"{st.arc_idx}: probe {st.name}'s plan says "
                    f"'{letter}' but earlier probes said "
                    f"'{letter_for_arc_idx[st.arc_idx]}'"
                )
        letter_for_arc_idx[st.arc_idx] = letter
    print(f"  Arc-letter mapping (from plan): {letter_for_arc_idx}")

    x = np.zeros(n_arcs + PHASE1_PER_PROBE_VARS * len(statics))
    for arc_idx, letter in letter_for_arc_idx.items():
        x[arc_idx] = float(arcs_by_letter[letter])

    for i, st in enumerate(statics):
        pp = probe_poses[st.name]
        ml = float(pp["slider_ml"])
        spin_deg = float(pp["spin"])
        depth = float(pp["past_target_mm"])
        off_R, off_A = (float(v) for v in pp["offsets_RA"])
        sx = float(np.cos(np.deg2rad(spin_deg)))
        sy = float(np.sin(np.deg2rad(spin_deg)))
        off_p = n_arcs + PHASE1_PER_PROBE_VARS * i
        x[off_p + 0] = ml
        x[off_p + 1] = sx
        x[off_p + 2] = sy
        x[off_p + 3] = off_R
        x[off_p + 4] = off_A
        x[off_p + 5] = depth
    return x


def _print_pose(label, x, statics, n_arcs):
    print(f"\n--- {label} ---")
    print(f"  arc_aps: {[float(v) for v in x[:n_arcs]]}")
    for i, st in enumerate(statics):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        sx = float(x[off + 1])
        sy = float(x[off + 2])
        spin = float(np.degrees(np.arctan2(sy, sx)))
        print(f"  {st.name:<8} arc_idx={st.arc_idx} ml={float(x[off]):+7.2f} "
              f"spin={spin:+7.2f}°  off_R={float(x[off+3]):+5.2f} "
              f"off_A={float(x[off+4]):+5.2f}  depth={float(x[off+5]):+5.2f}")


def _report_fcl(label, x, validator):
    s = validator.slacks(x)
    if s.size == 0:
        print(f"  {label}: no FCL pairs")
        return
    n_viol = int((s < -1e-4).sum())
    print(f"  {label}: n_viol={n_viol}/{len(s)}  "
          f"min_slack={s.min():+.4f}  feasible={n_viol == 0}")
    if n_viol:
        for name, slack in validator.violating_pairs(x):
            print(f"      {name}: {slack:+.4f} mm")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--plan", type=Path,
                   default=Path("examples/836656-config-T12.plan.yml"))
    p.add_argument("--polish-pkl", type=Path,
                   default=Path("/tmp/full_polish_lbfgsb_augmented.pkl"))
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
        probes, holes, cand.ha, cand.aa,
        bvh_cache=bvh_cache, sdf_by_name=sdf_by_name,
    )
    n_arcs = jc.n_arcs
    n_probes = len(statics)
    coverage_data = build_coverage_data(probes, statics)
    print(f"\nCand #{args.cand}: n_probes={n_probes}, n_arcs={n_arcs}, "
          f"violation_fn={data['violation_fn'][args.cand]:.2f}")
    print(f"  Probe → hole:")
    for st in statics:
        print(f"    {st.name:<8} → hole #{st.assigned_hole.id}  "
              f"arc_idx={st.arc_idx}")

    validator = make_fcl_validator(
        statics, n_arcs, fixtures=fixtures, fixture_bvhs=fixture_bvhs,
    )

    # ===== Pose 1: manual plan.yml =====
    x_manual = _manual_x_from_plan(args.plan, statics, n_arcs)
    _print_pose("Pose 1: MANUAL plan.yml", x_manual, statics, n_arcs)
    _report_fcl("FCL @ manual", x_manual, validator)

    # ===== Pose 2: Stage 2 polished =====
    x_s2 = reduced_to_phase1(jc.reduced_y, n_arcs, n_probes)
    _print_pose("Pose 2: STAGE 2 polished (off=depth=0)",
                x_s2, statics, n_arcs)
    _report_fcl("FCL @ Stage 2 polished", x_s2, validator)

    # ===== Pose 3: Augmented =====
    x_aug = np.asarray(data["augmented_phase1_x"][args.cand],
                       dtype=np.float64)
    _print_pose("Pose 3: AUGMENTED (offset polish exit)",
                x_aug, statics, n_arcs)
    _report_fcl("FCL @ Augmented", x_aug, validator)

    # ===== Pose 4: Phase 2 chain exit =====
    print("\nRunning Phase 1 + Phase 2 chain from augmented warm-start...",
          flush=True)
    bounds = phase1_bounds(n_arcs, n_probes)
    p1_fun, p1_jac = make_phase1_objective(
        statics, n_arcs, coverage_data=coverage_data,
        fixtures=fixtures, weights=Phase1Weights(),
    )
    r1 = minimize(p1_fun, x_aug, jac=p1_jac, method="L-BFGS-B",
                  bounds=bounds, options=dict(maxiter=80, ftol=1e-5,
                                              gtol=1e-5))
    x_p1 = np.asarray(r1.x, dtype=np.float64)
    p2 = make_phase2(
        statics, n_arcs, coverage_data=coverage_data,
        fixtures=fixtures,
        weights=Phase2Weights(min_clearance_mm=0.3),
    )
    r2 = minimize(p2["fun"], x_p1, jac=p2["jac"], method="trust-constr",
                  bounds=bounds, constraints=p2["constraints_nlc"],
                  options=dict(maxiter=80, xtol=1e-6, gtol=1e-5,
                               initial_tr_radius=1.0, verbose=0))
    x_p2 = np.asarray(r2.x, dtype=np.float64)
    _print_pose("Pose 4: PHASE 2 chain exit", x_p2, statics, n_arcs)
    _report_fcl("FCL @ Phase 2 exit", x_p2, validator)

    # ===== Delta summary =====
    print("\n=== Per-probe pose deltas (Phase 2 exit − manual) ===")
    for ai in range(n_arcs):
        print(f"  arc_aps[{ai}]: manual={x_manual[ai]:+6.2f}  "
              f"→ P2={x_p2[ai]:+6.2f}  (Δ={x_p2[ai] - x_manual[ai]:+6.2f})")
    for i, st in enumerate(statics):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        sx_m = float(x_manual[off + 1])
        sy_m = float(x_manual[off + 2])
        spin_m = float(np.degrees(np.arctan2(sy_m, sx_m)))
        sx_p = float(x_p2[off + 1])
        sy_p = float(x_p2[off + 2])
        spin_p = float(np.degrees(np.arctan2(sy_p, sx_p)))
        d_ml = float(x_p2[off + 0] - x_manual[off + 0])
        d_spin = ((spin_p - spin_m + 180.0) % 360.0) - 180.0
        d_oR = float(x_p2[off + 3] - x_manual[off + 3])
        d_oA = float(x_p2[off + 4] - x_manual[off + 4])
        d_dep = float(x_p2[off + 5] - x_manual[off + 5])
        print(f"  {st.name:<8} Δml={d_ml:+7.2f} Δspin={d_spin:+7.2f}° "
              f"Δoff_R={d_oR:+5.2f} Δoff_A={d_oA:+5.2f} "
              f"Δdep={d_dep:+5.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
