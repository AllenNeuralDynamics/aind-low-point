"""Re-run the chain (Phase 1 + Phase 2 + FCL validator) on a set of
candidates and save each as a standalone ConfigModel YAML that the
trame app can open.

Inputs: an augmented pkl (with ``augmented_phase1_x`` and
``violation_fn``) plus a list of cand indices. For each cand, runs the
chain from the augmented warm-start, then writes:

  ``plan-{rank:03d}-{feas_tag}-cand{idx}-cov{C}.yml``

where ``feas_tag`` ∈ ``feas`` / ``fail`` based on the FCL validator.
The cands are ordered by FCL feasibility first, then coverage descending.

Run::
    uv run --python 3.13 python -m scripts.save_chain_plans \\
        examples/836656-config-T12.yml /tmp/836656-holes.yml \\
        --polish-pkl /tmp/full_polish_lbfgsb_augmented.pkl \\
        --cands 5211 1040 1642 ... 4195 \\
        --out-dir examples/836656-config-T12_chain_alternatives
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
)
from aind_low_point.optimization.stage3_phase2_jax import (
    Phase2Weights,
    make_phase2,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from aind_low_point.runtime import save_plan_to_config
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    phase1_bounds,
)


def _apply_x_to_plan_state(plan_state, x, statics, n_arcs):
    """Mutate plan_state to reflect Phase 1/2's 45-dim ``x``.

    Converts (sx, sy) → spin via atan2. Arc letters a/b/c/… in
    arc-idx order.
    """
    arc_aps = x[:n_arcs]
    arc_letters = [chr(ord("a") + i) for i in range(n_arcs)]
    plan_state.kinematics.arc_angles = {
        arc_letters[i]: float(arc_aps[i]) for i in range(n_arcs)
    }
    for i, st in enumerate(statics):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        ml = float(x[off + 0])
        sx = float(x[off + 1])
        sy = float(x[off + 2])
        off_R = float(x[off + 3])
        off_A = float(x[off + 4])
        depth = float(x[off + 5])
        spin = float(np.degrees(np.arctan2(sy, sx)))
        plan = plan_state.probes[st.name]
        plan.arc_id = arc_letters[st.arc_idx]
        plan.bind_ap_to_arc = True
        plan.ap_local = 0.0
        plan.ml_local = ml
        plan.spin = spin
        plan.offsets_RA = (off_R, off_A)
        plan.past_target_mm = depth


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--polish-pkl", type=Path,
                   default=Path("/tmp/full_polish_lbfgsb_augmented.pkl"))
    p.add_argument("--cands", type=int, nargs="+", required=True,
                   help="Candidate indices to re-chain + save")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--p2-iter", type=int, default=80)
    p.add_argument("--min-clear", type=float, default=0.3)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

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
    cov_at_aug = data["coverage_at_aug"]
    viol_fn = data["violation_fn"]

    rows = []
    for cand_idx in args.cands:
        cand_idx = int(cand_idx)
        cand = data["candidates"][cand_idx]
        jc = data["results"][cand_idx]
        statics = _build_probe_static(
            probes, holes, cand.ha, cand.aa,
            bvh_cache=bvh_cache, sdf_by_name=sdf_by_name,
        )
        n_arcs = jc.n_arcs
        n_probes = len(statics)
        coverage_data = build_coverage_data(probes, statics)

        x_aug = np.asarray(data["augmented_phase1_x"][cand_idx],
                            dtype=np.float64)
        bounds = phase1_bounds(n_arcs, n_probes)

        # Phase 1
        p1_fun, p1_jac = make_phase1_objective(
            statics, n_arcs, coverage_data=coverage_data,
            fixtures=fixtures, weights=Phase1Weights(),
        )
        r1 = minimize(p1_fun, x_aug, jac=p1_jac, method="L-BFGS-B",
                      bounds=bounds,
                      options=dict(maxiter=80, ftol=1e-5, gtol=1e-5))
        x1 = np.asarray(r1.x, dtype=np.float64)

        # Phase 2
        p2 = make_phase2(
            statics, n_arcs, coverage_data=coverage_data,
            fixtures=fixtures,
            weights=Phase2Weights(min_clearance_mm=args.min_clear),
        )
        r2 = minimize(p2["fun"], x1, jac=p2["jac"], method="trust-constr",
                      bounds=bounds, constraints=p2["constraints_nlc"],
                      options=dict(maxiter=args.p2_iter, xtol=1e-6,
                                   gtol=1e-5, initial_tr_radius=1.0,
                                   verbose=0))
        x2 = np.asarray(r2.x, dtype=np.float64)

        # FCL validator
        validator = make_fcl_validator(
            statics, n_arcs, fixtures=fixtures, fixture_bvhs=fixture_bvhs,
        )
        s_fcl = validator.slacks(x2)
        fcl_min = float(s_fcl.min()) if s_fcl.size else 0.0
        feas = bool(s_fcl.size == 0 or s_fcl.min() >= -1e-4)

        rows.append((cand_idx, feas, float(cov_at_aug[cand_idx]),
                     float(viol_fn[cand_idx]), fcl_min, x2, statics, n_arcs))
        print(f"  cand#{cand_idx:>5}: feas={feas} cov={cov_at_aug[cand_idx]:.2f} "
              f"viol_fn={viol_fn[cand_idx]:.2f} fcl_min={fcl_min:+.4f}",
              flush=True)

    # Sort: feasible first, then coverage descending
    rows.sort(key=lambda r: (not r[1], -r[2]))

    # Write configs
    print(f"\nWriting {len(rows)} configs to {args.out_dir}...")
    summary_lines = [
        "# Chain-output plans",
        "",
        "Each row is a candidate re-chained from the augmented warm-start.",
        "FEAS = FCL validator says no collision. Files are full ConfigModel "
        "YAMLs and can be opened in trame.",
        "",
        "| rank | feas | cand | coverage | viol_fn | fcl_min | file |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for rank, (cand_idx, feas, cov, viol, fcl_min, x2, statics, n_arcs) \
            in enumerate(rows, start=1):
        # Fresh runtime/plan_state for each cand (avoid leaking pose state)
        cfg_local = ConfigModel.from_yaml(args.config)
        rt_local = build_runtime_from_config(cfg_local)
        _apply_x_to_plan_state(rt_local.plan_state, x2, statics, n_arcs)
        candidate_cfg = save_plan_to_config(rt_local.plan_state, cfg_local)
        tag = "feas" if feas else "fail"
        fname = (f"plan-{rank:03d}-{tag}-cand{cand_idx:05d}"
                 f"-cov{cov:05.2f}-viol{viol:06.2f}.yml")
        path = args.out_dir / fname
        with open(path, "w") as f:
            yaml.safe_dump(
                candidate_cfg.model_dump(mode="json"),
                f, sort_keys=False, default_flow_style=False,
            )
        summary_lines.append(
            f"| {rank} | {'yes' if feas else 'no'} | {cand_idx} | "
            f"{cov:.2f} | {viol:.2f} | {fcl_min:+.4f} | `{fname}` |"
        )
    summary_path = args.out_dir / "README.md"
    summary_path.write_text("\n".join(summary_lines))
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
