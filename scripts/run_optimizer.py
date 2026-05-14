"""Run the three-level placement optimizer on a config + holes file.

Loads a YAML config + the YAML produced by ``extract_implant_holes.py``,
runs ``optimization.optimize()``, and writes the optimized plan back
out as a new config YAML.

Usage::

    uv run --python 3.13 python scripts/run_optimizer.py \\
        examples/836656-config.yml \\
        /tmp/836656-holes.yml \\
        --max-num-arcs 4 \\
        -o examples/836656-config_opt.yml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.geometry import HoleSection
from aind_low_point.optimization.holes import Hole, load_holes
from aind_low_point.optimization.optimize import (
    OptimizationResult,
    ProbeStaticInfo,
    format_plan_table,
    optimize,
)
from aind_low_point.runtime import (
    build_runtime_from_config,
    detect_shank_tips_local,
    save_plan_to_config,
)
from aind_low_point.runtime.transforms import compile_all_transforms


def _transform_holes(holes: list[Hole], R: np.ndarray, t: np.ndarray) -> list[Hole]:
    """Apply a rigid transform (R, t) to every hole's positions and axis.
    Oval ``a/b/theta`` (in the per-axis basis) are invariant under
    rigid rotation, so they're preserved as-is."""
    out: list[Hole] = []
    for h in holes:
        new_axis = R @ np.asarray(h.axis, dtype=np.float64)
        new_axis = new_axis / np.linalg.norm(new_axis)
        new_ref = R @ np.asarray(h.ref_point, dtype=np.float64) + t
        new_sections = [
            HoleSection(
                axis=new_axis,
                center=R @ np.asarray(s.center, dtype=np.float64) + t,
                a=s.a,
                b=s.b,
                theta=s.theta,
            )
            for s in h.sections
        ]
        out.append(
            Hole(id=h.id, axis=new_axis, ref_point=new_ref, sections=new_sections)
        )
    return out


def _probe_static_info(plan_state, runtime, name: str) -> ProbeStaticInfo:
    """Build a ProbeStaticInfo for one probe by pulling its target,
    kind, and shank tips from the runtime."""
    plan = plan_state.probes[name]
    target_lps = None
    if plan.target_key is not None:
        target_pts = plan_state.target_index.get(plan.target_key)
        if target_pts is not None:
            target_lps = np.asarray(target_pts, dtype=np.float64).reshape(-1, 3).mean(0)
    if target_lps is None and plan.target_point_RAS is not None:
        from aind_anatomical_utils.coordinate_systems import (
            convert_coordinate_system,
        )

        ras = np.asarray(plan.target_point_RAS, dtype=np.float64).reshape(1, 3)
        target_lps = convert_coordinate_system(ras, "RAS", "LPS").reshape(3)
    if target_lps is None:
        raise RuntimeError(
            f"Probe {name}: no target_key or target_point_RAS — optimizer "
            f"needs an LPS target."
        )

    asset_key = f"probe:{plan.kind}"
    geom = runtime.asset_catalog.get_geometry(asset_key)
    # geom.raw is the canonicalized probe mesh (LPS-mm, shank 1 tip at origin)
    if hasattr(geom, "raw"):
        tips_local = detect_shank_tips_local(geom.raw)
    else:
        tips_local = np.zeros((1, 3), dtype=np.float64)
    return ProbeStaticInfo(
        name=name,
        target_LPS=target_lps,
        kind=plan.kind,
        shank_tips_local=tips_local,
    )


def _apply_result_to_plan_state(plan_state, result: OptimizationResult) -> None:
    """Mutate ``plan_state`` in place to reflect the optimizer's output.

    Maps the optimizer's flat variable vector to per-probe ProbePlan
    fields and per-arc kinematic angles. Arc letters are assigned
    a/b/c/... in the order arcs appear in ``result``.
    """
    n_arcs = result.n_arcs
    arc_aps = result.x[:n_arcs]
    # Assign deterministic arc letters by ascending arc-index so the
    # mapping is reproducible.
    arc_letters = [chr(ord("a") + i) for i in range(n_arcs)]
    plan_state.kinematics.arc_angles = {
        arc_letters[i]: float(arc_aps[i]) for i in range(n_arcs)
    }

    layout_probe_names = sorted(result.probe_to_hole.keys())
    for probe_idx, name in enumerate(layout_probe_names):
        offset = n_arcs + 5 * probe_idx
        ml, spin, off_R, off_A, depth = result.x[offset : offset + 5]
        plan = plan_state.probes[name]
        plan.arc_id = arc_letters[result.probe_to_arc_idx[name]]
        plan.bind_ap_to_arc = True
        plan.ap_local = 0.0
        plan.ml_local = float(ml)
        plan.spin = float(spin)
        plan.offsets_RA = (float(off_R), float(off_A))
        plan.past_target_mm = float(depth)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path, help="Path to input config YAML")
    p.add_argument("holes", type=Path, help="Path to holes YAML")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to write optimized config (default: <config>_opt.yml)",
    )
    p.add_argument(
        "--max-num-arcs",
        type=int,
        default=4,
        help="Max number of arcs the optimizer can use",
    )
    p.add_argument("--min-num-arcs", type=int, default=1, help="Min number of arcs")
    p.add_argument(
        "--arc-count-penalty-deg2",
        type=float,
        default=25.0,
        help="Cost added per arc beyond --min-num-arcs (deg^2). "
        "Default 25.0 prefers fewer arcs; set 0.0 to remove the "
        "preference and let the inner loop decide on arc count.",
    )
    p.add_argument(
        "--k-holes", type=int, default=5, help="Top-K hole assignments to evaluate"
    )
    p.add_argument(
        "--k-arcs",
        type=int,
        default=3,
        help="Top-K arc assignments per hole assignment",
    )
    p.add_argument(
        "--no-cma",
        action="store_true",
        help="Skip CMA-ES global stage; SLSQP polish only",
    )
    p.add_argument(
        "--slsqp-soft",
        action="store_true",
        help="Use soft penalties (legacy) instead of native SLSQP "
        "inequality constraints during the polish step.",
    )
    p.add_argument(
        "--no-two-stage-inner",
        action="store_true",
        help="Skip the Stage-A feasibility solve before the constrained "
        "SLSQP polish. Stage A minimises Σ ReLU(g_j(x))² to land "
        "near the feasibility tube; default is enabled.",
    )
    p.add_argument(
        "--feasibility-max-iter",
        type=int,
        default=80,
        help="Max SLSQP iterations for the Stage-A feasibility solve.",
    )
    p.add_argument(
        "--slsqp-max-iter",
        type=int,
        default=100,
        help="Max SLSQP iterations for the Stage-B coverage polish. "
        "Bump higher (e.g. 300) when residual violations look like "
        "convergence rather than basin issues.",
    )
    p.add_argument(
        "--no-final-feasibility-cleanup",
        action="store_true",
        help="Skip the Stage-C feasibility re-projection. Stage C runs "
        "feasibility_solve from the Stage-B output and keeps "
        "whichever (B, C) candidate has the better lex key. Default "
        "is enabled — disable only when comparing against legacy "
        "two-stage runs.",
    )
    p.add_argument(
        "--polish-method",
        type=str,
        default="SLSQP",
        choices=["SLSQP", "trust-constr"],
        help="Stage-B polish method. SLSQP (default) is fast but can "
        "drift off the feasibility tube; trust-constr is interior-"
        "point and maintains feasibility throughout, at higher "
        "per-iteration cost.",
    )
    p.add_argument(
        "--feasibility-threshold",
        type=float,
        default=0.0,
        help="Lex-rank tiebreaker: plans with max_violation <= ε are "
        "treated as 'feasible enough' and ranked by coverage "
        "instead. Default 0 = strict feasibility-first. Set ε to "
        "the physical 'slop budget' the manual workflow tolerates "
        "(e.g. 0.5 if 0.5 mm of clearance overlap or 0.5° of "
        "AP/ML-sep margin is fine in practice).",
    )
    p.add_argument(
        "--cma-stage-multipliers",
        type=str,
        default="0.1,1.0,10.0",
        help="Comma-separated feasibility-penalty multipliers per CMA stage. "
        "Empty string disables homotopy and runs single-stage. "
        "Default '0.1,1.0,10.0' = 3 stages from soft to hard.",
    )
    p.add_argument(
        "--min-arc-ap-sep-deg",
        type=float,
        default=16.0,
        help="Min AP separation between arc centroids (rig limit)",
    )
    p.add_argument(
        "--arc-sep-shortfall-weight",
        type=float,
        default=10.0,
        help="Soft penalty (deg^-2) on AP-centroid pairs closer than "
        "--min-arc-ap-sep-deg. Default 10.0 lets the inner loop "
        "rescue marginally-spaced partitions; pass 'inf' to "
        "recover the legacy hard-AP-sep middle-layer filter.",
    )
    p.add_argument(
        "--threading-oval-tolerance",
        type=float,
        default=0.0,
        help="Allow ``g_thread <= tol`` instead of strict ``g <= 0`` "
        "(``g = (u/a)² + (v/b)² − 1`` at the shaft-section "
        "intersection). ``tol = K² − 1`` corresponds to 'shaft "
        "within K oval-radii of slot centre'. Default 0 = strict.",
    )
    p.add_argument(
        "--clearance-overlap-allowance-mm",
        type=float,
        default=0.0,
        help="Allow headstage capsules to overlap by up to this many mm "
        "before the clearance constraint flags them. Default 0 = "
        "strict; non-zero values trade model strictness for "
        "matching observed manual practice.",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose log")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    plan_state = runtime.plan_state
    holes = load_holes(args.holes)
    # Holes are extracted from the implant OBJ in its local LPS frame;
    # transform them into subject LPS via implant_to_lps so they line
    # up with the targets.
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes = _transform_holes(holes, R, t)
        print(f"Applied implant_to_lps transform to {len(holes)} hole(s)")
    print(f"Loaded {len(holes)} hole spec(s) from {args.holes}")
    print(f"Loaded {len(plan_state.probes)} probe(s) from {args.config}")

    probes = [
        _probe_static_info(plan_state, runtime, name) for name in plan_state.probes
    ]

    print(
        f"Running optimizer (max_num_arcs={args.max_num_arcs}, "
        f"k_holes={args.k_holes}, k_arcs={args.k_arcs})..."
    )
    stage_mults_str = args.cma_stage_multipliers.strip()
    if stage_mults_str:
        stage_mults = tuple(float(x) for x in stage_mults_str.split(","))
    else:
        stage_mults = ()
    result = optimize(
        probes,
        holes,
        max_num_arcs=args.max_num_arcs,
        min_num_arcs=args.min_num_arcs,
        arc_count_penalty_deg2=args.arc_count_penalty_deg2,
        k_holes=args.k_holes,
        k_arcs=args.k_arcs,
        min_arc_ap_sep_deg=args.min_arc_ap_sep_deg,
        arc_sep_shortfall_weight=args.arc_sep_shortfall_weight,
        threading_oval_tolerance=args.threading_oval_tolerance,
        clearance_overlap_allowance_mm=args.clearance_overlap_allowance_mm,
        final_feasibility_cleanup=not args.no_final_feasibility_cleanup,
        polish_method=args.polish_method,
        feasibility_threshold=args.feasibility_threshold,
        use_cma=not args.no_cma,
        cma_stage_multipliers=stage_mults,
        slsqp_constrained=not args.slsqp_soft,
        two_stage_inner=not args.no_two_stage_inner,
        feasibility_max_iter=args.feasibility_max_iter,
        slsqp_max_iter=args.slsqp_max_iter,
        verbose=args.verbose,
    )
    if result is None:
        print("Optimizer returned no feasible solution.")
        return 1

    print(f"\nOptimization result (cost={result.cost:.3f}):")
    print(f"  n_arcs = {result.n_arcs}")
    print(f"  arc AP centroids = {[f'{v:+.1f}°' for v in result.arc_centroids_deg]}")
    print(f"  probe → hole : {result.probe_to_hole}")
    print(f"  probe → arc  : {result.probe_to_arc_idx}")
    print(
        f"  breakdown: coverage={result.breakdown.coverage_total:.3f}, "
        f"thread={result.breakdown.threading_penalty:.3f}, "
        f"clear={result.breakdown.clearance_penalty:.3f}, "
        f"kin={result.breakdown.kinematic_penalty:.3f}"
    )

    if result.alternatives:
        print(f"\nLex-ranked candidates ({len(result.alternatives)} total):")
        print(format_plan_table(result.alternatives))
        feasible_count = sum(1 for c in result.alternatives if c.feasible)
        print(
            f"\n{feasible_count}/{len(result.alternatives)} candidates feasible. "
            f"Best plan is "
            f"{'feasible' if result.alternatives[0].feasible else 'INFEASIBLE'}."
        )

    _apply_result_to_plan_state(plan_state, result)
    new_cfg = save_plan_to_config(plan_state, cfg)

    out_path = args.output
    if out_path is None:
        out_path = args.config.with_stem(args.config.stem + "_opt")
    with open(out_path, "w") as f:
        yaml.safe_dump(
            new_cfg.model_dump(mode="json"),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    print(f"\nWrote optimized config to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
