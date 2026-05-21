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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from aind_low_point.config import ConfigModel, PlanningModel
from aind_low_point.optimization.geometry import HoleSection
from aind_low_point.optimization.holes import Hole, load_holes
from aind_low_point.optimization.joint_rerank import JointWeights, optimize_joint
from aind_low_point.optimization.optimize import (
    OptimizationResult,
    PlanCandidate,
    ProbeStaticInfo,
    best_fit_hole_id_at_pose,
    format_plan_table,
    optimize,
    polish_seed,
)
from aind_low_point.runtime import (
    build_runtime_from_config,
    detect_shank_tips_local,
    save_plan_to_config,
)
from aind_low_point.runtime.export import apply_plan_model_to_state
from aind_low_point.runtime.transforms import compile_all_transforms
from aind_low_point.scene import resolve_base_geometry
from aind_low_point.state_change import PlanStore


def _maybe_build_sdfs(probes, runtime):
    """Build (or load-from-cache) the per-probe SDFs for the
    ``--sdf-clearance`` path. Returns a ``{probe_name: ProbeSDF}`` dict.
    """
    from aind_low_point.optimization.sdf import build_probe_sdf

    sdf_by_name: dict = {}
    print("Building SDFs for clearance backend...")
    for p in probes:
        mesh = runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        sdf_by_name[p.name] = build_probe_sdf(mesh)
    return sdf_by_name


@dataclass(frozen=True)
class RetroDensityOpts:
    """When set, the optimizer's per-probe target becomes a masked
    point cloud drawn from a labelled-cell asset (e.g. retro/rabies
    points), clipped to the intersection of brain and structure masks.

    The per-probe density switches from a single-point Gaussian on the
    centroid to an equally-weighted Gaussian mixture on the masked
    points with bandwidth ``sigma_mm``. ``target_LPS`` is set to the
    centroid of the masked cloud for LSAP pose-bank anchoring.
    """

    retro_asset_key: str = "retro-targets"
    common_mask_keys: tuple[str, ...] = ("brain",)
    per_probe_mask_fmt: str = "structure:{probe}"
    sigma_mm: float = 0.3


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


def _resolve_masked_retro_points(
    runtime, probe_name: str, opts: RetroDensityOpts
) -> np.ndarray:
    """Return the retro-asset points in world LPS, clipped to the
    intersection of all configured masks. Raises with a clear message
    if any asset is missing or the masked cloud is empty."""
    catalog, scene = runtime.asset_catalog, runtime.scene
    retro_t = resolve_base_geometry(catalog, scene, opts.retro_asset_key)
    if retro_t is None:
        raise RuntimeError(
            f"--retro-density: asset {opts.retro_asset_key!r} not in scene"
        )
    points = np.asarray(retro_t.raw, dtype=np.float64)
    mask_keys = list(opts.common_mask_keys) + [
        opts.per_probe_mask_fmt.format(probe=probe_name)
    ]
    keep = np.ones(len(points), dtype=bool)
    for mk in mask_keys:
        mt = resolve_base_geometry(catalog, scene, mk)
        if mt is None:
            raise RuntimeError(f"--retro-density: mask asset {mk!r} not in scene")
        keep &= np.asarray(mt.raw.contains(points), dtype=bool)
    masked = points[keep]
    if masked.shape[0] == 0:
        raise RuntimeError(
            f"--retro-density: probe {probe_name!r} has zero points in "
            f"{' ∩ '.join(mask_keys)} — check mask alignment"
        )
    return masked


def _probe_static_info(
    plan_state,
    runtime,
    name: str,
    retro_opts: RetroDensityOpts | None = None,
) -> ProbeStaticInfo:
    """Build a ProbeStaticInfo for one probe by pulling its target,
    kind, and shank tips from the runtime.

    When ``retro_opts`` is provided, the target switches from a single
    point (centroid or inline) to a masked retro-point cloud with the
    centroid serving as ``target_LPS`` for LSAP pose-bank anchoring.
    """
    plan = plan_state.probes[name]
    target_lps = None
    target_points = None
    if retro_opts is not None:
        target_points = _resolve_masked_retro_points(runtime, name, retro_opts)
        target_lps = target_points.mean(0)
    else:
        if plan.target_key is not None:
            target_pts = plan_state.target_index.get(plan.target_key)
            if target_pts is not None:
                target_lps = (
                    np.asarray(target_pts, dtype=np.float64).reshape(-1, 3).mean(0)
                )
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
        collision_mesh = geom.raw  # full probe mesh, used as BVH for clearance
    else:
        tips_local = np.zeros((1, 3), dtype=np.float64)
        collision_mesh = None
    sigma = retro_opts.sigma_mm if retro_opts is not None else 0.5
    return ProbeStaticInfo(
        name=name,
        target_LPS=target_lps,
        kind=plan.kind,
        shank_tips_local=tips_local,
        density_sigma_mm=sigma,
        collision_mesh=collision_mesh,
        target_points=target_points,
    )


def _apply_candidate_to_plan_state(plan_state, cand: PlanCandidate) -> None:
    """Mutate ``plan_state`` in place to reflect one PlanCandidate's
    arc angles + per-probe pose. Arc letters assigned a/b/c/... in
    ascending arc-index order so the mapping is reproducible.

    The optimizer's ``x`` is laid out in input-probes order (the order
    of the ``probes`` list passed to :func:`optimize`, preserved on
    ``cand.probe_to_hole.keys()`` via dict insertion order). Reading in
    any other order — e.g. alphabetical — assigns one probe's optimized
    ``(ml, spin, offsets, depth)`` to a different probe and silently
    corrupts the saved plan.
    """
    n_arcs = cand.n_arcs
    arc_aps = cand.x[:n_arcs]
    arc_letters = [chr(ord("a") + i) for i in range(n_arcs)]
    plan_state.kinematics.arc_angles = {
        arc_letters[i]: float(arc_aps[i]) for i in range(n_arcs)
    }
    layout_probe_names = list(cand.probe_to_hole.keys())
    for probe_idx, name in enumerate(layout_probe_names):
        offset = n_arcs + 5 * probe_idx
        ml, spin, off_R, off_A, depth = cand.x[offset : offset + 5]
        plan = plan_state.probes[name]
        plan.arc_id = arc_letters[cand.probe_to_arc_idx[name]]
        plan.bind_ap_to_arc = True
        plan.ap_local = 0.0
        plan.ml_local = float(ml)
        plan.spin = float(spin)
        plan.offsets_RA = (float(off_R), float(off_A))
        plan.past_target_mm = float(depth)


def _apply_result_to_plan_state(plan_state, result: OptimizationResult) -> None:
    """Apply the lex-best plan from ``result`` to ``plan_state``.

    Thin wrapper around :func:`_apply_candidate_to_plan_state` that
    constructs a ``PlanCandidate``-shaped view of the result's top
    fields. Preserved for back-compat with the result-based callsite.
    """
    n_arcs = result.n_arcs
    arc_aps = result.x[:n_arcs]
    arc_letters = [chr(ord("a") + i) for i in range(n_arcs)]
    plan_state.kinematics.arc_angles = {
        arc_letters[i]: float(arc_aps[i]) for i in range(n_arcs)
    }
    layout_probe_names = list(result.probe_to_hole.keys())
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


def _save_alternatives(
    plan_state,
    cfg: ConfigModel,
    result: OptimizationResult,
    out_dir: Path,
) -> Path:
    """Write every alternative as a full ConfigModel YAML in ``out_dir``.

    Files are named ``plan-NN-cov-V-viol-V.yml`` where NN is the lex
    rank (01 = best) and V is the candidate's coverage and max
    violation (zero-padded for sortability). Also writes
    ``summary.md`` with a markdown table of the alternatives' metrics
    for quick scanning.

    Returns the directory path. The caller is responsible for ordering
    this call so that ``plan_state``'s final state matches the lex-best
    plan (see callsite in ``main``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = [
        "# Alternative plans",
        "",
        "Each row is a candidate from the optimizer's lex-ranked list. "
        "`plan-01-...yml` is the best plan (also written to the main "
        "`--output` path); the rest are alternatives. All files are full "
        "ConfigModel YAMLs and can be re-opened in trame.",
        "",
        "| # | feasible? | max viol | coverage | min clearance (mm) | "
        "min AP sep (°) | min ML sep (°) | dominant group | file |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, cand in enumerate(result.alternatives, start=1):
        cov = cand.coverage
        viol = cand.max_violation
        # Snapshot of plan_state into a candidate state, then serialise.
        _apply_candidate_to_plan_state(plan_state, cand)
        candidate_cfg = save_plan_to_config(plan_state, cfg)
        fname = f"plan-{rank:02d}-cov-{cov:06.2f}-viol-{viol:.4g}.yml"
        path = out_dir / fname
        with open(path, "w") as f:
            yaml.safe_dump(
                candidate_cfg.model_dump(mode="json"),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        summary_lines.append(
            f"| {rank} | {'yes' if cand.feasible else 'no'} | "
            f"{viol:.4g} | {cov:.3f} | "
            f"{cand.min_headstage_clearance_mm:.3f} | "
            f"{cand.min_arc_ap_sep_deg:.2f} | "
            f"{cand.min_intra_arc_ml_sep_deg:.2f} | "
            f"{cand.dominant_violation_group} | "
            f"`{fname}` |"
        )
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    return out_dir


def _run_seed_polish(
    probes, plan_state, holes, *, args, stage_mults, subject_from_rig_rot=None
):
    """Run the inner solve from a seed plan that was already applied to
    ``plan_state`` by the caller. Skips outer + middle layers entirely.

    Builds the discrete (probe→hole, probe→arc) assignments from the seed
    plan's current PlanningState (hole auto-detected per-probe by static
    threading max_g; arc ordering by ascending AP angle), packages an x0
    vector from the per-probe plan variables, and calls ``polish_seed``.
    """
    probe_names = [p.name for p in probes]

    # Arc letter → arc_idx by ascending AP angle (matches optimizer's
    # arc_0..arc_{n-1} convention).
    arc_letters_used: dict[str, float] = {}
    for name in probe_names:
        plan = plan_state.probes[name]
        if plan.arc_id is None:
            print(f"Probe {name}: arc_id is None in seed plan; aborting.")
            return 1
        arc_letter: str = plan.arc_id
        ap = float(plan_state.kinematics.get_arc(arc_letter))
        arc_letters_used[arc_letter] = ap
    sorted_letters = sorted(arc_letters_used, key=lambda k: arc_letters_used[k])
    letter_to_idx = {letter: i for i, letter in enumerate(sorted_letters)}
    arc_centroids_deg = tuple(arc_letters_used[L] for L in sorted_letters)
    print(
        f"Seed arcs (ascending AP): "
        f"{[(L, arc_letters_used[L]) for L in sorted_letters]}"
    )

    # Per-probe best-fit hole at the seed pose.
    probe_to_hole: dict[str, int] = {}
    probe_to_arc_idx: dict[str, int] = {}
    seed_pose_max_g: dict[str, float] = {}
    for ps in probes:
        plan = plan_state.probes[ps.name]
        assert plan.arc_id is not None
        ap = float(plan_state.kinematics.get_arc(plan.arc_id))
        hole_id, max_g = best_fit_hole_id_at_pose(
            ps,
            holes,
            ap_deg=ap,
            ml_deg=float(plan.ml_local),
            spin_deg=float(plan.spin),
            off_R_mm=float(plan.offsets_RA[0]),
            off_A_mm=float(plan.offsets_RA[1]),
            past_target_mm=float(plan.past_target_mm),
        )
        probe_to_hole[ps.name] = hole_id
        probe_to_arc_idx[ps.name] = letter_to_idx[plan.arc_id]
        seed_pose_max_g[ps.name] = max_g
    print("\nSeed probe → (hole, arc), with static threading max_g at seed pose:")
    for name in probe_names:
        plan = plan_state.probes[name]
        print(
            f"  {name:>4}  kind={plan.kind:<18} arc={plan.arc_id}  "
            f"hole={probe_to_hole[name]:>2}  max_g={seed_pose_max_g[name]:+.4f}"
        )

    # Build x0 from the per-probe plan variables. Layout: arc APs first,
    # then (ml, spin, off_R, off_A, depth) per probe in probe-list order.
    n_arcs = len(sorted_letters)
    n_vars = n_arcs + 5 * len(probe_names)
    x0 = np.zeros(n_vars, dtype=np.float64)
    for L, idx in letter_to_idx.items():
        x0[idx] = arc_letters_used[L]
    for p_idx, name in enumerate(probe_names):
        plan = plan_state.probes[name]
        off = n_arcs + 5 * p_idx
        x0[off + 0] = float(plan.ml_local)
        x0[off + 1] = float(plan.spin)
        x0[off + 2] = float(plan.offsets_RA[0])
        x0[off + 3] = float(plan.offsets_RA[1])
        x0[off + 4] = float(plan.past_target_mm)

    print(
        f"\nRunning seed polish "
        f"(cma={'on' if args.seed_use_cma else 'off'}, "
        f"two_stage={not args.no_two_stage_inner}, "
        f"polish_method={args.polish_method}, "
        f"final_cleanup={not args.no_final_feasibility_cleanup})..."
    )
    cand: PlanCandidate = polish_seed(
        probes,
        holes,
        probe_to_hole=probe_to_hole,
        probe_to_arc_idx=probe_to_arc_idx,
        arc_centroids_deg=arc_centroids_deg,
        x0=x0,
        use_cma=args.seed_use_cma,
        cma_stage_multipliers=stage_mults,
        slsqp_max_iter=args.slsqp_max_iter,
        slsqp_constrained=not args.slsqp_soft,
        two_stage_inner=not args.no_two_stage_inner,
        feasibility_max_iter=args.feasibility_max_iter,
        final_feasibility_cleanup=not args.no_final_feasibility_cleanup,
        polish_method=args.polish_method,
        feasibility_threshold=args.feasibility_threshold,
        threading_oval_tolerance=args.threading_oval_tolerance,
        clearance_overlap_allowance_mm=args.clearance_overlap_allowance_mm,
        min_arc_ap_sep_deg=args.min_arc_ap_sep_deg,
        subject_from_rig_rot=subject_from_rig_rot,
        verbose=args.verbose,
    )

    print(f"\nSeed-polish result (cost={cand.cost:.3f}):")
    print(f"  feasible (strict)            : {cand.feasible}")
    print(f"  max_violation (any group)    : {cand.max_violation:.4f}")
    print(f"  sum_violation_sq             : {cand.sum_violation_sq:.4f}")
    print(f"  coverage_total               : {cand.coverage:.4f}")
    print(f"  min_headstage_clearance (mm) : {cand.min_headstage_clearance_mm:.3f}")
    print(f"  min_arc_ap_sep (deg)         : {cand.min_arc_ap_sep_deg:.2f}")
    print(f"  min_intra_arc_ml_sep (deg)   : {cand.min_intra_arc_ml_sep_deg:.2f}")
    print(f"  dominant violation group     : {cand.dominant_violation_group}")
    print("  per-group max violation:")
    print(f"    threading       : {cand.max_violation_threading:.4f}")
    print(f"    clearance       : {cand.max_violation_clearance:.4f}")
    print(f"    arc_ap_sep      : {cand.max_violation_arc_ap_sep:.4f}")
    print(f"    intra_arc_ml_sep: {cand.max_violation_intra_arc_ml_sep:.4f}")

    print("\nPer-probe x: seed → after polish (units: deg, deg, mm, mm, mm):")
    print(f"  arc AP angles seed : {[f'{a:+.2f}' for a in arc_centroids_deg]}")
    print(f"  arc AP angles final: {[f'{cand.x[i]:+.2f}' for i in range(n_arcs)]}")
    for p_idx, name in enumerate(probe_names):
        off = n_arcs + 5 * p_idx
        seed_vars = x0[off : off + 5]
        final_vars = cand.x[off : off + 5]
        print(
            f"  {name:>4}  arc={plan_state.probes[name].arc_id} "
            f"hole={probe_to_hole[name]:>2}\n"
            f"        seed : ml={seed_vars[0]:+6.2f}  spin={seed_vars[1]:+7.2f}  "
            f"off=({seed_vars[2]:+.3f},{seed_vars[3]:+.3f})  "
            f"depth={seed_vars[4]:+.3f}\n"
            f"        final: ml={final_vars[0]:+6.2f}  spin={final_vars[1]:+7.2f}  "
            f"off=({final_vars[2]:+.3f},{final_vars[3]:+.3f})  "
            f"depth={final_vars[4]:+.3f}"
        )

    # Apply the polished result to plan_state and write out an updated
    # config so the user can visualise it in trame.
    _apply_seed_polish_to_plan_state(plan_state, cand, sorted_letters, probe_names)
    from aind_low_point.runtime import save_plan_to_config as _save_plan_to_config

    new_cfg = _save_plan_to_config(plan_state, ConfigModel.from_yaml(args.config))
    out_path = args.output
    if out_path is None:
        out_path = args.config.with_stem(args.config.stem + "_seed_polished")
    with open(out_path, "w") as f:
        yaml.safe_dump(
            new_cfg.model_dump(mode="json"),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    print(f"\nWrote polished seed config to {out_path}")
    return 0


def _apply_seed_polish_to_plan_state(
    plan_state, cand: PlanCandidate, sorted_letters, probe_names
):
    """Map the polished ``cand.x`` back into ``plan_state``.

    Preserves the seed plan's arc letters (instead of renaming to a/b/c)
    so the output config keeps the user-recognisable arc names.
    """
    n_arcs = len(sorted_letters)
    arc_aps = cand.x[:n_arcs]
    plan_state.kinematics.arc_angles = {
        sorted_letters[i]: float(arc_aps[i]) for i in range(n_arcs)
    }
    for p_idx, name in enumerate(probe_names):
        off = n_arcs + 5 * p_idx
        ml, spin, off_R, off_A, depth = cand.x[off : off + 5]
        plan = plan_state.probes[name]
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
    p.add_argument(
        "--seed-plan",
        type=Path,
        default=None,
        help="If set, skip the LSAP + arc-partition layers and run the "
        "inner solve from this plan YAML as a warm start. The probe→hole "
        "assignment is auto-detected at the seed pose (per-probe best-fit "
        "by static threading max_g). Use to diagnose whether the optimizer "
        "is search-bound (manual seed stays put after polish) or polish-"
        "bound (manual seed drifts away).",
    )
    p.add_argument(
        "--seed-use-cma",
        action="store_true",
        help="When using --seed-plan, run CMA-ES around the seed before "
        "the SLSQP polish. Default is to skip CMA so the seed pose is "
        "preserved into the polish; flip on to test how a CMA restart "
        "interacts with a known-good warm start.",
    )
    p.add_argument(
        "--joint-rerank",
        action="store_true",
        help="Use the joint (H, A) reranking stage between the discrete "
        "layers and the full inner solve. Dispatches through "
        "``optimize_joint`` instead of ``optimize``.",
    )
    p.add_argument(
        "--k-holes-pool",
        type=int,
        default=50,
        help="(--joint-rerank only) Wide LSAP hole-assignment pool fed "
        "to the joint reranker. Default 50.",
    )
    p.add_argument(
        "--k-arcs-pool",
        type=int,
        default=20,
        help="(--joint-rerank only) Arc partitions per LSAP candidate "
        "fed to the joint reranker. Default 20.",
    )
    p.add_argument(
        "--k-joint",
        type=int,
        default=15,
        help="(--joint-rerank only) Number of joint candidates passed "
        "into the full inner solve after reranking. Default 15.",
    )
    p.add_argument(
        "--reduced-slsqp-max-iter",
        type=int,
        default=50,
        help="(--joint-rerank only) Max iterations for the reduced "
        "SLSQP scoring stage. Default 50.",
    )
    p.add_argument(
        "--save-alternatives",
        type=Path,
        default=None,
        help="Directory to write every alternative plan (lex-ranked) "
        "as its own full ConfigModel YAML. Default: write to "
        "``<output_stem>_alternatives/`` next to --output. The main "
        "--output still receives the lex-best plan. Also writes a "
        "`summary.md` with each plan's metrics. Pass "
        "--no-save-alternatives to disable.",
    )
    p.add_argument(
        "--no-save-alternatives",
        action="store_true",
        help="Disable writing alternative plans. By default the "
        "optimizer writes alternatives next to --output.",
    )
    p.add_argument(
        "--retro-density",
        action="store_true",
        help="Replace each probe's single-point target with the masked "
        "retro/labelled-cell cloud (brain ∩ structure mesh). Switches "
        "the optimizer's coverage density to a Gaussian mixture over "
        "those points. See --retro-* flags for tunables.",
    )
    p.add_argument(
        "--retro-asset",
        default="retro-targets",
        help="(--retro-density) Scene asset key for the cell-point cloud. "
        "Default 'retro-targets'.",
    )
    p.add_argument(
        "--retro-common-masks",
        nargs="*",
        default=["brain"],
        help="(--retro-density) Asset keys whose mesh.contains() AND-filters "
        "the cloud for every probe. Default ['brain'].",
    )
    p.add_argument(
        "--retro-per-probe-mask-fmt",
        default="structure:{probe}",
        help="(--retro-density) Per-probe extra mask key with {probe} "
        "substituted to the probe's name. Default 'structure:{probe}'.",
    )
    p.add_argument(
        "--retro-density-sigma-mm",
        type=float,
        default=0.3,
        help="(--retro-density) Gaussian mixture kernel bandwidth (mm). Default 0.3.",
    )
    p.add_argument(
        "--sdf-clearance",
        action="store_true",
        help="Use SDF-based pairwise probe clearance (JAX, analytic "
        "Jacobian) instead of FCL BVH + finite-diff. Faster + smooth "
        "gradients through overlap. Pre-builds a voxel SDF per probe "
        "kind (~5 s/kind, cached to ~/.cache/aind_low_point/sdfs/).",
    )
    p.add_argument(
        "--atlas-stage1",
        action="store_true",
        help="(--joint-rerank only) Replace Stage 1 LSAP with the target-"
        "aligned pose-feasibility atlas. Each (probe, hole) is gated by "
        "whether the probe can thread the hole AND reach its target. "
        "Cuts the H pool from ~1000 to ~50-100 for typical K=7 problems. "
        "See `dev/minlp_assignment_brief.md`.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="(--joint-rerank only) Number of parallel workers for the "
        "inner SLSQP stage. Default = os.cpu_count() - 2. Set to 1 for "
        "sequential debugging.",
    )
    p.add_argument(
        "--batched-stage2",
        action="store_true",
        help="(--joint-rerank only) Replace the sequential per-candidate "
        "scipy SLSQP polish with a JAX-batched Adam optimizer. Polishes "
        "ALL (HA, AA) candidates in parallel; trades absolute precision "
        "for throughput (~10-100× speedup at large pools). Stage 3 full "
        "SLSQP still runs on top-K. See dev/target_valid_atlas_design.md "
        "Phase 5.",
    )
    p.add_argument(
        "--batched-adam-steps",
        type=int,
        default=2000,
        help="(--batched-stage2 only) Adam iteration count per polish.",
    )
    p.add_argument(
        "--batched-adam-lr",
        type=float,
        default=0.05,
        help="(--batched-stage2 only) Adam learning rate.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="JAX device for the SDF clearance + reduced objective. "
        "'auto' picks GPU when --sdf-clearance is on (kernel work "
        "dominates and GPU is ~10x faster steady-state), else CPU. "
        "Has no effect when --sdf-clearance is off.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Print per-stage wall-time breakdown after the run.",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose log")
    args = p.parse_args()

    # JAX device selection must happen before any jax import.
    import os
    resolved_device = args.device
    if resolved_device == "auto":
        resolved_device = "gpu" if args.sdf_clearance else "cpu"
    if resolved_device == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
    print(f"JAX device: {resolved_device}")

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    plan_state = runtime.plan_state

    if args.seed_plan is not None:
        raw = yaml.safe_load(args.seed_plan.read_text())
        plan_model = PlanningModel.model_validate(raw)
        store = PlanStore(plan_state)
        apply_plan_model_to_state(plan_model, store)
        print(f"Applied seed plan from {args.seed_plan}")
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

    retro_opts: RetroDensityOpts | None = None
    if args.retro_density:
        retro_opts = RetroDensityOpts(
            retro_asset_key=args.retro_asset,
            common_mask_keys=tuple(args.retro_common_masks),
            per_probe_mask_fmt=args.retro_per_probe_mask_fmt,
            sigma_mm=args.retro_density_sigma_mm,
        )
        print(
            f"Retro-density mode: asset={retro_opts.retro_asset_key}, "
            f"common_masks={list(retro_opts.common_mask_keys)}, "
            f"per-probe={retro_opts.per_probe_mask_fmt}, "
            f"sigma={retro_opts.sigma_mm} mm"
        )
    probes = [
        _probe_static_info(plan_state, runtime, name, retro_opts=retro_opts)
        for name in plan_state.probes
    ]
    if retro_opts is not None:
        for p in probes:
            n = 0 if p.target_points is None else p.target_points.shape[0]
            print(
                f"  probe {p.name}: N={n} masked retro pts, "
                f"centroid={p.target_LPS.round(2).tolist()}"
            )

    stage_mults_str = args.cma_stage_multipliers.strip()
    if stage_mults_str:
        stage_mults = tuple(float(x) for x in stage_mults_str.split(","))
    else:
        stage_mults = ()

    # Subject-to-rig rotation from the planning state (built from config).
    subject_from_rig_rot, _ = plan_state.kinematics.subject_from_rig.rotate_translate
    subject_from_rig_rot = np.asarray(subject_from_rig_rot, dtype=np.float64)
    if np.allclose(subject_from_rig_rot, np.eye(3)):
        subject_from_rig_rot = None
    else:
        print("Using subject_from_rig rotation from config (non-identity head tilt).")

    if args.seed_plan is not None:
        return _run_seed_polish(
            probes,
            plan_state,
            holes,
            args=args,
            stage_mults=stage_mults,
            subject_from_rig_rot=subject_from_rig_rot,
        )

    if args.joint_rerank:
        # Detect the seed-equivalent (H, A) for diagnostic ranking when
        # the config has a baked-in plan; pass None otherwise.
        seed_to_hole: dict[str, int] | None = None
        seed_to_arc_idx: dict[str, int] | None = None
        try:
            tmp_seed_h: dict[str, int] = {}
            tmp_seed_a_letters: dict[str, float] = {}
            for ps in probes:
                plan_n = plan_state.probes[ps.name]
                if plan_n.arc_id is None:
                    raise RuntimeError("no arc_id on probe")
                ap_n = float(plan_state.kinematics.get_arc(plan_n.arc_id))
                hole_id, _ = best_fit_hole_id_at_pose(
                    ps,
                    holes,
                    ap_deg=ap_n,
                    ml_deg=float(plan_n.ml_local),
                    spin_deg=float(plan_n.spin),
                    off_R_mm=float(plan_n.offsets_RA[0]),
                    off_A_mm=float(plan_n.offsets_RA[1]),
                    past_target_mm=float(plan_n.past_target_mm),
                )
                tmp_seed_h[ps.name] = int(hole_id)
                tmp_seed_a_letters[plan_n.arc_id] = ap_n
            sorted_letters_seed = sorted(
                tmp_seed_a_letters, key=lambda k: tmp_seed_a_letters[k]
            )
            letter_to_idx_seed = {L: i for i, L in enumerate(sorted_letters_seed)}
            tmp_seed_a: dict[str, int] = {}
            for ps in probes:
                plan_n = plan_state.probes[ps.name]
                assert plan_n.arc_id is not None
                tmp_seed_a[ps.name] = letter_to_idx_seed[plan_n.arc_id]
            seed_to_hole = tmp_seed_h
            seed_to_arc_idx = tmp_seed_a
            if args.verbose:
                print(
                    f"[run_optimizer] detected seed (H, A) from baked plan: "
                    f"hole={seed_to_hole}, arc_idx={seed_to_arc_idx}"
                )
        except Exception as e:
            if args.verbose:
                print(f"[run_optimizer] no seed plan detected ({e})")

        print(
            f"Running joint-rerank optimizer ("
            f"max_num_arcs={args.max_num_arcs}, "
            f"k_holes_pool={args.k_holes_pool}, k_arcs_pool={args.k_arcs_pool}, "
            f"k_joint={args.k_joint})..."
        )
        import time as _time
        _t_opt_start = _time.perf_counter()
        result = optimize_joint(
            probes,
            holes,
            max_num_arcs=args.max_num_arcs,
            min_num_arcs=args.min_num_arcs,
            arc_count_penalty_deg2=args.arc_count_penalty_deg2,
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
            subject_from_rig_rot=subject_from_rig_rot,
            k_holes_pool=args.k_holes_pool,
            k_arcs_pool=args.k_arcs_pool,
            k_joint=args.k_joint,
            joint_weights=JointWeights(
                threading_oval_tolerance=args.threading_oval_tolerance,
                min_arc_ap_sep_deg=args.min_arc_ap_sep_deg,
            ),
            reduced_slsqp_max_iter=args.reduced_slsqp_max_iter,
            seed_to_hole=seed_to_hole,
            seed_to_arc_idx=seed_to_arc_idx,
            n_workers=args.n_workers,
            sdf_by_name=(
                _maybe_build_sdfs(probes, runtime) if args.sdf_clearance else None
            ),
            use_atlas_stage1=args.atlas_stage1,
            batched_stage2=args.batched_stage2,
            batched_adam_steps=args.batched_adam_steps,
            batched_adam_lr=args.batched_adam_lr,
            verbose=args.verbose,
        )
        _t_opt_end = _time.perf_counter()
        if args.profile:
            print(f"[profile] optimize_joint wall: {_t_opt_end - _t_opt_start:.2f}s")
            try:
                from aind_low_point.optimization.joint_rerank_jax import (
                    cache_stats as _jax_stats_2,
                )
                s2 = _jax_stats_2()
                print(
                    f"[profile] Stage2 JAX cache: {s2['entries']} entries, "
                    f"{s2['hits']} hits, {s2['misses']} misses"
                )
            except Exception:
                pass
            try:
                from aind_low_point.optimization.stage3_jax import (
                    cache_stats as _jax_stats_3,
                )
                s3 = _jax_stats_3()
                print(
                    f"[profile] Stage3 JAX cache: {s3['entries']} entries, "
                    f"{s3['hits']} hits, {s3['misses']} misses"
                )
            except Exception:
                pass
            try:
                from aind_low_point.optimization.joint_rerank import (
                    stage2_timings as _s2_t,
                )
                t = _s2_t()
                total = sum(t.values()) or 1.0
                print("[profile] Stage 2 component breakdown:")
                for k in sorted(t, key=lambda k: -t[k]):
                    print(f"  {k:<20} {t[k]:7.2f}s  ({100 * t[k] / total:4.1f}%)")
            except Exception:
                pass
    else:
        print(
            f"Running optimizer (max_num_arcs={args.max_num_arcs}, "
            f"k_holes={args.k_holes}, k_arcs={args.k_arcs})..."
        )
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
            subject_from_rig_rot=subject_from_rig_rot,
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

    # Resolve the main --output path first so we can derive a default
    # alternatives directory next to it.
    out_path = args.output
    if out_path is None:
        out_path = args.config.with_stem(args.config.stem + "_opt")

    # Save alternatives FIRST (each call mutates plan_state to that
    # candidate's pose); the subsequent _apply_result_to_plan_state
    # restores plan_state to the lex-best plan for the main --output.
    alts_target: Path | None = None
    if not args.no_save_alternatives and result.alternatives:
        alts_target = (
            args.save_alternatives
            if args.save_alternatives is not None
            else out_path.with_name(out_path.stem + "_alternatives")
        )
    if alts_target is not None:
        alts_dir = _save_alternatives(plan_state, cfg, result, alts_target)
        print(
            f"\nWrote {len(result.alternatives)} alternative plan(s) to "
            f"{alts_dir} (see summary.md)."
        )

    _apply_result_to_plan_state(plan_state, result)
    new_cfg = save_plan_to_config(plan_state, cfg)
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
