"""Score a manually-authored plan against the optimizer's constraint model.

Loads a config YAML + matching plan YAML, threads them through the same
``OptimizerContext`` the three-level driver uses, and prints the same
``PlanCandidate`` metrics (max-violation per group, min clearances /
separations, coverage, lex_key at various ε) that the optimizer reports.

Auto-detects each probe's bore at the manual pose by picking the hole
whose threading ``max(g_thread)`` over the manual pose's shanks is
smallest — i.e. the bore the manual plan implies the probe is using.

Usage::

    uv run --python 3.13 python scripts/score_manual_plan.py \\
        examples/836656-config-T12.yml scratch/0283-300-04.holes.yml \\
        --plan examples/836656-config-T12.plan.yml \\
        --threading-oval-tolerance 3.0 \\
        --clearance-overlap-allowance-mm 1.5 \\
        --thresholds 0,1,2,3

Pass ``--plan`` to use a sidecar plan YAML; without it the plan baked
into the config is scored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Make sibling scripts importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aind_low_point.config import ConfigModel, PlanningModel
from aind_low_point.optimization.arc_assignment import ArcAssignment
from aind_low_point.optimization.density import gaussian_density
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.kinematics import (
    pose_from_optimizer_vars,
    shank_capsules_from_pose,
)
from aind_low_point.optimization.objective import (
    OptimizerContext,
    ProbeContext,
    VariableLayout,
    evaluate_constraints,
    evaluate_objective,
    feasibility_violation_squared,
    shaft_section_oval_value,
)
from aind_low_point.optimization.optimize import (
    _build_plan_candidate as build_plan_candidate,
)
from aind_low_point.optimization.recording import (
    RECORDING_GEOMETRY,
    RecordingGeometry,
    get_recording_geometry,
    recording_center_local_for_kind,
)
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.export import apply_plan_model_to_state
from aind_low_point.runtime.transforms import compile_all_transforms
from aind_low_point.state_change import PlanStore
from scripts.run_optimizer import _probe_static_info, _transform_holes


def _best_fit_hole_id(
    probe_ctx_base: ProbeContext,
    holes,
    ml_deg: float,
    spin_deg: float,
    off_R_mm: float,
    off_A_mm: float,
    past_target_mm: float,
    ap_deg: float,
    shaft_length_mm: float,
    shank_radius_mm: float,
) -> tuple[int, float]:
    """Pick the hole whose max(g_thread) at the manual pose is smallest.

    Returns ``(hole_id, max_g)``.
    """
    tips = np.asarray(probe_ctx_base.shank_tips_local, dtype=np.float64)
    if tips.shape[0] > 0:
        pivot_local = np.array(
            [
                float(tips[:, 0].mean()),
                float(tips[:, 1].mean()),
                float(probe_ctx_base.recording_geom.active_center_mm),
            ],
            dtype=np.float64,
        )
    else:
        pivot_local = recording_center_local_for_kind(probe_ctx_base.kind)
    R, pose_tip = pose_from_optimizer_vars(
        target_LPS=probe_ctx_base.target_LPS,
        ap_deg=ap_deg,
        ml_deg=ml_deg,
        spin_deg=spin_deg,
        offset_R_mm=off_R_mm,
        offset_A_mm=off_A_mm,
        past_target_mm=past_target_mm,
        recording_center_local=pivot_local,
    )
    shanks = shank_capsules_from_pose(
        R,
        pose_tip,
        probe_ctx_base.shank_tips_local,
        shaft_length_mm=shaft_length_mm,
        shank_radius_mm=shank_radius_mm,
    )
    best_id = -1
    best_max_g = float("inf")
    for h in holes:
        max_g = max(
            shaft_section_oval_value(sh, sec) for sh in shanks for sec in h.sections
        )
        if max_g < best_max_g:
            best_max_g = max_g
            best_id = int(h.id)
    return best_id, float(best_max_g)


def main():  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path, help="Path to input config YAML")
    p.add_argument("holes", type=Path, help="Path to holes YAML")
    p.add_argument("--plan", type=Path, default=None, help="Optional sidecar plan YAML")
    p.add_argument(
        "--threading-oval-tolerance",
        type=float,
        default=0.0,
        help="Threading slack tolerance (g_thread <= tol).",
    )
    p.add_argument(
        "--clearance-overlap-allowance-mm",
        type=float,
        default=0.0,
        help="Headstage-headstage overlap allowance (mm).",
    )
    p.add_argument(
        "--min-arc-ap-sep-deg",
        type=float,
        default=16.0,
        help="Rig min AP separation between arcs (deg).",
    )
    p.add_argument(
        "--min-within-arc-ml-sep-deg",
        type=float,
        default=16.0,
        help="Rig min within-arc ML separation (deg).",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default="0,0.5,1,2,3",
        help="Comma-separated feasibility ε values for lex_key reporting.",
    )
    p.add_argument(
        "--fixed-holes",
        type=str,
        default=None,
        help="Override auto-detected probe→hole as 'NAME:ID,NAME:ID,...'. "
        "Use to score against a known assignment (e.g. the one documented "
        "in the YAML's comments).",
    )
    args = p.parse_args()

    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    plan_state = runtime.plan_state

    if args.plan is not None:
        raw = yaml.safe_load(args.plan.read_text())
        plan_model = PlanningModel.model_validate(raw)
        store = PlanStore(plan_state)
        apply_plan_model_to_state(plan_model, store)
        print(f"Applied plan overlay from {args.plan}")

    holes = load_holes(args.holes)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R_imp, t_imp = T.rotate_translate
        holes = _transform_holes(holes, R_imp, t_imp)
        print(f"Applied implant_to_lps to {len(holes)} hole(s)")

    fixed_holes: dict[str, int] = {}
    if args.fixed_holes:
        for item in args.fixed_holes.split(","):
            name, hole_id = item.split(":")
            fixed_holes[name.strip()] = int(hole_id)

    probe_names = list(plan_state.probes.keys())

    # Determine arc_id → arc_idx mapping by sorting arc letters by their
    # AP angle ascending. This matches the optimizer's `arc_0..arc_{n-1}`
    # convention (ordered by AP centroid).
    arc_letters_used: dict[str, float] = {}
    for name in probe_names:
        plan = plan_state.probes[name]
        if plan.arc_id is None:
            raise RuntimeError(f"Probe {name}: arc_id is None in the manual plan")
        arc_letter: str = plan.arc_id
        ap = plan_state.kinematics.get_arc(arc_letter)
        arc_letters_used[arc_letter] = float(ap)
    sorted_letters = sorted(arc_letters_used, key=lambda k: arc_letters_used[k])
    letter_to_idx = {letter: i for i, letter in enumerate(sorted_letters)}
    arc_centroids_deg = tuple(arc_letters_used[letter] for letter in sorted_letters)
    n_arcs = len(sorted_letters)
    arc_ids_opt = tuple(f"arc_{i}" for i in range(n_arcs))
    print(
        f"Arcs (sorted ascending AP): "
        f"{[(letter, arc_letters_used[letter]) for letter in sorted_letters]}"
    )

    # Build a placeholder ProbeContext per probe so we can auto-detect
    # the assigned hole at the manual pose. Hole gets re-bound to the
    # detected one in the second pass.
    fallback_geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))
    probe_static_list = []
    for name in probe_names:
        probe_static_list.append(_probe_static_info(plan_state, runtime, name))

    # Stage 1: detect or honour-fixed hole per probe.
    chosen_hole_ids: dict[str, int] = {}
    detected_max_g: dict[str, float] = {}
    sl_default = 10.0
    sr_default = 0.05
    for ps in probe_static_list:
        name = ps.name
        plan = plan_state.probes[name]
        # Build a minimal ProbeContext stub for the geometry helpers.
        if ps.kind in RECORDING_GEOMETRY:
            geom = get_recording_geometry(ps.kind)
        else:
            geom = fallback_geom
        ctx_stub = ProbeContext(
            name=name,
            target_LPS=np.asarray(ps.target_LPS, dtype=np.float64),
            kind=ps.kind,
            arc_id="arc_0",  # unused for the geometry test
            shank_tips_local=np.asarray(ps.shank_tips_local, dtype=np.float64),
            assigned_hole=holes[0],  # placeholder
            density_fn=gaussian_density(ps.target_LPS, ps.density_sigma_mm),
            recording_geom=geom,
        )
        ap = float(plan_state.kinematics.get_arc(plan.arc_id))
        ml = float(plan.ml_local)
        spin = float(plan.spin)
        off_R, off_A = (float(plan.offsets_RA[0]), float(plan.offsets_RA[1]))
        depth = float(plan.past_target_mm)
        if name in fixed_holes:
            chosen = fixed_holes[name]
            h = next(h for h in holes if h.id == chosen)
            tips_local = ctx_stub.shank_tips_local
            tips = np.asarray(tips_local, dtype=np.float64)
            if tips.shape[0] > 0:
                pivot_local = np.array(
                    [
                        float(tips[:, 0].mean()),
                        float(tips[:, 1].mean()),
                        float(ctx_stub.recording_geom.active_center_mm),
                    ],
                    dtype=np.float64,
                )
            else:
                pivot_local = recording_center_local_for_kind(ctx_stub.kind)
            R_pose, pose_tip = pose_from_optimizer_vars(
                target_LPS=ctx_stub.target_LPS,
                ap_deg=ap,
                ml_deg=ml,
                spin_deg=spin,
                offset_R_mm=off_R,
                offset_A_mm=off_A,
                past_target_mm=depth,
                recording_center_local=pivot_local,
            )
            shanks = shank_capsules_from_pose(
                R_pose,
                pose_tip,
                ctx_stub.shank_tips_local,
                shaft_length_mm=sl_default,
                shank_radius_mm=sr_default,
            )
            max_g = max(
                shaft_section_oval_value(sh, sec) for sh in shanks for sec in h.sections
            )
            chosen_hole_ids[name] = chosen
            detected_max_g[name] = float(max_g)
        else:
            hole_id, max_g = _best_fit_hole_id(
                ctx_stub,
                holes,
                ml_deg=ml,
                spin_deg=spin,
                off_R_mm=off_R,
                off_A_mm=off_A,
                past_target_mm=depth,
                ap_deg=ap,
                shaft_length_mm=sl_default,
                shank_radius_mm=sr_default,
            )
            chosen_hole_ids[name] = hole_id
            detected_max_g[name] = max_g

    print("\nProbe → hole (auto-detected best-fit, max_g at manual pose):")
    for name in probe_names:
        plan = plan_state.probes[name]
        print(
            f"  {name:>4}  kind={plan.kind:<18}  arc={plan.arc_id}  "
            f"hole={chosen_hole_ids[name]:>2}  max_g={detected_max_g[name]:+.4f}"
        )

    # Stage 2: build the OptimizerContext with the chosen holes and
    # construct x from the plan.
    layout = VariableLayout(arc_ids=arc_ids_opt, probe_names=tuple(probe_names))
    holes_by_id = {h.id: h for h in holes}
    probe_contexts: list[ProbeContext] = []
    probe_to_arc_idx: dict[str, int] = {}
    headstage_objs: list[object] = []  # fcl.CollisionObject | None
    from aind_low_point.optimization.headstages import (
        make_fcl_bvh,
        make_fcl_convex,
    )

    for ps in probe_static_list:
        name = ps.name
        plan = plan_state.probes[name]
        assert plan.arc_id is not None
        arc_idx = letter_to_idx[plan.arc_id]
        probe_to_arc_idx[name] = arc_idx
        if ps.kind in RECORDING_GEOMETRY:
            geom = get_recording_geometry(ps.kind)
        else:
            geom = fallback_geom
        probe_contexts.append(
            ProbeContext(
                name=name,
                target_LPS=np.asarray(ps.target_LPS, dtype=np.float64),
                kind=ps.kind,
                arc_id=f"arc_{arc_idx}",
                shank_tips_local=np.asarray(ps.shank_tips_local, dtype=np.float64),
                assigned_hole=holes_by_id[chosen_hole_ids[name]],
                density_fn=gaussian_density(ps.target_LPS, ps.density_sigma_mm),
                recording_geom=geom,
            )
        )
        hh = getattr(ps, "headstage_hull", None)
        if hh is not None:
            headstage_objs.append(make_fcl_convex(hh))
        elif ps.collision_mesh is not None:
            # Match optimize.py: use full-mesh BVH when no headstage hull
            # is set, so we get the exact pairwise clearance the SLSQP
            # path is constrained against.
            headstage_objs.append(make_fcl_bvh(ps.collision_mesh))
        else:
            headstage_objs.append(None)
    ctx = OptimizerContext(
        layout=layout,
        probes=tuple(probe_contexts),
        threading_oval_tolerance=args.threading_oval_tolerance,
        clearance_overlap_allowance_mm=args.clearance_overlap_allowance_mm,
        min_arc_ap_sep_deg=args.min_arc_ap_sep_deg,
        min_within_arc_ml_sep_deg=args.min_within_arc_ml_sep_deg,
        headstage_fcl_objs=tuple(headstage_objs),
    )

    x = np.zeros(layout.n_vars, dtype=np.float64)
    for letter, idx in letter_to_idx.items():
        x[idx] = arc_letters_used[letter]
    for p_idx, name in enumerate(probe_names):
        plan = plan_state.probes[name]
        off = layout.num_arcs + 5 * p_idx
        x[off + 0] = float(plan.ml_local)
        x[off + 1] = float(plan.spin)
        x[off + 2] = float(plan.offsets_RA[0])
        x[off + 3] = float(plan.offsets_RA[1])
        x[off + 4] = float(plan.past_target_mm)

    cv = evaluate_constraints(x, ctx)
    breakdown = evaluate_objective(x, ctx)
    viol_sq = feasibility_violation_squared(x, ctx)

    # Build a PlanCandidate via the optimizer's own helper so the
    # reported numbers exactly match what the optimizer would print.
    ha = HoleAssignment(probe_to_hole=dict(chosen_hole_ids), cost=0.0)
    aa = ArcAssignment(
        probe_to_arc_idx=dict(probe_to_arc_idx),
        arc_centroids_deg=arc_centroids_deg,
        cost=0.0,
    )
    cand = build_plan_candidate(
        x,
        ctx,
        breakdown,
        ha=ha,
        aa=aa,
        n_arcs=n_arcs,
    )

    import itertools as _it

    print("\nPer-pair clearance (signed mm, negative = penetration):")
    pair_idx = list(_it.combinations(range(len(probe_names)), 2))
    for (i, j), c in zip(pair_idx, np.asarray(cv.clearance, dtype=np.float64)):
        marker = " ✗" if c < 0 else ""
        print(f"  {probe_names[i]:>5} – {probe_names[j]:<5}  clear={c:+7.3f}{marker}")

    print("\nConstraint slack summary (slack >= 0 ⇒ feasible):")
    for name, arr in (
        ("threading", cv.threading),
        ("clearance", cv.clearance),
        ("arc_ap_separation", cv.arc_ap_separation),
        ("intra_arc_ml_separation", cv.intra_arc_ml_separation),
    ):
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            print(f"  {name:<24}  (empty)")
            continue
        viol = float(np.maximum(0.0, -a).max())
        print(
            f"  {name:<24}  n={a.size:>3}  min={float(a.min()):+.4f}  "
            f"max={float(a.max()):+.4f}  max_violation={viol:+.4f}"
        )

    print("\nPlan candidate metrics:")
    print(f"  feasible (strict)            : {cand.feasible}")
    print(f"  max_violation (any group)    : {cand.max_violation:.4f}")
    print(f"  sum_violation_sq (= viol²)   : {cand.sum_violation_sq:.4f}")
    print(f"  feasibility_violation²       : {viol_sq:.4f}")
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

    print("\nObjective breakdown (with default weights):")
    print(f"  inner cost (J)        : {breakdown.total:.3f}")
    print(f"  coverage_total        : {breakdown.coverage_total:.3f}")
    print(f"  threading_penalty     : {breakdown.threading_penalty:.3f}")
    print(f"  clearance_penalty     : {breakdown.clearance_penalty:.3f}")
    print(f"  kinematic_penalty     : {breakdown.kinematic_penalty:.3f}")

    print("\nLex-key under varying feasibility thresholds:")
    eps_list = [float(s) for s in args.thresholds.split(",")]
    for eps in eps_list:
        eff_viol, sum_viol_sq, neg_cov = cand.lex_key(eps)
        print(
            f"  ε={eps:>4.2f}  →  ({eff_viol:.4f}, {sum_viol_sq:.4f}, "
            f"-cov={neg_cov:.4f})"
        )

    print("\nPer-probe variable vector echo (units: deg, deg, mm, mm, mm):")
    print(f"  arc AP angles: {[f'{a:+.1f}' for a in arc_centroids_deg]}")
    for p_idx, name in enumerate(probe_names):
        off = layout.num_arcs + 5 * p_idx
        ml, spin, oR, oA, depth = x[off : off + 5]
        plan = plan_state.probes[name]
        ev = breakdown.per_probe_evals[p_idx]
        print(
            f"  {name:>4}  arc={plan.arc_id} hole={chosen_hole_ids[name]:>2}  "
            f"ml={ml:+6.2f}  spin={spin:+7.2f}  "
            f"off=({oR:+.3f},{oA:+.3f})  depth={depth:+.3f}  "
            f"cov={ev.coverage:.3f}  "
            f"max_g={float(ev.threading_gs.max()):+.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
