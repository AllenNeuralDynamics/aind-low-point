"""Diagnose the optimizer's outer + middle search layers against a seed plan.

For a config + plan pair, this script answers two questions:

  (a) Is the seed plan's probe→hole assignment enumerated by the LSAP +
      Murty top-K search? If not — what's its rank, and what does the
      LSAP cost breakdown say (target-angle vs static-threading vs
      coverage vs interference)?

  (b) Given the seed plan's hole assignment, is the seed's arc partition
      among the top-K enumerated by ``solve_top_k_arc_assignments``? If
      not — what's its rank, and what's the cost gap?

Usage::

    uv run --python 3.13 python scripts/diagnose_search.py \\
        examples/836656-config-T12.yml /tmp/836656-holes.yml \\
        --plan examples/836656-config-T12.plan.yml \\
        --k-holes 50 --k-arcs 50 --max-num-arcs 3 --min-num-arcs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aind_low_point.config import ConfigModel, PlanningModel  # noqa: E402
from aind_low_point.optimization.arc_assignment import (  # noqa: E402
    required_aps_deg_for_assignment,
    solve_top_k_arc_assignments,
)
from aind_low_point.optimization.hole_assignment import (  # noqa: E402
    AssignmentProbe,
    CostWeights,
    angle_to_target_rad,
    build_cost_matrix,
    multi_pose_evaluate,
    pairwise_interference_penalty,
    solve_top_k_assignments,
)
from aind_low_point.optimization.holes import load_holes  # noqa: E402
from aind_low_point.runtime import build_runtime_from_config  # noqa: E402
from aind_low_point.runtime.export import apply_plan_model_to_state  # noqa: E402
from aind_low_point.runtime.transforms import compile_all_transforms  # noqa: E402
from aind_low_point.state_change import PlanStore  # noqa: E402
from scripts.run_optimizer import _probe_static_info, _transform_holes  # noqa: E402


def _seed_hole_assignment(plan_state, probes, holes):
    """Detect seed probe→hole by static threading max_g (best-fit per probe)."""
    from aind_low_point.optimization.optimize import best_fit_hole_id_at_pose

    out: dict[str, int] = {}
    out_max_g: dict[str, float] = {}
    for ps in probes:
        plan = plan_state.probes[ps.name]
        assert plan.arc_id is not None
        ap = float(plan_state.kinematics.get_arc(plan.arc_id))
        hid, max_g = best_fit_hole_id_at_pose(
            ps,
            holes,
            ap_deg=ap,
            ml_deg=float(plan.ml_local),
            spin_deg=float(plan.spin),
            off_R_mm=float(plan.offsets_RA[0]),
            off_A_mm=float(plan.offsets_RA[1]),
            past_target_mm=float(plan.past_target_mm),
        )
        out[ps.name] = hid
        out_max_g[ps.name] = max_g
    return out, out_max_g


def _assignment_cost(
    cost_matrix: np.ndarray,
    probe_names: list[str],
    holes_id_to_idx: dict[int, int],
    probe_to_hole: dict[str, int],
) -> tuple[float, list[float]]:
    """Sum cost matrix entries for the (name → hole_id) assignment.

    Returns ``(total_cost, per_probe_costs)``.
    """
    per_probe: list[float] = []
    for i, name in enumerate(probe_names):
        hole_id = probe_to_hole[name]
        j = holes_id_to_idx[hole_id]
        per_probe.append(float(cost_matrix[i, j]))
    return float(sum(per_probe)), per_probe


def _partition_signature(probe_to_arc_idx: dict[str, int]) -> tuple[tuple[str, ...], ...]:
    """Canonical equivalence-class signature for a probe→arc partition.

    Two partitions are equivalent iff they have the same set of
    probe-groups (arc labels are interchangeable). Returns a tuple of
    sorted-name tuples, sorted lexicographically — so swapping arc 0
    with arc 1 produces the same signature.
    """
    groups: dict[int, list[str]] = {}
    for name, idx in probe_to_arc_idx.items():
        groups.setdefault(idx, []).append(name)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--k-holes", type=int, default=50)
    p.add_argument("--k-arcs", type=int, default=50)
    p.add_argument("--max-num-arcs", type=int, default=4)
    p.add_argument("--min-num-arcs", type=int, default=1)
    p.add_argument(
        "--min-arc-ap-sep-deg",
        type=float,
        default=16.0,
    )
    p.add_argument(
        "--arc-sep-shortfall-weight",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--arc-count-penalty-deg2",
        type=float,
        default=25.0,
    )
    args = p.parse_args()

    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    plan_state = runtime.plan_state
    raw = yaml.safe_load(args.plan.read_text())
    plan_model = PlanningModel.model_validate(raw)
    apply_plan_model_to_state(plan_model, PlanStore(plan_state))
    print(f"Applied plan from {args.plan}")

    holes = load_holes(args.holes)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        R, t = compiled["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    holes_id_to_idx = {h.id: i for i, h in enumerate(holes)}
    print(f"Loaded {len(holes)} holes")

    probe_names = list(plan_state.probes.keys())
    probes_static = [_probe_static_info(plan_state, runtime, n) for n in probe_names]

    # Seed hole assignment
    seed_to_hole, seed_max_g = _seed_hole_assignment(plan_state, probes_static, holes)
    print(f"\nSeed probe → hole (best-fit by max_g at seed pose):")
    for n in probe_names:
        print(
            f"  {n:>4}  hole={seed_to_hole[n]:>2}  max_g={seed_max_g[n]:+.4f}"
        )

    # =================================================================
    # (a) LSAP / Murty diagnosis
    # =================================================================
    assignment_probes = [
        AssignmentProbe(
            name=ps.name,
            target_LPS=np.asarray(ps.target_LPS, dtype=np.float64),
            shank_tips_local=np.asarray(ps.shank_tips_local, dtype=np.float64),
            kind=ps.kind,
            density_sigma_mm=ps.density_sigma_mm,
        )
        for ps in probes_static
    ]
    weights = CostWeights()
    cost_matrix = build_cost_matrix(assignment_probes, holes, weights=weights)

    # Per-component cost breakdown (re-evaluate the same primitives so
    # we can print α·angle, β·max_g, η·violation, γ·interference, −δ·cov
    # separately per (probe, hole)).
    K, N = cost_matrix.shape
    angle_mat = np.zeros((K, N))
    max_g_mat = np.zeros((K, N))
    violation_mat = np.zeros((K, N))
    coverage_mat = np.zeros((K, N))
    for i, ap in enumerate(assignment_probes):
        for j, h in enumerate(holes):
            angle_mat[i, j] = angle_to_target_rad(ap.target_LPS, h)
            score = multi_pose_evaluate(ap, h)
            max_g_mat[i, j] = score.min_max_g
            violation_mat[i, j] = score.min_violation_sq
            coverage_mat[i, j] = score.max_coverage
    interference_mat = pairwise_interference_penalty(assignment_probes, holes)

    seed_cost_total, _ = _assignment_cost(
        cost_matrix, probe_names, holes_id_to_idx, seed_to_hole
    )
    print(f"\n=== (a) LSAP / Murty diagnosis ===")
    print(f"Seed hole assignment total cost = {seed_cost_total:.4f}")
    print("Per-probe LSAP cost breakdown (seed assignment):")
    print(
        f"  {'probe':>4}  {'hole':>4}  {'α·angle':>10}  {'β·max_g':>10}  "
        f"{'η·viol²':>10}  {'γ·intf':>10}  {'−δ·cov':>10}  {'TOTAL':>10}  "
        f"{'reject?':>8}"
    )
    for i, name in enumerate(probe_names):
        j = holes_id_to_idx[seed_to_hole[name]]
        a = weights.alpha_target_angle * angle_mat[i, j]
        b = weights.beta_clearance * max_g_mat[i, j]
        e = weights.eta_violation * violation_mat[i, j]
        g = weights.gamma_interference * interference_mat[i, j]
        c = -weights.delta_coverage * coverage_mat[i, j]
        total = cost_matrix[i, j]
        reject = "YES" if violation_mat[i, j] > weights.violation_reject_threshold else ""
        print(
            f"  {name:>4}  {seed_to_hole[name]:>4}  {a:>+10.4f}  {b:>+10.4f}  "
            f"{e:>+10.4f}  {g:>+10.4f}  {c:>+10.4f}  {total:>+10.4f}  {reject:>8}"
        )

    # Top-K LSAP enumeration via Murty.
    print(
        f"\nEnumerating top-{args.k_holes} hole assignments via LSAP + Murty..."
    )
    top_k = solve_top_k_assignments(
        assignment_probes, holes, k=args.k_holes, weights=weights
    )
    print(f"  Got {len(top_k)} feasible assignments.")

    seed_canonical = {n: seed_to_hole[n] for n in probe_names}
    seed_rank = -1
    for rank, ha in enumerate(top_k, start=1):
        if dict(ha.probe_to_hole) == seed_canonical:
            seed_rank = rank
            break
    print(
        f"\nSeed hole assignment rank in top-{args.k_holes}: "
        f"{'**' + str(seed_rank) + '**' if seed_rank > 0 else 'NOT FOUND'}"
    )

    print("\nTop-10 LSAP assignments (sorted by total cost):")
    print(f"  {'rank':>4}  {'cost':>10}  {'matches seed?':>14}  probe→hole")
    for rank, ha in enumerate(top_k[:10], start=1):
        matches = sum(
            1 for n in probe_names if ha.probe_to_hole.get(n) == seed_to_hole[n]
        )
        match_str = f"{matches}/{len(probe_names)}"
        is_seed = " ←SEED" if dict(ha.probe_to_hole) == seed_canonical else ""
        mapping = ", ".join(
            f"{n}:{ha.probe_to_hole[n]}" for n in probe_names
        )
        print(
            f"  {rank:>4}  {ha.cost:>10.4f}  {match_str:>14}  "
            f"{mapping}{is_seed}"
        )
    if seed_rank > 10 and seed_rank > 0:
        idx = seed_rank - 1
        ha = top_k[idx]
        matches = sum(
            1 for n in probe_names if ha.probe_to_hole.get(n) == seed_to_hole[n]
        )
        mapping = ", ".join(f"{n}:{ha.probe_to_hole[n]}" for n in probe_names)
        print(
            f"  {seed_rank:>4}  {ha.cost:>10.4f}  {matches}/{len(probe_names):>4}  "
            f"{mapping}  ←SEED"
        )

    # =================================================================
    # (b) Arc partition diagnosis
    # =================================================================
    print(f"\n=== (b) Arc partition diagnosis (using seed hole assignment) ===")
    aps_dict = required_aps_deg_for_assignment(seed_to_hole, holes)
    print("Required AP per probe (from hole axis):")
    for n in probe_names:
        plan = plan_state.probes[n]
        assert plan.arc_id is not None
        seed_ap = float(plan_state.kinematics.get_arc(plan.arc_id))
        print(
            f"  {n:>4}  hole={seed_to_hole[n]:>2}  required_AP={aps_dict[n]:+6.2f}°  "
            f"seed_arc={plan.arc_id}  seed_arc_AP={seed_ap:+6.2f}°  "
            f"tilt={seed_ap - aps_dict[n]:+6.2f}°"
        )

    # Build the seed's arc partition (probe→arc_idx with arc-idx
    # assigned by ascending AP across the seed's used arc letters).
    arc_letters_used: dict[str, float] = {}
    for n in probe_names:
        plan = plan_state.probes[n]
        assert plan.arc_id is not None
        letter: str = plan.arc_id
        arc_letters_used[letter] = float(plan_state.kinematics.get_arc(letter))
    sorted_letters = sorted(arc_letters_used, key=lambda k: arc_letters_used[k])
    letter_to_idx = {L: i for i, L in enumerate(sorted_letters)}
    seed_arc_idx: dict[str, int] = {}
    for n in probe_names:
        plan_n = plan_state.probes[n]
        assert plan_n.arc_id is not None
        seed_arc_idx[n] = letter_to_idx[plan_n.arc_id]
    seed_centroids = tuple(arc_letters_used[L] for L in sorted_letters)
    seed_sig = _partition_signature(seed_arc_idx)
    print(
        f"\nSeed arc partition: probe→arc_idx = {seed_arc_idx}; "
        f"AP centroids = {[f'{c:+.2f}' for c in seed_centroids]}"
    )

    arc_top_k = solve_top_k_arc_assignments(
        seed_to_hole,
        holes,
        max_num_arcs=args.max_num_arcs,
        min_num_arcs=args.min_num_arcs,
        k=args.k_arcs,
        min_arc_ap_sep_deg=args.min_arc_ap_sep_deg,
        arc_sep_shortfall_weight=args.arc_sep_shortfall_weight,
        arc_count_penalty_deg2=args.arc_count_penalty_deg2,
    )
    print(
        f"\nEnumerated {len(arc_top_k)} arc partitions for the seed hole "
        f"assignment (top-{args.k_arcs}, max_num_arcs={args.max_num_arcs}, "
        f"min_num_arcs={args.min_num_arcs})."
    )

    seed_rank_arc = -1
    for rank, aa in enumerate(arc_top_k, start=1):
        if _partition_signature(dict(aa.probe_to_arc_idx)) == seed_sig:
            seed_rank_arc = rank
            break

    print(
        f"\nSeed arc partition rank in top-{args.k_arcs}: "
        f"{'**' + str(seed_rank_arc) + '**' if seed_rank_arc > 0 else 'NOT FOUND'}"
    )

    print("\nTop-10 arc partitions (sorted by partitioner cost):")
    print(
        f"  {'rank':>4}  {'cost':>10}  {'centroids':>30}  {'n_arcs':>6}  partition"
    )
    for rank, aa in enumerate(arc_top_k[:10], start=1):
        cents = ", ".join(f"{c:+.1f}" for c in aa.arc_centroids_deg)
        is_seed = " ←SEED" if _partition_signature(dict(aa.probe_to_arc_idx)) == seed_sig else ""
        mapping = ", ".join(f"{n}:{aa.probe_to_arc_idx.get(n, '-')}" for n in probe_names)
        print(
            f"  {rank:>4}  {aa.cost:>10.4f}  {cents:>30}  {len(aa.arc_centroids_deg):>6}  "
            f"{mapping}{is_seed}"
        )
    if seed_rank_arc > 10 and seed_rank_arc > 0:
        idx = seed_rank_arc - 1
        aa = arc_top_k[idx]
        cents = ", ".join(f"{c:+.1f}" for c in aa.arc_centroids_deg)
        mapping = ", ".join(f"{n}:{aa.probe_to_arc_idx.get(n, '-')}" for n in probe_names)
        print(
            f"  {seed_rank_arc:>4}  {aa.cost:>10.4f}  {cents:>30}  "
            f"{len(aa.arc_centroids_deg):>6}  {mapping}  ←SEED"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
