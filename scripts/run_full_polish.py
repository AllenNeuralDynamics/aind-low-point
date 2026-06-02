"""Full Stage 2 polish on the deduped arc-first pool, with adaptive
retry for boundary candidates.

End-to-end pipeline:
  1. Build visibility atlas
  2. Enumerate arc-first principled candidates (~8k for 836656/T12)
  3. Build SDFs + pose_features
  4. Run polish_all_adaptive:
     - phase 1: single midpoint-seed polish via batched spin restore +
                multiproc SLSQP
     - phase 2: identify boundary candidates (0.05 ≤ max_viol ≤ 2.0)
     - phase 3: polish them with 2 extra AP-quartile seeds
     - phase 4: take lex-best per candidate across initial + retries
  5. Save results to disk; report top-K, manual rank, feasibility stats

Expected wall time on 836656/T12: ~20-25 min on 8 cores.

Run::

    uv run --python 3.13 python -m scripts.run_full_polish \\
        examples/836656-config-T12.yml scratch/0283-300-04.holes.yml \\
        --output /tmp/full_polish_T12.pkl
"""

from __future__ import annotations

import argparse
import os as _os
import pickle
import time
from dataclasses import replace
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.arc_first_principled import (
    enumerate_arc_first_candidates,
    find_target_in_candidates,
)
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import JointWeights
from aind_low_point.optimization.parallel_stage2 import polish_all_adaptive
from aind_low_point.optimization.pose_features import precompute_pose_features
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.visibility_atlas import build_visibility_atlas
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes


MANUAL_H_836656_T12 = {
    "MD": 3, "BLA": 4, "PL": 1, "VM": 7, "RSP": 5, "CA1": 10, "CLA": 12
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--output", type=Path, default=Path("/tmp/full_polish.pkl"))
    p.add_argument("--n-top", type=int, default=128)
    p.add_argument("--n-spin", type=int, default=72)
    p.add_argument("--per-arc-cap", type=int, default=50)
    p.add_argument("--global-cap", type=int, default=200_000)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--reduced-slsqp-max-iter", type=int, default=50)
    p.add_argument("--boundary-lo", type=float, default=0.05)
    p.add_argument("--boundary-hi", type=float, default=2.0)
    p.add_argument("--no-adaptive", action="store_true",
                   help="Skip the adaptive retry phase; single-seed polish only")
    p.add_argument("--n-cands", type=int, default=None,
                   help="Limit to the first N candidates after enumeration; "
                        "for small probes/smoke tests")
    p.add_argument("--spin-restore-chunk", type=str, default="auto",
                   help="Batched-spin-restore chunk size (GPU memory knob). "
                        "Default 'auto' queries free VRAM and picks a safe "
                        "value. Pass an int to override.")
    p.add_argument("--vram-safety-margin", type=float, default=0.2,
                   help="Auto-chunk reserves this fraction of TOTAL VRAM as "
                        "headroom (default 0.2 = 20%%). Only used when "
                        "--spin-restore-chunk=auto.")
    p.add_argument("--skip-augment", action="store_true",
                   help="Skip the post-polish offset-augment + violation-eval "
                        "steps. By default these run and produce a pkl with "
                        "``augmented_phase1_x`` + ``violation_fn`` that Stage "
                        "3 chains can consume directly.")
    args = p.parse_args()

    t_total0 = time.perf_counter()

    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, name)
        for name in runtime.plan_state.probes
    ]
    holes_list = load_holes(args.holes)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes_list = _transform_holes(holes_list, R, t)

    print("Building visibility atlas...", flush=True)
    t0 = time.perf_counter()
    atlas = build_visibility_atlas(
        probes, holes_list, n_top=args.n_top, n_spin=args.n_spin, verbose=False,
    )
    print(f"  atlas: {time.perf_counter() - t0:.1f}s", flush=True)

    print("Enumerating arc-first candidates...", flush=True)
    t0 = time.perf_counter()
    candidates = enumerate_arc_first_candidates(
        probes, atlas, max_arcs=3, max_probes_per_arc=4,
        per_arc_max_hole_tuples=args.per_arc_cap,
        global_max_candidates=args.global_cap,
        verbose=False,
    )
    print(f"  enumeration: {time.perf_counter() - t0:.1f}s "
          f"({len(candidates)} candidates)", flush=True)

    manual_rank = find_target_in_candidates(candidates, MANUAL_H_836656_T12)
    print(f"  manual rank in deduped pool: {manual_rank}", flush=True)

    if args.n_cands is not None and args.n_cands < len(candidates):
        # Keep manual rank in scope when probing
        keep_idxs = list(range(args.n_cands))
        if manual_rank is not None and manual_rank >= args.n_cands:
            keep_idxs[-1] = manual_rank
            print(f"  swapped slot {args.n_cands - 1} for manual (rank "
                  f"{manual_rank}) so the probe still includes it")
        candidates = [candidates[i] for i in keep_idxs]
        manual_rank = next(
            (i for i, c in enumerate(candidates)
             if MANUAL_H_836656_T12 == {p: c.ha.probe_to_hole[p]
                                         for p in MANUAL_H_836656_T12}),
            None,
        )
        print(f"  limited to {len(candidates)} candidates for probe run "
              f"(manual is rank {manual_rank} in probe)", flush=True)

    print("Building SDFs + pose features...", flush=True)
    t0 = time.perf_counter()
    sdf_by_name = {p.name: build_probe_sdf_from_alpha_wrap(
        runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
    ) for p in probes}
    pose_features = precompute_pose_features(probes, holes_list)
    print(f"  SDFs + pose features: {time.perf_counter() - t0:.1f}s",
          flush=True)

    weights = replace(JointWeights(), min_arc_ap_sep_deg=16.0)

    if args.spin_restore_chunk == "auto":
        from aind_low_point.optimization.parallel_stage2 import (
            estimate_spin_restore_chunk,
        )
        chunk = estimate_spin_restore_chunk(
            n_surf=int(sdf_by_name[probes[0].name].surface_points.shape[0]),
            K=len(probes), total_B=len(candidates),
            safety_margin=args.vram_safety_margin,
        )
        print(f"  spin_restore_chunk=auto → {chunk} "
              f"(from free VRAM probe)")
    else:
        chunk = int(args.spin_restore_chunk)

    print()
    print("=" * 70)
    print("Full polish (adaptive)")
    print("=" * 70)
    t0 = time.perf_counter()
    if args.no_adaptive:
        from aind_low_point.optimization.parallel_stage2 import (
            polish_all_with_batched_spin_restore,
        )
        results = polish_all_with_batched_spin_restore(
            candidates, probes, holes_list, pose_features,
            weights=weights,
            reduced_slsqp_max_iter=args.reduced_slsqp_max_iter,
            sdf_by_name=sdf_by_name,
            n_workers=args.n_workers,
            spin_restore_chunk=chunk,
            n_arcs=3,
            verbose=True,
        )
    else:
        results = polish_all_adaptive(
            candidates, probes, holes_list, pose_features, atlas,
            weights=weights,
            reduced_slsqp_max_iter=args.reduced_slsqp_max_iter,
            sdf_by_name=sdf_by_name,
            n_workers=args.n_workers,
            boundary_lo=args.boundary_lo,
            boundary_hi=args.boundary_hi,
            spin_restore_chunk=chunk,
            n_arcs=3,
            verbose=True,
        )
    print(f"  polish wall: {time.perf_counter() - t0:.1f}s", flush=True)

    # ------- Save results -------
    print()
    print(f"Saving results to {args.output}...", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "candidates": candidates,
            "results": results,
            "manual_rank": manual_rank,
            "n_workers": args.n_workers,
        }, f)
    print(f"  saved ({args.output.stat().st_size / 1e6:.1f} MB)")

    # ------- Summary -------
    max_viols = np.array([float(r.metrics.max_violation) for r in results])
    coverages = np.array([float(r.metrics.approximate_coverage) for r in results])
    n_feasible = int(np.sum(max_viols <= 1e-3))
    n_near_feasible = int(np.sum(max_viols <= 0.5))
    n_polish_succeeded = int(np.sum(max_viols < 100))

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  total candidates polished: {len(results)}")
    print(f"  strictly feasible (max_viol ≤ 0.001): {n_feasible}")
    print(f"  near-feasible    (max_viol ≤ 0.5):   {n_near_feasible}")
    print(f"  polish succeeded (max_viol < 100):   {n_polish_succeeded}")
    print(f"  max_viol  min={max_viols.min():.4f} "
          f"median={float(np.median(max_viols)):.4f} "
          f"max={max_viols.max():.4f}")

    # Sort by lex_key, report top-10
    sorted_results = sorted(
        enumerate(results), key=lambda kv: kv[1].metrics.lex_key()
    )
    print()
    print("Top-10 by lex_key (max_viol, sum_viol², -coverage):")
    print(f"  {'rank':>4} {'cand#':>6} {'max_viol':>10} {'sum_viol²':>12} "
          f"{'coverage':>10}")
    for rank, (cand_idx, jc) in enumerate(sorted_results[:10]):
        m = jc.metrics
        marker = ""
        if manual_rank is not None and cand_idx == manual_rank:
            marker = "  [MANUAL]"
        print(f"  {rank:>4d} {cand_idx:>6d} {m.max_violation:>10.4f} "
              f"{m.sum_violation_sq:>12.4f} {m.approximate_coverage:>10.4f}"
              f"{marker}")

    if manual_rank is not None:
        manual_jc = results[manual_rank]
        manual_rank_post = next(
            (rank for rank, (idx, _) in enumerate(sorted_results)
             if idx == manual_rank), None
        )
        print()
        print(f"Manual (#{manual_rank}): "
              f"max_viol={manual_jc.metrics.max_violation:.4f} "
              f"sum_viol²={manual_jc.metrics.sum_violation_sq:.4f} "
              f"coverage={manual_jc.metrics.approximate_coverage:.4f} "
              f"→ post-polish lex-rank #{manual_rank_post}")

    # ------- Augment + violation eval (production Stage 2 → Stage 3 bridge) -------
    if not args.skip_augment:
        import subprocess
        import sys
        print()
        print("=" * 70)
        print("Augmenting with offset polish + violation eval")
        print("=" * 70)
        subprocess.run([
            sys.executable, "-u", "-m", "scripts.augment_polish_with_offsets",
            str(args.config), str(args.holes),
            "--in-pkl", str(args.output),
            "--out-pkl", str(args.output),
            "--n-workers", str(args.n_workers),
        ], check=True)
        subprocess.run([
            sys.executable, "-u", "-m", "scripts.eval_violation_at_augmented",
            str(args.config), str(args.holes),
            "--in-pkl", str(args.output),
            "--n-workers", str(args.n_workers),
        ], check=True)

    print()
    print(f"Total wall: {time.perf_counter() - t_total0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
