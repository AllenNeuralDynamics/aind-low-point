"""Profile a polish run on a small batch to find optimization targets.

Builds atlas, enumerates candidates, samples N=20 (default), and runs
the full polish pipeline in single-worker mode so the global
``_STAGE2_TIMINGS`` accumulator captures per-component breakdown.

Outputs:
  - Phase wall times (atlas, enumeration, SDFs, batched spin restore,
    scipy SLSQP polish)
  - Per-cand averages
  - _STAGE2_TIMINGS breakdown (build_probe_static / build_starts /
    spin_restore / slsqp / metric_eval)
  - Estimated 8K-cand wall extrapolation

Run::

    uv run --python 3.13 python -m scripts.profile_polish \\
        examples/836656-config-T12.yml scratch/0283-300-04.holes.yml --n-sample 20
"""

from __future__ import annotations

import argparse
import os as _os
import random
import time
from dataclasses import replace
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.arc_first_principled import (
    enumerate_arc_first_candidates,
)
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import (
    JointWeights,
    _STAGE2_TIMINGS,
    stage2_timings,
)
from aind_low_point.optimization.parallel_stage2 import (
    polish_all_with_batched_spin_restore,
)
from aind_low_point.optimization.pose_features import precompute_pose_features
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.visibility_atlas import build_visibility_atlas
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--n-sample", type=int, default=20)
    p.add_argument("--n-top", type=int, default=128)
    p.add_argument("--n-spin", type=int, default=72)
    p.add_argument("--per-arc-cap", type=int, default=50)
    p.add_argument("--global-cap", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-full-pool-size", type=int, default=8908,
                   help="For wall-time extrapolation")
    args = p.parse_args()

    phase_t = {}

    t = time.perf_counter()
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
        R, t_ = T.rotate_translate
        holes_list = _transform_holes(holes_list, R, t_)
    phase_t["setup"] = time.perf_counter() - t

    t = time.perf_counter()
    atlas = build_visibility_atlas(probes, holes_list,
                                    n_top=args.n_top, n_spin=args.n_spin,
                                    verbose=False)
    phase_t["atlas"] = time.perf_counter() - t

    t = time.perf_counter()
    candidates = enumerate_arc_first_candidates(
        probes, atlas, max_arcs=3, max_probes_per_arc=4,
        per_arc_max_hole_tuples=args.per_arc_cap,
        global_max_candidates=args.global_cap,
        verbose=False,
    )
    phase_t["arc_first_enum"] = time.perf_counter() - t

    t = time.perf_counter()
    sdf_by_name = {p.name: build_probe_sdf_from_alpha_wrap(
        runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
    ) for p in probes}
    phase_t["sdf_build"] = time.perf_counter() - t

    t = time.perf_counter()
    pose_features = precompute_pose_features(probes, holes_list)
    phase_t["pose_features"] = time.perf_counter() - t

    random.seed(args.seed)
    n = min(args.n_sample, len(candidates))
    sample = random.sample(candidates, n)
    print(f"Sampling {n} from {len(candidates)} candidates")

    weights = replace(JointWeights(), min_arc_ap_sep_deg=16.0)

    # Reset _STAGE2_TIMINGS so we measure only this run
    for k in list(_STAGE2_TIMINGS.keys()):
        _STAGE2_TIMINGS[k] = 0.0

    print()
    print(f"Polishing {n} candidates (n_workers=1, sequential — for "
          f"_STAGE2_TIMINGS visibility)...", flush=True)
    t = time.perf_counter()
    results = polish_all_with_batched_spin_restore(
        sample, probes, holes_list, pose_features,
        weights=weights, sdf_by_name=sdf_by_name,
        n_workers=1,        # sequential so _STAGE2_TIMINGS accumulates in parent
        n_arcs=3,
        verbose=True,
    )
    phase_t["polish_full"] = time.perf_counter() - t

    # Print breakdown
    print()
    print("=" * 70)
    print("Phase wall times (this run)")
    print("=" * 70)
    print(f"  {'phase':<25}{'wall (s)':>10}{'per cand (ms)':>15}")
    for name, sec in phase_t.items():
        per_cand_ms = sec / n * 1000 if name == "polish_full" else float("nan")
        ms_str = f"{per_cand_ms:>15.1f}" if not np.isnan(per_cand_ms) else " " * 15
        print(f"  {name:<25}{sec:>10.2f}{ms_str}")

    s2 = stage2_timings()
    total_s2 = sum(s2.values())
    print()
    print("=" * 70)
    print(f"Stage 2 component timing (n={n} candidates, accumulated)")
    print("=" * 70)
    for k, v in sorted(s2.items(), key=lambda kv: -kv[1]):
        pct = (v / total_s2 * 100) if total_s2 > 0 else 0
        per_cand_ms = v / n * 1000
        print(f"  {k:<25}{v:>8.2f}s  {pct:>5.1f}%   {per_cand_ms:>7.1f} ms/cand")
    print(f"  {'TOTAL':<25}{total_s2:>8.2f}s")

    # ------- Wall extrapolation to full pool -------
    print()
    print("=" * 70)
    print(f"Wall-time extrapolation to {args.target_full_pool_size} candidates")
    print("=" * 70)
    # Atlas, arc_first, sdf, pose_features are one-time
    one_time = phase_t["atlas"] + phase_t["arc_first_enum"] + phase_t["sdf_build"] + phase_t["pose_features"]
    # Polish scales with n_cands / n_workers
    polish_per_cand_seq = phase_t["polish_full"] / n
    # Spin restore is roughly constant compile + linear runtime
    print(f"  one-time setup           : {one_time:.1f}s")
    for nw in [1, 8, 16]:
        polish_full_seq = polish_per_cand_seq * args.target_full_pool_size
        polish_full_par = polish_full_seq / nw
        total = one_time + polish_full_par
        print(f"  full polish ({nw}-worker)  : ~{polish_full_par:.0f}s  "
              f"=> {total / 60:.1f} min total")

    # ------- Polish quality sanity -------
    max_viols = np.array([r.metrics.max_violation for r in results])
    print()
    print("Polish quality on sample:")
    print(f"  min={max_viols.min():.4f}  "
          f"median={float(np.median(max_viols)):.4f}  "
          f"max={max_viols.max():.4f}")
    feasible_count = int(np.sum(max_viols <= 1e-3))
    print(f"  strictly feasible: {feasible_count}/{n}")

    # ------- Low-hanging suggestions -------
    print()
    print("=" * 70)
    print("Low-hanging optimization targets (highest cost / per-cand wins)")
    print("=" * 70)
    targets = sorted(s2.items(), key=lambda kv: -kv[1])
    for k, v in targets[:3]:
        per = v / n * 1000
        print(f"  • {k}: {per:.0f} ms/cand "
              f"({v / total_s2 * 100:.0f}% of polish wall)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
