"""Compare round-robin (batched, per-probe) vs pairwise (sequential)
spin restore methods on a sample of candidates.

For each test candidate:
  1. Build initial y0 from arc midpoint AP + atlas-anchor ml/spin
  2. Method A: batched round-robin spin restore (current
     ``make_batched_spin_restore``, per-probe coord descent × n_rounds)
  3. Method B: sequential pairwise spin restore (existing
     ``spin_restore_jax``, per-pair joint 8×8 grid greedy)
  4. Polish each restored y0 via scipy SLSQP (skip_spin_restore=True),
     compare polished max_viol per method

Stratify by AP-envelope width (narrow / medium / wide) so we can see
where pairwise gains matter most.

Run::

    uv run --python 3.13 python -m scripts.compare_spin_restore_methods \\
        examples/836656-config-T12.yml scratch/0283-300-04.holes.yml --n-per-bin 10
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
from aind_low_point.optimization.spin_restore_jax import spin_restore_jax

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.arc_first_principled import (
    enumerate_arc_first_candidates,
)
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import (
    JointWeights,
    _build_probe_static,
)
from aind_low_point.optimization.parallel_stage2 import (
    _per_arc_envelopes,
    polish_all,
)
from aind_low_point.optimization.pose_features import precompute_pose_features
from aind_low_point.optimization.sdf import build_probe_sdf
from aind_low_point.optimization.visibility_atlas import build_visibility_atlas
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes


def build_initial_y0(cand, probes, n_arcs: int = 3) -> np.ndarray:
    """y0 from arc_centroids_deg + per-probe (ml_seed, spin_seed)."""
    K = len(probes)
    y0 = np.zeros(n_arcs + 2 * K, dtype=np.float64)
    for arc_idx in range(min(n_arcs, len(cand.aa.arc_centroids_deg))):
        y0[arc_idx] = float(cand.aa.arc_centroids_deg[arc_idx])
    for i, p in enumerate(probes):
        y0[n_arcs + 2 * i] = float(cand.ml_seed.get(p.name, 0.0))
        y0[n_arcs + 2 * i + 1] = float(cand.spin_seed.get(p.name, 0.0))
    return y0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--n-per-bin", type=int, default=10)
    p.add_argument("--n-top", type=int, default=128)
    p.add_argument("--n-spin", type=int, default=72)
    p.add_argument("--per-arc-cap", type=int, default=50)
    p.add_argument("--global-cap", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-workers", type=int, default=8)
    args = p.parse_args()

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

    print("Atlas + enumeration...", flush=True)
    t0 = time.perf_counter()
    atlas = build_visibility_atlas(
        probes, holes_list, n_top=args.n_top, n_spin=args.n_spin, verbose=False
    )
    candidates = enumerate_arc_first_candidates(
        probes,
        atlas,
        max_arcs=3,
        max_probes_per_arc=4,
        per_arc_max_hole_tuples=args.per_arc_cap,
        global_max_candidates=args.global_cap,
        verbose=False,
    )
    print(
        f"  done in {time.perf_counter() - t0:.1f}s ({len(candidates)} candidates)",
        flush=True,
    )

    print("SDFs + pose features...", flush=True)
    sdf_by_name = {
        p.name: build_probe_sdf(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    pose_features = precompute_pose_features(probes, holes_list)

    # Stratify by max-arc envelope width
    max_widths = []
    for c in candidates:
        envs = _per_arc_envelopes(c, atlas)
        widths = [(e[1] - e[0]) if e is not None else 0.0 for e in envs]
        max_widths.append(max(widths) if widths else 0.0)
    max_widths = np.array(max_widths)

    bins = {
        "narrow (<5°)": [i for i in range(len(candidates)) if max_widths[i] < 5.0],
        "medium (5-15°)": [
            i for i in range(len(candidates)) if 5.0 <= max_widths[i] < 15.0
        ],
        "wide (>=15°)": [i for i in range(len(candidates)) if max_widths[i] >= 15.0],
    }
    random.seed(args.seed)
    sample_per_bin: dict[str, list[int]] = {}
    for name, ids in bins.items():
        n = min(args.n_per_bin, len(ids))
        sample_per_bin[name] = random.sample(ids, n) if n > 0 else []
        print(f"  bin {name:<18}: {len(ids)} total, sampled {n}")

    flat_sample = []
    for name, ids in sample_per_bin.items():
        flat_sample.extend([(name, i) for i in ids])
    N = len(flat_sample)
    if N == 0:
        print("No candidates.")
        return 1

    sample_cands = [candidates[i] for _, i in flat_sample]
    weights = replace(JointWeights(), min_arc_ap_sep_deg=16.0)
    n_arcs = 3

    # ----- Initial y0 -----
    print()
    print("Building initial y0 per candidate (midpoint AP + anchor seeds)")
    y0_init = np.stack(
        [build_initial_y0(c, probes, n_arcs=n_arcs) for c in sample_cands]
    )  # (N, n_vars)

    # ===== Method A: batched round-robin =====
    print()
    print(f"Method A: batched round-robin spin restore ({N} cands)", flush=True)
    import jax.numpy as jnp

    from aind_low_point.optimization.batched_spin_restore import (
        make_batched_spin_restore,
    )
    from aind_low_point.optimization.batched_static import (
        build_batched_probe_static,
    )

    pairs = [(c.ha, c.aa) for c in sample_cands]
    bs = build_batched_probe_static(
        pairs,
        probes,
        holes_list,
        n_arcs=n_arcs,
        sdf_by_name=sdf_by_name,
    )
    spin_restore_RR = make_batched_spin_restore(
        bs,
        weights,
        n_spins=8,
        n_rounds=2,
    )
    t0 = time.perf_counter()
    y0_RR = np.asarray(spin_restore_RR(jnp.asarray(y0_init)), dtype=np.float64)
    print(
        f"  done in {time.perf_counter() - t0:.1f}s  "
        f"({(time.perf_counter() - t0) / N * 1000:.0f} ms/cand)"
    )

    import jax

    del spin_restore_RR, bs
    jax.clear_caches()

    # ===== Method B: sequential pairwise =====
    print()
    print(f"Method B: sequential pairwise spin restore ({N} cands)", flush=True)
    y0_PW = np.zeros_like(y0_init)
    t0 = time.perf_counter()
    for i, c in enumerate(sample_cands):
        statics = _build_probe_static(
            probes,
            holes_list,
            c.ha,
            c.aa,
            sdf_by_name=sdf_by_name,
        )
        y0_PW[i] = spin_restore_jax(y0_init[i], statics, n_arcs)
    t_pw = time.perf_counter() - t0
    print(f"  done in {t_pw:.1f}s  ({t_pw / N * 1000:.0f} ms/cand)")

    # ===== Polish each restored y0 =====
    print()
    print(
        f"Polishing {N} candidates × 2 methods = {2 * N} polishes "
        f"(skip_spin_restore=True)",
        flush=True,
    )
    polish_pairs = [(c.ha, c.aa, float("nan")) for c in sample_cands]
    # Polish A
    t0 = time.perf_counter()
    results_RR = polish_all(
        polish_pairs,
        probes,
        holes_list,
        pose_features,
        weights=weights,
        sdf_by_name=sdf_by_name,
        n_workers=args.n_workers,
        y0_per_candidate=[y0_RR[i] for i in range(N)],
        skip_spin_restore=True,
        verbose=False,
    )
    t_a = time.perf_counter() - t0
    print(f"  polish (A): {t_a:.1f}s")
    t0 = time.perf_counter()
    results_PW = polish_all(
        polish_pairs,
        probes,
        holes_list,
        pose_features,
        weights=weights,
        sdf_by_name=sdf_by_name,
        n_workers=args.n_workers,
        y0_per_candidate=[y0_PW[i] for i in range(N)],
        skip_spin_restore=True,
        verbose=False,
    )
    t_b = time.perf_counter() - t0
    print(f"  polish (B): {t_b:.1f}s")

    # ===== Report =====
    print()
    print("=" * 80)
    print("Per-candidate max_viol (A = round-robin, B = pairwise)")
    print("=" * 80)
    mv_A = np.array([r.metrics.max_violation for r in results_RR])
    mv_B = np.array([r.metrics.max_violation for r in results_PW])
    print(f"  {'bin':<18}{'cand':>6}{'mv_RR':>10}{'mv_PW':>10}{'PW-RR':>10}  comment")
    for i, (bin_name, cand_idx) in enumerate(flat_sample):
        d = mv_B[i] - mv_A[i]
        comment = ""
        if abs(d) > 0.1:
            if d > 0:
                comment = "  RR better"
            else:
                comment = "  PW better"
        print(
            f"  {bin_name:<18}{cand_idx:>6}{mv_A[i]:>10.4f}"
            f"{mv_B[i]:>10.4f}{d:>+10.4f}{comment}"
        )

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  RR median max_viol : {float(np.median(mv_A)):.4f}")
    print(f"  PW median max_viol : {float(np.median(mv_B)):.4f}")
    print(
        f"  PW wins (|PW - RR| > 0.1, PW lower) : "
        f"{int(np.sum((mv_B + 0.1 < mv_A)))} / {N}"
    )
    print(
        f"  RR wins (|PW - RR| > 0.1, RR lower) : "
        f"{int(np.sum((mv_A + 0.1 < mv_B)))} / {N}"
    )
    print(
        f"  effectively tied (|PW - RR| ≤ 0.1)  : "
        f"{int(np.sum(np.abs(mv_A - mv_B) <= 0.1))} / {N}"
    )

    # Speed
    print()
    print(
        f"  spin restore wall: RR ~{t_a:.0f}ms/cand (batched), "
        f"PW ~{t_pw / N * 1000:.0f}ms/cand (sequential)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
