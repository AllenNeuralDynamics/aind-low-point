"""Check FCL pairwise clearance among Phase 2's "strictly feasible"
candidates (max_violation ≤ 0.001 per P2 metric eval).

Question: how many of the 142 P2-feasibles actually have FCL-detected
overlap that SDF missed?

For each P2-feasible candidate, compute FCL signed pairwise clearance
at the polished pose (offsets=0, depth=0) using the same hybrid
distance/collide query Phase 3's _push_restore_full_x uses. Classify:
  - Truly FCL-feasible: min pairwise clearance ≥ 0
  - Minor FCL violation: -0.5 ≤ min < 0 mm
  - Major FCL violation: min < -0.5 mm

Report distribution + breakdown by manual position.
"""

from __future__ import annotations

import argparse
import os as _os
import pickle
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import fcl
import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import (
    _build_probe_static,
    _signed_pair_clearance,
)
from aind_low_point.optimization.kinematics import pose_from_optimizer_vars
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes


def main() -> int:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--polish-pkl", type=Path, default=Path("/tmp/full_polish_T12.pkl"))
    p.add_argument("--feasibility-threshold", type=float, default=0.001)
    p.add_argument(
        "--no-fixtures",
        action="store_true",
        help="Skip probe-vs-fixture FCL checks (Stage 2 doesn't "
        "optimize fixture clearance, so probe-probe is the "
        "more meaningful parity check)",
    )
    args = p.parse_args()

    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes_list = load_holes(args.holes)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes_list = _transform_holes(holes_list, R, t)

    # FCL-only path — no SDF needed.
    # Build fixture BVHs for probe-vs-fixture FCL check (cone, well, headframe).
    fixture_bvhs: dict[str, fcl.BVHModel] = {}
    if not args.no_fixtures:
        for asset_name in ("cone", "well", "headframe"):
            try:
                mesh = runtime.asset_catalog.get_geometry(asset_name).raw
            except KeyError:
                continue
            fixture_bvhs[asset_name] = make_fcl_bvh(mesh)
        if fixture_bvhs:
            print(f"  fixtures: {list(fixture_bvhs)}")
    else:
        print("  (skipping probe-vs-fixture checks per --no-fixtures)")
    bvh_cache = {
        p.name: (
            make_fcl_bvh(p.collision_mesh) if p.collision_mesh is not None else None
        )
        for p in probes
    }

    with open(args.polish_pkl, "rb") as f:
        data = pickle.load(f)
    candidates_arc = data["candidates"]
    results = data["results"]
    manual_rank_in_pool = data["manual_rank"]

    # Find P2-feasibles
    feas_idxs = [
        i
        for i, r in enumerate(results)
        if r.metrics.max_violation <= args.feasibility_threshold
    ]
    print(f"P2-feasibles (max_viol ≤ {args.feasibility_threshold}): {len(feas_idxs)}")

    # For each P2-feasible, compute FCL min pairwise clearance at the
    # polished pose (offsets=0, depth=0)
    fcl_min_clearances: list[float] = []
    pair_details: list[dict] = []
    for k, cand_idx in enumerate(feas_idxs):
        cand = candidates_arc[cand_idx]
        jc = results[cand_idx]
        y = np.asarray(jc.reduced_y, dtype=np.float64)
        n_arcs = jc.n_arcs
        statics = _build_probe_static(
            probes,
            holes_list,
            cand.ha,
            cand.aa,
            bvh_cache=bvh_cache,
        )
        # Patch B layout: (arc_aps, (ml, sx, sy) × P).
        for i, st in enumerate(statics):
            off = n_arcs + 3 * i
            ml = float(y[off])
            sx = float(y[off + 1])
            sy = float(y[off + 2])
            spin = float(np.degrees(np.arctan2(sy, sx)))
            ap = float(y[st.arc_idx])
            R_w, pose_tip = pose_from_optimizer_vars(
                target_LPS=st.target_LPS,
                ap_deg=ap,
                ml_deg=ml,
                spin_deg=spin,
                offset_R_mm=0.0,
                offset_A_mm=0.0,
                past_target_mm=0.0,
                recording_center_local=st.pivot_local,
            )
            if st.bvh_obj is not None:
                st.bvh_obj.setTransform(
                    fcl.Transform(
                        np.ascontiguousarray(R_w, dtype=np.float64),
                        np.ascontiguousarray(pose_tip, dtype=np.float64),
                    )
                )
        # Min pairwise FCL clearance + worst pair (probe-probe + probe-fixture)
        K = len(statics)
        min_d = float("inf")
        worst_pair = None
        for a in range(K):
            ba = statics[a].bvh_obj
            if ba is None:
                continue
            for b in range(a + 1, K):
                bb = statics[b].bvh_obj
                if bb is None:
                    continue
                d = _signed_pair_clearance(ba, bb)
                if d < min_d:
                    min_d = d
                    worst_pair = (statics[a].name, statics[b].name)
            # Probe-vs-fixture (fixture transforms are identity).
            for fx_name, fx_bvh in fixture_bvhs.items():
                d = _signed_pair_clearance(ba, fx_bvh)
                if d < min_d:
                    min_d = d
                    worst_pair = (statics[a].name, fx_name)
        fcl_min_clearances.append(min_d)
        pair_details.append(
            {
                "cand_idx": cand_idx,
                "min_fcl_clearance": min_d,
                "worst_pair": worst_pair,
                "is_manual": cand_idx == manual_rank_in_pool,
            }
        )
        if (k + 1) % 30 == 0:
            print(f"  processed {k + 1}/{len(feas_idxs)}", flush=True)

    arr = np.array(fcl_min_clearances)
    truly_clear = arr >= 0
    minor_violation = (arr < 0) & (arr >= -0.5)
    major_violation = arr < -0.5

    print()
    print("=" * 78)
    print("FCL pairwise min-clearance among P2-feasibles")
    print("=" * 78)
    print(
        f"  Truly FCL-clear (min ≥ 0):           "
        f"{int(truly_clear.sum()):>3} / {len(arr)} "
        f"({truly_clear.mean() * 100:.1f}%)"
    )
    print(
        f"  Minor violation (-0.5 ≤ min < 0):    "
        f"{int(minor_violation.sum()):>3} / {len(arr)} "
        f"({minor_violation.mean() * 100:.1f}%)"
    )
    print(
        f"  Major violation (min < -0.5):        "
        f"{int(major_violation.sum()):>3} / {len(arr)} "
        f"({major_violation.mean() * 100:.1f}%)"
    )
    print()
    print("  FCL min clearance distribution:")
    print(
        f"    min={arr.min():.4f}  median={float(np.median(arr)):.4f}  "
        f"max={arr.max():.4f}"
    )
    for p_ in [5, 25, 50, 75, 95]:
        print(f"    {p_}th percentile: {float(np.percentile(arr, p_)):.4f}")

    # Penetration depth distribution among violators
    viol = arr[arr < 0]
    if viol.size > 0:
        print()
        print(f"  Penetration depth distribution (n={viol.size}):")
        print(f"    min penetration: {-viol.max():.4f} mm")
        print(f"    median penetration: {-float(np.median(viol)):.4f} mm")
        print(f"    max penetration: {-viol.min():.4f} mm")

    # Manual specifically
    manual_entry = next((d for d in pair_details if d["is_manual"]), None)
    if manual_entry is not None:
        print()
        print(
            f"Manual (cand #{manual_rank_in_pool}): "
            f"min FCL clearance = {manual_entry['min_fcl_clearance']:.4f}  "
            f"worst pair = {manual_entry['worst_pair']}"
        )

    # Worst FCL violators
    sorted_details = sorted(pair_details, key=lambda d: d["min_fcl_clearance"])
    print()
    print("Top 10 worst FCL violators among P2-feasibles:")
    for d in sorted_details[:10]:
        m = "  [MANUAL]" if d["is_manual"] else ""
        print(
            f"  cand #{d['cand_idx']:>5}: FCL min={d['min_fcl_clearance']:.4f} "
            f"({d['worst_pair'][0]}/{d['worst_pair'][1]}){m}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
