"""Roundtrip test for BatchedProbeStatic.

Builds a single Stage 2 candidate (from the 836656/T12 manual plan)
two ways:
  1. The existing ``_build_probe_static`` → list[_ProbeStatic]
  2. The new ``build_batched_probe_static`` with B=1

For every field in _ProbeStatic, check that the batched array's
[b=0, i=<probe_idx>, ...] slice matches the per-probe value within
float32 tolerance.

Run::

    uv run --python 3.13 python -m scripts.test_batched_static_roundtrip
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.arc_assignment import ArcAssignment
from aind_low_point.optimization.batched_static import (
    build_batched_probe_static,
)
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes


def parse_manual_plan(plan_path: Path, probe_names: list[str]):
    with open(plan_path) as f:
        mp = yaml.safe_load(f)
    arcs = mp["arcs"]
    _arc_letters_in_order = list(arcs.keys())
    # canonical = ascending AP
    arc_letters_sorted = sorted(arcs.keys(), key=lambda k: arcs[k])
    letter_to_idx = {k: i for i, k in enumerate(arc_letters_sorted)}
    arc_centroids = tuple(arcs[k] for k in arc_letters_sorted)
    probe_to_arc_idx = {}
    probe_to_hole = {}
    # Hole IDs from a known manual mapping
    manual_H = {"MD": 3, "BLA": 4, "PL": 1, "VM": 7, "RSP": 5, "CA1": 10, "CLA": 12}
    for name in probe_names:
        spec = mp["probes"][name]
        probe_to_arc_idx[name] = letter_to_idx[spec["arc"]]
        probe_to_hole[name] = manual_H[name]
    return (
        HoleAssignment(probe_to_hole=probe_to_hole, cost=0.0),
        ArcAssignment(
            probe_to_arc_idx=probe_to_arc_idx,
            arc_centroids_deg=arc_centroids,
            cost=0.0,
        ),
    )


def main() -> int:
    cfg_path = Path("examples/836656-config-T12.yml")
    holes_path = Path("scratch/0283-300-04.holes.yml")
    plan_path = Path("examples/836656-config-T12.plan.yml")

    print("Setup...")
    cfg = ConfigModel.from_yaml(cfg_path)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, name)
        for name in runtime.plan_state.probes
    ]
    holes_list = load_holes(holes_path)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes_list = _transform_holes(holes_list, R, t)

    probe_names = [p.name for p in probes]
    ha, aa = parse_manual_plan(plan_path, probe_names)

    print(f"Manual candidate: HA={ha.probe_to_hole}")
    print(f"                  AA={aa.probe_to_arc_idx}")
    print(f"                  arc_centroids_deg={aa.arc_centroids_deg}")

    # Reference: existing single-candidate build
    reference = _build_probe_static(probes, holes_list, ha, aa)

    # Batched: B=1 with same candidate
    batched = build_batched_probe_static(
        [(ha, aa)],
        probes,
        holes_list,
    )

    print()
    print("Roundtrip check (B=1 vs reference):")
    print(f"  K={batched.K} n_arcs={batched.n_arcs} S={batched.S} SH={batched.SH}")

    ok = True
    for i, ref in enumerate(reference):
        active = bool(batched.probe_active_mask[0, i])
        if not active:
            print(f"  probe {i} {ref.name:<5}: marked INACTIVE but ref exists ✗")
            ok = False
            continue

        # arc_idx
        if int(batched.probe_arc_idx[0, i]) != ref.arc_idx:
            print(
                f"  probe {i} {ref.name:<5}: arc_idx mismatch "
                f"batched={int(batched.probe_arc_idx[0, i])} ref={ref.arc_idx} ✗"
            )
            ok = False

        # target_LPS
        if not np.allclose(
            np.asarray(batched.probe_target_lps[0, i]),
            ref.target_LPS.astype(np.float32),
            atol=1e-5,
        ):
            print(f"  probe {i} {ref.name:<5}: target_LPS mismatch ✗")
            ok = False

        # pivot_local
        if not np.allclose(
            np.asarray(batched.probe_pivot_local[0, i]),
            ref.pivot_local.astype(np.float32),
            atol=1e-5,
        ):
            print(
                f"  probe {i} {ref.name:<5}: pivot_local mismatch "
                f"batched={np.asarray(batched.probe_pivot_local[0, i])} "
                f"ref={ref.pivot_local} ✗"
            )
            ok = False

        # shank tips (padded)
        nsh_ref = ref.shank_tips_local.shape[0]
        nsh_batched = int(batched.probe_shank_mask[0, i].sum())
        if nsh_ref != nsh_batched:
            print(
                f"  probe {i} {ref.name:<5}: shank count mismatch "
                f"batched={nsh_batched} ref={nsh_ref} ✗"
            )
            ok = False
        elif not np.allclose(
            np.asarray(batched.probe_shank_tips[0, i, :nsh_ref]),
            ref.shank_tips_local[:nsh_ref].astype(np.float32),
            atol=1e-5,
        ):
            print(f"  probe {i} {ref.name:<5}: shank tips mismatch ✗")
            ok = False

        # sections
        S_ref = ref.section_axes.shape[0]
        active_sections = int(batched.section_mask[0, i].sum())
        if S_ref != active_sections:
            print(
                f"  probe {i} {ref.name:<5}: section count mismatch "
                f"batched={active_sections} ref={S_ref} ✗"
            )
            ok = False
            continue

        # section fields
        bm = batched
        sec_ok = True
        for fname, ref_arr, bat_arr in [
            ("axes", ref.section_axes, bm.section_axes[0, i, :S_ref]),
            ("e1", ref.section_e1, bm.section_e1[0, i, :S_ref]),
            ("e2", ref.section_e2, bm.section_e2[0, i, :S_ref]),
            ("centers", ref.section_centers, bm.section_centers[0, i, :S_ref]),
            ("cos_theta", ref.section_cos_theta, bm.section_cos_theta[0, i, :S_ref]),
            ("sin_theta", ref.section_sin_theta, bm.section_sin_theta[0, i, :S_ref]),
            ("a", ref.section_a, bm.section_a[0, i, :S_ref]),
            ("b", ref.section_b, bm.section_b[0, i, :S_ref]),
        ]:
            r = ref_arr.astype(np.float32)
            b = np.asarray(bat_arr)
            if not np.allclose(b, r, atol=1e-5):
                max_d = float(np.max(np.abs(b - r)))
                print(
                    f"  probe {i} {ref.name:<5}: section {fname} mismatch "
                    f"max|d|={max_d:.2e} ✗"
                )
                sec_ok = False
                ok = False
        if sec_ok:
            print(
                f"  probe {i} {ref.name:<5}: arc={ref.arc_idx} "
                f"shanks={nsh_ref} sections={S_ref} ✓"
            )

    # Bounds spot-check
    n_arcs = batched.n_arcs
    print("\nBounds (B=1):")
    print(
        f"  arc_aps lo/hi: {float(batched.bounds_lo[0, 0]):.0f} / "
        f"{float(batched.bounds_hi[0, 0]):.0f}"
    )
    print(
        f"  ml lo/hi:      {float(batched.bounds_lo[0, n_arcs]):.0f} / "
        f"{float(batched.bounds_hi[0, n_arcs]):.0f}"
    )
    print(
        f"  spin lo/hi:    {float(batched.bounds_lo[0, n_arcs + 1]):.0f} / "
        f"{float(batched.bounds_hi[0, n_arcs + 1]):.0f}"
    )

    print()
    if ok:
        print("PASS — batched static roundtrips against _build_probe_static.")
        return 0
    else:
        print("FAIL — at least one field mismatched. See above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
