"""Convert a plan-only YAML into a Pinpoint-style target CSV (``rutter-plan-csv``).

Builds the runtime from a base config, applies the plan-only ``PlanningModel``
(the ``*.plan.yml`` files written by ``rutter-emit``), then runs the canonical
``export_plan_geometry`` exporter and flattens its per-probe geometry into the
insertion-plan CSV columns the GUI ``_save_plan`` handler produced
(``scripts/reference_k3d_notebook.py``):

    structure, probe_type, ap_arc_id, ap_angle, ap_rig_angle, ml_angle, spin,
    target_pt_{R,A,S}, ideal_pt_{R,A,S}, hole, distance_past_target,
    position_bearing_shank

``target_pt_*`` is the executed landing — the 3D RAS tip of the position-bearing
shank at its final recording pose (``tip_RAS_mm``: offsets, depth, and angles
applied). Because it is consumed as the final tip position, ``distance_past_target``
is 0 (no further advance past it). ``ideal_pt_*`` is the anatomical target centroid
in RAS that we aimed at (no offset). ``hole`` is parsed from the plan filename
encoding (e.g. ``bla9_ca17_cla2_md6_pl1_rsp5_vm12``).

Run (CPU is plenty; avoids grabbing the GPU):
  JAX_PLATFORMS=cpu uv run --python 3.13 rutter-plan-csv \\
    scratch/837229-config_rerun_plans/plans/plan-10-cov03.96-...plan.yml \\
    --config examples/837229-config.yml --out scratch/plan-10-837229.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import yaml

from aind_rutter.build.assemble import build_runtime_from_config
from aind_rutter.config import ConfigModel, PlanningModel
from aind_rutter.plan_io.replay import apply_plan_model_to_state
from aind_rutter.plan_io.rig_export import export_plan_geometry
from aind_rutter.state_change import PlanStore

COLUMNS = [
    "structure",
    "probe_type",
    "ap_arc_id",
    "ap_angle",
    "ap_rig_angle",
    "ml_angle",
    "spin",
    "target_pt_R",
    "target_pt_A",
    "target_pt_S",
    "ideal_pt_R",
    "ideal_pt_A",
    "ideal_pt_S",
    "hole",
    "distance_past_target",
    "position_bearing_shank",
]


def _holes_from_filename(name: str, probe_names: list[str]) -> dict[str, int]:
    """Parse hole numbers from a plan filename, driven by the probe roster.

    For each probe (longest name first so ``CA1`` wins over ``CA``), find
    ``<lower-probe><digits>`` in the filename (e.g. ``ca17`` → CA1 hole 7) and
    take the digits as the hole.
    """
    holes: dict[str, int] = {}
    for probe in sorted(probe_names, key=len, reverse=True):
        m = re.search(rf"(?:^|[_-]){probe.lower()}(\d+)", name)
        if m:
            holes[probe] = int(m.group(1))
    return holes


def plan_to_rows(plan_path: Path, config_path: Path) -> list[dict[str, object]]:
    """Build the runtime, apply the plan, and flatten the export to CSV rows."""
    cfg = ConfigModel.from_yaml(str(config_path))
    bundle = build_runtime_from_config(cfg)
    store = PlanStore(bundle.plan_state)

    raw = yaml.safe_load(plan_path.read_text())
    plan_model = PlanningModel(**raw)
    apply_plan_model_to_state(plan_model, store)

    payload = export_plan_geometry(
        store.state,
        bundle.asset_catalog,
        source_config=str(config_path),
        scene=bundle.scene,
    )

    probes = payload["probes"]
    holes = _holes_from_filename(plan_path.name, list(probes))

    rows: list[dict[str, object]] = []
    for name, p in probes.items():
        ideal = p["target"]["position_RAS_mm"]  # anatomical target centroid, no offset
        tip = p["tip_RAS_mm"]  # position-bearing shank tip (depth + angles applied)
        rig = p["angles_rig_deg"]
        subj = p["angles_subject_deg"]
        arc_id = (p.get("arc") or {}).get("id")
        rows.append(
            {
                "structure": name,
                "probe_type": p["kind"],
                "ap_arc_id": arc_id,
                "ap_angle": subj["ap"],
                "ap_rig_angle": rig["ap"],
                "ml_angle": subj["ml"],
                "spin": subj["spin"],
                # target_pt = the executed landing: 3D RAS tip of the
                # position-bearing shank at its final recording pose (offsets,
                # depth, and angles already applied).
                "target_pt_R": tip[0],
                "target_pt_A": tip[1],
                "target_pt_S": tip[2],
                # ideal_pt = the anatomical target centroid we aimed at (no offset).
                "ideal_pt_R": ideal[0] if ideal else None,
                "ideal_pt_A": ideal[1] if ideal else None,
                "ideal_pt_S": ideal[2] if ideal else None,
                "hole": holes.get(name),
                # target_pt is consumed as the final tip position, so there is no
                # further advance past it: depth past target is 0 by construction.
                "distance_past_target": 0.0,
                # 1-indexed shank whose tip target_pt reports.
                "position_bearing_shank": p["position_bearing_shank"],
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    """Write flattened plan rows to *out* with the canonical column order."""
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rutter-plan-csv",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("plan", type=Path, help="path to the *.plan.yml file")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("examples/837229-config.yml"),
        help="base config the plan was emitted from",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output CSV path (default: alongside the plan, .csv suffix)",
    )
    args = ap.parse_args(argv)

    rows = plan_to_rows(args.plan, args.config)
    out = args.out or args.plan.with_suffix(".csv")
    write_csv(rows, out)
    print(f"wrote {len(rows)} probes → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
