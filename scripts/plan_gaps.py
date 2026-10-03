#!/usr/bin/env python3
"""List how close a plan brings each pair the collision check tests, closest first.

The FCL gate only says whether a plan collides; this says by how much it does
not, and where, which is what to read when a plan looks tight in the viewer.

    scripts/plan_gaps.py <config.yml> <plan.plan.yml> --probes-only --limit 10

The plan is applied the way the viewer's "Load plan" applies it, so the gaps are
the ones the viewer would show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from aind_rutter.build.assemble import build_runtime_from_config
from aind_rutter.collision.adapter import CollisionAdapter
from aind_rutter.collision.fcl import FCLBackend
from aind_rutter.config import ConfigModel, PlanningModel
from aind_rutter.plan_io.replay import apply_plan_model_to_state
from aind_rutter.session.store import PlanStore


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("plan", type=Path)
    p.add_argument("--limit", type=int, default=None, help="Closest pairs to list.")
    p.add_argument(
        "--probes-only", action="store_true", help="Leave out probe-fixture pairs."
    )
    args = p.parse_args()

    runtime = build_runtime_from_config(ConfigModel.from_yaml(args.config))
    store = PlanStore(runtime.plan_state)
    plan = PlanningModel.model_validate(yaml.safe_load(args.plan.read_text()))
    apply_plan_model_to_state(plan, store)
    backend = FCLBackend()
    CollisionAdapter(
        backend=backend, scene=runtime.scene, assets=runtime.asset_catalog
    ).rebuild(store.state)

    gaps = backend.gaps()
    if args.probes_only:
        gaps = [g for g in gaps if "probe:" in g.id1 and "probe:" in g.id2]
    print(f"{'gap (mm)':>9}  {'pair':<36}  closest approach, LPS (mm)")
    for g in gaps[: args.limit]:
        x, y, z = g.point_LPS
        print(
            f"{g.distance_mm:9.3f}  {g.id1 + ' – ' + g.id2:<36}"
            f"  ({x:6.2f}, {y:6.2f}, {z:6.2f})"
        )


if __name__ == "__main__":
    main()
