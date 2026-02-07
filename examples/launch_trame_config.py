"""Launch the trame + PyVista probe planner from a YAML config file.

Loads a ConfigModel, builds the runtime via build_runtime_from_config(),
then wires the trame controller + PyVista backend.

Usage:
    uv run --python 3.13 python examples/launch_trame_config.py \\
        examples/786864-config.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyvista as pv
import yaml

from aind_low_point.build_runtime import build_runtime_from_config
from aind_low_point.collisions import (
    CollisionAdapter,
    CollisionHandler,
    CollisionState,
)
from aind_low_point.config import ConfigModel
from aind_low_point.fcl_backend import FCLBackend
from aind_low_point.pyvista_backend import PyVistaBackend
from aind_low_point.rendering import (
    OverlayResolver,
    OverlayState,
    RendererAdapter,
    RenderHandler,
    on_collisions_changed_lambda,
)
from aind_low_point.state_change import PlanStore
from aind_low_point.trame_controller import TrameController


def main():
    parser = argparse.ArgumentParser(description="Trame probe planner")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--ccf-volume",
        type=Path,
        default=None,
        help="Path to warped CCF segmentation volume (.nrrd)",
    )
    args = parser.parse_args()

    config_path: Path = args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # --- Load config and build runtime ---
    raw = yaml.safe_load(config_path.read_text())
    cfg = ConfigModel.model_validate(raw)
    bundle = build_runtime_from_config(cfg)

    catalog = bundle.asset_catalog
    scene = bundle.scene
    plan_state = bundle.plan_state

    # --- PyVista plotter ---
    pl = pv.Plotter()
    pl.add_axes()
    pl.camera_position = "iso"

    # --- Backends ---
    pv_backend = PyVistaBackend(plotter=pl)
    fcl_backend = FCLBackend()

    # --- Overlay / collision state ---
    overlay_state = OverlayState()
    overlay_resolver = OverlayResolver(overlays=overlay_state)

    # --- Adapters ---
    render_adapter = RendererAdapter(
        backend=pv_backend,
        scene=scene,
        assets=catalog,
        overlays=overlay_resolver,
    )
    coll_adapter = CollisionAdapter(
        backend=fcl_backend,
        scene=scene,
        assets=catalog,
    )

    # --- Collision handler ---
    coll_state = CollisionState()
    on_coll = on_collisions_changed_lambda(render_adapter, scene, overlay_state)
    coll_handler = CollisionHandler(
        scene=scene,
        adapter=coll_adapter,
        state=coll_state,
        on_state_changed=on_coll,
    )

    # --- Store with subscribers ---
    store = PlanStore(plan_state)
    render_handler = RenderHandler(
        scene=scene,
        adapter=render_adapter,
        get_collision_state=lambda: coll_handler.state,
    )
    store.subscribe(render_handler)
    store.subscribe(coll_handler)

    # --- Initial build ---
    render_adapter.build(plan_state)
    coll_adapter.rebuild(plan_state)

    # --- CCF overlay (optional) ---
    ccf_overlay = None
    if args.ccf_volume is not None:
        from aind_low_point.ccf_overlay import CCFOverlayManager

        ccf_volume_path: Path = args.ccf_volume
        if not ccf_volume_path.exists():
            print(f"CCF volume not found: {ccf_volume_path}", file=sys.stderr)
            sys.exit(1)
        ccf_overlay = CCFOverlayManager(plotter=pl, volume_path=ccf_volume_path)

    # --- Controller ---
    controller = TrameController(
        store=store,
        assets=catalog,
        plotter=pl,
        render_adapter=render_adapter,
        collision_handler=coll_handler,
        overlays_resolver=overlay_resolver,
        ccf_overlay=ccf_overlay,
    )

    server = controller.build_app()
    pv_backend._flush_callback = server.controller.view_update
    if ccf_overlay is not None:
        ccf_overlay.flush_callback = server.controller.view_update

    print(f"Loaded config: {config_path}")
    print(f"  Probes: {sorted(plan_state.probes.keys())}")
    print(f"  Arcs: {sorted(plan_state.kinematics.arc_angles.keys())}")
    print(f"  Scene nodes: {len(scene.nodes)}")
    print("Starting trame server...")
    server.start()


if __name__ == "__main__":
    main()
