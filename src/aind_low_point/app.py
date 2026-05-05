"""High-level application builder for the trame probe planner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyvista as pv

from aind_low_point.build_runtime import build_runtime_from_config
from aind_low_point.collisions import (
    CollisionAdapter,
    CollisionHandler,
    CollisionState,
)
from aind_low_point.config import ConfigModel
from aind_low_point.fcl_backend import FCLBackend
from aind_low_point.pyvista_backend import DebouncedFlush, PyVistaBackend
from aind_low_point.rendering import (
    OverlayResolver,
    OverlayState,
    RendererAdapter,
    RenderHandler,
    on_collisions_changed_lambda,
)
from aind_low_point.state_change import AsyncLatestWorker, PlanStore
from aind_low_point.trame_controller import TrameController

if TYPE_CHECKING:
    from trame.app.singleton import Server


def build_trame_app(
    cfg: ConfigModel,
    *,
    ccf_volume: Path | None = None,
    save_path: Path | None = None,
    export_plan_path: Path | None = None,
    source_config_path: Path | None = None,
) -> Server:
    """Build and return a ready-to-start trame server from a ConfigModel.

    Parameters
    ----------
    cfg
        Validated configuration model.
    ccf_volume
        Optional path to a warped CCF segmentation volume (.nrrd).

    Returns
    -------
    trame Server
        Call ``server.start()`` to launch.
    """
    bundle = build_runtime_from_config(cfg)
    catalog = bundle.asset_catalog
    scene = bundle.scene
    plan_state = bundle.plan_state

    pl = pv.Plotter()
    pl.add_axes()
    pl.camera_position = "iso"

    pv_backend = PyVistaBackend(plotter=pl)
    fcl_backend = FCLBackend()

    overlay_state = OverlayState()
    overlay_resolver = OverlayResolver(overlays=overlay_state)

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

    coll_state = CollisionState()
    on_coll = on_collisions_changed_lambda(render_adapter, scene, overlay_state)
    coll_handler = CollisionHandler(
        scene=scene,
        adapter=coll_adapter,
        state=coll_state,
        on_state_changed=on_coll,
    )

    store = PlanStore(plan_state)
    render_handler = RenderHandler(
        scene=scene,
        adapter=render_adapter,
        get_collision_state=lambda: coll_handler.state,
    )
    store.subscribe(render_handler)

    render_adapter.build(plan_state)
    coll_adapter.rebuild(plan_state)

    # Collision runs in a background thread; results delivered via event loop.
    import asyncio

    loop = asyncio.get_event_loop()
    async_coll = AsyncLatestWorker(
        prepare=coll_handler.prepare,
        work=coll_handler.work,
        deliver=lambda result: coll_handler.deliver(result, store.state),
        post_to_main=loop.call_soon_threadsafe,
    )
    store.subscribe(async_coll)

    ccf_overlay = None
    if ccf_volume is not None:
        from aind_low_point.ccf_overlay import CCFOverlayManager

        ccf_overlay = CCFOverlayManager(plotter=pl, volume_path=ccf_volume)

    on_save = None
    if save_path is not None:
        from aind_low_point.build_runtime import save_plan_to_config

        def on_save():
            updated = save_plan_to_config(store.state, cfg)
            import yaml

            data = updated.model_dump(mode="json")
            with open(save_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"Saved plan to {save_path}")

    on_export_plan = None
    if export_plan_path is not None:
        from aind_low_point.build_runtime import export_plan_geometry

        src_str = (
            str(source_config_path) if source_config_path is not None else None
        )

        def on_export_plan():
            import yaml

            payload = export_plan_geometry(
                store.state, catalog, source_config=src_str
            )
            with open(export_plan_path, "w") as f:
                yaml.safe_dump(
                    payload, f, default_flow_style=False, sort_keys=False
                )
            print(f"Exported plan geometry to {export_plan_path}")

    controller = TrameController(
        store=store,
        assets=catalog,
        plotter=pl,
        render_adapter=render_adapter,
        collision_handler=coll_handler,
        overlays_resolver=overlay_resolver,
        ccf_overlay=ccf_overlay,
        on_save=on_save,
        on_export_plan=on_export_plan,
    )

    server = controller.build_app()
    debounced_flush = DebouncedFlush(server.controller.view_update, delay_s=0.03)
    pv_backend._flush_callback = debounced_flush
    if ccf_overlay is not None:
        ccf_overlay.flush_callback = debounced_flush

    return server
