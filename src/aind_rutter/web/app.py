"""High-level application builder for the trame probe planner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyvista as pv

from aind_rutter.build.assemble import build_runtime_from_config
from aind_rutter.collision.adapter import CollisionAdapter, CollisionState
from aind_rutter.collision.fcl import FCLBackend
from aind_rutter.collision.worker import CollisionHandler
from aind_rutter.config import ConfigModel
from aind_rutter.plan_io.plan_set import load_plan_set
from aind_rutter.render.adapter import (
    RendererAdapter,
    RenderHandler,
    on_collisions_changed_lambda,
)
from aind_rutter.render.overlays import OverlayResolver, OverlayState
from aind_rutter.render.pyvista import DebouncedFlush, PyVistaBackend
from aind_rutter.session.store import PlanStore
from aind_rutter.session.worker import AsyncLatestWorker
from aind_rutter.web.controller import Download, TrameController

if TYPE_CHECKING:
    from trame.app.singleton import Server


def build_trame_app(
    cfg: ConfigModel,
    *,
    ccf_volume: Path | None = None,
    plan_path: Path | None = None,
    source_config_path: Path | None = None,
    apply_plan_on_start: bool = False,
    plans: Path | None = None,
) -> Server:
    """Build and return a ready-to-start trame server from a ConfigModel.

    Parameters
    ----------
    cfg
        Validated configuration model.
    ccf_volume
        Optional path to a warped CCF segmentation volume (.nrrd).
    plan_path
        Optional plan-only ``PlanningModel`` YAML to apply at startup.
    source_config_path
        Optional path recorded as the source config in exported plans; its
        stem names the files the Files tab offers to save.
    apply_plan_on_start
        When ``True`` and ``plan_path`` exists, apply the plan to the
        planning state once at startup so the app opens with the plan
        already loaded (the "Load plan" button remains wired regardless).
    plans
        Optional emit output directory whose plans the Plans tab steps
        through; they must share ``cfg``'s probe roster.

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
        from aind_rutter.web.ccf_regions import CCFOverlayManager

        ccf_overlay = CCFOverlayManager(plotter=pl, volume_path=ccf_volume)

    import yaml

    from aind_rutter.plan_io.rig_export import export_plan_geometry
    from aind_rutter.plan_io.roundtrip import (
        planning_state_to_plan_model,
        replace_plan_block,
        save_plan_to_config,
    )

    stem = source_config_path.stem if source_config_path is not None else "rutter"
    src_str = str(source_config_path) if source_config_path is not None else None

    # Read once, beside ``cfg``, so the saved config is the one that loaded.
    written = (
        source_config_path.read_text(encoding="utf-8")
        if source_config_path is not None
        else None
    )

    def config_text() -> str:
        updated = save_plan_to_config(store.state, cfg)
        if written is not None:
            return replace_plan_block(written, updated.plan)
        return yaml.dump(
            updated.model_dump(mode="json"), default_flow_style=False, sort_keys=False
        )

    def plan_text() -> str:
        plan = planning_state_to_plan_model(store.state, cfg.plan)
        return yaml.safe_dump(
            plan.model_dump(mode="json"), default_flow_style=False, sort_keys=False
        )

    def poses_text() -> str:
        payload = export_plan_geometry(
            store.state, catalog, source_config=src_str, scene=scene
        )
        return yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)

    downloads = {
        "config": Download(f"{stem}_out.yml", config_text),
        "plan": Download(f"{stem}.plan.yml", plan_text),
        "poses": Download(f"{stem}.poses.yml", poses_text),
    }

    on_load_plan = None
    if plan_path is not None:
        from aind_rutter.config import PlanningModel
        from aind_rutter.plan_io.replay import apply_plan_model_to_state

        def on_load_plan():
            if not plan_path.exists():
                print(f"Plan YAML not found at {plan_path}; nothing to load")
                return
            with open(plan_path) as f:
                raw = yaml.safe_load(f)
            # A rig-poses export (written by export_plan_geometry / the
            # Export-rig-poses button) is a human-readable geometry *report*, not a
            # loadable PlanningModel — it carries derived fields like
            # ``tip_RAS_mm`` and is keyed by ``plan_export_version``. Skip it
            # with a clear message rather than dumping a wall of schema errors.
            if isinstance(raw, dict) and "plan_export_version" in raw:
                print(
                    f"Plan YAML at {plan_path} is an exported geometry report "
                    f"(plan_export_version={raw['plan_export_version']}), not a "
                    "loadable plan; skipping. Use a plan saved via 'Save plan'."
                )
                return
            try:
                loaded = PlanningModel.model_validate(raw)
            except Exception as exc:
                print(f"Plan YAML at {plan_path} failed to validate: {exc}")
                return
            apply_plan_model_to_state(loaded, store)
            print(f"Loaded plan from {plan_path}")

        if apply_plan_on_start and plan_path.exists():
            on_load_plan()

    controller = TrameController(
        store=store,
        assets=catalog,
        plotter=pl,
        render_adapter=render_adapter,
        collision_handler=coll_handler,
        overlays_resolver=overlay_resolver,
        ccf_overlay=ccf_overlay,
        on_load_plan=on_load_plan,
        downloads=downloads,
        plan_set=None if plans is None else load_plan_set(plans),
    )

    controller.apply_default_view()

    server = controller.build_app()
    debounced_flush = DebouncedFlush(server.controller.view_update, delay_s=0.03)
    pv_backend._flush_callback = debounced_flush
    if ccf_overlay is not None:
        ccf_overlay.flush_callback = debounced_flush

    return server
