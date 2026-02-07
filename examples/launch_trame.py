"""Launch the trame + PyVista probe planner with synthetic data.

Wires the full domain stack (PlanStore, RendererAdapter, CollisionHandler)
to the TrameController, using procedural geometry so no config/data files
are needed.

Usage:
    uv run --python 3.13 python examples/launch_trame.py
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import trimesh

from aind_low_point.assets import AssetCatalog, AssetSpec, TargetSpec
from aind_low_point.collisions import (
    CollisionAdapter,
    CollisionHandler,
    CollisionState,
)
from aind_low_point.core import (
    Material,
    MeshTransformable,
    PointsTransformable,
)
from aind_low_point.fcl_backend import FCLBackend
from aind_low_point.planning import Kinematics, PlanningState, ProbePlan
from aind_low_point.pyvista_backend import PyVistaBackend
from aind_low_point.rendering import (
    OverlayResolver,
    OverlayState,
    RendererAdapter,
    RenderHandler,
    on_collisions_changed_lambda,
)
from aind_low_point.scene import NodeInstance, Scene
from aind_low_point.state_change import PlanStore
from aind_low_point.trame_controller import TrameController

# ---------------------------------------------------------------------------
# 1) Synthetic geometry
# ---------------------------------------------------------------------------


def _make_probe_mesh() -> trimesh.Trimesh:
    """Capsule-like probe: cylinder + cone tip."""
    shaft = trimesh.creation.cylinder(radius=0.15, height=5.0)
    # shift so base is at z=0, tip at z=-5
    shaft.apply_translation([0, 0, -2.5])
    tip = trimesh.creation.cone(radius=0.15, height=0.4)
    tip.apply_translation([0, 0, -5.2])
    return trimesh.util.concatenate([shaft, tip])


def _make_brain_mesh() -> trimesh.Trimesh:
    """Ellipsoid stand-in for the brain volume."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # scale to brain-ish ellipsoid (mm): ~6 x 5 x 4
    sphere.apply_scale([6.0, 5.0, 4.0])
    return sphere


def _make_target_points() -> np.ndarray:
    """A single target point inside the brain."""
    return np.array([[0.0, 0.0, -1.5]], dtype=np.float64)


# ---------------------------------------------------------------------------
# 2) Build domain objects
# ---------------------------------------------------------------------------


def build_demo_runtime():
    probe_mesh = _make_probe_mesh()
    brain_mesh = _make_brain_mesh()
    target_pts = _make_target_points()

    # -- Asset catalog --
    probe_spec = AssetSpec(
        key="probe:np2",
        kind="mesh",
        default_material=Material("probe", color_hex_str="#4682B4", opacity=0.9),
        mesh=MeshTransformable(probe_mesh),
    )
    brain_spec = AssetSpec(
        key="brain",
        kind="mesh",
        default_material=Material("brain", color_hex_str="#F5DEB3", opacity=0.3),
        mesh=MeshTransformable(brain_mesh),
    )
    target_spec = TargetSpec(
        key="target:demo",
        source_key="brain",
        points=PointsTransformable(target_pts),
    )

    catalog = AssetCatalog(
        assets={"probe:np2": probe_spec, "brain": brain_spec},
        targets={"target:demo": target_spec},
    )

    # -- Scene --
    scene = Scene()
    scene.upsert(
        NodeInstance(
            key="brain",
            asset_key="brain",
            tags={"anatomy"},
        )
    )
    scene.upsert(
        NodeInstance(
            key="probe:A",
            asset_key="probe:np2",
            tags={"probe", "dynamic"},
            extras={"pose_source_probe": "A"},
        )
    )
    scene.upsert(
        NodeInstance(
            key="target:demo",
            asset_key="target:demo",
            tags={"target"},
        )
    )

    # -- Planning state --
    kinematics = Kinematics(arc_angles={"arc1": 0.0})
    probes = {
        "A": ProbePlan(
            kind="np2",
            arc_id="arc1",
            target_key="target:demo",
        ),
    }
    plan_state = PlanningState(
        kinematics=kinematics,
        probes=probes,
        target_index={"target:demo": target_pts.mean(axis=0)},
    )

    return catalog, scene, plan_state


# ---------------------------------------------------------------------------
# 3) Wire everything and launch
# ---------------------------------------------------------------------------


def main():
    catalog, scene, plan_state = build_demo_runtime()

    # Plotter
    pl = pv.Plotter()
    pl.add_axes()
    pl.camera_position = "iso"

    # Backends
    pv_backend = PyVistaBackend(plotter=pl)
    fcl_backend = FCLBackend()

    # Overlay / collision state
    overlay_state = OverlayState()
    overlay_resolver = OverlayResolver(overlays=overlay_state)

    # Adapters
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

    # Collision handler (+ overlay callback)
    coll_state = CollisionState()
    on_coll = on_collisions_changed_lambda(render_adapter, scene, overlay_state)
    coll_handler = CollisionHandler(
        scene=scene,
        adapter=coll_adapter,
        state=coll_state,
        on_state_changed=on_coll,
    )

    # Store
    store = PlanStore(plan_state)
    render_handler = RenderHandler(
        scene=scene,
        adapter=render_adapter,
        get_collision_state=lambda: coll_handler.state,
    )
    store.subscribe(render_handler)
    store.subscribe(coll_handler)

    # Initial build
    render_adapter.build(plan_state)
    coll_adapter.rebuild(plan_state)

    # Controller
    controller = TrameController(
        store=store,
        assets=catalog,
        plotter=pl,
        render_adapter=render_adapter,
        collision_handler=coll_handler,
        overlays_resolver=overlay_resolver,
    )

    # Wire flush callback so PyVistaBackend.flush() triggers trame view_update
    server = controller.build_app()
    pv_backend._flush_callback = server.controller.view_update

    print("Starting trame server...")
    server.start()


if __name__ == "__main__":
    main()
