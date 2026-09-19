"""`build_trame_app` on a synthetic subject, with no display.

The web frontend had no test at all, so a change to the render, collision or
store wiring surfaced only when someone launched the app. Building the server is
enough to exercise all of it: the plotter, both adapters, the collision worker
and the controller's reactive state are wired during the build, and only
``server.start()`` needs a browser.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import pyvista as pv
import yaml

from aind_rutter.config import ConfigModel
from tests.synthetic_subject import SyntheticSubject, write_subject

APPLIED_SPIN_DEG = 137.0


@pytest.fixture(scope="module", autouse=True)
def offscreen() -> Iterator[None]:
    """Render to a buffer: the test machine may have no X display."""
    previous = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    try:
        yield
    finally:
        pv.OFF_SCREEN = previous


@pytest.fixture(scope="module")
def subject(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSubject:
    return write_subject(tmp_path_factory.mktemp("subject"))


def _build(subject: SyntheticSubject, **kwargs):
    from aind_rutter.web.app import build_trame_app

    return build_trame_app(ConfigModel.from_yaml(subject.config), **kwargs)


def test_app_builds_and_publishes_the_plan_to_its_state(
    subject: SyntheticSubject,
) -> None:
    state = _build(subject).state
    assert state.probes == ["P1", "P2"]
    assert state.probe == "P1"
    assert state.arcs == ["a", "b"]
    assert state.targets == ["target:brain"]
    assert state.probe_kinds == ["2.1", "quadbase-alpha"]


def test_selected_probe_readouts_follow_the_plan(subject: SyntheticSubject) -> None:
    state = _build(subject).state
    # P1 is selected first: arc a at +10 deg, no spin, no offsets.
    assert state.ap_tilt == pytest.approx(10.0)
    assert state.spin == 0
    assert state.depth == pytest.approx(0.0)
    assert state.offset_r == pytest.approx(0.0)
    assert state.offset_a == pytest.approx(0.0)


def test_a_plan_file_can_be_applied_at_startup(
    subject: SyntheticSubject, tmp_path: Path
) -> None:
    config = ConfigModel.from_yaml(subject.config)
    plan = config.plan.model_copy(deep=True)
    plan.probes["P1"].spin = APPLIED_SPIN_DEG
    plan_path = tmp_path / "plan.yml"
    plan_path.write_text(yaml.safe_dump(plan.model_dump(mode="json")))

    state = _build(subject, plan_path=plan_path, apply_plan_on_start=True).state
    assert state.spin == int(APPLIED_SPIN_DEG)


def test_a_missing_plan_file_does_not_stop_the_build(
    subject: SyntheticSubject, tmp_path: Path
) -> None:
    state = _build(
        subject, plan_path=tmp_path / "absent.yml", apply_plan_on_start=True
    ).state
    assert state.probes == ["P1", "P2"]


def test_recenter_frames_the_brain_where_the_scene_puts_it(
    subject: SyntheticSubject,
) -> None:
    """The synthetic brain's node carries ``headframe_to_lps``, so its world
    centroid is not its mesh centroid. Framing the mesh aims the camera at a
    place nothing is drawn."""

    import numpy as np

    from aind_rutter.build import build_runtime_from_config
    from aind_rutter.build.queries import brain_world_mesh
    from aind_rutter.web.controller import TrameController

    bundle = build_runtime_from_config(ConfigModel.from_yaml(subject.config))
    world = brain_world_mesh(bundle.asset_catalog, bundle.scene)
    raw = bundle.asset_catalog.assets["brain"].mesh.raw
    assert not np.allclose(world.centroid, raw.centroid), "transform is a no-op"

    # recenter_view reads the catalog, the scene and the plotter and nothing
    # else, so the collision and overlay collaborators stay out of it.
    controller = TrameController(
        store=None,
        assets=bundle.asset_catalog,
        plotter=pv.Plotter(off_screen=True),
        render_adapter=SimpleNamespace(scene=bundle.scene),
        collision_handler=None,
        overlays_resolver=None,
    )
    controller.recenter_view()

    focal = np.asarray(controller.plotter.camera.focal_point, dtype=np.float64)
    np.testing.assert_allclose(focal, world.centroid, atol=1e-6)


def test_recenter_falls_back_when_the_scene_has_no_brain(
    subject: SyntheticSubject,
) -> None:
    """Never onto the untransformed mesh: an unplaced brain has no world
    position to frame, so the whole scene is the honest answer."""

    from aind_rutter.build import build_runtime_from_config
    from aind_rutter.domain.scene import Scene
    from aind_rutter.web.controller import TrameController

    bundle = build_runtime_from_config(ConfigModel.from_yaml(subject.config))
    empty = Scene(nodes={})
    plotter = pv.Plotter(off_screen=True)
    before = tuple(plotter.camera.focal_point)
    controller = TrameController(
        store=None,
        assets=bundle.asset_catalog,
        plotter=plotter,
        render_adapter=SimpleNamespace(scene=empty),
        collision_handler=None,
        overlays_resolver=None,
    )
    controller.recenter_view()
    assert tuple(controller.plotter.camera.focal_point) == before


class _FakeState:
    """Stands in for trame's reactive state: attributes, and a `with` block
    that batches writes. Dunder lookup is on the type, so this cannot be a
    SimpleNamespace."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _NullBackend:
    """Draws nothing. What these tests exercise is the adapter's material
    resolution, which runs whether or not a node has been drawn."""

    def create_mesh(self, *a, **k) -> None: ...

    def update_mesh(self, *a, **k) -> None: ...

    def create_points(self, *a, **k) -> None: ...

    def update_points(self, *a, **k) -> None: ...

    def remove(self, *a, **k) -> None: ...

    def flush(self, *a, **k) -> None: ...

    def has_node(self, node_id: str) -> bool:
        return True


def _controller_for(subject: SyntheticSubject):
    """A controller over the synthetic runtime, with a real renderer adapter
    over a backend that draws nothing."""
    from aind_rutter.build import build_runtime_from_config
    from aind_rutter.render.adapter import RendererAdapter
    from aind_rutter.session.store import PlanStore
    from aind_rutter.web.controller import TrameController

    bundle = build_runtime_from_config(ConfigModel.from_yaml(subject.config))
    return TrameController(
        store=PlanStore(bundle.plan_state),
        assets=bundle.asset_catalog,
        plotter=pv.Plotter(off_screen=True),
        render_adapter=RendererAdapter(
            backend=_NullBackend(), scene=bundle.scene, assets=bundle.asset_catalog
        ),
        collision_handler=None,
        overlays_resolver=None,
    )


def test_the_sliders_show_the_angles_the_probe_is_actually_in(
    subject: SyntheticSubject,
) -> None:
    """The readout resolved AP, ML and spin itself and skipped the clamp, so a
    plan past a rig limit put a number on screen that nothing rendered."""
    from aind_rutter.domain.pose import ProbePose, resolved_angles

    controller = _controller_for(subject)
    plan = controller.store.state.probes["P2"]
    plan.bind_ap_to_arc = False
    plan.ap_local = 140.0  # past the +75 deg AP limit
    plan.ml_local = -95.0  # past the -42 deg ML limit
    plan.spin = 400.0

    state = _FakeState(probe="P2")
    controller._load_probe_state(state)

    pose = ProbePose.from_planning_state(controller.store.state, "P2")
    assert (state.ap_tilt, state.ml_tilt) == (pose.ap, pose.ml)
    assert state.spin == int(round(pose.spin))
    assert (state.ap_tilt, state.ml_tilt) == resolved_angles(
        "P2", controller.store.state
    )[:2]
    assert state.ap_tilt < 140.0 and state.ml_tilt > -95.0


def test_default_opacities_survive_a_repaint(subject: SyntheticSubject) -> None:
    """They used to be written onto the actor, which the next repaint restores
    from the node's material — and a collision flip repaints."""
    controller = _controller_for(subject)
    scene = controller.render_adapter.scene
    implant = next(nid for nid, node in scene.nodes.items() if "implant" in node.tags)
    other = next(
        nid
        for nid, node in scene.nodes.items()
        if "fixture" in node.tags and "implant" not in node.tags
    )

    controller.apply_default_opacities()

    # The table's first matching row wins, so a node tagged both implant and
    # fixture takes the implant value.
    assert scene.nodes[implant].material_override.opacity == pytest.approx(0.2)
    assert scene.nodes[other].material_override.opacity == pytest.approx(0.6)

    # What a repaint resolves from is the override, so the value holds.
    resolved = controller.render_adapter._resolve_material(scene.nodes[implant])
    assert resolved.opacity == pytest.approx(0.2)


def test_the_default_opacity_keeps_the_rest_of_the_material(
    subject: SyntheticSubject,
) -> None:
    """Only opacity is overridden; colour and the rest come from the config."""
    controller = _controller_for(subject)
    scene = controller.render_adapter.scene
    implant = next(nid for nid, node in scene.nodes.items() if "implant" in node.tags)
    base = controller.assets.get_spec(scene.nodes[implant].asset_key).default_material

    controller.apply_default_opacities()

    override = scene.nodes[implant].material_override
    assert override.color_hex_str == base.color_hex_str
    assert (override.visible, override.wireframe) == (base.visible, base.wireframe)
    assert override.opacity != base.opacity
