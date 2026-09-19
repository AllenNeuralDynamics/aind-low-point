"""Changing a probe's kind has to move its mesh, however the change arrives.

A probe node's ``asset_key`` restates its plan's ``kind``, and only the kind
dropdown used to update both. A plan loaded from YAML dispatched the command and
left the scene pointing at the old mesh, so the drawn probe and the collider kept
the previous geometry while every readout used the new one.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from aind_rutter.domain.catalog import AssetCatalog, AssetSpec
from aind_rutter.domain.plan import (
    PlanningState,
    ProbePlan,
    probe_asset_key,
    probe_node_id,
    reconcile_probe_assets,
)
from aind_rutter.domain.rig import Kinematics
from aind_rutter.domain.scene import NodeInstance, Scene
from aind_rutter.domain.transforms import MeshTransformable

PROBE = "P1"
OLD_KIND = "2.1"
NEW_KIND = "quadbase-alpha"


def _catalog(*kinds: str) -> AssetCatalog:
    import trimesh

    assets = {}
    for i, kind in enumerate(kinds):
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0 + i))
        assets[probe_asset_key(kind)] = AssetSpec(
            key=probe_asset_key(kind),
            kind="mesh",
            mesh=MeshTransformable(mesh),
            collidable=True,
        )
    return AssetCatalog(assets=assets, targets={})


def _scene() -> Scene:
    scene = Scene()
    scene.upsert(
        NodeInstance(
            key=probe_node_id(PROBE),
            asset_key=probe_asset_key(OLD_KIND),
            tags={"probe"},
        )
    )
    return scene


def _plan(kind: str) -> PlanningState:
    return PlanningState(
        kinematics=Kinematics(arc_angles={"a": 0.0}),
        probes={
            PROBE: ProbePlan(kind=kind, arc_id="a", target_point_RAS=(0.0, 0.0, -1.0))
        },
    )


def test_a_changed_kind_repoints_the_node() -> None:
    scene, catalog = _scene(), _catalog(OLD_KIND, NEW_KIND)
    changed = reconcile_probe_assets(scene, _plan(NEW_KIND), catalog)
    assert changed == [probe_node_id(PROBE)]
    assert scene.nodes[probe_node_id(PROBE)].asset_key == probe_asset_key(NEW_KIND)


def test_an_unchanged_kind_reports_nothing() -> None:
    scene, catalog = _scene(), _catalog(OLD_KIND, NEW_KIND)
    assert reconcile_probe_assets(scene, _plan(OLD_KIND), catalog) == []


def test_reconciling_twice_reports_the_change_once() -> None:
    scene, catalog = _scene(), _catalog(OLD_KIND, NEW_KIND)
    plan = _plan(NEW_KIND)
    assert reconcile_probe_assets(scene, plan, catalog)
    assert reconcile_probe_assets(scene, plan, catalog) == []


def test_a_kind_with_no_asset_leaves_the_node_alone() -> None:
    """Pointing a node at a missing asset would fail at render instead."""
    scene, catalog = _scene(), _catalog(OLD_KIND)
    with pytest.warns(UserWarning, match="no asset"):
        assert reconcile_probe_assets(scene, _plan(NEW_KIND), catalog) == []
    assert scene.nodes[probe_node_id(PROBE)].asset_key == probe_asset_key(OLD_KIND)


def test_a_probe_with_no_node_is_skipped() -> None:
    catalog = _catalog(OLD_KIND, NEW_KIND)
    assert reconcile_probe_assets(Scene(), _plan(NEW_KIND), catalog) == []


class _RecordingBackend:
    """The render-backend calls that matter here: create, update, remove."""

    def __init__(self) -> None:
        self.nodes: dict[str, np.ndarray] = {}
        self.removed: list[str] = []
        self.updated: list[str] = []

    def create_mesh(self, node_id, *, vertices, **_kwargs):
        self.nodes[node_id] = np.asarray(vertices)

    def update_mesh(self, node_id, **_kwargs):
        self.updated.append(node_id)

    def remove(self, node_ids):
        for node_id in node_ids:
            self.nodes.pop(node_id, None)
            self.removed.append(node_id)

    def has_node(self, node_id):
        return node_id in self.nodes

    def flush(self):
        pass

    def highlight(self, node_id, **_kwargs):
        pass


def _bounds(vertices: np.ndarray) -> float:
    return float(np.ptp(vertices[:, 2]))


def test_the_renderer_rebuilds_the_node_it_had_drawn() -> None:
    """update_mesh takes new points but not new faces, so it has to recreate."""
    from aind_rutter.render.adapter import RendererAdapter

    scene, catalog = _scene(), _catalog(OLD_KIND, NEW_KIND)
    backend = _RecordingBackend()
    adapter = RendererAdapter(backend=cast(Any, backend), scene=scene, assets=catalog)

    adapter.build(_plan(OLD_KIND))
    drawn_first = _bounds(backend.nodes[probe_node_id(PROBE)])
    assert backend.removed == []

    node = scene.nodes[probe_node_id(PROBE)]
    adapter.sync_nodes(_plan(NEW_KIND), [node])
    assert backend.removed == [probe_node_id(PROBE)]
    assert _bounds(backend.nodes[probe_node_id(PROBE)]) != drawn_first


def test_a_loaded_plan_moves_the_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reported path: a plan-only YAML that names a different kind.

    Everything downstream of the dispatch is wired as the app wires it, so this
    fails if the swap goes back to living in the kind dropdown.
    """
    from aind_rutter.config import PlanningModel
    from aind_rutter.plan_io.replay import apply_plan_model_to_state
    from aind_rutter.render.adapter import RendererAdapter, RenderHandler
    from aind_rutter.session.store import PlanStore

    scene, catalog = _scene(), _catalog(OLD_KIND, NEW_KIND)
    backend = _RecordingBackend()
    adapter = RendererAdapter(backend=cast(Any, backend), scene=scene, assets=catalog)
    store = PlanStore(_plan(OLD_KIND))
    adapter.build(store.state)
    store.subscribe(RenderHandler(scene=scene, adapter=adapter))

    loaded = PlanningModel.model_validate(
        {
            "arcs": {"a": 0.0},
            "probes": {
                PROBE: {
                    "kind": NEW_KIND,
                    "arc": "a",
                    "slider_ml": 0.0,
                    "spin": 0.0,
                    "past_target_mm": 0.0,
                    "offsets_RA": [0.0, 0.0],
                    "target": {"kind": "inline", "point_RAS": [0.0, 0.0, -1.0]},
                }
            },
        }
    )
    apply_plan_model_to_state(loaded, store)

    assert store.state.probes[PROBE].kind == NEW_KIND
    assert scene.nodes[probe_node_id(PROBE)].asset_key == probe_asset_key(NEW_KIND)
    assert probe_node_id(PROBE) in backend.removed
