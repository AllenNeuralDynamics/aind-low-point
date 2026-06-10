from __future__ import annotations

import numpy as np
import trimesh

from aind_low_point.assets import AssetCatalog, AssetSpec
from aind_low_point.core import AffineTransform, MeshTransformable, TransformChain
from aind_low_point.planning import Kinematics, PlanningState
from aind_low_point.runtime.build import CollisionLabelIndex, RuntimeBundle
from aind_low_point.runtime.scene_geometry import (
    fixture_node_keys,
    implant_world_geometry,
    world_geometry_for_node,
)
from aind_low_point.scene import NodeInstance, Scene, resolve_base_geometry


def _runtime_with_scene_nodes() -> RuntimeBundle:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    catalog = AssetCatalog(
        assets={
            "asset:well": AssetSpec(
                key="asset:well",
                kind="mesh",
                mesh=MeshTransformable(mesh),
            ),
            "asset:implant": AssetSpec(
                key="asset:implant",
                kind="mesh",
                mesh=MeshTransformable(mesh),
            ),
        }
    )
    scene = Scene()
    scene.upsert(
        NodeInstance(
            key="well-node",
            asset_key="asset:well",
            transform=TransformChain.new(
                [AffineTransform(translation=np.array([1.0, 2.0, 3.0]))]
            ),
            tags={"fixture", "well"},
        )
    )
    scene.upsert(
        NodeInstance(
            key="implant-node",
            asset_key="asset:implant",
            tags={"fixture", "implant"},
        )
    )
    return RuntimeBundle(
        asset_catalog=catalog,
        targets_pts={},
        scene=scene,
        collision_labels=CollisionLabelIndex(label_to_bit={}, bit_to_label={}),
        plan_state=PlanningState(kinematics=Kinematics(), probes={}),
    )


def test_resolve_base_geometry_uses_node_asset_key() -> None:
    runtime = _runtime_with_scene_nodes()

    transformed = resolve_base_geometry(
        runtime.asset_catalog, runtime.scene, "well-node"
    )

    assert transformed is not None
    assert np.allclose(transformed.raw.centroid, [1.0, 2.0, 3.0])


def test_fixture_node_keys_exclude_implants() -> None:
    runtime = _runtime_with_scene_nodes()

    assert fixture_node_keys(runtime) == ("well-node",)


def test_world_geometry_helpers_return_node_context() -> None:
    runtime = _runtime_with_scene_nodes()

    well = world_geometry_for_node(runtime, "well-node")
    implant = implant_world_geometry(runtime)

    assert well is not None
    assert well.node_key == "well-node"
    assert well.asset_key == "asset:well"
    assert np.allclose(well.raw.centroid, [1.0, 2.0, 3.0])
    assert implant is not None
    assert implant.node_key == "implant-node"
