"""Where things are in the world"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Optional,
    Set,
)

from aind_low_point.assets import AssetCatalog
from aind_low_point.core import (
    AffineTransform,
    Material,
    TransformChain,
    Transformed,
)


@dataclass(slots=True)
class NodeInstance:
    key: str  # unique per-node, e.g., "probe:PL"
    asset_key: str  # foreign key to AssetSpec.key, e.g., "probe:2.1"
    transform: TransformChain = field(
        default_factory=lambda: TransformChain.new([AffineTransform.identity()])
    )
    tags: Set[str] = field(default_factory=set)
    material_override: Optional[Material] = None
    enabled: bool = True

    # Per-instance constraints/locks (e.g., calibration)
    locked_axes: Set[str] = field(
        default_factory=set
    )  # {"ap_tilt", "ml_tilt", "spin", "x", "y", "z"}
    # e.g., calibration_rt
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Scene:
    nodes: dict[str, NodeInstance] = field(default_factory=dict)

    def upsert(self, node: NodeInstance):
        self.nodes[node.key] = node

    def remove(self, node_id: str):
        self.nodes.pop(node_id, None)

    def by_tag(self, tag: str):
        return [n for n in self.nodes.values() if tag in n.tags]


def resolve_base_pose(scene: Scene, id: str) -> Optional[TransformChain]:
    node = scene.nodes.get(id)
    if not node:
        return None
    # Resolve the base pose by following the transform chain
    return node.transform


def resolve_base_geometry(
    catalog: AssetCatalog, scene: Scene, id: str
) -> Optional[Transformed]:
    pose = resolve_base_pose(scene, id)
    if not pose:
        return None
    node = scene.nodes.get(id)
    if not node:
        return None
    # Look up the geometry in the catalog
    geometry = catalog.get_geometry(node.asset_key)
    if not geometry:
        return None
    return Transformed(geometry, pose)
