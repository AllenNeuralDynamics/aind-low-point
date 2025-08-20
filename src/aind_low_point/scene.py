"""Where things are in the world"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Optional,
    Set,
)

from aind_low_point.core import (
    AffineTransform,
    Material,
    TransformChain,
)


@dataclass(slots=True)
class NodeInstance:
    id: str  # unique per-node, e.g., "probe:PL"
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
        self.nodes[node.id] = node

    def remove(self, node_id: str):
        self.nodes.pop(node_id, None)

    def by_tag(self, tag: str):
        return [n for n in self.nodes.values() if tag in n.tags]
