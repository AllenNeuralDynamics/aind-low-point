"""Runtime helpers for semantic, world-frame scene geometry."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aind_low_point.core import Transformed
from aind_low_point.runtime.build import RuntimeBundle
from aind_low_point.scene import NodeInstance, resolve_base_geometry

FIXTURE_TAGS: frozenset[str] = frozenset({"fixture", "cone", "well", "headframe"})
FIXTURE_EXCLUDED_TAGS: frozenset[str] = frozenset({"implant"})


@dataclass(frozen=True)
class WorldGeometry:
    """A scene node's catalog geometry after applying its scene transform."""

    node_key: str
    asset_key: str
    tags: frozenset[str]
    transformed: Transformed[Any, Any]

    @property
    def raw(self) -> Any:
        return self.transformed.raw


def head_pitch_deg_from_runtime(runtime: RuntimeBundle) -> float:
    """Return rig AP pitch in degrees, expressed in the subject frame."""
    rotation = np.asarray(
        runtime.plan_state.kinematics.subject_from_rig.rotate_translate[0],
        dtype=float,
    )
    return float(np.rad2deg(np.arctan2(rotation[2, 1], rotation[1, 1])))


def _tag_set(values: Iterable[str] | None) -> frozenset[str]:
    return frozenset(values or ())


def scene_nodes_by_tags(
    runtime: RuntimeBundle,
    *,
    include_any: Iterable[str],
    exclude_any: Iterable[str] = (),
) -> tuple[NodeInstance, ...]:
    """Return enabled scene nodes whose tags match the include/exclude filters."""
    include = _tag_set(include_any)
    exclude = _tag_set(exclude_any)
    out: list[NodeInstance] = []
    for node in runtime.scene.nodes.values():
        if not node.enabled:
            continue
        tags = frozenset(node.tags or ())
        if include and not (tags & include):
            continue
        if exclude and tags & exclude:
            continue
        out.append(node)
    return tuple(out)


def fixture_node_keys(
    runtime: RuntimeBundle,
    *,
    include_tags: Iterable[str] = FIXTURE_TAGS,
    exclude_tags: Iterable[str] = FIXTURE_EXCLUDED_TAGS,
) -> tuple[str, ...]:
    """Scene node keys for static fixture-like geometry, excluding implants."""
    return tuple(
        node.key
        for node in scene_nodes_by_tags(
            runtime, include_any=include_tags, exclude_any=exclude_tags
        )
    )


def world_geometry_for_node(
    runtime: RuntimeBundle, node_key: str
) -> WorldGeometry | None:
    """Resolve a scene node's geometry in world LPS coordinates."""
    node = runtime.scene.nodes.get(node_key)
    if node is None:
        return None
    transformed = resolve_base_geometry(runtime.asset_catalog, runtime.scene, node_key)
    if transformed is None:
        return None
    return WorldGeometry(
        node_key=node.key,
        asset_key=node.asset_key,
        tags=frozenset(node.tags or ()),
        transformed=transformed,
    )


def world_geometries_for_nodes(
    runtime: RuntimeBundle, node_keys: Sequence[str]
) -> tuple[WorldGeometry, ...]:
    """Resolve several scene nodes, skipping nodes without concrete geometry."""
    out: list[WorldGeometry] = []
    for node_key in node_keys:
        geometry = world_geometry_for_node(runtime, node_key)
        if geometry is not None:
            out.append(geometry)
    return tuple(out)


def fixture_world_geometries(runtime: RuntimeBundle) -> tuple[WorldGeometry, ...]:
    """World LPS geometry for static fixture-like scene nodes."""
    return world_geometries_for_nodes(runtime, fixture_node_keys(runtime))


def world_geometries_for_asset(
    runtime: RuntimeBundle, asset_key: str
) -> tuple[WorldGeometry, ...]:
    """World LPS geometry for every enabled scene node using ``asset_key``."""
    return tuple(
        geometry
        for node in runtime.scene.nodes.values()
        if node.enabled
        and node.asset_key == asset_key
        and (geometry := world_geometry_for_node(runtime, node.key)) is not None
    )


def implant_world_geometry(
    runtime: RuntimeBundle, *, node_key: str = "implant"
) -> WorldGeometry | None:
    """Return the world-frame implant geometry, preferring the canonical node key."""
    geometry = world_geometry_for_node(runtime, node_key)
    if geometry is not None:
        return geometry
    for node in runtime.scene.nodes.values():
        if node.enabled and "implant" in set(node.tags or ()):
            geometry = world_geometry_for_node(runtime, node.key)
            if geometry is not None:
                return geometry
    by_asset = world_geometries_for_asset(runtime, "implant")
    return by_asset[0] if by_asset else None
