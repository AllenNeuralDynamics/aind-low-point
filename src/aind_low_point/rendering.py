"""Rendering protocol and adapter"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
)

import numpy as np
from aind_mri_utils.plots import hex_string_to_int

from aind_low_point.core import (
    Material,
)
from aind_low_point.planning import PlanningState
from aind_low_point.scene import Scene


@dataclass(frozen=True)
class ViewMaterial:
    color: int
    opacity: float
    wireframe: bool
    visible: bool


BlendMode = Literal["replace", "multiply", "screen", "alpha_over"]


@dataclass(frozen=True)
class OverlaySpec:
    color: int  # 0xRRGGBB
    alpha: float = 0.6  # 0..1
    blend: BlendMode = "alpha_over"
    priority: int = 0  # higher wins when conflicts
    source: str = "generic"  # "collision" | "hover" | "selection" | ...
    ttl_ms: Optional[int] = None  # optional auto-expire; None = persistent


@dataclass(slots=True)
class OverlayState:
    # node_id -> list of overlays currently active
    by_node: dict[str, List[OverlaySpec]] = field(default_factory=dict)

    def set(self, node_id: str, *specs: OverlaySpec) -> None:
        self.by_node[node_id] = list(specs)

    def set_for_source(self, node_ids: list[str], spec: OverlaySpec) -> None:
        for nid in node_ids:
            lst = [s for s in self.by_node.get(nid, []) if s.source != spec.source]
            lst.append(spec)
            self.by_node[nid] = lst

    def add(self, node_id: str, spec: OverlaySpec) -> None:
        self.by_node.setdefault(node_id, []).append(spec)

    def clear_source(self, source: str, node_ids: list[str] = []) -> None:
        if not node_ids:
            node_ids = list(self.by_node.keys())
        for nid in node_ids:
            lst = self.by_node.get(nid, [])
            kept = [s for s in lst if s.source != source]
            if kept:
                self.by_node[nid] = kept
            else:
                self.by_node.pop(nid)

    def clear_node(self, node_id: str) -> None:
        self.by_node.pop(node_id, None)

    def clear_all(self) -> None:
        self.by_node.clear()


@dataclass(frozen=True)
class CollisionOverlayStyle:
    default_color: int = 0xFF0000  # red
    default_alpha: float = 0.65


def _blend_over(base_rgb: int, over_rgb: int, alpha: float) -> int:
    br, bg, bb = (base_rgb >> 16) & 255, (base_rgb >> 8) & 255, base_rgb & 255
    or_, og, ob = (over_rgb >> 16) & 255, (over_rgb >> 8) & 255, over_rgb & 255
    r = int(round((1 - alpha) * br + alpha * or_))
    g = int(round((1 - alpha) * bg + alpha * og))
    b = int(round((1 - alpha) * bb + alpha * ob))
    return (r << 16) | (g << 8) | b


def material_to_view(m: Material) -> ViewMaterial:
    color_int = (
        hex_string_to_int(m.color_hex_str)
        if isinstance(m.color_hex_str, str)
        else int(m.color_hex_str)
    )
    return ViewMaterial(
        color=color_int,
        opacity=float(m.opacity),
        wireframe=bool(m.wireframe),
        visible=bool(m.visible),
    )


@dataclass
class OverlayResolver:
    overlays: OverlayState

    def apply(self, node_id: str, base_vm: ViewMaterial) -> ViewMaterial:
        specs = self.overlays.by_node.get(node_id)
        if not specs:
            return base_vm
        # choose highest priority (or fold in order; customize if needed)
        spec = max(specs, key=lambda s: s.priority)
        if spec.blend == "replace":
            return ViewMaterial(
                color=spec.color,
                opacity=base_vm.opacity,
                wireframe=base_vm.wireframe,
                visible=base_vm.visible,
            )
        # default alpha-over on color only
        new_color = _blend_over(base_vm.color, spec.color, spec.alpha)
        return ViewMaterial(
            color=new_color,
            opacity=base_vm.opacity,
            wireframe=base_vm.wireframe,
            visible=base_vm.visible,
        )


@dataclass
class CollisionOverlay:
    overlay_color: int = 0xFF0000  # red
    overlay_alpha: float = 0.65  # mix-in strength

    def color_for(self, base_color: int, colliding: bool) -> int:
        return (
            _blend_over(base_color, self.overlay_color, self.overlay_alpha)
            if colliding
            else base_color
        )


class RenderBackend(Protocol):
    def create_mesh(
        self,
        node_id: str,
        *,
        name: str,
        vertices: np.ndarray,
        indices: np.ndarray,
        material: ViewMaterial,
    ) -> None: ...
    def update_mesh(
        self,
        node_id: str,
        *,
        vertices: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        material: ViewMaterial | None = None,
    ) -> None: ...
    def create_points(
        self,
        node_id: str,
        *,
        name: str,
        positions: np.ndarray,
        material: ViewMaterial,
        point_size: float = 1.0,
    ) -> None: ...
    def update_points(
        self,
        node_id: str,
        *,
        positions: np.ndarray | None = None,
        material: ViewMaterial | None = None,
    ) -> None: ...
    def remove(self, node_ids: Iterable[str]) -> None: ...


@dataclass
class RenderHandler:
    scene: Scene
    adapter: RendererAdapter
    # optional shared view-state (e.g., overlays from collisions)
    get_collision_state: Callable[[], CollisionState] | None = None

    def __call__(self, plan: PlanningState, changed_ids: List[str]) -> None:
        # map probe ids → scene nodes; extend as needed
        nodes = [self.scene.nodes.get(f"probe:{pid}") for pid in changed_ids]
        nodes = [n for n in nodes if n is not None]
        # let the adapter apply overlays if provided
        self.adapter.sync_nodes(
            plan,
            nodes,
            coll=self.get_collision_state() if self.get_collision_state else None,
        )


def on_collisions_changed_lambda(
    renderer_adapter: RendererAdapter, scene: Scene, overlays_state: OverlayState
):
    def _on_collisions_changed(
        state: CollisionState, flips: Set[str], plan: PlanningState
    ) -> None:
        # update overlays by source "collision"
        overlays_state.clear_source("collision")
        if state.hot:
            spec = OverlaySpec(
                color=0xFF0000, alpha=0.65, source="collision", priority=30
            )
            overlays_state.set_for_source(list(state.hot), spec)

        # repaint only nodes whose hot/cold status flipped
        nodes = [scene.nodes[nid] for nid in flips if nid in scene.nodes]
        if nodes:
            renderer_adapter.sync_nodes(
                plan, nodes
            )  # adapter reads overlays internally

    return _on_collisions_changed
