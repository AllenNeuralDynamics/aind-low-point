"""Overlay specs and how they blend into a node's material."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from aind_mri_utils.plots import hex_string_to_int

from aind_rutter.domain.catalog import Material


@dataclass(frozen=True)
class ViewMaterial:
    color: int
    opacity: float
    wireframe: bool
    visible: bool
    point_size: float = 5.0


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

    def set_for_source(self, node_ids: list[str], spec: OverlaySpec) -> None:
        for nid in node_ids:
            lst = [s for s in self.by_node.get(nid, []) if s.source != spec.source]
            lst.append(spec)
            self.by_node[nid] = lst

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
        point_size=float(m.point_size),
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
                point_size=base_vm.point_size,
            )
        # default alpha-over on color only
        new_color = _blend_over(base_vm.color, spec.color, spec.alpha)
        return ViewMaterial(
            color=new_color,
            opacity=base_vm.opacity,
            wireframe=base_vm.wireframe,
            visible=base_vm.visible,
            point_size=base_vm.point_size,
        )
