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

from aind_low_point.assets import AssetCatalog
from aind_low_point.collisions import CollisionState
from aind_low_point.core import (
    Material,
    MeshTransformable,
    PointsTransformable,
)
from aind_low_point.planning import PlanningState, PoseResolver
from aind_low_point.scene import NodeInstance, Scene


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
        model_matrix: np.ndarray | None = None,
    ) -> None: ...
    def update_mesh(
        self,
        node_id: str,
        *,
        vertices: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        material: ViewMaterial | None = None,
        model_matrix: np.ndarray | None = None,
    ) -> None: ...
    def create_points(
        self,
        node_id: str,
        *,
        name: str,
        positions: np.ndarray,
        material: ViewMaterial,
        point_size: float = 1.0,
        model_matrix: np.ndarray | None = None,
    ) -> None: ...
    def update_points(
        self,
        node_id: str,
        *,
        positions: np.ndarray | None = None,
        material: ViewMaterial | None = None,
        model_matrix: np.ndarray | None = None,
    ) -> None: ...
    def remove(self, node_ids: Iterable[str]) -> None: ...
    def has_node(self, node_id: str) -> bool: ...
    def flush(self) -> None: ...


def _rt_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous affine matrix from rotation R and translation t."""
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


@dataclass
class RendererAdapter:
    backend: RenderBackend
    scene: Scene
    assets: AssetCatalog
    overlays: OverlayResolver | None = None

    # ----- public API -----
    def build(self, plan: PlanningState, coll: CollisionState | None = None) -> None:
        resolver = self._make_resolver(plan)
        hot = coll.hot if coll else frozenset()
        for node in self.scene.nodes.values():
            self._upsert_node(node, resolver, node.key in hot)
        self.backend.flush()

    def sync_nodes(
        self,
        plan: PlanningState,
        nodes: Iterable[NodeInstance],
        coll: CollisionState | None = None,
    ) -> None:
        resolver = self._make_resolver(plan)
        hot = coll.hot if coll else frozenset()
        for node in nodes:
            self._upsert_node(node, resolver, node.key in hot)
        self.backend.flush()

    def repaint_materials(self, node_ids: Iterable[str]) -> None:
        """Update only materials/overlays for given nodes. No pose recompute."""
        for nid in node_ids:
            node = self.scene.nodes.get(nid)
            if not node or not node.enabled:
                continue
            mat = self._resolve_material(node)
            base_vm = material_to_view(mat)
            vm = self.overlays.apply(nid, base_vm) if self.overlays else base_vm
            if self.backend.has_node(nid):
                geom = self.assets.get_geometry(node.asset_key)
                if isinstance(geom, MeshTransformable):
                    self.backend.update_mesh(nid, material=vm)
                elif isinstance(geom, PointsTransformable):
                    self.backend.update_points(nid, material=vm)
        self.backend.flush()

    def remove(self, node_ids: Iterable[str]) -> None:
        self.backend.remove(node_ids)

    # ----- internals -----
    def _make_resolver(self, plan: PlanningState) -> PoseResolver:
        # ``catalog`` flows through to ProbePose so each probe's
        # ``pivot_LPS`` shows up in pose.tip. Pivot is baked once
        # there; the legacy ``get_pivot_for_asset`` wrap stays at its
        # no-op default to avoid double-application.
        return PoseResolver(scene=self.scene, plan=plan, catalog=self.assets)

    def _resolve_material(self, node: NodeInstance) -> Material:
        if node.material_override is not None:
            return node.material_override
        spec = self.assets.get_spec(node.asset_key)
        return spec.default_material

    def _upsert_node(
        self, node: NodeInstance, resolver: PoseResolver, colliding: bool
    ) -> None:
        if not node.enabled:
            return

        # Material: override > spec default
        mat = self._resolve_material(node)
        base_vm = material_to_view(mat)
        vm = self.overlays.apply(node.key, base_vm) if self.overlays else base_vm

        # Geometry + pose via generic catalog
        geom = self.assets.get_geometry(node.asset_key)
        R, t = resolver.world_rt_for_node(node)
        M = _rt_to_matrix(R, t)

        if isinstance(geom, MeshTransformable):
            base_mesh = geom.raw

            if self.backend.has_node(node.key):
                self.backend.update_mesh(node.key, material=vm, model_matrix=M)
            else:
                self.backend.create_mesh(
                    node.key,
                    name=node.key,
                    vertices=base_mesh.vertices,
                    indices=base_mesh.faces,
                    material=vm,
                    model_matrix=M,
                )

        elif isinstance(geom, PointsTransformable):
            if self.backend.has_node(node.key):
                self.backend.update_points(node.key, material=vm, model_matrix=M)
            else:
                self.backend.create_points(
                    node.key,
                    name=node.key,
                    positions=geom.raw,
                    material=vm,
                    point_size=vm.point_size,
                    model_matrix=M,
                )

        else:
            raise ValueError(
                f"Unsupported geometry type for {node.asset_key}: {type(geom)}"
            )


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

        # repaint only nodes whose hot/cold status flipped (material-only, no pose)
        if flips:
            renderer_adapter.repaint_materials(flips)

    return _on_collisions_changed
