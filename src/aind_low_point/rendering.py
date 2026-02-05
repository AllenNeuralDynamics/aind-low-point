"""Rendering protocol and adapter"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
)

import numpy as np
import trimesh
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


## LRU of pose
# ---- pose signature ----
def _pose_signature(R: np.ndarray, t: np.ndarray, *, tol: float = 1e-6) -> bytes:
    qR = np.round(R / tol).astype(np.int64).ravel()
    qt = np.round(t / tol).astype(np.int64).ravel()
    return b"v1|" + qR.tobytes() + b"|" + qt.tobytes()


Key = Tuple[str, bytes]  # (mesh_id, pose_sig)


# ---- cache entry ----
@dataclass(slots=True)
class _CacheEntry:
    vertices: np.ndarray  # (N,3) float64
    faces: np.ndarray  # (M,3) int32/64 (passed through; backend re-casts as needed)


# ---- LRU cache of transformed vertices ----
class _TransformCache:
    def __init__(self, maxsize: int = 256):
        self.maxsize = int(maxsize)
        self._od: "OrderedDict[Key, _CacheEntry]" = OrderedDict()

    def get_or_compute(
        self, mesh_id: str, base: trimesh.Trimesh, R: np.ndarray, t: np.ndarray
    ) -> _CacheEntry:
        key = (mesh_id, _pose_signature(R, t))
        hit = self._od.get(key)
        if hit is not None:
            self._od.move_to_end(key)
            return hit
        # transform
        v = (base.vertices @ R.T) + t
        entry = _CacheEntry(vertices=v.astype(np.float64, copy=False), faces=base.faces)
        self._od[key] = entry
        if len(self._od) > self.maxsize:
            self._od.popitem(last=False)
        return entry

    def clear(self) -> None:
        self._od.clear()

    def invalidate_mesh(self, mesh_id: str) -> None:
        """Drop all cache entries derived from a mesh key (e.g. topology changed)."""
        to_del = [k for k in self._od.keys() if k[0] == mesh_id]
        for k in to_del:
            self._od.pop(k, None)


@dataclass
class RendererAdapter:
    backend: RenderBackend
    scene: Scene
    assets: AssetCatalog
    cache: _TransformCache = _TransformCache(maxsize=256)
    overlays: OverlayResolver | None = None

    # ----- public API -----
    def build(self, plan: PlanningState, coll: CollisionState | None = None) -> None:
        resolver = self._make_resolver(plan)
        hot = coll.hot if coll else frozenset()
        for node in self.scene.nodes.values():
            self._upsert_node(node, resolver, node.key in hot)

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

    def remove(self, node_ids: Iterable[str]) -> None:
        self.backend.remove(node_ids)

    def invalidate_mesh_key(self, mesh_key: str) -> None:
        """Call this if a base mesh topology changes (forces re-transform)."""
        self.cache.invalidate_mesh(mesh_key)

    # ----- internals -----
    def _make_resolver(self, plan: PlanningState) -> PoseResolver:
        return PoseResolver(
            scene=self.scene,
            plan=plan,
            get_pivot_for_asset=lambda key: getattr(
                self.assets.get_spec(key), "pivot_LPS", None
            ),
        )

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

        if isinstance(geom, MeshTransformable):
            base_mesh = geom.raw  # trimesh.Trimesh

            # LRU cache for dynamic probe meshes
            if node.extras.get("pose_source_probe"):
                entry = self.cache.get_or_compute(node.asset_key, base_mesh, R, t)
                v, f = entry.vertices, entry.faces
            else:
                v = (base_mesh.vertices @ R.T) + t
                f = base_mesh.faces

            if node.key in getattr(self.backend, "_handles", {}):
                self.backend.update_mesh(
                    node.key, vertices=v, indices=None, material=vm
                )
            else:
                self.backend.create_mesh(
                    node.key, name=node.key, vertices=v, indices=f, material=vm
                )

        elif isinstance(geom, PointsTransformable):
            pts = geom.transformed(R, t)

            if node.key in getattr(self.backend, "_handles", {}):
                self.backend.update_points(node.key, positions=pts, material=vm)
            else:
                self.backend.create_points(
                    node.key,
                    name=node.key,
                    positions=pts,
                    material=vm,
                    point_size=0.5,
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

        # repaint only nodes whose hot/cold status flipped
        nodes = [scene.nodes[nid] for nid in flips if nid in scene.nodes]
        if nodes:
            renderer_adapter.sync_nodes(
                plan, nodes
            )  # adapter reads overlays internally

    return _on_collisions_changed
