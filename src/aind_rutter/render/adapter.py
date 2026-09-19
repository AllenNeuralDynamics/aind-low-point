"""The render backend protocol, and the adapter that drives it from a scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Protocol, Set

import numpy as np

from aind_rutter.collision.adapter import CollisionState
from aind_rutter.domain.catalog import AssetCatalog, Material
from aind_rutter.domain.plan import PlanningState, reconcile_probe_assets
from aind_rutter.domain.pose import PoseResolver
from aind_rutter.domain.scene import NodeInstance, Scene
from aind_rutter.domain.transforms import MeshTransformable, PointsTransformable
from aind_rutter.render.overlays import (
    OverlayResolver,
    OverlaySpec,
    OverlayState,
    ViewMaterial,
    material_to_view,
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
    def highlight(
        self,
        node_id: str | None,
        *,
        color: str = "#ffffff",
        width: float = 3.0,
    ) -> None: ...


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
    # Which asset each drawn node was built from. A node whose asset changed
    # needs its handle rebuilt, and update_mesh replaces points without faces.
    _asset_of: dict[str, str] = field(default_factory=dict)

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
        reconcile_probe_assets(self.scene, plan, self.assets)
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

        # A node pointing at a different asset than the one it was drawn from
        # has to be recreated: update_mesh takes new points but not new faces.
        if self._asset_of.get(node.key, node.asset_key) != node.asset_key:
            self.backend.remove([node.key])
        self._asset_of[node.key] = node.asset_key

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
