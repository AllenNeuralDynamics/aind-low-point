"""Logic to do collision testing"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
)

import fcl
import numpy as np
import trimesh

from aind_low_point.assets import AssetCatalog
from aind_low_point.common import Capability
from aind_low_point.core import MeshTransformable, Pair
from aind_low_point.planning import PlanningState, PoseResolver
from aind_low_point.scene import NodeInstance, Scene


## Collision detection
# ---- result types ----
@dataclass(frozen=True)
class Contact:
    position: np.ndarray  # (3,), float64
    normal: np.ndarray  # (3,), float64 (from o1 into o2)
    penetration_depth: float


@dataclass(frozen=True)
class CollisionPair:
    id1: str
    id2: str
    contacts: Tuple[Contact, ...]  # empty if enable_contact=False


# ---- specs the backend accepts (domain-free) ----
@dataclass(frozen=True)
class ObjSpec:
    node_id: str
    geom: fcl.CollisionGeometry  # already built (BVH, box, etc.)
    transform: fcl.Transform  # pose in world coords (LPS)
    group: int = 0  # collision group bitmask
    mask: int = 0  # which groups this object collides with


@dataclass(slots=True)
class CollisionState:
    # all pairs in collision (sorted tuple so (a,b)==(b,a))
    pairs: FrozenSet[Pair] = field(default_factory=frozenset)
    # convenience: any node that participates in *any* collision
    hot: FrozenSet[str] = field(default_factory=frozenset)

    def replace(self, pairs: set[Pair]) -> "CollisionState":
        spairs = frozenset(tuple(sorted(p)) for p in pairs)
        hot = frozenset({nid for p in spairs for nid in p})
        return CollisionState(pairs=spairs, hot=hot)


class CollisionBackend(Protocol):
    def rebuild(self, specs: Iterable[ObjSpec]) -> None: ...
    def sync(self, specs: Iterable[ObjSpec]) -> None: ...
    def update_transforms(
        self, transforms: Iterable[Tuple[str, "fcl.Transform"]]
    ) -> None: ...
    def remove(self, node_ids: Iterable[str]) -> None: ...
    def collide_internal(
        self, *, enable_contacts: bool, max_contacts: int
    ) -> List[CollisionPair]: ...
    def collide_one_to_many(
        self, spec: ObjSpec, *, enable_contacts: bool, max_contacts: int
    ) -> List[CollisionPair]: ...


def default_include(node: NodeInstance, catalog: AssetCatalog) -> bool:
    spec = catalog.get_spec(node.asset_key)
    return spec.kind == "mesh" and bool(spec.caps & Capability.COLLIDABLE)


@dataclass
class CollisionAdapter:
    backend: CollisionBackend
    scene: Scene
    assets: AssetCatalog
    include: Callable[[NodeInstance, AssetCatalog], bool] = default_include

    # ---- lifecycle wiring ----
    def rebuild(self, plan: PlanningState) -> None:
        resolver = self._make_resolver(plan)
        specs = [
            s
            for n in self.scene.nodes.values()
            if self.include(n, self.assets)
            for s in [self._spec_for_node(n, resolver)]
            if s
        ]
        self.backend.rebuild(specs)

    def on_store_change(
        self, plan: PlanningState, changed_probe_names: List[str]
    ) -> None:
        resolver = self._make_resolver(plan)
        nodes: List[NodeInstance] = []
        for pname in changed_probe_names:
            nid = f"probe:{pname}"
            node = self.scene.nodes.get(nid)
            if node and self.include(node, self.assets):
                nodes.append(node)
        if not nodes:
            return
        specs = [self._spec_for_node(n, resolver) for n in nodes]
        self.backend.sync([s for s in specs if s is not None])

    def update_probe_transforms(
        self, plan: PlanningState, changed_probe_names: List[str]
    ) -> None:
        """Transform-only update for moved probes. No BVH rebuild."""
        resolver = self._make_resolver(plan)
        transforms: List[Tuple[str, fcl.Transform]] = []
        for pname in changed_probe_names:
            nid = f"probe:{pname}"
            node = self.scene.nodes.get(nid)
            if node and self.include(node, self.assets):
                R, t = resolver.world_rt_for_node(node)
                tf = _rt_to_transform(R, t, name=f"pose:{nid}")
                transforms.append((nid, tf))
        if transforms:
            self.backend.update_transforms(transforms)

    def remove_nodes(self, node_ids: Iterable[str]) -> None:
        self.backend.remove(node_ids)

    # ---- queries (pass-through to backend) ----
    def collide_internal(
        self, *, enable_contacts: bool = True, max_contacts: int = 100
    ) -> List[CollisionPair]:
        return self.backend.collide_internal(
            enable_contacts=enable_contacts, max_contacts=max_contacts
        )

    def collide_one_to_many(
        self, mesh: trimesh.Trimesh, R: np.ndarray, t: np.ndarray, *, name: str
    ) -> List[CollisionPair]:
        spec = ObjSpec(
            node_id=name,
            geom=_bvh_from_mesh(mesh, name=name),
            transform=_rt_to_transform(R, t, name=f"pose:{name}"),
        )
        return self.backend.collide_one_to_many(
            spec, enable_contacts=True, max_contacts=8
        )

    # ---- internals ----
    def _make_resolver(self, plan: PlanningState) -> PoseResolver:
        # ``catalog`` flows through to ProbePose so each probe's
        # ``pivot_LPS`` shows up in pose.tip (recording-array center
        # at target). The legacy ``get_pivot_for_asset`` callback is
        # left at its no-op default — the pivot is already baked.
        return PoseResolver(
            scene=self.scene, plan=plan, catalog=self.assets
        )

    def _spec_for_node(
        self, node: NodeInstance, resolver: PoseResolver
    ) -> Optional[ObjSpec]:
        geom = self.assets.get_geometry(node.asset_key)
        if not isinstance(geom, MeshTransformable):
            return None
        base = geom.raw
        bvh = _bvh_from_mesh(base, name=node.asset_key)
        R, t = resolver.world_rt_for_node(node)
        tf = _rt_to_transform(R, t, name=f"pose:{node.key}")
        spec = self.assets.get_spec(node.asset_key)
        return ObjSpec(
            node_id=node.key,
            geom=bvh,
            transform=tf,
            group=spec.collidable_group,
            mask=spec.collidable_mask,
        )


def objects_in_collision(collision_pairs: List[CollisionPair]) -> List[Tuple[str, str]]:
    objects_in_collision = set()
    for coll_pair in collision_pairs:
        objects_in_collision.add((coll_pair.id1, coll_pair.id2))
    return list(objects_in_collision)


def _diff_hot(curr: CollisionState, prev: CollisionState | None = None) -> set[str]:
    # nodes that flipped collision state
    if prev is None:
        return set(curr.hot)
    return set((prev.hot - curr.hot) | (curr.hot - prev.hot))


def _ensure_fcl_arrays(
    mesh: trimesh.Trimesh, *, name: str
) -> tuple[np.ndarray, np.ndarray]:
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise FCLInputError(f"{name}: vertices must be (N,3)")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3 or mesh.faces.size == 0:
        raise FCLInputError(f"{name}: faces must be (M,3) and non-empty")
    v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    f = np.ascontiguousarray(mesh.faces, dtype=np.int32)  # FCL wants int32
    if not np.isfinite(v).all():
        raise FCLInputError(f"{name}: NaN/Inf in vertices")
    vmax = v.shape[0] - 1
    if f.min() < 0 or f.max() > vmax:
        raise FCLInputError(f"{name}: face index out of range [0..{vmax}]")
    return v, f


def _bvh_from_mesh(mesh: trimesh.Trimesh, *, name: str) -> fcl.BVHModel:
    v, f = _ensure_fcl_arrays(mesh, name=name)
    m = fcl.BVHModel()
    m.beginModel(v.shape[0], f.shape[0])
    m.addSubModel(v, f)
    m.endModel()
    return m


def _rt_to_transform(R: np.ndarray, t: np.ndarray, *, name: str) -> fcl.Transform:
    R = np.ascontiguousarray(R, dtype=np.float64).reshape(3, 3)
    t = np.ascontiguousarray(t, dtype=np.float64).reshape(
        3,
    )
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise FCLInputError(f"{name}: non-finite R/t")
    return fcl.Transform(R, t)


class FCLInputError(ValueError):
    pass


@dataclass
class CollisionHandler:
    scene: Scene
    adapter: CollisionAdapter
    state: CollisionState = field(default_factory=CollisionState)
    on_state_changed: Optional[
        Callable[[CollisionState, Set[str], PlanningState], None]
    ] = None
    _prev_state: CollisionState | None = None

    def __call__(self, plan: PlanningState, changed_ids: List[str]) -> None:
        # keep backend up-to-date for moved probes (transform-only, no BVH rebuild)
        moved = [pid for pid in changed_ids if f"probe:{pid}" in self.scene.nodes]
        if moved:
            self.adapter.update_probe_transforms(plan, moved)

        # recompute collisions
        pairs = self.adapter.collide_internal(enable_contacts=False)
        new_pairs = {(p.id1, p.id2) for p in pairs}
        new_state = self.state.replace(new_pairs)

        # notify only if something flipped
        flips = _diff_hot(new_state, self._prev_state)
        self._prev_state = new_state
        self.state = new_state
        if flips and self.on_state_changed:
            self.on_state_changed(new_state, flips, plan)

    # --- factored methods for async worker ---

    def prepare(
        self, plan: PlanningState, changed_ids: List[str]
    ) -> List[Tuple[str, fcl.Transform]]:
        """Main thread: compute (node_id, fcl.Transform) pairs."""
        resolver = self.adapter._make_resolver(plan)
        transforms: List[Tuple[str, fcl.Transform]] = []
        for pid in changed_ids:
            nid = f"probe:{pid}"
            node = self.scene.nodes.get(nid)
            if node and self.adapter.include(node, self.adapter.assets):
                R, t = resolver.world_rt_for_node(node)
                tf = _rt_to_transform(R, t, name=f"pose:{nid}")
                transforms.append((nid, tf))
        return transforms

    def work(
        self, transforms: List[Tuple[str, fcl.Transform]]
    ) -> Tuple[CollisionState, Set[str]]:
        """Worker thread: update transforms + run collision detection."""
        if transforms:
            self.adapter.backend.update_transforms(transforms)
        pairs = self.adapter.collide_internal(
            enable_contacts=False,
        )
        new_pairs = {(p.id1, p.id2) for p in pairs}
        new_state = self.state.replace(new_pairs)
        flips = _diff_hot(new_state, self._prev_state)
        self._prev_state = new_state
        self.state = new_state
        return (new_state, flips)

    def deliver(
        self,
        result: Tuple[CollisionState, Set[str]],
        plan: PlanningState,
    ) -> None:
        """Main thread: update overlays if collision state changed."""
        new_state, flips = result
        if flips and self.on_state_changed:
            self.on_state_changed(new_state, flips, plan)
