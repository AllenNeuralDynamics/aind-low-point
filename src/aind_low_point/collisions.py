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

from aind_low_point.assets import (
    AssetCatalog,
)
from aind_low_point.core import Pair
from aind_low_point.planning import PlanningState
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
    def remove(self, node_ids: Iterable[str]) -> None: ...
    def collide_internal(
        self, *, enable_contacts: bool, max_contacts: int
    ) -> List[CollisionPair]: ...
    def collide_one_to_many(
        self, spec: ObjSpec, *, enable_contacts: bool, max_contacts: int
    ) -> List[CollisionPair]: ...


def default_include(node: NodeInstance) -> bool:
    # Exclude brain + structures; include only meshes
    if node.geom.kind != "mesh":
        return False
    if node.geom.key == "brain" or node.geom.key.startswith("structure:"):
        return False
    return True


@dataclass
class CollisionAdapter:
    backend: CollisionBackend
    scene: Scene
    assets: AssetCatalog
    include: Callable[[NodeInstance], bool] = default_include

    # ---- lifecycle wiring ----
    def rebuild(self, plan: PlanningState) -> None:
        specs = [
            s
            for n in self.scene.nodes.values()
            if self.include(n)
            for s in [self._spec_for_node(n, plan)]
            if s
        ]
        self.backend.rebuild(specs)

    def on_store_change(
        self, plan: PlanningState, changed_probe_names: List[str]
    ) -> None:
        # Only probes move; map probe names -> scene nodes
        nodes: List[NodeInstance] = []
        for pname in changed_probe_names:
            nid = f"probe:{pname}"
            node = self.scene.nodes.get(nid)
            if node and self.include(node):
                nodes.append(node)
        if not nodes:
            return
        specs = [self._spec_for_node(n, plan) for n in nodes]
        self.backend.sync([s for s in specs if s is not None])

    def remove_nodes(self, node_ids: Iterable[str]) -> None:
        self.backend.remove(node_ids)

    # ---- queries (pass-through to backend) ----
    def collide_internal(
        self, *, enable_contacts: bool = True, max_contacts: int = 8
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

    # ---- domain → backend spec ----
    def _spec_for_node(
        self, node: NodeInstance, plan: PlanningState
    ) -> Optional[ObjSpec]:
        if node.geom.kind != "mesh":
            return None
        base = self._resolve_mesh(node.geom.key, self.assets)
        bvh = _bvh_from_mesh(base, name=node.geom.key)  # unique geometry per node
        R, t = self._pose_for_node(node, plan)
        tf = _rt_to_transform(R, t, name=f"pose:{node.id}")
        return ObjSpec(node_id=node.id, geom=bvh, transform=tf)

    def _pose_for_node(
        self, node: NodeInstance, plan: PlanningState
    ) -> Tuple[np.ndarray, np.ndarray]:
        if node.id.startswith("probe:"):
            pname = node.id.split(":", 1)[1]
            return plan.probes[pname].pose.chain().composed_transform
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    def _resolve_mesh(self, key: str, a: AssetCatalog) -> trimesh.Trimesh:
        if key == "implant":
            return a.implant_mesh.raw
        if key == "headframe":
            return a.headframe_mesh.raw
        if key == "well":
            return a.well_mesh.raw
        if key == "cone":
            return a.cone_mesh.raw
        if key.startswith("hole:"):
            return a.hole_models[int(key.split(":", 1)[1])].raw
        if key.startswith("probe:"):
            return a.probe_models[key.split(":", 1)[1]].raw  # TYPE, not name
        # brain/structures are intentionally excluded by include()
        raise KeyError(f"Unknown mesh key for collision: {key}")


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
        # keep backend up-to-date for moved probes
        moved = [pid for pid in changed_ids if f"probe:{pid}" in self.scene.nodes]
        if moved:
            self.adapter.on_store_change(plan, moved)

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
