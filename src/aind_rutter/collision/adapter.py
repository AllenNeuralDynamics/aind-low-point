"""Keeping an FCL world in step with the scene, and asking it what overlaps.

Which pairs get tested is a rule over `collidable` and `role`, not per-asset
labels: both sides collidable and at least one a probe. See `pair_bits`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
)

import fcl
import numpy as np
import trimesh

from aind_rutter.collision.geometry import bvh_from_mesh, rt_to_fcl_transform
from aind_rutter.domain.catalog import AssetCatalog
from aind_rutter.domain.enums import Role
from aind_rutter.domain.plan import PlanningState, probe_node_id, reconcile_probe_assets
from aind_rutter.domain.pose import PoseResolver
from aind_rutter.domain.scene import NodeInstance, Scene
from aind_rutter.domain.transforms import MeshTransformable, Pair


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


# The pair filter, as two bits. A pair is tested when both sides are collidable
# and at least one is a probe: probes hit fixtures and each other, fixtures do
# not hit each other. This replaced per-asset group/mask labels, which across
# every subject config only ever expressed these two patterns.
_PROBE = 1
_FIXTURE = 2


def pair_bits(spec) -> tuple[int, int]:
    """``(group, mask)`` for one asset; the backend tests a pair when each
    side's mask admits the other's group."""
    if not spec.collidable:
        return 0, 0
    if spec.role is Role.PROBE:
        return _PROBE, _PROBE | _FIXTURE
    return _FIXTURE, _PROBE


def default_include(node: NodeInstance, catalog: AssetCatalog) -> bool:
    spec = catalog.get_spec(node.asset_key)
    return spec.kind == "mesh" and spec.collidable


@dataclass
class CollisionAdapter:
    backend: CollisionBackend
    scene: Scene
    assets: AssetCatalog
    include: Callable[[NodeInstance, AssetCatalog], bool] = default_include
    # Which asset each registered node was built from. The backend's sync
    # updates a known node's pose and keeps its geometry, so a node whose asset
    # changed has to be dropped before it is synced again.
    _asset_of: dict[str, str] = field(default_factory=dict)

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
        reconcile_probe_assets(self.scene, plan, self.assets)
        resolver = self._make_resolver(plan)
        nodes: List[NodeInstance] = []
        for pname in changed_probe_names:
            node = self.scene.nodes.get(probe_node_id(pname))
            if node and self.include(node, self.assets):
                nodes.append(node)
        if not nodes:
            return
        self.remove_nodes(
            [
                n.key
                for n in nodes
                if self._asset_of.get(n.key, n.asset_key) != n.asset_key
            ]
        )
        for node in nodes:
            self._asset_of[node.key] = node.asset_key
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
                tf = rt_to_fcl_transform(R, t, name=f"pose:{nid}")
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
            geom=bvh_from_mesh(mesh, name=name),
            transform=rt_to_fcl_transform(R, t, name=f"pose:{name}"),
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
        return PoseResolver(scene=self.scene, plan=plan, catalog=self.assets)

    def _spec_for_node(
        self, node: NodeInstance, resolver: PoseResolver
    ) -> Optional[ObjSpec]:
        geom = self.assets.get_geometry(node.asset_key)
        if not isinstance(geom, MeshTransformable):
            return None
        base = geom.raw
        bvh = bvh_from_mesh(base, name=node.asset_key)
        R, t = resolver.world_rt_for_node(node)
        tf = rt_to_fcl_transform(R, t, name=f"pose:{node.key}")
        spec = self.assets.get_spec(node.asset_key)
        group, mask = pair_bits(spec)
        return ObjSpec(
            node_id=node.key,
            geom=bvh,
            transform=tf,
            group=group,
            mask=mask,
        )
