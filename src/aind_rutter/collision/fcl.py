"""FCL backend for collision testing"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import (
    Iterable,
    List,
    Optional,
    Tuple,
)

import fcl
import numpy as np

from aind_rutter.collision.adapter import (
    CollisionBackend,
    CollisionPair,
    Contact,
    ObjSpec,
)


@dataclass
class FCLBackend(CollisionBackend):
    _mgr: fcl.DynamicAABBTreeCollisionManager = field(
        default_factory=fcl.DynamicAABBTreeCollisionManager
    )
    _node_to_obj: dict[str, fcl.CollisionObject] = field(default_factory=dict)
    _node_to_geom: dict[str, fcl.CollisionGeometry] = field(
        default_factory=dict
    )  # keep geom alive so id() stays stable
    _geomid_to_node: dict[int, str] = field(
        default_factory=dict
    )  # id(CollisionGeometry) -> node_id
    _node_to_geomid: dict[str, int] = field(
        default_factory=dict
    )  # node_id -> id(CollisionGeometry)
    # group/mask bits for collision filtering (group & mask must overlap)
    _node_to_group: dict[str, int] = field(default_factory=dict)
    _node_to_mask: dict[str, int] = field(default_factory=dict)
    # One manager, two threads: the async worker updates transforms and runs
    # the collision pass while the main thread may be rebuilding after a probe
    # kind change. Reentrant so a future nested call cannot deadlock.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def _record(self, s: ObjSpec) -> fcl.CollisionObject:
        """Register one spec in every map a later query reads.

        ``rebuild`` and ``sync`` both add objects, and when ``sync`` recorded
        fewer maps than ``rebuild`` the node it added was invisible to the
        group/mask filter and so never reported as colliding.
        """
        geom_id = id(s.geom)
        cob = fcl.CollisionObject(s.geom, s.transform)
        self._node_to_obj[s.node_id] = cob
        # Holding the geometry is what keeps ``geom_id`` from being recycled
        # onto a different object while ``_geomid_to_node`` still maps it.
        self._node_to_geom[s.node_id] = s.geom
        self._geomid_to_node[geom_id] = s.node_id
        self._node_to_geomid[s.node_id] = geom_id
        self._node_to_group[s.node_id] = s.group
        self._node_to_mask[s.node_id] = s.mask
        return cob

    def _forget(self, node_id: str) -> fcl.CollisionObject | None:
        cob = self._node_to_obj.pop(node_id, None)
        self._node_to_geom.pop(node_id, None)
        self._node_to_group.pop(node_id, None)
        self._node_to_mask.pop(node_id, None)
        geom_id = self._node_to_geomid.pop(node_id, None)
        if geom_id is not None:
            self._geomid_to_node.pop(geom_id, None)
        return cob

    def rebuild(self, specs: Iterable[ObjSpec]) -> None:
        with self._lock:
            self._mgr.clear()
            self._node_to_obj.clear()
            self._node_to_geom.clear()
            self._geomid_to_node.clear()
            self._node_to_geomid.clear()
            self._node_to_group.clear()
            self._node_to_mask.clear()

            objs = [self._record(s) for s in specs]
            if objs:
                self._mgr.registerObjects(objs)
            self._mgr.setup()

    def sync(self, specs: Iterable[ObjSpec]) -> None:
        with self._lock:
            for s in specs:
                cob = self._node_to_obj.get(s.node_id)
                if cob is None:
                    self._mgr.registerObject(self._record(s))
                else:
                    # pose update only (geometry assumed same)
                    cob.setTransform(s.transform)
                    self._mgr.update(cob)
            self._mgr.update()

    def update_transforms(
        self, transforms: Iterable[Tuple[str, fcl.Transform]]
    ) -> None:
        """Update poses for existing nodes. Skips unknown node_ids."""
        with self._lock:
            for node_id, tf in transforms:
                cob = self._node_to_obj.get(node_id)
                if cob is not None:
                    cob.setTransform(tf)
                    self._mgr.update(cob)
            self._mgr.update()

    def remove(self, node_ids: Iterable[str]) -> None:
        with self._lock:
            for nid in node_ids:
                cob = self._forget(nid)
                if cob is not None:
                    self._mgr.unregisterObject(cob)
            self._mgr.update()

    # ---- queries ----
    def collide_internal(
        self, *, enable_contacts: bool, max_contacts: int
    ) -> List[CollisionPair]:
        gid_map = dict(self._geomid_to_node)
        group_map = self._node_to_group
        mask_map = self._node_to_mask
        found_pairs: List[Tuple[str, str]] = []

        def _cb(o1, o2, _cdata):
            req = fcl.CollisionRequest(enable_contact=False, num_max_contacts=1)
            res = fcl.CollisionResult()
            fcl.collide(o1, o2, req, res)
            if res.is_collision and res.contacts:
                c = res.contacts[0]
                n1 = gid_map.get(id(c.o1))
                n2 = gid_map.get(id(c.o2))
                if n1 is None or n2 is None:
                    return False
                # Group/mask filter: n1's mask must include n2's group
                # and vice versa
                g1, m1 = group_map.get(n1, 0), mask_map.get(n1, 0)
                g2, m2 = group_map.get(n2, 0), mask_map.get(n2, 0)
                if (m1 & g2) and (m2 & g1):
                    k = (n1, n2) if n1 <= n2 else (n2, n1)
                    found_pairs.append(k)
            return False  # never stop early

        with self._lock:
            self._mgr.collide(fcl.CollisionData(), _cb)
        return [CollisionPair(id1=a, id2=b, contacts=()) for a, b in set(found_pairs)]

    def collide_one_to_many(
        self, spec: ObjSpec, *, enable_contacts: bool, max_contacts: int
    ) -> List[CollisionPair]:
        req = fcl.CollisionRequest(
            enable_contact=bool(enable_contacts), num_max_contacts=int(max_contacts)
        )
        cdata = fcl.CollisionData(request=req)
        ext = fcl.CollisionObject(spec.geom, spec.transform)
        with self._lock:
            self._mgr.collide(ext, cdata, fcl.defaultCollisionCallback)
        # add a temporary mapping for the external object, using its geometry id
        ext_name_map = {id(ext.collision_geometry): spec.node_id}
        return self._pairs_from_contacts(cdata.result.contacts, extra_map=ext_name_map)

    # ---- helpers ----
    def _pairs_from_contacts(
        self,
        contacts: Iterable[fcl.Contact],
        *,
        extra_map: Optional[dict[int, str]] = None,
    ) -> List[CollisionPair]:
        gid_to_name: dict[int, str] = dict(self._geomid_to_node)
        if extra_map:
            gid_to_name.update(extra_map)

        groups: dict[Tuple[str, str], List[Contact]] = {}
        for c in contacts:
            n1 = gid_to_name.get(id(c.o1))
            n2 = gid_to_name.get(id(c.o2))
            if n1 is None or n2 is None:
                continue
            k = (n1, n2) if n1 <= n2 else (n2, n1)
            cc = Contact(
                position=np.asarray(c.pos, dtype=np.float64),
                normal=np.asarray(c.normal, dtype=np.float64),
                penetration_depth=float(c.penetration_depth),
            )
            groups.setdefault(k, []).append(cc)

        return [
            CollisionPair(id1=a, id2=b, contacts=tuple(cs))
            for (a, b), cs in groups.items()
        ]


# ---- simple guards at the fcl boundary ----
