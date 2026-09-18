"""FCL collision objects for probe and fixture meshes."""

from __future__ import annotations

import fcl
import numpy as np
import trimesh


def make_fcl_bvh(mesh: trimesh.Trimesh) -> fcl.CollisionObject:
    """Wrap a trimesh as an :class:`fcl.CollisionObject` over a BVH.

    A BVH costs 603 µs per pair against a convex hull's 3.8 µs, and buys
    the concavity between the shanks and the PCB: a hull of the whole
    probe collides there, a hull of the body alone misses the connector.

    Identity transform; callers must ``setTransform(...)`` per pose.
    """
    v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    f = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    bvh = fcl.BVHModel()
    bvh.beginModel(v.shape[0], f.shape[0])
    bvh.addSubModel(v, f)
    bvh.endModel()
    tf = fcl.Transform(np.eye(3), np.zeros(3))
    return fcl.CollisionObject(bvh, tf)
