"""Building FCL geometry and transforms from meshes, with the input guards.

FCL segfaults on a malformed vertex or face array rather than raising, so every
mesh crosses this boundary through a check first.
"""

from __future__ import annotations

import fcl
import numpy as np
import trimesh


class FCLInputError(ValueError):
    """A mesh or pose FCL cannot be given."""


def ensure_fcl_arrays(
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


def bvh_from_mesh(mesh: trimesh.Trimesh, *, name: str) -> fcl.BVHModel:
    v, f = ensure_fcl_arrays(mesh, name=name)
    m = fcl.BVHModel()
    m.beginModel(v.shape[0], f.shape[0])
    m.addSubModel(v, f)
    m.endModel()
    return m


def rt_to_fcl_transform(R: np.ndarray, t: np.ndarray, *, name: str) -> fcl.Transform:
    R = np.ascontiguousarray(R, dtype=np.float64).reshape(3, 3)
    t = np.ascontiguousarray(t, dtype=np.float64).reshape(
        3,
    )
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise FCLInputError(f"{name}: non-finite R/t")
    return fcl.Transform(R, t)
