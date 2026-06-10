"""Per-kind probe headstage geometry derived from canonical meshes.

The placement optimizer needs cheap, tight clearance queries between
neighbouring probes' headstages. A single capsule per probe is far too
loose for the real PCB-plus-cable slab geometry: the manual T12 plan
on 836656 is feasible in practice yet flags a ``-2 mm`` "collision"
under the placeholder 2 mm-radius capsule.

This module solves the geometry side of the problem: from a
canonicalized probe mesh (in the asset's local LPS-mm frame, origin at
the shank-0 tip) it extracts the "body" region above the shanks and
builds a convex hull. The hull is in the same local frame, so the
runtime can later wrap it in an :class:`fcl.CollisionObject` and update
its transform per inner-loop iteration via ``setTransform``.

The convex hull is exact for AIND's flagship probes (Neuropixels 2.0
single/quad-base, customHolder, dovetail) whose bodies are convex
slabs. Pipettes and degenerate test fixtures gracefully fall back to
``None``, in which case the caller skips them from the clearance
constraint.

Why not a generic capsule? Benchmarked alternatives on the real
canonicalised meshes:

* OBB stack (N=8 cap): 80 µs per-pair distance, 1.4-1.9× looser than
  the hull.
* **Convex hull + FCL Convex / GJK**: 3.8 µs per-pair distance, exact
  for convex bodies.
* Mesh BVH: 603 µs per-pair distance, exact for arbitrary geometry.

The hull is both tightest and fastest. The decision is locked in.
"""

from __future__ import annotations

import fcl
import numpy as np
import trimesh
from numpy.typing import NDArray


def detect_body_region(
    mesh: trimesh.Trimesh,
    *,
    n_bins: int = 40,
    bbox_jump_factor: float = 4.0,
) -> tuple[float, NDArray[np.floating]]:
    """Detect where a canonical probe mesh transitions from shanks to body.

    Bins the mesh vertices along ``z`` and computes the xy-bbox area
    per bin. The "shank" region is identified by the lower half of
    bins (small bbox area, mostly empty); the body starts at the first
    bin whose area exceeds ``bbox_jump_factor`` times the shank-region
    median area.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Canonicalised probe mesh in local LPS-mm. The shank-0 tip is
        at the origin and shanks ascend in ``+z``.
    n_bins : int, optional
        Number of ``z`` bins for the area scan. Default ``40``.
    bbox_jump_factor : float, optional
        Multiplier on the shank-baseline bbox area defining the body
        transition. Default ``4.0``.

    Returns
    -------
    body_start_z : float
        ``z`` value of the bin centre at which the body region begins.
        ``z.max()`` when no jump is detected (degenerate / no body).
    body_vertices : ndarray, shape (M, 3)
        Mesh vertices with ``z >= body_start_z``.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.shape[0] == 0:
        return 0.0, np.empty((0, 3), dtype=np.float64)
    z = vertices[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        return float(z_max), vertices[z >= z_max]

    bins = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bbox_areas = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        mask = (z >= bins[i]) & (z < bins[i + 1])
        if int(mask.sum()) < 4:
            continue
        v = vertices[mask]
        dx = float(v[:, 0].max() - v[:, 0].min())
        dy = float(v[:, 1].max() - v[:, 1].min())
        bbox_areas[i] = dx * dy

    lower_half = bbox_areas[: n_bins // 2]
    valid = lower_half[~np.isnan(lower_half)]
    median_shank_area = float(np.median(valid)) if valid.size >= 3 else 0.0
    threshold = bbox_jump_factor * max(median_shank_area, 1e-6)

    body_start = z_max
    for i, area in enumerate(bbox_areas):
        if np.isnan(area):
            continue
        if area > threshold:
            body_start = float(bin_centers[i])
            break

    body_mask = z >= body_start
    return body_start, vertices[body_mask]


def build_headstage_hull(
    mesh: trimesh.Trimesh,
    *,
    min_body_verts: int = 10,
) -> trimesh.Trimesh | None:
    """Build the convex hull of a canonical probe mesh's body region.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Canonical probe mesh in local LPS-mm.
    min_body_verts : int, optional
        Minimum number of body-region vertices required to attempt
        hull construction. Default ``10``.

    Returns
    -------
    trimesh.Trimesh or None
        Convex hull of the body vertices in the same local frame as
        the input mesh. ``None`` when:

        * the body region has fewer than ``min_body_verts`` vertices,
        * the hull constructor raises (degenerate / coplanar geometry).
    """
    _, body_verts = detect_body_region(mesh)
    if body_verts.shape[0] < min_body_verts:
        return None
    try:
        hull = trimesh.convex.convex_hull(body_verts)
    except Exception:
        # qhull raises on degenerate inputs; fall back to "no hull".
        return None
    if hull is None or len(hull.vertices) < 4 or len(hull.faces) == 0:
        return None
    return hull


def make_fcl_convex(hull: trimesh.Trimesh) -> fcl.CollisionObject:
    """Wrap a trimesh convex hull as an :class:`fcl.CollisionObject`.

    The convex polygon array uses FCL's prefix-count format: each
    polygon is preceded by its vertex count (always 3 for triangulated
    hulls). The collision object is created with the identity
    transform; callers should call ``obj.setTransform(...)`` to place
    it in the world before distance queries.

    Parameters
    ----------
    hull : trimesh.Trimesh
        Convex hull mesh (e.g. the output of :func:`build_headstage_hull`).

    Returns
    -------
    fcl.CollisionObject
        Wrapping an :class:`fcl.Convex` geometry on the hull vertices.
    """
    verts = np.ascontiguousarray(hull.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(hull.faces, dtype=np.intc)
    polygons = np.empty(4 * len(faces), dtype=np.intc)
    for i, f in enumerate(faces):
        polygons[4 * i] = 3
        polygons[4 * i + 1 : 4 * i + 4] = f
    convex = fcl.Convex(verts, len(faces), polygons)
    tf = fcl.Transform(np.eye(3), np.zeros(3))
    return fcl.CollisionObject(convex, tf)


def make_fcl_bvh(mesh: trimesh.Trimesh) -> fcl.CollisionObject:
    """Wrap an arbitrary (potentially non-convex) trimesh as an
    :class:`fcl.CollisionObject` backed by a :class:`fcl.BVHModel`.

    Use this for exact probe-vs-probe clearance: the body-region
    convex hull misses the silicon-body / connector region between
    the shanks and the wide PCB headstage, and the convex hull of
    the *whole* probe over-estimates collisions in concavities. The
    BVH gives the true signed distance via FCL's mesh-mesh GJK.

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
