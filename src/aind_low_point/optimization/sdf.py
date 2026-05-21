"""Signed Distance Field (SDF) generation + on-disk caching.

For pairwise probe clearance in the optimizer we need a *smooth*
signed-distance function with reliable gradients everywhere — including
through overlap, where FCL's BVH-vs-BVH GJK distance clamps to zero and
its witness-point gradient becomes noise. Precomputing the SDF as a
uniform voxel grid solves this: at inner-loop time we trilinearly
interpolate, getting a continuous signed distance and (via finite-diff
on the grid or analytic on the interp formula) a smooth gradient.

This module just builds and caches the grids. Lookup happens in
:mod:`aind_low_point.optimization.sdf_jax` (separate to keep JAX
imports out of the path that just needs a grid).

Generation uses ``libigl``'s pseudonormal SDF — typically 0.2–0.3 μs
per query point, so a 5 M-voxel grid for a single probe builds in
~1 s. Caches to ``~/.cache/aind_low_point/sdfs/`` keyed by mesh hash
+ spacing so we only pay once per (mesh, resolution) tuple.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import igl
import numpy as np
import trimesh
from numpy.typing import NDArray

DEFAULT_SPACING_MM: float = 0.2
DEFAULT_PAD_MM: float = 2.0


@dataclass(frozen=True)
class ProbeSDF:
    """Pre-computed SDF grid for one probe mesh (canonical local frame).

    The grid is sampled on a regular axis-aligned lattice:
    ``world_local_pt = origin + i * spacing`` for each integer index
    ``i`` along the three axes. Look up at a continuous local-frame
    point ``p`` by trilinear interpolation in
    :mod:`aind_low_point.optimization.sdf_jax`.

    Outside the grid bbox the SDF should be treated as ``+spacing * 10``
    or similar large positive — the probe is "definitely far" — since
    we don't bother computing SDF in empty regions of the bbox.

    ``surface_points`` is an ``(N, 3)`` set of points sampled
    uniformly on the mesh surface. Used as the query set in pairwise
    clearance — for a pair ``(a, b)``, we transform ``b``'s surface
    points into ``a``'s local frame and look up ``a``'s SDF at those
    points (and symmetrise).
    """

    grid: NDArray[np.floating]  # (Nx, Ny, Nz) float32
    origin: NDArray[np.floating]  # (3,) world local pt at grid[0,0,0]
    spacing: float  # voxel edge length (mm)
    surface_points: NDArray[np.floating]  # (N, 3) canonical local

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.grid.shape)  # type: ignore[return-value]

    @property
    def bbox_min(self) -> NDArray[np.floating]:
        return np.asarray(self.origin, dtype=np.float64)

    @property
    def bbox_max(self) -> NDArray[np.floating]:
        return self.bbox_min + (np.asarray(self.shape) - 1) * self.spacing


def _mesh_hash(mesh: trimesh.Trimesh) -> str:
    """Stable content hash of (vertices, faces). Used as cache key."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _cache_dir() -> Path:
    root = os.environ.get("AIND_LOW_POINT_CACHE_DIR")
    if root:
        return Path(root) / "sdfs"
    return Path.home() / ".cache" / "aind_low_point" / "sdfs"


def _cache_path(
    mesh: trimesh.Trimesh,
    *,
    spacing_mm: float,
    pad_mm: float,
    n_surface_points: int,
) -> Path:
    key = f"{_mesh_hash(mesh)}_s{spacing_mm}_p{pad_mm}_n{n_surface_points}"
    return _cache_dir() / f"{key}.npz"


def build_probe_sdf(
    mesh: trimesh.Trimesh,
    *,
    spacing_mm: float = DEFAULT_SPACING_MM,
    pad_mm: float = DEFAULT_PAD_MM,
    n_surface_points: int = 5000,
    use_cache: bool = True,
) -> ProbeSDF:
    """Build (or load) the SDF for one probe mesh.

    Parameters
    ----------
    mesh
        Probe mesh in canonical local frame (shank-0 tip at origin).
    spacing_mm
        Voxel edge length. ``0.2 mm`` is a good default — fine enough
        for sub-mm clearance reading on probe-body features, generates
        in ~4 s, ~67 MB memory.
    pad_mm
        Padding around the mesh bbox. The SDF returns large positive
        values near the bbox edges, so query points "in the padding"
        get sensible (positive) distances.
    n_surface_points
        Surface-point sample count for the pairwise clearance query
        set. Stored alongside the grid in the same cache record.
    use_cache
        When True (default), reuse the on-disk cache if present and
        re-write after a fresh build. Disable for debugging.
    """
    cpath = _cache_path(
        mesh, spacing_mm=spacing_mm, pad_mm=pad_mm,
        n_surface_points=n_surface_points,
    )
    if use_cache:
        if cpath.exists():
            with np.load(cpath) as data:
                return ProbeSDF(
                    grid=data["grid"].astype(np.float32),
                    origin=data["origin"].astype(np.float64),
                    spacing=float(data["spacing"]),
                    surface_points=data["surface_points"].astype(np.float64),
                )

    V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    F = np.ascontiguousarray(mesh.faces, dtype=np.int64)

    bbox_min = mesh.bounds[0] - pad_mm
    bbox_max = mesh.bounds[1] + pad_mm
    dims = np.ceil((bbox_max - bbox_min) / spacing_mm).astype(int) + 1
    # Build lattice. ``meshgrid(indexing='ij')`` gives (Nx, Ny, Nz, 3)
    # when stacked along the last axis.
    ax = [np.arange(int(dims[i])) * spacing_mm + bbox_min[i] for i in range(3)]
    grid_pts = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)

    # FAST_WINDING_NUMBER handles non-watertight meshes correctly;
    # PSEUDONORMAL returns wrong signs for non-watertight (which our
    # probe meshes are — connectors and cables leave open boundaries).
    S, _I, _C, _N = igl.signed_distance(
        grid_pts.astype(np.float64),
        V,
        F,
        sign_type=igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
    )
    sdf_grid = S.reshape(tuple(int(d) for d in dims)).astype(np.float32)

    # Surface-point sample set for pairwise clearance queries.
    surface_pts, _face_idx = trimesh.sample.sample_surface(mesh, n_surface_points)
    surface_pts = np.asarray(surface_pts, dtype=np.float64)

    out = ProbeSDF(
        grid=sdf_grid,
        origin=np.asarray(bbox_min, dtype=np.float64),
        spacing=float(spacing_mm),
        surface_points=surface_pts,
    )

    if use_cache:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cpath,
            grid=sdf_grid,
            origin=out.origin,
            spacing=np.array(spacing_mm, dtype=np.float64),
            surface_points=surface_pts,
        )
    return out
