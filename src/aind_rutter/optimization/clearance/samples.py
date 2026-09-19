"""Coarse-to-fine surface samples for probe body clearance queries.

A body clearance query looks up one probe's surface points in the other probe's SDF
and keeps the minimum. Area-uniform samples miss the closest point on small convex
features by up to half a millimetre, so samples here go where first contact can
happen, and are grouped for a two-pass query:

- fine points, drawn by area × convex curvature × closeness to the local convex
  hull, most of them in the band just above the shanks;
- coarse points (evenly spaced, plus a subset of the fine points), each owning the
  fine points nearest to it — a cell — with the cell's radius.

A query looks up the coarse points, ranks cells by ``coarse value − radius`` and looks
up the fine points of the best cells. No fine point in a cell reads below that bound,
because a distance field changes no faster than its query point moves.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError, cKDTree

from aind_rutter.optimization.clearance.voxel_sdf import (
    SURFACE_SAMPLE_SEED,
    _cache_dir,
    _mesh_hash,
)


@dataclass(frozen=True)
class SampleParams:
    """Sampling rules; every field is part of the cache key."""

    fine_n: int = 32_000
    band_mm: float = 10.0  # contact band: this far above the body's lowest point
    band_share: float = 0.8
    coarse_n: int = 1_000
    coarse_even: int = 500
    curvature_radius_mm: float = 0.5
    curvature_ref: float = 1.0  # 1/mm; weight saturates relative to this
    curvature_floor: float = 0.05
    hull_window_mm: float = 15.0
    hull_bin_mm: float = 5.0
    hull_scale_mm: float = 0.25
    hull_floor: float = 0.02
    seed: int = SURFACE_SAMPLE_SEED


@dataclass(frozen=True)
class ClearanceSamples:
    """Fine and coarse surface points in the probe's local frame, and the cells
    linking them. ``cells`` rows list fine indices padded with ``len(fine)``."""

    fine: NDArray[np.float32]  # (F, 3)
    coarse: NDArray[np.float32]  # (C, 3)
    cells: NDArray[np.int32]  # (C, W)
    radius: NDArray[np.float32]  # (C,) farthest cell member from its coarse point


def convex_mean_curvature(
    mesh: trimesh.Trimesh, points: NDArray, radius: float
) -> NDArray[np.float64]:
    """Convex part of the discrete mean curvature (1/mm) near each point.

    Sums signed dihedral angle × edge length / 2 over edges whose midpoint lies within
    ``radius`` and divides by the ball's cross-section; concave regions read 0.
    """
    ends = np.asarray(mesh.vertices)[mesh.face_adjacency_edges]
    midpoints = ends.mean(axis=1)
    length = np.linalg.norm(ends[:, 0] - ends[:, 1], axis=1)
    sign = np.where(mesh.face_adjacency_convex, 1.0, -1.0)
    contribution = sign * np.asarray(mesh.face_adjacency_angles) * length / 2.0
    pairs = cKDTree(points).sparse_distance_matrix(
        cKDTree(midpoints), radius, output_type="coo_matrix"
    )
    total = np.zeros(len(points))
    np.add.at(total, pairs.row, contribution[pairs.col])
    return np.maximum(total / (np.pi * radius**2), 0.0)


def local_hull_distance(
    points: NDArray, vertices: NDArray, *, zmin: float, bin_mm: float, window_mm: float
) -> NDArray[np.float64]:
    """Depth of each point inside the convex hull of the vertices within
    ``window_mm`` of its height (0 on or outside the hull)."""
    out = np.zeros(len(points))
    bins = np.floor((points[:, 2] - zmin) / bin_mm).astype(int)
    for b in np.unique(bins):
        centre = zmin + (b + 0.5) * bin_mm
        window = vertices[np.abs(vertices[:, 2] - centre) <= window_mm]
        try:
            planes = ConvexHull(window).equations
        except (QhullError, ValueError):  # too few or coplanar vertices
            continue
        m = bins == b
        # Inside a convex hull the nearest boundary is the nearest facet plane.
        depth = -(points[m] @ planes[:, :3].T + planes[:, 3])
        out[m] = np.clip(depth.min(axis=1), 0.0, None)
    return out


def face_weights(
    mesh: trimesh.Trimesh, params: SampleParams
) -> tuple[NDArray, NDArray]:
    """Per-face sampling weights inside and above the contact band."""
    vertices = np.asarray(mesh.vertices)
    zmin = float(vertices[:, 2].min())
    centers = np.asarray(mesh.triangles_center)
    area = np.asarray(mesh.area_faces)
    curvature = convex_mean_curvature(mesh, centers, params.curvature_radius_mm)
    depth = local_hull_distance(
        centers,
        vertices,
        zmin=zmin,
        bin_mm=params.hull_bin_mm,
        window_mm=params.hull_window_mm,
    )
    weight = (
        area
        * np.maximum(curvature / params.curvature_ref, params.curvature_floor)
        * np.maximum(np.exp(-depth / params.hull_scale_mm), params.hull_floor)
    )
    in_band = centers[:, 2] - zmin <= params.band_mm
    return weight * in_band, weight * ~in_band


def _cache_path(mesh: trimesh.Trimesh, params: SampleParams) -> Path:
    key = hashlib.sha256(repr(sorted(asdict(params).items())).encode()).hexdigest()[:12]
    return _cache_dir() / f"c2f_{_mesh_hash(mesh)}_{key}.npz"


def build_clearance_samples(
    mesh: trimesh.Trimesh,
    params: SampleParams | None = None,
    *,
    use_cache: bool = True,
) -> ClearanceSamples:
    """Fine points, coarse points, cells and radii for one probe body mesh.

    ``mesh`` is the body surface the SDF was built from (the α-wrap envelope with
    shanks stripped), in the probe's local frame with +z pointing up the body.
    """
    params = params or SampleParams()
    cpath = _cache_path(mesh, params)
    if use_cache and cpath.exists():
        with np.load(cpath) as data:
            return ClearanceSamples(
                fine=data["fine"], coarse=data["coarse"],
                cells=data["cells"], radius=data["radius"],
            )  # fmt: skip

    w_band, w_rest = face_weights(mesh, params)
    has_rest = w_rest.sum() > 0
    n_band = round(params.band_share * params.fine_n) if has_rest else params.fine_n
    fine_parts = []
    for i, (n, w) in enumerate([(n_band, w_band), (params.fine_n - n_band, w_rest)]):
        if n > 0:
            pts, _ = trimesh.sample.sample_surface(
                mesh, n, face_weight=w, seed=params.seed + i
            )
            fine_parts.append(pts)
    fine = np.vstack(fine_parts).astype(np.float32)

    even, _ = trimesh.sample.sample_surface_even(
        mesh, params.coarse_even, seed=params.seed + 2
    )
    even = np.asarray(even, np.float32)[: params.coarse_n]
    rng = np.random.default_rng(params.seed + 3)
    extra = rng.choice(len(fine), params.coarse_n - len(even), replace=False)
    coarse = np.vstack([even, fine[extra]]).astype(np.float32)

    distance, owner = cKDTree(coarse).query(fine)
    counts = np.bincount(owner, minlength=len(coarse))
    width = max(int(counts.max()), 1)
    cells = np.full((len(coarse), width), len(fine), np.int32)
    order = np.argsort(owner, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(len(fine)) - starts[owner[order]]
    cells[owner[order], slot] = order
    radius = np.zeros(len(coarse), np.float32)
    np.maximum.at(radius, owner, distance.astype(np.float32))

    samples = ClearanceSamples(fine=fine, coarse=coarse, cells=cells, radius=radius)
    if use_cache:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        # Workers start together; write-then-rename keeps a reader from seeing a
        # partial file.
        tmp = cpath.with_name(f"{cpath.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, **asdict(samples))
        os.replace(tmp, cpath)
    return samples
