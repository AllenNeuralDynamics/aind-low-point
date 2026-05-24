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
from dataclasses import dataclass, field
from pathlib import Path

import igl
import numpy as np
import trimesh
from numpy.typing import NDArray

DEFAULT_SPACING_MM: float = 0.2
DEFAULT_PAD_MM: float = 2.0


@dataclass(frozen=True)
class ProbeSDF:
    """Pre-computed dual-rep clearance data for one probe (body voxel
    SDF + analytic shank OBBs), all in canonical local frame.

    The grid is sampled on a regular axis-aligned lattice:
    ``world_local_pt = origin + i * spacing`` for each integer index
    ``i`` along the three axes. Look up at a continuous local-frame
    point ``p`` by trilinear or tricubic interpolation in
    :mod:`aind_low_point.optimization.sdf_jax`.

    Outside the grid bbox the SDF should be treated as ``+spacing * 10``
    or similar large positive — the probe is "definitely far" — since
    we don't bother computing SDF in empty regions of the bbox.

    ``surface_points`` is an ``(N, 3)`` set of points sampled
    uniformly on the body mesh surface. Used as the query set in
    pairwise clearance — for a pair ``(a, b)``, we transform ``b``'s
    surface points into ``a``'s local frame and look up ``a``'s SDF at
    those points (and symmetrise).

    ``shank_centers`` / ``shank_halves`` are ``(S, 3)`` arrays of
    per-shank OBB params in the same canonical local frame. Empty
    ``(0, 3)`` arrays when no shanks (legacy raw-mesh SDF path).
    """

    grid: NDArray[np.floating]  # (Nx, Ny, Nz) float32
    origin: NDArray[np.floating]  # (3,) world local pt at grid[0,0,0]
    spacing: float  # voxel edge length (mm)
    surface_points: NDArray[np.floating]  # (N, 3) canonical local
    shank_centers: NDArray[np.floating] = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    shank_halves: NDArray[np.floating] = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )

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
    sign_type: str = "fwn",
) -> Path:
    key = (
        f"{_mesh_hash(mesh)}_s{spacing_mm}_p{pad_mm}"
        f"_n{n_surface_points}_{sign_type}"
    )
    return _cache_dir() / f"{key}.npz"


def build_probe_sdf(
    mesh: trimesh.Trimesh,
    *,
    spacing_mm: float = DEFAULT_SPACING_MM,
    pad_mm: float = DEFAULT_PAD_MM,
    n_surface_points: int = 5000,
    use_cache: bool = True,
    sign_type: str = "fwn",
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
    sign_type
        ``"fwn"`` (default) → ``FAST_WINDING_NUMBER`` — correct signs
        on non-watertight inputs (raw CAD). ``"pseudonormal"`` →
        ``PSEUDONORMAL`` — sharper signs but requires watertight input
        (α-wrap envelopes). Use ``"pseudonormal"`` whenever the source
        is the α-wrap envelope, ``"fwn"`` for legacy raw CAD.
    """
    cpath = _cache_path(
        mesh, spacing_mm=spacing_mm, pad_mm=pad_mm,
        n_surface_points=n_surface_points, sign_type=sign_type,
    )
    if use_cache:
        if cpath.exists():
            with np.load(cpath) as data:
                shank_c = (
                    data["shank_centers"].astype(np.float64)
                    if "shank_centers" in data.files
                    else np.zeros((0, 3), dtype=np.float64)
                )
                shank_h = (
                    data["shank_halves"].astype(np.float64)
                    if "shank_halves" in data.files
                    else np.zeros((0, 3), dtype=np.float64)
                )
                return ProbeSDF(
                    grid=data["grid"].astype(np.float32),
                    origin=data["origin"].astype(np.float64),
                    spacing=float(data["spacing"]),
                    surface_points=data["surface_points"].astype(np.float64),
                    shank_centers=shank_c,
                    shank_halves=shank_h,
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

    if sign_type == "pseudonormal":
        sig = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    elif sign_type == "fwn":
        sig = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    else:
        raise ValueError(
            f"sign_type must be 'pseudonormal' or 'fwn', got {sign_type!r}"
        )
    S, _I, _C, _N = igl.signed_distance(
        grid_pts.astype(np.float64), V, F, sign_type=sig
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
            shank_centers=out.shank_centers,
            shank_halves=out.shank_halves,
        )
    return out


def build_probe_sdf_from_alpha_wrap(
    raw_mesh: trimesh.Trimesh,
    *,
    alpha_mm: float = 0.2,
    offset_mm: float = 0.15,
    spacing_mm: float = DEFAULT_SPACING_MM,
    pad_mm: float = DEFAULT_PAD_MM,
    n_surface_points: int = 5000,
    use_cache: bool = True,
    strip_shanks_first: bool = True,
) -> ProbeSDF:
    """Build a probe body SDF via α-wrap → PSEUDONORMAL signed distance,
    bundled with literature-floored shank OBBs for the dual-rep
    clearance kernel.

    Convenience wrapper: strips shanks → α-wraps the body → builds the
    SDF on the watertight envelope → extracts and floors shank OBBs
    from the raw mesh. Returns a single ``ProbeSDF`` carrying both.

    Use this whenever the source is surface-modeled raw CAD that
    wouldn't sign-correctly under PSEUDONORMAL directly.

    Set ``strip_shanks_first=False`` for fixtures (cone, well,
    headframe) — they have no shanks; stripping anything in the
    shank-zone classifier (z ≤ 10.5 mm) would erase the whole mesh.
    """
    from aind_low_point.optimization.envelope import (
        build_alpha_wrap_envelope,
        extract_shank_obbs,
        floor_shank_half_extents,
    )

    envelope = build_alpha_wrap_envelope(
        raw_mesh,
        alpha_mm=alpha_mm,
        offset_mm=offset_mm,
        strip_shanks_first=strip_shanks_first,
        use_cache=use_cache,
    )
    body_sdf = build_probe_sdf(
        envelope,
        spacing_mm=spacing_mm,
        pad_mm=pad_mm,
        n_surface_points=n_surface_points,
        use_cache=use_cache,
        sign_type="pseudonormal",
    )
    if strip_shanks_first:
        centers, halves = extract_shank_obbs(raw_mesh)
        halves = floor_shank_half_extents(halves)
    else:
        # Fixtures have no shanks.
        centers = np.zeros((0, 3), dtype=np.float64)
        halves = np.zeros((0, 3), dtype=np.float64)
    # Re-pack into a new ProbeSDF carrying the OBBs (the body cache may
    # have come back without them on a cache hit).
    from dataclasses import replace

    return replace(body_sdf, shank_centers=centers, shank_halves=halves)
