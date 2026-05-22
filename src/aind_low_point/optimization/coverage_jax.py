"""JAX-traceable coverage computation for Stage 3 Phase 1.

Two density backends, both differentiable:

- **Gaussian** (single target centroid): exact closed-form, no pre-bake.
  ``density(p) = exp(−||p − target||² / (2σ²))``.
- **Voxel KDE** (cloud of retro points pre-baked to a grid): trilinear
  lookup, same accuracy as the numpy ``voxel_kde_density`` modulo
  float32. Pre-bake done on the host via the existing numpy builder;
  the grid is passed into JAX as a static array.

The per-shank coverage integral is Simpson's rule along the active
recording range, summed across shanks per probe. ``shank_mask`` zeros
out padded shanks. Recording active range is per-probe (one (start,
end)) — all shanks of a kind share it (verified per
``RECORDING_GEOMETRY``).

Mode is selected per probe at trace time. The JIT trace bakes a
Gaussian branch or a KDE branch into the kernel for each probe; mixing
modes across probes is fine, just produces a larger trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

_DEFAULT_KDE_SPACING_MM = 0.1
_DEFAULT_KDE_PAD_SIGMAS = 4.0


def _build_kde_grid(
    target_points: np.ndarray,
    *,
    sigma_mm: float,
    spacing_mm: float = _DEFAULT_KDE_SPACING_MM,
    pad_sigmas: float = _DEFAULT_KDE_PAD_SIGMAS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deposit each target point's Gaussian kernel onto a voxel grid.

    Mirrors :func:`aind_low_point.optimization.density.voxel_kde_density`
    but returns ``(grid, origin, spacing_mm)`` directly instead of a
    closure. Same result; just no opaque function wrapper.
    """
    targets = np.asarray(target_points, dtype=np.float64)
    n = targets.shape[0]
    pad = pad_sigmas * sigma_mm
    bbox_min = targets.min(axis=0) - pad
    bbox_max = targets.max(axis=0) + pad
    dims = np.ceil((bbox_max - bbox_min) / spacing_mm).astype(int) + 1
    Nx, Ny, Nz = int(dims[0]), int(dims[1]), int(dims[2])
    grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    k_radius = int(np.ceil(pad_sigmas * sigma_mm / spacing_mm))
    ax = np.arange(-k_radius, k_radius + 1) * spacing_mm
    r2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
    kernel = np.exp(-r2 / (2.0 * sigma_mm * sigma_mm))
    inv_n = 1.0 / n
    for p in targets:
        idx = np.floor((p - bbox_min) / spacing_mm).astype(int)
        i_min = idx - k_radius
        i_max = idx + k_radius + 1
        i0_g = np.maximum(i_min, 0)
        i1_g = np.minimum(i_max, np.array([Nx, Ny, Nz]))
        if np.any(i0_g >= i1_g):
            continue
        i0_k = i0_g - i_min
        i1_k = i0_k + (i1_g - i0_g)
        grid[i0_g[0]:i1_g[0], i0_g[1]:i1_g[1], i0_g[2]:i1_g[2]] += kernel[
            i0_k[0]:i1_k[0], i0_k[1]:i1_k[1], i0_k[2]:i1_k[2]
        ]
    grid *= inv_n
    return grid, bbox_min, float(spacing_mm)


@dataclass(frozen=True)
class GaussianCoverageData:
    """Static per-probe data for the Gaussian-density coverage backend."""

    target_LPS: jnp.ndarray  # (3,)
    sigma_mm: float
    active_start_mm: float
    active_end_mm: float


@dataclass(frozen=True)
class KdeCoverageData:
    """Static per-probe data for the voxel-KDE coverage backend.

    The grid is pre-baked on the host by
    :func:`aind_low_point.optimization.density.voxel_kde_density`; we
    rebuild it here in JAX-friendly arrays and copy the grid contents
    out of the closure.
    """

    grid: jnp.ndarray            # (Nx, Ny, Nz)
    origin: jnp.ndarray          # (3,) world LPS coord of grid[0,0,0]
    spacing_mm: float
    active_start_mm: float
    active_end_mm: float


CoverageData = GaussianCoverageData | KdeCoverageData


def _simpson_weights_jnp(n: int) -> jnp.ndarray:
    """Composite Simpson 1/3 weights for odd ``n``; trapezoid for even.

    Constant in n at trace time — pre-compute once per (n, dtype) and
    pass into the JIT'd kernel as a static array.
    """
    if n < 2:
        return jnp.ones(max(n, 1), dtype=jnp.float32)
    if n % 2 == 0:
        w = np.ones(n, dtype=np.float32)
        w[0] = w[-1] = 0.5
    else:
        w = np.ones(n, dtype=np.float32)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w = w / 3.0
    return jnp.asarray(w)


def build_coverage_data_from_probe_context(
    probe_ctx: Any,
    recording_active_range_mm: tuple[float, float],
) -> CoverageData:
    """Inspect a Stage 3 ``ProbeContext`` and return the matching JAX
    coverage data.

    Uses the same selection logic as the legacy ``_build_inner_context``
    in optimize.py: if the probe has a ``target_points`` cloud, use
    voxel KDE; else Gaussian on ``target_LPS``.

    Parameters
    ----------
    probe_ctx
        Object with ``target_LPS``, ``target_points`` (optional),
        ``density_sigma_mm`` attributes.
    recording_active_range_mm
        ``(start, end)`` in mm along the shank — extracted from
        ``RecordingGeometry.active_ranges_mm[0]`` (we already verified
        all shanks of a kind share the same range).
    """
    sigma_mm = float(getattr(probe_ctx, "density_sigma_mm", 0.5))
    target_points = getattr(probe_ctx, "target_points", None)
    if target_points is not None and len(target_points) > 0:
        grid_np, origin_np, spacing = _build_kde_grid(
            np.asarray(target_points, dtype=np.float64), sigma_mm=sigma_mm
        )
        return KdeCoverageData(
            grid=jnp.asarray(grid_np, dtype=jnp.float32),
            origin=jnp.asarray(origin_np, dtype=jnp.float32),
            spacing_mm=spacing,
            active_start_mm=float(recording_active_range_mm[0]),
            active_end_mm=float(recording_active_range_mm[1]),
        )
    return GaussianCoverageData(
        target_LPS=jnp.asarray(probe_ctx.target_LPS, dtype=jnp.float32),
        sigma_mm=sigma_mm,
        active_start_mm=float(recording_active_range_mm[0]),
        active_end_mm=float(recording_active_range_mm[1]),
    )


def _trilinear_density(
    grid: jnp.ndarray,
    origin: jnp.ndarray,
    spacing_mm: float,
    query_pts: jnp.ndarray,  # (..., 3)
) -> jnp.ndarray:
    """Trilinear interpolation of a density grid at query points.

    Out-of-bounds queries return 0 (the GMM tail there is negligible
    given the pre-bake's ``pad_sigmas=4`` default).
    """
    coords = (query_pts - origin) / spacing_mm  # (..., 3) voxel units
    i0 = jnp.floor(coords).astype(jnp.int32)
    f = coords - i0
    Nx, Ny, Nz = grid.shape
    in_bounds = (
        (i0[..., 0] >= 0) & (i0[..., 0] < Nx - 1)
        & (i0[..., 1] >= 0) & (i0[..., 1] < Ny - 1)
        & (i0[..., 2] >= 0) & (i0[..., 2] < Nz - 1)
    )
    ix = jnp.clip(i0[..., 0], 0, Nx - 2)
    iy = jnp.clip(i0[..., 1], 0, Ny - 2)
    iz = jnp.clip(i0[..., 2], 0, Nz - 2)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    c000 = grid[ix, iy, iz]
    c100 = grid[ix + 1, iy, iz]
    c010 = grid[ix, iy + 1, iz]
    c110 = grid[ix + 1, iy + 1, iz]
    c001 = grid[ix, iy, iz + 1]
    c101 = grid[ix + 1, iy, iz + 1]
    c011 = grid[ix, iy + 1, iz + 1]
    c111 = grid[ix + 1, iy + 1, iz + 1]
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    val = c0 * (1 - fz) + c1 * fz
    return jnp.where(in_bounds, val, 0.0)


def _gaussian_density(
    target_LPS: jnp.ndarray,
    sigma_mm: float,
    query_pts: jnp.ndarray,  # (..., 3)
) -> jnp.ndarray:
    """``exp(−||q − target||² / (2σ²))`` evaluated at query points."""
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_mm * sigma_mm)
    d2 = jnp.sum((query_pts - target_LPS) ** 2, axis=-1)
    return jnp.exp(-d2 * inv_two_sigma_sq)


def probe_coverage(
    R: jnp.ndarray,             # (3, 3)
    t: jnp.ndarray,             # (3,)
    shank_tips_local: jnp.ndarray,  # (max_shanks, 3) padded
    shank_mask: jnp.ndarray,    # (max_shanks,) float 0/1
    cov_data: CoverageData,
    n_samples: int = 41,
) -> jnp.ndarray:
    """Coverage = Σ_shanks ∫_active density(shank_pose(s)) ds.

    All shanks share the same active range (per-probe-kind convention).
    Padded shanks contribute zero via ``shank_mask``.
    """
    # World-space shank axis (= probe pose +z direction)
    shank_dir = R @ jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    # Shank tip in world: tip_local @ R.T + t
    tips_world = shank_tips_local @ R.T + t  # (max_shanks, 3)

    # Sample points along the active range, per shank.
    s_vals = jnp.linspace(
        cov_data.active_start_mm, cov_data.active_end_mm, n_samples
    ).astype(jnp.float32)
    # points: (max_shanks, n_samples, 3)
    points = (
        tips_world[:, None, :]
        + s_vals[None, :, None] * shank_dir[None, None, :]
    )

    # Density evaluation: dispatch on the (Python-static) dataclass type.
    if isinstance(cov_data, GaussianCoverageData):
        values = _gaussian_density(
            cov_data.target_LPS, cov_data.sigma_mm, points
        )
    elif isinstance(cov_data, KdeCoverageData):
        values = _trilinear_density(
            cov_data.grid, cov_data.origin, cov_data.spacing_mm, points
        )
    else:
        raise TypeError(f"Unknown CoverageData type: {type(cov_data)}")

    # Simpson's-rule integral along the active range
    weights = _simpson_weights_jnp(n_samples)
    step = (cov_data.active_end_mm - cov_data.active_start_mm) / max(
        n_samples - 1, 1
    )
    per_shank = jnp.sum(values * weights[None, :], axis=-1) * step
    # Mask out padded shanks
    masked = per_shank * shank_mask
    return jnp.sum(masked)
