"""Target density model and per-probe coverage integral.

For v1 the density is a Gaussian on the target centroid:

    density(p) = exp(-||p − target||² / (2·σ²))

The optimizer's coverage objective integrates this density along each
shank's active recording range, summed over all shanks listed in the
probe kind's :class:`RecordingGeometry`. This naturally encourages
the optimizer to:

- Aim the shaft so the *shank row* is centered on the target (small
  perpendicular distance for every shank), not just the
  position-bearing shank.
- Set ``past_target_mm`` so the active recording region is centered
  on the target's projection along the shaft.

Numerics
--------
The integral is approximated by Simpson's rule on a fixed sample
count along each active range. Sample count is configurable; default
of 41 covers the typical 0.7–3 mm ranges with ~20–75 µm step — much
finer than the spatial scale of the density (σ ≈ 0.5 mm). A trapezoid
fallback is used when the sample count is even.

The implementation is pure-numpy and JAX-traceable as long as the
density callable is too. The default :func:`gaussian_density` factory
is JAX-friendly (uses ``np.exp`` only).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aind_low_point.optimization.geometry import Capsule
from aind_low_point.optimization.recording import RecordingGeometry

DensityFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]


def gaussian_density(target_LPS: ArrayLike, sigma_mm: float = 0.5) -> DensityFn:
    """Return a Gaussian density centered on ``target_LPS`` with
    isotropic standard deviation ``sigma_mm``.

    ``density(points)`` accepts a single ``(3,)`` point or a batched
    ``(N, 3)`` array and returns a scalar or ``(N,)`` array.
    """
    target = np.asarray(target_LPS, dtype=np.float64)
    inv_two_sigma_sq = 1.0 / (2.0 * float(sigma_mm) ** 2)

    def fn(points):
        p = np.asarray(points, dtype=np.float64)
        d2 = np.sum((p - target) ** 2, axis=-1)
        return np.exp(-d2 * inv_two_sigma_sq)

    return fn


def gaussian_mixture_density(
    target_points: ArrayLike, sigma_mm: float = 0.3
) -> DensityFn:
    """Return an equally-weighted Gaussian mixture density over
    ``target_points`` with isotropic kernel bandwidth ``sigma_mm``.

    Density value at query point ``p`` is
    ``(1/N) Σᵢ exp(-||p − xᵢ||² / (2·σ²))``, peaking at ``1.0`` on each
    target point. This matches the scale of :func:`gaussian_density`
    (which peaks at ``1.0`` at the target) so coverage line-integrals
    have comparable units regardless of which factory is used.

    ``density(points)`` accepts a single ``(3,)`` point or a batched
    ``(N, 3)`` array and returns a scalar or ``(N,)`` array.

    **Slow** for large N — at each query point this evaluates ``N``
    exponentials. Prefer :func:`voxel_kde_density` for production use
    when ``N`` is more than a few hundred; keep this around for tests
    and small-N reference checks.

    Raises
    ------
    ValueError
        If ``target_points`` is empty (zero rows) — silent zero coverage
        is the worst failure mode for the optimizer, so callers must
        catch the empty case before reaching here.
    """
    targets = np.asarray(target_points, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"target_points must have shape (N, 3); got {targets.shape}")
    n = targets.shape[0]
    if n == 0:
        raise ValueError("target_points is empty — cannot build mixture density")
    inv_two_sigma_sq = 1.0 / (2.0 * float(sigma_mm) ** 2)
    inv_n = 1.0 / n

    def fn(points):
        p = np.asarray(points, dtype=np.float64)
        # Broadcast: p shape (..., 3), targets (N, 3) -> (..., N, 3)
        diff = p[..., None, :] - targets
        d2 = np.sum(diff * diff, axis=-1)  # (..., N)
        return np.sum(np.exp(-d2 * inv_two_sigma_sq), axis=-1) * inv_n

    return fn


def voxel_kde_density(
    target_points: ArrayLike,
    sigma_mm: float = 0.3,
    *,
    spacing_mm: float = 0.1,
    pad_sigmas: float = 4.0,
) -> DensityFn:
    """Pre-bake a Gaussian-mixture density to a uniform voxel grid; the
    returned ``DensityFn`` is a trilinear lookup.

    Same continuous density as :func:`gaussian_mixture_density` —
    ``(1/N) Σᵢ exp(-||p − xᵢ||² / (2·σ²))`` — but evaluated by
    pre-depositing each target point onto a 3D grid (unnormalized
    Gaussian kernel of radius ``pad_sigmas·σ``) and trilinearly
    interpolating at query time.

    For ``N`` in the thousands and many thousands of query evaluations
    per optimizer run, this is several orders of magnitude faster than
    the on-the-fly mixture and matches it to within the voxel
    discretization error (≲ ``spacing_mm/2`` in the position of each
    kernel center). Default ``spacing_mm = 0.1`` mm at ``σ = 0.3`` mm
    gives ~3 samples per σ — more than enough for the coverage
    line-integral's needs.

    Outside the padded bounding box of ``target_points`` the density
    returns 0 (the GMM tail there is already << 1e-3 for ``pad_sigmas
    = 4``).

    Parameters
    ----------
    target_points : ArrayLike, shape (N, 3)
        Cloud of target locations in world LPS mm.
    sigma_mm : float
        Kernel bandwidth (same as ``gaussian_mixture_density``).
    spacing_mm : float
        Voxel edge length (mm). Smaller = more accurate, more memory.
    pad_sigmas : float
        Bounding-box padding in units of σ.

    Returns
    -------
    DensityFn
        ``density(points)`` accepts a single ``(3,)`` point or a batched
        ``(..., 3)`` array; returns a scalar or matching-shape array.

    Raises
    ------
    ValueError
        If ``target_points`` is empty.
    """
    targets = np.asarray(target_points, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"target_points must have shape (N, 3); got {targets.shape}")
    n = targets.shape[0]
    if n == 0:
        raise ValueError("target_points is empty — cannot build voxel-KDE density")

    sigma = float(sigma_mm)
    spacing = float(spacing_mm)
    pad = pad_sigmas * sigma
    bbox_min = targets.min(axis=0) - pad
    bbox_max = targets.max(axis=0) + pad
    dims = np.ceil((bbox_max - bbox_min) / spacing).astype(int) + 1
    Nx, Ny, Nz = int(dims[0]), int(dims[1]), int(dims[2])
    grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    # Pre-compute the kernel: unnormalized Gaussian (peak=1) sampled at
    # the lattice's relative offsets, so summing into ``grid`` matches
    # the continuous GMM up to voxel-snapping.
    k_radius = int(np.ceil(pad_sigmas * sigma / spacing))
    ax = np.arange(-k_radius, k_radius + 1) * spacing
    # ``r²`` over the (2k+1)³ block; np.add.outer keeps it simple.
    r2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
    kernel = np.exp(-r2 / (2.0 * sigma * sigma))

    inv_n = 1.0 / n
    for p in targets:
        idx = np.floor((p - bbox_min) / spacing).astype(int)
        i_min = idx - k_radius
        i_max = idx + k_radius + 1
        i0_g = np.maximum(i_min, 0)
        i1_g = np.minimum(i_max, np.array([Nx, Ny, Nz]))
        if np.any(i0_g >= i1_g):
            continue  # entirely outside grid (won't happen given pad)
        i0_k = i0_g - i_min
        i1_k = i0_k + (i1_g - i0_g)
        grid[
            i0_g[0] : i1_g[0],
            i0_g[1] : i1_g[1],
            i0_g[2] : i1_g[2],
        ] += kernel[
            i0_k[0] : i1_k[0],
            i0_k[1] : i1_k[1],
            i0_k[2] : i1_k[2],
        ]
    grid *= inv_n

    origin = bbox_min
    inv_spacing = 1.0 / spacing
    shape = grid.shape

    def fn(points):
        p = np.asarray(points, dtype=np.float64)
        # Continuous voxel coordinates.
        coords = (p - origin) * inv_spacing  # (..., 3)
        i0 = np.floor(coords).astype(np.int64)
        f = coords - i0  # fractional parts in [0, 1)
        # Out-of-bounds mask: any voxel index outside [0, Nx-2] etc. → 0.
        in_bounds = (
            (i0[..., 0] >= 0)
            & (i0[..., 0] < shape[0] - 1)
            & (i0[..., 1] >= 0)
            & (i0[..., 1] < shape[1] - 1)
            & (i0[..., 2] >= 0)
            & (i0[..., 2] < shape[2] - 1)
        )
        # Clamp indices so the gather below is safe; out-of-bounds entries
        # are zeroed via ``in_bounds`` after.
        ix = np.clip(i0[..., 0], 0, shape[0] - 2)
        iy = np.clip(i0[..., 1], 0, shape[1] - 2)
        iz = np.clip(i0[..., 2], 0, shape[2] - 2)
        fx = f[..., 0]
        fy = f[..., 1]
        fz = f[..., 2]
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
        out = c0 * (1 - fz) + c1 * fz
        return np.where(in_bounds, out, 0.0)

    return fn


def _simpson_weights(n: int) -> NDArray[np.floating]:
    """Composite Simpson 1/3 weights for an odd ``n``. For even ``n``
    falls back to trapezoidal."""
    if n < 2:
        return np.ones(max(n, 1), dtype=np.float64)
    if n % 2 == 0:
        # Trapezoidal for even sample count.
        w = np.ones(n, dtype=np.float64)
        w[0] = w[-1] = 0.5
        return w
    # Simpson 1/3: 1, 4, 2, 4, 2, ..., 4, 1, divided by 3.
    w = np.ones(n, dtype=np.float64)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w / 3.0


def integrate_density_along_shank(
    density_fn: DensityFn,
    shank: Capsule,
    *,
    start_mm: float,
    end_mm: float,
    n_samples: int = 41,
) -> float:
    """Line-integral of ``density_fn`` along the shank's axis between
    ``start_mm`` and ``end_mm`` measured from ``shank.p0`` (the tip).

    The integration variable is *physical arc length along the
    shank*, not capsule parameter — so the result is in
    ``density × mm`` units.
    """
    p0 = np.asarray(shank.p0, dtype=np.float64)
    p1 = np.asarray(shank.p1, dtype=np.float64)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return 0.0
    unit = direction / length
    s = np.linspace(start_mm, end_mm, n_samples)
    points = p0[None, :] + s[:, None] * unit[None, :]
    values = density_fn(points)
    weights = _simpson_weights(n_samples)
    span = float(end_mm - start_mm)
    if span <= 0.0:
        return 0.0
    h = span / (n_samples - 1) if n_samples > 1 else span
    return float(np.sum(values * weights) * h)


def coverage(
    density_fn: DensityFn,
    shank_capsules: list[Capsule],
    recording_geom: RecordingGeometry,
    *,
    n_samples: int = 41,
) -> float:
    """Total coverage = sum of per-shank line integrals over each
    shank's active recording range.

    The number of capsules and the number of active ranges in
    ``recording_geom`` must match. Mismatches raise ``ValueError``
    with a clear message — defensive because the optimizer surfaces
    silent zero-coverage as 'no good poses found,' which is the worst
    failure mode.
    """
    if len(shank_capsules) != recording_geom.n_shanks:
        raise ValueError(
            f"shank count mismatch: got {len(shank_capsules)} capsules vs. "
            f"{recording_geom.n_shanks} ranges in recording geometry"
        )
    total = 0.0
    for shank, (start_mm, end_mm) in zip(
        shank_capsules, recording_geom.active_ranges_mm
    ):
        total += integrate_density_along_shank(
            density_fn,
            shank,
            start_mm=start_mm,
            end_mm=end_mm,
            n_samples=n_samples,
        )
    return total
