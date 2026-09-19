"""Trilinear lookup into a voxel signed-distance grid.

The grid is built by :mod:`aind_rutter.optimization.clearance.voxel_sdf`,
which is kept free of JAX so it imports without a backend.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def trilinear_sdf(
    grid: Array,
    origin: Array,
    spacing: Array,
    query_local: Array,
    out_of_bounds_value: Array = jnp.array(1e3),
    n_real: Array | None = None,
) -> Array:
    """Trilinear interpolation of an SDF voxel grid at ``query_local``
    points (which must already be in the probe's canonical local frame).

    Parameters
    ----------
    grid : (Nx, Ny, Nz) array of signed distances (mm).
    origin : (3,) the local-frame position of ``grid[0, 0, 0]``.
    spacing : scalar voxel edge length (mm).
    query_local : (..., 3) query points in the same local frame as the grid.
    out_of_bounds_value : scalar returned for points outside the grid bbox.
        Default 1e3 mm — "definitely far positive", safe for clearance.
    n_real : optional (3,) real grid extent ``(nx, ny, nz)``. When the grid is a
        padded slot of a uniform table (the swept-pair table — see
        ``clearance_sweep``), the true extent is smaller than ``grid.shape``;
        pass it so in-bounds / clip use the real extent and out-of-extent queries
        return ``out_of_bounds_value`` exactly as for the unpadded grid. May be a
        traced (dynamic) value so it works under ``vmap`` gather. ``None`` ⇒ use
        ``grid.shape`` (unchanged behaviour for every existing caller).

    Returns
    -------
    (...,) interpolated signed distances. Differentiable w.r.t.
    ``query_local`` and ``grid``.
    """
    grid = jnp.asarray(grid)
    # Value dtype follows the grid: fp32 normally, bf16 when the grid is
    # stored bf16 (mixed-precision kernel — bf16 gather/blend, fp32 output).
    vdt = grid.dtype
    if n_real is None:
        Nx, Ny, Nz = grid.shape
    else:
        Nx, Ny, Nz = n_real[0], n_real[1], n_real[2]
    # Addressing stays fp32 — voxel indices must be exact regardless of the
    # value precision.
    coords = (
        query_local.astype(jnp.float32) - jnp.asarray(origin, jnp.float32)
    ) / jnp.asarray(spacing, jnp.float32)  # (..., 3) in voxel units
    i0 = jnp.floor(coords).astype(jnp.int32)
    f = coords - i0  # fractional parts

    in_bounds = (
        (i0[..., 0] >= 0)
        & (i0[..., 0] < Nx - 1)
        & (i0[..., 1] >= 0)
        & (i0[..., 1] < Ny - 1)
        & (i0[..., 2] >= 0)
        & (i0[..., 2] < Nz - 1)
    )
    ix = jnp.clip(i0[..., 0], 0, Nx - 2)
    iy = jnp.clip(i0[..., 1], 0, Ny - 2)
    iz = jnp.clip(i0[..., 2], 0, Nz - 2)
    # Interp weights carried at the value dtype (bf16 when the grid is bf16).
    fx, fy, fz = (
        f[..., 0].astype(vdt),
        f[..., 1].astype(vdt),
        f[..., 2].astype(vdt),
    )

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
    # interp value carried at the grid dtype; cast back to fp32 so the
    # cross-point reduction (soft-min / top-k) downstream accumulates fp32.
    interp = (c0 * (1 - fz) + c1 * fz).astype(jnp.float32)
    return jnp.where(in_bounds, interp, jnp.asarray(out_of_bounds_value, jnp.float32))


def trilinear_sdf_stacked(
    grids: Array,
    origins: Array,
    spacings: Array,
    n_reals: Array,
    which: Array,
    query_local: Array,
    out_of_bounds_value: float = 1e3,
) -> Array:
    """:func:`trilinear_sdf` over a table of same-shape grids, reading each query
    point from grid ``which``.

    One gather serves points bound for different probes' SDFs; separate lookups
    would each have to cover every point. ``grids`` is (K, Nx, Ny, Nz) with real
    extents ``n_reals`` (K, 3), ``origins`` (K, 3) and ``spacings`` (K,); ``which``
    is an integer array broadcastable to ``query_local.shape[:-1]``.
    """
    which = jnp.broadcast_to(which, query_local.shape[:-1])
    n = n_reals[which]
    coords = (
        query_local.astype(jnp.float32) - jnp.asarray(origins, jnp.float32)[which]
    ) / jnp.asarray(spacings, jnp.float32)[which][..., None]
    i0 = jnp.floor(coords).astype(jnp.int32)
    f = (coords - i0).astype(grids.dtype)
    in_bounds = jnp.all((i0 >= 0) & (i0 < n - 1), axis=-1)
    idx = jnp.clip(i0, 0, n - 2)
    ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    def corner(dx: int, dy: int, dz: int) -> Array:
        return grids[which, ix + dx, iy + dy, iz + dz]

    c00 = corner(0, 0, 0) * (1 - fx) + corner(1, 0, 0) * fx
    c01 = corner(0, 0, 1) * (1 - fx) + corner(1, 0, 1) * fx
    c10 = corner(0, 1, 0) * (1 - fx) + corner(1, 1, 0) * fx
    c11 = corner(0, 1, 1) * (1 - fx) + corner(1, 1, 1) * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    interp = (c0 * (1 - fz) + c1 * fz).astype(jnp.float32)
    return jnp.where(in_bounds, interp, jnp.asarray(out_of_bounds_value, jnp.float32))
