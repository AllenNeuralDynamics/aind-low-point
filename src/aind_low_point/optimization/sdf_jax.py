"""JAX-traceable kernels: pose math, SDF trilinear lookup, pairwise
signed clearance via SDF.

Companion to :mod:`aind_low_point.optimization.sdf` (which builds the
voxel grids). This module does the inner-loop work that needs to be
differentiable for SLSQP's Jacobian.

The core function is :func:`pairwise_signed_clearance`: for two probes
``(a, b)`` at world poses ``(R_a, t_a)`` and ``(R_b, t_b)``, transform
``b``'s surface samples into ``a``'s canonical local frame, look up
``a``'s SDF at those points, take the min; symmetrise. The result is
signed (negative inside, positive outside) and smooth — including
through overlap, where FCL's BVH distance clamps at zero.

Use ``jax.grad(pairwise_signed_clearance)`` to get analytic gradients
w.r.t. the optimizer's variables; no finite-diff needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

# RAS → LPS sign flip applied to a (3, 3) rotation: R_lps = D R_ras D.
_D_RAS_TO_LPS = jnp.diag(jnp.array([-1.0, -1.0, 1.0]))


def _rot_x(angle_rad: Array) -> Array:
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    return jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def _rot_y(angle_rad: Array) -> Array:
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    return jnp.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def _rot_z(angle_rad: Array) -> Array:
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    return jnp.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def arc_angles_to_rotation(
    ap_deg: Array, ml_deg: Array, spin_deg: Array
) -> Array:
    """Convention-matched ``arc_angles_to_affine`` in JAX.

    Mirrors :func:`aind_mri_utils.arc_angles.arc_angles_to_affine` with
    ``invert_AP=True, invert_rotation=True`` (the AIND convention). The
    rotation maps probe canonical local frame to world (LPS), so a
    point ``p_local`` becomes ``R @ p_local`` in world coords.
    """
    # invert_AP=True, invert_rotation=True → angles in deg
    ap = -jnp.deg2rad(ap_deg)
    ml = jnp.deg2rad(ml_deg)
    spin = -jnp.deg2rad(spin_deg)
    # RAS-frame Euler XYZ: RX(ap) RY(ml) RZ(spin)
    R_ras = _rot_x(ap) @ _rot_y(ml) @ _rot_z(spin)
    # RAS → LPS conjugation
    return _D_RAS_TO_LPS @ R_ras @ _D_RAS_TO_LPS


def pose_from_optimizer_vars(
    target_LPS: Array,
    ap_deg: Array,
    ml_deg: Array,
    spin_deg: Array,
    offset_R_mm: Array,
    offset_A_mm: Array,
    past_target_mm: Array,
    recording_center_local: Array,
) -> tuple[Array, Array]:
    """JAX-traceable companion of
    :func:`optimization.kinematics.pose_from_optimizer_vars`.

    Returns ``(R, pose_tip_world)``. ``pose_tip_world`` is the position
    of the probe's local origin (= shank-0 tip in canonical) such that
    the recording-array centre lands at ``target + offset_LPS −
    past_target · shaft_dir``.
    """
    R = arc_angles_to_rotation(ap_deg, ml_deg, spin_deg)
    # off_RAS = (off_R, off_A, 0) → off_LPS = (-off_R, -off_A, 0)
    off_LPS = jnp.stack([-offset_R_mm, -offset_A_mm, jnp.zeros_like(offset_R_mm)])
    adjusted_target = target_LPS + off_LPS
    zero = jnp.zeros_like(past_target_mm)
    insertion_vec = R @ jnp.stack([zero, zero, -past_target_mm])
    pose_tip = adjusted_target + insertion_vec - R @ recording_center_local
    return R, pose_tip


def trilinear_sdf(
    grid: Array,
    origin: Array,
    spacing: Array,
    query_local: Array,
    out_of_bounds_value: Array = jnp.array(1e3),
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

    Returns
    -------
    (...,) interpolated signed distances. Differentiable w.r.t.
    ``query_local`` and ``grid``.
    """
    grid = jnp.asarray(grid)
    Nx, Ny, Nz = grid.shape
    coords = (query_local - origin) / spacing  # (..., 3) in voxel units
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
    interp = c0 * (1 - fz) + c1 * fz
    return jnp.where(in_bounds, interp, out_of_bounds_value)


def pairwise_signed_clearance(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    sdf_a_grid: Array,
    sdf_a_origin: Array,
    sdf_a_spacing: Array,
    sdf_b_grid: Array,
    sdf_b_origin: Array,
    sdf_b_spacing: Array,
    surface_a: Array,  # (N, 3) — a's surface in a's canonical local
    surface_b: Array,  # (N, 3) — b's surface in b's canonical local
) -> Array:
    """Minimum signed distance between two probes via SDF lookup.

    Negative ⇒ overlap; positive ⇒ clear. Smooth + differentiable
    everywhere via trilinear interpolation. Symmetrised by querying
    both ``a → b`` and ``b → a`` (the surface-vs-volume formulation is
    asymmetric otherwise — sampling density differences would make one
    direction tighter than the other).
    """
    # Transform b's surface points (b's local) into a's local frame:
    # world_b = R_b @ surf_b + t_b
    # local_in_a = R_a^T @ (world_b - t_a)
    world_b = surface_b @ R_b.T + t_b  # (N, 3)
    local_in_a = (world_b - t_a) @ R_a  # equivalent to R_a^T @ (world_b - t_a).T
    sd_b_in_a = trilinear_sdf(
        sdf_a_grid, sdf_a_origin, sdf_a_spacing, local_in_a
    )
    # Symmetric direction
    world_a = surface_a @ R_a.T + t_a
    local_in_b = (world_a - t_b) @ R_b
    sd_a_in_b = trilinear_sdf(
        sdf_b_grid, sdf_b_origin, sdf_b_spacing, local_in_b
    )
    return jnp.minimum(jnp.min(sd_b_in_a), jnp.min(sd_a_in_b))


@jax.jit
def pairwise_signed_clearance_jit(
    R_a, t_a, R_b, t_b,
    sdf_a_grid, sdf_a_origin, sdf_a_spacing,
    sdf_b_grid, sdf_b_origin, sdf_b_spacing,
    surface_a, surface_b,
):
    return pairwise_signed_clearance(
        R_a, t_a, R_b, t_b,
        sdf_a_grid, sdf_a_origin, sdf_a_spacing,
        sdf_b_grid, sdf_b_origin, sdf_b_spacing,
        surface_a, surface_b,
    )
