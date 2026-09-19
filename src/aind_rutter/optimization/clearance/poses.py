"""Rotations and the optimizer-variable-to-world-pose map.

All JAX-traceable: the constrained solver differentiates through these to
reach the arc, ML and spin variables.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

_D_RAS_TO_LPS = jnp.diag(jnp.array([-1.0, -1.0, 1.0]))


def unit_circle_penalty(sx: Array, sy: Array) -> Array:
    """Soft penalty pulling ``(sx, sy)`` toward the unit circle.

    Returns ``sum((sx² + sy² − 1)²)`` across all probes. The Patch B
    reparameterisation uses ``spin_deg = atan2(sy, sx)`` which only
    reads the DIRECTION of ``(sx, sy)``; magnitude is geometrically
    irrelevant but the optimizer can wander away from the unit circle
    (e.g. toward the origin where ``atan2`` gradients are undefined,
    or toward the bounds where the (sx, sy) Hessian is poorly
    conditioned).

    This penalty keeps the magnitude consistent across stages so
    poses are interchangeable between reduced / Phase 1 / Phase 2 and
    so any downstream consumer reading ``(sx, sy)`` as a unit vector
    gets a well-conditioned value.
    """
    radii_sq = sx * sx + sy * sy
    return jnp.sum((radii_sq - 1.0) ** 2)


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


def arc_angles_to_rotation(ap_deg: Array, ml_deg: Array, spin_deg: Array) -> Array:
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


def spin_deg_from_sxy(sx: Array, sy: Array) -> Array:
    """Convert ``(sx, sy)`` rotation parameterization to ``spin_deg``.

    The optimizer's reduced/full y vector parameterizes spin as a 2D
    vector ``(sx, sy) ∝ (cos θ, sin θ)`` on the unit circle, avoiding
    the ±180° wraparound discontinuity that bound-clipped scalar-angle
    scalar-angle layout. Internally, the existing
    :func:`pose_from_optimizer_vars` API still takes ``spin_deg`` so we
    convert at the unpacking site via ``atan2(sy, sx)``.

    The norm of ``(sx, sy)`` is irrelevant — only the direction
    matters. JAX's ``arctan2`` is well-defined and differentiable
    everywhere except at the origin; optimizer bounds keep us away from
    it.
    """
    return jnp.degrees(jnp.arctan2(sy, sx))
