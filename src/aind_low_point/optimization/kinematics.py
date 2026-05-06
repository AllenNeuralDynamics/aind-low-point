"""Probe pose adapter for the optimizer.

Wraps the rig kinematics in a few thin functions:

- :func:`pose_from_optimizer_vars` — convert the optimizer's continuous
  variables ``(ap, ml, spin, offset_R, offset_A, past_target_mm)`` into
  ``(R, pose_tip_world)``. Matches the manual-mode convention in
  :class:`planning.ProbePose` exactly so the optimizer's output can be
  re-applied via the existing dispatch path.
- :func:`shank_capsules_from_pose` — given ``(R, pose_tip_world)`` and
  per-probe-asset shank-tip positions in the local frame (from
  :func:`runtime.shanks.detect_shank_tips_local`), build one
  :class:`Capsule` per shank in world LPS-mm.
- :func:`pose_at_hole_best_fit` — for the LSAP cost matrix: a static
  pose that perfectly aligns the probe's shaft with a given hole's
  axis and the shank-row with the slot's major axis. No optimizer
  variables; closed-form from the hole spec.
- :func:`required_ap_deg` — approximate AP angle for the middle layer's
  arc clustering. Pure function of the hole axis.

Numpy-only for v1; the operations are all standard linear algebra and
trig, which JAX can trace as-is once we wire ``jax.numpy`` into
``objective.py``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.arc_angles import arc_angles_to_affine

from aind_low_point.optimization.geometry import Capsule
from aind_low_point.optimization.holes import Hole


def pose_from_optimizer_vars(
    *,
    target_LPS: ArrayLike,
    ap_deg: float,
    ml_deg: float,
    spin_deg: float,
    offset_R_mm: float,
    offset_A_mm: float,
    past_target_mm: float,
    recording_center_local: ArrayLike | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Manual-mode convention ⇒ ``(R, pose_tip_world)``.

    ``R`` rotates the probe local frame to world LPS. ``pose_tip_world``
    is the world LPS position of the probe's local origin — i.e. the
    position-bearing shank's tip in the AIND canonicalization.

    ``recording_center_local`` is the recording-array center in the
    canonical local frame (``(centroid_x, centroid_y, active_center_mm)``
    from :class:`RecordingGeometry`). When provided, the formula
    subtracts ``R @ recording_center_local`` from ``pose_tip_world`` so
    that the recording-array center lands at ``adjusted_target + R @
    [0, 0, -past_target_mm]``. When ``None`` (default), the legacy
    "tip-on-target" formula is used — used by tests that pre-date the
    pivot redesign.
    """
    R = arc_angles_to_affine(float(ap_deg), float(ml_deg), float(spin_deg))
    off_RAS = np.array(
        [float(offset_R_mm), float(offset_A_mm), 0.0], dtype=np.float64
    )
    off_LPS = convert_coordinate_system(off_RAS, "RAS", "LPS")
    adjusted_target = np.asarray(target_LPS, dtype=np.float64) + off_LPS
    insertion_vec = R @ np.array(
        [0.0, 0.0, -float(past_target_mm)], dtype=np.float64
    )
    pose_tip = adjusted_target + insertion_vec
    if recording_center_local is not None:
        pose_tip = pose_tip - R @ np.asarray(
            recording_center_local, dtype=np.float64
        )
    return R, pose_tip


def shank_capsules_from_pose(
    R: NDArray[np.floating],
    pose_tip_world: ArrayLike,
    shank_tips_local: NDArray[np.floating],
    *,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
) -> list[Capsule]:
    """Build per-shank :class:`Capsule` in world LPS-mm.

    ``shank_tips_local`` is the ``(N, 3)`` array from
    :func:`runtime.shanks.detect_shank_tips_local`. Each shank is
    represented as a capsule extending ``shaft_length_mm`` along the
    probe's local +z (i.e. up the shaft toward the base/headstage),
    rotated into world. The threading-constraint check uses the
    capsule's *axis line* through both endpoints, so ``shank_radius_mm``
    only matters for capsule-vs-capsule clearance.
    """
    pose_tip_world = np.asarray(pose_tip_world, dtype=np.float64)
    shaft_dir_world = R @ np.array([0.0, 0.0, 1.0])
    capsules: list[Capsule] = []
    for tip_local in np.asarray(shank_tips_local, dtype=np.float64):
        tip_world = R @ tip_local + pose_tip_world
        base_world = tip_world + shaft_length_mm * shaft_dir_world
        capsules.append(
            Capsule(p0=tip_world, p1=base_world, radius=shank_radius_mm)
        )
    return capsules


def pose_at_hole_best_fit(
    hole: Hole,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Static "perfect-alignment" pose for the LSAP cost matrix.

    Builds ``(R, pose_tip_world)`` such that:

    - The probe's local +z direction maps to ``-hole.axis`` in world
      (the shaft enters down the bore from the top).
    - The probe's local +x direction (the shank-row direction in NP 2.0
      canonicalization) maps to the slot's major axis.
    - ``pose_tip_world`` is the bottom section's center — the bore's
      deepest extent and the natural reference point for "probe enters
      the brain here."

    The orthonormal frame is constructed via cross products to ensure
    right-handedness; if ``slot_major`` and ``-axis`` are not exactly
    perpendicular (rounding, axis fit error), the second axis basis is
    re-orthonormalized.
    """
    axis = np.asarray(hole.axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    slot_major = hole.slot_major_dir()

    z_col = -axis                    # local +z → world -axis (shaft-down)
    x_col = slot_major               # local +x → world slot major
    y_col = np.cross(z_col, x_col)
    y_col_norm = float(np.linalg.norm(y_col))
    if y_col_norm < 1e-12:
        # Pathological: slot_major parallel to axis. Pick any perpendicular.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, axis)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        y_col = np.cross(z_col, helper)
        y_col = y_col / np.linalg.norm(y_col)
    else:
        y_col = y_col / y_col_norm
    # Re-orthonormalize x against (y, z) for numerical robustness.
    x_col = np.cross(y_col, z_col)
    x_col = x_col / np.linalg.norm(x_col)
    R = np.column_stack([x_col, y_col, z_col])

    pose_tip = np.asarray(hole.sections[-1].center, dtype=np.float64).copy()
    return R, pose_tip


def required_ap_deg(hole_axis_LPS: ArrayLike) -> float:
    """Approximate AP angle that aligns the probe shaft with a bore.

    Used as the clustering key for the middle layer's probe→arc
    assignment and as the initial guess for ``ap_arc_deg``. Defined as
    the angle (in degrees) between the world ``+z`` axis (probe nominal
    "pointing down" direction) and ``hole_axis``, projected onto the
    LPS ``(y, z)`` plane — a reasonable proxy for the AIND rig's
    AP rotation plane. The clustering is invariant to the exact rig
    convention as long as this function is monotonic in true required-AP
    across the relevant range.
    """
    a = np.asarray(hole_axis_LPS, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return float(np.rad2deg(np.arctan2(a[1], a[2])))
