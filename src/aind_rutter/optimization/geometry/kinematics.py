"""Probe pose adapter for the optimizer.

Wraps the rig kinematics in a few thin functions:

- :func:`pose_from_optimizer_vars` — convert the optimizer's continuous
  variables ``(ap, ml, spin, offset_R, offset_A, past_target_mm)`` into
  ``(R, pose_tip_world)``. Matches the manual-mode convention in
  :class:`planning.ProbePose` exactly so the optimizer's output can be
  re-applied via the existing dispatch path.
- :func:`shank_capsules_from_pose` — given ``(R, pose_tip_world)`` and
  per-probe-asset shank-tip positions in the local frame (from
  :func:`domain.pose.detect_shank_tips_local`), build one
  :class:`Capsule` per shank in world LPS-mm.
- :func:`pose_at_hole_best_fit` — static
  pose that perfectly aligns the probe's shaft with a given hole's
  axis and the shank-row with the slot's major axis. No optimizer
  variables; closed-form from the hole spec.
- :func:`required_ap_deg` — approximate AP angle for the middle layer's
  arc clustering. Pure function of the hole axis.

Numpy-only for v1; the operations are all standard linear algebra and
trig, which JAX can trace as-is once callers use the JAX kernels in
``optimization.sdf.kernels`` and ``optimization.objectives``.
"""

from __future__ import annotations

import numpy as np
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.arc_angles import arc_angles_to_affine
from numpy.typing import ArrayLike, NDArray


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

    ``(ap_deg, ml_deg, spin_deg)`` are interpreted in subject anatomical
    LPS, matching the legacy convention: ``ap=0, ml=0, spin=0`` means
    "probe vertical in subject LPS." The rig's head-tilt offset
    (``Kinematics.subject_from_rig``) only impacts the *reachable* set
    of subject-frame angles (via the optimizer's bounds and rig-limit
    constraints); it does not change what a given (ap, ml, spin) value
    means geometrically.
    """
    R = arc_angles_to_affine(float(ap_deg), float(ml_deg), float(spin_deg))
    off_RAS = np.array([float(offset_R_mm), float(offset_A_mm), 0.0], dtype=np.float64)
    off_LPS = convert_coordinate_system(off_RAS, "RAS", "LPS")
    adjusted_target = np.asarray(target_LPS, dtype=np.float64) + off_LPS
    insertion_vec = R @ np.array([0.0, 0.0, -float(past_target_mm)], dtype=np.float64)
    pose_tip = adjusted_target + insertion_vec
    if recording_center_local is not None:
        pose_tip = pose_tip - R @ np.asarray(recording_center_local, dtype=np.float64)
    return R, pose_tip
