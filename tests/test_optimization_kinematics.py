"""Tests for ``aind_rutter.optimization.geometry.kinematics``."""

from __future__ import annotations

import numpy as np

from aind_rutter.optimization.geometry.kinematics import pose_from_optimizer_vars


def test_pose_zero_rotations_no_offset_no_depth():
    """Zero rotations + no offsets + zero depth → pose_tip equals target."""
    R, tip = pose_from_optimizer_vars(
        target_LPS=[1.0, 2.0, 3.0],
        ap_deg=0.0,
        ml_deg=0.0,
        spin_deg=0.0,
        offset_R_mm=0.0,
        offset_A_mm=0.0,
        past_target_mm=0.0,
    )
    assert np.allclose(R, np.eye(3))
    assert np.allclose(tip, [1.0, 2.0, 3.0])


def test_pose_zero_rotations_depth_shifts_minus_z():
    """At zero rotations the insertion vector is exactly [0, 0, -depth]."""
    _, tip = pose_from_optimizer_vars(
        target_LPS=[0, 0, 0],
        ap_deg=0.0,
        ml_deg=0.0,
        spin_deg=0.0,
        offset_R_mm=0.0,
        offset_A_mm=0.0,
        past_target_mm=5.0,
    )
    assert np.allclose(tip, [0.0, 0.0, -5.0])


def test_pose_offset_RAS_to_LPS():
    """offset_R_mm=2, offset_A_mm=3 in RAS → (-2, -3, 0) in LPS."""
    _, tip = pose_from_optimizer_vars(
        target_LPS=[0, 0, 0],
        ap_deg=0.0,
        ml_deg=0.0,
        spin_deg=0.0,
        offset_R_mm=2.0,
        offset_A_mm=3.0,
        past_target_mm=0.0,
    )
    # RAS [2, 3, 0] → LPS [-2, -3, 0] (R/A flip signs, S unchanged).
    assert np.allclose(tip, [-2.0, -3.0, 0.0])


def test_pose_matches_planning_probepose():
    """Numerical equivalence with planning.ProbePose.from_planning_state
    style construction. Reproduces the exact formula for a non-trivial pose."""
    from aind_anatomical_utils.coordinate_systems import (
        convert_coordinate_system,
    )
    from aind_mri_utils.arc_angles import arc_angles_to_affine

    target = np.array([1.0, 2.0, 3.0])
    ap, ml, spin = 10.0, -5.0, 30.0
    off_R, off_A, depth = 0.5, -0.3, 4.0

    R_ref = arc_angles_to_affine(ap, ml, spin)
    off_lps = convert_coordinate_system(np.array([off_R, off_A, 0.0]), "RAS", "LPS")
    tip_ref = target + off_lps + R_ref @ np.array([0, 0, -depth])

    R, tip = pose_from_optimizer_vars(
        target_LPS=target,
        ap_deg=ap,
        ml_deg=ml,
        spin_deg=spin,
        offset_R_mm=off_R,
        offset_A_mm=off_A,
        past_target_mm=depth,
    )
    assert np.allclose(R, R_ref)
    assert np.allclose(tip, tip_ref)
