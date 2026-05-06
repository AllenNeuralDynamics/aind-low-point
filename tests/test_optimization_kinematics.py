"""Tests for ``aind_low_point.optimization.kinematics``."""

from __future__ import annotations

import numpy as np
import pytest

from aind_low_point.optimization.geometry import (
    HoleSection,
    shaft_section_oval_value,
)
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.kinematics import (
    pose_at_hole_best_fit,
    pose_from_optimizer_vars,
    required_ap_deg,
    shank_capsules_from_pose,
)


# -- pose_from_optimizer_vars ----------------------------------------------


def test_pose_zero_rotations_no_offset_no_depth():
    """Zero rotations + no offsets + zero depth → pose_tip equals target."""
    R, tip = pose_from_optimizer_vars(
        target_LPS=[1.0, 2.0, 3.0],
        ap_deg=0.0, ml_deg=0.0, spin_deg=0.0,
        offset_R_mm=0.0, offset_A_mm=0.0,
        past_target_mm=0.0,
    )
    assert np.allclose(R, np.eye(3))
    assert np.allclose(tip, [1.0, 2.0, 3.0])


def test_pose_zero_rotations_depth_shifts_minus_z():
    """At zero rotations the insertion vector is exactly [0, 0, -depth]."""
    _, tip = pose_from_optimizer_vars(
        target_LPS=[0, 0, 0],
        ap_deg=0.0, ml_deg=0.0, spin_deg=0.0,
        offset_R_mm=0.0, offset_A_mm=0.0,
        past_target_mm=5.0,
    )
    assert np.allclose(tip, [0.0, 0.0, -5.0])


def test_pose_offset_RAS_to_LPS():
    """offset_R_mm=2, offset_A_mm=3 in RAS → (-2, -3, 0) in LPS."""
    _, tip = pose_from_optimizer_vars(
        target_LPS=[0, 0, 0],
        ap_deg=0.0, ml_deg=0.0, spin_deg=0.0,
        offset_R_mm=2.0, offset_A_mm=3.0,
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
    off_lps = convert_coordinate_system(
        np.array([off_R, off_A, 0.0]), "RAS", "LPS"
    )
    tip_ref = target + off_lps + R_ref @ np.array([0, 0, -depth])

    R, tip = pose_from_optimizer_vars(
        target_LPS=target,
        ap_deg=ap, ml_deg=ml, spin_deg=spin,
        offset_R_mm=off_R, offset_A_mm=off_A,
        past_target_mm=depth,
    )
    assert np.allclose(R, R_ref)
    assert np.allclose(tip, tip_ref)


# -- shank_capsules_from_pose ----------------------------------------------


def test_shank_capsules_identity_pose_at_origin():
    """R=I, pose_tip=origin: shank tips in world equal local positions."""
    R = np.eye(3)
    pose_tip = np.zeros(3)
    tips_local = np.array([
        [-0.375, 0.0, 0.0],
        [-0.125, 0.0, 0.0],
        [+0.125, 0.0, 0.0],
        [+0.375, 0.0, 0.0],
    ])
    capsules = shank_capsules_from_pose(
        R, pose_tip, tips_local, shaft_length_mm=10.0, shank_radius_mm=0.05
    )
    assert len(capsules) == 4
    for cap, expected_tip in zip(capsules, tips_local):
        assert np.allclose(cap.p0, expected_tip)
        # At identity rotation, shaft direction is +z.
        assert np.allclose(cap.p1, expected_tip + np.array([0, 0, 10.0]))
        assert cap.radius == pytest.approx(0.05)


def test_shank_capsules_translation_only():
    """R=I, pose_tip=(1,2,3): tips are translated by pose_tip."""
    R = np.eye(3)
    pose_tip = np.array([1.0, 2.0, 3.0])
    tips_local = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    capsules = shank_capsules_from_pose(
        R, pose_tip, tips_local, shaft_length_mm=5.0, shank_radius_mm=0.04
    )
    assert np.allclose(capsules[0].p0, [1.0, 2.0, 3.0])
    assert np.allclose(capsules[1].p0, [1.5, 2.0, 3.0])


def test_shank_capsules_rotated_shaft_direction():
    """Local +z direction is transformed to R @ +z in world."""
    # 90° rotation around y-axis: local +z → world +x
    R = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    pose_tip = np.zeros(3)
    tips_local = np.array([[0.0, 0.0, 0.0]])
    capsules = shank_capsules_from_pose(
        R, pose_tip, tips_local, shaft_length_mm=2.0, shank_radius_mm=0.0
    )
    # Tip at origin, base = origin + 2 * (R @ +z) = origin + 2*[1, 0, 0]
    assert np.allclose(capsules[0].p1, [2.0, 0.0, 0.0])


# -- pose_at_hole_best_fit -------------------------------------------------


def _make_hole(*, axis, center, theta=0.0, a=0.6, b=0.35) -> Hole:
    axis_arr = np.asarray(axis, dtype=float)
    sec = HoleSection(
        axis=axis_arr,
        center=np.asarray(center, dtype=float),
        a=a, b=b, theta=theta,
    )
    return Hole(id=0, axis=axis_arr, ref_point=np.asarray(center), sections=[sec, sec])


def test_pose_at_hole_best_fit_axis_aligned_hole():
    """Axis = +z, slot major along default = +y (theta=0).

    cap_basis([0,0,1]) → (e1=+y, e2=-x). slot_major (theta=0) = e1 = +y.
    pose_at_hole_best_fit:
      z_col = -axis = -z   → local +z → world -z (shaft enters going down)
      x_col = slot_major = +y → local +x → world +y (shank-row)
      y_col = z_col × x_col = -z × +y = +x → local +y → world +x
    """
    hole = _make_hole(axis=[0, 0, 1], center=[0, 0, 0])
    R, tip = pose_at_hole_best_fit(hole)
    assert np.allclose(R[:, 0], [0, 1, 0], atol=1e-9)
    assert np.allclose(R[:, 1], [1, 0, 0], atol=1e-9)
    assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-9)
    # Determinant ±1 (rotation matrix).
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    # pose_tip is the bottom section's center.
    assert np.allclose(tip, [0, 0, 0])


def test_pose_at_hole_best_fit_orthonormal():
    """For a tilted axis the constructed R is still orthonormal."""
    axis = np.array([0.3, -0.2, 0.93])
    axis /= np.linalg.norm(axis)
    hole = _make_hole(axis=axis, center=[1, 1, 1], theta=0.7)
    R, _ = pose_at_hole_best_fit(hole)
    # Columns are orthonormal
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    # Right-handed
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    # Local +z maps to -axis
    assert np.allclose(R @ [0, 0, 1], -axis, atol=1e-9)


def test_pose_at_hole_best_fit_threads_the_hole():
    """A 4-shank probe placed at hole-best-fit threads every section."""
    # Axis-aligned hole at origin with the slot major along +y (theta=0).
    axis_arr = np.asarray([0, 0, 1], dtype=float)
    sections = [
        HoleSection(
            axis=axis_arr, center=np.array([0.0, 0.0, 0.5]),
            a=0.65, b=0.42, theta=0.0,
        ),  # top (chamfer, wider)
        HoleSection(
            axis=axis_arr, center=np.array([0.0, 0.0, 0.0]),
            a=0.60, b=0.35, theta=0.0,
        ),
        HoleSection(
            axis=axis_arr, center=np.array([0.0, 0.0, -0.5]),
            a=0.60, b=0.35, theta=0.0,
        ),  # bottom (straight bore, tightest)
    ]
    hole = Hole(
        id=0, axis=axis_arr, ref_point=np.zeros(3), sections=sections
    )
    R, tip = pose_at_hole_best_fit(hole)
    # 4 shanks at 250 µm pitch along local +x (slot-major in world)
    tips_local = np.array([
        [-0.375, 0.0, 0.0],
        [-0.125, 0.0, 0.0],
        [+0.125, 0.0, 0.0],
        [+0.375, 0.0, 0.0],
    ])
    capsules = shank_capsules_from_pose(
        R, tip, tips_local, shaft_length_mm=10.0, shank_radius_mm=0.0
    )
    # Every shank threads every section — max g should be < 0.
    worst = max(
        shaft_section_oval_value(c, s)
        for c in capsules
        for s in sections
    )
    assert worst < 0.0, (
        f"best-fit pose should thread the hole; worst g = {worst:.3f}"
    )
    # The outer shanks at ±0.375 along +y (slot major):
    # g_bot = (0/0.6)² + (0.375/0.35)² − 1 ≈ +0.148 → outside!
    # Wait: local +x maps to world +y (slot major). So the outer shanks
    # are at world y = ±0.375 (in slot-major direction). Slot major has
    # half-extent a=0.60, so they're well inside on that axis. The
    # *minor* axis half-extent b=0.35 is what would clip — but along
    # the minor we're at zero.
    assert worst == pytest.approx((0.375 / 0.60) ** 2 - 1.0, abs=1e-9)


# -- required_ap_deg --------------------------------------------------------


def test_required_ap_zero_for_vertical_axis():
    """A bore aligned with world +z → required AP angle ≈ 0°."""
    assert required_ap_deg([0, 0, 1]) == pytest.approx(0.0, abs=1e-9)


def test_required_ap_sign_convention():
    """Bore tilted in +y direction (anterior) → positive required AP."""
    axis = np.array([0.0, 0.5, np.sqrt(0.75)])  # tilted ~30° toward +y
    ap = required_ap_deg(axis)
    assert ap == pytest.approx(30.0, abs=0.5)


def test_required_ap_negative_y_tilt():
    """Bore tilted in -y direction → negative required AP."""
    axis = np.array([0.0, -0.5, np.sqrt(0.75)])  # tilted ~30° toward -y
    ap = required_ap_deg(axis)
    assert ap == pytest.approx(-30.0, abs=0.5)


def test_required_ap_monotonic_in_y_component():
    """For increasing y component, required AP monotonically increases."""
    aps = []
    for ydir in np.linspace(-0.6, 0.6, 13):
        axis = np.array([0.1, ydir, np.sqrt(max(0.0, 1 - 0.01 - ydir**2))])
        aps.append(required_ap_deg(axis))
    assert all(a < b for a, b in zip(aps, aps[1:]))
