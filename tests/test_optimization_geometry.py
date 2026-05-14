"""Tests for ``aind_low_point.optimization.geometry``.

Covers:
  - point_to_segment_dist
  - segment_to_segment_dist (skew, parallel, intersecting, degenerate)
  - capsule_capsule_dist
  - cap_basis (orthonormality + same convention as the extractor)
  - section_oval_value (axis-aligned, rotated)
  - shaft_section_oval_value (intersection then oval check)
"""

from __future__ import annotations

import numpy as np
import pytest

from aind_low_point.optimization.geometry import (
    Capsule,
    HoleSection,
    cap_basis,
    capsule_capsule_dist,
    point_to_segment_dist,
    section_oval_value,
    segment_to_segment_dist,
    shaft_section_oval_value,
)

# -- point_to_segment_dist --------------------------------------------------


def test_point_to_segment_perpendicular():
    # Segment along +x; point straight above the midpoint.
    d = point_to_segment_dist([0.5, 1.0, 0.0], [0, 0, 0], [1, 0, 0])
    assert d == pytest.approx(1.0)


def test_point_to_segment_endpoint_clip():
    # Point past the +x end of the segment — should clip to that endpoint.
    d = point_to_segment_dist([2.0, 0.0, 0.0], [0, 0, 0], [1, 0, 0])
    assert d == pytest.approx(1.0)


def test_point_to_segment_degenerate():
    # Zero-length "segment" reduces to point distance.
    d = point_to_segment_dist([3, 4, 0], [0, 0, 0], [0, 0, 0])
    assert d == pytest.approx(5.0)


# -- segment_to_segment_dist ------------------------------------------------


def test_segment_segment_skew():
    # Two unit segments offset along z; they don't intersect, gap = 1.
    d = segment_to_segment_dist([0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1])
    assert d == pytest.approx(1.0)


def test_segment_segment_parallel():
    # Parallel segments along x at y = 0 and y = 1.
    d = segment_to_segment_dist([0, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0])
    assert d == pytest.approx(1.0)


def test_segment_segment_intersect():
    # Two segments crossing at (0.5, 0.5, 0).
    d = segment_to_segment_dist([0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0])
    assert d == pytest.approx(0.0, abs=1e-12)


def test_segment_segment_endpoint_clip():
    # Two segments along x, end-to-end with a gap of 0.5 between them.
    d = segment_to_segment_dist([0, 0, 0], [1, 0, 0], [1.5, 0, 0], [2.5, 0, 0])
    assert d == pytest.approx(0.5)


def test_segment_segment_degenerate():
    # Both "segments" are points 5 mm apart.
    d = segment_to_segment_dist([0, 0, 0], [0, 0, 0], [3, 4, 0], [3, 4, 0])
    assert d == pytest.approx(5.0)


# -- capsule_capsule_dist ---------------------------------------------------


def test_capsule_capsule_clearance():
    # Two parallel capsules with axes 4 mm apart, radii 1 each → 2 mm clearance.
    c1 = Capsule(np.array([0, 0, 0]), np.array([2, 0, 0]), radius=1.0)
    c2 = Capsule(np.array([0, 4, 0]), np.array([2, 4, 0]), radius=1.0)
    assert capsule_capsule_dist(c1, c2) == pytest.approx(2.0)


def test_capsule_capsule_overlap_negative():
    # Same axes 1 mm apart, radii 1 each → overlap of 1 mm (negative).
    c1 = Capsule(np.array([0, 0, 0]), np.array([2, 0, 0]), radius=1.0)
    c2 = Capsule(np.array([0, 1, 0]), np.array([2, 1, 0]), radius=1.0)
    assert capsule_capsule_dist(c1, c2) == pytest.approx(-1.0)


def test_capsule_capsule_just_touching():
    # Axes 2 mm apart, radii 1 each → touching, signed distance == 0.
    c1 = Capsule(np.array([0, 0, 0]), np.array([2, 0, 0]), radius=1.0)
    c2 = Capsule(np.array([0, 2, 0]), np.array([2, 2, 0]), radius=1.0)
    assert capsule_capsule_dist(c1, c2) == pytest.approx(0.0, abs=1e-12)


# -- cap_basis --------------------------------------------------------------


def test_cap_basis_orthonormal_z_axis():
    e1, e2 = cap_basis([0, 0, 1])
    axis = np.array([0, 0, 1])
    assert np.dot(e1, axis) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(e2, axis) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(e1, e2) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(e1) == pytest.approx(1.0)
    assert np.linalg.norm(e2) == pytest.approx(1.0)


def test_cap_basis_orthonormal_tilted():
    axis = np.array([0.3, -0.4, 0.866])
    axis /= np.linalg.norm(axis)
    e1, e2 = cap_basis(axis)
    assert abs(np.dot(e1, axis)) < 1e-9
    assert abs(np.dot(e2, axis)) < 1e-9
    assert abs(np.dot(e1, e2)) < 1e-9
    assert np.linalg.norm(e1) == pytest.approx(1.0)
    assert np.linalg.norm(e2) == pytest.approx(1.0)


def test_cap_basis_handles_x_aligned_axis():
    # Helper switches to y when axis is too close to +x.
    e1, e2 = cap_basis([1, 0, 0])
    axis = np.array([1, 0, 0])
    assert abs(np.dot(e1, axis)) < 1e-9
    assert abs(np.dot(e2, axis)) < 1e-9
    assert abs(np.dot(e1, e2)) < 1e-9


# -- section_oval_value -----------------------------------------------------


def test_section_oval_center_is_inside():
    sec = HoleSection(
        axis=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        a=0.6,
        b=0.3,
        theta=0.0,
    )
    assert section_oval_value([0, 0, 0], sec) == pytest.approx(-1.0)


def test_section_oval_on_perimeter():
    sec = HoleSection(
        axis=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        a=0.6,
        b=0.3,
        theta=0.0,
    )
    e1, e2 = cap_basis([0, 0, 1])
    # Point at distance b along e2 — should be on perimeter (g = 0).
    perim = sec.center + sec.b * e2
    assert section_oval_value(perim, sec) == pytest.approx(0.0, abs=1e-9)
    # Point at distance a along e1.
    perim2 = sec.center + sec.a * e1
    assert section_oval_value(perim2, sec) == pytest.approx(0.0, abs=1e-9)


def test_section_oval_outside_is_positive():
    sec = HoleSection(
        axis=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        a=0.6,
        b=0.3,
        theta=0.0,
    )
    e1, _ = cap_basis([0, 0, 1])
    # Twice the major half-extent → g = (1.2 / 0.6)^2 - 1 = 3.
    point = sec.center + 2 * sec.a * e1
    assert section_oval_value(point, sec) == pytest.approx(3.0)


def test_section_oval_rotation_swaps_axes():
    # theta = pi/2 should swap which world direction sees the major half-extent.
    sec = HoleSection(
        axis=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        a=0.6,
        b=0.3,
        theta=np.pi / 2,
    )
    e1, e2 = cap_basis([0, 0, 1])
    # e2 direction was minor (b=0.3); now after rotation 90°, e2 is major.
    along_e2 = sec.center + sec.a * e2
    assert section_oval_value(along_e2, sec) == pytest.approx(0.0, abs=1e-9)
    along_e1 = sec.center + sec.b * e1
    assert section_oval_value(along_e1, sec) == pytest.approx(0.0, abs=1e-9)


# -- shaft_section_oval_value -----------------------------------------------


def test_shaft_through_center_is_inside():
    # Vertical shaft passing exactly through (0, 0, 0); section at z=0.
    shaft = Capsule(np.array([0, 0, -1]), np.array([0, 0, 5]), radius=0.0)
    sec = HoleSection(
        axis=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        a=0.6,
        b=0.3,
        theta=0.0,
    )
    assert shaft_section_oval_value(shaft, sec) == pytest.approx(-1.0)


def test_shaft_offset_inside():
    # Shaft offset by (0.3 e1, 0.1 e2) from oval center — inside since
    # (0.3/0.6)^2 + (0.1/0.3)^2 = 0.25 + 0.111 = 0.361, g = -0.639.
    sec = HoleSection(
        axis=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        a=0.6,
        b=0.3,
        theta=0.0,
    )
    e1, e2 = cap_basis(sec.axis)
    base = sec.center + 0.3 * e1 + 0.1 * e2
    shaft = Capsule(base + np.array([0, 0, -1]), base + np.array([0, 0, 5]), 0.0)
    g = shaft_section_oval_value(shaft, sec)
    assert g == pytest.approx(-0.639, abs=1e-3)
    assert g < 0.0


def test_shaft_outside_oval():
    # Shaft offset 1.0 mm along e1 — past a = 0.6.
    sec = HoleSection(
        axis=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        a=0.6,
        b=0.3,
        theta=0.0,
    )
    e1, _ = cap_basis(sec.axis)
    base = sec.center + 1.0 * e1
    shaft = Capsule(base + np.array([0, 0, -1]), base + np.array([0, 0, 5]), 0.0)
    g = shaft_section_oval_value(shaft, sec)
    assert g == pytest.approx((1.0 / 0.6) ** 2 - 1.0)
    assert g > 0.0


def test_shaft_tilted_intersection_correct():
    # 30°-tilted shaft starting at the section center along -axis,
    # tilting toward +e1. Line p(t) = base + t * (sin(30°) e1 + cos(30°) axis).
    # Meets plane (axis dot (p - center) = 0) at t = 1 / cos(30°), where
    # the projection onto e1 is tan(30°).
    sec = HoleSection(
        axis=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        a=0.6,
        b=0.3,
        theta=0.0,
    )
    e1, _ = cap_basis(sec.axis)
    axis = sec.axis / np.linalg.norm(sec.axis)
    p0 = sec.center - axis  # 1 mm below the plane along -axis
    p1 = p0 + np.sin(np.pi / 6) * e1 + np.cos(np.pi / 6) * axis
    shaft = Capsule(p0, p1, radius=0.0)
    g = shaft_section_oval_value(shaft, sec)
    expected_g = (np.tan(np.pi / 6) / 0.6) ** 2 - 1.0
    assert g == pytest.approx(expected_g)


def test_shaft_parallel_to_plane_is_inf():
    # Shaft parallel to the section plane (no unique intersection).
    sec = HoleSection(
        axis=np.array([0, 0, 1]),
        center=np.array([0, 0, 0]),
        a=0.6,
        b=0.3,
        theta=0.0,
    )
    shaft = Capsule(np.array([0, 0, 1]), np.array([1, 0, 1]), radius=0.0)
    assert shaft_section_oval_value(shaft, sec) == float("inf")
