"""Geometric primitives for the placement optimizer.

Conventions
-----------
- All inputs in LPS-mm.
- A *capsule* is a swept sphere: a finite line segment with a radius. The
  probe shaft and headstage each get one (or several, in the multi-shank
  case).
- A *hole section* is a planar oval ``(center, axis, a, b, theta)`` where
  ``axis`` is the section plane's unit normal, ``(a, b)`` are the oval's
  major and minor half-extents, and ``theta`` is the rotation of the major
  axis relative to the ``e1`` basis vector built from ``axis`` (see
  :func:`cap_basis`).

All functions are pure-numpy. JAX/autograd variants will live in
``optimization.geometry_jax`` once the inner loop is wired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class Capsule:
    """Swept-sphere capsule: a segment between ``p0`` and ``p1`` with
    spherical end-caps of radius ``radius``."""

    p0: NDArray[np.floating]
    p1: NDArray[np.floating]
    radius: float


@dataclass(frozen=True, slots=True)
class HoleSection:
    """One planar oval cross-section of a bore.

    ``theta`` rotates the oval's major axis CCW from ``e1`` toward ``e2``
    in the basis returned by :func:`cap_basis(axis)`.
    """

    axis: NDArray[np.floating]
    center: NDArray[np.floating]
    a: float
    b: float
    theta: float


def cap_basis(axis: ArrayLike) -> tuple[NDArray, NDArray]:
    """Build an orthonormal ``(e1, e2)`` basis perpendicular to ``axis``.

    Same convention as ``scripts/extract_implant_holes.py`` so that
    ``theta`` from the extracted YAML lines up unchanged.
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, a)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, helper)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def point_to_segment_dist(p: ArrayLike, a: ArrayLike, b: ArrayLike) -> float:
    """Closest distance from point ``p`` to segment ``[a, b]``."""
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-24:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def segment_to_segment_dist(
    p0: ArrayLike, p1: ArrayLike, q0: ArrayLike, q1: ArrayLike
) -> float:
    """Closest distance between segments ``[p0, p1]`` and ``[q0, q1]``.

    Standard parametric formulation (Eberly), robust to parallel and
    degenerate (zero-length) cases.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    EPS = 1e-12

    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(p0 - q0))
    if a <= EPS:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= EPS:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if denom != 0.0:
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    closest1 = p0 + s * d1
    closest2 = q0 + t * d2
    return float(np.linalg.norm(closest1 - closest2))


def capsule_capsule_dist(c1: Capsule, c2: Capsule) -> float:
    """Signed distance between two capsules.

    Positive = clearance between surfaces (mm).
    Zero     = touching.
    Negative = interpenetration depth (mm).
    """
    seg_d = segment_to_segment_dist(c1.p0, c1.p1, c2.p0, c2.p1)
    return seg_d - (c1.radius + c2.radius)


def line_plane_intersect(
    line_p: ArrayLike,
    line_d: ArrayLike,
    plane_p: ArrayLike,
    plane_n: ArrayLike,
) -> float | None:
    """Return parameter ``t`` such that ``line_p + t * line_d`` lies on
    the plane defined by ``(plane_p, plane_n)``. Returns ``None`` when
    the line is parallel to the plane (no unique intersection)."""
    line_d = np.asarray(line_d, dtype=float)
    plane_n = np.asarray(plane_n, dtype=float)
    denom = float(np.dot(line_d, plane_n))
    if abs(denom) < 1e-12:
        return None
    return float(
        np.dot(
            np.asarray(plane_p, dtype=float) - np.asarray(line_p, dtype=float), plane_n
        )
        / denom
    )


def section_oval_value(point_3d: ArrayLike, section: HoleSection) -> float:
    """Evaluate ``g(point) = (u/a)^2 + (v/b)^2 - 1`` for the projection
    of ``point_3d`` onto the section's local 2D frame.

    ``g <= 0`` means the projected point lies inside the oval; ``g > 0``
    means outside.

    NB: assumes ``point_3d`` is already on the section plane. Use
    :func:`shaft_section_oval_value` to project first.
    """
    e1, e2 = cap_basis(section.axis)
    rel = np.asarray(point_3d, dtype=float) - section.center
    u_world = float(np.dot(rel, e1))
    v_world = float(np.dot(rel, e2))
    c, s = np.cos(section.theta), np.sin(section.theta)
    u = c * u_world + s * v_world
    v = -s * u_world + c * v_world
    return (u / section.a) ** 2 + (v / section.b) ** 2 - 1.0


def shaft_section_oval_value(shaft: Capsule, section: HoleSection) -> float:
    """Find where the shaft's *axis line* intersects the section plane,
    then evaluate the oval inequality at that point.

    Returns ``+inf`` when the shaft is parallel to the section plane
    (typically a wildly misaligned probe — should be rejected by the
    optimizer's outer loop, not the threading constraint itself).
    """
    line_p = np.asarray(shaft.p0, dtype=float)
    line_d = np.asarray(shaft.p1, dtype=float) - line_p
    t = line_plane_intersect(line_p, line_d, section.center, section.axis)
    if t is None:
        return float("inf")
    point_on_plane = line_p + t * line_d
    return section_oval_value(point_on_plane, section)
