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
