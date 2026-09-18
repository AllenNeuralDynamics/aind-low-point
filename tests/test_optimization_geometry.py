"""Tests for ``aind_rutter.optimization.geometry.primitives``."""

from __future__ import annotations

import numpy as np
import pytest

from aind_rutter.optimization.geometry import cap_basis


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
