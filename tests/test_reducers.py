"""Tests for the reducer registry — built-in reducers used by TargetSpec."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from aind_low_point.build_runtime import _REDUCER_REGISTRY
from aind_low_point.runtime.reducers import EmptyReductionError


def _bilateral_box() -> trimesh.Trimesh:
    """A 2 mm cube centred at the origin — symmetric in ML."""
    return trimesh.creation.box(extents=(2.0, 2.0, 2.0))


class TestPointsInRegionCenterMass:
    """``points_in_region_center_mass`` containment tests."""

    def test_mean_of_contained_points(self):
        region = _bilateral_box()  # 2 mm cube at origin
        pts = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [5.0, 5.0, 5.0]])
        c = _REDUCER_REGISTRY["points_in_region_center_mass"](region, points=pts)
        # The far point is outside the cube; the two inside average to origin.
        assert c == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    def test_empty_selection_raises_empty_reduction_error(self):
        region = _bilateral_box()
        pts = np.array([[5.0, 5.0, 5.0]])  # outside region
        with pytest.raises(EmptyReductionError):
            _REDUCER_REGISTRY["points_in_region_center_mass"](region, points=pts)
