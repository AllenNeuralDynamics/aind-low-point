"""Tests for the reducer registry — built-in reducers used by TargetSpec."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from aind_low_point.build_runtime import _REDUCER_REGISTRY


def _bilateral_box() -> trimesh.Trimesh:
    """A 2 mm cube centred at the origin — symmetric in ML."""
    return trimesh.creation.box(extents=(2.0, 2.0, 2.0))


def _shifted_box(centre_ml: float) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.apply_translation([centre_ml, 0.0, 0.0])
    return box


class TestHemisphereCenterMass:
    def test_left_of_symmetric_box(self):
        m = _bilateral_box()
        c = _REDUCER_REGISTRY["hemisphere_center_mass"](m, hemisphere="left")
        assert c[0] > 0
        assert c[0] == pytest.approx(0.5, abs=1e-3)
        assert c[1] == pytest.approx(0.0, abs=1e-3)
        assert c[2] == pytest.approx(0.0, abs=1e-3)

    def test_right_of_symmetric_box(self):
        m = _bilateral_box()
        c = _REDUCER_REGISTRY["hemisphere_center_mass"](m, hemisphere="right")
        assert c[0] < 0
        assert c[0] == pytest.approx(-0.5, abs=1e-3)

    def test_l_alias(self):
        m = _bilateral_box()
        c1 = _REDUCER_REGISTRY["hemisphere_center_mass"](m, hemisphere="left")
        c2 = _REDUCER_REGISTRY["hemisphere_center_mass"](m, hemisphere="L")
        np.testing.assert_allclose(c1, c2, atol=1e-6)

    def test_r_alias_case_insensitive(self):
        m = _bilateral_box()
        c1 = _REDUCER_REGISTRY["hemisphere_center_mass"](m, hemisphere="right")
        c2 = _REDUCER_REGISTRY["hemisphere_center_mass"](m, hemisphere="r")
        c3 = _REDUCER_REGISTRY["hemisphere_center_mass"](m, hemisphere="RIGHT")
        np.testing.assert_allclose(c1, c2, atol=1e-6)
        np.testing.assert_allclose(c1, c3, atol=1e-6)

    def test_invalid_hemisphere_rejected(self):
        m = _bilateral_box()
        with pytest.raises(ValueError, match="hemisphere"):
            _REDUCER_REGISTRY["hemisphere_center_mass"](m, hemisphere="middle")

    def test_falls_back_to_centroid_when_all_one_side(self):
        # Box entirely on the left (positive ML); asking for the right
        # hemisphere should fall back to the overall mean rather than
        # returning NaN or crashing.
        m = _shifted_box(centre_ml=2.0)
        c = _REDUCER_REGISTRY["hemisphere_center_mass"](m, hemisphere="right")
        # All vertices have ML > 0, so the right-side mask is empty and
        # we fall back to the full vertex mean.
        assert np.isfinite(c).all()
        assert c[0] > 0  # mean of all vertices, all on the left

    def test_custom_split_plane(self):
        # Box centred at ML=1, asking for "left of plane=2" should be
        # the part with ML < 2; everything qualifies → centre near 1.
        m = _shifted_box(centre_ml=1.0)
        c = _REDUCER_REGISTRY["hemisphere_center_mass"](
            m, hemisphere="right", plane=2.0
        )
        # Right-of-plane=2 means x < 2; the whole box qualifies.
        assert c[0] == pytest.approx(1.0, abs=1e-3)

    def test_works_on_ndarray_input(self):
        # When the source is a point cloud (N, 3), the slice-plane path
        # isn't available — should fall through to the vertex-mask
        # branch and still split correctly.
        pts = np.array(
            [
                [+1.0, 0.0, 0.0],
                [+2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
            ]
        )
        cl = _REDUCER_REGISTRY["hemisphere_center_mass"](pts, hemisphere="left")
        cr = _REDUCER_REGISTRY["hemisphere_center_mass"](pts, hemisphere="right")
        assert cl[0] == pytest.approx(1.5, abs=1e-6)
        assert cr[0] == pytest.approx(-1.5, abs=1e-6)
