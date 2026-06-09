"""Tests for the reducer registry — built-in reducers used by TargetSpec."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from aind_low_point.build_runtime import _REDUCER_REGISTRY
from aind_low_point.runtime.reducers import EmptyReductionError


def _write_signed_volume(path: str, label: int = 685) -> None:
    """A lateralized annotation: +label on the right (low physical x), -label on
    the left (high physical x). Mirrors ``test_ccf_annotation_hemisphere``: LPS
    +x is LEFT, origin set so physical x straddles 0.
    """
    nx, ny, nz = 10, 6, 6
    arr = np.zeros((nz, ny, nx), dtype=np.int32)
    arr[2:4, 2:4, 1:4] = label  # right hemisphere (+label), physical x ~ -4..-2
    arr[2:4, 2:4, 6:9] = -label  # left hemisphere (-label), physical x ~ +1..+3
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((-5.0, 0.0, 0.0))
    sitk.WriteImage(img, path)


# A small cloud: 2 points in the right (+label) blob, 2 in the left (-label)
# blob, 1 well outside any region.
_RIGHT_PTS = np.array([[-3.0, 2.2, 2.2], [-2.2, 2.2, 2.2]])
_LEFT_PTS = np.array([[1.2, 2.2, 2.2], [2.2, 2.2, 2.2]])
_OUTSIDE = np.array([[0.0, 0.0, 0.0]])


class TestPointsInRegionCenterMass:
    """Voxel-label selection of retro points within a CCF region."""

    def _cloud(self) -> np.ndarray:
        return np.vstack([_RIGHT_PTS, _LEFT_PTS, _OUTSIDE])

    def test_both_hemispheres_selects_all_region_points(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_signed_volume(p)
        c = _REDUCER_REGISTRY["points_in_region_center_mass"](
            None,
            points=self._cloud(),
            annotation_path=p,
            label_id=685,
            hemisphere="both",
        )
        # The outside point is excluded; the 4 region points average.
        assert c == pytest.approx(np.vstack([_RIGHT_PTS, _LEFT_PTS]).mean(0))

    def test_left_selects_negated_label_only(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_signed_volume(p)
        c = _REDUCER_REGISTRY["points_in_region_center_mass"](
            None,
            points=self._cloud(),
            annotation_path=p,
            label_id=685,
            hemisphere="left",
        )
        assert c == pytest.approx(_LEFT_PTS.mean(0))

    def test_right_selects_positive_label_only(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_signed_volume(p)
        c = _REDUCER_REGISTRY["points_in_region_center_mass"](
            None,
            points=self._cloud(),
            annotation_path=p,
            label_id=685,
            hemisphere="right",
        )
        assert c == pytest.approx(_RIGHT_PTS.mean(0))

    def test_brain_mask_further_restricts(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_signed_volume(p)
        # Brain mask non-zero only over the left blob's voxels.
        mask = np.zeros((6, 6, 10), dtype=np.uint8)
        mask[2:4, 2:4, 6:9] = 1
        mimg = sitk.GetImageFromArray(mask)
        mimg.SetSpacing((1.0, 1.0, 1.0))
        mimg.SetOrigin((-5.0, 0.0, 0.0))
        mpath = str(tmp_path / "brain.nii.gz")
        sitk.WriteImage(mimg, mpath)
        c = _REDUCER_REGISTRY["points_in_region_center_mass"](
            None,
            points=self._cloud(),
            annotation_path=p,
            label_id=685,
            hemisphere="both",
            brain_mask_paths=(mpath,),
        )
        # Bilateral region ∩ left-only brain mask → just the left points.
        assert c == pytest.approx(_LEFT_PTS.mean(0))

    def test_empty_selection_raises(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_signed_volume(p)
        with pytest.raises(EmptyReductionError):
            _REDUCER_REGISTRY["points_in_region_center_mass"](
                None,
                points=_OUTSIDE,
                annotation_path=p,
                label_id=685,
                hemisphere="both",
            )


class TestPointsMean:
    def test_mean_of_cloud(self):
        pts = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        c = _REDUCER_REGISTRY["points_mean"](pts)
        assert c == pytest.approx([2.0, 3.0, 4.0])

    def test_empty_raises(self):
        with pytest.raises(EmptyReductionError):
            _REDUCER_REGISTRY["points_mean"](np.zeros((0, 3)))
