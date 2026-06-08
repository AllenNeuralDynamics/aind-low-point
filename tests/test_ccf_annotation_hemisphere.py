"""Hemisphere-aware selection of lateralized CCF annotation regions.

Covers the label-level L/R split that replaced the geometric
``hemisphere_center_mass`` midsagittal cut, which collapsed for
near-midline nuclei (see ``ccf_annotation_region``).
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from aind_low_point.config import AssetSpecModel, DerivedTargetSpecModel
from aind_low_point.runtime.loaders import ccf_annotation_region


def _write_lateralized_volume(path: str, label: int = 685) -> None:
    """A signed annotation: a +label blob on the right, -label on the left.

    LPS +x is LEFT, and SimpleITK array axes are (z, y, x). We place the
    +label voxels at low x (right hemisphere) and -label voxels at high x
    (left hemisphere), with origin/spacing such that physical x straddles 0.
    """
    nx, ny, nz = 10, 6, 6
    arr = np.zeros((nz, ny, nx), dtype=np.int32)
    # right hemisphere (+label) on the lower-x half, left (-label) upper-x half
    arr[2:4, 2:4, 1:4] = label
    arr[2:4, 2:4, 6:9] = -label
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((-5.0, 0.0, 0.0))  # x in [-5, 4], midline near 0
    sitk.WriteImage(img, path)


class TestHemisphereSelection:
    def test_right_selects_positive_only(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        mesh = ccf_annotation_region(p, label_id=685, hemisphere="right")
        # +label blob sits at low array-x → high physical x (LPS left? no:
        # origin -5 + index → index 1..4 maps to x ~ -4..-1, i.e. right side)
        assert mesh.vertices[:, 0].mean() < 0

    def test_left_selects_negative_only(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        mesh = ccf_annotation_region(p, label_id=685, hemisphere="left")
        # -label blob at index 6..9 → physical x ~ +1..+4
        assert mesh.vertices[:, 0].mean() > 0

    def test_left_and_right_are_distinct(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        left = ccf_annotation_region(p, label_id=685, hemisphere="left")
        right = ccf_annotation_region(p, label_id=685, hemisphere="right")
        # The two hemispheres' centroids are on opposite sides of midline —
        # the exact failure the geometric split produced for midline nuclei.
        assert np.sign(left.vertices[:, 0].mean()) != np.sign(
            right.vertices[:, 0].mean()
        )

    def test_both_spans_full_range(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        both = ccf_annotation_region(p, label_id=685, hemisphere="both")
        left = ccf_annotation_region(p, label_id=685, hemisphere="left")
        right = ccf_annotation_region(p, label_id=685, hemisphere="right")
        lo = min(left.vertices[:, 0].min(), right.vertices[:, 0].min())
        hi = max(left.vertices[:, 0].max(), right.vertices[:, 0].max())
        assert both.vertices[:, 0].min() == pytest.approx(lo, abs=1e-6)
        assert both.vertices[:, 0].max() == pytest.approx(hi, abs=1e-6)

    def test_both_matches_legacy_on_unsigned_volume(self, tmp_path):
        """A non-lateralized (all-positive) annotation still works."""
        p = str(tmp_path / "unsigned.nii.gz")
        nx, ny, nz = 10, 6, 6
        arr = np.zeros((nz, ny, nx), dtype=np.int32)
        arr[2:4, 2:4, 1:4] = 685
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetOrigin((-5.0, 0.0, 0.0))
        sitk.WriteImage(img, p)
        mesh = ccf_annotation_region(p, label_id=685, hemisphere="both")
        assert len(mesh.vertices) > 0

    def test_no_match_raises(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p, label=685)
        with pytest.raises(ValueError, match="no voxels matched"):
            ccf_annotation_region(p, label_id=999, hemisphere="left")

    def test_invalid_hemisphere_rejected(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        with pytest.raises(ValueError, match="hemisphere must be"):
            ccf_annotation_region(p, label_id=685, hemisphere="middle")


class TestDerivedHemisphereExpand:
    """The config-level expansion that re-loads per-hemisphere at load time."""

    def _vm_asset(self) -> AssetSpecModel:
        return AssetSpecModel(
            key="structure:VM",
            src="/some/lateralized.nii.gz",
            loader="ccf_annotation_region",
            loader_kwargs={"acronym": "VM"},
        )

    def test_hemisphere_injects_loader_kwargs_and_drops_source_key(self):
        derived = DerivedTargetSpecModel(
            derive_from=["structure:VM"],
            key_prefix="target:L:",
            hemisphere="left",
            reducer="mesh_center_mass",
            templates=["structure"],
        )
        out = derived.expand({"structure:VM": self._vm_asset()})
        assert len(out) == 1
        t = out[0]
        assert t.key == "target:L:VM"
        # Re-loads rather than reusing the bilateral source mesh.
        assert t.source_key is None
        assert t.loader == "ccf_annotation_region"
        assert t.loader_kwargs == {"acronym": "VM", "hemisphere": "left"}
        assert t.reducer == "mesh_center_mass"
        assert str(t.src) == "/some/lateralized.nii.gz"

    def test_without_hemisphere_keeps_source_key(self):
        derived = DerivedTargetSpecModel(
            derive_from=["structure:VM"],
            key_prefix="target:",
            reducer="mesh_center_mass",
        )
        out = derived.expand({"structure:VM": self._vm_asset()})
        assert out[0].source_key == "structure:VM"
        assert out[0].loader is None

    def test_hemisphere_without_source_asset_raises(self):
        derived = DerivedTargetSpecModel(
            derive_from=["structure:VM"],
            key_prefix="target:L:",
            hemisphere="left",
        )
        with pytest.raises(ValueError, match="requires source asset"):
            derived.expand({})
