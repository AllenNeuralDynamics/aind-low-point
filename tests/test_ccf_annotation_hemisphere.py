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
from aind_low_point.runtime.loaders import (
    ccf_annotation_region,
    ccf_region_membership,
    ccf_region_point_mask,
    ccf_region_voxel_points,
    voxel_values_at,
)


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


class TestVoxelCore:
    """The shared voxel-label selection core used by the reducer + optimizer."""

    def test_voxel_values_at_round_trips_labels_and_bounds(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        img = sitk.ReadImage(p)
        pts = np.array(
            [[-3.0, 2.2, 2.2], [2.2, 2.2, 2.2], [100.0, 0.0, 0.0]]
        )  # right blob, left blob, out of bounds
        vals, inb = voxel_values_at(img, pts)
        assert vals[0] == 685 and vals[1] == -685
        assert inb[0] and inb[1] and not inb[2]

    def test_region_membership_left_right_both(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        pts = np.array([[-3.0, 2.2, 2.2], [2.2, 2.2, 2.2], [0.0, 0.0, 0.0]])
        both = ccf_region_point_mask(p, pts, label_id=685, hemisphere="both")
        left = ccf_region_point_mask(p, pts, label_id=685, hemisphere="left")
        right = ccf_region_point_mask(p, pts, label_id=685, hemisphere="right")
        assert both.tolist() == [True, True, False]
        assert left.tolist() == [False, True, False]  # -685 blob is high-x (left)
        assert right.tolist() == [True, False, False]

    def test_point_mask_brain_mask_intersection(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        mask = np.zeros((6, 6, 10), dtype=np.uint8)
        mask[2:4, 2:4, 6:9] = 1  # left blob only
        mimg = sitk.GetImageFromArray(mask)
        mimg.SetSpacing((1.0, 1.0, 1.0))
        mimg.SetOrigin((-5.0, 0.0, 0.0))
        mpath = str(tmp_path / "brain.nii.gz")
        sitk.WriteImage(mimg, mpath)
        pts = np.array([[-3.0, 2.2, 2.2], [2.2, 2.2, 2.2]])
        keep = ccf_region_point_mask(
            p, pts, label_id=685, hemisphere="both", extra_mask_paths=(mpath,)
        )
        assert keep.tolist() == [False, True]

    def test_membership_helper_matches_point_mask(self, tmp_path):
        """The two callers (reducer one-shot vs optimizer cached) share the rule."""
        from aind_low_point.runtime.loaders import ccf_region_label_ids

        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        pts = np.array([[-3.0, 2.2, 2.2], [2.2, 2.2, 2.2], [0.0, 0.0, 0.0]])
        # Optimizer-style: sample once, then membership over cached labels.
        vals, inb = voxel_values_at(sitk.ReadImage(p), pts)
        ids = ccf_region_label_ids(label_id=685, hemisphere="left")
        cached = ccf_region_membership(vals, inb, ids)
        # Reducer-style: one-shot.
        oneshot = ccf_region_point_mask(p, pts, label_id=685, hemisphere="left")
        assert cached.tolist() == oneshot.tolist()

    def test_voxel_points_centroid_on_correct_side(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        left = ccf_region_voxel_points(p, label_id=685, hemisphere="left")
        right = ccf_region_voxel_points(p, label_id=685, hemisphere="right")
        assert left[:, 0].mean() > 0  # -label blob sits at high physical x
        assert right[:, 0].mean() < 0
        # Multi-label / descendants path: both hemispheres span the full range.
        both = ccf_region_voxel_points(p, label_id=685, hemisphere="both")
        assert both.shape[0] == left.shape[0] + right.shape[0]

    def test_voxel_points_no_match_raises(self, tmp_path):
        p = str(tmp_path / "lat.nii.gz")
        _write_lateralized_volume(p)
        with pytest.raises(ValueError, match="no voxels matched"):
            ccf_region_voxel_points(p, label_id=999, hemisphere="left")


class TestDerivedHemisphereExpand:
    """The config-level expansion that wires per-hemisphere voxel selection."""

    def _vm_asset(self, **kw) -> AssetSpecModel:
        return AssetSpecModel(
            key="structure:VM",
            src="/some/lateralized.nii.gz",
            loader="ccf_annotation_region",
            loader_kwargs={"acronym": "VM"},
            **kw,
        )

    def _retro_asset(self, **kw) -> AssetSpecModel:
        return AssetSpecModel(
            key="retro-targets",
            src="/some/retro_LPS.csv",
            loader="csv_points",
            kind="points",
            **kw,
        )

    def test_mesh_reducer_reloads_hemisphere_mesh(self):
        """Legacy mesh path (mesh_center_mass) re-loads the region as a mesh."""
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
        assert t.source_key is None
        assert t.loader == "ccf_annotation_region"
        assert t.loader_kwargs == {"acronym": "VM", "hemisphere": "left"}
        assert t.reducer == "mesh_center_mass"
        assert str(t.src) == "/some/lateralized.nii.gz"

    def test_points_mean_uses_voxel_points_loader(self):
        """Anatomical path: region voxel centroid via ccf_region_voxel_points."""
        derived = DerivedTargetSpecModel(
            derive_from=["structure:VM"],
            key_prefix="target:L:",
            hemisphere="left",
            reducer="points_mean",
        )
        out = derived.expand({"structure:VM": self._vm_asset()})
        t = out[0]
        assert t.source_key is None
        assert t.loader == "ccf_region_voxel_points"
        assert t.loader_kwargs == {"acronym": "VM", "hemisphere": "left"}
        assert t.reducer == "points_mean"
        assert str(t.src) == "/some/lateralized.nii.gz"

    def test_retro_injects_reducer_kwargs_and_keeps_source_key(self):
        """Retro path: keep the mesh source for rendering; select by voxel label."""
        derived = DerivedTargetSpecModel(
            derive_from=["structure:VM"],
            key_prefix="target:L:",
            hemisphere="left",
            reducer="points_in_region_center_mass",
            reducer_kwargs={"points_key": "retro-targets"},
        )
        out = derived.expand(
            {"structure:VM": self._vm_asset(), "retro-targets": self._retro_asset()}
        )
        t = out[0]
        assert t.source_key == "structure:VM"  # mesh renders, ignored by reducer
        assert t.loader is None
        assert t.reducer == "points_in_region_center_mass"
        assert t.reducer_kwargs == {
            "points_key": "retro-targets",
            "annotation_path": "/some/lateralized.nii.gz",
            "acronym": "VM",
            "hemisphere": "left",
        }

    def test_retro_guards_non_identity_canonicalization(self):
        """A retro points asset that is canonicalized off the volume frame is a
        silent-shift hazard — caught up front."""
        derived = DerivedTargetSpecModel(
            derive_from=["structure:VM"],
            key_prefix="target:L:",
            hemisphere="left",
            reducer="points_in_region_center_mass",
            reducer_kwargs={"points_key": "retro-targets"},
        )
        with pytest.raises(ValueError, match="identity canonicalization"):
            derived.expand(
                {
                    "structure:VM": self._vm_asset(),
                    "retro-targets": self._retro_asset(
                        canonicalization_ref="probe-mesh"
                    ),
                }
            )

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
            reducer="points_mean",
        )
        with pytest.raises(ValueError, match="requires source asset"):
            derived.expand({})
