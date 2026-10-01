"""Halves of a CCF region: the cut rule, the loaders, and what scoring reads."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from aind_rutter.build.loaders import (
    ccf_annotation_region,
    ccf_region_bounds,
    ccf_region_label_ids,
    ccf_region_point_mask,
    ccf_region_voxel_points,
)
from aind_rutter.ccf.regions import half_mask, parse_halves
from aind_rutter.config import AssetSpecModel, DerivedTargetSpecModel
from aind_rutter.optimization.pipeline.probe_setup import region_definition

VM, MD = 685, 362
# A 4 x 6 x 8 mm block per hemisphere at 1 mm: x 1..4 is left, x -4..-1 right.
BLOCK = {"x": (1, 5), "y": (0, 6), "z": (0, 8)}


def _write_volume(path: str) -> None:
    arr = np.zeros((8, 6, 10), np.int32)  # (z, y, x), x index 0..9 -> -5..4 mm
    arr[:, :, 6:10] = -VM
    arr[:, :, 1:5] = VM
    arr[0:2, 0:2, 6:8] = -MD  # a second structure inside the left block
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin((-5.0, 0.0, 0.0))
    sitk.WriteImage(img, path)


@pytest.fixture
def volume(tmp_path) -> str:
    path = str(tmp_path / "lat.nii.gz")
    _write_volume(path)
    return path


class TestHalfMask:
    PTS = np.array(
        [[x, y, z] for x in (1.0, 3.0) for y in (1.0, 3.0) for z in (1.0, 3.0)]
    )
    BOX = dict(lo=np.array([0.0, 0.0, 0.0]), hi=np.array([4.0, 4.0, 4.0]))

    @pytest.mark.parametrize(
        "half, axis, upper",
        [
            ("dorsal", 2, True),
            ("ventral", 2, False),
            ("posterior", 1, True),
            ("anterior", 1, False),
        ],
    )
    def test_each_half_is_one_side_of_the_middle(self, half, axis, upper):
        got = half_mask(self.PTS, [half], hemisphere="both", **self.BOX)
        assert got.tolist() == ((self.PTS[:, axis] > 2.0) == upper).tolist()

    def test_medial_is_toward_the_midline_of_its_hemisphere(self):
        low_x = (self.PTS[:, 0] < 2.0).tolist()
        left = half_mask(self.PTS, ["medial"], hemisphere="left", **self.BOX)
        right = half_mask(self.PTS, ["medial"], hemisphere="right", **self.BOX)
        assert left.tolist() == low_x  # +x is left, so the midline is at low x
        assert right.tolist() == [not v for v in low_x]
        lateral = half_mask(self.PTS, ["lateral"], hemisphere="left", **self.BOX)
        assert lateral.tolist() == right.tolist()

    def test_halves_combine_and_opposites_partition(self):
        both = half_mask(
            self.PTS, ["dorsal", "posterior"], hemisphere="both", **self.BOX
        )
        assert both.sum() == 2
        for a, b in (("dorsal", "ventral"), ("anterior", "posterior")):
            one = half_mask(self.PTS, [a], hemisphere="both", **self.BOX)
            other = half_mask(self.PTS, [b], hemisphere="both", **self.BOX)
            assert (one ^ other).all()

    def test_medial_without_a_hemisphere_is_refused(self):
        with pytest.raises(ValueError, match="needs hemisphere"):
            half_mask(self.PTS, ["medial"], hemisphere="both", **self.BOX)

    def test_an_unknown_half_and_a_contradiction_are_refused(self):
        with pytest.raises(ValueError, match="keep must name halves"):
            parse_halves(["dorsla"])
        with pytest.raises(ValueError, match="both dorsal and ventral"):
            parse_halves(["dorsal", "ventral"])


class TestLoaders:
    def test_keep_halves_the_voxels_along_each_axis(self, volume):
        whole = ccf_region_voxel_points(volume, label_id=VM, hemisphere="right")
        assert len(whole) == 4 * 6 * 8
        dorsal = ccf_region_voxel_points(
            volume, label_id=VM, hemisphere="right", keep=["dorsal"]
        )
        assert len(dorsal) == len(whole) // 2 and dorsal[:, 2].min() > 3.5
        corner = ccf_region_voxel_points(
            volume,
            label_id=VM,
            hemisphere="right",
            keep=["dorsal", "medial", "anterior"],
        )
        assert len(corner) == len(whole) // 8
        assert corner[:, 0].min() >= -2.5  # the right hemisphere's midline is at high x

    def test_the_cut_is_at_the_middle_of_the_whole_region(self, volume):
        # MD takes a corner out of left VM; VM's own box is unchanged by it.
        lo, hi = ccf_region_bounds(
            sitk.ReadImage(volume), ccf_region_label_ids(label_id=VM, hemisphere="left")
        )
        assert lo.tolist() == [1.0, 0.0, 0.0] and hi.tolist() == [4.0, 5.0, 7.0]
        ventral = ccf_region_voxel_points(
            volume, label_id=VM, hemisphere="left", keep=["ventral"]
        )
        assert len(ventral) == 4 * 6 * 4 - 8 and ventral[:, 2].max() <= 3.5

    def test_the_mesh_wraps_the_kept_voxels(self, volume):
        kw = dict(label_id=VM, hemisphere="right", keep=["posterior", "lateral"])
        mesh = ccf_annotation_region(volume, **kw)
        voxels = ccf_region_voxel_points(volume, **kw)
        lo, hi = mesh.bounds
        np.testing.assert_allclose(lo, voxels.min(axis=0) - 0.5, atol=0.51)
        np.testing.assert_allclose(hi, voxels.max(axis=0) + 0.5, atol=0.51)

    def test_point_membership_agrees_with_the_voxels(self, volume):
        kw = dict(label_id=VM, hemisphere="left", keep=["dorsal", "lateral"])
        every = ccf_region_voxel_points(volume, label_id=VM, hemisphere="both")
        inside = ccf_region_point_mask(volume, every, **kw)
        kept = ccf_region_voxel_points(volume, **kw)
        assert sorted(map(tuple, every[inside])) == sorted(map(tuple, kept))

    def test_several_acronyms_are_one_region(self, volume):
        vm = ccf_region_voxel_points(volume, acronym="VM", hemisphere="left")
        md = ccf_region_voxel_points(volume, acronym="MD", hemisphere="left")
        both = ccf_region_voxel_points(volume, acronym=["VM", "MD"], hemisphere="left")
        assert len(md) == 8 and len(both) == len(vm) + len(md) == 4 * 6 * 8

    def test_a_cut_that_leaves_nothing_is_refused(self, volume):
        with pytest.raises(ValueError, match="no voxels matched"):
            ccf_region_voxel_points(
                volume, label_id=MD, hemisphere="right", keep=["dorsal"]
            )


class TestWhatScoringReads:
    def _spec(self, loader_kwargs, metadata, loader="ccf_annotation_region"):
        return SimpleNamespace(
            loader=loader, loader_kwargs=loader_kwargs, metadata=metadata
        )

    def test_the_loader_arguments_define_the_region(self):
        told = {"acronym": "CP", "hemisphere": "left", "keep": ["dorsal", "medial"]}
        assert region_definition(self._spec(told, {})) == {
            "acronym": "CP",
            "label_id": None,
            "hemisphere": "left",
            "keep": ("dorsal", "medial"),
        }

    def test_metadata_restates_what_it_names(self):
        spec = self._spec(
            {"acronym": "CA1"}, {"ccf_acronym": "CA1", "hemisphere": "left"}
        )
        got = region_definition(spec)
        assert got["hemisphere"] == "left" and got["keep"] == ()

    def test_another_loader_is_read_from_metadata_alone(self):
        spec = self._spec({"acronym": "CP"}, {"ccf_acronym": "MD"}, loader="trimesh")
        got = region_definition(spec)
        assert got["acronym"] == "MD" and got["hemisphere"] == "both"

    def test_a_derived_hemisphere_target_carries_the_cut(self):
        asset = AssetSpecModel(
            key="structure:pDMS",
            src="/some/lateralized.nii.gz",
            loader="ccf_annotation_region",
            loader_kwargs={"acronym": "CP", "keep": ["dorsal", "posterior"]},
        )
        derived = DerivedTargetSpecModel(
            derive_from=["structure:pDMS"],
            key_prefix="target:L:",
            hemisphere="left",
            reducer="points_mean",
        )
        (target,) = derived.expand({"structure:pDMS": asset})
        assert target.loader == "ccf_region_voxel_points"
        assert target.loader_kwargs == {
            "acronym": "CP",
            "keep": ["dorsal", "posterior"],
            "hemisphere": "left",
        }
