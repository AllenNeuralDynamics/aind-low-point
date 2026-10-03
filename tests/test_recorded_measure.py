"""The per-region recorded share: its geometry, its frame, and emit's manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from aind_rutter.build.assemble import build_runtime_from_config
from aind_rutter.config import ConfigModel, PlanningModel
from aind_rutter.domain.scene import resolve_base_geometry
from aind_rutter.optimization.pipeline import emit
from aind_rutter.optimization.pipeline.measure import (
    ActiveSegment,
    active_segments,
    recorded_share,
    region_clouds,
    region_names,
    shank_geometry,
)
from aind_rutter.optimization.pipeline.payloads import write_handoff
from aind_rutter.optimization.pipeline.probe_setup import (
    RetroDensityOpts,
    retro_anchor_LPS,
)
from aind_rutter.optimization.pipeline.settings import EmitSettings
from aind_rutter.plan_io.plan_set import RecordedMeasure, load_plan_set
from aind_rutter.plan_io.replay import apply_plan_model_to_state
from aind_rutter.session.store import PlanStore
from tests.synthetic_subject import BRAIN_CENTER_LPS, write_subject

VM_LEFT = -685  # the lateralized annotation negates the left hemisphere's id
SPACING_MM = 0.2
BLOCK_VOXELS = (10, 10, 20)  # x, y, z: 2 x 2 x 4 mm around the brain's centre


def _write_annotation(path: Path) -> None:
    pad = 2
    nx, ny, nz = (n + 2 * pad for n in BLOCK_VOXELS)
    arr = np.zeros((nz, ny, nx), np.int32)
    arr[pad:-pad, pad:-pad, pad:-pad] = VM_LEFT
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((SPACING_MM,) * 3)
    centre = (np.array([nx, ny, nz]) - 1) * SPACING_MM / 2
    img.SetOrigin(tuple(np.array(BRAIN_CENTER_LPS) - centre))
    sitk.WriteImage(img, str(path))


@pytest.fixture(scope="module")
def config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic subject with a VM structure that probe P1 records."""
    subject = write_subject(tmp_path_factory.mktemp("subject"))
    annotation = subject.root / "annotation.nii.gz"
    _write_annotation(annotation)
    data = yaml.safe_load(subject.config.read_text())
    data["assets"].append(
        {
            "key": "structure:VM",
            "kind": "mesh",
            "role": "anatomy",
            "loader": "ccf_annotation_region",
            "loader_kwargs": {"acronym": "VM", "hemisphere": "left"},
            "src": str(annotation),
            "material_ref": "structure",
            "transform": "headframe_to_lps",
            "scene_tags": ["static"],
            "metadata": {"ccf_acronym": "VM", "hemisphere": "left"},
        }
    )
    data["plan"]["probes"]["P1"]["regions"] = ["VM"]
    path = subject.root / "with-structure.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


class TestRecordedShare:
    POINTS = np.array(
        [[0.0, 0.0, z] for z in (-0.5, 0.0, 1.0, 2.0, 2.5)]
        + [[0.3, 0.0, 1.0], [0.0, 0.1, 1.0]]
    )
    SEGMENT = ActiveSegment(np.zeros(3), np.array([0.0, 0.0, 1.0]), 2.0)

    def test_counts_points_within_the_radius_of_the_segment_and_its_ends(self):
        # Along the axis the two points 0.5 mm past either end are out; beside
        # it the one at 0.1 mm is in and the one at 0.3 mm is not.
        assert recorded_share(self.POINTS, [self.SEGMENT], 0.2) == pytest.approx(4 / 7)

    def test_the_ends_are_rounded(self):
        assert recorded_share(self.POINTS, [self.SEGMENT], 0.5) == pytest.approx(1.0)

    def test_any_segment_counts_once(self):
        twice = [self.SEGMENT, self.SEGMENT]
        assert recorded_share(self.POINTS, twice, 0.2) == pytest.approx(4 / 7)
        far = ActiveSegment(np.array([0.3, 0.0, 0.9]), np.array([0.0, 0.0, 1.0]), 0.2)
        assert recorded_share(self.POINTS, [self.SEGMENT, far], 0.2) == pytest.approx(
            5 / 7
        )

    def test_no_segments_record_nothing(self):
        assert recorded_share(self.POINTS, [], 0.2) == 0.0


class TestOnTheSubject:
    @pytest.fixture(scope="class")
    def built(self, config_path: Path):
        cfg = ConfigModel.from_yaml(config_path)
        return cfg, build_runtime_from_config(cfg)

    def test_a_probe_without_regions_is_named_after_its_structure(self, built):
        assert region_names(built[1].plan_state) == ["VM", "P2"]

    def test_voxels_share_the_scene_frame_of_the_structure_mesh(self, built):
        cfg, runtime = built
        clouds = region_clouds(cfg, runtime, None)
        assert list(clouds) == ["VM"]  # nothing is drawn for P2
        assert len(clouds["VM"]) == np.prod(BLOCK_VOXELS)
        mesh = resolve_base_geometry(
            runtime.asset_catalog, runtime.scene, "structure:VM"
        )
        lo, hi = mesh.raw.bounds
        assert (clouds["VM"] >= lo - 1e-6).all() and (clouds["VM"] <= hi + 1e-6).all()
        # The headframe transform moved them: this is not the file's own frame.
        assert not np.allclose(clouds["VM"].mean(axis=0), BRAIN_CENTER_LPS, atol=0.05)

    def test_retro_density_scores_the_cells_inside_the_structure(self, built):
        cfg, runtime = built
        # The synthetic brain is a mesh, which the retro brain mask cannot read.
        opts = RetroDensityOpts(common_mask_keys=())
        cells = region_clouds(cfg, runtime, opts)["VM"]
        mesh = resolve_base_geometry(
            runtime.asset_catalog, runtime.scene, "structure:VM"
        )
        lo, hi = mesh.raw.bounds
        every = resolve_base_geometry(
            runtime.asset_catalog, runtime.scene, "retro-targets"
        ).raw
        inside = ((every >= lo) & (every <= hi)).all(axis=1)
        assert 0 < len(cells) < len(every)
        assert len(cells) == pytest.approx(inside.sum(), abs=3)  # box vs voxel edges

    @pytest.mark.parametrize("opts", [None, RetroDensityOpts(common_mask_keys=())])
    def test_a_cut_structure_is_scored_on_its_kept_part(
        self, config_path: Path, built, opts, tmp_path: Path
    ):
        whole = len(region_clouds(*built, opts)["VM"])
        parts = []
        for half in ("dorsal", "ventral"):
            data = yaml.safe_load(config_path.read_text())
            data["assets"][-1]["loader_kwargs"]["keep"] = [half]
            path = tmp_path / f"{half}.yml"
            path.write_text(yaml.safe_dump(data, sort_keys=False))
            cfg = ConfigModel.from_yaml(path)
            try:
                clouds = region_clouds(cfg, build_runtime_from_config(cfg), opts)
            except RuntimeError as exc:  # the retro mask refuses an empty half
                assert "zero retro points" in str(exc)
                parts.append(0)
            else:
                parts.append(len(clouds["VM"]))
        assert max(parts) < whole or opts is not None
        assert sum(parts) == whole

    def test_every_shank_gives_one_segment_centred_on_the_target(self, built):
        _, runtime = built
        shanks = shank_geometry(runtime)
        segments = active_segments(runtime.plan_state, runtime, shanks)
        assert len(segments) == 1 + 4  # the 2.1, then the quad base
        for seg in segments:
            assert np.linalg.norm(seg.axis) == pytest.approx(1.0)
        # P1 aims at the brain's centre with nothing past it, so the middle of
        # its active stretch is the target.
        p1 = segments[0]
        target = runtime.plan_state.target_index["target:brain"]
        np.testing.assert_allclose(
            p1.start + p1.axis * p1.length_mm / 2, np.ravel(target), atol=1e-6
        )


def _handoff(path: Path) -> None:
    # arc APs, then per probe: ML, spin as (cos, sin), offsets R and A, depth.
    pose = np.array([10.0, -20.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                    + [-6.0, 0.0, 1.0, 0.0, 0.0, 0.5])  # fmt: skip
    record = {
        "idx": 0,
        "rank": 0,
        "n_arcs": 2,
        "fcl": 0.05,
        "coverage": 1.0,
        "pose": pose,
        "nit": 1,
        "secs": 0.0,
        "hole": {"P1": 1, "P2": 2},
        "partition": [["P1"], ["P2"]],
        "probe_to_arc_idx": {"P1": 0, "P2": 1},
        "arc_centroids_deg": [10.0, -20.0],
        "min_clear": 0.1,
    }
    write_handoff(path, {"all": [record], "ranked": [record], "config": {}})


@pytest.mark.parametrize("radius, expect", [(0.2, "part"), (50.0, "all"), (None, None)])
def test_emit_writes_the_recorded_share(
    config_path: Path, tmp_path: Path, radius, expect, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUTTER_RETRO_DENSITY", raising=False)
    _handoff(tmp_path / "handoff.json")
    out = tmp_path / "emitted"
    settings = EmitSettings(
        config=config_path,
        handoff=tmp_path / "handoff.json",
        outdir=out,
        recorded_radius_mm=radius,
    )
    assert emit.run(settings) == 0
    plan_set = load_plan_set(out)
    (entry,) = plan_set.entries
    if expect is None:
        assert plan_set.measure is None and entry.recorded == {}
        assert entry.mean_recorded() is None
        return
    assert plan_set.measure == RecordedMeasure("voxels", radius)
    assert list(entry.recorded) == ["VM"]
    if expect == "all":
        assert entry.recorded["VM"] == 1.0
    else:
        # One shank down a 2 x 2 mm column reaches a 0.4 mm core of it.
        assert 0.02 < entry.recorded["VM"] < 0.2
    assert entry.mean_recorded() == entry.recorded["VM"]
    assert json.loads((out / "manifest.json").read_text())["version"] == 2


def test_a_retro_plan_aims_where_the_solver_did(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = yaml.safe_load(config_path.read_text())
    data["plan"]["probes"]["P2"]["regions"] = ["VM"]  # retro density needs a mask
    config = tmp_path / "retro.yml"
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    # The synthetic brain is a mesh, which the retro brain mask cannot read.
    opts = RetroDensityOpts(common_mask_keys=())
    monkeypatch.setattr(emit, "retro_opts_from_env", lambda runtime=None: opts)
    _handoff(tmp_path / "handoff.json")
    out = tmp_path / "emitted"
    settings = EmitSettings(
        config=config, handoff=tmp_path / "handoff.json", outdir=out
    )
    assert emit.run(settings) == 0

    cfg = ConfigModel.from_yaml(config)
    runtime = build_runtime_from_config(cfg)
    anchor = retro_anchor_LPS(runtime, "P1", opts)
    (entry,) = load_plan_set(out).entries
    plan = PlanningModel.model_validate(
        yaml.safe_load((out / "plans" / entry.file).read_text())
    )
    store = PlanStore(runtime.plan_state)
    apply_plan_model_to_state(plan, store)
    p1 = active_segments(store.state, runtime, shank_geometry(runtime))[0]
    # P1 has no offset and no depth past its target, so its active stretch is
    # centred on whatever it aimed at.
    np.testing.assert_allclose(p1.start + p1.axis * p1.length_mm / 2, anchor, atol=1e-6)
    config_target = np.ravel(runtime.plan_state.target_index["target:brain"])
    assert np.linalg.norm(anchor - config_target) > 0.05
