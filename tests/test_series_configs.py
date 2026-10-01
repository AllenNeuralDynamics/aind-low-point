"""Series spec to configs: what is written, what it builds to, and the lock."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from aind_rutter.build.assemble import build_runtime_from_config
from aind_rutter.config import ConfigModel
from aind_rutter.optimization.pipeline.measure import region_clouds
from aind_rutter.optimization.pipeline.probe_setup import RetroDensityOpts
from aind_rutter.series.configs import STRUCTURE_KEY, config_path, write_configs
from aind_rutter.series.lock import LOCK_NAME, file_sha256
from aind_rutter.series.spec import SeriesSpec
from tests.synthetic_subject import BRAIN_CENTER_LPS, write_subject

Q, S = "quadbase-alpha", "2.1"
VM, MD = 685, 362
SPACING_MM = 0.2
INLINE_RAS = (-2.5, 0.25, -4.0)


def _write_annotation(path: Path) -> None:
    """Left VM above left MD, each 2 x 2 x 2 mm, around the brain's centre."""
    arr = np.zeros((24, 14, 14), np.int32)  # z, y, x
    arr[12:22, 2:12, 2:12] = -VM
    arr[2:12, 2:12, 2:12] = -MD
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((SPACING_MM,) * 3)
    centre = (np.array([14, 14, 24]) - 1) * SPACING_MM / 2
    img.SetOrigin(tuple(np.array(BRAIN_CENTER_LPS) - centre))
    sitk.WriteImage(img, str(path))


def _spec(subject, annotation: Path, **edits: Any) -> SeriesSpec:
    one = {
        "base_config": str(subject.config),
        "holes": str(subject.holes),
        "region_asset": {
            "src": str(annotation),
            "material_ref": "structure",
            "transform": "headframe_to_lps",
        },
        "region_target": {"transform": "headframe_to_lps"},
    }
    data = {
        "name": "pair",
        "hemisphere": "left",
        "subjects": {"m1": one, "m2": {**one, "sets": ["whole"]}},
        "inventory": {Q: 1, S: 2},
        "regions": {"vVM": {"ccf": "VM", "keep": ["ventral"]}},
        "sets": {
            "whole": {
                "regions": ["VM", "MD"],
                "allocations": {
                    "2p": [
                        {"kind": S, "regions": ["VM"]},
                        {"kind": Q, "regions": ["MD"]},
                    ],
                    "1p": [
                        {
                            "kind": Q,
                            "regions": ["MD", "VM"],
                            "name": "VM_MD",
                            "points_RAS": {"VM": list(INLINE_RAS)},
                        }
                    ],
                },
            },
            "cut": {
                "regions": ["vVM", "MD", "VM"],
                "allocations": {
                    "3p": [{"kind": S, "regions": [r]} for r in ("vVM", "MD")]
                    + [{"kind": Q, "regions": ["VM"]}]
                },
            },
        },
    }
    return SeriesSpec.model_validate(data | edits)


@pytest.fixture(scope="module")
def subject(tmp_path_factory: pytest.TempPathFactory):
    return write_subject(tmp_path_factory.mktemp("subject"))


@pytest.fixture(scope="module")
def annotation(subject) -> Path:
    path = subject.root / "annotation.nii.gz"
    _write_annotation(path)
    return path


@pytest.fixture(scope="module")
def run(subject, annotation, tmp_path_factory: pytest.TempPathFactory):
    run_dir = tmp_path_factory.mktemp("run")
    spec = _spec(subject, annotation)
    return run_dir, spec, write_configs(spec, run_dir)


class TestWhatIsWritten:
    def test_one_config_per_subject_set_and_allocation(self, run):
        run_dir, _, written = run
        got = [str(c.path.relative_to(run_dir)) for c in written]
        assert got == [
            "m1/configs/whole/2p.yml",
            "m1/configs/whole/1p.yml",
            "m1/configs/cut/3p.yml",
            "m2/configs/whole/2p.yml",
            "m2/configs/whole/1p.yml",
        ]
        assert all(c.path.is_file() for c in written)

    def test_the_probes_are_the_allocations_on_the_bases_arcs_in_turn(self, run):
        run_dir, _, _ = run
        plan = yaml.safe_load(config_path(run_dir, "m1", "cut", "3p").read_text())[
            "plan"
        ]
        assert plan["arcs"] == {"a": 10.0, "b": -20.0}
        assert plan["probes"] == {
            "vVM": {
                "kind": S,
                "arc": "a",
                "target": {"kind": "catalog", "key": "target:vVM"},
                "regions": ["vVM"],
            },
            "MD": {
                "kind": S,
                "arc": "b",
                "target": {"kind": "catalog", "key": "target:MD"},
                "regions": ["MD"],
            },
            "VM": {
                "kind": Q,
                "arc": "a",
                "target": {"kind": "catalog", "key": "target:VM"},
                "regions": ["VM"],
            },
        }

    def test_a_shared_probe_anchors_on_its_first_region(self, run):
        run_dir, _, _ = run
        probes = yaml.safe_load(config_path(run_dir, "m1", "whole", "1p").read_text())[
            "plan"
        ]["probes"]
        assert probes == {
            "VM_MD": {
                "kind": Q,
                "arc": "a",
                "target": {"kind": "catalog", "key": "target:MD"},
                "also_record": [{"kind": "inline", "point_RAS": list(INLINE_RAS)}],
                "regions": ["MD", "VM"],
            }
        }

    def test_a_region_is_one_structure_asset_carrying_its_cut(self, run):
        run_dir, _, _ = run
        cfg = yaml.safe_load(config_path(run_dir, "m1", "cut", "3p").read_text())
        assets = {a["key"]: a for a in cfg["assets"]}
        assert assets["structure:vVM"]["loader_kwargs"] == {
            "acronym": "VM",
            "hemisphere": "left",
            "keep": ["ventral"],
        }
        assert assets["structure:VM"]["loader_kwargs"] == {
            "acronym": "VM",
            "hemisphere": "left",
        }
        assert assets["structure:MD"]["transform"] == "headframe_to_lps"
        assert "brain" in assets  # the base's own assets stay

    def test_the_structure_key_is_the_one_scoring_looks_up(self):
        assert STRUCTURE_KEY == RetroDensityOpts().region_mask_fmt.replace(
            "{region}", "{label}"
        )


class TestWhatItBuildsTo:
    @pytest.fixture(scope="class")
    def built(self, run):
        run_dir, _, _ = run
        cfg = ConfigModel.from_yaml(config_path(run_dir, "m1", "cut", "3p"))
        runtime = build_runtime_from_config(cfg)
        return runtime, region_clouds(cfg, runtime, None)

    def test_every_region_has_a_cloud_and_a_target_inside_it(self, built):
        runtime, clouds = built
        assert set(clouds) == {"vVM", "MD", "VM"}
        for label, cloud in clouds.items():
            target = np.asarray(runtime.targets_pts[f"target:{label}"]).reshape(3)
            nearest = np.linalg.norm(cloud - target, axis=1).min()
            assert nearest < SPACING_MM

    def test_the_cut_region_is_the_lower_half(self, built):
        _, clouds = built
        assert len(clouds["vVM"]) * 2 == len(clouds["VM"])
        # The subject's transform tilts z by two degrees, well under the cut.
        assert clouds["vVM"][:, 2].mean() < clouds["VM"][:, 2].mean()

    def test_a_stated_point_is_where_the_probe_aims(self, run):
        run_dir, _, _ = run
        cfg = ConfigModel.from_yaml(config_path(run_dir, "m1", "whole", "1p"))
        plan = build_runtime_from_config(cfg).plan_state.probes["VM_MD"]
        assert plan.target_key == "target:MD"
        assert plan.regions == ("MD", "VM")
        r, a, s = INLINE_RAS
        assert plan.also_record[0].point_LPS == pytest.approx((-r, -a, s))


class TestTheLock:
    def test_records_the_spec_the_configs_and_every_input(self, run, subject):
        run_dir, spec, written = run
        lock = yaml.safe_load((run_dir / LOCK_NAME).read_text())
        assert lock["series"] == "pair"
        assert SeriesSpec.model_validate(lock["spec"]) == spec
        assert lock["configs"] == {
            str(c.path.relative_to(run_dir)): file_sha256(c.path) for c in written
        }
        inputs = lock["inputs"]
        for path in (
            subject.config,
            subject.holes,
            subject.root / "annotation.nii.gz",
            subject.root / "meshes" / "implant.obj",
            subject.root / "meshes" / "probe-2.1.obj",
        ):
            assert inputs[str(path.resolve())] == file_sha256(path)
        assert set(lock["aind_rutter"]) == {"version", "commit"}

    def test_a_missing_input_has_no_hash(self, subject, annotation, tmp_path: Path):
        spec = _spec(subject, annotation)
        gone = str(tmp_path / "no.holes.yml")
        data = spec.model_dump(mode="json")
        data["subjects"]["m1"]["holes"] = gone
        write_configs(SeriesSpec.model_validate(data), tmp_path)
        lock = yaml.safe_load((tmp_path / LOCK_NAME).read_text())
        assert lock["inputs"][gone] is None
        assert lock["spec_file"] is None


class TestRefusals:
    def _base(self, subject, tmp_path: Path, edit) -> Path:
        base = yaml.safe_load(subject.config.read_text())
        edit(base)
        path = tmp_path / "base.yml"
        path.write_text(yaml.safe_dump(base, sort_keys=False))
        return path

    def _with_base(self, subject, annotation, base: Path, **edits: Any) -> SeriesSpec:
        data = _spec(subject, annotation, **edits).model_dump(mode="json")
        for one in data["subjects"].values():
            one["base_config"] = str(base)
        return SeriesSpec.model_validate(data)

    def test_a_base_that_already_defines_a_region(self, subject, annotation, tmp_path):
        base = self._base(
            subject,
            tmp_path,
            lambda b: b["assets"].append({**b["assets"][0], "key": "structure:MD"}),
        )
        with pytest.raises(ValueError, match="m1 / whole / 2p.*already defines"):
            write_configs(self._with_base(subject, annotation, base), tmp_path / "run")

    def test_a_base_without_arcs(self, subject, annotation, tmp_path):
        base = self._base(subject, tmp_path, lambda b: b["plan"].pop("arcs"))
        with pytest.raises(ValueError, match="no plan.arcs"):
            write_configs(self._with_base(subject, annotation, base), tmp_path / "run")

    def test_a_kind_the_base_has_no_probe_for(self, subject, annotation, tmp_path):
        data = _spec(subject, annotation).model_dump(mode="json")
        data["inventory"] = {"2.4": 3}
        for probe in data["sets"]["whole"]["allocations"]["2p"]:
            probe["kind"] = "2.4"
        del data["sets"]["whole"]["allocations"]["1p"], data["sets"]["cut"]
        with pytest.raises(ValueError, match="m1 / whole / 2p.*does not load"):
            write_configs(SeriesSpec.model_validate(data), tmp_path / "run")
