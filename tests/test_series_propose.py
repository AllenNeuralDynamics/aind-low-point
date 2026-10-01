"""Proposed allocations: what is written, and that the spec takes it back."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from aind_rutter.config import ConfigModel
from aind_rutter.optimization.pipeline import propose as pipeline_propose
from aind_rutter.optimization.pipeline.propose import (
    Proposal,
    Proposals,
    ProposedProbe,
)
from aind_rutter.series import cli
from aind_rutter.series.configs import write_configs
from aind_rutter.series.propose import (
    propose_set,
    proposed_allocations,
    proposed_path,
    write_proposals,
)
from aind_rutter.series.spec import SeriesSpec
from tests.synthetic_subject import write_subject
from tests.test_series_configs import Q, S, _write_annotation


def _data(subject, annotation: Path) -> dict[str, Any]:
    """VM above MD under the first bore, and no allocation stated for either set."""
    return {
        "name": "pair",
        "hemisphere": "left",
        "subjects": {
            "m1": {
                "base_config": str(subject.config),
                "holes": str(subject.holes),
                "region_asset": {
                    "src": str(annotation),
                    "material_ref": "structure",
                    "transform": "headframe_to_lps",
                },
                "region_target": {"transform": "headframe_to_lps"},
            }
        },
        "inventory": {Q: 1, S: 2},
        "sets": {
            "both": {"regions": ["VM", "MD"]},
            "upper": {"regions": ["VM"], "solve": {"recorded_radius_mm": 0.3}},
        },
        "propose": {"max_points": 500},
    }


@pytest.fixture(scope="module")
def series(tmp_path_factory: pytest.TempPathFactory):
    subject = write_subject(tmp_path_factory.mktemp("subject"))
    annotation = subject.root / "annotation.nii.gz"
    _write_annotation(annotation)
    return _data(subject, annotation), tmp_path_factory.mktemp("run")


@pytest.fixture(scope="module")
def proposed(series) -> dict[str, Any]:
    data, run_dir = series
    said: list[str] = []
    (path,) = write_proposals(
        SeriesSpec.model_validate(data), run_dir, sets=["both"], report=said.append
    )
    assert path == proposed_path(run_dir, "m1", "both")
    return {"written": yaml.safe_load(path.read_text()), "said": "\n".join(said)}


class TestOnTheSubject:
    def test_one_proposal_per_probe_count_the_bores_allow(self, proposed):
        written = proposed["written"]
        # Two bores, so at most two probes; one line reaches both regions.
        assert list(written["allocations"]) == ["1p", "2p"]
        (shared,) = written["allocations"]["1p"]
        assert sorted(shared["regions"]) == ["MD", "VM"]
        assert sorted(shared["points_RAS"]) == ["MD", "VM"]
        assert all("points_RAS" not in p for p in written["allocations"]["2p"])

    def test_the_prediction_names_every_region_and_each_probes_bore(self, proposed):
        written = proposed["written"]
        for name, predicted in written["predicted"].items():
            assert sorted(predicted["regions"]) == ["MD", "VM"]
            assert all(0 < share <= 1 for share in predicted["regions"].values())
            assert predicted["mean"] == pytest.approx(
                np.mean(list(predicted["regions"].values())), abs=1e-3
            )
            bores = list(predicted["bores"].values())
            assert len(set(bores)) == len(written["allocations"][name])
            assert set(bores) <= {1, 2}

    def test_a_region_alone_records_most_on_the_wider_probe(self, proposed):
        for by_kind in proposed["written"]["alone"].values():
            assert by_kind[Q] > by_kind[S] > 0
        assert "share of each region's voxels" in proposed["said"]

    def test_pasted_into_the_spec_it_makes_configs_that_load(self, series, proposed):
        data, run_dir = series
        stated = copy.deepcopy(data)
        stated["sets"]["both"]["allocations"] = proposed["written"]["allocations"]
        spec = SeriesSpec.model_validate(stated)
        written = write_configs(spec, run_dir)
        assert [c.path.stem for c in written] == ["1p", "2p"]
        for config in written:
            probes = ConfigModel.from_yaml(config.path).plan.probes
            want = proposed["written"]["allocations"][config.path.stem]
            assert [p.kind for p in probes.values()] == [p["kind"] for p in want]


class TestTheSpecsPart:
    @pytest.fixture
    def asked(self, series, monkeypatch) -> dict[str, Any]:
        """What the pipeline is asked for, with a canned answer returned."""
        got: dict[str, Any] = {}

        def stand_in(config, holes, inventory, options):
            got.update(config=config, holes=holes, inventory=inventory, options=options)
            return Proposals(("VM",), "voxels", {"VM": {S: 0.5}}, ())

        monkeypatch.setattr(pipeline_propose, "propose", stand_in)
        return got

    def test_the_radius_and_the_cloud_are_the_sets(self, series, asked):
        data, run_dir = series
        spec = SeriesSpec.model_validate(data)
        propose_set(spec, run_dir, "m1", "upper")
        assert asked["options"].radius_mm == 0.3
        assert asked["options"].retro is False
        assert asked["options"].max_points == 500
        assert dict(asked["inventory"]) == {Q: 1, S: 2}
        regions = ConfigModel.from_yaml(asked["config"]).plan.probes
        assert [p.regions for p in regions.values()] == [["VM"]]

    def test_the_command_takes_a_subject_and_sets(
        self, series, asked, tmp_path, monkeypatch, capsys
    ):
        data, _ = series
        path = tmp_path / "pair.series.yml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        argv = ["propose", str(path), str(tmp_path / "run"), "--set", "upper"]
        monkeypatch.setattr("sys.argv", ["rutter-series", *argv])
        with pytest.raises(SystemExit) as stop:
            cli.main()
        assert stop.value.code == 0
        found = sorted((tmp_path / "run" / "m1" / "proposed").iterdir())
        assert [f.name for f in found] == ["upper.regions.yml", "upper.yml"]
        assert "upper: share of each region's voxels" in capsys.readouterr().out


def _probe(kind: str, *regions: str, point: bool = False) -> ProposedProbe:
    points = {r: np.array([1.0, 2.0, -3.0]) for r in regions} if point else {}
    return ProposedProbe(kind, regions, dict.fromkeys(regions, 0.5), 3, points)


def test_proposals_are_named_by_probe_count_and_points_are_ras():
    result = Proposals(
        ("MD", "VM"),
        "retro",
        {},
        (
            Proposal((_probe(Q, "MD", "VM", point=True),), 0.5),
            Proposal((_probe(S, "MD"), _probe(Q, "VM")), 0.5),
            Proposal((_probe(Q, "MD"), _probe(S, "VM")), 0.4),
        ),
    )
    got = proposed_allocations(result)
    assert list(got["allocations"]) == ["1p", "2p", "2p-2"]
    assert got["allocations"]["1p"] == [
        {
            "kind": Q,
            "regions": ["MD", "VM"],
            "points_RAS": {"MD": [-1.0, -2.0, -3.0], "VM": [-1.0, -2.0, -3.0]},
        }
    ]
    assert got["allocations"]["2p-2"][0] == {"kind": Q, "regions": ["MD"]}
    assert got["predicted"]["2p"] == {
        "mean": 0.5,
        "regions": {"MD": 0.5, "VM": 0.5},
        "bores": {"MD": 3, "VM": 3},
    }
