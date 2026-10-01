"""The series spec: what it reads and every statement it refuses."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from aind_rutter.series.spec import SeriesSpec, load_spec

Q, S = "quadbase-alpha", "2.1"


def spec_dict() -> dict[str, Any]:
    return {
        "name": "pair",
        "hemisphere": "left",
        "subjects": {
            "m1": {"base_config": "base.yml", "holes": "holes.yml"},
            "m2": {"base_config": "/abs/base.yml", "holes": "h.yml", "sets": ["one"]},
        },
        "inventory": {Q: 1, S: 2},
        "regions": {
            "vCA1": {"ccf": "CA1", "keep": ["ventral"]},
            "SI_HDB": {"ccf": ["SI", "NDB"]},
        },
        "sets": {
            "one": {
                "regions": ["MD", "VM", "vCA1"],
                "allocations": {
                    "3p": [
                        {"kind": Q, "regions": ["MD"]},
                        {"kind": S, "regions": ["VM"]},
                        {"kind": S, "regions": ["vCA1"]},
                    ],
                    "2p": [
                        {
                            "kind": Q,
                            "regions": ["VM", "MD"],
                            "name": "MD_VM",
                            "points_RAS": {"MD": [0.5, -1.0, -3.0]},
                        },
                        {"kind": S, "regions": ["vCA1"]},
                    ],
                },
            },
            "two": {
                "regions": ["SI_HDB"],
                "allocations": {"1p": [{"kind": S, "regions": ["SI_HDB"]}]},
            },
        },
    }


def _edited(path: str, value: Any) -> dict[str, Any]:
    """``spec_dict`` with the dotted ``path`` set to ``value``."""
    data = copy.deepcopy(spec_dict())
    *parents, last = path.split(".")
    node: Any = data
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node[key]
    node[last] = value
    return data


class TestReading:
    def test_a_region_is_its_definition_or_the_acronym_itself(self):
        spec = SeriesSpec.model_validate(spec_dict())
        assert spec.region("vCA1").ccf == ("CA1",)
        assert spec.region("vCA1").keep == ("ventral",)
        assert spec.region("SI_HDB").ccf == ("SI", "NDB")
        assert spec.region("MD").ccf == ("MD",) and spec.region("MD").keep == ()

    def test_a_subject_runs_every_set_unless_it_lists_some(self):
        spec = SeriesSpec.model_validate(spec_dict())
        assert spec.sets_of("m1") == ("one", "two")
        assert spec.sets_of("m2") == ("one",)

    def test_a_probe_is_named_after_its_regions_unless_it_says(self):
        one = SeriesSpec.model_validate(spec_dict()).sets["one"].allocations
        assert [p.probe_name for p in one["3p"]] == ["MD", "VM", "vCA1"]
        assert one["2p"][0].probe_name == "MD_VM"
        renamed = _edited("sets.one.allocations.2p.0.name", None)
        got = SeriesSpec.model_validate(renamed).sets["one"].allocations["2p"]
        assert got[0].probe_name == "VM_MD"

    def test_relative_paths_are_relative_to_the_file(self, tmp_path: Path):
        path = tmp_path / "specs" / "pair.series.yml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump(spec_dict()))
        spec = load_spec(path)
        assert spec.subjects["m1"].base_config == tmp_path / "specs" / "base.yml"
        assert spec.subjects["m1"].holes == tmp_path / "specs" / "holes.yml"
        assert spec.subjects["m2"].base_config == Path("/abs/base.yml")

    def test_a_ranker_table_is_relative_to_the_file_too(self, tmp_path: Path):
        data = spec_dict()
        data["solve"] = {"select_by": "learned", "ranker_table": "ranker/a.npz"}
        data["sets"]["two"]["solve"] = {"ranker_table": "ranker/b.npz"}
        path = tmp_path / "pair.series.yml"
        path.write_text(yaml.safe_dump(data))
        spec = load_spec(path)
        assert spec.solve_for("one").ranker_table == tmp_path / "ranker" / "a.npz"
        assert spec.solve_for("two").ranker_table == tmp_path / "ranker" / "b.npz"
        assert spec.solve_for("two").select_by == "learned"


REFUSED = [
    ("propose", {"probes": [5]}, "Extra inputs"),
    ("propose", {"max_group": 0}, "greater than or equal to 1"),
    ("propose", {"min_kept": 1.5}, "less than or equal to 1"),
    ("schedule", {"first": ["three"]}, "schedule first: unknown sets"),
    ("schedule", {"within": {"three": 2}}, "schedule within: unknown sets"),
    ("sets.one.score", "cells", "centroid"),
    ("hemisphere", "both", "left"),
    ("name", "a/b", "series name"),
    ("inventory", {Q: 0, S: 2}, "inventory"),
    ("regions.vCA1.ccf", "CA9", "not CCF acronyms"),
    ("regions.vCA1.keep", ["ventral", "dorsal"], "both"),
    ("regions.vCA1.keep", ["upper"], "keep"),
    ("regions.vCA1.cut", ["ventral"], "Extra inputs"),
    ("subjects.m2.sets", ["three"], "unknown sets"),
    ("subjects.m1.region_asset", {"loader": "trimesh"}, "may not state"),
    ("subjects.m1.region_target", {"hemisphere": "right"}, "may not state"),
    ("sets.one.regions", ["MD", "VM", "vCA1", "dCA1"], "neither defined nor CCF"),
    ("sets.one.regions", ["MD", "VM", "vCA1", "PL"], "records none of"),
    ("sets.one.regions", ["MD", "VM"], "does not list"),
    ("sets.two.regions", ["SI_HDB", "SI_HDB"], "listed twice"),
    ("sets.two.allocations.1p", [{"kind": Q, "regions": []}], "at least 1"),
    ("sets.one.allocations.3p.1.regions", ["VM", "MD"], "more than one probe"),
    ("sets.one.allocations.3p.1.kind", Q, "the inventory has 1"),
    ("sets.one.allocations.3p.1.kind", "2.4", "not in the inventory"),
    ("sets.one.allocations.3p.1.name", "MD", "used twice"),
    ("sets.one.allocations.3p.1.name", "V:M", "probe names"),
    ("sets.one.allocations.3p.1.points_RAS", {"MD": [0, 0, 0]}, "not on this probe"),
    ("sets.one.allocations.3p.1.points_RAS", {"VM": [0, 0]}, "points_RAS"),
]


@pytest.mark.parametrize("path, value, says", REFUSED)
def test_refuses(path: str, value: Any, says: str):
    with pytest.raises(ValidationError, match=says):
        SeriesSpec.model_validate(_edited(path, value))


def test_a_set_may_state_no_allocation_yet():
    data = spec_dict()
    del data["sets"]["two"]["allocations"]
    assert SeriesSpec.model_validate(data).sets["two"].allocations == {}


def test_an_allocation_name_is_a_file_name():
    data = spec_dict()
    data["sets"]["two"]["allocations"] = {"a b": [{"kind": S, "regions": ["SI_HDB"]}]}
    with pytest.raises(ValidationError, match="allocation names"):
        SeriesSpec.model_validate(data)
