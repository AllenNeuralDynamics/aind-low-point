"""A config keeps what its author wrote: a saved plan replaces only the plan
block, and a relative location is read against the config's folder."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from aind_rutter.config import ConfigModel, PlanningModel
from aind_rutter.config.locations import Locator
from aind_rutter.plan_io.roundtrip import replace_plan_block

WRITTEN = """\
# Subject 123, set 0.
version: 1
paths:
  data: /mnt/Data/rutter   # this machine
  atlas: ${.data}/atlas.nii.gz
plan:
  # old arcs
  arcs:
    a: 0.0

# Assets follow.
assets:
- key: mesh:a
  src: ${paths.data}/a.obj
"""

NEW_PLAN = PlanningModel(arcs={"a": 12.5, "b": -3.0})


def test_only_the_plan_block_changes() -> None:
    out = replace_plan_block(WRITTEN, NEW_PLAN)
    head, _, rest = out.partition("plan:")
    assert head == WRITTEN.partition("plan:")[0]
    assert rest.endswith(
        "\n# Assets follow.\nassets:\n- key: mesh:a\n  src: ${paths.data}/a.obj\n"
    )
    assert "# old arcs" not in out
    assert yaml.safe_load(out)["plan"] == NEW_PLAN.model_dump(mode="json")


def test_a_last_plan_block_is_replaced_to_the_end() -> None:
    written = "version: 1\nplan:\n  arcs:\n    a: 0.0\n"
    out = replace_plan_block(written, NEW_PLAN)
    assert out.startswith("version: 1\nplan:\n")
    assert yaml.safe_load(out) == {
        "version": 1,
        "plan": NEW_PLAN.model_dump(mode="json"),
    }


@pytest.mark.parametrize("written", ["version: 1\n", "version: 1", ""])
def test_a_config_without_a_plan_gains_one(written: str) -> None:
    out = replace_plan_block(written, NEW_PLAN)
    assert out.startswith(written)
    assert yaml.safe_load(out)["plan"] == NEW_PLAN.model_dump(mode="json")


def test_a_top_level_it_cannot_splice_is_refused() -> None:
    with pytest.raises(ValueError, match="could not be replaced in place"):
        replace_plan_block("{version: 1, plan: {arcs: {a: 0.0}}}\n", NEW_PLAN)


def test_a_saved_config_loads_where_the_original_did(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    config = tmp_path / "c.yml"
    config.write_text(WRITTEN.replace("/mnt/Data/rutter", str(tmp_path / "data")))
    saved = tmp_path / "saved.yml"
    saved.write_text(replace_plan_block(config.read_text(), NEW_PLAN))
    cfg = ConfigModel.from_yaml(saved, Locator(machine_file=None))
    assert cfg.plan.arcs == {"a": 12.5, "b": -3.0}
    assert str(cfg.assets[0].src) == str(tmp_path / "data" / "a.obj")


def _load(config: Path) -> ConfigModel:
    return ConfigModel.from_yaml(config, Locator(machine_file=None))


def test_a_relative_location_is_read_against_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "proj"
    (folder / "transforms").mkdir(parents=True)
    (folder / "transforms" / "t.tfm").write_text("")
    config = folder / "c.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"meshes": "../shared"},
                "assets": [
                    {"key": "mesh:a", "src": "${paths.meshes}/a.obj"},
                    {"key": "mesh:b", "src": "/abs/b.obj"},
                    {"key": "mesh:c", "src": "Z:/ephys/c.obj"},
                    {"keys": ["structure:PL"], "src": "masks/{name}.nrrd"},
                ],
                "transforms": {
                    "t": {
                        "sequence": [{"kind": "sitk_file", "path": "transforms/t.tfm"}]
                    }
                },
            }
        )
    )
    monkeypatch.chdir(tmp_path)  # anywhere but the config's folder
    cfg = _load(config)
    src = {a.key: str(a.src) for a in cfg.assets}
    assert src["mesh:a"] == os.path.normpath(tmp_path / "shared" / "a.obj")
    assert src["mesh:b"] == str(Path("/abs/b.obj"))
    assert src["mesh:c"] == str(Path("Z:/ephys/c.obj"))
    assert src["structure:PL"] == str(folder / "masks" / "PL.nrrd")
    path = cfg.transforms["t"].sequence[0].path  # type: ignore[union-attr]
    assert path == folder / "transforms" / "t.tfm"


def test_a_config_built_in_code_keeps_the_working_directory() -> None:
    cfg = ConfigModel.model_validate({"assets": [{"key": "mesh:a", "src": "a.obj"}]})
    assert cfg.assets[0].src == Path("a.obj")
