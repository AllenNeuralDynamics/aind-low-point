"""Where a config's ``paths:`` names resolve: each source, and the order they win in."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from aind_rutter.config import ConfigModel
from aind_rutter.config.locations import (
    CONFIG_SOURCE,
    OVERRIDE_SOURCE,
    PATHS_FILE_NAME,
    Locator,
    load_resolved,
    locate,
    parse_overrides,
    read_paths_file,
)
from aind_rutter.optimization.pipeline.settings import PipelineSettings


def _write(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A config two folders below a project root, with no paths file yet."""
    _write(
        tmp_path / "proj" / "configs" / "c.yml",
        {
            "paths": {
                "data": "/written/data",
                "atlas": "${.data}/atlas.nii.gz",
                "meshes": "/written/meshes",
            }
        },
    )
    return tmp_path


def _config(tree: Path) -> Path:
    return tree / "proj" / "configs" / "c.yml"


def _paths(tree: Path, locator: Locator) -> dict:
    return load_resolved(_config(tree), locator)["paths"]


def _isolated(**kw) -> Locator:
    return Locator(machine_file=None, **kw)


def test_the_config_value_stands_when_nothing_else_sets_it(tree: Path) -> None:
    assert _paths(tree, _isolated()) == {
        "data": "/written/data",
        "atlas": "/written/data/atlas.nii.gz",
        "meshes": "/written/meshes",
    }


def test_a_name_derived_from_another_follows_it(tree: Path) -> None:
    _write(tree / "proj" / PATHS_FILE_NAME, {"data": "/here/data"})
    assert _paths(tree, _isolated())["atlas"] == "/here/data/atlas.nii.gz"


def test_sources_win_in_order(tree: Path) -> None:
    machine = _write(tree / "machine.yml", {"data": "/machine", "meshes": "/m"})
    far = _write(tree / "proj" / PATHS_FILE_NAME, {"data": "/far", "meshes": "/far"})
    _write(tree / "proj" / "configs" / PATHS_FILE_NAME, {"data": "/near"})
    given = _write(tree / "given.yml", {"data": "/given"})
    given_second = _write(tree / "given2.yml", {"data": "/given2"})
    config = _config(tree)

    def winners(**kw) -> dict[str, tuple[str, str]]:
        found = locate(
            config,
            {"data": "/written/data", "meshes": "/written/meshes"},
            Locator(machine_file=machine, **kw),
        )
        return {n: (loc.value, loc.source) for n, loc in found.items()}

    near = str(tree / "proj" / "configs" / PATHS_FILE_NAME)
    assert winners() == {"data": ("/near", near), "meshes": ("/far", str(far))}
    assert winners(files=(given, given_second))["data"] == ("/given", str(given))
    assert winners(files=(given,), overrides={"data": "/cli"})["data"] == (
        "/cli",
        OVERRIDE_SOURCE,
    )
    assert winners(search_upward=False) == {
        "data": ("/machine", str(machine)),
        "meshes": ("/m", str(machine)),
    }


def test_the_environment_ranks_below_the_command_line(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    given = _write(tree / "given.yml", {"data": "/given"})
    env_a = _write(tree / "a.yml", {"data": "/env-a", "meshes": "/env-a"})
    env_b = _write(tree / "b.yml", {"meshes": "/env-b"})
    monkeypatch.setenv("RUTTER_PATHS", os.pathsep.join([str(env_a), str(env_b)]))
    locator = Locator.from_environment([given])
    assert locator.files == (given, env_a, env_b)
    assert PipelineSettings().paths == (env_a, env_b)
    found = _paths(tree, locator)
    assert (found["data"], found["meshes"]) == ("/given", "/env-a")


def test_the_machine_file_sits_in_the_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUTTER_CONFIG_DIR", str(tmp_path))
    assert Locator().machine_file == tmp_path / "paths.yml"


def test_names_the_config_does_not_declare_are_ignored(tree: Path) -> None:
    _write(tree / "proj" / PATHS_FILE_NAME, {"other": "/x"})
    assert "other" not in _paths(tree, _isolated())


def test_an_override_for_an_undeclared_name_is_refused(tree: Path) -> None:
    with pytest.raises(ValueError, match="dta.*declared: atlas, data, meshes"):
        _paths(tree, _isolated(overrides={"dta": "/x"}))


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("sub/dir", "{base}/sub/dir"),
        ("../up", "{base}/../up"),
        ("/abs", "/abs"),
        ("Z:/ephys/data", "Z:/ephys/data"),
        ("C:\\ephys\\data", "C:\\ephys\\data"),
        ("\\\\server\\share", "\\\\server\\share"),
    ],
)
def test_a_relative_folder_is_taken_against_its_file(
    tmp_path: Path, written: str, expected: str
) -> None:
    """An absolute folder, in either platform's spelling, is kept as written."""
    path = _write(tmp_path / "f" / PATHS_FILE_NAME, {"data": written})
    if "{base}" in expected:
        expected = os.path.normpath(expected.format(base=tmp_path / "f"))
    assert read_paths_file(path)["data"] == expected


def test_home_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    path = _write(tmp_path / "f.yml", {"data": "~/data"})
    assert read_paths_file(path)["data"] == str(tmp_path / "data")
    assert parse_overrides(["data=~/data"]) == {"data": str(tmp_path / "data")}


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ({"data": 3}, "must be a folder"),
        ({"data": {"pack": "hardware/v1"}}, "must be a folder"),
        ({"data": "${x}/y"}, "interpolation"),
        (["a", "b"], "maps names to folders"),
    ],
)
def test_a_malformed_paths_file_is_refused(
    tmp_path: Path, content: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        read_paths_file(_write(tmp_path / "f.yml", content))


def test_an_empty_paths_file_sets_nothing(tmp_path: Path) -> None:
    path = tmp_path / "f.yml"
    path.write_text("")
    assert read_paths_file(path) == {}


@pytest.mark.parametrize("item", ["data", "=x", "data=", " = "])
def test_a_malformed_override_is_refused(item: str) -> None:
    with pytest.raises(ValueError, match="NAME=DIR"):
        parse_overrides([item])


def test_a_name_left_unset_fails_naming_every_file_searched(tmp_path: Path) -> None:
    config = _write(tmp_path / "c.yml", {"paths": {"data": "???"}})
    machine = tmp_path / "machine.yml"
    with pytest.raises(LookupError) as e:
        load_resolved(config, Locator(machine_file=machine))
    assert "nothing sets paths data" in str(e.value)
    assert str(tmp_path / PATHS_FILE_NAME) in str(e.value)
    assert str(machine) in str(e.value)
    _write(machine, {"data": "/found"})
    assert load_resolved(config, Locator(machine_file=machine)) == {
        "paths": {"data": "/found"}
    }


def test_a_config_without_paths_loads_unchanged(tmp_path: Path) -> None:
    config = _write(tmp_path / "c.yml", {"version": 1})
    assert load_resolved(config, _isolated()) == {"version": 1}


def test_from_yaml_reads_the_located_folders(tree: Path) -> None:
    _write(tree / "proj" / PATHS_FILE_NAME, {"meshes": "/located"})
    cfg = ConfigModel.from_yaml(_config(tree), _isolated())
    assert cfg.paths.model_dump()["meshes"] == "/located"
    assert ConfigModel.from_yaml(_config(tree)).paths.model_dump()["meshes"] == (
        "/located"
    )


def test_the_pipeline_locates_through_its_settings(tree: Path) -> None:
    given = _write(tree / "given.yml", {"data": "/given"})
    locator = PipelineSettings(paths=(given,)).locator()
    assert locator.files == (given,)
    assert _paths(tree, locator)["data"] == "/given"


def test_the_source_of_a_config_value_is_the_config(tree: Path) -> None:
    found = locate(_config(tree), {"data": "/written/data"}, _isolated())
    assert found["data"].source == CONFIG_SOURCE


def test_rutter_paths_reports_each_name_and_its_source(tree: Path) -> None:
    _write(tree / "proj" / PATHS_FILE_NAME, {"data": str(tree)})
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "aind_rutter.config.cli",
            str(_config(tree)),
            "--path",
            "meshes=/cli",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert f"set by {tree / 'proj' / PATHS_FILE_NAME}" in out
    assert "set by --path" in out
    assert "/cli  [missing]" in out
    assert f"{tree}\n" in out  # exists, so not marked
