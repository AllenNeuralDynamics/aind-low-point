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
    CACHE_SOURCE,
    CONFIG_SOURCE,
    OVERRIDE_SOURCE,
    PATHS_FILE_NAME,
    Locator,
    PathsFile,
    load_resolved,
    locate,
    parse_overrides,
    read_paths_file,
    record_pack_folder,
)
from aind_rutter.optimization.pipeline.settings import PipelineSettings
from aind_rutter.packs import PackCache, PackRef, PackSettings
from aind_rutter.packs.manifest import build_manifest
from aind_rutter.packs.store import publish, store_url


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
    assert read_paths_file(path).names["data"] == expected


def test_home_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    path = _write(tmp_path / "f.yml", {"data": "~/data"})
    assert read_paths_file(path).names["data"] == str(tmp_path / "data")
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
    assert read_paths_file(path) == PathsFile({}, {}, {})


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


# ---- packs ----

PACK_FILES = {
    "cone/cone.obj": b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    "cone/cone.mtl": b"newmtl a\n",
    "masks/PL.nrrd": b"nrrd",
    "masks/MOp.nrrd": b"nrrd",
    "unused/big.step": b"x" * 1000,
}


@pytest.fixture
def pack_store(tmp_path: Path) -> str:
    src = tmp_path / "pack-src"
    for rel, data in PACK_FILES.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(data)
    url = store_url(str(tmp_path / "store"))
    publish(src, build_manifest(src, PackRef("hardware", "v1")), url)
    return url


def _pack_locator(tmp_path: Path, store: str, **kw) -> Locator:
    cache = PackCache(PackSettings(pack_store=store, cache=tmp_path / "cache"))
    return Locator(machine_file=None, cache=cache, **kw)


def _pack_config(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "proj" / "c.yml",
        {
            "paths": {"hw": {"pack": "hardware/v1"}},
            "assets": [
                {"key": "mesh:cone", "src": "${paths.hw}/cone/cone.obj"},
                {"keys": ["structure:PL"], "src": "${paths.hw}/masks/{name}.nrrd"},
            ],
        },
    )


def _fetched(locator: Locator) -> list[str]:
    folder = locator.pack_cache().version_dir(PackRef("hardware", "v1")) / "files"
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.*"))


def test_a_pack_value_comes_from_the_cache(tmp_path: Path, pack_store: str) -> None:
    locator = _pack_locator(tmp_path, pack_store)
    found = locate(
        tmp_path / "c.yml", {"hw": {"pack": "hardware/v1"}, "d": "/d"}, locator
    )
    assert found["hw"].source == CACHE_SOURCE
    assert found["hw"].pack == PackRef("hardware", "v1")
    assert found["hw"].value.endswith(os.path.join("hardware", "v1", "files"))
    assert found["d"].pack is None


def test_loading_fetches_only_what_the_config_names(
    tmp_path: Path, pack_store: str
) -> None:
    locator = _pack_locator(tmp_path, pack_store)
    cfg = ConfigModel.from_yaml(_pack_config(tmp_path), locator)
    assert Path(str(cfg.assets[0].src)).read_bytes() == PACK_FILES["cone/cone.obj"]
    assert _fetched(locator) == ["cone/cone.mtl", "cone/cone.obj", "masks/PL.nrrd"]


def test_a_paths_file_can_locate_a_pack(tmp_path: Path, pack_store: str) -> None:
    config = _pack_config(tmp_path)
    copy = tmp_path / "copy"
    _write(tmp_path / "proj" / PATHS_FILE_NAME, {"packs": {"hardware/v1": str(copy)}})
    found = locate(config, {"hw": {"pack": "hardware/v1"}}, _isolated())
    assert (found["hw"].value, found["hw"].source) == (
        str(copy),
        f"{tmp_path / 'proj' / PATHS_FILE_NAME} (packs)",
    )
    assert found["hw"].cached is None


def test_a_name_line_beats_a_pack_line_in_one_file(tmp_path: Path) -> None:
    paths_file = _write(
        tmp_path / PATHS_FILE_NAME, {"hw": "/by-name", "packs": {"hardware/v1": "/p"}}
    )
    found = locate(
        tmp_path / "c.yml",
        {"hw": {"pack": "hardware/v1"}},
        _isolated(files=(paths_file,)),
    )
    assert found["hw"].value == os.path.normpath("/by-name")


def test_only_the_machine_file_sets_the_store(tmp_path: Path) -> None:
    paths_file = _write(tmp_path / PATHS_FILE_NAME, {"pack_store": "file:///x/"})
    with pytest.raises(ValueError, match="only the machine"):
        read_paths_file(paths_file)
    assert read_paths_file(paths_file, machine=True).settings == {
        "pack_store": "file:///x/"
    }


@pytest.mark.parametrize("value", [{"pack": "Hardware"}, {"pack": "a/b", "x": 1}])
def test_a_malformed_pack_value_is_refused(tmp_path: Path, value: dict) -> None:
    with pytest.raises(ValueError):
        locate(tmp_path / "c.yml", {"hw": value}, _isolated())


def test_recording_a_pack_folder_keeps_the_rest(tmp_path: Path) -> None:
    machine = tmp_path / "paths.yml"
    written = "# this machine\ndata: /mnt/data  # big disk\npacks:\n  other/v1: /o\n"
    machine.write_text(written)
    record_pack_folder(machine, PackRef("hardware", "v1"), Path("/copies/hw"))
    record_pack_folder(machine, PackRef("hardware", "v1"), Path("/copies/hw2"))
    text = machine.read_text()
    assert text.startswith("# this machine\ndata: /mnt/data  # big disk\npacks:\n")
    assert read_paths_file(machine, machine=True).packs == {
        "hardware/v1": os.path.normpath("/copies/hw2"),
        "other/v1": os.path.normpath("/o"),
    }
    fresh = tmp_path / "new" / "paths.yml"
    record_pack_folder(fresh, PackRef("hardware", "v1"), Path("/c"))
    assert read_paths_file(fresh).packs == {"hardware/v1": os.path.normpath("/c")}


def test_offline_on_the_locator_outranks_the_cache_it_is_given(
    tmp_path: Path, pack_store: str
) -> None:
    online = _pack_locator(tmp_path, pack_store)
    online.pack_cache().open(PackRef("hardware", "v1"))
    offline = _pack_locator(tmp_path, pack_store, offline=True)
    assert offline.pack_cache().settings.offline
    with pytest.raises(FileNotFoundError, match="RUTTER_OFFLINE"):
        ConfigModel.from_yaml(_pack_config(tmp_path), offline)
    assert PipelineSettings(offline=True).locator().offline


def test_the_lock_keys_a_pack_file_by_pack_and_a_located_one_by_name(
    tmp_path: Path, pack_store: str
) -> None:
    from aind_rutter.series.lock import referenced_files

    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "sub").mkdir(parents=True)
    (elsewhere / "sub" / "t.tfm").write_text("")
    written = tmp_path / "written.obj"
    written.write_text("")
    config = _write(
        tmp_path / "proj" / "c.yml",
        {
            "paths": {"hw": {"pack": "hardware/v1"}, "other": "/nowhere"},
            "a": "${paths.hw}/cone/cone.obj",
            "b": "${paths.other}/sub/t.tfm",
            "c": str(written),
        },
    )
    _write(tmp_path / "proj" / PATHS_FILE_NAME, {"other": str(elsewhere)})
    found = referenced_files(config, _pack_locator(tmp_path, pack_store))
    # A companion (cone.mtl) is fetched beside its file but is not an input.
    assert set(found) == {
        "hardware/v1:cone/cone.obj",
        "${paths.other}/sub/t.tfm",
        str(written.resolve()),
    }
    assert found["${paths.other}/sub/t.tfm"] == elsewhere / "sub" / "t.tfm"


def test_each_location_is_reported_once(
    tree: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tree / "proj" / PATHS_FILE_NAME, {"meshes": "/located/once"})
    with caplog.at_level("INFO", logger="aind_rutter.config.locations"):
        _paths(tree, _isolated())
        _paths(tree, _isolated())
    lines = [
        r.getMessage() for r in caplog.records if "/located/once" in r.getMessage()
    ]
    assert lines == [
        f"paths.meshes = {os.path.normpath('/located/once')} "
        f"(from {tree / 'proj' / PATHS_FILE_NAME})"
    ]
