"""``rutter-calibrate`` adds calibration sources to a config's plan block."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aind_rutter.config import ConfigModel
from aind_rutter.config.locations import Locator
from aind_rutter.plan_io import calibrate
from aind_rutter.plan_io.calibrate import (
    Args,
    Reticle,
    add_calibration,
    match_codes,
    planned_angles,
    source_entry,
    written_path,
)

PROBE = {
    "kind": "2.1",
    "arc": "a",
    "slider_ml": 0.0,
    "target": {"kind": "inline", "point_RAS": [0.0, 0.0, 0.0]},
}
RETICLE = Reticle("4", (-0.042, 0.056, -0.007), 0.37)


def _dir(where: str) -> dict:
    return {"directory": where, "reticle": "4"}


def _plan() -> dict:
    return {
        "arcs": {"a": 30.0, "b": -10.0},
        "probes": {
            "P1": {**PROBE, "slider_ml": 20.0},
            "P2": {**PROBE, "slider_ml": -5.0},
            "P3": {**PROBE, "arc": "b", "slider_ml": 10.0},
        },
    }


def test_codes_match_the_planned_angles() -> None:
    fitted = {"111": (-12.0, 11.0), "222": (31.0, -4.0), "333": (29.0, 22.0)}
    assert match_codes(fitted, planned_angles(_plan())) == {
        "P1": "333",
        "P2": "222",
        "P3": "111",
    }


def test_an_unbound_probe_is_planned_at_its_own_ap() -> None:
    plan = _plan()
    plan["probes"]["P1"].update(bind_ap_to_arc=False, ap_local=5.0)
    assert planned_angles(plan)["P1"] == (5.0, 20.0)


def test_a_source_is_written_through_the_narrowest_containing_name(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vast"
    run = root / "cal" / "log_1"
    run.mkdir(parents=True)
    located = {
        "vast": str(root),
        "calibrations": str(root / "cal"),
        "here": str(run),  # names the run itself, so it loses to a root
        "packed": {"pack": "hardware/v1"},
    }
    assert written_path(run, located) == "${paths.calibrations}/log_1"
    assert written_path(run, {"here": str(run)}) == "${paths.here}"
    assert written_path(run, {}) == run.resolve().as_posix()


def test_a_second_source_stacks_after_the_first() -> None:
    once = add_calibration(_plan(), [_dir("d1")], RETICLE, "4", {"P1": "333"})
    twice = add_calibration(
        once, [_dir("d2"), {"file": "m.xlsx"}], None, None, {"P1": "444", "P2": "222"}
    )
    cals = twice["calibrations"]
    assert cals["sources"] == [_dir("d1"), _dir("d2"), {"file": "m.xlsx"}]
    assert cals["probe_to_code"] == {"P1": "444", "P2": "222"}
    assert twice["reticles"]["4"] == RETICLE.as_yaml()
    assert [twice["probes"][p].get("calibrated") for p in ("P1", "P2", "P3")] == [
        True,
        True,
        None,
    ]
    # Re-adding a source moves it to the end, where it wins.
    again = add_calibration(twice, [_dir("d1")], None, "4", {})
    assert again["calibrations"]["sources"] == [
        _dir("d2"),
        {"file": "m.xlsx"},
        _dir("d1"),
    ]


def test_a_directory_entry_names_its_reticle() -> None:
    assert source_entry("w.xlsx", False, "4") == {"file": "w.xlsx"}
    assert source_entry("d", True, "4") == _dir("d")
    with pytest.raises(ValueError, match="needs a reticle"):
        source_entry("d", True, None)


@pytest.mark.parametrize(
    "plan_update, reticle, name, match",
    [
        ({"calibrations": {"files": {"x": {}}}}, RETICLE, "4", "stacked"),
        ({}, None, "H", "not declared"),
        (
            {"reticles": {"4": {"offset_RAS": [0, 0, 0], "rotation_z": 0}}},
            RETICLE,
            "4",
            "other values",
        ),
    ],
)
def test_refusals(plan_update, reticle, name, match) -> None:
    with pytest.raises(ValueError, match=match):
        add_calibration({**_plan(), **plan_update}, [_dir("d")], reticle, name, {})


def test_an_unknown_probe_is_refused() -> None:
    with pytest.raises(ValueError, match="no such probe"):
        add_calibration(_plan(), [_dir("d")], RETICLE, "4", {"GHOST": "1"})


def test_run_writes_a_config_that_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "cal"
    run = runs / "log_1"
    run.mkdir(parents=True)
    config = tmp_path / "config.yml"
    config.write_text(
        "# kept\n"
        + yaml.safe_dump(
            {
                "version": 1,
                "paths": {"cal_root": str(runs)},
                "assets": [{"key": "probe:2.1", "kind": "mesh", "src": "x.obj"}],
                "plan": _plan(),
            },
            sort_keys=False,
        )
    )
    plan_file = tmp_path / "plan.yml"
    plan_file.write_text(yaml.safe_dump(_plan()))
    manual = runs / "manual.xlsx"
    manual.touch()
    # The workbook refits 333 and adds 444; it wins for 333 as the later source.
    fits = {
        run: {"111": (-12.0, 11.0), "222": (31.0, -4.0), "333": (0.0, 0.0)},
        manual: {"333": (29.0, 22.0), "444": (80.0, 80.0)},
    }
    monkeypatch.setattr(calibrate, "fitted_angles", lambda s, r: fits[s])
    monkeypatch.setattr(
        "aind_rutter.build.calibration.load_calibrations", lambda c, r: {}
    )

    calibrate.run(
        Args(
            config=config,
            sources=(run, manual),
            out=tmp_path / "out.yml",
            reticle="4",
            reticle_offset=RETICLE.offset_RAS,
            reticle_rotation=RETICLE.rotation_z,
            probe=("P3=111",),
            match_plan=plan_file,
        )
    )

    text = (tmp_path / "out.yml").read_text()
    assert text.startswith("# kept\n")
    cfg = ConfigModel.from_yaml(tmp_path / "out.yml", Locator(machine_file=None))
    parallax, workbook = cfg.plan.calibrations.sources
    assert parallax.reticle == "4" and Path(parallax.directory) == run
    assert Path(workbook.file) == manual
    assert "${paths.cal_root}/log_1" in text
    assert "${paths.cal_root}/manual.xlsx" in text
    assert cfg.plan.calibrations.probe_to_code == {
        "P3": "111",
        "P1": "333",
        "P2": "222",
    }
    assert all(p.calibrated for p in cfg.plan.probes.values())
    assert not list(tmp_path.glob(".*.staged"))


def test_run_needs_one_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--out or --in-place"):
        calibrate.run(Args(config=tmp_path / "c.yml", sources=(tmp_path,)))


def test_a_reticle_without_a_directory_is_refused(tmp_path: Path) -> None:
    config = tmp_path / "c.yml"
    config.write_text(yaml.safe_dump({"plan": _plan()}))
    manual = tmp_path / "m.xlsx"
    manual.touch()
    with pytest.raises(ValueError, match="none is given"):
        calibrate.run(
            Args(config=config, sources=(manual,), in_place=True, reticle="4")
        )
