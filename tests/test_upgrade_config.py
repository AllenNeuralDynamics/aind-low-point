"""Upgrading a config written before `mr_signal` must not move a decision."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aind_rutter.common import MRSignal
from aind_rutter.config import ConfigModel

upgrade_mod = pytest.importorskip("scripts.upgrade_config")

LEGACY = {
    "version": 1,
    "imaging": {
        "magnet_frequency_MHz": 599.0,
        "chem_shift_ppm_default": 3.9,
        "chem_shift_apply_by_role": ["anatomy", "target"],
    },
    "assets": [
        {"key": "brain", "kind": "mesh", "src": "b.obj", "role": "anatomy"},
        {"key": "headframe", "kind": "mesh", "src": "h.obj", "role": "geometry"},
        {"key": "probe:2.1", "kind": "mesh", "src": "p.obj", "role": "geometry"},
    ],
    "targets": [
        {
            "key": "target:MD",
            "kind": "points",
            "src": "m.csv",
            "loader": "csv_points",
            "role": "target",
        },
        {
            "key": "target:hole:1",
            "kind": "points",
            "src": "h.csv",
            "loader": "csv_points",
            "role": "target",
            "chem_shift_policy": "off",
        },
    ],
}


@pytest.fixture
def legacy(tmp_path: Path) -> Path:
    path = tmp_path / "legacy.yml"
    path.write_text(yaml.safe_dump(LEGACY, sort_keys=False))
    return path


def test_the_legacy_config_will_not_load_strictly(legacy: Path) -> None:
    """The reason the script exists, and why it loads with the flag."""
    with pytest.raises(Exception, match="mr_signal is required"):
        ConfigModel.from_yaml(legacy)
    assert ConfigModel.from_yaml(legacy, require_mr_signal=False)


def test_every_decision_survives_the_upgrade(legacy: Path) -> None:
    before = upgrade_mod.decisions(
        ConfigModel.from_yaml(legacy, require_mr_signal=False)
    )
    upgrade_mod.upgrade(legacy, dry_run=False)
    after = upgrade_mod.decisions(ConfigModel.from_yaml(legacy))
    assert after == before
    assert before == {
        "brain": True,
        "headframe": False,
        "probe:2.1": False,
        "target:MD": True,
        "target:hole:1": False,
    }


def test_the_upgraded_config_loads_strictly(legacy: Path) -> None:
    upgrade_mod.upgrade(legacy, dry_run=False)
    cfg = ConfigModel.from_yaml(legacy)
    assert {str(s.key): s.mr_signal for s in [*cfg.assets, *cfg.targets]} == {
        "brain": MRSignal.WATER,
        "headframe": MRSignal.FAT,
        "probe:2.1": MRSignal.NONE,
        "target:MD": MRSignal.WATER,
        "target:hole:1": MRSignal.FAT,
    }


def test_the_superseded_fields_are_removed(legacy: Path) -> None:
    """They are ignored once `mr_signal` is set, and `extra=forbid` later."""
    upgrade_mod.upgrade(legacy, dry_run=False)
    text = legacy.read_text()
    assert "chem_shift_policy" not in text
    assert "chem_shift_apply_by_role" not in text


def test_running_it_twice_changes_nothing(legacy: Path) -> None:
    upgrade_mod.upgrade(legacy, dry_run=False)
    once = legacy.read_text()
    with pytest.raises(upgrade_mod.Unchanged):
        upgrade_mod.upgrade(legacy, dry_run=False)
    assert legacy.read_text() == once


def test_a_dry_run_writes_nothing(legacy: Path) -> None:
    original = legacy.read_text()
    upgrade_mod.upgrade(legacy, dry_run=True)
    assert legacy.read_text() == original


def test_an_atlas_config_is_skipped(tmp_path: Path) -> None:
    """No imaging block, so there is no chemical shift to describe."""
    path = tmp_path / "atlas.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "assets": [{"key": "brain", "kind": "mesh", "src": "b.obj"}],
            }
        )
    )
    with pytest.raises(upgrade_mod.Unchanged, match="atlas"):
        upgrade_mod.upgrade(path, dry_run=False)


def test_a_bulk_declaration_gets_one_signal(tmp_path: Path) -> None:
    """A `keys:` entry expands to many assets that share the declaration."""
    path = tmp_path / "bulk.yml"
    doc = {
        "version": 1,
        "imaging": dict(LEGACY["imaging"]),
        "assets": [
            {
                "keys": ["structure:MD", "structure:PL"],
                "kind": "mesh",
                "src": "{name}.obj",
                "role": "anatomy",
            }
        ],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    upgrade_mod.upgrade(path, dry_run=False)
    cfg = ConfigModel.from_yaml(path)
    assert [a.mr_signal for a in cfg.assets] == [MRSignal.WATER, MRSignal.WATER]
