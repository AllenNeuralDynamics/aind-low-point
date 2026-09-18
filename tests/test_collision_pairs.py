"""Which pairs get an FCL test, decided by a rule rather than per-asset labels.

Every asset used to carry `collision.group` and `collision.mask` label lists,
compiled to bits, and `FCLBackend` tested a pair when each side's mask admitted
the other's group. Across all five tracked configs those labels only ever
expressed two patterns — probes admit fixtures and probes, fixtures admit probes
— so the pair filter is a rule about probe-ness, not data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aind_rutter.collisions import pair_bits
from aind_rutter.common import Role
from aind_rutter.config import ConfigModel
from aind_rutter.runtime.build import resolve_collidable
from tests.config_semantics import CONFIGS


@dataclass
class _Spec:
    collidable: bool
    role: Role


def _tested(a: _Spec, b: _Spec) -> bool:
    g1, m1 = pair_bits(a)
    g2, m2 = pair_bits(b)
    return bool(m1 & g2) and bool(m2 & g1)


PROBE = _Spec(True, Role.PROBE)
FIXTURE = _Spec(True, Role.FIXTURE)
INERT = _Spec(False, Role.ANATOMY)


@pytest.mark.parametrize(
    ("a", "b", "tested"),
    [
        (PROBE, PROBE, True),
        (PROBE, FIXTURE, True),
        (FIXTURE, FIXTURE, False),
        (PROBE, INERT, False),
        (FIXTURE, INERT, False),
        (INERT, INERT, False),
    ],
    ids=[
        "probe-probe",
        "probe-fixture",
        "fixture-fixture",
        "probe-inert",
        "fixture-inert",
        "inert-inert",
    ],
)
def test_the_rule(a: _Spec, b: _Spec, tested: bool) -> None:
    assert _tested(a, b) is tested


def test_a_non_collidable_probe_is_tested_against_nothing() -> None:
    """`collidable` and probe-ness are separate questions."""
    assert not _tested(_Spec(False, Role.PROBE), PROBE)


@pytest.mark.parametrize("path", CONFIGS)
def test_no_config_still_carries_collision_labels(path: str) -> None:
    """`collision.group`/`mask` are migrated away; the rule decides now.

    Which pairs each config yields is pinned in tests/config_semantics.json.
    """
    cfg = ConfigModel.from_yaml(path)
    labelled = [
        str(s.key)
        for s in [*cfg.assets, *cfg.targets]
        if s.collision.group or s.collision.mask
    ]
    assert not labelled, f"{path}: still labelled: {labelled}"


@pytest.mark.parametrize("path", CONFIGS)
def test_every_collidable_asset_is_a_probe_or_a_fixture(path: str) -> None:
    """Anything else collidable would be tested against nothing at all."""
    cfg = ConfigModel.from_yaml(path)
    stranded = [
        str(s.key)
        for s in [*cfg.assets, *cfg.targets]
        if resolve_collidable(s) and s.role not in (Role.PROBE, Role.FIXTURE)
    ]
    assert not stranded, f"{path}: collidable but neither probe nor fixture: {stranded}"


def test_a_template_carries_collidability_for_its_assets() -> None:
    """Where the real configs state it: once per template, not once per asset."""
    cfg = ConfigModel.model_validate(
        {
            "version": 1,
            "asset_templates": {
                "hardware": {"kind": "mesh", "role": "fixture", "collidable": True}
            },
            "assets": [
                {
                    "key": "well",
                    "src": "w.obj",
                    "loader": "trimesh",
                    "templates": ["hardware"],
                },
                {
                    "key": "cone",
                    "src": "c.obj",
                    "loader": "trimesh",
                    "templates": ["hardware"],
                },
            ],
        }
    )
    assert [resolve_collidable(a) for a in cfg.assets] == [True, True]
    assert {a.role for a in cfg.assets} == {Role.FIXTURE}


def test_an_asset_may_opt_out_of_its_template() -> None:
    """A fixture that is drawn but never collided against."""
    cfg = ConfigModel.model_validate(
        {
            "version": 1,
            "asset_templates": {
                "hardware": {"kind": "mesh", "role": "fixture", "collidable": True}
            },
            "assets": [
                {
                    "key": "decor",
                    "src": "d.obj",
                    "loader": "trimesh",
                    "templates": ["hardware"],
                    "collidable": False,
                },
            ],
        }
    )
    assert resolve_collidable(cfg.assets[0]) is False
