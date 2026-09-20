"""The flag rework may change how a config says things, not what it means.

`tests/config_semantics.json` is the contract: every tracked config's per-asset
chemical-shift decision and ppm, collidability, and tag and collision labels, as
they resolve today. A diff in that file is a behaviour change, and for chemical
shift it is a change that silently moves geometry — regenerate it only when the
move is the point.

Regenerate with ``uv run --python 3.13 python -m tests.config_semantics``.
"""

from __future__ import annotations

import json

import pytest

from tests.config_semantics import (
    CONFIGS,
    GOLDEN,
    MissingSubjectData,
    fixtures_for,
    pairs_for,
    scene_for,
    semantics_for,
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text())


def or_skip(fn, path: str):
    """``fn(path)``, or skip when this machine cannot see the subject data.

    Every tracked subject config points at a lab share. Off that network the
    config still validates as YAML but its declared inputs are absent, and
    there is nothing to compare against the golden file. Any other validation
    failure is raised, so a genuinely broken config still fails here.
    """
    try:
        return fn(path)
    except MissingSubjectData as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("path", CONFIGS)
def test_a_config_resolves_to_the_recorded_decisions(path: str, golden: dict) -> None:
    assert or_skip(semantics_for, path) == golden[path]["specs"]


def test_the_golden_file_covers_every_tracked_config(golden: dict) -> None:
    """A config added to the corpus without a regenerate would pass vacuously."""
    assert set(golden) == set(CONFIGS)


@pytest.mark.parametrize("path", CONFIGS)
def test_no_cad_geometry_is_chemically_shifted(path: str) -> None:
    """The implant, well, cone, headframe and probes are not in image space.

    Chemical shift displaces what the scanner measured. Geometry that came from
    CAD or from the implant frame must never move, whichever flag carries the
    decision.
    """
    resolved = or_skip(semantics_for, path)
    physical = [
        key
        for key in resolved
        if key.split(":")[0] in {"probe", "implant", "well", "cone", "headframe"}
        or key in {"implant", "well", "cone", "headframe"}
    ]
    assert physical, f"{path}: no CAD geometry found — the key spellings moved"
    shifted = [key for key in physical if resolved[key]["chem_shift"]]
    assert not shifted, f"{path}: CAD geometry chemically shifted: {sorted(shifted)}"


@pytest.mark.parametrize("path", CONFIGS)
def test_bore_targets_are_not_shifted_but_annotation_targets_are(path: str) -> None:
    """The distinction `Role` cannot draw, and the reason for 13 `off` overrides.

    A bore centre and an annotation centroid are both targets; only the second
    is in image space.
    """
    resolved = or_skip(semantics_for, path)
    if not any(spec["chem_shift"] for spec in resolved.values()):
        pytest.skip("config declares no imaging, so nothing is shifted")
    bores = [k for k in resolved if k.startswith("target:hole:")]
    regions = [k for k in resolved if k.startswith(("target:L:", "target:R:"))]
    assert not [k for k in bores if resolved[k]["chem_shift"]]
    assert all(resolved[k]["chem_shift"] for k in regions)


@pytest.mark.parametrize("path", CONFIGS)
def test_a_config_yields_the_recorded_collision_pairs(path: str, golden: dict) -> None:
    """Which pairs the backend tests.

    These were per-asset `collision.group`/`mask` labels compiled to bits; they
    are now the rule "both collidable, at least one a probe", which
    tests/test_collision_pairs.py checks in isolation. The counts here are the
    ones the labels produced.
    """
    assert or_skip(pairs_for, path) == golden[path]["pairs"]


@pytest.mark.parametrize("path", CONFIGS)
def test_a_config_produces_the_recorded_scene_nodes(path: str, golden: dict) -> None:
    """Which nodes exist, and what each is tagged.

    A probe *kind* asset is a template with no placement; the nodes the planner
    poses come from `plan.probes`. Tagging a template must not invent a node.
    """
    assert or_skip(scene_for, path) == golden[path]["scene"]


@pytest.mark.parametrize("path", CONFIGS)
def test_a_config_yields_the_recorded_fixture_set(path: str, golden: dict) -> None:
    """What the optimizer treats as a static obstacle."""
    assert or_skip(fixtures_for, path) == golden[path]["fixtures"]


def test_absent_data_is_told_apart_from_a_broken_config(tmp_path) -> None:
    """The skip must be narrow. A config whose share is unmounted is skipped;
    one that is genuinely wrong still fails, or the corpus would go quiet the
    moment someone broke a config off the network."""
    import yaml
    from pydantic import ValidationError

    from tests.config_semantics import ROOT, load_config

    absent = tmp_path / "absent.yml"
    absent.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "transforms": {
                    "t": {
                        "sequence": [
                            {"kind": "sitk_file", "path": str(tmp_path / "nope.h5")}
                        ]
                    }
                },
                "assets": [],
            }
        )
    )
    with pytest.raises(MissingSubjectData) as caught:
        load_config(
            str(absent.relative_to(ROOT))
            if absent.is_relative_to(ROOT)
            else str(absent)
        )
    assert "nope.h5" in str(caught.value)

    broken = tmp_path / "broken.yml"
    broken.write_text(yaml.safe_dump({"version": 1, "assets": [{"no_such": "field"}]}))
    with pytest.raises(ValidationError):
        load_config(str(broken))
