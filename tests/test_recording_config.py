"""A config declares where a probe's electrodes are.

`RECORDING_GEOMETRY` used to be the only source, so a subject using a probe the
table had never heard of got tip-on-target silently — the same answer a pipette
gets deliberately. Three of the table's five entries were holder variants of one
another, added because the table had to chase the configs.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aind_rutter.build.assemble import resolve_recording
from aind_rutter.config import ConfigModel
from aind_rutter.domain.probe_kinds import RECORDING_GEOMETRY

RANGES = [[0.2, 3.065], [0.2, 3.065]]


def _config(*, kind: str, **asset_over: Any) -> dict[str, Any]:
    asset = {
        "key": f"probe:{kind}",
        "kind": "mesh",
        "src": "p.obj",
        "loader": "trimesh",
        **asset_over,
    }
    return {
        "version": 1,
        "assets": [
            asset,
            {"key": "t", "kind": "points", "src": "t.csv", "loader": "csv_points"},
        ],
        "targets": [{"key": "target:t", "source_key": "t", "reducer": "mesh_centroid"}],
        "plan": {
            "arcs": {"a": 0.0},
            "probes": {
                "P1": {"kind": kind, "arc": "a", "target": "target:t"},
            },
        },
    }


def test_an_unregistered_kind_is_refused() -> None:
    """The defect: this used to plan fine, aiming with the tip."""
    with pytest.raises(ValidationError, match="has no recording geometry"):
        ConfigModel.model_validate(_config(kind="np-3.0"))


def test_the_error_names_the_probe_and_the_kind() -> None:
    with pytest.raises(ValidationError, match=r"probe kind 'np-3\.0'"):
        ConfigModel.model_validate(_config(kind="np-3.0"))


def test_declaring_the_geometry_admits_a_new_kind() -> None:
    """No code change for a probe the built-in table never heard of."""
    cfg = ConfigModel.model_validate(
        _config(kind="np-3.0", recording={"active_ranges_mm": RANGES})
    )
    geom = resolve_recording(cfg.assets[0])
    assert geom is not None
    assert geom.n_shanks == 2
    assert geom.active_ranges_mm == ((0.2, 3.065), (0.2, 3.065))


def test_an_array_less_probe_says_so() -> None:
    """A pipette targets with its tip, and now that is stated rather than
    inferred from a kind being absent."""
    cfg = ConfigModel.model_validate(_config(kind="pipette", recording="none"))
    assert resolve_recording(cfg.assets[0]) is None


def test_a_registered_kind_needs_no_declaration() -> None:
    cfg = ConfigModel.model_validate(_config(kind="2.1"))
    geom = resolve_recording(cfg.assets[0])
    assert geom == RECORDING_GEOMETRY["2.1"]


def test_a_declaration_overrides_the_built_in_table() -> None:
    """A subject recording from a different bank than the default."""
    cfg = ConfigModel.model_validate(
        _config(kind="2.1", recording={"active_ranges_mm": [[1.0, 2.0]]})
    )
    geom = resolve_recording(cfg.assets[0])
    assert geom is not None
    assert geom.active_ranges_mm == ((1.0, 2.0),)
    assert geom.active_center_mm == pytest.approx(1.5)


def test_a_descending_range_is_refused() -> None:
    with pytest.raises(ValidationError, match="end > start"):
        ConfigModel.model_validate(
            _config(kind="np-3.0", recording={"active_ranges_mm": [[2.0, 1.0]]})
        )


def test_an_empty_range_list_is_refused() -> None:
    with pytest.raises(ValidationError):
        ConfigModel.model_validate(
            _config(kind="np-3.0", recording={"active_ranges_mm": []})
        )


def test_the_resolved_geometry_reaches_the_probe_context(tmp_path) -> None:
    """End to end: what the optimizer reads comes from the config."""
    import yaml

    from aind_rutter.build.assemble import build_runtime_from_config
    from aind_rutter.build.probe_context import probe_context_from_runtime
    from tests.synthetic_subject import write_subject

    subject = write_subject(tmp_path)
    document = yaml.safe_load(subject.config.read_text())
    for asset in document["assets"]:
        if asset["key"] == "probe:2.1":
            asset["recording"] = {"active_ranges_mm": [[0.5, 1.5]]}
    path = tmp_path / "recording.yml"
    path.write_text(yaml.safe_dump(document))

    runtime = build_runtime_from_config(ConfigModel.from_yaml(path))
    context = probe_context_from_runtime(runtime, "P1")
    assert context.recording is not None
    assert context.recording.active_ranges_mm == ((0.5, 1.5),)
