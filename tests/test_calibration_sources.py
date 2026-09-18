"""A calibration source declares a reticle only when it needs one.

`CalibrationSourceModel` forbids naming a reticle beside `file:`, because a
single calibration file carries its own — but the cross-reference pass demanded
that every source's reticle be declared, so a `file:` entry failed with
"reticle 'None' not in plan.reticles" and the whole legacy mode was unusable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from aind_rutter.config import ConfigModel

RETICLE = {"offset_RAS": [0.0, 0.0, 0.0], "rotation_z": 0.0}
PROBE = {
    "kind": "2.1",
    "arc": "a",
    "spin": 0.0,
    "slider_ml": 0.0,
    "past_target_mm": 0.0,
    "offsets_RA": [0.0, 0.0],
    "target": {"kind": "inline", "point_RAS": [0.0, 0.0, 0.0]},
}


# A probe roster needs a matching catalog asset; src is not opened during
# validation, so a name is enough.
PROBE_ASSET = {"key": "probe:2.1", "kind": "mesh", "src": "unread.obj"}


def _config(calibrations: dict[str, Any], *, reticles: dict | None = None) -> dict:
    return {
        "version": 1,
        "assets": [dict(PROBE_ASSET)],
        "plan": {
            "arcs": {"a": 0.0},
            "probes": {"P1": dict(PROBE)},
            "reticles": reticles if reticles is not None else {},
            "calibrations": calibrations,
        },
    }


@pytest.fixture
def calibration_file(tmp_path: Path) -> Path:
    path = tmp_path / "manual_calibration.xlsx"
    path.write_bytes(b"")
    return path


@pytest.fixture
def parallax_directory(tmp_path: Path) -> Path:
    path = tmp_path / "parallax"
    path.mkdir()
    return path


def test_a_file_source_needs_no_reticle(calibration_file: Path) -> None:
    """The defect: this raised even though the model forbids naming a reticle."""
    config = ConfigModel.model_validate(
        _config({"files": {"manual": {"file": str(calibration_file)}}})
    )
    assert config.plan.calibrations.files["manual"].reticle is None


def test_a_file_source_may_not_name_one(calibration_file: Path) -> None:
    with pytest.raises(ValidationError, match="must not be provided"):
        ConfigModel.model_validate(
            _config(
                {"files": {"manual": {"file": str(calibration_file), "reticle": "r"}}},
                reticles={"r": RETICLE},
            )
        )


def test_a_directory_source_still_needs_its_reticle_declared(
    parallax_directory: Path,
) -> None:
    with pytest.raises(ValidationError, match="not in plan.reticles"):
        ConfigModel.model_validate(
            _config(
                {
                    "files": {
                        "parallax": {
                            "directory": str(parallax_directory),
                            "reticle": "absent",
                        }
                    }
                },
                reticles={"present": RETICLE},
            )
        )


def test_a_declared_reticle_satisfies_a_directory_source(
    parallax_directory: Path,
) -> None:
    config = ConfigModel.model_validate(
        _config(
            {
                "files": {
                    "parallax": {
                        "directory": str(parallax_directory),
                        "reticle": "present",
                    }
                }
            },
            reticles={"present": RETICLE},
        )
    )
    assert config.plan.calibrations.files["parallax"].reticle == "present"


def test_a_stacked_file_source_needs_no_reticle(calibration_file: Path) -> None:
    config = ConfigModel.model_validate(
        _config(
            {
                "sources": [{"file": str(calibration_file)}],
                "probe_to_code": {"P1": "1"},
            }
        )
    )
    assert config.plan.calibrations.sources[0].reticle is None


def test_a_stacked_directory_source_is_checked_too(parallax_directory: Path) -> None:
    """Stacked mode had no reticle check at all; it failed later, at load."""
    with pytest.raises(ValidationError, match=r"sources\[0\]"):
        ConfigModel.model_validate(
            _config(
                {
                    "sources": [
                        {"directory": str(parallax_directory), "reticle": "absent"}
                    ],
                    "probe_to_code": {"P1": "1"},
                },
                reticles={"present": RETICLE},
            )
        )
