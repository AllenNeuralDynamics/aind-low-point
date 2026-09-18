"""Both phases read the values they share through one model.

Phase 1 parsed `COV_NORM` as `== "1"` while Phase 2 let pydantic parse it, so
`COV_NORM=true` normalized the coverage of one phase and not the other. Phase 1
also read the bare `CONFIG` while Phase 2 preferred `RUTTER_CONFIG`, so an
exported `RUTTER_CONFIG` sent the two phases to different subjects.

Each module resolves these at import, so the checks run in a clean interpreter
with the environment set before it starts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Every module that resolves the subject, and the name it exposes it under.
SUBJECT_READERS = {
    "aind_rutter.optimization.pipeline.restore": "CONFIG",
    "aind_rutter.optimization.pipeline.enumeration": "CONFIG",
    "aind_rutter.optimization.pipeline.emit": "CONFIG",
}
TRUTHY = ["1", "true", "TRUE", "yes", "on"]
FALSY = ["0", "false", "no", "off"]


def _read(expression: str, env: dict[str, str]) -> str:
    """Evaluate `expression` in a clean interpreter; return its printed value."""
    result = subprocess.run(
        [sys.executable, "-c", f"print({expression})"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "PLATFORM": "cpu", **env},
        cwd=ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize(("module", "name"), sorted(SUBJECT_READERS.items()))
def test_every_stage_prefers_the_prefixed_subject(module: str, name: str) -> None:
    """RUTTER_CONFIG wins over CONFIG, in the stages as in Phase 2's settings."""
    value = _read(
        f"__import__({module!r}, fromlist=[{name!r}]).{name}",
        {"CONFIG": "plain.yml", "RUTTER_CONFIG": "prefixed.yml"},
    )
    assert value == "prefixed.yml"


@pytest.mark.parametrize(("module", "name"), sorted(SUBJECT_READERS.items()))
def test_every_stage_still_accepts_the_bare_subject(module: str, name: str) -> None:
    value = _read(
        f"__import__({module!r}, fromlist=[{name!r}]).{name}", {"CONFIG": "plain.yml"}
    )
    assert value == "plain.yml"


@pytest.mark.parametrize("setting", TRUTHY)
def test_phase_one_reads_the_same_true_values_as_phase_two(setting: str) -> None:
    """`COV_NORM=true` used to normalize Phase 2's coverage and not Phase 1's."""
    value = _read(
        "__import__('aind_rutter.optimization.pipeline.phase1_pool',"
        " fromlist=['COV_NORM']).COV_NORM",
        {"COV_NORM": setting},
    )
    assert value == "True"


@pytest.mark.parametrize("setting", FALSY)
def test_phase_one_reads_the_same_false_values_as_phase_two(setting: str) -> None:
    value = _read(
        "__import__('aind_rutter.optimization.pipeline.phase1_pool',"
        " fromlist=['COV_NORM']).COV_NORM",
        {"COV_NORM": setting},
    )
    assert value == "False"


@pytest.mark.parametrize("setting", TRUTHY + FALSY)
def test_the_two_phases_agree_on_every_spelling(setting: str) -> None:
    both = _read(
        "(__import__('aind_rutter.optimization.pipeline.phase1_pool',"
        " fromlist=['COV_NORM']).COV_NORM,"
        " __import__('aind_rutter.optimization.pipeline.settings',"
        " fromlist=['PipelineSettings']).PipelineSettings().cov_norm)",
        {"COV_NORM": setting},
    )
    first, second = both.strip("()").split(",")
    assert first.strip() == second.strip()


def test_a_value_neither_phase_can_parse_is_refused() -> None:
    """A spelling that used to fall through to False now fails at startup."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import aind_rutter.optimization.pipeline.phase1_pool",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "COV_NORM": "perhaps"},
        cwd=ROOT,
        timeout=300,
    )
    assert result.returncode != 0
    assert "cov_norm" in result.stderr.lower()
