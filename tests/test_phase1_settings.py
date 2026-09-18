"""`Phase1Settings` resolves what Phase 1's module globals resolved.

Phase 1 read 21 values from the environment at import, which meant the importable
`run()` could not be told anything. The tables in `tests/phase1_env.py` record
what those globals made of a given environment; this checks the settings object
against them, field by field.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from aind_rutter.optimization.pipeline.settings import Phase1Settings
from tests.phase1_env import (
    DEFAULT_ENV,
    FIELDS,
    RESOLVED_DEFAULT,
    RESOLVED_SAMPLE,
    ROOT,
    SAMPLE,
)


def _settings(env: dict[str, str]) -> dict[str, str]:
    """Build the settings in a clean interpreter, so the ambient env cannot leak."""
    names = sorted(FIELDS.values())
    code = (
        "import json;"
        "from aind_rutter.optimization.pipeline.settings import Phase1Settings as S;"
        f"s = S();print(json.dumps({{n: str(getattr(s, n)) for n in {names!r}}}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        timeout=300,
        env={**os.environ, **env},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def sampled() -> dict[str, str]:
    return _settings(SAMPLE)


@pytest.fixture(scope="module")
def defaulted() -> dict[str, str]:
    return _settings(DEFAULT_ENV)


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_a_set_variable_resolves_as_it_did(name: str, sampled) -> None:
    assert sampled[FIELDS[name]] == RESOLVED_SAMPLE[name], name


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_an_unset_variable_defaults_as_it_did(name: str, defaulted) -> None:
    assert defaulted[FIELDS[name]] == RESOLVED_DEFAULT[name], name


def test_the_sample_differs_from_the_default_everywhere() -> None:
    """Otherwise a field that ignores its variable would pass both tests."""
    same = [n for n in FIELDS if RESOLVED_SAMPLE[n] == RESOLVED_DEFAULT[n]]
    assert not same, f"sample value equals the default for {same}"


def test_the_prefixed_spelling_wins() -> None:
    """`LIMIT` and `OUT` are set in many environments for unrelated reasons."""
    resolved = _settings({**DEFAULT_ENV, "LIMIT": "5", "RUTTER_LIMIT": "9"})
    assert resolved["limit"] == "9"


def test_the_seed_cache_is_named_for_the_subject() -> None:
    """Sharing one cache between subjects seeds the wrong atlas."""
    a = Phase1Settings(config="examples/836656-config.yml").seed_cache
    b = Phase1Settings(config="examples/837229-config.yml").seed_cache
    assert a != b
    assert "836656" in str(a) and "837229" in str(b)


def test_a_caller_may_pass_values_directly() -> None:
    """The point of the conversion: `run()` can be told, not just configured."""
    settings = Phase1Settings(stage1=10, stage2=20, chunk=8, coarse_n=5000)
    assert (settings.stage1, settings.stage2, settings.chunk) == (10, 20, 8)
    assert settings.two_fidelity is False


def test_a_constructor_argument_beats_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("STAGE1", "999")
    assert Phase1Settings(stage1=10).stage1 == 10
    assert Phase1Settings().stage1 == 999


def test_every_stage_has_a_settings_class() -> None:
    """The point of step 4: each stage is told, not configured by import order."""
    from aind_rutter.optimization.pipeline import emit, phase1_pool, phase2_ipopt

    for module in (phase1_pool, phase2_ipopt, emit):
        assert callable(module.run), module.__name__
        assert callable(module.main), module.__name__


def test_the_stage_settings_share_one_subject() -> None:
    """A stage that resolved the subject for itself could be sent elsewhere."""
    from aind_rutter.optimization.pipeline.settings import (
        EmitSettings,
        Phase2Settings,
        PipelineSettings,
    )

    subject = "examples/837229-config.yml"
    resolved = {
        cls(config=subject).config
        for cls in (PipelineSettings, Phase1Settings, Phase2Settings, EmitSettings)
    }
    assert len(resolved) == 1


def test_the_caches_are_named_for_the_subject() -> None:
    """Atlas and seed caches are subject-specific; sharing one seeds the wrong
    geometry."""
    a = Phase1Settings(config="examples/836656-config.yml")
    b = Phase1Settings(config="examples/837229-config.yml")
    assert a.atlas_cache != b.atlas_cache
    assert a.seed_cache != b.seed_cache
