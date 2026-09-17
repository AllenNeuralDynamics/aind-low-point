"""Settings resolve as arguments over environment over defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aind_rutter.optimization.pipeline.settings import Phase2Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a known environment.

    The pipeline drivers export these names, so a shell that had sourced one
    would otherwise make the default assertions fail for reasons unrelated to
    the code under test.
    """
    for field in Phase2Settings.model_fields.values():
        for name in getattr(field.validation_alias, "choices", ()) or ():
            monkeypatch.delenv(str(name), raising=False)


def test_defaults_match_the_shipped_solver_configuration() -> None:
    s = Phase2Settings()
    assert (s.lam_clear, s.ip_hist, s.ip_acc_tol, s.ip_acc_iter) == (0.0, 60, 5.0, 8)
    assert (s.ip_tol, s.ip_cvtol) == (1e-4, 1e-4)
    assert (s.solver, s.hess, s.ip_mu, s.well) == ("ipopt", "none", "adaptive", "thick")


def test_environment_supplies_values_under_the_names_the_pipeline_has_always_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IP_HIST", "6")
    monkeypatch.setenv("LAM_CLEAR", "5.0")
    monkeypatch.setenv("P2_DIAG", "1")
    monkeypatch.setenv("HOLES", "scratch/other.holes.yml")
    s = Phase2Settings()
    assert (s.ip_hist, s.lam_clear, s.p2_diag) == (6, 5.0, True)
    assert s.holes == Path("scratch/other.holes.yml")


def test_arguments_win_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IP_HIST", "6")
    monkeypatch.setenv("LAM_CLEAR", "5.0")
    s = Phase2Settings(ip_hist=99)
    assert s.ip_hist == 99, "an explicit argument must outrank the environment"
    assert s.lam_clear == 5.0, "unsupplied fields still come from the environment"


@pytest.mark.parametrize("field", ["poses", "out"])
def test_a_pickle_payload_path_fails_at_construction(field: str) -> None:
    with pytest.raises(ValidationError, match="json"):
        Phase2Settings(**{field: "scratch/run.pkl"})
    assert getattr(Phase2Settings(**{field: "scratch/run.json.gz"}), field) == Path(
        "scratch/run.json.gz"
    )


def test_a_pickle_payload_path_from_the_environment_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUT", "scratch/handoff.pkl")
    with pytest.raises(ValidationError, match="json"):
        Phase2Settings()


def test_the_prefixed_spelling_outranks_the_generic_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CONFIG is set for unrelated reasons in many environments.
    monkeypatch.setenv("CONFIG", "examples/generic.yml")
    monkeypatch.setenv("RUTTER_CONFIG", "examples/ours.yml")
    assert Phase2Settings().config == Path("examples/ours.yml")


def test_unparseable_and_unknown_values_fail_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HESS", "dense2")
    with pytest.raises(ValidationError):
        Phase2Settings()
    monkeypatch.delenv("HESS")
    with pytest.raises(ValidationError):
        Phase2Settings(no_such_knob=1)


def test_settings_are_frozen_so_a_worker_cannot_mutate_them() -> None:
    s = Phase2Settings()
    with pytest.raises(ValidationError):
        s.ip_hist = 6
