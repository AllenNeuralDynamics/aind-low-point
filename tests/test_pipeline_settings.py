"""Settings resolve as arguments over environment over defaults."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from aind_rutter.optimization.objectives.feasibility import FeasibilitySpec
from aind_rutter.optimization.objectives.objective_spec import ObjectiveSpec
from aind_rutter.optimization.pipeline.settings import (
    RETIRED_VARIABLES,
    Phase1Settings,
    Phase2Settings,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a known environment.

    The pipeline drivers export these names, so a shell that had sourced one
    would otherwise make the default assertions fail for reasons unrelated to
    the code under test.
    """
    for name in {**Phase1Settings.model_fields, **Phase2Settings.model_fields}:
        monkeypatch.delenv(f"RUTTER_{name.upper()}", raising=False)
    for name in RETIRED_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_match_the_shipped_solver_configuration() -> None:
    s = Phase2Settings()
    assert (s.lam_clear, s.ip_hist, s.ip_acc_tol, s.ip_acc_iter) == (0.0, 60, 5.0, 8)
    assert (s.ip_tol, s.ip_cvtol) == (1e-4, 1e-4)
    assert (s.p2_iter, s.ip_mu_max, s.drop_dead_rows, s.ip_bound_mult_init) == (
        300,
        1e5,
        True,
        "mu-based",
    )
    assert (s.solver, s.hess, s.ip_mu, s.well) == ("ipopt", "none", "adaptive", "thick")
    assert (s.lam_thread, s.grid_dtype) == (1.0, "bfloat16")


def test_environment_supplies_values_under_the_prefixed_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUTTER_IP_HIST", "6")
    monkeypatch.setenv("RUTTER_LAM_CLEAR", "5.0")
    monkeypatch.setenv("RUTTER_P2_DIAG", "1")
    monkeypatch.setenv("RUTTER_HOLES", "scratch/other.holes.yml")
    s = Phase2Settings()
    assert (s.ip_hist, s.lam_clear, s.p2_diag) == (6, 5.0, True)
    assert s.holes == Path("scratch/other.holes.yml")


def test_arguments_win_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUTTER_IP_HIST", "6")
    monkeypatch.setenv("RUTTER_LAM_CLEAR", "5.0")
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
    monkeypatch.setenv("RUTTER_OUT", "scratch/handoff.pkl")
    with pytest.raises(ValidationError, match="json"):
        Phase2Settings()


def test_only_the_prefixed_spelling_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CONFIG, OUT, LIMIT, N and WORKERS are set for unrelated reasons in
    # ordinary shells; a stage reading one of them optimized the wrong subject.
    monkeypatch.setenv("CONFIG", "examples/generic.yml")
    assert Phase2Settings().config == Phase2Settings.model_fields["config"].default
    monkeypatch.setenv("RUTTER_CONFIG", "examples/ours.yml")
    assert Phase2Settings().config == Path("examples/ours.yml")


def test_unparseable_and_unknown_values_fail_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUTTER_HESS", "dense2")
    with pytest.raises(ValidationError):
        Phase2Settings()
    monkeypatch.delenv("RUTTER_HESS")
    with pytest.raises(ValidationError):
        Phase2Settings(no_such_knob=1)


def test_settings_are_frozen_so_a_worker_cannot_mutate_them() -> None:
    s = Phase2Settings()
    with pytest.raises(ValidationError):
        s.ip_hist = 6


def test_the_default_objective_is_the_specs() -> None:
    assert Phase2Settings().objective_spec() == ObjectiveSpec()


def test_both_phases_build_one_objective_from_their_shared_settings() -> None:
    changed = dict(
        cov_norm=True, cov_alpha=0.4, cov_weight=3.0, lam_clear=2.0, tau_clear=0.3,
        smooth_reward=True, lam_thread=0.5, smooth_thread_reward=True,
    )  # fmt: skip
    objective = Phase1Settings(**changed).objective_spec()
    assert objective == Phase2Settings(**changed).objective_spec()
    assert objective == ObjectiveSpec(
        coverage_gain=3.0, cov_alpha=0.4, lambda_margin_clear=2.0, tau_clear_mm=0.3,
        smooth_clearance_reward=True, lambda_margin_thread=0.5,
        smooth_thread_reward=True,
    )  # fmt: skip


def test_every_stage_reads_one_feasibility_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Phase1Settings().feasibility_spec() == FeasibilitySpec(
        min_clearance_mm=0.1, softmin_correction=True
    )
    monkeypatch.setenv("RUTTER_MIN_CLEARANCE", "0.3")
    spec = Phase1Settings().feasibility_spec()
    assert spec.min_clearance_mm == 0.3
    assert spec == Phase2Settings().feasibility_spec()


@pytest.mark.parametrize("settings_cls", [Phase1Settings, Phase2Settings])
@pytest.mark.parametrize("name", sorted(RETIRED_VARIABLES))
def test_a_retired_variable_is_refused_by_name(
    settings_cls: type, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, "0.2")
    instead = re.escape(RETIRED_VARIABLES[name])
    with pytest.raises(ValidationError, match=f"{name} is retired: {instead}"):
        settings_cls()
