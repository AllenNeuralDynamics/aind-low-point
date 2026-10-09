"""A config's ``plan.rig_limits``: parsed, refused when inverted, carried into the
planning state and back out unchanged."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aind_rutter.build.assemble import build_plan_state_from_config
from aind_rutter.config import ConfigModel
from aind_rutter.domain.rig import DEFAULT_AP_RANGE_DEG, DEFAULT_ML_RANGE_DEG
from aind_rutter.plan_io.roundtrip import planning_state_to_plan_model
from tests.config_factories import ConfigFactory


def _config(rig_limits: dict | None) -> ConfigModel:
    raw = ConfigFactory.minimal_config()
    if rig_limits is not None:
        raw["plan"]["rig_limits"] = rig_limits
    return ConfigModel.model_validate(raw)


def test_a_config_without_limits_gets_the_default_rig() -> None:
    limits = build_plan_state_from_config(_config(None)).kinematics.limits
    assert (limits.ap_deg.lo, limits.ap_deg.hi) == DEFAULT_AP_RANGE_DEG
    assert (limits.ml_deg.lo, limits.ml_deg.hi) == DEFAULT_ML_RANGE_DEG


def test_asymmetric_limits_reach_the_kinematics() -> None:
    cfg = _config({"ap_deg": [-75, 46]})
    limits = build_plan_state_from_config(cfg).kinematics.limits
    assert (limits.ap_deg.lo, limits.ap_deg.hi) == (-75.0, 46.0)
    assert (limits.ml_deg.lo, limits.ml_deg.hi) == DEFAULT_ML_RANGE_DEG


@pytest.mark.parametrize("bad", [{"ap_deg": [46, -75]}, {"ml_deg": [10, 10]}])
def test_an_inverted_range_is_refused(bad) -> None:
    with pytest.raises(ValidationError, match="lo < hi"):
        _config(bad)


def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        _config({"ap_max": 46})


def test_saving_a_plan_keeps_its_limits() -> None:
    cfg = _config({"ap_deg": [-75, 46], "ml_deg": [-30, 42]})
    state = build_plan_state_from_config(cfg)
    assert (
        planning_state_to_plan_model(state, cfg.plan).rig_limits == cfg.plan.rig_limits
    )
