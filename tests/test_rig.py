"""A config's ``plan.rig``: parsed, refused when malformed, carried into the
planning state and back out unchanged, and read by the pipeline's specs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aind_rutter.build.assemble import build_plan_state_from_config
from aind_rutter.config import ConfigModel
from aind_rutter.domain.rig import (
    DEFAULT_AP_RANGE_DEG,
    DEFAULT_ML_RANGE_DEG,
    ArcRig,
    JointRange,
)
from aind_rutter.optimization.pipeline.payloads import pinned_spec_differences
from aind_rutter.optimization.pipeline.settings import Phase1Settings, Phase2Settings
from aind_rutter.plan_io.roundtrip import planning_state_to_plan_model
from tests.config_factories import ConfigFactory


def _config(rig: dict | None) -> ConfigModel:
    raw = ConfigFactory.minimal_config()
    if rig is not None:
        raw["plan"]["rig"] = rig
    return ConfigModel.model_validate(raw)


def _rig(rig: dict | None) -> ArcRig:
    return build_plan_state_from_config(_config(rig)).kinematics.rig


def test_a_config_without_a_rig_gets_the_default_rig() -> None:
    assert _rig(None) == ArcRig()


def test_a_configs_rig_reaches_the_kinematics() -> None:
    rig = _rig(
        {
            "kind": "arc",
            "name": "dhc",
            "ap_deg": [-75, 46],
            "min_arc_ap_sep_deg": 18,
            "min_intra_arc_ml_sep_deg": 14,
        }
    )
    assert rig.name == "dhc"
    assert (rig.ap_deg.lo, rig.ap_deg.hi) == (-75.0, 46.0)
    assert rig.ml_range_deg == DEFAULT_ML_RANGE_DEG
    assert (rig.min_arc_ap_sep_deg, rig.min_intra_arc_ml_sep_deg) == (18.0, 14.0)


@pytest.mark.parametrize("bad", [{"ap_deg": [46, -75]}, {"ml_deg": [10, 10]}])
def test_an_inverted_range_is_refused(bad) -> None:
    with pytest.raises(ValidationError, match="lo < hi"):
        _config(bad)


@pytest.mark.parametrize(
    "bad", [{"ap_max": 46}, {"kind": "octopus"}, {"min_arc_ap_sep_deg": 0}]
)
def test_a_malformed_rig_is_refused(bad) -> None:
    with pytest.raises(ValidationError):
        _config(bad)


def test_saving_a_plan_keeps_its_rig() -> None:
    cfg = _config({"ap_deg": [-75, 46], "ml_deg": [-30, 42], "name": "dhc"})
    state = build_plan_state_from_config(cfg)
    assert planning_state_to_plan_model(state, cfg.plan).rig == cfg.plan.rig


def test_the_caps_follow_the_separations() -> None:
    rig = ArcRig(ap_deg=JointRange(-75, 46), min_intra_arc_ml_sep_deg=21.0)
    assert rig.max_arcs() == 121 // 16 + 1
    assert rig.max_probes_per_arc() == 84 // 21 + 1
    assert ArcRig().reachable_ap_window_deg(14.0) == (
        DEFAULT_AP_RANGE_DEG[0] - 14.0,
        DEFAULT_AP_RANGE_DEG[1] - 14.0,
    )


def test_the_feasibility_spec_takes_the_rigs_separations() -> None:
    rig = ArcRig(min_arc_ap_sep_deg=18.0, min_intra_arc_ml_sep_deg=14.0)
    spec = Phase1Settings().with_rig(rig).feasibility_spec()
    assert (spec.min_arc_ap_sep_deg, spec.min_intra_arc_ml_sep_deg) == (18.0, 14.0)


def test_specs_without_a_rig_are_refused() -> None:
    with pytest.raises(RuntimeError, match="no rig attached"):
        Phase1Settings().feasibility_spec()


def test_a_pool_pins_only_ranges_off_the_default() -> None:
    assert Phase1Settings().with_rig(ArcRig()).pinned_specs()["rig"] == {}
    dhc = ArcRig(ap_deg=JointRange(-75.0, 46.0))
    assert Phase1Settings().with_rig(dhc).pinned_specs()["rig"] == {
        "ap_deg": [-75.0, 46.0]
    }


def test_phase2_refuses_a_pool_built_for_another_rig() -> None:
    dhc = ArcRig(ap_deg=JointRange(-75.0, 46.0))
    pool = {"records": [], **Phase1Settings().with_rig(dhc).pinned_specs()}
    default = Phase2Settings().with_rig(ArcRig()).pinned_specs()
    assert pinned_spec_differences(pool, default) == [
        "rig.ap_deg: [-75.0, 46.0] vs None"
    ]
    # A pool recorded before rigs were pinned matches the default rig.
    del pool["rig"]
    assert pinned_spec_differences(pool, default) == []
