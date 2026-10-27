"""A probe's manipulator bias: the plan keeps the dial, the probe points at dial
plus bias, and a calibration replaces both."""

from __future__ import annotations

import numpy as np
import pytest

from aind_rutter.domain.catalog import AssetCatalog
from aind_rutter.domain.commands import (
    BindProbeAPToArc,
    SetProbeLocalAngles,
    apply_planning_command,
)
from aind_rutter.domain.plan import PlanningState, ProbePlan
from aind_rutter.domain.pose import dialed_angles, resolved_angles
from aind_rutter.domain.rig import Kinematics
from aind_rutter.domain.transforms import AffineTransform
from aind_rutter.plan_io.rig_export import export_plan_geometry

HEAD_PITCH_DEG = 14.0


def _rx(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _state(ap_bias: float = 2.0, ml_bias: float = -1.5) -> PlanningState:
    probe = ProbePlan(
        kind="neuropixels",
        arc_id="a",
        ml_local=10.0,
        ap_bias=ap_bias,
        ml_bias=ml_bias,
    )
    return PlanningState(
        kinematics=Kinematics(
            arc_angles={"a": -5.0},
            subject_from_rig=AffineTransform(rotation=_rx(HEAD_PITCH_DEG)),
        ),
        probes={"P": probe},
    )


def test_the_probe_points_at_its_dial_plus_its_bias() -> None:
    state = _state()
    assert dialed_angles("P", state)[:2] == (-5.0, 10.0)
    assert resolved_angles("P", state)[:2] == pytest.approx((-3.0, 8.5))


def test_unbinding_keeps_the_dial_so_the_probe_does_not_move() -> None:
    state = _state()
    before = resolved_angles("P", state)
    apply_planning_command(state, BindProbeAPToArc(name="P", bind=False))
    assert state.probes["P"].ap_local == -5.0
    assert resolved_angles("P", state) == pytest.approx(before)


def test_a_bias_is_set_by_command() -> None:
    state = _state(0.0, 0.0)
    apply_planning_command(state, SetProbeLocalAngles(name="P", ap_bias=1.0))
    assert resolved_angles("P", state)[0] == pytest.approx(-4.0)
    assert state.probes["P"].ml_bias == 0.0


def test_the_rig_export_gives_the_dial_and_the_bias() -> None:
    payload = export_plan_geometry(_state(), AssetCatalog(assets={}))
    probe = payload["probes"]["P"]
    assert probe["angles_rig_deg"]["ap"] == pytest.approx(-5.0 + HEAD_PITCH_DEG)
    assert probe["angles_rig_deg"]["ap"] == pytest.approx(
        payload["arc_angles_rig_deg"]["a"]
    )
    assert probe["angles_rig_deg"]["ml"] == pytest.approx(10.0)
    assert probe["bias_deg"] == {"ap": 2.0, "ml": -1.5}
    unbiased = export_plan_geometry(_state(0.0, 0.0), AssetCatalog(assets={}))
    assert "bias_deg" not in unbiased["probes"]["P"]
