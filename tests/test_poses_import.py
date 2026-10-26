"""A rig-poses export, read back over the plan it came from, reopens the session."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aind_rutter.build.assemble import build_runtime_from_config
from aind_rutter.config import ConfigModel
from aind_rutter.domain.commands import (
    SetArcAngle,
    SetProbeCalibrated,
    SetProbeLocalAngles,
    SetProbeOffsetsRA,
    SetProbePastTarget,
    SetProbePositionBearingShank,
)
from aind_rutter.domain.transforms import AffineTransform
from aind_rutter.plan_io.poses import plan_from_poses
from aind_rutter.plan_io.replay import apply_plan_model_to_state
from aind_rutter.plan_io.rig_export import export_plan_geometry
from aind_rutter.session.store import PlanStore
from tests.synthetic_subject import write_subject

POSE_FIELDS = (
    "kind",
    "target",
    "arc",
    "angles_rig_deg",
    "offsets_RA_mm",
    "past_target_mm",
    "position_bearing_shank",
    "tip_RAS_mm",
    "newscale_tip_mm",
)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory: pytest.TempPathFactory) -> ConfigModel:
    subject = write_subject(tmp_path_factory.mktemp("poses"))
    return ConfigModel.from_yaml(subject.config)


def _session(cfg: ConfigModel, calibrate: str | None) -> tuple[PlanStore, object]:
    runtime = build_runtime_from_config(cfg)
    state = runtime.plan_state
    if calibrate is not None:
        cal = AffineTransform(
            rotation=np.array([[0.98, -0.17, 0.0], [0.17, 0.98, 0.0], [0.0, 0.0, 1.0]]),
            translation=np.array([1.0, -2.0, 3.0]),
        )
        state = replace(state, calibrations={calibrate: cal})
    store = PlanStore(state)
    if calibrate is not None:
        store.dispatch(SetProbeCalibrated(name=calibrate, calibrated=True))
    return store, runtime.asset_catalog


def _export(store: PlanStore, catalog) -> dict:
    return export_plan_geometry(store.state, catalog)


def _assert_same_poses(a: dict, b: dict) -> None:
    assert a["arc_angles_rig_deg"] == pytest.approx(b["arc_angles_rig_deg"])
    for name, pose in a["probes"].items():
        for field in POSE_FIELDS:
            want, got = pose.get(field), b["probes"][name].get(field)
            if isinstance(want, dict):
                for k, v in want.items():
                    assert got[k] == (
                        v if not isinstance(v, float) else pytest.approx(v)
                    )
            elif isinstance(want, list):
                np.testing.assert_allclose(got, want, atol=1e-9)
            else:
                assert got == (
                    want if not isinstance(want, float) else pytest.approx(want)
                )


@pytest.mark.parametrize("calibrate", [None, "P1"])
def test_poses_reopen_the_session_they_were_exported_from(cfg, calibrate) -> None:
    store, catalog = _session(cfg, calibrate)
    store.dispatch(SetArcAngle(arc_id="b", ap_deg=-17.5))
    store.dispatch(
        SetProbeLocalAngles(name="P2", ap_local=None, ml_local=-4.0, spin=75.0)
    )
    store.dispatch(
        SetProbeLocalAngles(name="P1", ap_local=None, ml_local=6.0, spin=-30.0)
    )
    store.dispatch(SetProbeOffsetsRA(name="P1", R_mm=0.3, A_mm=-0.2))
    store.dispatch(SetProbeOffsetsRA(name="P2", R_mm=-0.1, A_mm=0.4))
    store.dispatch(SetProbePastTarget(name="P2", past_target_mm=-0.7))
    store.dispatch(SetProbePositionBearingShank(name="P2", position_bearing_shank=3))
    exported = _export(store, catalog)

    fresh, _ = _session(cfg, calibrate)
    apply_plan_model_to_state(plan_from_poses(cfg.plan, exported), fresh)

    _assert_same_poses(exported, _export(fresh, catalog))


def test_a_calibrated_probe_keeps_its_planned_ml(cfg) -> None:
    """The export's ML is the calibration's; the plan's own ML survives."""
    store, catalog = _session(cfg, "P1")
    exported = _export(store, catalog)
    assert exported["probes"]["P1"]["angles_rig_deg"]["ml"] != pytest.approx(
        cfg.plan.probes["P1"].slider_ml
    )
    restored = plan_from_poses(cfg.plan, exported)
    assert restored.probes["P1"].slider_ml == cfg.plan.probes["P1"].slider_ml


def test_a_probe_the_plan_lacks_is_refused(cfg) -> None:
    store, catalog = _session(cfg, None)
    exported = _export(store, catalog)
    exported["probes"]["P9"] = exported["probes"]["P1"]
    with pytest.raises(ValueError, match="P9"):
        plan_from_poses(cfg.plan, exported)
