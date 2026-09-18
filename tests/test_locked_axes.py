"""Which pose controls a probe cannot drive, decided once.

The rule used to be written in four places that disagreed: `runtime/build.py`
locked on `calibrated` alone, while the resolver and both slider handlers
required a calibration to actually be loaded. A probe declaring `calibrated`
with no calibration file got grayed-out sliders that were in fact live.
"""

from __future__ import annotations

import numpy as np
import pytest

from aind_rutter.planning import (
    CALIBRATION_LOCKED_AXES,
    Kinematics,
    PlanningState,
    ProbePlan,
    locked_axes_for,
)


class _Calibration:
    """Enough of a calibration for the lock rule; the rotation is not read."""

    rotation = np.eye(3)


def _state(*, calibrated: bool, has_calibration: bool) -> PlanningState:
    state = PlanningState(
        probes={
            "P1": ProbePlan(kind="2.1", arc_id="a", calibrated=calibrated),
        },
        kinematics=Kinematics(arc_angles={"a": 0.0}),
    )
    if has_calibration:
        state.calibrations["P1"] = _Calibration()
    return state


def test_a_calibrated_probe_with_its_calibration_locks_both_tilts() -> None:
    state = _state(calibrated=True, has_calibration=True)
    assert locked_axes_for(state, "P1") == CALIBRATION_LOCKED_AXES


def test_declaring_calibrated_without_one_locks_nothing() -> None:
    """The disagreement: the sliders used to gray out here, wrongly.

    With no calibration loaded the resolver falls back to the arc and the local
    angle, so both controls do work.
    """
    assert locked_axes_for(_state(calibrated=True, has_calibration=False), "P1") == (
        frozenset()
    )


def test_an_uncalibrated_probe_locks_nothing() -> None:
    assert locked_axes_for(_state(calibrated=False, has_calibration=True), "P1") == (
        frozenset()
    )


def test_being_bound_to_an_arc_is_not_a_lock() -> None:
    """The AP control still works when arc-bound — it moves the whole arc."""
    state = _state(calibrated=False, has_calibration=False)
    state.probes["P1"].bind_ap_to_arc = True
    assert "ap_tilt" not in locked_axes_for(state, "P1")


def test_an_unknown_probe_locks_nothing() -> None:
    assert locked_axes_for(_state(calibrated=True, has_calibration=True), "P9") == (
        frozenset()
    )


@pytest.mark.parametrize("axis", sorted(CALIBRATION_LOCKED_AXES))
def test_the_locked_names_are_the_slider_names(axis: str) -> None:
    """The set is consumed as a Vue `includes(...)` over slider ids."""
    assert axis in {"ap_tilt", "ml_tilt"}
