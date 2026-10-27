"""``rutter-tracks``: implant fits from insertion tracks, manipulator angle
biases and the plan check, on synthetic bores."""

from __future__ import annotations

import numpy as np
import pytest

from aind_rutter.optimization.geometry.holes import Hole
from aind_rutter.optimization.geometry.primitives import HoleSection
from aind_rutter.plan_io.insertion_tracks import (
    SHANK_HALF_WIDTH_MM,
    PlacedTrack,
    Track,
    bore_clearance,
    fit_implants,
    lines_clearance,
)
from aind_rutter.plan_io.manipulator_bias import (
    Bias,
    Sample,
    dial_sheet,
    planned_angles,
    summarize,
)
from aind_rutter.plan_io.plan_check import (
    ProbeCheck,
    dialed_plan,
    rethread,
    rethreaded_plan,
)

UP = np.array([0.0, 0.0, 1.0])
DOWN = -UP
RADIUS = 0.4


def _hole(k: int, top: tuple[float, float, float]) -> Hole:
    top_ = np.array(top)
    sections = [
        HoleSection(UP, top_ + s * UP, RADIUS, RADIUS, 0.0) for s in (0.0, -0.5, -1.0)
    ]
    return Hole(k, UP, top_, sections)


HOLES = {
    1: _hole(1, (2.0, -2.0, 0.0)),
    2: _hole(2, (-1.5, -1.0, 0.2)),
    3: _hole(3, (0.5, 2.5, -0.3)),
}


def test_clearance_is_the_gap_to_the_bore_wall() -> None:
    hole = HOLES[1]
    p = hole.ref_point + np.array([0.1, 0.0, 1.0])
    assert bore_clearance(p, DOWN, hole) == pytest.approx(RADIUS - 0.1)
    # Moving the bore by the line's offset centres it.
    assert bore_clearance(p, DOWN, hole, np.array([0.1, 0.0, 0.0])) == pytest.approx(
        RADIUS
    )
    assert bore_clearance(p, DOWN, hole, np.array([-0.5, 0.0, 0.0])) < 0


def _track(subject: str, manipulator: str, k: int, offset: np.ndarray) -> PlacedTrack:
    """A vertical track through hole ``k`` of an implant moved by ``offset``."""
    p = HOLES[k].ref_point + offset + np.array([0.0, 0.0, 2.0])
    t = Track(subject, "s", "cal", manipulator, k, (0.0, 0.0, 0.0), "2.1", 0.0)
    return PlacedTrack(t, k, [(p, DOWN)])


def test_a_fit_moves_the_implant_until_every_track_threads() -> None:
    moved = np.array([0.0, 0.5, 0.0])  # beyond the bore: no track threads at design
    placed = [_track("m", "x", k, moved) for k in HOLES]
    assert max(lines_clearance(p.lines, HOLES[p.hole]) for p in placed) < 0
    fit = fit_implants(placed, HOLES, errors="none")
    assert min(fit.clearance) > 0
    assert np.linalg.norm(fit.shifts["m"] - moved) < RADIUS - SHANK_HALF_WIDTH_MM


def test_a_manipulator_error_is_shared_by_its_tracks_only() -> None:
    """One manipulator reads 0.6 mm off on every track; the others read true."""
    off = np.array([0.6, 0.0, 0.0])
    placed = [_track("m", "good", k, np.zeros(3)) for k in HOLES]
    placed += [_track("m", "bad", k, off) for k in HOLES]
    fit = fit_implants(placed, HOLES, errors="manipulator", sigma_mm=1.0)
    assert min(fit.clearance) > 0
    assert fit.group_errors["bad"][0] - fit.group_errors["good"][0] > 0.6 - 2 * RADIUS


def test_planned_angles_follow_the_arc_unless_unbound() -> None:
    plan = {
        "arcs": {"a": -10.0},
        "probes": {
            "P1": {"arc": "a", "slider_ml": 5.0},
            "P2": {
                "arc": "a",
                "slider_ml": -3.0,
                "bind_ap_to_arc": False,
                "ap_local": 7.0,
            },
        },
    }
    assert planned_angles(plan) == {"P1": ("a", -10.0, 5.0), "P2": ("a", 7.0, -3.0)}


def _sample(m: str, d_ap: float, d_ml: float) -> Sample:
    return Sample(m, "s", "c", "P", "a", 0.0, 0.0, d_ap, d_ml)


def test_ml_is_corrected_only_where_its_sign_holds() -> None:
    biases = summarize(
        [
            _sample("steady", 3.0, 2.0),
            _sample("steady", 3.4, 2.4),
            _sample("steady", 2.6, 1.6),
        ]
        + [_sample("noisy", 1.0, 3.0), _sample("noisy", 2.0, -2.0)]
    )
    assert biases["steady"].ml_correction == pytest.approx(2.0)
    assert biases["noisy"].ml_correction == 0.0
    assert biases["noisy"].ap_correction == pytest.approx(1.5)


def test_an_arc_is_dialed_back_by_its_manipulators_mean_bias() -> None:
    plan = {
        "arcs": {"a": 0.0, "b": -20.0},
        "probes": {
            "P1": {"arc": "a", "slider_ml": 10.0},
            "P2": {"arc": "a", "slider_ml": -10.0},
            "P3": {"arc": "b", "slider_ml": 0.0},
        },
    }
    biases = {"m1": Bias(3, 2.0, 0.5, 2.0, 0.5), "m2": Bias(3, 4.0, 0.5, 0.0, 3.0)}
    sheet = dial_sheet(plan, {"P1": "m1", "P2": "m2", "P3": "new"}, biases, 14.0)
    assert sheet.arcs["a"]["dial_rig_ap"] == pytest.approx(14.0 - 3.0)
    # A manipulator with no record takes the measured ones' mean.
    assert sheet.arcs["b"]["dial_rig_ap"] == pytest.approx(-6.0 - 3.0)
    assert sheet.probes["P1"]["dial_ml"] == pytest.approx(8.0)
    assert sheet.probes["P2"]["dial_ml"] == pytest.approx(-10.0)
    # Each lands off the shared arc by its own bias less the arc's mean.
    assert sheet.probes["P1"]["landing_ap_offset"] == pytest.approx(-1.0)
    assert sheet.probes["P2"]["landing_ap_offset"] == pytest.approx(1.0)


def test_rethread_moves_a_probe_back_inside_by_the_margin() -> None:
    hole = HOLES[2]
    lines = [(hole.ref_point + np.array([0.5, 0.0, 1.0]), DOWN)]
    v = rethread(lines, hole, np.zeros(3), 0.05)
    moved = [(p + np.r_[v, 0.0], d) for p, d in lines]
    assert lines_clearance(moved, hole) == pytest.approx(0.05, abs=1e-4)
    assert v[0] < 0 and abs(v[1]) < 1e-6
    assert not rethread(moved, hole, np.zeros(3), 0.0).any()


def test_a_rethreaded_plan_moves_only_the_offsets() -> None:
    plan = {
        "arcs": {"a": 1.0},
        "probes": {"P": {"arc": "a", "offsets_RA": [0.1, -0.2]}, "Q": {"arc": "a"}},
    }
    rows = [
        ProbeCheck("P", 1, -0.1, -0.2, (0.3, 0.4)),
        ProbeCheck("Q", 2, 0.1, 0.0, (0.0, 0.0)),
    ]
    out = rethreaded_plan(plan, rows)
    assert out["probes"]["P"]["offsets_RA"] == pytest.approx([0.4, 0.2])
    assert out["probes"]["Q"]["offsets_RA"] == [0.0, 0.0]
    assert plan["probes"]["P"]["offsets_RA"] == [0.1, -0.2]


def test_a_dialed_plan_lands_each_probe_at_plan_plus_its_arc_leftover() -> None:
    plan = {
        "arcs": {"a": 0.0},
        "probes": {
            "P1": {"arc": "a", "slider_ml": 10.0},
            "P2": {"arc": "a", "slider_ml": -10.0},
        },
    }
    biases = {"m1": Bias(3, 2.0, 0.5, 2.0, 0.5), "m2": Bias(3, 4.0, 0.5, 0.0, 3.0)}
    sheet = dial_sheet(plan, {"P1": "m1", "P2": "m2"}, biases, 14.0)
    out = dialed_plan(plan, sheet)
    assert out["arcs"]["a"] == pytest.approx(-3.0)
    for name, d in sheet.probes.items():
        p = out["probes"][name]
        landed_ap = out["arcs"]["a"] + p["ap_bias"]
        assert landed_ap == pytest.approx(d["landing_ap_offset"])
        assert p["slider_ml"] + p["ml_bias"] == pytest.approx(
            plan["probes"][name]["slider_ml"]
        )
