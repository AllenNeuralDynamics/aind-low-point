"""Tests for ``planning.kinematic_violations``."""

from __future__ import annotations

from aind_low_point.planning import (
    Kinematics,
    PlanningState,
    PoseLimits,
    ProbePlan,
    kinematic_violations,
)


def _state(arc_angles: dict[str, float], probes: dict[str, ProbePlan]) -> PlanningState:
    return PlanningState(
        kinematics=Kinematics(arc_angles=dict(arc_angles), limits=PoseLimits()),
        probes=dict(probes),
    )


def _probe(arc: str | None, ml: float = 0.0) -> ProbePlan:
    return ProbePlan(kind="2.4", arc_id=arc, ml_local=ml)


class TestArcAPSeparation:
    def test_arcs_far_apart_are_fine(self):
        s = _state({"a": 0.0, "b": 30.0}, {})
        v = kinematic_violations(s)
        assert v["arc_ap"] == set()

    def test_arcs_too_close_flags_pair(self):
        # 16° threshold; 0 vs 10 is < 16°
        s = _state({"a": 0.0, "b": 10.0}, {})
        v = kinematic_violations(s)
        assert v["arc_ap"] == {("a", "b")}

    def test_three_arcs_all_close(self):
        s = _state({"a": 0.0, "b": 5.0, "c": 10.0}, {})
        v = kinematic_violations(s)
        # All three pairs are <16° apart
        assert v["arc_ap"] == {("a", "b"), ("a", "c"), ("b", "c")}

    def test_threshold_boundary_excluded(self):
        # Exactly 16° apart is NOT a violation (we use strict <).
        s = _state({"a": 0.0, "b": 16.0}, {})
        assert kinematic_violations(s)["arc_ap"] == set()


class TestWithinArcMLSeparation:
    def test_two_probes_far_apart_on_one_arc(self):
        s = _state(
            {"a": 0.0},
            {"P1": _probe("a", ml=-20.0), "P2": _probe("a", ml=20.0)},
        )
        assert kinematic_violations(s)["within_arc_ml"] == set()

    def test_two_probes_too_close_on_one_arc(self):
        s = _state(
            {"a": 0.0},
            {"P1": _probe("a", ml=0.0), "P2": _probe("a", ml=10.0)},
        )
        assert kinematic_violations(s)["within_arc_ml"] == {("P1", "P2")}

    def test_close_probes_on_different_arcs_dont_clash(self):
        # ML=0 and ML=10 on DIFFERENT arcs do NOT trigger the within-arc
        # constraint (different arcs have separate sliders / hardware).
        s = _state(
            {"a": 0.0, "b": 30.0},  # arcs 30° apart so no AP clash
            {"P1": _probe("a", ml=0.0), "P2": _probe("b", ml=10.0)},
        )
        assert kinematic_violations(s)["within_arc_ml"] == set()

    def test_unbound_probe_ignored(self):
        # A probe without an arc_id can't violate within-arc rules.
        s = _state(
            {"a": 0.0},
            {"P1": _probe("a", ml=0.0), "P2": _probe(None, ml=5.0)},
        )
        assert kinematic_violations(s)["within_arc_ml"] == set()


class TestPairOrderingStable:
    def test_pairs_are_sorted_for_dedup(self):
        s = _state(
            {"a": 0.0, "b": 5.0},
            {"P2": _probe("a", ml=0.0), "P1": _probe("a", ml=2.0)},
        )
        v = kinematic_violations(s)
        assert v["arc_ap"] == {("a", "b")}
        # Within the tuple, names are sorted regardless of dict order.
        assert v["within_arc_ml"] == {("P1", "P2")}


class TestCustomThreshold:
    def test_pose_limits_threshold_is_respected(self):
        s = PlanningState(
            kinematics=Kinematics(
                arc_angles={"a": 0.0, "b": 5.0},
                limits=PoseLimits(
                    min_arc_ap_separation_deg=3.0,
                    min_within_arc_ml_separation_deg=3.0,
                ),
            ),
            probes={
                "P1": _probe("a", ml=0.0),
                "P2": _probe("a", ml=4.0),
            },
        )
        v = kinematic_violations(s)
        # arcs 5° apart > new 3° threshold → no clash
        assert v["arc_ap"] == set()
        # MLs 4° apart > new 3° threshold → no clash
        assert v["within_arc_ml"] == set()
