"""Tests for ``planning.kinematic_violations``."""

from __future__ import annotations

import numpy as np

from aind_rutter.domain.plan import PlanningState, ProbePlan, kinematic_violations
from aind_rutter.domain.rig import JointRange, Kinematics, PoseLimits
from aind_rutter.domain.transforms import AffineTransform


def _state(arc_angles: dict[str, float], probes: dict[str, ProbePlan]) -> PlanningState:
    return PlanningState(
        kinematics=Kinematics(arc_angles=dict(arc_angles), limits=PoseLimits()),
        probes=dict(probes),
    )


def _pitch(deg: float) -> AffineTransform:
    """A head pitch of ``deg`` about L, as the config's ``subject_from_rig``."""
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return AffineTransform(rotation=np.array([[1, 0, 0], [0, c, -s], [0, s, c]]))


def _probe(arc: str | None, ml: float = 0.0) -> ProbePlan:
    return ProbePlan(kind="2.4", arc_id=arc, ml_local=ml)


def _riders(*arcs: str) -> dict[str, ProbePlan]:
    return {f"P_{a}": _probe(a) for a in arcs}


class TestArcAPSeparation:
    def test_arcs_far_apart_are_fine(self):
        s = _state({"a": 0.0, "b": 30.0}, _riders("a", "b"))
        v = kinematic_violations(s)
        assert v["arc_ap"] == set()

    def test_arcs_too_close_flags_pair(self):
        # 16° threshold; 0 vs 10 is < 16°
        s = _state({"a": 0.0, "b": 10.0}, _riders("a", "b"))
        v = kinematic_violations(s)
        assert v["arc_ap"] == {("a", "b")}

    def test_three_arcs_all_close(self):
        s = _state({"a": 0.0, "b": 5.0, "c": 10.0}, _riders("a", "b", "c"))
        v = kinematic_violations(s)
        # All three pairs are <16° apart
        assert v["arc_ap"] == {("a", "b"), ("a", "c"), ("b", "c")}

    def test_threshold_boundary_excluded(self):
        # Exactly 16° apart is NOT a violation (we use strict <).
        s = _state({"a": 0.0, "b": 16.0}, _riders("a", "b"))
        assert kinematic_violations(s)["arc_ap"] == set()

    def test_an_arc_without_probes_is_ignored(self):
        s = _state({"a": 0.0, "b": 5.0, "c": 40.0}, _riders("a", "c"))
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
            {"P2": _probe("a", ml=0.0), "P1": _probe("a", ml=2.0), "P3": _probe("b")},
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
                "P3": _probe("b"),
            },
        )
        v = kinematic_violations(s)
        # arcs 5° apart > new 3° threshold → no clash
        assert v["arc_ap"] == set()
        # MLs 4° apart > new 3° threshold → no clash
        assert v["within_arc_ml"] == set()


class TestAngularRanges:
    """The rig's AP range is in rig angles (subject AP plus head pitch) and may
    be asymmetric; leaving either range is reported, never clamped."""

    @staticmethod
    def _limited(arc_ap: float, ml: float, pitch_deg: float) -> PlanningState:
        return PlanningState(
            kinematics=Kinematics(
                arc_angles={"a": arc_ap},
                limits=PoseLimits(
                    ap_deg=JointRange(-75.0, 46.0), ml_deg=JointRange(-42.0, 42.0)
                ),
                subject_from_rig=_pitch(pitch_deg),
            ),
            probes={"P1": _probe("a", ml=ml)},
        )

    def test_rig_ap_is_subject_ap_plus_pitch(self):
        # Subject 32 at pitch 14 is rig 46, the edge: inside.
        assert kinematic_violations(self._limited(32.0, 0.0, 14.0))["ap_range"] == set()
        # Subject 33 is rig 47: outside, though well inside the symmetric ±75.
        v = kinematic_violations(self._limited(33.0, 0.0, 14.0))
        assert v["ap_range"] == {("P1",)}

    def test_the_low_end_keeps_its_own_limit(self):
        assert (
            kinematic_violations(self._limited(-89.0, 0.0, 14.0))["ap_range"] == set()
        )
        v = kinematic_violations(self._limited(-90.0, 0.0, 14.0))
        assert v["ap_range"] == {("P1",)}

    def test_ml_outside_range_flagged(self):
        v = kinematic_violations(self._limited(0.0, 45.0, 14.0))
        assert v["ml_range"] == {("P1",)}
        assert v["ap_range"] == set()

    def test_setting_an_arc_past_the_range_keeps_the_value(self):
        s = self._limited(0.0, 0.0, 14.0)
        assert s.kinematics.set_arc("a", 60.0) == 60.0
        assert s.kinematics.arc_angles["a"] == 60.0
