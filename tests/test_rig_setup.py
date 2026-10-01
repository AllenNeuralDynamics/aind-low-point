"""Tests for ``domain.rig_setup``: a plan's rig setup and the rework between two."""

from __future__ import annotations

import itertools

import pytest

from aind_rutter.domain.plan import PlanningState, ProbePlan
from aind_rutter.domain.rig import Kinematics
from aind_rutter.domain.rig_setup import (
    ArcSetup,
    RigSetup,
    SetupChanges,
    SetupCosts,
    Slot,
    rig_setup,
    setup_changes,
)

Q, S = "quadbase-dovetail", "2.1"


def _arc(ap: float, *slots: tuple[str, float, float]) -> ArcSetup:
    return ArcSetup(ap, tuple(Slot(*s) for s in slots))


BASE: RigSetup = (
    _arc(-20.0, (Q, -18.0, 0.0), (S, 10.0, 90.0)),
    _arc(10.0, (S, 0.0, 180.0)),
)


class TestRigSetup:
    def test_arcs_by_ap_and_probes_by_ml(self):
        state = PlanningState(
            kinematics=Kinematics(arc_angles={"a": 30.0, "b": -10.0}),
            probes={
                "P1": ProbePlan(kind=S, arc_id="a", ml_local=20.0, spin=-90.0),
                "P2": ProbePlan(kind=Q, arc_id="a", ml_local=-5.0),
                "P3": ProbePlan(kind=S, arc_id="b", ml_local=0.0, spin=45.0),
            },
        )
        assert rig_setup(state) == (
            _arc(-10.0, (S, 0.0, 45.0)),
            _arc(30.0, (Q, -5.0, 0.0), (S, 20.0, 270.0)),
        )

    def test_an_arc_without_probes_and_a_probe_without_an_arc_are_left_out(self):
        state = PlanningState(
            kinematics=Kinematics(arc_angles={"a": 0.0, "b": 40.0}),
            probes={
                "P1": ProbePlan(kind=S, arc_id="a"),
                "P2": ProbePlan(kind=S, arc_id=None),
            },
        )
        assert rig_setup(state) == (_arc(0.0, (S, 0.0, 0.0)),)


class TestSetupChanges:
    def test_the_same_setup_takes_nothing(self):
        assert setup_changes(BASE, BASE) == SetupChanges()

    def test_angles_within_tolerance_take_nothing(self):
        after = (
            _arc(-19.8, (Q, -18.3, 3.0), (S, 10.0, 90.0)),
            _arc(10.0, (S, 0.0, 180.0)),
        )
        assert setup_changes(BASE, after) == SetupChanges()

    def test_each_slide_and_turn_is_counted(self):
        after = (
            _arc(-25.0, (Q, -18.0, 0.0), (S, 14.0, 120.0)),
            _arc(10.0, (S, 0.0, 180.0)),
        )
        assert setup_changes(BASE, after) == SetupChanges(
            arc_moves=1, slider_moves=1, spins=1
        )

    def test_spin_wraps(self):
        before = (_arc(0.0, (S, 0.0, 359.0)),)
        after = (_arc(0.0, (S, 0.0, 2.0)),)
        assert setup_changes(before, after) == SetupChanges()

    def test_a_kind_swap_is_one_off_and_one_on(self):
        after = (BASE[0], _arc(10.0, (Q, 0.0, 180.0)))
        assert setup_changes(BASE, after) == SetupChanges(mounts=2)

    def test_a_probe_added_between_two_leaves_them_alone(self):
        after = (
            _arc(-20.0, (Q, -18.0, 0.0), (S, -2.0, 0.0), (S, 10.0, 90.0)),
            BASE[1],
        )
        assert setup_changes(BASE, after) == SetupChanges(mounts=1)

    def test_an_arc_added_pays_for_its_probes(self):
        after = (*BASE, _arc(40.0, (S, -10.0, 0.0), (S, 10.0, 0.0)))
        assert setup_changes(BASE, after) == SetupChanges(arc_mounts=1, mounts=2)

    def test_probes_on_an_arc_cannot_pass_each_other(self):
        before = (_arc(0.0, (Q, -10.0, 0.0), (S, 10.0, 0.0)),)
        after = (_arc(0.0, (S, -10.0, 0.0), (Q, 10.0, 0.0)),)
        # One probe stays and slides across; the other comes off and goes back on.
        assert setup_changes(before, after) == SetupChanges(mounts=2, slider_moves=1)

    def test_arcs_cannot_pass_each_other(self):
        before = (_arc(-20.0, (Q, 0.0, 0.0)), _arc(20.0, (S, 0.0, 0.0)))
        after = (_arc(-20.0, (S, 0.0, 0.0)), _arc(20.0, (Q, 0.0, 0.0)))
        changes = setup_changes(before, after)
        assert changes.arc_moves == 0
        assert SetupCosts().of(changes) > 2 * SetupCosts().arc_move

    def test_costs_decide_between_remounting_and_sliding(self):
        before = (_arc(0.0, (S, 0.0, 0.0)),)
        after = (_arc(30.0, (S, 0.0, 0.0)),)
        assert setup_changes(before, after) == SetupChanges(arc_moves=1)
        dear_slide = SetupCosts(arc_move=100.0, mount=1.0, arc_mount=1.0)
        assert setup_changes(before, after, dear_slide) == SetupChanges(
            arc_mounts=2, mounts=2
        )

    @pytest.mark.parametrize(
        "other", [(), (BASE[1],), (_arc(-40.0, (S, 5.0, 0.0)), BASE[1])]
    )
    def test_the_way_back_costs_the_same(self, other: RigSetup):
        costs = SetupCosts()
        there, back = setup_changes(BASE, other), setup_changes(other, BASE)
        assert costs.of(there) == costs.of(back)

    def test_matches_brute_force_on_one_arc(self):
        """Every way of pairing slots in order, against the alignment."""
        costs = SetupCosts()
        kinds = [Q, S]
        slots = [Slot(k, ml, 0.0) for k in kinds for ml in (-20.0, 0.0, 20.0)]

        def brute(xs: tuple[Slot, ...], ys: tuple[Slot, ...]) -> float:
            best = float("inf")
            for n in range(min(len(xs), len(ys)) + 1):
                for i in itertools.combinations(range(len(xs)), n):
                    for j in itertools.combinations(range(len(ys)), n):
                        cost = costs.mount * (len(xs) + len(ys) - 2 * n)
                        for a, b in zip(i, j):
                            if xs[a].kind != ys[b].kind:
                                cost += 2 * costs.mount
                            else:
                                cost += costs.slider_move * (
                                    xs[a].ml_deg != ys[b].ml_deg
                                )
                        best = min(best, cost)
            return best

        arcs = [
            tuple(sorted(c, key=lambda s: s.ml_deg))
            for n in range(1, 4)
            for c in itertools.combinations(slots, n)
            if len({s.ml_deg for s in c}) == n
        ]
        for xs, ys in itertools.product(arcs, repeat=2):
            changes = setup_changes((ArcSetup(0.0, xs),), (ArcSetup(0.0, ys),))
            # Keeping the arc is never dearer than replacing it at these costs.
            assert costs.of(changes) == brute(xs, ys)
