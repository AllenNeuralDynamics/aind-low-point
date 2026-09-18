"""Rig-readability ordering belongs to the exported document, not the session.

`export_plan_geometry` is a reporter, and it used to relabel the arcs of whatever
state it was handed — which, from the app's Export-plan button, is the live
session. It renamed them without a dispatch, so nothing watching the plan heard
about it, the UI kept showing the old labels, and the next save wrote the new
ones.
"""

from __future__ import annotations

import numpy as np

from aind_rutter.assets import AssetCatalog
from aind_rutter.core import AffineTransform
from aind_rutter.planning import Kinematics, PlanningState, ProbePlan
from aind_rutter.runtime.export import export_plan_geometry, reorder_plan_for_rig

# Deliberately not in AP order, so relabelling is not the identity: by AP the
# order is z (+20), x (-5), y (-40), which becomes a, b, c.
ARC_ANGLES = {"x": -5.0, "y": -40.0, "z": 20.0}


def _probe(arc_id: str, ml: float) -> ProbePlan:
    return ProbePlan(
        kind="neuropixels",
        arc_id=arc_id,
        bind_ap_to_arc=True,
        ml_local=ml,
        past_target_mm=1.0,
        offsets_RA=(0.0, 0.0),
    )


def _state() -> PlanningState:
    return PlanningState(
        kinematics=Kinematics(
            arc_angles=dict(ARC_ANGLES),
            subject_from_rig=AffineTransform(rotation=np.eye(3)),
        ),
        probes={
            "P_low_ml": _probe("z", -8.0),
            "P_high_ml": _probe("z", 12.0),
            "P_on_y": _probe("y", 0.0),
            "P_on_x": _probe("x", 4.0),
        },
    )


def test_arcs_are_relabelled_most_positive_ap_first() -> None:
    ordered = reorder_plan_for_rig(_state())
    assert ordered.kinematics.arc_angles == {"a": 20.0, "b": -5.0, "c": -40.0}


def test_probes_are_ordered_by_arc_then_descending_ml() -> None:
    ordered = reorder_plan_for_rig(_state())
    assert list(ordered.probes) == ["P_high_ml", "P_low_ml", "P_on_x", "P_on_y"]
    assert [p.arc_id for p in ordered.probes.values()] == ["a", "a", "b", "c"]


def test_the_input_state_is_left_alone() -> None:
    """The defect: this used to relabel the caller's arcs in place."""
    state = _state()
    before_arcs = dict(state.kinematics.arc_angles)
    before_order = list(state.probes)
    before_ids = {name: plan.arc_id for name, plan in state.probes.items()}

    reorder_plan_for_rig(state)

    assert state.kinematics.arc_angles == before_arcs
    assert list(state.probes) == before_order
    assert {n: p.arc_id for n, p in state.probes.items()} == before_ids


def test_the_copy_does_not_share_probe_objects_with_the_input() -> None:
    state = _state()
    ordered = reorder_plan_for_rig(state)
    for name, plan in ordered.probes.items():
        assert plan is not state.probes[name]


def test_pose_values_survive_the_relabelling() -> None:
    state = _state()
    ordered = reorder_plan_for_rig(state)
    for name, plan in ordered.probes.items():
        source = state.probes[name]
        assert plan.ml_local == source.ml_local
        assert plan.past_target_mm == source.past_target_mm
        assert plan.kind == source.kind


def test_exporting_does_not_relabel_the_session() -> None:
    """The reported path: the app hands the Export-plan button its live state."""
    state = _state()
    payload = export_plan_geometry(state, AssetCatalog(assets={}))

    assert set(payload["arc_angles_subject_deg"]) == {"a", "b", "c"}
    assert state.kinematics.arc_angles == ARC_ANGLES
    assert state.probes["P_on_x"].arc_id == "x"
