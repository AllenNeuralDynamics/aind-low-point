"""Reading a plan config back into a planning state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aind_rutter.config import PlanningModel

if TYPE_CHECKING:
    from aind_rutter.state_change import PlanStore


def apply_plan_model_to_state(plan: PlanningModel, store: "PlanStore") -> list[str]:
    """Apply a loaded :class:`PlanningModel` to a live :class:`PlanStore`.

    Issues per-arc and per-probe planning commands through ``store.dispatch``
    so the controller's subscribers (renderer, collisions, readouts)
    fan out the resulting changes the same way they would for any
    user-initiated edit.

    Probes named in ``plan.probes`` that aren't already in the store's
    state are skipped with a warning printed to stdout — adding/removing
    probes is a config-level concern (the optimizer plumbing assumes
    a fixed probe roster) and not something a plan-only YAML should
    silently do. Returns the list of probe names actually touched.

    Arc angles are dispatched first so any probe bound to that arc
    sees the new AP via the inner reducer's resolved-angles helper.
    """
    from aind_rutter.domain.commands import (
        AssignProbeArc,
        SetArcAngle,
        SetProbeCalibrated,
        SetProbeKind,
        SetProbeLocalAngles,
        SetProbeOffsetsRA,
        SetProbePastTarget,
        SetProbePositionBearingShank,
        SetProbeTarget,
    )

    for arc_id, ap_deg in plan.arcs.items():
        store.dispatch(SetArcAngle(arc_id=str(arc_id), ap_deg=float(ap_deg)))

    touched: list[str] = []
    for name, decl in plan.probes.items():
        if name not in store.state.probes:
            print(f"apply_plan_model_to_state: skipping unknown probe {name!r}")
            continue
        # Kind first — switching kind affects what's a valid pose.
        store.dispatch(SetProbeKind(name=name, kind=str(decl.kind)))
        store.dispatch(
            AssignProbeArc(
                name=name,
                arc_id=decl.arc,
                bind_ap_to_arc=bool(decl.bind_ap_to_arc),
            )
        )
        store.dispatch(
            SetProbeLocalAngles(
                name=name,
                ap_local=(float(decl.ap_local) if decl.ap_local is not None else None),
                ml_local=float(decl.slider_ml),
                spin=float(decl.spin),
            )
        )
        store.dispatch(
            SetProbeOffsetsRA(
                name=name,
                R_mm=float(decl.offsets_RA[0]),
                A_mm=float(decl.offsets_RA[1]),
            )
        )
        store.dispatch(
            SetProbePastTarget(name=name, past_target_mm=float(decl.past_target_mm))
        )
        store.dispatch(
            SetProbePositionBearingShank(
                name=name,
                position_bearing_shank=int(decl.position_bearing_shank),
            )
        )
        store.dispatch(SetProbeCalibrated(name=name, calibrated=bool(decl.calibrated)))
        # Target is always last — it can clear a stale target_key while
        # setting a new RAS-only target without an intervening invalid
        # state, since SetProbeTarget rejects "neither set" in one shot.
        target_key = None
        target_pt_RAS = None
        if hasattr(decl.target, "key"):
            target_key = str(decl.target.key) if decl.target.key else None
        if hasattr(decl.target, "point_RAS"):
            pts = decl.target.point_RAS
            if pts is not None and len(pts) == 3:
                target_pt_RAS = (
                    float(pts[0]),
                    float(pts[1]),
                    float(pts[2]),
                )
        if target_key is not None or target_pt_RAS is not None:
            store.dispatch(
                SetProbeTarget(
                    name=name,
                    target_key=target_key,
                    target_point_RAS=target_pt_RAS,
                )
            )
        touched.append(name)
    return touched
