"""Commands to change the plan state"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from aind_low_point.planning import PlanningState, _resolved_angles


@dataclass(frozen=True)
class SetProbeLocalAngles:
    """Edit per-probe local angles (used when not bound to arc or when you unbind)."""

    name: str
    ap_local: Optional[float] = None  # deg
    ml_local: Optional[float] = None  # deg
    spin: Optional[float] = None  # deg


@dataclass(frozen=True)
class SetProbeOffsetsRA:
    """Set absolute R/A offsets (in mm)."""

    name: str
    R_mm: Optional[float] = None
    A_mm: Optional[float] = None


@dataclass(frozen=True)
class NudgeProbeOffsetsRA:
    """Nudge offsets (delta in mm)."""

    name: str
    dR_mm: float = 0.0
    dA_mm: float = 0.0


@dataclass(frozen=True)
class SetProbePastTarget:
    """Set relative depth (mm). Positive increases insertion past the target."""

    name: str
    past_target_mm: float


@dataclass(frozen=True)
class NudgeProbePastTarget:
    """Nudge depth (delta mm)."""

    name: str
    d_mm: float


@dataclass(frozen=True)
class SetProbeTarget:
    """
    Choose a target. Exactly one of target_key or target_point_RAS must be provided.
    Passing None clears that field; use to switch between the two forms.
    """

    name: str
    target_key: Optional[str] = None
    target_point_RAS: Optional[Tuple[float, float, float]] = None


# ---------- Arc & policy edits ----------


@dataclass(frozen=True)
class SetArcAngle:
    """Set AP angle for an arc (deg)."""

    arc_id: str
    ap_deg: float


@dataclass(frozen=True)
class AssignProbeArc:
    """Assign/unassign probe to an arc and optionally (un)bind AP to arc."""

    name: str
    arc_id: Optional[str]  # None = unassign from arc
    bind_ap_to_arc: Optional[bool] = None


@dataclass(frozen=True)
class BindProbeAPToArc:
    """Bind/unbind AP to the probe’s current arc."""

    name: str
    bind: bool
    freeze_effective_on_unbind: bool = (
        True  # capture current AP into ap_local on unbind
    )


@dataclass(frozen=True)
class SetProbeCalibrated:
    """Mark plan as 'use calibration if available'."""

    name: str
    calibrated: bool


PlanningCommand = Union[
    SetProbeLocalAngles,
    SetProbeOffsetsRA,
    NudgeProbeOffsetsRA,
    SetProbePastTarget,
    NudgeProbePastTarget,
    SetProbeTarget,
    SetArcAngle,
    AssignProbeArc,
    BindProbeAPToArc,
    SetProbeCalibrated,
]


def apply_planning_command(ps: PlanningState, cmd: PlanningCommand) -> List[str]:
    """
    Mutates PlanningState in place.
    Returns a list of probe names that should be re-resolved/re-rendered.
    """
    changed: Set[str] = set()

    if isinstance(cmd, SetArcAngle):
        # clamp to limits (and apply any separation policy if you added it)
        ap = ps.kinematics.set_arc(cmd.arc_id, cmd.ap_deg)
        # any non-calibrated probe bound to this arc is affected
        for name, plan in ps.probes.items():
            if plan.arc_id == cmd.arc_id and plan.bind_ap_to_arc:
                # calibrated probes ignore arc changes
                if not (plan.calibrated and name in ps.calibrations):
                    changed.add(name)
        return sorted(changed)

    if isinstance(cmd, AssignProbeArc):
        plan = ps.probes[cmd.name]
        plan.arc_id = cmd.arc_id
        if cmd.bind_ap_to_arc is not None:
            plan.bind_ap_to_arc = bool(cmd.bind_ap_to_arc)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, BindProbeAPToArc):
        plan = ps.probes[cmd.name]
        if cmd.bind:
            plan.bind_ap_to_arc = True
        else:
            # freeze current effective AP into ap_local so unbinding doesn't jump
            if cmd.freeze_effective_on_unbind:
                eff_ap, _, _ = _resolved_angles(
                    cmd.name, ps
                )  # helper from earlier reply
                plan.ap_local = eff_ap
            plan.bind_ap_to_arc = False
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbeCalibrated):
        plan = ps.probes[cmd.name]
        plan.calibrated = bool(cmd.calibrated)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbeLocalAngles):
        plan = ps.probes[cmd.name]
        if cmd.ap_local is not None:
            plan.ap_local = ps.kinematics.limits.ap_deg.clamp(cmd.ap_local)
        if cmd.ml_local is not None:
            plan.ml_local = ps.kinematics.limits.ml_deg.clamp(cmd.ml_local)
        if cmd.spin is not None:
            plan.spin = ps.kinematics.limits.spin_deg.clamp(cmd.spin)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbeOffsetsRA):
        plan = ps.probes[cmd.name]
        R, A = plan.offsets_RA
        if cmd.R_mm is not None:
            R = float(cmd.R_mm)
        if cmd.A_mm is not None:
            A = float(cmd.A_mm)
        plan.offsets_RA = (R, A)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, NudgeProbeOffsetsRA):
        plan = ps.probes[cmd.name]
        R, A = plan.offsets_RA
        plan.offsets_RA = (R + float(cmd.dR_mm), A + float(cmd.dA_mm))
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbePastTarget):
        plan = ps.probes[cmd.name]
        plan.past_target_mm = float(cmd.past_target_mm)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, NudgeProbePastTarget):
        plan = ps.probes[cmd.name]
        plan.past_target_mm = float(plan.past_target_mm + cmd.d_mm)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbeTarget):
        plan = ps.probes[cmd.name]
        # ensure exactly one is set
        if (cmd.target_key is None) == (cmd.target_point_RAS is None):
            raise ValueError(
                "SetProbeTarget: specify exactly one of target_key or target_point_RAS"
            )
        plan.target_key = cmd.target_key
        plan.target_point_RAS = cmd.target_point_RAS
        changed.add(cmd.name)
        return sorted(changed)

    # Unknown command: no-op
    return sorted(changed)
