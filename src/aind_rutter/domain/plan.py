"""What the planner holds: a probe plan per probe, and the state around it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple
from warnings import warn

import numpy as np

from aind_rutter.domain.rig import Kinematics
from aind_rutter.domain.scene import Scene
from aind_rutter.domain.transforms import (
    AffineTransform,
    Float3,
    TransformedPoints,
)

if TYPE_CHECKING:
    from aind_rutter.domain.catalog import AssetCatalog


def probe_asset_key(kind: str) -> str:
    """The catalog key holding the mesh for a probe of this kind."""
    return f"probe:{kind}"


def probe_node_id(name: str) -> str:
    """The scene node showing the probe of this name.

    Spelled like an asset key and meaning something else entirely: this one is
    keyed by the probe's name, that one by its kind.
    """
    return f"probe:{name}"


def reconcile_probe_assets(
    scene: Scene, plan: PlanningState, catalog: "AssetCatalog"
) -> list[str]:
    """Point every probe node at the asset for its plan's kind.

    A probe node's ``asset_key`` restates ``ProbePlan.kind``, so a change to the
    kind has to reach both. Reconciling where the geometry is consumed means a
    dispatch is enough, whoever made it — a plan loaded from YAML swaps meshes
    the same way the kind dropdown does.

    Returns the ids of nodes whose asset changed, so a caller holding a handle
    built from the old mesh can drop it. A kind with no asset in the catalog
    leaves its node alone, since pointing it at nothing would fail at render.
    """
    changed: list[str] = []
    for name, probe_plan in plan.probes.items():
        node = scene.nodes.get(probe_node_id(name))
        if node is None:
            continue
        key = probe_asset_key(probe_plan.kind)
        if node.asset_key == key:
            continue
        if key not in catalog.assets:
            warn(f"probe {name!r}: no asset {key!r}; keeping {node.asset_key!r}")
            continue
        node.asset_key = key
        changed.append(node.key)
    return changed


@dataclass(frozen=True, slots=True)
class RegionRef:
    """A further region a probe records, named like its target: one of the two."""

    key: Optional[str] = None
    point_LPS: Optional[Tuple[float, float, float]] = None


# Plan for probe location
@dataclass(slots=True)
class ProbePlan:
    kind: str
    arc_id: Optional[
        str
    ]  # which arc this probe belongs to (None = not bound to any arc)
    # angle sources / bindings
    bind_ap_to_arc: bool = True  # if True and not calibrated → AP comes from arc
    # per-probe local angles (always present so you can edit them; used when
    # not bound / not calibrated)
    ap_local: float = 0.0  # deg
    ml_local: float = 0.0  # deg
    spin: float = 0.0  # deg
    # targeting
    past_target_mm: float = 0.0
    # In-plane offset from the target, LPS mm. The config and the sliders say
    # (R, A); the planning boundary flips both signs on the way in.
    offsets_LP: Tuple[float, float] = (0.0, 0.0)
    target_key: Optional[str] = None
    target_point_LPS: Optional[Tuple[float, float, float]] = None  # ad-hoc fallback
    # Regions scored beside the target. The target still anchors the pose.
    also_record: Tuple[RegionRef, ...] = ()
    # Structure names for the target, then each ``also_record`` entry; empty
    # when the probe is named after its one structure.
    regions: Tuple[str, ...] = ()
    # The shank whose tip is the kinematic pivot (1-indexed). Drives
    # which shank's tip lands at the inline target, which shank's RAS
    # is shown in the readout, and along which shank brain-surface
    # depth is measured. See ``ProbeDeclModel.position_bearing_shank``.
    position_bearing_shank: int = 1
    # calibration policy
    calibrated: bool = (
        False  # if True and calibration exists → AP/ML come from calibration
    )


def kinematic_violations(
    state: "PlanningState",
) -> dict[str, set[tuple[str, ...]]]:
    """Detect probes that violate the rig's angular ranges and pairwise
    separations.

    Returns a dict with four keys:
    ``"arc_ap"`` — set of (arc_a, arc_b) pairs whose AP angles are <
    ``min_arc_ap_separation_deg`` apart, among arcs that carry a probe
    (every probe on either arc is affected).
    ``"within_arc_ml"`` — set of (probe_a, probe_b) pairs on the same arc
    whose effective ML angles are < ``min_within_arc_ml_separation_deg``
    apart.
    ``"ap_range"`` — set of (probe,) whose rig AP (subject AP plus head pitch,
    from its arc when bound) is outside ``limits.ap_deg``.
    ``"ml_range"`` — set of (probe,) whose ML is outside ``limits.ml_deg``.

    Sorted within each tuple so e.g. (a, b) and (b, a) hash the same.
    """
    limits = state.kinematics.limits
    ap_thr = float(limits.min_arc_ap_separation_deg)
    ml_thr = float(limits.min_within_arc_ml_separation_deg)

    by_arc: dict[str, list[str]] = {}
    for name, plan in state.probes.items():
        if plan.arc_id is None:
            continue
        by_arc.setdefault(plan.arc_id, []).append(name)

    # An arc no probe rides has nothing to keep apart, wherever it sits.
    arc_violations: set[tuple[str, ...]] = set()
    arcs = sorted(a for a in state.kinematics.arc_angles if a in by_arc)
    for i, a in enumerate(arcs):
        for b in arcs[i + 1 :]:
            if (
                abs(state.kinematics.arc_angles[a] - state.kinematics.arc_angles[b])
                < ap_thr
            ):
                arc_violations.add(tuple(sorted((a, b))))

    ml_violations: set[tuple[str, ...]] = set()
    for arc_id, members in by_arc.items():
        members.sort()
        for i, a in enumerate(members):
            ml_a = state.probes[a].ml_local
            for b in members[i + 1 :]:
                ml_b = state.probes[b].ml_local
                if abs(ml_a - ml_b) < ml_thr:
                    ml_violations.add(tuple(sorted((a, b))))

    pitch = state.kinematics.head_pitch_deg
    ap_range: set[tuple[str, ...]] = set()
    ml_range: set[tuple[str, ...]] = set()
    for name, plan in state.probes.items():
        bound = plan.arc_id is not None and plan.bind_ap_to_arc
        subject_ap = (
            state.kinematics.arc_angles.get(plan.arc_id, plan.ap_local)
            if bound
            else plan.ap_local
        )
        if not limits.ap_deg.lo <= subject_ap + pitch <= limits.ap_deg.hi:
            ap_range.add((name,))
        if not limits.ml_deg.lo <= plan.ml_local <= limits.ml_deg.hi:
            ml_range.add((name,))

    return {
        "arc_ap": arc_violations,
        "within_arc_ml": ml_violations,
        "ap_range": ap_range,
        "ml_range": ml_range,
    }


@dataclass(slots=True)
class PlanningState:
    kinematics: Kinematics
    probes: dict[str, ProbePlan]
    calibrations: dict[str, AffineTransform] = field(
        default_factory=dict
    )  # probe_name → calibration transform
    target_index: dict[str, Float3] = field(default_factory=dict)


# TODO: make node targets update with node pose
def resolve_target_LPS(
    plan: ProbePlan,
    target_index: Mapping[str, np.ndarray],
    *,
    assets_fallback: Optional[dict[str, TransformedPoints]] = None,
    points_LPS: Optional[np.ndarray] = None,
    strict: bool = False,
) -> np.ndarray:
    """The ``(3,)`` LPS point a probe plan aims at.

    A plan names its target one way or the other, never both: the config model's
    target is a discriminated union, and ``SetProbeTarget`` refuses a command
    that sets neither or both. Setting both is therefore a corrupted state
    rather than a case to resolve, and it raises instead of picking a winner —
    the app and the optimizer once picked different ones.

    ``points_LPS`` overrides the plan with an already-resolved cloud, which is
    how a per-probe target point set is passed in. ``strict`` decides what an
    unresolvable target costs: the optimizer must not quietly aim at the origin,
    while the app has to keep drawing.
    """
    if points_LPS is not None:
        return np.asarray(points_LPS, dtype=np.float64).reshape(-1, 3).mean(0)

    if plan.target_key and plan.target_point_LPS is not None:
        raise ValueError(
            f"probe plan names both target_key {plan.target_key!r} and an inline "
            f"point {plan.target_point_LPS!r}; exactly one is allowed"
        )

    if plan.target_key:
        pts = target_index.get(plan.target_key)
        if pts is None and assets_fallback is not None:
            tp = assets_fallback.get(plan.target_key)
            if tp is not None:
                pts = tp.raw  # already in LPS if your assets pipeline canonicalized it
        if pts is not None:
            pts = np.asarray(pts, dtype=np.float64)
            return pts if pts.ndim == 1 else pts.reshape(-1, 3).mean(0)
        if strict:
            raise RuntimeError(f"No target in the index for key {plan.target_key!r}")
        warn(f"Missing target for key: {plan.target_key!r}; using origin.")
        return np.zeros(3, dtype=np.float64)

    if plan.target_point_LPS is not None:
        return np.asarray(plan.target_point_LPS, dtype=np.float64)

    if strict:
        raise RuntimeError(
            "Probe plan has no target_key or target_point_LPS; "
            "a runtime target point is required."
        )
    warn("ProbePlan has neither target_key nor target_point_LPS; using origin.")
    return np.zeros(3, dtype=np.float64)


def resolve_region_points_LPS(
    plan: ProbePlan,
    target_index: Mapping[str, np.ndarray],
    *,
    strict: bool = True,
) -> np.ndarray | None:
    """``(K, 3)`` LPS: the target's point, then each of ``also_record``'s.

    ``None`` when the probe records only its target.
    """
    if not plan.also_record:
        return None
    points = [resolve_target_LPS(plan, target_index, strict=strict)]
    for ref in plan.also_record:
        one = ProbePlan(
            kind=plan.kind,
            arc_id=None,
            target_key=ref.key,
            target_point_LPS=ref.point_LPS,
        )
        points.append(resolve_target_LPS(one, target_index, strict=strict))
    return np.stack(points)


CALIBRATION_LOCKED_AXES = frozenset({"ap_tilt", "ml_tilt"})


def locked_axes_for(ps: "PlanningState", name: str) -> frozenset[str]:
    """Pose axes this probe's own controls cannot change.

    A calibrated probe whose calibration is loaded takes both tilts from the
    measured rotation, so an edit to either would be resolved away. Declaring
    ``calibrated`` without a calibration present locks nothing — the tilts fall
    back to the arc and the local angle.

    Being bound to an arc is not a lock: the AP control still works, it moves
    the whole arc.
    """
    plan = ps.probes.get(name)
    if plan is None:
        return frozenset()
    if plan.calibrated and ps.calibrations.get(name) is not None:
        return CALIBRATION_LOCKED_AXES
    return frozenset()
