"""What the planner holds: a probe plan per probe, and the state around it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple
from warnings import warn

import numpy as np
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system

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
    offsets_RA: Tuple[float, float] = (0.0, 0.0)
    target_key: Optional[str] = None
    target_point_RAS: Optional[Tuple[float, float, float]] = None  # ad-hoc fallback
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
    """Detect probes that violate the rig's pairwise angular separation
    requirements.

    Returns a dict with two keys:
    ``"arc_ap"`` — set of (arc_a, arc_b) pairs whose AP angles are <
    ``min_arc_ap_separation_deg`` apart (every probe on either arc is
    affected).
    ``"within_arc_ml"`` — set of (probe_a, probe_b) pairs on the same arc
    whose effective ML angles are < ``min_within_arc_ml_separation_deg``
    apart.

    Sorted within each tuple so e.g. (a, b) and (b, a) hash the same.
    """
    limits = state.kinematics.limits
    ap_thr = float(limits.min_arc_ap_separation_deg)
    ml_thr = float(limits.min_within_arc_ml_separation_deg)

    arc_violations: set[tuple[str, ...]] = set()
    arcs = sorted(state.kinematics.arc_angles.keys())
    for i, a in enumerate(arcs):
        for b in arcs[i + 1 :]:
            if (
                abs(state.kinematics.arc_angles[a] - state.kinematics.arc_angles[b])
                < ap_thr
            ):
                arc_violations.add(tuple(sorted((a, b))))

    by_arc: dict[str, list[str]] = {}
    for name, plan in state.probes.items():
        if plan.arc_id is None:
            continue
        by_arc.setdefault(plan.arc_id, []).append(name)

    ml_violations: set[tuple[str, ...]] = set()
    for arc_id, members in by_arc.items():
        members.sort()
        for i, a in enumerate(members):
            ml_a = state.probes[a].ml_local
            for b in members[i + 1 :]:
                ml_b = state.probes[b].ml_local
                if abs(ml_a - ml_b) < ml_thr:
                    ml_violations.add(tuple(sorted((a, b))))

    return {"arc_ap": arc_violations, "within_arc_ml": ml_violations}


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

    if plan.target_key and plan.target_point_RAS is not None:
        raise ValueError(
            f"probe plan names both target_key {plan.target_key!r} and an inline "
            f"point {plan.target_point_RAS!r}; exactly one is allowed"
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

    if plan.target_point_RAS is not None:
        ras = np.asarray(plan.target_point_RAS, dtype=np.float64)
        return convert_coordinate_system(ras, "RAS", "LPS")

    if strict:
        raise RuntimeError(
            "Probe plan has no target_key or target_point_RAS; "
            "a runtime target point is required."
        )
    warn("ProbePlan has neither target_key nor target_point_RAS; using origin.")
    return np.zeros(3, dtype=np.float64)


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
