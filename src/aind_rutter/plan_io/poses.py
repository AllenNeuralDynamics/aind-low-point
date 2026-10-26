"""Reading a rig-poses export back over the plan it was exported from."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aind_rutter.config import (
    CatalogTargetRefModel,
    InlineTargetRefModel,
    PlanningModel,
)

POSES_EXPORT_VERSION = 3


def is_poses_export(raw: Any) -> bool:
    return isinstance(raw, Mapping) and "plan_export_version" in raw


def plan_from_poses(plan: PlanningModel, poses: Mapping[str, Any]) -> PlanningModel:
    """``plan`` with each exported probe's pose, as ``export_plan_geometry``
    wrote it, laid over the probe of the same name.

    The export holds the angles in use, so a probe it read through its
    calibration (one with ``newscale_tip_mm``) keeps the plan's AP and ML:
    those came from the calibration, and writing them as planned angles
    would change the plan the calibration departs from. Spin is never
    calibrated and is always restored. Arcs come back in subject angles.
    """
    version = poses.get("plan_export_version")
    if version != POSES_EXPORT_VERSION:
        raise ValueError(
            f"poses export version {version!r}; this reads {POSES_EXPORT_VERSION}"
        )
    unknown = sorted(set(poses["probes"]) - set(plan.probes))
    if unknown:
        raise ValueError(
            f"poses export names probes the plan does not have: {', '.join(unknown)}"
        )
    pitch = float(poses["head_pitch_about_L_deg"])
    arcs = dict(plan.arcs)
    arcs.update(
        {str(k): float(v) - pitch for k, v in poses["arc_angles_rig_deg"].items()}
    )

    probes = dict(plan.probes)
    for name, pose in poses["probes"].items():
        decl = plan.probes[name]
        angles = pose["angles_rig_deg"]
        update: dict[str, Any] = {
            "kind": pose["kind"],
            "arc": (pose.get("arc") or {}).get("id", decl.arc),
            "spin": float(angles["spin"]),
            "offsets_RA": [float(v) for v in pose["offsets_RA_mm"]],
            "past_target_mm": float(pose["past_target_mm"]),
            "position_bearing_shank": int(pose["position_bearing_shank"]),
            "target": _target(decl.target, pose["target"]),
        }
        if "newscale_tip_mm" not in pose:
            update["slider_ml"] = float(angles["ml"])
            if not decl.bind_ap_to_arc:
                update["ap_local"] = float(angles["ap"]) - pitch
        probes[name] = decl.model_copy(update=update)
    return plan.model_copy(update={"arcs": arcs, "probes": probes})


def _target(planned: Any, exported: Mapping[str, Any]) -> Any:
    key = exported.get("key")
    if key is None:
        return InlineTargetRefModel(
            point_RAS=[float(v) for v in exported["position_RAS_mm"]]
        )
    # Same key keeps the plan's reference, whose kind (catalog or node) the
    # export does not record.
    if getattr(planned, "key", None) == key:
        return planned
    return CatalogTargetRefModel(key=str(key))
