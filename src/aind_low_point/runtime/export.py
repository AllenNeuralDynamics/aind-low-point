"""Plan export and round-trip: planning_state_to_plan_model, save_plan_to_config,
export_plan_geometry, and the _depth_along_probe_axis helper."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np
import trimesh
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.arc_angles import arc_angles_to_affine

from aind_low_point.assets import AssetCatalog
from aind_low_point.config import (
    CatalogTargetRefModel,
    ConfigModel,
    InlineTargetRefModel,
    NodeTargetRefModel,
    PlanningModel,
    ProbeDeclModel,
)
from aind_low_point.planning import PlanningState, ProbePlan, ProbePose


def _reconstruct_target_ref(
    probe: ProbePlan,
    original_probes: dict[str, ProbeDeclModel],
    probe_name: str,
) -> "CatalogTargetRefModel | NodeTargetRefModel | InlineTargetRefModel":
    """Reconstruct a TargetRef from a ProbePlan.

    Priority:
    1. If target_point_RAS is set → InlineTargetRefModel
    2. If target_key matches the original → reuse original TargetRef (preserves kind)
    3. Otherwise → CatalogTargetRefModel
    """
    if probe.target_point_RAS is not None:
        return InlineTargetRefModel(point_RAS=list(probe.target_point_RAS))
    orig = original_probes.get(probe_name)
    if (
        orig is not None
        and hasattr(orig.target, "key")
        and probe.target_key == orig.target.key
    ):
        return orig.target
    if probe.target_key is None:
        return CatalogTargetRefModel(key="")
    return CatalogTargetRefModel(key=probe.target_key)


def planning_state_to_plan_model(
    state: PlanningState,
    original: PlanningModel,
) -> PlanningModel:
    """Convert a mutated PlanningState back to a PlanningModel.

    Parameters
    ----------
    state
        The runtime planning state (possibly mutated by commands).
    original
        The original PlanningModel from the config (used to preserve
        calibrations, reticles, and target ref kinds).

    Returns
    -------
    PlanningModel
        A new PlanningModel reflecting the current state.
    """
    probes: dict[str, ProbeDeclModel] = {}
    for name, plan in state.probes.items():
        target_ref = _reconstruct_target_ref(plan, original.probes, name)
        orig_decl = original.probes.get(name)
        probes[name] = ProbeDeclModel(
            kind=plan.kind,
            arc=plan.arc_id,
            slider_ml=plan.ml_local,
            spin=plan.spin,
            ap_local=plan.ap_local,
            bind_ap_to_arc=plan.bind_ap_to_arc,
            target=target_ref,
            past_target_mm=plan.past_target_mm,
            offsets_RA=list(plan.offsets_RA),
            calibrated=plan.calibrated,
            auto_scene=orig_decl.auto_scene if orig_decl else True,
            scene_tags=orig_decl.scene_tags if orig_decl else ["probe", "dynamic"],
        )

    return PlanningModel(
        arcs=dict(state.kinematics.arc_angles),
        probes=probes,
        reticles=original.reticles,
        calibrations=original.calibrations,
    )


def _depth_along_probe_axis(
    tip_lps: np.ndarray,
    probe_axis_world: np.ndarray,
    brain_mesh: trimesh.Trimesh,
) -> Optional[float]:
    """Distance from *tip_lps* to the nearest brain-surface intersection
    along ``+probe_axis_world`` (i.e. tip → base direction). Returns
    None if no intersection is found in that half-space (probe likely
    not yet inside the brain)."""
    try:
        locs, _, _ = brain_mesh.ray.intersects_location(
            ray_origins=tip_lps[None, :],
            ray_directions=probe_axis_world[None, :],
        )
    except Exception:
        return None
    if len(locs) == 0:
        return None
    dists = np.linalg.norm(locs - tip_lps, axis=1)
    return float(dists.min())


def export_plan_geometry(
    plan_state: PlanningState,
    catalog: "AssetCatalog",
    *,
    brain_asset_key: str = "brain",
    source_config: Optional[str] = None,
) -> dict[str, Any]:
    """Produce the minimal geometric summary needed to execute a plan.

    Unlike ``save_plan_to_config`` (which round-trips the entire config),
    this returns just the per-probe placement information an
    experimenter cares about: probe type, target identity and RAS
    coordinate, resolved angles, offsets, depth past target, the final
    tip position in RAS, and (when a brain mesh asset is available) the
    depth of the tip below the brain surface measured along the probe
    axis.

    The dict is yaml-serialisable. Intended for ``yaml.safe_dump``.
    """
    brain_mesh = None
    brain_spec = catalog.assets.get(brain_asset_key)
    if brain_spec is not None and brain_spec.mesh is not None:
        brain_mesh = brain_spec.mesh.raw

    probes_out: dict[str, Any] = {}
    for name, plan in plan_state.probes.items():
        pose = ProbePose.from_planning_state(plan_state, name, catalog=catalog)
        tip_lps = np.asarray(pose.tip, dtype=np.float64)
        tip_ras = convert_coordinate_system(tip_lps, "LPS", "RAS")

        target_ras = None
        if plan.target_key is not None and plan.target_key in plan_state.target_index:
            tlps = np.asarray(plan_state.target_index[plan.target_key], dtype=np.float64)
            tlps = tlps.flatten() if tlps.ndim > 1 else tlps
            target_ras = convert_coordinate_system(tlps[:3], "LPS", "RAS").tolist()
        elif plan.target_point_RAS is not None:
            target_ras = list(plan.target_point_RAS)

        depth = None
        if brain_mesh is not None:
            R = arc_angles_to_affine(pose.ap, pose.ml, pose.spin)
            probe_axis = R @ np.array([0.0, 0.0, 1.0])
            depth = _depth_along_probe_axis(tip_lps, probe_axis, brain_mesh)

        probes_out[name] = {
            "kind": plan.kind,
            "target": {
                "key": plan.target_key,
                "position_RAS_mm": target_ras,
            },
            "arc": {"id": plan.arc_id} if plan.arc_id else None,
            "angles_deg": {
                "ap": float(pose.ap),
                "ml": float(pose.ml),
                "spin": float(pose.spin),
            },
            "offsets_RA_mm": [float(plan.offsets_RA[0]), float(plan.offsets_RA[1])],
            "past_target_mm": float(plan.past_target_mm),
            "tip_RAS_mm": [float(c) for c in tip_ras],
            "depth_from_brain_surface_mm": depth,
        }

    return {
        "plan_export_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_config": source_config,
        "arc_angles_deg": dict(plan_state.kinematics.arc_angles),
        "probes": probes_out,
    }


def save_plan_to_config(
    state: PlanningState,
    original_config: ConfigModel,
) -> ConfigModel:
    """Produce a new ConfigModel with the plan section updated from state.

    Everything except ``plan`` is preserved from the original config.
    The returned model can be serialized to YAML via
    ``model.model_dump(mode="json")``.

    Parameters
    ----------
    state
        The runtime planning state.
    original_config
        The original ConfigModel (used as the base for non-plan sections).

    Returns
    -------
    ConfigModel
        A new ConfigModel ready for serialization.
    """
    new_plan = planning_state_to_plan_model(state, original_config.plan)
    data = original_config.model_dump(mode="json")
    data["plan"] = new_plan.model_dump(mode="json")

    # Strip auto-generated scene nodes so the validator can re-generate
    # them for the (possibly changed) set of probes / assets.
    explicit_keys = original_config.scene._explicit_node_keys
    if explicit_keys is not None:
        data["scene"]["nodes"] = [
            n for n in data["scene"]["nodes"] if n["key"] in explicit_keys
        ]

    return ConfigModel.model_validate(data)
