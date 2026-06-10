"""Plan export and round-trip: planning_state_to_plan_model, save_plan_to_config,
export_plan_geometry, and the _depth_along_probe_axis helper."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aind_low_point.scene import Scene
    from aind_low_point.state_change import PlanStore

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
            position_bearing_shank=plan.position_bearing_shank,
            calibrated=plan.calibrated,
            auto_scene=orig_decl.auto_scene if orig_decl else True,
            scene_tags=orig_decl.scene_tags if orig_decl else ["probe", "dynamic"],
        )

    return PlanningModel(
        arcs=dict(state.kinematics.arc_angles),
        subject_from_rig=original.subject_from_rig,
        probes=probes,
        reticles=original.reticles,
        calibrations=original.calibrations,
    )


def reorder_plan_for_rig(state: PlanningState) -> None:
    """In-place rig-readability ordering of a PlanningState (cosmetic; pose
    semantics unchanged).

    - **Arcs relabelled by AP**: ``a`` is the most-positive-AP arc, ``b`` next,
      etc. (both ``kinematics.arc_angles`` keys and each probe's ``arc_id``).
    - **Probes ordered** by arc ascending (``a`` first) then ML descending
      (positive ML first) — the order an experimenter reads them off at the rig.

    Mutates the underlying dicts in place (clear+update) so it works regardless
    of dataclass field mutability.
    """
    arc_angles = dict(state.kinematics.arc_angles)
    order = sorted(arc_angles, key=lambda k: -arc_angles[k])  # most +AP first
    remap = {old: chr(ord("a") + i) for i, old in enumerate(order)}
    state.kinematics.arc_angles.clear()
    for old in order:
        state.kinematics.arc_angles[remap[old]] = arc_angles[old]
    for plan in state.probes.values():
        if plan.arc_id in remap:
            plan.arc_id = remap[plan.arc_id]
    ordered = dict(
        sorted(
            state.probes.items(),
            key=lambda kv: (kv[1].arc_id or "z", -float(kv[1].ml_local or 0.0)),
        )
    )
    state.probes.clear()
    state.probes.update(ordered)


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
    scene: Optional["Scene"] = None,
) -> dict[str, Any]:
    """Produce the minimal geometric summary needed to execute a plan.

    Unlike ``save_plan_to_config`` (which round-trips the entire config),
    this returns just the per-probe placement information an
    experimenter cares about: probe type, target identity and RAS
    coordinate, resolved angles, offsets, depth past target, the final
    tip position in RAS, and (when a brain mesh asset is available) the
    depth of the tip below the brain surface measured along the probe
    axis.

    Pass ``scene`` when the brain asset has a scene-node-level transform
    (e.g. ``transform: headframe_to_lps``); the brain mesh will be
    resolved through that transform so the depth ray cast happens in
    world LPS. Without a scene the catalog's raw mesh is used, which is
    correct only when the brain is authored directly in LPS.

    The dict is yaml-serialisable. Intended for ``yaml.safe_dump``.
    """
    # Rig-readability ordering: arcs relabelled by AP (a = most +AP), probes
    # sorted by arc then ML-descending. Cosmetic; does not change poses.
    reorder_plan_for_rig(plan_state)
    brain_mesh = None
    if scene is not None:
        from aind_low_point.scene import resolve_base_geometry

        # Find the scene node carrying this brain asset.
        for k, n in scene.nodes.items():
            if n.asset_key == brain_asset_key:
                wrap = resolve_base_geometry(catalog, scene, k)
                if wrap is not None:
                    brain_mesh = wrap.raw
                break
    if brain_mesh is None:
        brain_spec = catalog.assets.get(brain_asset_key)
        if brain_spec is not None and brain_spec.mesh is not None:
            brain_mesh = brain_spec.mesh.raw

    from aind_low_point.runtime.shanks import detect_shank_tips_local

    # Head-tilt offset between subject-anatomical AP and rig-mechanical AP.
    # See ``_head_pitch_about_L_deg`` in optimization/optimize.py for the
    # same derivation. Pulled from kinematics so per-mouse head pitch
    # propagates into the rig-frame angle readout below.
    R_sfr, _ = plan_state.kinematics.subject_from_rig.rotate_translate
    R_sfr_arr = np.asarray(R_sfr, dtype=np.float64)
    head_pitch_about_L = float(np.rad2deg(np.arctan2(R_sfr_arr[2, 1], R_sfr_arr[1, 1])))

    probes_out: dict[str, Any] = {}
    for name, plan in plan_state.probes.items():
        pose = ProbePose.from_planning_state(plan_state, name, catalog=catalog)
        R = arc_angles_to_affine(pose.ap, pose.ml, pose.spin)
        # Tip readout = world position of the position-bearing shank's
        # tip, not necessarily shank-1. ``pose.tip`` is shank-1 (the
        # canonical local origin); shift to the named shank.
        asset_key = f"probe:{plan.kind}"
        spec = catalog.assets.get(asset_key)
        local_tips = (
            detect_shank_tips_local(spec.mesh.raw)
            if spec is not None and spec.mesh is not None
            else np.zeros((0, 3), dtype=np.float64)
        )
        named_idx = max(0, int(plan.position_bearing_shank) - 1)
        if local_tips.shape[0] > 0:
            named_idx = min(named_idx, local_tips.shape[0] - 1)
            named_local = np.asarray(local_tips[named_idx], dtype=np.float64)
        else:
            named_local = np.zeros(3, dtype=np.float64)
        tip_lps = np.asarray(pose.tip, dtype=np.float64) + R @ named_local
        tip_ras = convert_coordinate_system(tip_lps, "LPS", "RAS")

        target_ras = None
        if plan.target_key is not None and plan.target_key in plan_state.target_index:
            tlps = np.asarray(
                plan_state.target_index[plan.target_key], dtype=np.float64
            )
            tlps = tlps.flatten() if tlps.ndim > 1 else tlps
            target_ras = convert_coordinate_system(tlps[:3], "LPS", "RAS").tolist()
        elif plan.target_point_RAS is not None:
            target_ras = list(plan.target_point_RAS)

        depth = None
        if brain_mesh is not None:
            probe_axis = R @ np.array([0.0, 0.0, 1.0])
            # Depth measured from the named shank's tip along the shaft.
            depth = _depth_along_probe_axis(tip_lps, probe_axis, brain_mesh)

        # Subject-anatomical angles: (ap, ml, spin) as stored on the plan.
        # ap=0, ml=0, spin=0 means the probe is vertical in subject LPS.
        #
        # Rig-mechanical angles: what an experimenter dials into the rig.
        # The mouse head is mounted nose-DOWN by the head pitch, so a
        # subject-vertical probe (subject_ap = 0, lambda-bregma) requires a
        # rig angle of +head_pitch: ``rig_ap = subject_ap + head_pitch_about_L``
        # (aind-mri-utils convention; see dev memory rig_ap_sign_convention).
        # ML/spin are unaffected by a pure-L-axis head pitch.
        ap_rig = float(pose.ap) + head_pitch_about_L
        probes_out[name] = {
            "kind": plan.kind,
            "target": {
                "key": plan.target_key,
                "position_RAS_mm": target_ras,
            },
            "arc": {"id": plan.arc_id} if plan.arc_id else None,
            # Rig-mechanical angles first — these are what's dialed at the rig.
            "angles_rig_deg": {
                "ap": ap_rig,
                "ml": float(pose.ml),
                "spin": float(pose.spin),
            },
            "angles_subject_deg": {
                "ap": float(pose.ap),
                "ml": float(pose.ml),
                "spin": float(pose.spin),
            },
            "offsets_RA_mm": [float(plan.offsets_RA[0]), float(plan.offsets_RA[1])],
            "past_target_mm": float(plan.past_target_mm),
            "tip_RAS_mm": [float(c) for c in tip_ras],
            "depth_from_brain_surface_mm": depth,
        }

    # v2: per-probe angles split into ``angles_subject_deg`` (anatomical,
    # ap=0=probe vertical in subject) and ``angles_rig_deg`` (mechanical,
    # the dial values an experimenter would set on the rig — differs
    # from subject AP by the head pitch). v1 had a single ``angles_deg``
    # block that conflated the two.
    return {
        "plan_export_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_config": source_config,
        "head_pitch_about_L_deg": head_pitch_about_L,
        "arc_angles_subject_deg": dict(plan_state.kinematics.arc_angles),
        "arc_angles_rig_deg": {
            arc_id: float(ap) - head_pitch_about_L
            for arc_id, ap in plan_state.kinematics.arc_angles.items()
        },
        "probes": probes_out,
    }


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
    from aind_low_point.commands import (
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
