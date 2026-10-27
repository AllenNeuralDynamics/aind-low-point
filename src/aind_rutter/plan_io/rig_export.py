"""The rig-facing per-probe summary, and the arc ordering it is read in."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Optional

import numpy as np
import trimesh
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system

from aind_rutter.domain.catalog import AssetCatalog
from aind_rutter.domain.plan import PlanningState
from aind_rutter.domain.pose import ProbePose
from aind_rutter.domain.rig import probe_rotation
from aind_rutter.domain.scene import Scene


def reorder_plan_for_rig(
    state: PlanningState, *, relabel_arcs: bool = True
) -> PlanningState:
    """Return a copy of ``state`` ordered for rig readability.

    - **Arcs in AP order**, most positive first, and with ``relabel_arcs``
      renamed to match: ``a`` is the most-positive-AP arc, ``b`` next, etc.
      (both ``kinematics.arc_angles`` keys and each probe's ``arc_id``). The
      emitter relabels, so a solved plan loads with AP-ordered letters; the rig
      export does not, so it always reads with the letters the UI shows. An arc
      no probe is on is left out, since nothing at the rig is set by it.
    - **Probes ordered** by arc in that order, then ML descending (positive ML
      first) — the order an experimenter reads them off at the rig.

    Cosmetic: pose semantics are unchanged, only the labels and the order. The
    input is left alone, because the relabelling is a property of the written
    document rather than of the session that produced it — renaming a live
    session's arcs would change what the next save writes, and would do it
    without a dispatch, so nothing watching the plan would hear about it.
    """
    in_use = {plan.arc_id for plan in state.probes.values()}
    arc_angles = {k: v for k, v in state.kinematics.arc_angles.items() if k in in_use}
    order = sorted(arc_angles, key=lambda k: -arc_angles[k])  # most +AP first
    remap = {
        old: chr(ord("a") + i) if relabel_arcs else old for i, old in enumerate(order)
    }
    rank = {remap[old]: i for i, old in enumerate(order)}
    probes = {
        name: replace(plan, arc_id=remap.get(plan.arc_id, plan.arc_id))
        for name, plan in state.probes.items()
    }
    return replace(
        state,
        kinematics=replace(
            state.kinematics,
            arc_angles={remap[old]: arc_angles[old] for old in order},
        ),
        probes=dict(
            sorted(
                probes.items(),
                key=lambda kv: (
                    rank.get(kv[1].arc_id, len(rank)),
                    -float(kv[1].ml_local or 0.0),
                ),
            )
        ),
    )


def depth_along_probe_axis(
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
    axis. Angles are rig angles only, the values dialed at the rig. A probe
    that is calibrated, with its calibration loaded, also carries the
    NewScale position its tip should read.

    Pass ``scene`` when the brain asset has a scene-node-level transform
    (e.g. ``transform: headframe_to_lps``); the brain mesh will be
    resolved through that transform so the depth ray cast happens in
    world LPS. Without a scene the catalog's raw mesh is used, which is
    correct only when the brain is authored directly in LPS.

    The dict is yaml-serialisable. Intended for ``yaml.safe_dump``.
    """
    # Arcs keep the session's letters so the sheet matches the UI; probes are
    # sorted for the rig. Leaves the caller's state, maybe a live session, alone.
    plan_state = reorder_plan_for_rig(plan_state, relabel_arcs=False)
    from aind_rutter.build.calibration import lps_to_newscale
    from aind_rutter.build.queries import brain_world_mesh
    from aind_rutter.domain.pose import (
        detect_shank_tips_local,
        dialed_angles,
        named_shank_tip_world,
    )

    # A catalog with no scene means the brain is authored directly in LPS.
    brain_mesh = brain_world_mesh(catalog, scene, brain_asset_key, fallback_to_raw=True)

    # Head-tilt offset between subject-anatomical AP and rig-mechanical AP, from
    # the extractor the optimizer bounds use, so the rig-frame readout can't
    # drift from them.
    head_pitch_about_L = plan_state.kinematics.head_pitch_deg

    probes_out: dict[str, Any] = {}
    for name, plan in plan_state.probes.items():
        pose = ProbePose.from_planning_state(plan_state, name, catalog=catalog)
        R = probe_rotation(pose.ap, pose.ml, pose.spin)
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
        tip_lps = named_shank_tip_world(
            pose.tip, R, local_tips, plan.position_bearing_shank
        )
        tip_ras = convert_coordinate_system(tip_lps, "LPS", "RAS")

        target_ras = None
        if plan.target_key is not None and plan.target_key in plan_state.target_index:
            tlps = np.asarray(
                plan_state.target_index[plan.target_key], dtype=np.float64
            )
            tlps = tlps.flatten() if tlps.ndim > 1 else tlps
            target_ras = convert_coordinate_system(tlps[:3], "LPS", "RAS").tolist()
        elif plan.target_point_LPS is not None:
            target_ras = [
                float(v)
                for v in convert_coordinate_system(
                    np.asarray(plan.target_point_LPS, float).reshape(1, 3),
                    "LPS",
                    "RAS",
                ).reshape(3)
            ]

        depth = None
        if brain_mesh is not None:
            probe_axis = R @ np.array([0.0, 0.0, 1.0])
            # Depth measured from the named shank's tip along the shaft.
            depth = depth_along_probe_axis(tip_lps, probe_axis, brain_mesh)

        # The plan stores subject angles; the rig is dialed in rig angles. The
        # head is mounted nose-down by the head pitch, so
        # ``rig_ap = subject_ap + head_pitch_about_L``; ML and spin are
        # unaffected by a pitch about L. The rig is set to the dial, not to
        # where the manipulator's bias then carries the probe.
        dial_ap, dial_ml, dial_spin = dialed_angles(name, plan_state)
        ap_rig = float(dial_ap) + head_pitch_about_L
        entry: dict[str, Any] = {
            "kind": plan.kind,
            "target": {
                "key": plan.target_key,
                "position_RAS_mm": target_ras,
            },
            "arc": {"id": plan.arc_id} if plan.arc_id else None,
            "angles_rig_deg": {
                "ap": ap_rig,
                "ml": float(dial_ml),
                "spin": float(dial_spin),
            },
            "offsets_RA_mm": [-float(plan.offsets_LP[0]), -float(plan.offsets_LP[1])],
            "past_target_mm": float(plan.past_target_mm),
            "position_bearing_shank": int(plan.position_bearing_shank),
            "tip_RAS_mm": [float(c) for c in tip_ras],
            "depth_from_brain_surface_mm": depth,
        }
        if plan.ap_bias or plan.ml_bias:
            entry["bias_deg"] = {"ap": float(plan.ap_bias), "ml": float(plan.ml_bias)}
        cal = plan_state.calibrations.get(name)
        if plan.calibrated and cal is not None:
            entry["newscale_tip_mm"] = [float(c) for c in lps_to_newscale(tip_lps, cal)]
        probes_out[name] = entry

    # Rig angles only: a document carrying both frames was misread at the rig.
    return {
        "plan_export_version": 3,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_config": source_config,
        "head_pitch_about_L_deg": head_pitch_about_L,
        # Same sign as each probe's ``angles_rig_deg``.
        "arc_angles_rig_deg": {
            arc_id: float(ap) + head_pitch_about_L
            for arc_id, ap in plan_state.kinematics.arc_angles.items()
        },
        "probes": probes_out,
    }
