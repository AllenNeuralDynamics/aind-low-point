"""One probe's resolved pose, and where its shank tips land.

`ProbePose` is the single numpy pose formula; the optimizer's JAX twin is
parity-tested against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import trimesh
from aind_mri_utils.arc_angles import arc_angles_to_affine
from aind_mri_utils.reticle_calibrations import find_probe_angle
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import fclusterdata

from aind_rutter.domain.plan import (
    CALIBRATION_LOCKED_AXES,
    PlanningState,
    locked_axes_for,
    probe_asset_key,
    resolve_target_LPS,
)
from aind_rutter.domain.scene import NodeInstance, Scene
from aind_rutter.domain.transforms import (
    AffineTransform,
    TransformChain,
    TransformedPoints,
)

if TYPE_CHECKING:
    from aind_rutter.domain.catalog import AssetCatalog


def resolved_angles(name: str, ps: PlanningState) -> tuple[float, float, float]:
    plan = ps.probes[name]
    cal = ps.calibrations.get(name)

    if CALIBRATION_LOCKED_AXES <= locked_axes_for(ps, name):
        ap, ml = find_probe_angle(cal.rotation)  # locked to calibration
    else:
        # AP: from arc if bound, else local; ML: always per-probe local
        ap = (
            ps.kinematics.get_arc(plan.arc_id)
            if (plan.arc_id and plan.bind_ap_to_arc)
            else plan.ap_local
        )
        ml = plan.ml_local
    # clamp to rig limits
    ap, ml, spin = ps.kinematics.clamp_angles(ap, ml, plan.spin)
    return ap, ml, spin


# Run time
@dataclass(slots=True)
class ProbePose:
    # subject anatomical LPS frame: positive ap = mouse pitch down
    # (CW looking into right ML axis), 0 = probe vertical in subject LPS.
    ap: float = 0.0
    # subject anatomical LPS frame: positive ml = mouse roll right
    # (CCW looking into front AP axis), 0 = midline.
    ml: float = 0.0
    # subject anatomical LPS frame: positive spin = mouse yaw right
    # (CW looking into superior DV axis), 0 = sites facing left.
    spin: float = 0.0
    tip: NDArray = field(default_factory=lambda: np.zeros(3))  # LPS

    def transform(self) -> AffineTransform:
        # ``(ap, ml, spin)`` interpreted in subject anatomical LPS:
        # ap=0, ml=0, spin=0 means probe vertical in subject coordinates.
        # The rig's head-tilt offset (``Kinematics.subject_from_rig``)
        # does not enter here — it only constrains which (ap, ml, spin)
        # values are mechanically reachable on the rig.
        R = arc_angles_to_affine(self.ap, self.ml, self.spin)
        t = self.tip
        return AffineTransform(R, t)

    def chain(self) -> TransformChain:
        return TransformChain([self.transform()])

    @classmethod
    def from_planning_state(
        cls,
        ps: PlanningState,
        probe_name: str,
        *,
        assets_targets_fallback: Optional[dict[str, TransformedPoints]] = None,
        catalog: Optional["AssetCatalog"] = None,
    ) -> ProbePose:
        """
        Resolve a live pose from PlanningState (no mutations).
        - AP comes from calibration if plan.calibrated and matrix is present,
        else from arc if bound, else local.
        - ML comes from calibration if present/allowed, else local.
        - Spin is always the per-probe plan spin.
        - Target is taken from planning.target_index (or assets fallback) +
        offsets_LP.

        ``ProbePose.tip`` continues to mean **the world position of the
        position-bearing shank's tip** (= world position of the probe's
        local origin in canonical convention) — used by readouts like
        Tip-RAS, ``_count_overinserted_shanks``, the FCL collision
        adapter, and the renderer.

        The kinematic *pivot* — the world point that lands at
        ``adjusted_target + R @ [0, 0, -past_target_mm]`` — is now the
        **center of the recording array**, not the position shank's
        tip. This means ``past_target_mm = 0`` puts the recording bank
        exactly on the target (rather than the tip on the target, which
        recorded *above* the target). The position shank's tip is
        ``recording_center_local`` mm deeper along the shaft, which is
        what we want.

        Mathematically: ``pose.tip = adjusted_target + R @ [0, 0,
        -past_target_mm] - R @ pivot_local``. The pivot comes from the
        per-asset ``AssetSpec.pivot_LPS`` (canonical-local frame, set
        at runtime build from the actual canonicalized mesh) when a
        ``catalog`` is provided. Without a catalog, falls back to the
        kind-keyed ``recording_center_local_for_kind`` (assumes shank
        layout from ``RECORDING_GEOMETRY``; mostly fine for
        single-shank probes, off-by-row-direction for multi-shank
        without catalog access — pass the catalog).
        """
        plan = ps.probes[probe_name]

        # --- angles (AP/ML) ---
        ap_deg, ml_deg, spin_deg = resolved_angles(probe_name, ps)

        # --- target + offsets, both already LPS ---
        tgt_LPS = resolve_target_LPS(
            plan, ps.target_index, assets_fallback=assets_targets_fallback
        )
        off_LPS = np.array(
            [plan.offsets_LP[0], plan.offsets_LP[1], 0.0], dtype=np.float64
        )
        adjusted_target = tgt_LPS + off_LPS

        # --- pivot lookup ---
        # Pivot is the recording-array centre in the canonical local
        # frame (746764b semantic): ``past_target_mm = 0`` lands the
        # recording bank on target. ``position_bearing_shank`` is a
        # *reporting* setting — it doesn't change the kinematic pivot,
        # only which shank's tip the GUI reports as the RAS readout.
        pivot_local: Optional[np.ndarray] = None
        if catalog is not None:
            asset_key = probe_asset_key(plan.kind)
            spec = catalog.assets.get(asset_key)
            if spec is not None and spec.pivot_LPS is not None:
                pivot_local = np.asarray(spec.pivot_LPS, dtype=np.float64)
        if pivot_local is None:
            from aind_rutter.domain.probe_kinds import recording_center_local_for_kind

            pivot_local = recording_center_local_for_kind(plan.kind)

        # --- tip from depth, orientation, and pivot ---
        # ``(ap, ml, spin)`` are subject-frame angles; pose composition
        # is the legacy ``arc_angles_to_affine`` directly. The rig's
        # head-tilt offset (``ps.kinematics.subject_from_rig``) only
        # affects which subject-frame values are mechanically reachable
        # on the rig (handled in the optimizer's bounds + rig-limit
        # checks), not the meaning of stored angles.
        R_probe = arc_angles_to_affine(ap_deg, ml_deg, spin_deg)
        insertion_vec = R_probe @ np.array(
            [0.0, 0.0, -float(plan.past_target_mm)], dtype=np.float64
        )
        # Subtract R @ pivot_local so the recording-array centre lands
        # at adjusted_target + insertion_vec (and the canonical-local
        # origin = shank-1 ends up at pose.tip).
        tip = adjusted_target + insertion_vec - R_probe @ pivot_local
        tip = ps.kinematics.clamp_xyz(tip)

        return cls(ap=ap_deg, ml=ml_deg, spin=spin_deg, tip=tip)


# run time
@dataclass
class PoseResolver:
    scene: Scene
    plan: PlanningState
    # Optional catalog reference. When provided, ``_probe_chain`` passes
    # it to ``ProbePose.from_planning_state`` so the pose construction
    # picks up each probe asset's ``pivot_LPS`` directly. Strongly
    # recommended for callers (rendering / collisions) — without it
    # multi-shank probes fall back to the kind-keyed approximation.
    catalog: Optional["AssetCatalog"] = None

    # ---- final world transform = base ∘ dynamic ----
    def world_chain_for_node(self, node: "NodeInstance") -> TransformChain:
        base = node.transform
        dyn = self._dynamic_chain_for_node(node)
        return TransformChain.new([*base.elements, *dyn.elements])

    def world_rt_for_node(self, node: "NodeInstance") -> tuple[np.ndarray, np.ndarray]:
        return self.world_chain_for_node(node).composed_transform

    # ---- dynamic pose for a probe (no scene knowledge) ----
    def _probe_chain(self, probe_name: str) -> TransformChain:
        pose = ProbePose.from_planning_state(
            self.plan, probe_name, catalog=self.catalog
        )
        return pose.chain()

    # ---- dynamic transform for a scene node (may be identity) ----
    def _dynamic_chain_for_node(self, node: "NodeInstance") -> TransformChain:
        probe_name: Optional[str] = node.extras.get("pose_source_probe")
        if not probe_name:
            return TransformChain.new([AffineTransform.identity()])

        # The pivot is baked into ``ProbePose.tip`` via ``catalog``, so the
        # dynamic chain needs no ±pivot wrap.
        return self._probe_chain(probe_name)


def detect_shank_tips_local(
    mesh: trimesh.Trimesh,
    *,
    z_tolerance_mm: float = 0.05,
    cluster_radius_mm: float = 0.15,
) -> NDArray[np.float64]:
    """Return shank-tip positions in the probe's local frame as ``(N, 3)``.

    The probe mesh is assumed to be canonicalized so the shaft runs in
    the local ``-z`` direction (tip at ``min-z``, base at ``max-z``) —
    that's the convention enforced by the ``probe-mesh`` canonicalization
    used throughout the codebase. Vertices within ``z_tolerance_mm`` of
    ``min-z`` are taken as candidate tip points; single-link clustering
    over their ``xy`` positions with ``cluster_radius_mm`` as the
    threshold groups within-shank tip neighbours together while keeping
    adjacent shanks (≥ 250 µm apart for NP 2.0 four-shank) separated.

    Parameters
    ----------
    mesh
        Probe mesh in local LPS mm.
    z_tolerance_mm
        Vertices with ``z ≤ min(z) + z_tolerance_mm`` are candidates.
        Default 50 µm — a probe shank tip is sharper than this in
        practice.
    cluster_radius_mm
        Single-link clustering threshold on ``xy`` distance. Default
        150 µm: smaller than the inter-shank pitch (250 µm on NP 2.0
        four-shank) so adjacent shanks stay separate, larger than
        within-shank vertex spread (~70 µm) so each shank groups into
        one cluster.

    Returns
    -------
    Float array shape ``(N, 3)`` of shank-tip centroids in probe local
    mm. Returns a single ``[0, 0, min(z)]`` if the mesh is empty.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) == 0:
        return np.zeros((1, 3), dtype=np.float64)

    z = verts[:, 2]
    z_min = float(z.min())
    bottom = verts[z <= z_min + z_tolerance_mm]
    if len(bottom) == 0:
        return np.array([[0.0, 0.0, z_min]], dtype=np.float64)
    if len(bottom) == 1:
        return bottom.copy()

    labels = fclusterdata(
        bottom[:, :2],
        t=cluster_radius_mm,
        criterion="distance",
        method="single",
        metric="euclidean",
    )
    n_clusters = int(labels.max())
    centroids = np.zeros((n_clusters, 3), dtype=np.float64)
    for cid in range(1, n_clusters + 1):
        centroids[cid - 1] = bottom[labels == cid].mean(axis=0)
    # Sort for stable output (helps tests + visual diff)
    order = np.lexsort(centroids[:, :3].T[::-1])
    return centroids[order]


def named_shank_tip_world(
    pose_tip: ArrayLike,
    rotation: ArrayLike,
    local_tips: ArrayLike,
    shank_number: int,
) -> NDArray[np.float64]:
    """World LPS position of the position-bearing shank's tip.

    ``pose.tip`` is the world position of the probe's canonical local origin,
    which is the first shank. ``shank_number`` is the 1-based position the
    config calls ``position_bearing_shank``; it is clamped to the shanks the
    mesh actually has, and a mesh with no detected tips reads back ``pose_tip``
    unchanged.

    The app's readout and the rig export both need this, and both had their own
    copy — a disagreement here means the number on screen is not the number
    handed to the rig.
    """
    tip = np.asarray(pose_tip, dtype=np.float64)
    tips = np.asarray(local_tips, dtype=np.float64)
    if tips.size == 0:
        return tip
    index = min(max(0, int(shank_number) - 1), tips.shape[0] - 1)
    return tip + np.asarray(rotation, dtype=np.float64) @ tips[index]
