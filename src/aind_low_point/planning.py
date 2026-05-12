"""Planning state - the domain logic"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    Optional,
    Set,
    TYPE_CHECKING,
    Tuple,
)
from warnings import warn

import numpy as np
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.arc_angles import arc_angles_to_affine
from aind_mri_utils.reticle_calibrations import (
    find_probe_angle,
)
from numpy.typing import NDArray

from aind_low_point.core import (
    AffineTransform,
    Float3,
    TransformChain,
    TransformedPoints,
)
from aind_low_point.scene import NodeInstance, Scene

if TYPE_CHECKING:
    from aind_low_point.assets import AssetCatalog


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


@dataclass(frozen=True, slots=True)
class JointRange:
    lo: float
    hi: float

    def clamp(self, v: float) -> float:
        return float(min(max(v, self.lo), self.hi))


@dataclass(frozen=True, slots=True)
class PoseLimits:
    # angular limits (deg)
    ap_deg: JointRange = field(default_factory=lambda: JointRange(-60.0, 60.0))
    ml_deg: JointRange = field(default_factory=lambda: JointRange(-60.0, 60.0))
    spin_deg: JointRange = field(default_factory=lambda: JointRange(-180.0, 180.0))
    # translational work envelope (mm); set to None if unbounded
    x_mm: Optional[JointRange] = None
    y_mm: Optional[JointRange] = None
    z_mm: Optional[JointRange] = None
    # Hardware angular-exclusion: AP between any two arcs and ML between
    # any two probes on the same arc must each differ by at least this
    # many degrees, otherwise the manipulators / arc structure can't be
    # set up that way physically. Default matches the AIND rig (16°).
    min_arc_ap_separation_deg: float = 16.0
    min_within_arc_ml_separation_deg: float = 16.0

    def clamp_angles(
        self, ap: float, ml: float, spin: float
    ) -> Tuple[float, float, float]:
        return (
            self.ap_deg.clamp(ap),
            self.ml_deg.clamp(ml),
            self.spin_deg.clamp(spin),
        )

    def clamp_xyz(self, tip_lps: np.ndarray) -> np.ndarray:
        t = np.asarray(tip_lps, dtype=np.float64).copy()
        if self.x_mm:
            t[0] = self.x_mm.clamp(t[0])
        if self.y_mm:
            t[1] = self.y_mm.clamp(t[1])
        if self.z_mm:
            t[2] = self.z_mm.clamp(t[2])
        return t


# --- Kinematics model (no knowledge of probes) ------------------------------


@dataclass(slots=True)
class Kinematics:
    """
    Rig-wide kinematics parameters.
    - arc_angles: shared AP tilt per arc id (deg)
    - limits: mechanical/operational joint limits
    - coupled_axes: which DOFs are shared by all probes on the same arc
      (names match ProbePose fields: ap_deg, ml_deg, spin_deg, x_mm, y_mm, z_mm)
    """

    arc_angles: dict[str, float] = field(
        default_factory=dict
    )  # e.g., {"a": 12.0, "b": -8.0}
    limits: PoseLimits = field(default_factory=PoseLimits)
    coupled_axes: Set[str] = field(
        default_factory=lambda: {"ap_deg"}
    )  # today: AP tilt is arc-coupled

    # convenience helpers
    def get_arc(self, arc_id: str) -> float:
        return float(self.arc_angles[arc_id])

    def set_arc(self, arc_id: str, ap_deg: float) -> float:
        """Clamp and store AP for an arc; return the value actually stored."""
        clamped = self.limits.ap_deg.clamp(ap_deg)
        self.arc_angles[arc_id] = clamped
        return clamped

    def clamp_angles(
        self, ap: float, ml: float, spin: float
    ) -> Tuple[float, float, float]:
        return self.limits.clamp_angles(ap, ml, spin)

    def clamp_xyz(self, tip_lps: np.ndarray) -> np.ndarray:
        return self.limits.clamp_xyz(tip_lps)

    def is_axis_coupled(self, axis_name: str) -> bool:
        """UI can call this to gray controls; mechanics layer just declares policy."""
        return axis_name in self.coupled_axes


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
        for b in arcs[i + 1:]:
            if abs(state.kinematics.arc_angles[a]
                   - state.kinematics.arc_angles[b]) < ap_thr:
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
            for b in members[i + 1:]:
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
def _resolve_target_LPS_from_plan(
    plan: ProbePlan,
    target_index: dict[str, np.ndarray],
    assets_fallback: Optional[dict[str, TransformedPoints]] = None,
) -> np.ndarray:
    """Return a single (3,) LPS point for the plan's target."""
    # Inline ad-hoc target
    if plan.target_point_RAS is not None:
        ras = np.asarray(plan.target_point_RAS, dtype=float)
        return convert_coordinate_system(ras, "RAS", "LPS")

    # Catalog target by key
    if plan.target_key:
        pts = target_index.get(plan.target_key)
        if pts is None and assets_fallback is not None:
            tp = assets_fallback.get(plan.target_key)
            if tp is not None:
                pts = tp.raw  # already in LPS if your assets pipeline canonicalized it
        if pts is None:
            warn(f"Missing target for key: {plan.target_key!r}; using origin.")
            return np.zeros(3, dtype=float)
        return pts if pts.ndim == 1 else pts.mean(axis=0)

    warn("ProbePlan has neither target_key nor target_point_RAS; using origin.")
    return np.zeros(3, dtype=float)


def _resolved_angles(name: str, ps: PlanningState) -> tuple[float, float, float]:
    plan = ps.probes[name]
    cal = ps.calibrations.get(name)

    if plan.calibrated and cal is not None:
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
    # rig convention: positive is mouse pitch down (CW looking into right ML
    # axis), 0 is vertical
    ap: float = 0.0
    # rig convention: positive is mouse roll right (CCW looking into the front
    # AP axis), 0 is midline
    ml: float = 0.0
    # rig convention: positive is mouse yaw right, 0 is sites facing (left?)
    # (CW looking into superior DV axis)
    spin: float = 0.0
    tip: NDArray = field(default_factory=lambda: np.zeros(3))  # LPS

    def transform(self) -> AffineTransform:
        # Compute the transformation matrix from the probe's location
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
        offsets_RA.

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
        ap_deg, ml_deg, spin_deg = _resolved_angles(probe_name, ps)

        # --- target + offsets (RAS→LPS) ---
        tgt_LPS = _resolve_target_LPS_from_plan(
            plan, ps.target_index, assets_fallback=assets_targets_fallback
        )
        off_RAS = np.array(
            [plan.offsets_RA[0], plan.offsets_RA[1], 0.0], dtype=np.float64
        )
        off_LPS = convert_coordinate_system(off_RAS, "RAS", "LPS")
        adjusted_target = tgt_LPS + off_LPS

        # --- pivot lookup ---
        # Pivot is the recording-array centre in the canonical local
        # frame (746764b semantic): ``past_target_mm = 0`` lands the
        # recording bank on target. ``position_bearing_shank`` is a
        # *reporting* setting — it doesn't change the kinematic pivot,
        # only which shank's tip the GUI reports as the RAS readout.
        pivot_local: Optional[np.ndarray] = None
        if catalog is not None:
            asset_key = f"probe:{plan.kind}"
            spec = catalog.assets.get(asset_key)
            if spec is not None and spec.pivot_LPS is not None:
                pivot_local = np.asarray(spec.pivot_LPS, dtype=np.float64)
        if pivot_local is None:
            from aind_low_point.optimization.recording import (
                recording_center_local_for_kind,
            )

            pivot_local = recording_center_local_for_kind(plan.kind)

        # --- tip from depth, orientation, and pivot ---
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
@dataclass(slots=True)
class Probe:
    probe_type: str
    pose: ProbePose


# get_pivot_for_asset: asset_key -> local-space pivot (LPS mm) or None
GetPivotFn = Callable[[str], Optional[np.ndarray]]


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
    # Legacy callback hook. Kept for backward compatibility but should
    # be left at the default ``None``-returning function: pivot is now
    # baked into ``ProbePose.tip`` via the ``catalog`` route, and
    # double-wrapping here would shift the probe twice. Non-probe
    # assets that need an asset-level pivot can still use this; for
    # probe assets pass ``catalog`` instead.
    get_pivot_for_asset: GetPivotFn = (
        lambda _key: None
    )

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

        dyn = self._probe_chain(probe_name)

        # If the asset needs rotation about a local pivot (e.g., tip),
        # wrap the dynamic pose with +pivot / -pivot translations.
        # Probe pivots come through ``catalog`` and are already baked
        # into ``ProbePose.tip``; this path is now only used by other
        # asset types that opt in via ``get_pivot_for_asset``.
        pivot = self.get_pivot_for_asset(node.asset_key)
        if pivot is not None:
            T_p = AffineTransform(
                rotation=np.eye(3), translation=np.asarray(pivot, float)
            )
            T_m = AffineTransform(
                rotation=np.eye(3), translation=-np.asarray(pivot, float)
            )
            return TransformChain.new([T_p, *dyn.elements, T_m])

        return dyn
