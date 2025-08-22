"""Planning state - the domain logic"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    Optional,
    Set,
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
from aind_low_point.scene import NodeInstance


# Plan for probe location
@dataclass(slots=True)
class ProbePlan:
    probe_type: str
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
    target_key: Optional[str] = None  # preferred: reference into asset catalog
    target_point_RAS: Optional[Tuple[float, float, float]] = None  # ad-hoc fallback
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
      (by convention these names match your ProbePose fields: 'ap_deg', 'ml_deg', 'spin_deg', 'x_mm', 'y_mm', 'z_mm')
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


@dataclass(slots=True)
class PlanningState:
    kinematics: Kinematics
    probes: dict[str, ProbePlan]
    calibrations: dict[str, AffineTransform] = field(
        default_factory=dict
    )  # probe_name → calibration transform
    target_index: dict[str, Float3] = field(default_factory=dict)


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
        # AP: from arc if bound, else local; ML: always per-probe local unless you add another binding flag
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
    ) -> ProbePose:
        """
        Resolve a live pose from PlanningState (no mutations).
        - AP comes from calibration if plan.calibrated and matrix is present,
        else from arc if bound, else local.
        - ML comes from calibration if present/allowed, else local.
        - Spin is always the per-probe plan spin.
        - Target is taken from planning.target_index (or assets fallback) +
        offsets_RA.
        """
        plan = ps.probes[probe_name]
        cal = ps.calibrations.get(probe_name)

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

        # --- tip from depth and orientation ---
        R_probe = arc_angles_to_affine(ap_deg, ml_deg, spin_deg)
        insertion_vec = R_probe @ np.array(
            [0.0, 0.0, -float(plan.past_target_mm)], dtype=np.float64
        )
        tip = adjusted_target + insertion_vec
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
    planning: PlanningState
    get_pivot_for_asset: GetPivotFn = (
        lambda _key: None
    )  # default: rotate around asset origin

    # ---- dynamic pose for a probe (no scene knowledge) ----
    def _probe_chain(self, probe_name: str) -> TransformChain:
        pose = ProbePose.from_planning_state(self.planning, probe_name)
        return pose.chain()

    # ---- dynamic transform for a scene node (may be identity) ----
    def dynamic_chain_for_node(self, node: "NodeInstance") -> TransformChain:
        probe_name: Optional[str] = node.extras.get("pose_source_probe")
        if not probe_name:
            return TransformChain.new([AffineTransform.identity()])

        dyn = self._probe_chain(probe_name)

        # If the asset needs rotation about a local pivot (e.g., tip),
        # wrap the dynamic pose with +pivot / -pivot translations
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

    # ---- final world transform = base ∘ dynamic ----
    def world_chain_for_node(self, node: "NodeInstance") -> TransformChain:
        base = node.transform
        dyn = self.dynamic_chain_for_node(node)
        return TransformChain.new([*base.elements, *dyn.elements])

    def world_rt_for_node(self, node: "NodeInstance") -> tuple[np.ndarray, np.ndarray]:
        return self.world_chain_for_node(node).composed_transform
