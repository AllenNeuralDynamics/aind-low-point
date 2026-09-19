"""Rig mechanics: the arc manipulator's joints and their limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from aind_rutter.domain.transforms import AffineTransform


@dataclass(frozen=True, slots=True)
class JointRange:
    lo: float
    hi: float

    def clamp(self, v: float) -> float:
        return float(min(max(v, self.lo), self.hi))


# Rig-mechanical angular limits (deg), as half-ranges about 0. SINGLE SOURCE OF
# TRUTH: ``PoseLimits`` defaults and every optimizer search bound / enumeration
# window derive from these — change a limit here and it propagates everywhere
# (import ``AP_LIMIT_DEG`` / ``ML_LIMIT_DEG`` rather than hardcoding literals).
AP_LIMIT_DEG: float = 75.0
ML_LIMIT_DEG: float = 42.0


@dataclass(frozen=True, slots=True)
class PoseLimits:
    # angular limits (deg)
    ap_deg: JointRange = field(
        default_factory=lambda: JointRange(-AP_LIMIT_DEG, AP_LIMIT_DEG)
    )
    ml_deg: JointRange = field(
        default_factory=lambda: JointRange(-ML_LIMIT_DEG, ML_LIMIT_DEG)
    )
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

    def max_arcs(self) -> int:
        """Kinematic max arc count: how many arcs fit in the AP range at the
        arc-AP separation. There is NO hardware rail-count limit — arcs are
        angularly limited only. The realized count is additionally ≤ the
        probe count (one probe per arc minimum)."""
        span = self.ap_deg.hi - self.ap_deg.lo
        return int(span // self.min_arc_ap_separation_deg) + 1

    def max_probes_per_arc(self) -> int:
        """Kinematic max probes on one arc: how many fit in the ML range at
        the within-arc ML separation. NOT a hardware mount-count limit (the
        rig takes more than 4 per arc); the 16° ML exclusion is the only
        constraint. Also ≤ the probe count in practice."""
        span = self.ml_deg.hi - self.ml_deg.lo
        return int(span // self.min_within_arc_ml_separation_deg) + 1

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
      (names match ProbePose fields: ap_deg, ml_deg, spin_deg, x_mm, y_mm, z_mm)
    - subject_from_rig: rotation taking rig-frame vectors to subject
      anatomical LPS. Default identity. Encodes the mounted mouse's head
      tilt so that (ap, ml, spin) variables are interpreted in the rig's
      mechanical frame while all geometry stays in subject LPS. Composed
      as ``R_in_subject = subject_from_rig.rotation @
      arc_angles_to_affine(ap, ml, spin)``. The inverse,
      ``rig_from_subject = subject_from_rig.invert()``, is used to
      project subject-frame quantities (e.g. hole axes) into the rig
      frame for required-AP/ML scoring.
    """

    arc_angles: dict[str, float] = field(
        default_factory=dict
    )  # e.g., {"a": 12.0, "b": -8.0}
    limits: PoseLimits = field(default_factory=PoseLimits)
    subject_from_rig: AffineTransform = field(default_factory=AffineTransform.identity)

    # convenience helpers
    def get_arc(self, arc_id: str) -> float:
        return float(self.arc_angles[arc_id])

    @property
    def rig_from_subject_rotation(self) -> "NDArray":
        """3×3 rotation taking subject-LPS vectors into the rig frame.

        Computed as the transpose of ``subject_from_rig.rotation`` since
        the transform is rigid (rotation only on this axis). Cached
        re-computation per call is fine — it's a single matrix transpose.
        """
        R, _ = self.subject_from_rig.rotate_translate
        return np.asarray(R, dtype=np.float64).T

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
