"""Rig mechanics: the arc manipulator's joints, limits and separations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from aind_mri_utils.arc_angles import arc_angles_to_affine
from numpy.typing import NDArray

from aind_rutter.domain.transforms import AffineTransform


@dataclass(frozen=True, slots=True)
class JointRange:
    lo: float
    hi: float

    def clamp(self, v: float) -> float:
        return float(min(max(v, self.lo), self.hi))


# Rig-mechanical angular ranges (deg, rig angles) for a config that states
# none; a config's ``plan.rig`` replaces them.
DEFAULT_AP_RANGE_DEG: tuple[float, float] = (-75.0, 75.0)
DEFAULT_ML_RANGE_DEG: tuple[float, float] = (-42.0, 42.0)
DEFAULT_SEPARATION_DEG = 16.0


def head_pitch_deg(subject_from_rig: AffineTransform) -> float:
    """Head pitch (deg) about L from a ``subject_from_rig`` transform; rig AP is
    subject AP plus this."""
    rotation = np.asarray(subject_from_rig.rotate_translate[0], dtype=float)
    return float(np.rad2deg(np.arctan2(rotation[2, 1], rotation[1, 1])))


def probe_rotation(ap_deg: float, ml_deg: float, spin_deg: float) -> NDArray:
    """Probe-local to subject-LPS rotation from subject-frame arc angles; the
    optimizer's JAX twin is parity-tested against it."""
    return arc_angles_to_affine(float(ap_deg), float(ml_deg), float(spin_deg))


class ArcRider(Protocol):
    """What an arc rig reads of a probe's plan."""

    arc_id: Optional[str]
    bind_ap_to_arc: bool
    ap_local: float
    ml_local: float


@dataclass(frozen=True, slots=True)
class ArcRig:
    """An arc manipulator rig: probes ride arcs, each arc sets one AP angle for
    its probes and each probe sets its own ML and spin.

    Everything the code knows about this rig's mechanics reads it from here:
    its ranges, the separations it needs between arcs and between probes on one
    arc, how many arcs and probes per arc those leave room for, and which
    planned angles it cannot set.
    """

    name: str | None = None
    # Angular ranges (deg) in rig angles; ML is the same in the subject frame,
    # AP is shifted by head pitch (see reachable_ap_window_deg).
    ap_deg: JointRange = field(
        default_factory=lambda: JointRange(*DEFAULT_AP_RANGE_DEG)
    )
    ml_deg: JointRange = field(
        default_factory=lambda: JointRange(*DEFAULT_ML_RANGE_DEG)
    )
    spin_deg: JointRange = field(default_factory=lambda: JointRange(-180.0, 180.0))
    # translational work envelope (mm); set to None if unbounded
    x_mm: Optional[JointRange] = None
    y_mm: Optional[JointRange] = None
    z_mm: Optional[JointRange] = None
    # Closest two arcs' APs, and two probes' MLs on one arc, may be (deg); the
    # manipulators collide nearer than this.
    min_arc_ap_sep_deg: float = DEFAULT_SEPARATION_DEG
    min_intra_arc_ml_sep_deg: float = DEFAULT_SEPARATION_DEG

    @property
    def ml_range_deg(self) -> tuple[float, float]:
        return self.ml_deg.lo, self.ml_deg.hi

    def reachable_ap_window_deg(self, head_pitch_deg: float) -> tuple[float, float]:
        """The subject-frame arc AP angles the rig can reach, given head pitch.

        Rig AP is subject AP plus head pitch, because the head is mounted
        nose-down, so the rig's AP range is shifted by ``-head_pitch_deg`` in
        the subject frame: at 14°, a rig range of [-75, +46] is [-89, +32].

        Every search bound, enumeration window and coverage ceiling has to use
        this same window: a ceiling computed over poses the solver cannot reach
        normalizes coverage by an unreachable maximum.
        """
        return self.ap_deg.lo - head_pitch_deg, self.ap_deg.hi - head_pitch_deg

    def max_arcs(self) -> int:
        """How many arcs fit in the AP range at the arc separation; there is no
        rail count, and the realized count is also at most the probe count."""
        span = self.ap_deg.hi - self.ap_deg.lo
        return int(span // self.min_arc_ap_sep_deg) + 1

    def max_probes_per_arc(self) -> int:
        """How many probes fit on one arc in the ML range at the ML separation;
        there is no mount count."""
        span = self.ml_deg.hi - self.ml_deg.lo
        return int(span // self.min_intra_arc_ml_sep_deg) + 1

    def violations(
        self,
        arc_angles: Mapping[str, float],
        probes: Mapping[str, ArcRider],
        head_pitch_deg: float,
    ) -> dict[str, set[tuple[str, ...]]]:
        """Planned angles this rig cannot set, by kind:

        ``"arc_ap"`` — (arc, arc) pairs, among arcs carrying a probe, nearer
        than ``min_arc_ap_sep_deg``; ``"within_arc_ml"`` — (probe, probe)
        pairs on one arc nearer in ML than ``min_intra_arc_ml_sep_deg``;
        ``"ap_range"`` — (probe,) whose rig AP (subject AP plus head pitch,
        from its arc when bound) is outside ``ap_deg``; ``"ml_range"`` —
        (probe,) whose ML is outside ``ml_deg``. Each tuple is sorted.
        """
        by_arc: dict[str, list[str]] = {}
        for name, plan in probes.items():
            if plan.arc_id is not None:
                by_arc.setdefault(plan.arc_id, []).append(name)

        # An arc no probe rides has nothing to keep apart, wherever it sits.
        arc_violations: set[tuple[str, ...]] = set()
        arcs = sorted(a for a in arc_angles if a in by_arc)
        for i, a in enumerate(arcs):
            for b in arcs[i + 1 :]:
                if abs(arc_angles[a] - arc_angles[b]) < self.min_arc_ap_sep_deg:
                    arc_violations.add((a, b))

        ml_violations: set[tuple[str, ...]] = set()
        for members in by_arc.values():
            members.sort()
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    gap = abs(probes[a].ml_local - probes[b].ml_local)
                    if gap < self.min_intra_arc_ml_sep_deg:
                        ml_violations.add((a, b))

        ap_range: set[tuple[str, ...]] = set()
        ml_range: set[tuple[str, ...]] = set()
        for name, plan in probes.items():
            subject_ap = plan.ap_local
            if plan.arc_id is not None and plan.bind_ap_to_arc:
                subject_ap = arc_angles.get(plan.arc_id, plan.ap_local)
            if not self.ap_deg.lo <= subject_ap + head_pitch_deg <= self.ap_deg.hi:
                ap_range.add((name,))
            if not self.ml_deg.lo <= plan.ml_local <= self.ml_deg.hi:
                ml_range.add((name,))

        return {
            "arc_ap": arc_violations,
            "within_arc_ml": ml_violations,
            "ap_range": ap_range,
            "ml_range": ml_range,
        }

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
    - rig: the rig's mechanics (ranges and separations)
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
    rig: ArcRig = field(default_factory=ArcRig)
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

    @property
    def head_pitch_deg(self) -> float:
        return head_pitch_deg(self.subject_from_rig)

    def set_arc(self, arc_id: str, ap_deg: float) -> float:
        """Store AP for an arc, unclamped: manual planning may leave the rig's
        range, which the kinematic overlay reports."""
        self.arc_angles[arc_id] = float(ap_deg)
        return self.arc_angles[arc_id]

    def clamp_xyz(self, tip_lps: np.ndarray) -> np.ndarray:
        return self.rig.clamp_xyz(tip_lps)
