"""Per-(probe, hole) static pose features for the joint reranker.

Stage 0 of the joint reranking layer (between the LSAP and arc
partitioner, and the full inner solve). For every (probe, hole) pair
this module precomputes a small set of static features that the
reduced-SLSQP scoring stage in :mod:`joint_rerank` consumes:

- The closed-form rig ``(ap, ml)`` that aligns the probe shaft with
  the bore-to-target unit vector (a useful warm start for the per-arc
  AP and per-probe ML variables in the reduced search).
- The slot major-axis spin (``π/2 − slot_theta``), matching the warm
  start :func:`_build_initial_x` already uses in the full inner solve.
- The range of rig AP values around the bore-aligning AP where the
  probe can be threaded with ``max_g ≤ threading_oval_tolerance``
  (when the ml is set by the same formula as above). This is a 1-D
  sweep; the connected feasible interval containing (or nearest to)
  the bore-aligning AP is returned.
- A static threading "max_g" reading at the per-pair best pose (from
  :func:`multi_pose_evaluate`).
- A static coverage reading at the target-aligned pose.

Pose features are static — they depend only on the probe-hole pairing,
not on the joint assignment. Computing them once up front amortises
the per-(H, A) reduced-SLSQP cost in :mod:`joint_rerank`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.geometry import shaft_section_oval_value
from aind_low_point.optimization.hole_assignment import (
    AssignmentProbe,
    multi_pose_evaluate,
)
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.kinematics import (
    pose_from_optimizer_vars,
    shank_capsules_from_pose,
)
from aind_low_point.optimization.optimize import ProbeStaticInfo
from aind_low_point.optimization.recording import (
    RecordingGeometry,
    get_recording_geometry,
)


@dataclass(frozen=True)
class PoseFeatures:
    """Static features for one (probe, hole) pair.

    Attributes
    ----------
    required_ap_deg
        Rig ``ap`` that aligns the probe shaft with the bore-to-target
        unit vector ``b`` (subject LPS): ``atan2(b_y, -b_z)`` in
        degrees. When the target lies on the bore axis this equals
        :func:`kinematics.required_ap_deg`; for off-axis targets the
        two differ by the tilt needed to reach the target.
    required_ml_deg
        Companion ``ml`` from the same closed form: ``arcsin(b_x)``
        in degrees.
    slot_theta_rad
        Slot major-axis angle (radians). Used as the warm-start
        ``spin`` in the joint reranker via ``π/2 − slot_theta``,
        matching :func:`optimize._build_initial_x`.
    ap_interval_deg
        Connected range of rig ``ap`` values (around ``required_ap``)
        where the probe threads the bore with ``max_g`` at or below
        ``threading_oval_tolerance``, holding ``ml = ml(ap)`` to keep
        the shaft pointed at the target. ``(required_ap, required_ap)``
        when no sampled AP is feasible.
    static_max_g
        Best-of-bank threading ``max_g`` from
        :func:`hole_assignment.multi_pose_evaluate`.
    static_coverage
        Best-of-bank coverage (target-aligned pose) from the same.
    """

    required_ap_deg: float
    required_ml_deg: float
    slot_theta_rad: float
    ap_interval_deg: tuple[float, float]
    static_max_g: float
    static_coverage: float


def _bore_to_target_unit(hole: Hole, target_LPS: NDArray) -> NDArray:
    """Unit vector from the hole's bottom-section center to the target.

    Returns the negated downward bore axis (``-hole.axis``) when the
    target coincides with the bore center — defensive zero-norm guard
    rather than a NaN propagation.
    """
    center = np.asarray(hole.sections[-1].center, dtype=np.float64)
    target = np.asarray(target_LPS, dtype=np.float64)
    b = target - center
    n = float(np.linalg.norm(b))
    if n < 1e-12:
        axis = np.asarray(hole.axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        return -axis
    return b / n


def required_ap_ml_for_target(hole: Hole, target_LPS: NDArray) -> tuple[float, float]:
    """Closed-form rig ``(ap, ml)`` that aligns the shaft with the
    bore-to-target unit vector ``b``.

    Returns ``(ap_deg, ml_deg)`` such that

    .. code-block::

        arc_angles_to_affine(ap, ml, 0) @ [0, 0, -1] == b

    Algebra: with ``b = (bx, by, bz)``, the rig kinematics give
    ``ml = arcsin(bx)`` and ``ap = atan2(by, -bz)``. The shaft "points
    at the target" in the sense that the probe's local ``-z`` direction
    (down the shaft) equals ``b``.
    """
    b = _bore_to_target_unit(hole, target_LPS)
    ml = float(np.rad2deg(np.arcsin(np.clip(float(b[0]), -1.0, 1.0))))
    ap = float(np.rad2deg(np.arctan2(float(b[1]), -float(b[2]))))
    return ap, ml


def _ml_for_ap(ap_deg: float, b: NDArray) -> float:
    """ML (deg) that minimises shaft-vs-target angular error at a given AP.

    With the rig parameterisation ``arc_angles_to_affine(ap, ml, 0) @
    [0, 0, -1] = (sin(ml), cos(ml)·sin(ap), -cos(ml)·cos(ap))``, the
    target-pointing ML at a fixed AP is given by

    .. code-block::

        ml = atan2(b_x, sin(ap)·b_y - cos(ap)·b_z)

    derived from the inner product of the shaft direction with the
    target direction.
    """
    ap_rad = float(np.deg2rad(ap_deg))
    denom = float(np.sin(ap_rad) * float(b[1]) - np.cos(ap_rad) * float(b[2]))
    return float(np.rad2deg(np.arctan2(float(b[0]), denom)))


def _pivot_local(
    probe: ProbeStaticInfo | AssignmentProbe,
    recording_geom: RecordingGeometry,
) -> NDArray:
    """Per-probe local-frame pivot point used by the inner solve.

    Matches :func:`evaluate_probe`: ``(centroid_x, centroid_y,
    active_center_mm)`` where the centroid is over the probe's
    canonicalised shank tips. For zero-tip probes (defensive only),
    falls back to ``(0, 0, active_center_mm)``.
    """
    tips = np.asarray(probe.shank_tips_local, dtype=np.float64)
    if tips.shape[0] > 0:
        return np.array(
            [
                float(tips[:, 0].mean()),
                float(tips[:, 1].mean()),
                float(recording_geom.active_center_mm),
            ],
            dtype=np.float64,
        )
    return np.array(
        [0.0, 0.0, float(recording_geom.active_center_mm)], dtype=np.float64
    )


def _max_g_at_pose(
    probe: ProbeStaticInfo | AssignmentProbe,
    hole: Hole,
    *,
    ap_deg: float,
    ml_deg: float,
    spin_deg: float,
    pivot_local: NDArray,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
) -> float:
    """Worst-case threading ``g`` across (shank × section) at a pose.

    Centres the recording array on the target (pivot anchoring matches
    the inner loop). Returns ``+inf`` when there are no shanks.
    """
    R, pose_tip = pose_from_optimizer_vars(
        target_LPS=probe.target_LPS,
        ap_deg=ap_deg,
        ml_deg=ml_deg,
        spin_deg=spin_deg,
        offset_R_mm=0.0,
        offset_A_mm=0.0,
        past_target_mm=0.0,
        recording_center_local=pivot_local,
    )
    capsules = shank_capsules_from_pose(
        R,
        pose_tip,
        probe.shank_tips_local,
        shaft_length_mm=shaft_length_mm,
        shank_radius_mm=shank_radius_mm,
    )
    if not capsules:
        return float("inf")
    gs = [shaft_section_oval_value(sh, sec) for sh in capsules for sec in hole.sections]
    return float(max(gs)) if gs else float("inf")


def _connected_interval(
    feasible: list[bool],
    aps: list[float],
    center_ap: float,
) -> tuple[float, float]:
    """Connected run of ``True`` entries in ``feasible`` containing or
    nearest to ``center_ap``.

    Returns the ``(min_ap, max_ap)`` of the chosen run. When no entry is
    feasible, returns ``(center_ap, center_ap)`` (zero-width interval).
    """
    n = len(feasible)
    if n == 0 or not any(feasible):
        return (float(center_ap), float(center_ap))

    # Find maximal True runs.
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not feasible[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and feasible[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1

    # Prefer the run that contains center_ap (smallest |ap - center_ap|).
    # Equivalent: pick the run minimising the distance from center_ap to
    # the run's [min, max] interval.
    def run_distance(run: tuple[int, int]) -> float:
        lo, hi = aps[run[0]], aps[run[1]]
        if center_ap < lo:
            return lo - center_ap
        if center_ap > hi:
            return center_ap - hi
        return 0.0

    best = min(runs, key=run_distance)
    return (float(aps[best[0]]), float(aps[best[1]]))


def precompute_pose_features(
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    *,
    threading_oval_tolerance: float = 0.0,
    ap_sweep_half_deg: float = 25.0,
    ap_sweep_step_deg: float = 1.0,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
) -> dict[tuple[str, int], PoseFeatures]:
    """Precompute :class:`PoseFeatures` for every (probe, hole) pair.

    Parameters
    ----------
    probes, holes
        Inputs to the optimizer driver. Order is preserved.
    threading_oval_tolerance
        Threshold for declaring a swept AP feasible: ``max_g ≤
        tolerance`` ⇒ feasible.
    ap_sweep_half_deg
        Half-width of the AP sweep around ``required_ap``. Default
        25° covers the realistic rig envelope around the bore-aligning
        AP without ballooning the per-pair cost.
    ap_sweep_step_deg
        Sweep resolution in degrees. Default 1° gives 51 samples per
        pair at the default half-width.

    Returns
    -------
    dict
        Keyed by ``(probe.name, hole.id)``.
    """
    fallback_geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))
    geom_cache: dict[str, RecordingGeometry] = {}

    def _geom(kind: str) -> RecordingGeometry:
        if kind in geom_cache:
            return geom_cache[kind]
        try:
            geom = get_recording_geometry(kind)
        except KeyError:
            geom = fallback_geom
        geom_cache[kind] = geom
        return geom

    out: dict[tuple[str, int], PoseFeatures] = {}
    for probe in probes:
        geom = _geom(probe.kind)
        pivot_local = _pivot_local(probe, geom)
        # AssignmentProbe is the shape multi_pose_evaluate consumes; the
        # extra fields (density_sigma_mm) are inherited from probe.
        ap_probe = AssignmentProbe(
            name=probe.name,
            target_LPS=np.asarray(probe.target_LPS, dtype=np.float64),
            shank_tips_local=np.asarray(probe.shank_tips_local, dtype=np.float64),
            kind=probe.kind,
            density_sigma_mm=probe.density_sigma_mm,
        )
        for hole in holes:
            ap_req, ml_req = required_ap_ml_for_target(hole, probe.target_LPS)
            spin_warm_rad = float(np.pi / 2 - hole.slot_theta_rad)
            spin_warm_deg = float(np.rad2deg(spin_warm_rad))
            b = _bore_to_target_unit(hole, probe.target_LPS)

            # Sweep AP around required_ap, hold ml = ml(ap), record
            # feasibility of the threading constraint.
            half = float(ap_sweep_half_deg)
            step = float(ap_sweep_step_deg)
            n_steps = int(round(2 * half / step)) + 1
            aps = [ap_req - half + i * step for i in range(n_steps)]
            feasible: list[bool] = []
            for ap_sample in aps:
                ml_sample = _ml_for_ap(ap_sample, b)
                max_g = _max_g_at_pose(
                    probe,
                    hole,
                    ap_deg=ap_sample,
                    ml_deg=ml_sample,
                    spin_deg=spin_warm_deg,
                    pivot_local=pivot_local,
                    shaft_length_mm=shaft_length_mm,
                    shank_radius_mm=shank_radius_mm,
                )
                feasible.append(max_g <= threading_oval_tolerance)
            interval = _connected_interval(feasible, aps, ap_req)

            score = multi_pose_evaluate(
                ap_probe,
                hole,
                shaft_length_mm=shaft_length_mm,
                shank_radius_mm=shank_radius_mm,
            )

            out[(probe.name, int(hole.id))] = PoseFeatures(
                required_ap_deg=float(ap_req),
                required_ml_deg=float(ml_req),
                slot_theta_rad=float(hole.slot_theta_rad),
                ap_interval_deg=(float(interval[0]), float(interval[1])),
                static_max_g=float(score.min_max_g),
                static_coverage=float(score.max_coverage),
            )
    return out
