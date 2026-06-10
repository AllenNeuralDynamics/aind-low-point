"""Per-pair pose-bank scoring used by pose feature precomputation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.geometry import shaft_section_oval_value
from aind_low_point.optimization.geometry.holes import Hole
from aind_low_point.optimization.geometry.kinematics import (
    pose_at_hole_best_fit,
    shank_capsules_from_pose,
)
from aind_low_point.optimization.geometry.recording import (
    RecordingGeometry,
    get_recording_geometry,
)
from aind_low_point.optimization.objectives.density import coverage, gaussian_density


@dataclass(frozen=True)
class PoseBankProbe:
    """Per-probe shape consumed by pose-bank scoring."""

    name: str
    target_LPS: NDArray[np.floating]
    shank_tips_local: NDArray[np.floating]
    kind: str = "2.1"
    density_sigma_mm: float = 0.5


@dataclass(frozen=True)
class MultiPoseScore:
    """Best-of-bank threading and coverage aggregates for one probe-hole pair."""

    min_violation_sq: float
    min_max_g: float
    max_coverage: float


def _rotation_about_axis(axis: NDArray, angle_rad: float) -> NDArray:
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def _rotate_to(from_dir: NDArray, to_dir: NDArray) -> NDArray:
    f = np.asarray(from_dir, dtype=np.float64)
    t = np.asarray(to_dir, dtype=np.float64)
    cos = float(np.clip(np.dot(f, t), -1.0, 1.0))
    if cos > 1.0 - 1e-12:
        return np.eye(3)
    if cos < -1.0 + 1e-12:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, f)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        ax = np.cross(f, helper)
        ax /= np.linalg.norm(ax)
        return _rotation_about_axis(ax, np.pi)
    ax = np.cross(f, t)
    ax /= np.linalg.norm(ax)
    return _rotation_about_axis(ax, float(np.arccos(cos)))


def _build_pose_bank(
    probe: PoseBankProbe,
    hole: Hole,
    pivot_local: NDArray,
    *,
    tilt_deg: float = 2.0,
) -> list[tuple[NDArray, NDArray]]:
    """Sample poses around a target-oriented insertion for pair scoring."""
    R_base, _ = pose_at_hole_best_fit(hole)
    target_LPS = np.asarray(probe.target_LPS, dtype=np.float64)
    pivot_local = np.asarray(pivot_local, dtype=np.float64)

    bore_dir = -np.asarray(hole.axis, dtype=np.float64)
    bore_dir /= np.linalg.norm(bore_dir)
    to_target = target_LPS - np.asarray(hole.sections[-1].center, dtype=np.float64)
    n = float(np.linalg.norm(to_target))
    target_dir = to_target / n if n >= 1e-9 else bore_dir

    R_target = _rotate_to(bore_dir, target_dir) @ R_base

    def _anchor(R: NDArray) -> NDArray:
        return target_LPS - R @ pivot_local

    poses: list[tuple[NDArray, NDArray]] = [(R_target, _anchor(R_target))]

    tilt_rad = float(np.deg2rad(tilt_deg))
    e_x = np.array([1.0, 0.0, 0.0])
    e_y = np.array([0.0, 1.0, 0.0])
    for sign in (+1.0, -1.0):
        for axis in (e_x, e_y):
            R = _rotation_about_axis(axis, sign * tilt_rad) @ R_target
            poses.append((R, _anchor(R)))
    return poses


def multi_pose_evaluate(
    probe: PoseBankProbe,
    hole: Hole,
    *,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
    tilt_deg: float = 2.0,
    coverage_n_samples: int = 41,
) -> MultiPoseScore:
    """Evaluate one probe-hole pair over a small target-oriented pose bank."""
    try:
        geom = get_recording_geometry(probe.kind)
    except Exception:
        geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))

    tips = np.asarray(probe.shank_tips_local, dtype=np.float64)
    if tips.shape[0] > 0:
        pivot_local = np.array(
            [
                float(tips[:, 0].mean()),
                float(tips[:, 1].mean()),
                float(geom.active_center_mm),
            ],
            dtype=np.float64,
        )
    else:
        pivot_local = np.array([0.0, 0.0, geom.active_center_mm], dtype=np.float64)
    poses = _build_pose_bank(probe, hole, pivot_local, tilt_deg=tilt_deg)

    density_fn = gaussian_density(probe.target_LPS, sigma_mm=probe.density_sigma_mm)

    min_vio_sq = float("inf")
    min_max_g = float("inf")
    max_cov = 0.0
    for R, tp in poses:
        capsules = shank_capsules_from_pose(
            R,
            tp,
            probe.shank_tips_local,
            shaft_length_mm=shaft_length_mm,
            shank_radius_mm=shank_radius_mm,
        )
        if not capsules:
            continue
        gs = np.array(
            [
                shaft_section_oval_value(cap, sec)
                for cap in capsules
                for sec in hole.sections
            ],
            dtype=np.float64,
        )
        if gs.size > 0:
            vio = float(np.sum(np.maximum(0.0, gs) ** 2))
            min_vio_sq = min(min_vio_sq, vio)
            min_max_g = min(min_max_g, float(np.max(gs)))
        if len(capsules) == geom.n_shanks:
            cov = coverage(density_fn, capsules, geom, n_samples=coverage_n_samples)
            max_cov = max(max_cov, float(cov))

    if min_vio_sq == float("inf"):
        min_vio_sq = 0.0
    if min_max_g == float("inf"):
        min_max_g = 0.0
    return MultiPoseScore(
        min_violation_sq=min_vio_sq,
        min_max_g=min_max_g,
        max_coverage=max_cov,
    )
