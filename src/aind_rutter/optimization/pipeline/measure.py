"""How much of each region a plan's active sites reach, on one scale for all kinds.

The objective's coverage is divided by a ceiling that depends on the probe kind,
so it cannot compare a quad base on a region with a single shank on it, nor two
configs that split the regions over different probes. This is the share of a
region's own points within a radius of any probe's active sites, every probe
counting: the retro cells labelled in it when the run scores retro density,
else its voxels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aind_rutter.build.assemble import RuntimeBundle, load_asset_geometry
from aind_rutter.build.probe_context import probe_context_from_runtime
from aind_rutter.config import ConfigModel
from aind_rutter.domain.plan import PlanningState
from aind_rutter.domain.pose import ProbePose
from aind_rutter.domain.scene import resolve_base_pose
from aind_rutter.optimization.pipeline.probe_setup import (
    RetroDensityOpts,
    masked_retro_points,
)

VOXEL_LOADERS = {"ccf_annotation_region": "ccf_region_voxel_points"}


@dataclass(frozen=True)
class ActiveSegment:
    """One shank's active stretch, from its deep end towards the base."""

    start: NDArray[np.float64]
    axis: NDArray[np.float64]  # unit
    length_mm: float


def region_names(plan_state: PlanningState) -> list[str]:
    """Every region some probe records, in probe order."""
    names: dict[str, None] = {}
    for name, plan in plan_state.probes.items():
        names.update(dict.fromkeys(plan.regions or (name,)))
    return list(names)


def _region_voxels(
    cfg: ConfigModel, runtime: RuntimeBundle, key: str
) -> NDArray[np.float64] | None:
    spec = runtime.asset_catalog.assets.get(key)
    if spec is None or spec.loader not in VOXEL_LOADERS:
        return None
    points = np.asarray(
        load_asset_geometry(cfg, key, loader=VOXEL_LOADERS[spec.loader]), np.float64
    )
    pose = resolve_base_pose(runtime.scene, key)
    return points if pose is None else np.asarray(pose.apply_to(points))


def region_clouds(
    cfg: ConfigModel, runtime: RuntimeBundle, retro_opts: RetroDensityOpts | None
) -> dict[str, NDArray[np.float64]]:
    """Each region's points in scene LPS; a region with no structure asset is absent."""
    mask_fmt = (retro_opts or RetroDensityOpts()).region_mask_fmt
    clouds: dict[str, NDArray[np.float64]] = {}
    for region in region_names(runtime.plan_state):
        key = mask_fmt.format(region=region)
        if key not in runtime.asset_catalog.assets:
            continue
        cloud = (
            _region_voxels(cfg, runtime, key)
            if retro_opts is None
            else masked_retro_points(runtime, region, retro_opts)
        )
        if cloud is not None:
            clouds[region] = cloud
    return clouds


def shank_geometry(
    runtime: RuntimeBundle,
) -> dict[str, tuple[NDArray[np.float64], tuple[tuple[float, float], ...]]]:
    """Per probe, its shank tips in the probe frame and each one's active range."""
    out = {}
    for name in runtime.plan_state.probes:
        context = probe_context_from_runtime(runtime, name)
        if context.recording is not None:
            out[name] = (context.shank_tips_local, context.recording.active_ranges_mm)
    return out


def active_segments(
    plan_state: PlanningState, runtime: RuntimeBundle, shanks: dict
) -> list[ActiveSegment]:
    """Where every shank records under ``plan_state``'s poses."""
    segments = []
    for name, (tips_local, ranges) in shanks.items():
        pose = ProbePose.from_planning_state(
            plan_state, name, catalog=runtime.asset_catalog
        )
        R = np.asarray(pose.transform().rotation)[:3, :3]
        up = R[:, 2]
        for tip_local, (lo, hi) in zip(tips_local, ranges):
            tip = np.asarray(pose.tip) + R @ tip_local
            segments.append(ActiveSegment(tip + lo * up, up, hi - lo))
    return segments


def recorded_share(
    points: NDArray[np.float64], segments: list[ActiveSegment], radius_mm: float
) -> float:
    """The share of ``points`` within ``radius_mm`` of any segment."""
    hit = np.zeros(len(points), bool)
    for seg in segments:
        rel = points - seg.start
        along = np.clip(rel @ seg.axis, 0.0, seg.length_mm)
        off = rel - along[:, None] * seg.axis
        hit |= np.einsum("ij,ij->i", off, off) <= radius_mm**2
    return float(hit.mean())
