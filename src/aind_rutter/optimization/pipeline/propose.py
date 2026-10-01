"""Propose allocations for the regions of one config, before any solve.

The config supplies the anatomy, the probe meshes and the rig pitch; its probes
only name the regions. Each proposal says which regions share a probe, which
kind each probe is, and the share of every region its line records: the same
figure emit writes for a solved plan, taken on a line no fixture or neighbour
has yet moved, so it is what a region can get and not what a plan will give.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aind_rutter.build.assemble import RuntimeBundle, build_runtime_from_config
from aind_rutter.build.queries import head_pitch_deg_from_runtime
from aind_rutter.config import ConfigModel
from aind_rutter.domain.plan import probe_asset_key
from aind_rutter.domain.pose import detect_shank_tips_local
from aind_rutter.domain.rig import reachable_ap_window_deg
from aind_rutter.domain.transforms import MeshTransformable
from aind_rutter.optimization.assignment.allocation import (
    Cover,
    ProbeShape,
    aim_points,
    best_per_count,
    covers,
    find_units,
)
from aind_rutter.optimization.geometry.holes import load_holes
from aind_rutter.optimization.pipeline.measure import region_clouds
from aind_rutter.optimization.pipeline.probe_setup import RetroDensityOpts


@dataclass(frozen=True)
class ProposeOptions:
    radius_mm: float = 0.2
    retro: bool = False  # score on the labelled cells, not the regions' voxels
    max_group: int = 3  # regions on one probe, at most
    min_kept: float = 0.1  # of a region's share alone, for it to share a probe
    per_count: int = 1  # proposals at each probe count
    angle_step_deg: float = 3.0
    spin_step_deg: float = 15.0
    depth_step_mm: float = 0.1
    # A larger cloud is sampled down to this, so its shares are estimates.
    max_points: int = 3000


@dataclass(frozen=True)
class ProposedProbe:
    kind: str
    regions: tuple[str, ...]  # the one nearest the stretch's middle first
    share: dict[str, float]
    hole: int
    # Where each region is aimed on the shared line; empty for a lone region.
    points_LPS: dict[str, NDArray[np.float64]]


@dataclass(frozen=True)
class Proposal:
    probes: tuple[ProposedProbe, ...]
    mean_share: float


@dataclass(frozen=True)
class Proposals:
    regions: tuple[str, ...]
    cloud: str  # "retro" or "voxels"
    alone: dict[str, dict[str, float]]  # region -> kind -> best share on its own probe
    proposals: tuple[Proposal, ...]


def probe_shape(runtime: RuntimeBundle, kind: str, count: int) -> ProbeShape:
    """The shanks and the recorded stretch of ``kind``, from its asset."""
    key = probe_asset_key(kind)
    spec = runtime.asset_catalog.assets.get(key)
    if spec is None or spec.recording is None:
        raise ValueError(f"{key} is not an asset with a recording array")
    ranges = set(spec.recording.active_ranges_mm)
    if len(ranges) != 1:
        raise ValueError(f"{kind}: shanks record different stretches: {sorted(ranges)}")
    geometry = runtime.asset_catalog.get_geometry(key)
    tips = (
        detect_shank_tips_local(geometry.raw)
        if isinstance(geometry, MeshTransformable)
        else np.zeros((1, 3))
    )
    offsets = np.asarray(tips, float) - np.asarray(tips, float).mean(axis=0)
    offsets[:, 2] = 0.0
    return ProbeShape(kind, offsets, next(iter(ranges)), count)


def _proposal(
    cover: Cover,
    regions: tuple[str, ...],
    clouds: list[NDArray],
    shapes: Mapping[str, ProbeShape],
    radius_mm: float,
) -> Proposal:
    probes = []
    for unit, line in sorted(cover.lines.items(), key=lambda kv: min(kv[0])):
        shape = shapes[line.kind]
        order, points = sorted(unit), {}
        if len(unit) > 1:
            at = aim_points(line, {r: clouds[r] for r in unit}, shape, radius_mm)
            middle = line.entry + (line.start_mm + shape.window_mm / 2) * line.direction
            order.sort(key=lambda r: float(np.linalg.norm(at[r] - middle)))
            points = {regions[r]: at[r] for r in order}
        probes.append(
            ProposedProbe(
                line.kind,
                tuple(regions[r] for r in order),
                {regions[r]: line.share[r] for r in order},
                line.hole,
                points,
            )
        )
    return Proposal(tuple(probes), cover.mean_share)


def propose(
    config: str | Path,
    holes: str | Path,
    inventory: Mapping[str, int],
    options: ProposeOptions = ProposeOptions(),
) -> Proposals:
    """Best allocations of ``inventory`` to the config's regions, by probe count."""
    cfg = ConfigModel.from_yaml(config)
    runtime = build_runtime_from_config(cfg)
    found = region_clouds(cfg, runtime, RetroDensityOpts() if options.retro else None)
    regions = tuple(found)
    rng = np.random.default_rng(0)
    clouds = [
        c
        if len(c) <= options.max_points
        else c[rng.choice(len(c), options.max_points, replace=False)]
        for c in found.values()
    ]
    shapes = [probe_shape(runtime, kind, count) for kind, count in inventory.items()]
    units = find_units(
        clouds,
        load_holes(holes),
        reachable_ap_window_deg(head_pitch_deg_from_runtime(runtime)),
        shapes,
        radius_mm=options.radius_mm,
        max_group=options.max_group,
        min_kept=options.min_kept,
        angle_step_deg=options.angle_step_deg,
        spin_step_deg=options.spin_step_deg,
        depth_step_mm=options.depth_step_mm,
    )
    alone = {
        name: {
            shape.kind: max(
                (
                    by_kind[shape.kind].total
                    for by_kind in units.get(frozenset([r]), {}).values()
                    if shape.kind in by_kind
                ),
                default=0.0,
            )
            for shape in shapes
        }
        for r, name in enumerate(regions)
    }
    by_kind = {shape.kind: shape for shape in shapes}
    chosen = best_per_count(covers(len(regions), units, shapes), options.per_count)
    return Proposals(
        regions,
        "retro" if options.retro else "voxels",
        alone,
        tuple(
            _proposal(c, regions, clouds, by_kind, options.radius_mm) for c in chosen
        ),
    )
