"""Probe-setup utilities shared across pipeline scripts.

This module exports the infrastructure helpers that production scripts need:

  - :class:`RetroDensityOpts` / :func:`retro_opts_from_env`
  - :func:`_transform_holes`
  - :func:`_probe_static_info`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aind_rutter.build.probe_context import probe_context_from_runtime
from aind_rutter.domain.plan import resolve_region_points_LPS
from aind_rutter.domain.scene import resolve_base_geometry
from aind_rutter.optimization.clearance.envelope import shank_half_extents_xy
from aind_rutter.optimization.geometry import HoleSection
from aind_rutter.optimization.geometry.holes import Hole, HoleWall
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo


@dataclass(frozen=True)
class RetroDensityOpts:
    """When set, the optimizer's per-probe target becomes a masked
    point cloud drawn from a labelled-cell asset (e.g. retro/rabies
    points), clipped to the intersection of brain and structure masks.

    The per-probe density switches from a single-point Gaussian on the
    centroid to an equally-weighted Gaussian mixture on the masked
    points with bandwidth ``sigma_mm``. ``target_LPS`` is set to the
    centroid of the masked cloud for pose-bank anchoring.
    """

    retro_asset_key: str = "retro-targets"
    common_mask_keys: tuple[str, ...] = ("brain",)
    region_mask_fmt: str = "structure:{region}"
    sigma_mm: float = 0.3


def retro_opts_from_env(runtime=None) -> "RetroDensityOpts | None":
    """``RetroDensityOpts()`` when ``RUTTER_RETRO_DENSITY`` is enabled in the env and
    the retro asset is present in ``runtime``; otherwise ``None``. Lets every
    driver opt into density (KDE) coverage uniformly with
    ``RUTTER_RETRO_DENSITY=1`` — off by
    default, so subjects without a retro cloud keep the single-point target."""
    import os as _os

    if _os.environ.get("RUTTER_RETRO_DENSITY", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return None
    opts = RetroDensityOpts()
    if runtime is not None:
        try:
            if runtime.asset_catalog.get_spec(opts.retro_asset_key) is None:
                return None
        except Exception:
            return None
    return opts


def _transform_holes(holes: list[Hole], R: np.ndarray, t: np.ndarray) -> list[Hole]:
    """Apply a rigid transform (R, t) to every hole's positions, axis and walls.
    Oval ``a/b/theta`` (in the per-axis basis) are invariant under
    rigid rotation, so they're preserved as-is."""
    out: list[Hole] = []
    for h in holes:
        new_axis = R @ np.asarray(h.axis, dtype=np.float64)
        new_axis = new_axis / np.linalg.norm(new_axis)
        new_ref = R @ np.asarray(h.ref_point, dtype=np.float64) + t
        new_sections = [
            HoleSection(
                axis=new_axis,
                center=R @ np.asarray(s.center, dtype=np.float64) + t,
                a=s.a,
                b=s.b,
                theta=s.theta,
            )
            for s in h.sections
        ]
        new_walls = tuple(
            HoleWall(
                point=R @ np.asarray(w.point, dtype=np.float64) + t,
                normal=R @ np.asarray(w.normal, dtype=np.float64),
            )
            for w in h.walls
        )
        out.append(
            Hole(
                id=h.id,
                axis=new_axis,
                ref_point=new_ref,
                sections=new_sections,
                walls=new_walls,
            )
        )
    return out


# Per-subject retro arrays, computed ONCE and shared across all probes: the
# corrected scene-LPS cloud, the brain-membership mask, and the CCF annotation
# label at each retro point. Keyed by (runtime id, retro key, mask keys, annot
# path) so the volume reads + 28k-point voxel lookups never repeat per probe.
_RETRO_VOXEL_CACHE: dict = {}


def _retro_voxel_base(runtime, opts: RetroDensityOpts, annot_path: str):
    catalog, scene = runtime.asset_catalog, runtime.scene
    key = (id(runtime), opts.retro_asset_key, tuple(opts.common_mask_keys), annot_path)
    hit = _RETRO_VOXEL_CACHE.get(key)
    if hit is not None:
        return hit
    import SimpleITK as sitk

    from aind_rutter.build.loaders import csv_points, voxel_values_at

    retro_t = resolve_base_geometry(catalog, scene, opts.retro_asset_key)
    if retro_t is None:
        raise RuntimeError(
            f"--retro-density: asset {opts.retro_asset_key!r} not in scene"
        )
    world = np.asarray(retro_t.raw, dtype=np.float64)
    retro_spec = catalog.get_spec(opts.retro_asset_key)
    subject = np.asarray(csv_points(str(retro_spec.source_path)), dtype=np.float64)
    if len(subject) != len(world):
        raise RuntimeError(
            f"--retro-density: raw CSV ({len(subject)}) vs scene cloud "
            f"({len(world)}) length mismatch — canonicalization changed point count"
        )
    brain_keep = np.ones(len(subject), dtype=bool)
    for mk in opts.common_mask_keys:
        spec = catalog.get_spec(mk)
        vals, inb = voxel_values_at(sitk.ReadImage(str(spec.source_path)), subject)
        brain_keep &= inb & (vals > 0)
    annot_vals, annot_inb = voxel_values_at(sitk.ReadImage(annot_path), subject)
    out = (world, subject, brain_keep, annot_vals, annot_inb)
    _RETRO_VOXEL_CACHE[key] = out
    return out


CCF_REGION_LOADERS = frozenset({"ccf_annotation_region", "ccf_region_voxel_points"})


def region_definition(spec) -> dict:
    """The CCF region a structure asset stands for.

    Its loader's own arguments, unless the metadata restates one: a config may
    draw a structure in both hemispheres and score it in one.
    """
    told = spec.loader_kwargs if spec.loader in CCF_REGION_LOADERS else {}
    meta = spec.metadata
    return {
        "acronym": meta.get("ccf_acronym")
        or meta.get("acronym")
        or told.get("acronym"),
        "label_id": meta.get("label_id", told.get("label_id")),
        "hemisphere": meta.get("hemisphere", told.get("hemisphere", "both")),
        "keep": tuple(meta.get("keep", told.get("keep", ()))),
    }


def masked_retro_points(runtime, region: str, opts: RetroDensityOpts) -> np.ndarray:
    """Retro points (corrected scene-LPS) inside ``region``'s CCF structure AND
    the brain, by direct voxel lookup against the SOURCE volumes."""
    import SimpleITK as sitk

    from aind_rutter.build.loaders import (
        ccf_region_bounds,
        ccf_region_label_ids,
        ccf_region_membership,
    )
    from aind_rutter.ccf.regions import half_mask

    catalog = runtime.asset_catalog
    mask_key = opts.region_mask_fmt.format(region=region)
    sspec = catalog.get_spec(mask_key)
    if sspec is None or sspec.source_path is None:
        raise RuntimeError(f"--retro-density: structure asset {mask_key!r} missing")
    annot_path = str(sspec.source_path)
    world, subject, brain_keep, annot_vals, annot_inb = _retro_voxel_base(
        runtime, opts, annot_path
    )
    region_def = region_definition(sspec)
    acronym, hemisphere = region_def["acronym"], region_def["hemisphere"]
    match_ids = ccf_region_label_ids(
        acronym=acronym, label_id=region_def["label_id"], hemisphere=hemisphere
    )
    keep = ccf_region_membership(
        annot_vals, annot_inb, match_ids, brain_keep=brain_keep
    )
    if region_def["keep"]:
        lo, hi = ccf_region_bounds(sitk.ReadImage(annot_path), match_ids)
        keep &= half_mask(
            subject, region_def["keep"], hemisphere=hemisphere, lo=lo, hi=hi
        )
    masked = world[keep]
    if masked.shape[0] == 0:
        raise RuntimeError(
            f"--retro-density: region {region!r} has zero retro points in "
            f"brain ∩ {acronym!r} — check mask/annotation alignment"
        )
    return masked


def region_weighted_cloud(
    clouds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None]:
    """One cloud from each region's, and its point weights.

    Each region's points share a weight of one, so every region scores what a
    probe aimed at it alone would however densely it is labelled. A single
    region keeps the builder's own equal weighting (``None``).
    """
    if len(clouds) == 1:
        return clouds[0], None
    weights = np.concatenate([np.full(len(c), 1.0 / len(c)) for c in clouds])
    return np.concatenate(clouds), weights


def _probe_static_info(
    plan_state,
    runtime,
    name: str,
    retro_opts: RetroDensityOpts | None = None,
    *,
    thread_corners: bool = True,
) -> ProbeStaticInfo:
    """Build a ProbeStaticInfo for one probe from the runtime.

    ``thread_corners`` threads the shank's cross-section corners, measured from
    the collision mesh, rather than its centreline.
    """
    plan = runtime.plan_state.probes[name]
    region_points = resolve_region_points_LPS(plan, runtime.plan_state.target_index)
    target_points = target_point_weights = anchor_points = None
    if retro_opts is not None:
        if region_points is not None and not plan.regions:
            raise ValueError(
                f"probe {name!r} lists also_record without regions, so retro "
                "density cannot tell which structure masks each cloud"
            )
        clouds = [
            masked_retro_points(runtime, region, retro_opts)
            for region in plan.regions or (name,)
        ]
        target_points, target_point_weights = region_weighted_cloud(clouds)
        # The target region alone anchors the pose, as without retro density.
        anchor_points = clouds[0]
    context = probe_context_from_runtime(runtime, name, target_points_LPS=anchor_points)
    sigma = retro_opts.sigma_mm if retro_opts is not None else 0.5
    shank_half_xy = None
    if thread_corners and context.collision_mesh is not None:
        shank_half_xy = shank_half_extents_xy(
            context.collision_mesh, context.shank_tips_local
        )
    return ProbeStaticInfo(
        name=name,
        target_LPS=context.target_LPS,
        kind=context.kind,
        shank_tips_local=context.shank_tips_local,
        recording=context.recording,
        density_sigma_mm=sigma,
        collision_mesh=context.collision_mesh,
        target_points=target_points,
        target_point_weights=target_point_weights,
        region_points=region_points,
        coverage_weight=context.coverage_weight,
        pivot_local=context.pivot_local,
        shank_half_xy=shank_half_xy,
    )
