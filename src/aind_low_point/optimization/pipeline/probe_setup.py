"""Probe-setup utilities shared across pipeline scripts.

This module exports the infrastructure helpers that production scripts need:

  - :class:`RetroDensityOpts` / :func:`retro_opts_from_env`
  - :func:`_transform_holes`
  - :func:`_probe_static_info`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aind_low_point.optimization.geometry import HoleSection
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.optimize import ProbeStaticInfo
from aind_low_point.runtime.probe_context import probe_context_from_runtime
from aind_low_point.scene import resolve_base_geometry


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
    per_probe_mask_fmt: str = "structure:{probe}"
    sigma_mm: float = 0.3


def retro_opts_from_env(runtime=None) -> "RetroDensityOpts | None":
    """``RetroDensityOpts()`` when ``RETRO_DENSITY`` is enabled in the env and the
    retro asset is present in ``runtime``; otherwise ``None``. Lets every driver
    opt into density (KDE) coverage uniformly with ``RETRO_DENSITY=1`` — off by
    default, so subjects without a retro cloud keep the single-point target."""
    import os as _os

    if _os.environ.get("RETRO_DENSITY", "0").lower() not in ("1", "true", "yes", "on"):
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
    """Apply a rigid transform (R, t) to every hole's positions and axis.
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
        out.append(
            Hole(id=h.id, axis=new_axis, ref_point=new_ref, sections=new_sections)
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

    from aind_low_point.runtime.loaders import csv_points, voxel_values_at

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
    out = (world, brain_keep, annot_vals, annot_inb)
    _RETRO_VOXEL_CACHE[key] = out
    return out


def _resolve_masked_retro_points(
    runtime, probe_name: str, opts: RetroDensityOpts
) -> np.ndarray:
    """Retro points (corrected scene-LPS) inside the per-probe CCF region AND
    the brain, by direct voxel lookup against the SOURCE volumes."""
    from aind_low_point.runtime.loaders import (
        ccf_region_label_ids,
        ccf_region_membership,
    )

    catalog = runtime.asset_catalog
    sspec = catalog.get_spec(opts.per_probe_mask_fmt.format(probe=probe_name))
    if sspec is None or sspec.source_path is None:
        raise RuntimeError(
            f"--retro-density: structure asset for probe {probe_name!r} missing"
        )
    annot_path = str(sspec.source_path)
    world, brain_keep, annot_vals, annot_inb = _retro_voxel_base(
        runtime, opts, annot_path
    )
    acronym = sspec.metadata.get("ccf_acronym") or sspec.metadata.get("acronym")
    match_ids = ccf_region_label_ids(
        acronym=acronym,
        label_id=sspec.metadata.get("label_id"),
        hemisphere=sspec.metadata.get("hemisphere", "both"),
    )
    keep = ccf_region_membership(
        annot_vals, annot_inb, match_ids, brain_keep=brain_keep
    )
    masked = world[keep]
    if masked.shape[0] == 0:
        raise RuntimeError(
            f"--retro-density: probe {probe_name!r} has zero retro points in "
            f"brain ∩ {acronym!r} — check mask/annotation alignment"
        )
    return masked


def _probe_static_info(
    plan_state,
    runtime,
    name: str,
    retro_opts: RetroDensityOpts | None = None,
) -> ProbeStaticInfo:
    """Build a ProbeStaticInfo for one probe from the runtime."""
    target_points = None
    if retro_opts is not None:
        target_points = _resolve_masked_retro_points(runtime, name, retro_opts)
    context = probe_context_from_runtime(runtime, name, target_points_LPS=target_points)
    sigma = retro_opts.sigma_mm if retro_opts is not None else 0.5
    return ProbeStaticInfo(
        name=name,
        target_LPS=context.target_LPS,
        kind=context.kind,
        shank_tips_local=context.shank_tips_local,
        density_sigma_mm=sigma,
        collision_mesh=context.collision_mesh,
        target_points=target_points,
        coverage_weight=context.coverage_weight,
    )
