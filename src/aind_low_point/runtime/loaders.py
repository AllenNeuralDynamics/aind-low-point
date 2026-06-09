"""Registry-driven file loaders and the GeometryOut type."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import SimpleITK as sitk
import trimesh
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_anatomical_utils.slicer import read_slicer_fcsv
from aind_mri_utils.meshes import mask_to_trimesh
from numpy.typing import NDArray

from aind_low_point.core import Float3, FloatNx3

GeometryOut = Union[
    trimesh.Trimesh,  # surface mesh
    FloatNx3,  # (N,3) points
    dict[str, Float3],  # (name, point) pairs
]

# ---- Registry core -----------------------------------------------------------
_GEOMETRY_LOADER_REGISTRY: dict[str, Callable[..., GeometryOut]] = {}


def register_loader_fn(fn: Callable[..., GeometryOut], name: Optional[str] = None):
    key = name or fn.__name__
    if key in _GEOMETRY_LOADER_REGISTRY:
        raise KeyError(f"Loader '{key}' already registered")
    _GEOMETRY_LOADER_REGISTRY[key] = fn
    return fn


def register_loader(arg: str | Callable[..., GeometryOut] | None = None):
    """Decorator to register a loader function by name."""

    def _wrap(fn: Callable[..., GeometryOut]):
        name = arg.__name__ if callable(arg) else arg
        return register_loader_fn(fn, name)

    if callable(arg):
        return _wrap(arg)
    else:
        return _wrap


def load_geometry(src: Union[str, Path], loader: str, **kwargs) -> GeometryOut:
    """Dispatch to a named loader. kwargs are passed to the loader."""
    fn = _GEOMETRY_LOADER_REGISTRY.get(loader)
    if fn is None:
        raise KeyError(
            (
                f"Unknown loader '{loader}'. Known: "
                f"{', '.join(sorted(_GEOMETRY_LOADER_REGISTRY)) or '(none)'}"
            )
        )
    return fn(str(src), **kwargs)


def trimesh_from_sitk_mask(mask: sitk.Image) -> trimesh.Trimesh:
    """Convert a SimpleITK mask image to a trimesh.

    ``mask_to_trimesh`` (aind_mri_utils >=0.12.1) zero-pads the mask before
    marching cubes, so a mask that reaches the volume boundary (e.g. a brain
    skull-strip filling the field of view) still closes into a watertight mesh.
    No local padding is needed here.
    """
    structure_mesh = mask_to_trimesh(mask)
    trimesh.repair.fix_normals(structure_mesh)
    trimesh.repair.fix_inversion(structure_mesh)
    return structure_mesh


@register_loader
def sitk_volume(path: str) -> trimesh.Trimesh:
    """Read a SimpleITK volume file (.nrrd, .nii, .nii.gz) and mesh it."""
    mask = sitk.ReadImage(path)
    return trimesh_from_sitk_mask(mask)


@register_loader
def load_trimesh_lps(path: str, src_coordinate_system: str = "ASR") -> trimesh.Trimesh:
    """Load a trimesh from a file and convert to LPS."""
    mesh = trimesh.load(path)
    vertices_lps = convert_coordinate_system(
        mesh.vertices, src_coordinate_system, "LPS"
    )
    mesh.vertices = vertices_lps
    return mesh


register_loader_fn(read_slicer_fcsv)


@register_loader("trimesh")
def _load_trimesh(path: str) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh")


@register_loader
def csv_points(path: str, max_points: int | None = None) -> NDArray[np.float64]:
    """Load an (N,3) point cloud from a CSV with x, y, z columns.

    Parameters
    ----------
    max_points
        If set, randomly subsample to this many points.
    """
    import pandas as pd

    df = pd.read_csv(path, index_col=0)
    pts = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if max_points is not None and len(pts) > max_points:
        idx = np.random.default_rng().choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def ccf_region_label_ids(
    *,
    acronym: str | None = None,
    label_id: int | None = None,
    include_descendants: bool = True,
    hemisphere: str = "both",
) -> list[int]:
    """Resolve a CCF region to the annotation-volume label ids that belong to it.

    Shared by ``ccf_annotation_region`` (which *meshes* the matching voxels) and
    the retro point-cloud masker (which tests point *membership* against the same
    ids) so both agree exactly. Applies the lateralized-annotation sign
    convention (IBL: left = negated id).
    """
    if acronym is None and label_id is None:
        raise ValueError("ccf_region_label_ids: must specify acronym or label_id")
    ids: set[int] = set()
    if acronym is not None:
        from aind_low_point.ccf_ontology import CCFOntology

        ontology = CCFOntology.from_bundled()
        if include_descendants:
            descendants = ontology.descendants_of(acronym, include_self=True)
            if not descendants:
                raise KeyError(f"CCF acronym {acronym!r} not in bundled ontology")
            ids.update(s.id for s in descendants)
        else:
            structure = ontology.find_by_acronym(acronym)
            if structure is None:
                raise KeyError(f"CCF acronym {acronym!r} not in bundled ontology")
            ids.add(structure.id)
    if label_id is not None:
        ids.add(int(label_id))
    hemi = hemisphere.lower()
    if hemi in {"both", "b"}:
        return list(ids) + [-i for i in ids]
    if hemi in {"left", "l"}:
        return [-i for i in ids]
    if hemi in {"right", "r"}:
        return list(ids)
    raise ValueError(
        f"hemisphere must be 'left', 'right', or 'both', got {hemisphere!r}"
    )


def voxel_values_at(
    sitk_img: sitk.Image, pts: NDArray[np.float64]
) -> tuple[NDArray[np.generic], NDArray[np.bool_]]:
    """Nearest-voxel value of a SimpleITK volume at physical points ``pts`` (in
    the volume's own physical space). Returns ``(values, in_bounds)`` with
    out-of-bounds points getting value 0.

    Uses SimpleITK's own ``TransformPhysicalPointToIndex`` for the physical→voxel
    mapping (handles origin/spacing/direction correctly) rather than re-deriving
    the affine by hand. The points are typically sampled once per subject and
    cached, so the work is paid once.

    SimpleITK has no batch ``TransformPhysicalPointToIndex``, so the
    physical→index map is built as an affine from the image's own origin /
    spacing / direction and applied to all points at once; the gather then
    reverses (x,y,z)→(z,y,x) for the numpy array in a single place. The affine is
    cross-checked against sitk's per-point transform on a few samples so the
    convention can never silently drift.
    """
    arr = sitk.GetArrayFromImage(sitk_img)  # (Z, Y, X)
    nz, ny, nx = arr.shape
    origin = np.asarray(sitk_img.GetOrigin(), dtype=np.float64)  # (x, y, z)
    spacing = np.asarray(sitk_img.GetSpacing(), dtype=np.float64)  # (x, y, z)
    direction = np.asarray(sitk_img.GetDirection(), dtype=np.float64).reshape(3, 3)
    # physical = origin + direction @ (spacing * index)  ⇒  invert for index
    rel = np.asarray(pts, np.float64) - origin
    cont = (rel @ np.linalg.inv(direction).T) / spacing
    idx = np.rint(cont).astype(np.intp)  # (N, 3) in (x, y, z)
    for k in range(min(5, len(pts))):  # cheap guard against convention drift
        ref = sitk_img.TransformPhysicalPointToIndex(tuple(float(v) for v in pts[k]))
        if tuple(int(v) for v in idx[k]) != tuple(ref):
            raise AssertionError(
                f"voxel index mismatch vs SimpleITK: {idx[k]} != {ref}"
            )
    inb = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < nx)
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < ny)
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < nz)
    )
    vals = np.zeros(len(pts), dtype=arr.dtype)
    g = idx[inb]
    vals[inb] = arr[g[:, 2], g[:, 1], g[:, 0]]  # arr is (z, y, x)
    return vals, inb


def ccf_region_membership(
    label_vals: NDArray[np.generic],
    in_bounds: NDArray[np.bool_],
    match_ids: list[int],
    *,
    brain_keep: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Boolean membership of pre-sampled annotation labels in a CCF region.

    The single membership rule shared by the config reducer (one-shot, via
    :func:`ccf_region_point_mask`) and the optimizer (which samples + caches the
    annotation labels per subject, then calls this per probe). A point belongs to
    the region iff its nearest-voxel label is in ``match_ids`` (from
    :func:`ccf_region_label_ids`), it is in bounds, and — if a ``brain_keep`` mask
    is given — it also lies inside the brain.
    """
    mask = np.isin(label_vals, match_ids) & in_bounds
    if brain_keep is not None:
        mask = mask & brain_keep
    return mask


def ccf_region_point_mask(
    annotation_path: str,
    points: NDArray[np.float64],
    *,
    acronym: str | None = None,
    label_id: int | None = None,
    include_descendants: bool = True,
    hemisphere: str = "both",
    extra_mask_paths: tuple[str, ...] = (),
) -> NDArray[np.bool_]:
    """Which ``points`` lie inside a CCF region by nearest-voxel label lookup.

    ``points`` must be in the same physical frame as the annotation volume at
    ``annotation_path`` (e.g. raw subject LPS mm for a CCF-warped-into-subject
    annotation). Membership is exact and robust to the nonlinear CCF→subject
    warp, unlike geometric containment of a meshed region.

    Each path in ``extra_mask_paths`` (e.g. a brain mask) is sampled the same way
    and AND-ed in as ``> 0`` — a point survives only if it is inside every mask.
    Returns a boolean array aligned row-for-row with ``points``.
    """
    pts = np.asarray(points, dtype=np.float64)
    match_ids = ccf_region_label_ids(
        acronym=acronym,
        label_id=label_id,
        include_descendants=include_descendants,
        hemisphere=hemisphere,
    )
    annot = sitk.ReadImage(annotation_path)
    annot_vals, annot_inb = voxel_values_at(annot, pts)
    brain_keep: NDArray[np.bool_] | None = None
    for mask_path in extra_mask_paths:
        vals, inb = voxel_values_at(sitk.ReadImage(str(mask_path)), pts)
        keep = inb & (vals > 0)
        brain_keep = keep if brain_keep is None else (brain_keep & keep)
    return ccf_region_membership(
        annot_vals, annot_inb, match_ids, brain_keep=brain_keep
    )


@register_loader
def ccf_region_voxel_points(
    path: str,
    *,
    acronym: str | None = None,
    label_id: int | None = None,
    include_descendants: bool = True,
    hemisphere: str = "both",
) -> NDArray[np.float64]:
    """Physical coordinates of every voxel labelled with a CCF region.

    The companion to :func:`ccf_annotation_region` (which *meshes* the matching
    voxels): this returns the matching voxels' physical centres as an ``(N, 3)``
    point cloud in the volume's native frame, so a downstream reducer (e.g.
    ``points_mean``) can take the region's voxel centroid. Selecting the region
    by label is exact and robust to the CCF→subject warp; ``hemisphere`` picks one
    side of a lateralized annotation (left = negated id). See
    :func:`ccf_region_label_ids` for the region/hemisphere semantics.

    The voxel→physical mapping uses the same forward affine as
    :func:`voxel_values_at` (``origin + direction @ (spacing * index)``).
    """
    match_ids = ccf_region_label_ids(
        acronym=acronym,
        label_id=label_id,
        include_descendants=include_descendants,
        hemisphere=hemisphere,
    )
    annotation = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(annotation)  # (Z, Y, X)
    zyx = np.argwhere(np.isin(arr, match_ids))  # (N, 3) in (z, y, x)
    if zyx.shape[0] == 0:
        raise ValueError(
            f"ccf_region_voxel_points: no voxels matched ids={sorted(match_ids)} "
            f"(hemisphere={hemisphere!r}) in {path}"
        )
    idx_xyz = zyx[:, ::-1].astype(np.float64)  # (N, 3) in (x, y, z)
    origin = np.asarray(annotation.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(annotation.GetSpacing(), dtype=np.float64)
    direction = np.asarray(annotation.GetDirection(), dtype=np.float64).reshape(3, 3)
    return origin + (idx_xyz * spacing) @ direction.T


@register_loader
def ccf_annotation_region(
    path: str,
    *,
    acronym: str | None = None,
    label_id: int | None = None,
    include_descendants: bool = True,
    hemisphere: str = "both",
) -> trimesh.Trimesh:
    """Mesh a single CCF region out of a label-mapped annotation volume.

    The volume at ``path`` must be a NIFTI/NRRD where each voxel's
    intensity is its CCF structure id (e.g. ``ccf_annotation_in_subject.nii.gz``
    produced by the AIND ANTs registration pipeline — already warped
    into subject space). Specify the structure either by ``acronym``
    (looked up in the bundled CCF ontology) or by ``label_id``
    directly.

    ``include_descendants=True`` (default) includes voxels labelled
    with any descendant structure of ``acronym`` — typical, since the
    annotation volume's voxels are tagged with leaf-level region IDs
    rather than the parent acronym a user normally types.

    ``hemisphere`` selects one side of a **lateralized** annotation, where
    left-hemisphere voxels carry the *negated* structure id (IBL
    convention, as produced by
    ``aind_registration_utils.annotations.lateralize_and_compact_ccf_image``):

    - ``"both"`` (default) — match ``±id`` (bilateral). On a non-lateralized
      annotation (only positive ids) this is identical to the legacy
      behaviour.
    - ``"left"`` — match ``-id`` only.
    - ``"right"`` — match ``+id`` only.

    Splitting hemispheres at label level is exact and robust to the
    nonlinear CCF→subject warp, unlike a geometric midsagittal cut on the
    meshed region (which collapses for near-midline nuclei).

    Returns the surface mesh of the (binary) thresholded mask, in the
    annotation volume's native frame.
    """
    match_ids = ccf_region_label_ids(
        acronym=acronym,
        label_id=label_id,
        include_descendants=include_descendants,
        hemisphere=hemisphere,
    )

    annotation = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(annotation)
    mask = np.isin(arr, match_ids).astype(np.uint8)
    if not mask.any():
        raise ValueError(
            f"ccf_annotation_region: no voxels matched ids={sorted(match_ids)} "
            f"(hemisphere={hemisphere!r}) in {path}"
        )
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(annotation)
    return trimesh_from_sitk_mask(mask_img)
