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
