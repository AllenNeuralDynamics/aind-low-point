"""Registry-driven point reducers."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import trimesh
from numpy.typing import NDArray

SourceGeo = Union[trimesh.Trimesh, NDArray[np.float64]]
ReduceOut = NDArray[np.float64]  # usually (3,) single point; could be (N,3)


class EmptyReductionError(ValueError):
    """A reducer had no input points to reduce over.

    Raised when a point-selecting reducer (e.g. region/hemisphere
    containment) selects zero points, so its output is undefined. The
    runtime build catches this to skip such a derived target rather than
    aborting the whole build — useful for unused contralateral targets in
    subjects with strictly ipsilateral retro labeling.
    """


_REDUCER_REGISTRY: dict[str, Callable[..., ReduceOut]] = {}


def register_reducer_fn(fn: Callable[..., ReduceOut], name: Optional[str] = None):
    key = name or fn.__name__
    if key in _REDUCER_REGISTRY:
        raise KeyError(f"Reducer '{key}' already registered")
    _REDUCER_REGISTRY[key] = fn
    return fn


def register_reducer(arg: str | Callable[..., ReduceOut] | None = None):
    def _wrap(fn: Callable[..., ReduceOut]):
        name = arg.__name__ if callable(arg) else arg
        return register_reducer_fn(fn, name)

    if callable(arg):
        return _wrap(arg)
    else:
        return _wrap


def reduce_target(source: SourceGeo, reducer: str, **kwargs) -> ReduceOut:
    fn = _REDUCER_REGISTRY.get(reducer)
    if fn is None:
        known = ", ".join(sorted(_REDUCER_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown reducer '{reducer}'. Known: {known}")
    return fn(source, **kwargs)


@register_reducer
def mesh_center_mass(source: SourceGeo, **_) -> ReduceOut:
    # Compute the center of mass for the given geometry.
    if isinstance(source, trimesh.Trimesh):
        return np.array(source.center_mass)
    raise TypeError(f"Unsupported source type: {type(source)}")


@register_reducer
def mesh_centroid(source: SourceGeo, **_) -> ReduceOut:
    # Compute the centroid for the given geometry.
    if isinstance(source, trimesh.Trimesh):
        return np.array(source.centroid)
    raise TypeError(f"Unsupported source type: {type(source)}")


@register_reducer
def points_mean(source: SourceGeo, **_) -> ReduceOut:
    """Mean of an ``(N, 3)`` point cloud.

    Pairs with the ``ccf_region_voxel_points`` loader to take a CCF region's
    voxel centroid (the anatomical region-centre target). ``source`` is the voxel
    point cloud (passed positionally by the build step).
    """
    pts = np.asarray(source, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise TypeError(f"points_mean needs an (N, 3) cloud, got {pts.shape}")
    if pts.shape[0] == 0:
        raise EmptyReductionError("points_mean: empty point cloud")
    return pts.mean(axis=0)


@register_reducer
def points_in_region_center_mass(
    _source: SourceGeo,
    *,
    points: NDArray[np.float64],
    annotation_path: str,
    acronym: str | None = None,
    label_id: int | None = None,
    hemisphere: str = "both",
    include_descendants: bool = True,
    brain_mask_paths: tuple[str, ...] = (),
    lookup_points: NDArray[np.float64] | None = None,
    **_,
) -> ReduceOut:
    """Mean of ``points`` whose nearest annotation voxel lies in a CCF region.

    **Frame invariant (see dev memory ``retro_target_native_lookup``):** an
    annotation/voxel lookup must be done in the annotation volume's NATIVE frame
    — with NO transform and NO chemical-shift applied to the query points. The
    chem-shift + scene transform belong on the *result* (moving it into the
    scene), not on the membership query. So this reducer takes two clouds:

    - ``lookup_points`` — the retro cloud in the annotation's native frame (raw,
      pre-canonicalization / pre-chem-shift), used ONLY to decide membership.
    - ``points`` — the SAME cloud in the scene frame (chem-shift applied at load),
      which is what gets averaged. Because chem-shift + the scene transform are
      affine, ``mean(points[sel]) == transform(mean(native[sel]))`` — so selecting
      natively and averaging in-scene yields the correctly-placed centroid, and
      the result matches the optimizer's retro anchor exactly.

    If ``lookup_points`` is omitted, membership falls back to ``points`` (correct
    only when the cloud has identity canonicalization and chem-shift off).
    ``_source`` (the structure mesh) is accepted for the build plumbing but
    ignored — selection is purely label-based.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    lookup = (
        pts if lookup_points is None else np.asarray(lookup_points, dtype=np.float64)
    )
    if lookup.shape != pts.shape:
        raise ValueError(
            f"lookup_points {lookup.shape} must match points {pts.shape}"
        )
    from aind_low_point.runtime.loaders import ccf_region_point_mask

    sel = ccf_region_point_mask(
        annotation_path,
        lookup,
        acronym=acronym,
        label_id=label_id,
        include_descendants=include_descendants,
        hemisphere=hemisphere,
        extra_mask_paths=tuple(brain_mask_paths),
    )
    if not sel.any():
        raise EmptyReductionError(
            "points_in_region_center_mass: no points inside region"
        )
    return pts[sel].mean(axis=0)
