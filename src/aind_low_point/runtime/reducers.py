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
def points_in_region_center_mass(
    source: SourceGeo,
    *,
    points: NDArray[np.float64],
    **_,
) -> ReduceOut:
    """Mean of ``points`` that lie inside the region mesh ``source``.

    This is the "average of retro-label points within a target structure"
    target: ``source`` is the structure mesh (the region), ``points`` is the
    retro point cloud (injected by the build step from a ``points_key`` in
    ``reducer_kwargs`` — both must already be in the same raw frame, which they
    are when the structure and the point asset share a transform).

    Hemisphere selection is handled at the loader level via
    ``DerivedTargetSpecModel.hemisphere`` (which re-loads the source mesh with
    ``ccf_annotation_region(hemisphere=...)`` before this reducer runs), so no
    geometric x > 0 cut is needed or supported here.
    """
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError(
            f"points_in_region_center_mass needs a mesh region, got {type(source)}"
        )
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    sel = source.contains(pts)
    n = int(sel.sum())
    if n == 0:
        raise EmptyReductionError(
            "points_in_region_center_mass: no points inside region"
        )
    return pts[sel].mean(axis=0)
