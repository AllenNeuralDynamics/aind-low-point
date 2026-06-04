"""Registry-driven point reducers."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import trimesh
from numpy.typing import NDArray

SourceGeo = Union[trimesh.Trimesh, NDArray[np.float64]]
ReduceOut = NDArray[np.float64]  # usually (3,) single point; could be (N,3)

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
    hemisphere: str | None = None,
    plane: float = 0.0,
    **_,
) -> ReduceOut:
    """Mean of ``points`` that lie inside the region mesh ``source``,
    optionally restricted to one LPS hemisphere.

    This is the "average of retro-label points within a target structure"
    target: ``source`` is the structure mesh (the region), ``points`` is the
    retro point cloud (injected by the build step from a ``points_key`` in
    ``reducer_kwargs`` — both must already be in the same raw frame, which they
    are when the structure and the point asset share a transform).

    Parameters
    ----------
    source : trimesh.Trimesh
        The region to test containment against.
    points : (N, 3) array
        Candidate points (same frame as ``source``).
    hemisphere : "left" | "right" | None
        If set, also require the point to be in that LPS hemisphere
        (``x > plane`` is LEFT, matching ``hemisphere_center_mass``).
    plane : float
        Sagittal split plane in raw mm (default 0 = midline).
    """
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError(
            f"points_in_region_center_mass needs a mesh region, got {type(source)}"
        )
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    sel = source.contains(pts)
    if hemisphere is not None:
        hemi = hemisphere.lower()
        if hemi in {"left", "l"}:
            sign = 1.0
        elif hemi in {"right", "r"}:
            sign = -1.0
        else:
            raise ValueError(f"hemisphere must be 'left'/'right', got {hemisphere!r}")
        sel = sel & ((pts[:, 0] - plane) * sign > 0)
    n = int(sel.sum())
    if n == 0:
        raise ValueError(
            "points_in_region_center_mass: no points inside region"
            + (f" (+{hemisphere} hemisphere)" if hemisphere else "")
        )
    return pts[sel].mean(axis=0)


@register_reducer
def hemisphere_center_mass(
    source: SourceGeo,
    *,
    hemisphere: str = "left",
    plane: float = 0.0,
    **_,
) -> ReduceOut:
    """Volumetric centre of mass of one LPS hemisphere of the source mesh.

    In LPS the L axis points to the patient's left, so vertices with
    ``x > plane`` are in the LEFT hemisphere; ``x < plane`` is the right.
    The mesh is sliced at ``x = plane`` (capped) and the resulting
    half-mesh's ``center_mass`` returned. Falls back to a vertex-mean of
    the requested side if slicing fails or yields an empty mesh.

    Parameters
    ----------
    hemisphere
        ``"left"`` / ``"l"`` or ``"right"`` / ``"r"``.
    plane
        Sagittal split plane in LPS mm (default 0 = midline).
    """
    hemi = hemisphere.lower()
    if hemi in {"left", "l"}:
        sign = 1.0
    elif hemi in {"right", "r"}:
        sign = -1.0
    else:
        raise ValueError(f"hemisphere must be 'left' or 'right', got {hemisphere!r}")

    if isinstance(source, trimesh.Trimesh):
        half = source.slice_plane(
            plane_origin=[float(plane), 0.0, 0.0],
            plane_normal=[sign, 0.0, 0.0],
            cap=True,
        )
        if half is not None and len(half.vertices) > 0 and not half.is_empty:
            return np.asarray(half.center_mass, dtype=np.float64)
        verts = np.asarray(source.vertices, dtype=np.float64)
    else:
        verts = np.asarray(source, dtype=np.float64)

    mask = (verts[:, 0] - plane) * sign > 0
    if not mask.any():
        return verts.mean(axis=0)
    return verts[mask].mean(axis=0)
