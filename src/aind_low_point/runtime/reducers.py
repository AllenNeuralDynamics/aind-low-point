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
        raise ValueError(
            f"hemisphere must be 'left' or 'right', got {hemisphere!r}"
        )

    if isinstance(source, trimesh.Trimesh):
        half = None
        try:
            half = source.slice_plane(
                plane_origin=[float(plane), 0.0, 0.0],
                plane_normal=[sign, 0.0, 0.0],
                cap=True,
            )
        except Exception:
            half = None
        if half is not None and len(half.vertices) > 0 and not half.is_empty:
            return np.asarray(half.center_mass, dtype=np.float64)
        verts = np.asarray(source.vertices, dtype=np.float64)
    else:
        verts = np.asarray(source, dtype=np.float64)

    mask = (verts[:, 0] - plane) * sign > 0
    if not mask.any():
        return verts.mean(axis=0)
    return verts[mask].mean(axis=0)
