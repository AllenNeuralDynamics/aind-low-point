"""Per-candidate static geometry builder and objective weights."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

import fcl
import numpy as np
from numpy.typing import NDArray

from aind_rutter.domain.probe_kinds import (
    RECORDING_GEOMETRY,
    RecordingGeometry,
    pivot_from_shank_tips,
)
from aind_rutter.optimization.assignment.assignments import (
    ArcAssignment,
    HoleAssignment,
)
from aind_rutter.optimization.geometry import cap_basis
from aind_rutter.optimization.geometry.headstages import make_fcl_bvh
from aind_rutter.optimization.geometry.holes import (
    Hole,
    pack_walls,
    threading_margin_mm,
)
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo


@dataclass(frozen=True)
class JointWeights:
    """Penalty weights shared by the reduced and full optimization stages."""

    lambda_thread: float = 100.0
    lambda_arc_ap: float = 100.0
    lambda_ml: float = 100.0
    lambda_bounds: float = 1.0
    lambda_clearance: float = 100.0
    lambda_coverage: float = 0.0
    lambda_unit_circle: float = 10.0
    comfortable_ap_deg: float = 50.0
    comfortable_ml_deg: float = 50.0
    min_arc_ap_sep_deg: float = 16.0
    min_intra_arc_ml_sep_deg: float = 16.0
    threading_oval_tolerance: float = 0.0
    min_clearance_mm: float = 0.0


@dataclass(frozen=True)
class _ProbeStatic:
    """Pre-built static geometry for one probe under one discrete assignment."""

    name: str
    target_LPS: NDArray
    shank_tips_local: NDArray
    pivot_local: NDArray
    assigned_hole: Hole
    arc_idx: int
    section_axes: NDArray
    section_e1: NDArray
    section_e2: NDArray
    section_centers: NDArray
    section_cos_theta: NDArray
    section_sin_theta: NDArray
    section_a: NDArray
    section_b: NDArray
    bvh_obj: fcl.CollisionObject | None = None
    sdf_data: dict | None = None
    kind: str = ""
    # Assigned hole's wall planes, padded by holes.pack_walls and margin-inset.
    wall_normals: NDArray = field(default_factory=lambda: pack_walls((), 0.0)[0])
    wall_offsets: NDArray = field(default_factory=lambda: pack_walls((), 0.0)[1])


class SDFPayload(dict):
    """A probe's SDF arrays on device.

    A ``dict`` subclass only so it can be weak-referenced: the caches below key
    on ``id`` and have to drop an entry before its object is collected, or a
    later object landing on the same address reads the previous probe's grids.
    """


_SDF_JNP_CACHE: dict[int, SDFPayload] = {}


def _sdf_jnp_payload(sdf) -> SDFPayload:
    """Return the cached JAX-array payload for a ``ProbeSDF``.

    Keyed on identity, because the arrays cannot be hashed and equality on them
    would cost more than the conversion it saves. The entry is dropped by a
    finalizer when the ``ProbeSDF`` is collected, which both bounds the cache
    and closes the window where its id could be reused.
    """
    key = id(sdf)
    cached = _SDF_JNP_CACHE.get(key)
    if cached is not None:
        return cached
    import jax.numpy as jnp

    payload = SDFPayload(
        grid=jnp.asarray(sdf.grid, dtype=jnp.float32),
        origin=jnp.asarray(sdf.origin, dtype=jnp.float32),
        spacing=jnp.asarray(sdf.spacing, dtype=jnp.float32),
        surface=jnp.asarray(sdf.surface_points, dtype=jnp.float32),
        shank_centers=jnp.asarray(sdf.shank_centers, dtype=jnp.float32),
        shank_halves=jnp.asarray(sdf.shank_halves, dtype=jnp.float32),
    )
    if getattr(sdf, "clearance", None) is not None:
        # Host arrays; build_padded_probe_tables stacks them per kind.
        payload["clearance"] = sdf.clearance
    _SDF_JNP_CACHE[key] = payload
    weakref.finalize(sdf, _SDF_JNP_CACHE.pop, key, None)
    return payload


def _build_probe_static(
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    ha: HoleAssignment,
    aa: ArcAssignment,
    bvh_cache: dict[str, fcl.CollisionObject | None] | None = None,
    sdf_by_name: dict | None = None,
) -> list[_ProbeStatic]:
    """Build per-probe static cache for the active optimization pipeline."""
    holes_by_id = {h.id: h for h in holes}
    fallback_geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))
    out: list[_ProbeStatic] = []
    for p in probes:
        # Config first, then the built-in table for the kind, then a
        # stand-in: a caller that builds statics by hand states only a kind.
        geom = p.recording or RECORDING_GEOMETRY.get(p.kind) or fallback_geom
        tips = np.asarray(p.shank_tips_local, dtype=np.float64)
        pivot = (
            np.asarray(p.pivot_local, dtype=np.float64)
            if getattr(p, "pivot_local", None) is not None
            else pivot_from_shank_tips(tips, geom.active_center_mm)
        )
        hole_id = ha.probe_to_hole[p.name]
        arc_idx = aa.probe_to_arc_idx[p.name]
        hole = holes_by_id[hole_id]

        sections = hole.sections
        s_axes = np.array([np.asarray(s.axis, dtype=np.float64) for s in sections])
        s_e1 = np.empty_like(s_axes)
        s_e2 = np.empty_like(s_axes)
        for k, s in enumerate(sections):
            e1, e2 = cap_basis(s.axis)
            s_e1[k] = e1
            s_e2[k] = e2
        s_centers = np.array([np.asarray(s.center, dtype=np.float64) for s in sections])
        s_thetas = np.array([float(s.theta) for s in sections])
        s_a = np.array([float(s.a) for s in sections])
        s_b = np.array([float(s.b) for s in sections])
        margin = threading_margin_mm()
        if margin:
            s_a = np.maximum(s_a - margin, 1e-3)
            s_b = np.maximum(s_b - margin, 1e-3)
        wall_normals, wall_offsets = pack_walls(hole.walls, margin)

        if bvh_cache is not None and p.name in bvh_cache:
            bvh_obj = bvh_cache[p.name]
        else:
            bvh_obj = (
                make_fcl_bvh(p.collision_mesh) if p.collision_mesh is not None else None
            )

        sdf_payload = None
        if sdf_by_name is not None and p.name in sdf_by_name:
            sdf_payload = _sdf_jnp_payload(sdf_by_name[p.name])

        out.append(
            _ProbeStatic(
                name=p.name,
                target_LPS=np.asarray(p.target_LPS, dtype=np.float64),
                shank_tips_local=tips,
                pivot_local=pivot,
                assigned_hole=hole,
                arc_idx=int(arc_idx),
                section_axes=s_axes,
                section_e1=s_e1,
                section_e2=s_e2,
                section_centers=s_centers,
                section_cos_theta=np.cos(s_thetas),
                section_sin_theta=np.sin(s_thetas),
                section_a=s_a,
                section_b=s_b,
                bvh_obj=bvh_obj,
                sdf_data=sdf_payload,
                kind=str(p.kind),
                wall_normals=wall_normals,
                wall_offsets=wall_offsets,
            )
        )
    return out
