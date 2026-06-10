"""Per-candidate static geometry builder and objective weights."""

from __future__ import annotations

from dataclasses import dataclass

import fcl
import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.enumeration.contracts import (
    ArcAssignment,
    HoleAssignment,
)
from aind_low_point.optimization.geometry import cap_basis
from aind_low_point.optimization.geometry.headstages import make_fcl_bvh
from aind_low_point.optimization.geometry.holes import Hole, threading_margin_mm
from aind_low_point.optimization.geometry.probes import ProbeStaticInfo
from aind_low_point.optimization.geometry.recording import (
    RecordingGeometry,
    get_recording_geometry,
)


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


_SDF_JNP_CACHE: dict[tuple, dict] = {}


def _sdf_jnp_payload(sdf) -> dict:
    """Return cached JAX-array payload for a ``ProbeSDF``."""
    key = (id(sdf),)
    cached = _SDF_JNP_CACHE.get(key)
    if cached is not None:
        return cached
    import jax.numpy as jnp

    payload = dict(
        grid=jnp.asarray(sdf.grid, dtype=jnp.float32),
        origin=jnp.asarray(sdf.origin, dtype=jnp.float32),
        spacing=jnp.asarray(sdf.spacing, dtype=jnp.float32),
        surface=jnp.asarray(sdf.surface_points, dtype=jnp.float32),
        shank_centers=jnp.asarray(sdf.shank_centers, dtype=jnp.float32),
        shank_halves=jnp.asarray(sdf.shank_halves, dtype=jnp.float32),
    )
    _SDF_JNP_CACHE[key] = payload
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
        try:
            geom = get_recording_geometry(p.kind)
        except KeyError:
            geom = fallback_geom
        tips = np.asarray(p.shank_tips_local, dtype=np.float64)
        if tips.shape[0] > 0:
            pivot = np.array(
                [
                    float(tips[:, 0].mean()),
                    float(tips[:, 1].mean()),
                    float(geom.active_center_mm),
                ],
                dtype=np.float64,
            )
        else:
            pivot = np.array([0.0, 0.0, float(geom.active_center_mm)], dtype=np.float64)
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
            )
        )
    return out
