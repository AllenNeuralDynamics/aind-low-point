"""Padded, batched per-candidate static data for Stage 2 batched polish.

The existing ``_ProbeStatic`` in ``joint_rerank.py`` is per-(probe, HA, AA)
with shapes that depend on the assigned hole's section count and the
probe's shank count. To run Stage 2 across many candidates in a single
JAX vmap, we need uniform shapes across the batch.

This module builds a ``BatchedProbeStatic`` with all relevant per-probe
data padded to fixed maxima ``(K, SH, S)`` and per-kind SDF tables
indexed by ``sdf_kind_id``. Mask arrays mark padded entries so the
objective can mask their contribution out.

Phase 1 of the batched-Stage-2 refactor — design in
``dev/target_valid_atlas_design.md`` and the conversation log of
2026-05-19.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.arc_assignment import ArcAssignment
from aind_low_point.optimization.geometry import cap_basis
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.optimization.holes import Hole, threading_margin_mm
from aind_low_point.optimization.optimize import ProbeStaticInfo
from aind_low_point.optimization.recording import (
    RecordingGeometry,
    get_recording_geometry,
)


@dataclass(frozen=True)
class BatchedProbeStatic:
    """Padded, batched per-candidate static data.

    Shape conventions:
        B       : batch size (candidates)
        K       : max probes (fixed; here 7 for 836656)
        n_arcs  : max arcs (fixed; here 3)
        S       : max sections per hole (fixed; here 3)
        SH      : max shanks per probe (fixed; here 4)
        N_kinds : number of distinct probe kinds across the batch
        GX,GY,GZ: padded SDF grid dims (max across kinds)
        N_surf  : surface-point count per kind (uniform)

    All arrays are ``jnp`` float32 unless noted. Padded entries are
    masked by ``probe_active_mask``, ``section_mask``, ``shank_mask``.
    """

    # ---- Per-probe geometry ----
    probe_arc_idx: jnp.ndarray  # (B, K) int32 — which arc (0..n_arcs-1)
    probe_active_mask: jnp.ndarray  # (B, K) bool
    probe_target_lps: jnp.ndarray  # (B, K, 3)
    probe_pivot_local: jnp.ndarray  # (B, K, 3)
    probe_shank_tips: jnp.ndarray  # (B, K, SH, 3)
    probe_shank_mask: jnp.ndarray  # (B, K, SH) bool

    # ---- Per-(probe, section) hole geometry ----
    section_axes: jnp.ndarray  # (B, K, S, 3)
    section_e1: jnp.ndarray  # (B, K, S, 3)
    section_e2: jnp.ndarray  # (B, K, S, 3)
    section_centers: jnp.ndarray  # (B, K, S, 3)
    section_cos_theta: jnp.ndarray  # (B, K, S)
    section_sin_theta: jnp.ndarray  # (B, K, S)
    section_a: jnp.ndarray  # (B, K, S)
    section_b: jnp.ndarray  # (B, K, S)
    section_mask: jnp.ndarray  # (B, K, S) bool

    # ---- SDF: indirect via kind table ----
    sdf_kind_id: jnp.ndarray  # (B, K) int32, -1 if no SDF
    sdf_grids: jnp.ndarray  # (N_kinds, GX, GY, GZ) float32
    sdf_grid_shapes: jnp.ndarray  # (N_kinds, 3) int32 — actual (Gx,Gy,Gz)
    sdf_origins: jnp.ndarray  # (N_kinds, 3)
    sdf_spacings: jnp.ndarray  # (N_kinds,)
    sdf_surface_points: jnp.ndarray  # (N_kinds, N_surf, 3)
    # Per-kind analytic shank OBB tables for dual-rep clearance.
    # Shape varies per kind (Sa_k), so kept as Python tuples and
    # looked up by Python-static kind_id at trace time.
    sdf_shank_centers_table: tuple[jnp.ndarray, ...]
    sdf_shank_halves_table: tuple[jnp.ndarray, ...]
    # Uniform padded mirror of the OBB tables (max_Sa across kinds) + validity
    # mask, so a DYNAMIC kind index can gather them (the tuples need static
    # kind_id). Padded rows are masked out in the clearance soft-min.
    sdf_shank_centers_padded: jnp.ndarray  # (N_kinds, max_Sa, 3)
    sdf_shank_halves_padded: jnp.ndarray  # (N_kinds, max_Sa, 3)
    sdf_shank_obb_mask: jnp.ndarray  # (N_kinds, max_Sa) 0/1

    # ---- Optimization bounds (per candidate) ----
    bounds_lo: jnp.ndarray  # (B, n_arcs + 3*K) — (ml, sx, sy)
    bounds_hi: jnp.ndarray  # (B, n_arcs + 3*K)

    # ---- Static dims ----
    K: int
    n_arcs: int
    S: int
    SH: int


def _build_per_kind_sdf_table(sdf_by_name: dict | None, probes: list[ProbeStaticInfo]):
    """Collect distinct SDFs across probes; return per-kind table arrays
    and a ``probe_name → kind_id`` lookup.

    Different probes of the same kind share the same SDF object (or at
    least the same grid contents). We dedup by Python object identity
    of the ``ProbeSDF`` to avoid copying.
    """
    if sdf_by_name is None:
        return None, {}

    seen: dict[int, int] = {}  # id(ProbeSDF) -> kind_id
    name_to_kind: dict[str, int] = {}
    kind_sdfs: list = []
    for p in probes:
        sdf = sdf_by_name.get(p.name)
        if sdf is None:
            continue
        oid = id(sdf)
        if oid not in seen:
            seen[oid] = len(kind_sdfs)
            kind_sdfs.append(sdf)
        name_to_kind[p.name] = seen[oid]

    if not kind_sdfs:
        return None, {}

    # Pad SDF grids to a common shape (max over all kinds)
    max_shape = tuple(max(int(s.grid.shape[d]) for s in kind_sdfs) for d in range(3))
    max_surf = max(int(s.surface_points.shape[0]) for s in kind_sdfs)

    grids = np.zeros((len(kind_sdfs), *max_shape), dtype=np.float32)
    shapes = np.zeros((len(kind_sdfs), 3), dtype=np.int32)
    origins = np.zeros((len(kind_sdfs), 3), dtype=np.float32)
    spacings = np.zeros((len(kind_sdfs),), dtype=np.float32)
    surfs = np.zeros((len(kind_sdfs), max_surf, 3), dtype=np.float32)
    shank_centers: list[jnp.ndarray] = []
    shank_halves: list[jnp.ndarray] = []

    for kid, sdf in enumerate(kind_sdfs):
        gx, gy, gz = sdf.grid.shape
        grids[kid, :gx, :gy, :gz] = sdf.grid.astype(np.float32)
        shapes[kid] = (gx, gy, gz)
        origins[kid] = np.asarray(sdf.origin, dtype=np.float32)
        spacings[kid] = float(sdf.spacing)
        n_surf = sdf.surface_points.shape[0]
        surfs[kid, :n_surf] = sdf.surface_points.astype(np.float32)
        shank_centers.append(jnp.asarray(sdf.shank_centers, dtype=jnp.float32))
        shank_halves.append(jnp.asarray(sdf.shank_halves, dtype=jnp.float32))

    table = dict(
        grids=jnp.asarray(grids),
        shapes=jnp.asarray(shapes),
        origins=jnp.asarray(origins),
        spacings=jnp.asarray(spacings),
        surfs=jnp.asarray(surfs),
        shank_centers=tuple(shank_centers),
        shank_halves=tuple(shank_halves),
    )
    return table, name_to_kind


def _ap_bounds_deg(head_pitch_deg: float) -> tuple[float, float]:
    """Per-arc AP bounds in degrees: the ±75° kinematic AP range expressed
    around the head pitch (matches ``PoseLimits.ap_deg``)."""
    return -75.0 + head_pitch_deg, 75.0 + head_pitch_deg


def build_batched_probe_static(
    candidates: list[tuple[HoleAssignment, ArcAssignment]],
    probes: list[ProbeStaticInfo],
    holes: list[Hole],
    *,
    n_arcs: int = 3,
    S: int = 3,
    SH: int = 4,
    sdf_by_name: dict | None = None,
    head_pitch_deg: float = 0.0,
) -> BatchedProbeStatic:
    """Build the padded, batched static for a list of candidates.

    Parameters
    ----------
    candidates : list of (HoleAssignment, ArcAssignment)
        Each entry is one Stage 2 candidate.
    probes : list[ProbeStaticInfo]
        Probe list (same for all candidates).
    holes : list[Hole]
        Hole list (same for all candidates); referenced by id via
        ``ha.probe_to_hole``.
    n_arcs, S, SH : int
        Padding dimensions. Caller ensures these meet the actual maxes
        across the batch.
    sdf_by_name : optional dict ``probe_name → ProbeSDF``
        When provided, the SDF table is built and ``sdf_kind_id`` is
        populated.
    head_pitch_deg : float
        Used in arc-AP bounds computation.
    """
    B = len(candidates)
    K = len(probes)
    holes_by_id = {h.id: h for h in holes}
    fallback_geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))

    # ---- Per-probe static fields shared across candidates ----
    # (target, kind, shank tips, pivot are the same for a given probe
    # regardless of the candidate's HA/AA)
    probe_target = np.zeros((K, 3), dtype=np.float32)
    probe_pivot = np.zeros((K, 3), dtype=np.float32)
    probe_tips_padded = np.zeros((K, SH, 3), dtype=np.float32)
    probe_shank_mask_per_k = np.zeros((K, SH), dtype=bool)
    for i, p in enumerate(probes):
        try:
            geom = get_recording_geometry(p.kind)
        except KeyError:
            geom = fallback_geom
        tips = np.asarray(p.shank_tips_local, dtype=np.float32)
        if tips.shape[0] > 0:
            pivot = np.array(
                [
                    float(tips[:, 0].mean()),
                    float(tips[:, 1].mean()),
                    float(geom.active_center_mm),
                ],
                dtype=np.float32,
            )
        else:
            pivot = np.array([0.0, 0.0, float(geom.active_center_mm)], dtype=np.float32)
        probe_target[i] = np.asarray(p.target_LPS, dtype=np.float32)
        probe_pivot[i] = pivot
        nsh = min(tips.shape[0], SH)
        probe_tips_padded[i, :nsh] = tips[:nsh]
        probe_shank_mask_per_k[i, :nsh] = True

    # SDF kind table (shared across batch)
    sdf_table, name_to_kind = _build_per_kind_sdf_table(sdf_by_name, probes)

    # ---- Per-candidate fields ----
    probe_arc_idx = np.zeros((B, K), dtype=np.int32)
    probe_active_mask = np.ones((B, K), dtype=bool)

    section_axes_np = np.zeros((B, K, S, 3), dtype=np.float32)
    section_e1_np = np.zeros((B, K, S, 3), dtype=np.float32)
    section_e2_np = np.zeros((B, K, S, 3), dtype=np.float32)
    section_centers_np = np.zeros((B, K, S, 3), dtype=np.float32)
    section_cos_np = np.zeros((B, K, S), dtype=np.float32)
    section_sin_np = np.zeros((B, K, S), dtype=np.float32)
    section_a_np = np.zeros((B, K, S), dtype=np.float32)
    section_b_np = np.zeros((B, K, S), dtype=np.float32)
    section_mask_np = np.zeros((B, K, S), dtype=bool)

    sdf_kind_id_np = -np.ones((B, K), dtype=np.int32)

    n_vars = n_arcs + 3 * K  # (ml, sx, sy) per probe under Patch B
    bounds_lo = np.zeros((B, n_vars), dtype=np.float32)
    bounds_hi = np.zeros((B, n_vars), dtype=np.float32)
    ap_lo, ap_hi = _ap_bounds_deg(head_pitch_deg)
    # ML / (sx, sy) bounds. Spin is parameterized as a 2D unit-circle
    # vector to remove the ±180° angle wrap; each component bounded
    # to ±1.5 (loose around the unit circle).
    ml_lo, ml_hi = -45.0, 45.0
    sxy_lo, sxy_hi = -1.5, 1.5

    _t_margin = threading_margin_mm()
    for b, (ha, aa) in enumerate(candidates):
        # Per-arc AP bounds
        for a in range(n_arcs):
            bounds_lo[b, a] = ap_lo
            bounds_hi[b, a] = ap_hi
        # Per-probe ml + (sx, sy) bounds + arc index + section data
        for i, p in enumerate(probes):
            bounds_lo[b, n_arcs + 3 * i] = ml_lo
            bounds_hi[b, n_arcs + 3 * i] = ml_hi
            bounds_lo[b, n_arcs + 3 * i + 1] = sxy_lo
            bounds_hi[b, n_arcs + 3 * i + 1] = sxy_hi
            bounds_lo[b, n_arcs + 3 * i + 2] = sxy_lo
            bounds_hi[b, n_arcs + 3 * i + 2] = sxy_hi
            probe_arc_idx[b, i] = aa.probe_to_arc_idx.get(p.name, 0)

            hole_id = ha.probe_to_hole.get(p.name)
            if hole_id is None:
                continue
            hole = holes_by_id[hole_id]
            sections = hole.sections
            ns = min(len(sections), S)
            for s_idx in range(ns):
                sec = sections[s_idx]
                axis = np.asarray(sec.axis, dtype=np.float32)
                e1, e2 = cap_basis(np.asarray(sec.axis, dtype=np.float64))
                section_axes_np[b, i, s_idx] = axis
                section_e1_np[b, i, s_idx] = e1.astype(np.float32)
                section_e2_np[b, i, s_idx] = e2.astype(np.float32)
                section_centers_np[b, i, s_idx] = np.asarray(
                    sec.center, dtype=np.float32
                )
                section_cos_np[b, i, s_idx] = float(np.cos(sec.theta))
                section_sin_np[b, i, s_idx] = float(np.sin(sec.theta))
                section_a_np[b, i, s_idx] = max(float(sec.a) - _t_margin, 1e-3)
                section_b_np[b, i, s_idx] = max(float(sec.b) - _t_margin, 1e-3)
                section_mask_np[b, i, s_idx] = True

            if p.name in name_to_kind:
                sdf_kind_id_np[b, i] = name_to_kind[p.name]

    # Broadcast per-probe-only fields to (B, K, ...)
    probe_target_lps = np.broadcast_to(probe_target[None], (B, K, 3)).copy()
    probe_pivot_local = np.broadcast_to(probe_pivot[None], (B, K, 3)).copy()
    probe_shank_tips = np.broadcast_to(probe_tips_padded[None], (B, K, SH, 3)).copy()
    probe_shank_mask = np.broadcast_to(probe_shank_mask_per_k[None], (B, K, SH)).copy()

    # SDF table (placeholders if no SDFs)
    if sdf_table is None:
        _n_kinds = 0
        sdf_grids = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
        sdf_grid_shapes = jnp.zeros((1, 3), dtype=jnp.int32)
        sdf_origins = jnp.zeros((1, 3), dtype=jnp.float32)
        sdf_spacings = jnp.zeros((1,), dtype=jnp.float32)
        sdf_surface_points = jnp.zeros((1, 1, 3), dtype=jnp.float32)
        sdf_shank_centers_table: tuple[jnp.ndarray, ...] = ()
        sdf_shank_halves_table: tuple[jnp.ndarray, ...] = ()
    else:
        _n_kinds = int(sdf_table["grids"].shape[0])
        sdf_grids = sdf_table["grids"]
        sdf_grid_shapes = sdf_table["shapes"]
        sdf_origins = sdf_table["origins"]
        sdf_spacings = sdf_table["spacings"]
        sdf_surface_points = sdf_table["surfs"]
        sdf_shank_centers_table = sdf_table["shank_centers"]
        sdf_shank_halves_table = sdf_table["shank_halves"]

    # Uniform padded mirror of the per-kind OBB tables (+ validity mask), for
    # dynamic-kind-index gather. Halves padded to 1.0 (a valid box; masked out
    # anyway), centers to 0.0. Built from the ragged tuples so it can't drift.
    if sdf_shank_centers_table:
        max_sa = (
            max(
                (int(np.asarray(c).shape[0]) for c in sdf_shank_centers_table),
                default=0,
            )
            or 1
        )
        nk = len(sdf_shank_centers_table)
        cen_np = np.zeros((nk, max_sa, 3), dtype=np.float32)
        hlv_np = np.ones((nk, max_sa, 3), dtype=np.float32)
        obbm_np = np.zeros((nk, max_sa), dtype=np.float32)
        for k, (c, h) in enumerate(
            zip(sdf_shank_centers_table, sdf_shank_halves_table)
        ):
            s = int(np.asarray(c).shape[0])
            if s:
                cen_np[k, :s] = np.asarray(c)
                hlv_np[k, :s] = np.asarray(h)
                obbm_np[k, :s] = 1.0
        sdf_shank_centers_padded = jnp.asarray(cen_np)
        sdf_shank_halves_padded = jnp.asarray(hlv_np)
        sdf_shank_obb_mask = jnp.asarray(obbm_np)
    else:
        sdf_shank_centers_padded = jnp.zeros((1, 1, 3), dtype=jnp.float32)
        sdf_shank_halves_padded = jnp.ones((1, 1, 3), dtype=jnp.float32)
        sdf_shank_obb_mask = jnp.zeros((1, 1), dtype=jnp.float32)

    return BatchedProbeStatic(
        probe_arc_idx=jnp.asarray(probe_arc_idx),
        probe_active_mask=jnp.asarray(probe_active_mask),
        probe_target_lps=jnp.asarray(probe_target_lps),
        probe_pivot_local=jnp.asarray(probe_pivot_local),
        probe_shank_tips=jnp.asarray(probe_shank_tips),
        probe_shank_mask=jnp.asarray(probe_shank_mask),
        section_axes=jnp.asarray(section_axes_np),
        section_e1=jnp.asarray(section_e1_np),
        section_e2=jnp.asarray(section_e2_np),
        section_centers=jnp.asarray(section_centers_np),
        section_cos_theta=jnp.asarray(section_cos_np),
        section_sin_theta=jnp.asarray(section_sin_np),
        section_a=jnp.asarray(section_a_np),
        section_b=jnp.asarray(section_b_np),
        section_mask=jnp.asarray(section_mask_np),
        sdf_kind_id=jnp.asarray(sdf_kind_id_np),
        sdf_grids=sdf_grids,
        sdf_grid_shapes=sdf_grid_shapes,
        sdf_origins=sdf_origins,
        sdf_spacings=sdf_spacings,
        sdf_surface_points=sdf_surface_points,
        sdf_shank_centers_table=sdf_shank_centers_table,
        sdf_shank_halves_table=sdf_shank_halves_table,
        sdf_shank_centers_padded=sdf_shank_centers_padded,
        sdf_shank_halves_padded=sdf_shank_halves_padded,
        sdf_shank_obb_mask=sdf_shank_obb_mask,
        bounds_lo=jnp.asarray(bounds_lo),
        bounds_hi=jnp.asarray(bounds_hi),
        K=K,
        n_arcs=n_arcs,
        S=S,
        SH=SH,
    )


def initial_y_from_aa(
    candidates: list[tuple[HoleAssignment, ArcAssignment]],
    probes: list[ProbeStaticInfo],
    *,
    n_arcs: int = 3,
) -> NDArray:
    """Construct initial y vector for each candidate from its
    ``ArcAssignment.arc_centroids_deg`` and zero ml/spin.

    Shape: ``(B, n_arcs + 2*K)``. Phase 3 will replace this with a
    proper warm-start builder that mirrors ``_build_starts``.
    """
    B = len(candidates)
    K = len(probes)
    # (ml, sx, sy) per probe under Patch B; default spin = 0° → (sx, sy) = (1, 0).
    y0 = np.zeros((B, n_arcs + 3 * K), dtype=np.float32)
    for b, (_ha, aa) in enumerate(candidates):
        ap_seq = aa.arc_centroids_deg
        for a in range(min(n_arcs, len(ap_seq))):
            y0[b, a] = float(ap_seq[a])
        for k in range(K):
            y0[b, n_arcs + 3 * k + 1] = 1.0  # sx
            y0[b, n_arcs + 3 * k + 2] = 0.0  # sy
    return y0
