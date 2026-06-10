"""Extract oriented hole specs from an implant mesh.

For each hole the tool produces:
    - axis_LPS    : unit vector along the bore (per-hole — bores need
                    not share the same axis).
    - sections    : list of {s_mm, center_LPS, a_mm, b_mm, theta_rad}
                    sampled at planes perpendicular to *that hole's*
                    axis. ``s_mm`` is the signed offset along axis_LPS
                    from the section nearest to the implant top.

Algorithm (topology-based, watertight-mesh-only)
------------------------------------------------
1. Canonicalize the input mesh (default ASR -> LPS).
2. For every face, cast an "outer ray" along ``+normal`` (away from
   solid material — into the bore hollow for bore walls; into the
   world for skirt/plate faces). Record the first-hit distance.
   - Bore-wall faces: ray crosses the bore hollow, hits the opposite
     wall of the *same* bore at distance ≈ bore diameter (~1 mm).
   - Skirt / plate-top / plate-bottom faces: ray escapes to infinity.
3. Filter to faces with finite outer-ray distance below a threshold
   (default 1.3 mm). These are the bore walls; everything else is
   discarded.
4. Build a graph on bore-wall faces with edges from
   (a) each outer-ray pair (within-bore by construction) and
   (b) face adjacency restricted to bore-wall faces.
   Connected components ⇒ individual bores. The mesh's genus equals
   the bore count, and outer-ray pairs cleanly connect each bore's
   wall halves; no merging heuristics needed.
5. For each bore: fit axis as the eigenvector of ``Σ n nᵀ`` with the
   smallest eigenvalue. Sample sections perpendicular to that axis at
   top/mid/bottom of the wall-vertex extent and fit oriented oval
   cross-sections.

Output
------
YAML with a ``holes`` list:
    holes:
      - id: 0
        axis_LPS: [...]
        ref_point_LPS: [...]
        sections:
          - {s_mm, center_LPS, a_mm, b_mm, theta_rad}
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
import yaml
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from scipy.sparse.csgraph import connected_components

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Section:
    s_mm: float  # signed offset along axis from a reference point
    center: np.ndarray  # 3D center of the inner loop (LPS-mm)
    a_mm: float  # half-extent of fitted oval major axis
    b_mm: float  # half-extent of fitted oval minor axis
    theta_rad: float  # rotation in the perpendicular plane


@dataclass
class Hole:
    axis: np.ndarray  # unit vector (LPS)
    ref_point: np.ndarray  # any point on the axis (LPS)
    sections: list[Section]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _hole_rings_3d(mesh, origin, axis_normal):
    """Slice perpendicular to ``axis_normal`` and return all rings
    (interior loops or small fragment exteriors) as 3D LPS-mm vertex
    arrays.
    """
    sec = mesh.section(plane_origin=origin, plane_normal=axis_normal)
    if sec is None:
        return []
    p2d, to_3d = sec.to_2D(normal=axis_normal)
    polys = list(p2d.polygons_full)
    if not polys:
        return []
    areas = np.asarray([p.area for p in polys])
    largest_idx = int(np.argmax(areas))
    largest = polys[largest_idx]
    other_max = (
        float(areas[np.arange(len(polys)) != largest_idx].max() or 0.0)
        if len(polys) > 1
        else 0.0
    )

    rings_xy: list[np.ndarray] = []
    if list(largest.interiors) and (
        areas[largest_idx] >= 5.0 * other_max or len(polys) == 1
    ):
        for ring in largest.interiors:
            rings_xy.append(np.asarray(ring.coords)[:, :2])
    else:
        for p in polys:
            if p.area < 0.005:  # noise speck
                continue
            rings_xy.append(np.asarray(p.exterior.coords)[:, :2])

    out = []
    for xy in rings_xy:
        xyz1 = np.column_stack([xy, np.zeros(len(xy)), np.ones(len(xy))])
        xyz_world = (to_3d @ xyz1.T).T[:, :3]
        out.append(xyz_world)
    return out


def _fit_oval_in_plane(points_3d, axis):
    """Fit oriented oval (a, b, theta) to a 3D ring whose plane normal
    is ``axis``. Returns ``(center, a_mm, b_mm, theta_rad, e1, e2)``.
    a_mm, b_mm are half-extents perpendicular to the axis. ``theta_rad``
    is the angle of the major axis from ``e1`` in the (e1, e2) basis.
    """
    p = np.asarray(points_3d)
    center = p.mean(axis=0)
    a = np.asarray(axis) / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, a)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    rel = p - center
    uv = np.column_stack([rel @ e1, rel @ e2])

    angles = np.deg2rad(np.arange(0, 90, 0.5))
    best_area = np.inf
    best_theta = 0.0
    best_extent = (0.0, 0.0)
    for ang in angles:
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s], [s, c]])
        ruv = uv @ rot.T
        ext = ruv.max(axis=0) - ruv.min(axis=0)
        area = ext[0] * ext[1]
        if area < best_area:
            best_area = area
            best_theta = ang
            best_extent = (ext[0] / 2, ext[1] / 2)
    a_mm, b_mm = sorted(best_extent, reverse=True)
    # Major-axis angle in the original (e1, e2) frame: when ``best_theta``
    # is the rotation we applied that aligns the major with u' axis,
    # the original major lies at -best_theta; when it aligns with v',
    # the original major lies at π/2 - best_theta.
    if best_extent[0] >= best_extent[1]:
        theta_final = -best_theta
    else:
        theta_final = np.pi / 2 - best_theta
    return center, float(a_mm), float(b_mm), float(theta_final), e1, e2


# ---------------------------------------------------------------------------
# Topology-based bore detection (watertight mesh, outer-ray SDF)
# ---------------------------------------------------------------------------


def _outer_ray_pairs(mesh: trimesh.Trimesh, ray_offset: float = 5e-4):
    """Cast a ray from each face's centroid along +normal (outward
    from the mesh's solid interior). Return per-face arrays:
      - ``dist[i]``: distance to the first hit (np.inf if escapes)
      - ``partner[i]``: face id of the first hit (-1 if escapes)

    For a watertight mesh whose normals point outward from the solid,
    this is the bore-hollow ray:
      - bore-wall faces' rays cross the hollow and hit the opposite
        wall of the same bore at distance ≈ bore diameter
      - all other faces escape to infinity
    """
    fn = mesh.face_normals
    fc = mesh.triangles_center
    n = len(mesh.faces)
    origins = fc + ray_offset * fn
    locs, ray_idx, hit_face = mesh.ray.intersects_location(
        origins, fn, multiple_hits=False
    )
    dist = np.full(n, np.inf)
    partner = -np.ones(n, dtype=np.int64)
    for ri, hf, loc in zip(ray_idx, hit_face, locs):
        d = float(np.linalg.norm(loc - origins[ri]))
        if d < dist[ri]:
            dist[ri] = d
            partner[ri] = int(hf)
    return dist, partner


def _bore_components(
    mesh: trimesh.Trimesh,
    *,
    max_outer_ray_mm: float,
    ray_offset_mm: float,
):
    """Identify per-bore face sets via the outer-ray SDF + adjacency.

    Returns a list of boolean face masks, one per bore.
    """
    n = len(mesh.faces)
    dist, partner = _outer_ray_pairs(mesh, ray_offset=ray_offset_mm)

    # Bore-wall faces: those with a finite, short outer-ray hit.
    is_wall = (dist > 0) & (dist < max_outer_ray_mm)

    # Edges:
    # 1. Outer-ray pairs (within-bore by construction)
    pair_edges: list[tuple[int, int]] = []
    for i in np.where(is_wall)[0]:
        j = int(partner[i])
        if j >= 0 and is_wall[j]:
            pair_edges.append((int(i), j))
    pair_edges_arr = (
        np.asarray(pair_edges, dtype=np.int64)
        if pair_edges
        else np.zeros((0, 2), dtype=np.int64)
    )

    # 2. Face adjacency, restricted to bore-wall faces. Adjacent strips
    #    of one bore wall are reliably connected here (no dihedral
    #    cutoff needed because we're already restricted to bore walls
    #    via the SDF gate).
    fa = mesh.face_adjacency
    adj_mask = is_wall[fa[:, 0]] & is_wall[fa[:, 1]]
    adj_edges = fa[adj_mask]

    if len(pair_edges_arr) + len(adj_edges) == 0:
        return [], dist

    edges = np.concatenate([pair_edges_arr, adj_edges], axis=0)
    data = np.ones(len(edges), dtype=bool)
    g = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n))
    g = g + g.T
    _, labels = connected_components(g, directed=False)
    masks: list[np.ndarray] = []
    for k in np.unique(labels[is_wall]):
        m = (labels == k) & is_wall
        if m.sum() < 1:
            continue
        masks.append(m)
    return masks, dist


def _component_axis(face_normals: np.ndarray) -> np.ndarray:
    """Bore axis = direction perpendicular to all wall-face normals
    (eigenvector of Σ n nᵀ with the smallest eigenvalue)."""
    M = np.einsum("ij,ik->jk", face_normals, face_normals)
    _, evecs = np.linalg.eigh(M)
    return evecs[:, 0]


def _bore_axis_extent(
    mesh: trimesh.Trimesh,
    fmask: np.ndarray,
    axis: np.ndarray,
    center: np.ndarray,
    *,
    margin: float = 0.0,
) -> tuple[float, float]:
    """Return (s_min, s_max) of bore-wall vertex projections onto the
    bore axis, measured from ``center``. Inset by ``margin`` so the
    sampled sections sit safely *inside* the wall extent rather than
    skimming its edges."""
    face_verts = mesh.faces[fmask].ravel()
    verts = mesh.vertices[np.unique(face_verts)]
    s = (verts - center) @ axis
    s_min, s_max = float(s.min()), float(s.max())
    if margin > 0 and s_max - s_min > 2 * margin:
        s_min += margin
        s_max -= margin
    return s_min, s_max


def _narrow_bore_span(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    axis: np.ndarray,
    *,
    narrow_mm: float = 0.40,
    flat_delta_mm: float = 0.03,
    step_mm: float = 0.02,
    reach_mm: float = 1.6,
) -> tuple[float, float] | None:
    """Find the constant-diameter channel span ``(s_bottom, s_flare)`` of a bore
    that flares into a chamfered mouth.

    Scans cross-sections perpendicular to ``axis`` at ``center + s*axis`` and
    records the nearest wall distance ``r(s)`` (the inscribed minor of the bore
    ring nearest the axis). ``r`` is large below the implant (exit), flat through
    the constant channel, and rises through the chamfer. Returns:

    - ``s_bottom`` = inferior end of the longest contiguous ``r <= narrow_mm``
      run (the implant bottom — ``r`` jumps to the exit zone just below it).
    - ``s_flare`` = superior end where ``r`` first departs the flat channel value
      by ``flat_delta_mm`` (the chamfer onset / top of the narrow inner bore).

    ``None`` if no narrow run is found (caller falls back to the wall extent).
    """
    ss = np.arange(-reach_mm, reach_mm, step_mm)
    r: list[float | None] = []
    for s in ss:
        o = center + s * axis
        sec = mesh.section(plane_origin=o, plane_normal=axis)
        rmin = None
        if sec is not None:
            cand = [
                float(np.linalg.norm(p - o, axis=1).min())
                for p in sec.discrete
                if np.linalg.norm(p.mean(0) - o) < 1.0
            ]
            if cand:
                rmin = min(cand)
        r.append(rmin)
    nar = [rv is not None and rv <= narrow_mm for rv in r]
    best_len, best = 0, None
    i = 0
    while i < len(ss):
        if nar[i]:
            j = i
            while j + 1 < len(ss) and nar[j + 1]:
                j += 1
            if j - i > best_len:
                best_len, best = j - i, (i, j)
            i = j + 1
        else:
            i += 1
    if best is None:
        return None
    i0, i1 = best
    run_r = [v for k in range(i0, i1 + 1) if (v := r[k]) is not None]
    rflat = min(run_r)
    s_bottom = float(ss[i0])
    s_flare = max(
        float(ss[k])
        for k in range(i0, i1 + 1)
        if (rk := r[k]) is not None and rk <= rflat + flat_delta_mm
    )
    return s_bottom, s_flare


def _section_at(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    axis: np.ndarray,
    *,
    s: float,
    max_ring_radius: float,
    max_xy_drift: float = 0.5,
) -> Section | None:
    """Slice perpendicular to ``axis`` at ``center + s * axis``. Pick
    the ring whose centroid is closest to the bore axis line (within
    ``max_ring_radius``) AND whose xy projection is close to the
    bore's xy center (within ``max_xy_drift``)."""
    origin = center + s * axis
    rings = _hole_rings_3d(mesh, origin, axis)
    if not rings:
        return None
    best = None
    best_d = np.inf
    for r in rings:
        rc = r.mean(axis=0)
        if np.linalg.norm(rc[:2] - center[:2]) > max_xy_drift:
            continue
        d = float(np.linalg.norm(np.cross(rc - center, axis)))
        if d < best_d:
            best_d = d
            best = r
    if best is None or best_d > max_ring_radius:
        return None
    c, a_mm, b_mm, theta, _, _ = _fit_oval_in_plane(best, axis)
    return Section(
        s_mm=float(s),
        center=c,
        a_mm=a_mm,
        b_mm=b_mm,
        theta_rad=theta,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def extract_holes(  # noqa: C901
    mesh: trimesh.Trimesh,
    *,
    axis_global: np.ndarray = np.array([0.0, 0.0, 1.0]),
    max_outer_ray_mm: float = 1.3,
    ray_offset_mm: float = 5e-4,
    min_face_count: int = 6,
    n_sections: int = 2,
    max_ring_radius: float = 1.0,
    section_inset_mm: float = 0.05,
    max_section_a: float = 1.0,
    min_section_a: float = 0.10,
    max_tilt_deg: float = 80.0,
    channel_a: float | None = None,
    channel_b: float | None = None,
    narrow_bore_span: bool = False,
) -> list[Hole]:
    if not mesh.is_watertight:
        print("warning: mesh is not watertight — outer-ray SDF may misclassify faces")

    axis_global = np.asarray(axis_global, dtype=float) / np.linalg.norm(axis_global)

    masks, _ = _bore_components(
        mesh,
        max_outer_ray_mm=max_outer_ray_mm,
        ray_offset_mm=ray_offset_mm,
    )
    masks = [m for m in masks if int(m.sum()) >= min_face_count]
    print(f"Found {len(masks)} bore(s)")

    fn = mesh.face_normals
    fc = mesh.triangles_center
    cos_max_tilt = np.cos(np.deg2rad(max_tilt_deg))
    holes: list[Hole] = []
    skipped: list[str] = []
    for bid, fmask in enumerate(masks):
        nf = int(fmask.sum())
        ctr = fc[fmask].mean(0)
        axis = _component_axis(fn[fmask])
        if np.dot(axis, axis_global) < 0:
            axis = -axis

        if float(axis @ axis_global) < cos_max_tilt:
            tilt = np.degrees(np.arccos(float(abs(axis @ axis_global))))
            skipped.append(
                f"  bore #{bid}: skip (tilt {tilt:.1f}° > {max_tilt_deg:.0f}°)"
            )
            continue

        # Section sampling range. Default = the full bore-wall vertex extent.
        # With narrow_bore_span, restrict to the constant-diameter channel
        # (implant bottom -> chamfer onset) so sections sample only the narrow
        # inner bore, not the flared mouth (whose wide cross-section inflates
        # the fit and over-constrains the entrance).
        if narrow_bore_span:
            span = _narrow_bore_span(mesh, ctr, axis)
            if span is None:
                skipped.append(f"  bore #{bid}: skip (no narrow-bore span)")
                continue
            s_min, s_max = span  # (s_bottom inferior, s_flare superior)
            # Inset inferior end by the same margin used in _bore_axis_extent so
            # _section_at samples safely away from the mesh edge (where the bore
            # ring may be incomplete and the wrong ring can be selected).
            s_min += section_inset_mm
        else:
            s_min, s_max = _bore_axis_extent(
                mesh, fmask, axis, ctr, margin=section_inset_mm
            )
        if (s_max - s_min) <= 0:
            skipped.append(f"  bore #{bid}: skip (zero axial extent)")
            continue

        s_samples = np.linspace(s_max, s_min, n_sections)
        sections: list[Section] = []
        for s in s_samples:
            sec = _section_at(
                mesh,
                ctr,
                axis,
                s=float(s),
                max_ring_radius=max_ring_radius,
            )
            if sec is None:
                continue
            if sec.a_mm > max_section_a:
                continue
            sections.append(sec)
        if not sections:
            skipped.append(f"  bore #{bid}: skip (no clean rings)")
            continue
        if max(s.a_mm for s in sections) < min_section_a:
            skipped.append(
                f"  bore #{bid}: skip (max a_mm={max(s.a_mm for s in sections):.3f}"
                f" < {min_section_a:.2f}; sliver)"
            )
            continue
        sections.sort(key=lambda s: -s.s_mm)
        # Spec override: the deep channel is a known constant 1.2 x 0.6 mm oval
        # (a=0.60, b=0.30). The per-section a/b fitted from this non-watertight,
        # chamfer-flared mesh are unreliable (minor inflated up to +0.09 mm), so
        # keep the fitted center/axis/theta but replace a/b with the spec. The
        # fitted a/b are still used above to reject garbage rings.
        if channel_a is not None and channel_b is not None:
            sections = [replace(s, a_mm=channel_a, b_mm=channel_b) for s in sections]
        # Fit theta once from all bore-wall face-centroids in the span (the same
        # dataset used for the axis estimate). More stable than per-section ring
        # fitting: uses many more points and avoids slice-level mesh noise.
        s_bore = (fc[fmask] - ctr) @ axis
        if narrow_bore_span:
            in_span = (s_bore >= s_min) & (s_bore <= s_max)
        else:
            in_span = np.ones(int(fmask.sum()), dtype=bool)
        span_fc = fc[fmask][in_span]
        if span_fc.shape[0] >= 4:
            _, _, _, theta_bore, _, _ = _fit_oval_in_plane(span_fc, axis)
            sections = [replace(s, theta_rad=theta_bore) for s in sections]
        holes.append(Hole(axis=axis, ref_point=ctr, sections=sections))
        s0 = sections[0]
        print(
            f"  bore #{bid}: nf={nf:>3}  "
            f"ctr=({ctr[0]:+.2f},{ctr[1]:+.2f},{ctr[2]:+.2f}) "
            f"axis=({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f})  "
            f"a={s0.a_mm:.3f} b={s0.b_mm:.3f}  "
            f"len={s_max - s_min:.3f} ({len(sections)} sections)"
        )
    if skipped:
        print("Skipped:")
        for line in skipped:
            print(line)
    return holes


# Reference (LPS_x, LPS_y) per diagram-id for the 14-hole `0283-300-04`
# implant. Positions are taken from the canonicalized mesh (post-LPS
# conversion in main()) of the production OBJ file. Each new mesh
# extraction is matched to these by nearest-unused-hole in (x, y).
#
# Layout summary:
#   0          — apex (small alignment hole)
#   1..5       — right column, single sub-column, anterior → posterior
#   6, 7       — centre column, right sub-col, anterior → posterior
#   8, 9       — centre column, mid sub-col,   anterior → posterior
#   10         — centre column, left sub-col   (single hole)
#   11..13     — left column, single sub-column, anterior → posterior
#
# A pure column-and-AP sort doesn't capture the centre column's three
# sub-columns (mfr ordering enumerates each sub-column right→left, then
# anterior→posterior within), so we use the explicit reference table.
_DIAGRAM_REFERENCE_POSITIONS_LPS_XY: tuple[tuple[float, float], ...] = (
    (1.84, -2.93),  # 0  — apex
    (1.05, -2.76),  # 1  — right column, most anterior
    (1.20, -1.89),  # 2
    (1.26, -0.50),  # 3
    (0.75, +1.92),  # 4
    (0.83, +3.32),  # 5  — right column, most posterior
    (1.98, +1.68),  # 6  — centre right sub-col, anterior
    (2.10, +3.94),  # 7  — centre right sub-col, posterior
    (2.54, +0.73),  # 8  — centre mid sub-col, anterior
    (2.53, +2.71),  # 9  — centre mid sub-col, posterior
    (2.96, +3.95),  # 10 — centre left sub-col
    (3.02, -0.90),  # 11 — left column, most anterior
    (3.68, +0.71),  # 12
    (4.08, +2.67),  # 13 — left column, most posterior
)


def _assign_diagram_ids(holes: list[Hole]) -> list[Hole]:
    """Re-assign hole IDs in the manufacturer's diagram convention.

    Specific to the 14-hole ``0283-300-04.obj`` implant. Each extracted
    hole is matched to its expected position in
    :data:`_DIAGRAM_REFERENCE_POSITIONS_LPS_XY` by nearest unused
    centroid in the canonicalized LPS ``(x, y)`` plane (depth ``z`` is
    ignored — the implant face is roughly a 2-D layout).

    Returns a list of ``Hole`` objects re-ordered so list-index ==
    diagram-id. If the hole count isn't 14, returns ``holes`` unchanged
    (the diagram-numbering step is a no-op for non-standard implants).

    Robust to small mesh variations: an extracted hole gets the closest
    unused diagram-id, so per-mouse mesh refits with sub-mm jitter
    still produce stable IDs as long as the layout is geometrically
    intact.
    """
    if len(holes) != len(_DIAGRAM_REFERENCE_POSITIONS_LPS_XY):
        print(
            f"  diagram-numbering: skipped (have {len(holes)} holes; "
            f"expected 14 for `0283-300-04`-style implant)"
        )
        return holes

    used: set[int] = set()
    new_holes: list[Hole | None] = [None] * len(holes)
    for did, (ex, ey) in enumerate(_DIAGRAM_REFERENCE_POSITIONS_LPS_XY):
        best_idx, best_dist = -1, float("inf")
        for i, h in enumerate(holes):
            if i in used:
                continue
            dx = float(h.ref_point[0]) - ex
            dy = float(h.ref_point[1]) - ey
            d = dx * dx + dy * dy
            if d < best_dist:
                best_dist, best_idx = d, i
        used.add(best_idx)
        new_holes[did] = holes[best_idx]
    if any(h is None for h in new_holes):  # defensive
        print("  diagram-numbering: skipped (matching failed)")
        return holes
    print(
        "  diagram-numbering: applied (matched to `0283-300-04` reference positions)."
    )
    return new_holes  # type: ignore[return-value]


def holes_to_yaml(holes: list[Hole]) -> dict:
    out: list[dict] = []
    for i, h in enumerate(holes):
        out.append(
            {
                "id": i,
                "axis_LPS": [round(float(x), 6) for x in h.axis],
                "ref_point_LPS": [round(float(x), 6) for x in h.ref_point],
                "sections": [
                    {
                        "s_mm": round(s.s_mm, 4),
                        "center_LPS": [round(float(x), 4) for x in s.center],
                        "a_mm": round(s.a_mm, 4),
                        "b_mm": round(s.b_mm, 4),
                        "theta_rad": round(s.theta_rad, 6),
                    }
                    for s in h.sections
                ],
            }
        )
    return {"holes": out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh", type=Path, help="Path to implant OBJ")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to write the YAML (default: <mesh>.holes.yml)",
    )
    p.add_argument(
        "--source-space",
        default="ASR",
        help="Coordinate system of the input mesh (canonicalize to LPS)",
    )
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor to mm")
    p.add_argument(
        "--axis",
        default="z",
        choices=["x", "y", "z"],
        help="Global probe axis (LPS frame)",
    )
    p.add_argument(
        "--n-sections",
        type=int,
        default=2,
        help="Sections per hole. A straight constant-oval channel is fully "
        "defined by two end sections (default 2); use 3 for tapered/curved bores.",
    )
    p.add_argument(
        "--max-outer-ray-mm",
        type=float,
        default=1.3,
        help="Outer-ray SDF threshold (mm). Bore walls have "
        "outer-ray hit ≈ bore diameter; everything else "
        "escapes to infinity. 1.3 mm covers typical "
        "implant bore sizes (1.2 × 0.6 mm).",
    )
    p.add_argument(
        "--ray-offset-mm",
        type=float,
        default=5e-4,
        help="Tiny offset along +normal so the source face doesn't hit itself (mm)",
    )
    p.add_argument(
        "--min-face-count",
        type=int,
        default=6,
        help="Reject bores with fewer than this many wall faces",
    )
    p.add_argument(
        "--max-tilt-deg",
        type=float,
        default=80.0,
        help="Reject merged bores whose axis is more tilted "
        "than this from the global axis (degrees)",
    )
    p.add_argument(
        "--max-ring-radius",
        type=float,
        default=1.0,
        help="Drop perpendicular-section rings whose centroid "
        "is farther than this from the bore axis (mm)",
    )
    p.add_argument(
        "--section-inset-mm",
        type=float,
        default=0.05,
        help="Inset section sample planes by this much from "
        "the wall-vertex extent (mm)",
    )
    p.add_argument(
        "--min-section-a",
        type=float,
        default=0.10,
        help="Reject bores whose largest section a_mm is below this",
    )
    p.add_argument(
        "--max-section-a",
        type=float,
        default=1.0,
        help="Reject bores whose largest section a_mm exceeds this",
    )
    p.add_argument(
        "--alpha-wrap",
        action="store_true",
        help="Alpha-wrap the (canonicalized) mesh to a watertight surface "
        "before extraction. The raw 0283-300-04 OBJ is ~45%% open edges, "
        "which corrupts the bore-face classification and section rings; "
        "wrapping repairs it. Run AFTER ASR->LPS canonicalization.",
    )
    p.add_argument(
        "--wrap-alpha",
        type=float,
        default=0.1,
        help="Alpha-wrap characteristic size (mm). Must be well under the "
        "bore feature (~0.3 mm radius) so the wrap enters the bore "
        "instead of bridging it. Default 0.1.",
    )
    p.add_argument(
        "--wrap-offset",
        type=float,
        default=0.015,
        help="Alpha-wrap offset (mm). Must be > 0 and < --wrap-alpha per "
        "CGAL; keep minimal given the small features (it shrinks the "
        "channel by ~offset). Default 0.015.",
    )
    p.add_argument(
        "--channel-a",
        type=float,
        default=None,
        help="Override fitted oval major half-extent with this known spec "
        "(mm), keeping the fitted center/axis/theta. The flared chamfer "
        "makes the mesh fit unreliable; for 0283-300-04 the deep channel "
        "is 1.2 x 0.6 mm => --channel-a 0.6 --channel-b 0.3.",
    )
    p.add_argument(
        "--channel-b",
        type=float,
        default=None,
        help="Override fitted oval minor half-extent with this known spec (mm).",
    )
    p.add_argument(
        "--narrow-bore-span",
        action="store_true",
        help="Place sections within the constant-diameter inner channel only "
        "(implant bottom -> chamfer onset), found by scanning the nearest-wall "
        "profile along the bore axis, instead of across the full wall extent. "
        "For chamfered/flared bores (0283-300-04) this keeps the threading "
        "sections in the narrow channel. Pair with --n-sections 2 (a straight "
        "constant-oval channel is fully defined by two end sections).",
    )
    p.add_argument(
        "--no-diagram-numbering",
        action="store_true",
        help="Skip the manufacturer's diagram-ID re-assignment step "
        "(apex + 5/5/3 anterior→posterior). For 14-hole "
        "0283-300-04 implants this is on by default and produces "
        "IDs that match the manufacturer's diagram. For other "
        "implants it's a no-op.",
    )
    args = p.parse_args()

    mesh = trimesh.load_mesh(args.mesh, process=False)
    if args.scale != 1.0:
        mesh.apply_scale(args.scale)
    v_lps = convert_coordinate_system(
        np.asarray(mesh.vertices), args.source_space, "LPS"
    )
    mesh = trimesh.Trimesh(v_lps, mesh.faces, process=True)
    print(
        f"Loaded {args.mesh.name}: verts={len(mesh.vertices)} "
        f"faces={len(mesh.faces)} bounds_LPS={mesh.bounds.tolist()}"
    )

    if args.alpha_wrap:
        from aind_low_point.optimization.sdf.envelope import build_alpha_wrap_envelope

        if not (0.0 < args.wrap_offset < args.wrap_alpha):
            raise SystemExit(
                f"--wrap-offset ({args.wrap_offset}) must be >0 and "
                f"< --wrap-alpha ({args.wrap_alpha}) per CGAL"
            )
        mesh = build_alpha_wrap_envelope(
            mesh,
            alpha_mm=args.wrap_alpha,
            offset_mm=args.wrap_offset,
            strip_shanks_first=False,
            use_cache=False,
        )
        print(
            f"Alpha-wrapped (alpha={args.wrap_alpha} offset={args.wrap_offset}): "
            f"verts={len(mesh.vertices)} faces={len(mesh.faces)} "
            f"watertight={mesh.is_watertight}"
        )

    axis_global = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[args.axis]
    holes = extract_holes(
        mesh,
        axis_global=np.asarray(axis_global, dtype=float),
        max_outer_ray_mm=args.max_outer_ray_mm,
        ray_offset_mm=args.ray_offset_mm,
        min_face_count=args.min_face_count,
        n_sections=args.n_sections,
        max_ring_radius=args.max_ring_radius,
        section_inset_mm=args.section_inset_mm,
        min_section_a=args.min_section_a,
        max_section_a=args.max_section_a,
        max_tilt_deg=args.max_tilt_deg,
        channel_a=args.channel_a,
        channel_b=args.channel_b,
        narrow_bore_span=args.narrow_bore_span,
    )
    if not args.no_diagram_numbering:
        holes = _assign_diagram_ids(holes)

    out_path = args.output or args.mesh.with_suffix(".holes.yml")
    payload = holes_to_yaml(holes)
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    print(f"\nWrote {len(holes)} hole spec(s) to {out_path}")


if __name__ == "__main__":
    main()
