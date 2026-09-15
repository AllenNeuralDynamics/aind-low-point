#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "cadquery-ocp==8.0.1.0.0",
#     "numpy",
#     "pyyaml",
#     "scipy",
#     "trimesh",
# ]
# ///
"""Extract implant bore geometry directly from a STEP B-rep.

Reads the implant solid with OpenCascade and writes, per bore:

- the bore axis and two oval cross-sections (top and bottom of the constant
  channel), each fitted to the exact section of the solid at that depth;
- any wall of the implant that intrudes into the channel, as a plane (point
  and normal, solid on the ``+normal`` side);
- an id, numbered in a dorsal view with anterior up and the mouse's left on
  the viewer's left: rows top to bottom, then left to right, by the centre of
  each bore's countersink.

Outputs are in the frame the planner reads the implant mesh in: the holes YAML
is canonical LPS, and the OBJ is ASR (``canonicalizations: obj-wavefront:
source_space: ASR``), both with the STEP's origin and in mm.

Channel detection works on sampled face normals, so oval extrusions that CAD
exports as B-spline surfaces qualify as well as exact cylinders: a channel wall
is a face whose normals all make the same small angle (the draft) with one
axis, facing that axis, spanning at least ``--min-span-deg`` around it.

Run standalone (this script does not import ``aind_rutter``; cadquery-ocp pins
a VTK the project does not lock)::

    uv run scripts/extract_implant_holes_step.py implant.step \\
        --holes-out implant.holes.yml --obj-out implant.obj
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml
from numpy.typing import ArrayLike, NDArray
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_MakeFace
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp, BRepGProp_Face
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import GeomAbs_Plane
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_IN,
    TopAbs_OUT,
    TopAbs_REVERSED,
)
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Face
from scipy.spatial import cKDTree

# Unit vector in LPS for each orientation letter of ``--part-frame``.
_LETTER_LPS = {
    "L": (1.0, 0.0, 0.0),
    "R": (-1.0, 0.0, 0.0),
    "P": (0.0, 1.0, 0.0),
    "A": (0.0, -1.0, 0.0),
    "S": (0.0, 0.0, 1.0),
    "I": (0.0, 0.0, -1.0),
}
_LPS_TO_ASR = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def _st(cls, name: str):
    """OCP 8 names static methods inconsistently (``Face`` vs ``Surface_s``)."""
    return getattr(cls, name + "_s", None) or getattr(cls, name)


def cap_basis(axis: ArrayLike) -> tuple[NDArray, NDArray]:
    """Build an orthonormal ``(e1, e2)`` basis perpendicular to ``axis``.

    Same convention as ``scripts/extract_implant_holes.py`` so that
    ``theta`` from the extracted YAML lines up unchanged.
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, a)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, helper)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def part_frame_matrices(code: str) -> tuple[NDArray, NDArray]:
    """``(part→LPS, part→ASR)`` rotations for an orientation code.

    ``code`` names the anatomical direction of the part's +x, +y, +z, e.g.
    ``"ALS"`` = x anterior, y left, z superior. Mirrored codes are rejected:
    a CAD export is a rigid copy of the physical part.
    """
    code = code.upper()
    if len(code) != 3 or any(c not in _LETTER_LPS for c in code):
        raise ValueError(f"part frame {code!r}: need three letters from RLAPSI")
    to_lps = np.column_stack([_LETTER_LPS[c] for c in code])
    if round(float(np.linalg.det(to_lps))) != 1:
        raise ValueError(f"part frame {code!r} is not a proper rotation")
    return to_lps, _LPS_TO_ASR @ to_lps


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _ellipse_fit(xy: NDArray) -> tuple[NDArray, float, float, float]:
    """Algebraic conic fit → ``(centre, a, b, theta)``; ``a >= b`` semi-axes and
    ``theta`` the major-axis angle. Raises if the conic is not an ellipse."""
    x, y = xy[:, 0], xy[:, 1]
    design = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    ca, cb, cc, cd, ce, cf = np.linalg.svd(design, full_matrices=False)[2][-1]
    quad = np.array([[ca, cb / 2], [cb / 2, cc]])
    centre = np.linalg.solve(2 * quad, [-cd, -ce])
    k = -(cf + 0.5 * (cd * centre[0] + ce * centre[1]))
    w, v = np.linalg.eigh(quad / k)
    if not np.all(w > 0):
        raise ValueError("points do not fit an ellipse")
    semi = 1.0 / np.sqrt(w)
    i = int(np.argmax(semi))
    return centre, float(semi[i]), float(semi[1 - i]), math.atan2(v[1, i], v[0, i])


def _ellipse_rms(xy: NDArray, centre, a: float, b: float, theta: float) -> float:
    t = np.linspace(0.0, 2 * np.pi, 20000, endpoint=False)
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    curve = np.asarray(centre) + (rot @ np.vstack([a * np.cos(t), b * np.sin(t)])).T
    d, _ = cKDTree(curve).query(xy)
    return float(np.sqrt(np.mean(d * d)))


def _angular_span_deg(rel2d: NDArray) -> float:
    ang = np.sort(np.mod(np.degrees(np.arctan2(rel2d[:, 1], rel2d[:, 0])), 360.0))
    return float(360.0 - np.diff(np.r_[ang, ang[0] + 360.0]).max())


def _point_line_distance(p: NDArray, q: NDArray, axis: NDArray) -> float:
    d = p - q
    return float(np.linalg.norm(d - (d @ axis) * axis))


def _wrap_half_pi(theta: float) -> float:
    """Ovals are symmetric under a half turn; keep theta in (-pi/2, pi/2]."""
    theta = math.remainder(theta, math.pi)
    return theta + math.pi if theta <= -math.pi / 2 else theta


# ---------------------------------------------------------------------------
# STEP reading and sampling
# ---------------------------------------------------------------------------


@dataclass
class FaceSamples:
    index: int  # 1-based, TopExp_Explorer order
    face: TopoDS_Face
    points: NDArray  # (N, 3) exact surface points (tessellation nodes)
    normals: NDArray  # (N, 3) outward unit normals (out of the solid)
    area: float
    is_plane: bool


def read_step(path: str | Path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise OSError(f"could not read STEP file {path}")
    reader.TransferRoots()
    return reader.OneShape()


def sample_faces(shape, deflection_mm: float) -> list[FaceSamples]:
    BRepMesh_IncrementalMesh(shape, deflection_mm, False, 0.05, True)
    out: list[FaceSamples] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    index = 0
    while explorer.More():
        face = _st(TopoDS, "Face")(explorer.Current())
        explorer.Next()
        index += 1
        loc = TopLoc_Location()
        tri = _st(BRep_Tool, "Triangulation")(face, loc)
        pts: list = []
        nrm: list = []
        if tri is not None:
            trsf = loc.Transformation()
            face_props = BRepGProp_Face(face)
            for i in range(1, tri.NbNodes() + 1):
                uv = tri.UVNode(i)
                p, n = gp_Pnt(), gp_Vec()
                face_props.Normal(uv.X(), uv.Y(), p, n)
                if n.Magnitude() < 1e-12:
                    continue
                pts.append(p.Transformed(trsf).Coord())
                nrm.append(np.array(n.Transformed(trsf).Coord()) / n.Magnitude())
        props = GProp_GProps()
        _st(BRepGProp, "SurfaceProperties")(face, props)
        points = np.asarray(pts, dtype=float).reshape(-1, 3)
        planar_type = BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane
        flat = len(points) >= 3 and (
            np.linalg.svd(points - points.mean(0), compute_uv=False)[-1]
            / math.sqrt(len(points))
            < 2e-3
        )
        out.append(
            FaceSamples(
                index=index,
                face=face,
                points=points,
                normals=np.asarray(nrm, dtype=float).reshape(-1, 3),
                area=props.Mass(),
                is_plane=bool(planar_type or flat),
            )
        )
    return out


def _boundary_points(face, n_per_edge: int = 200) -> NDArray:
    pts: list = []
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        edge = _st(TopoDS, "Edge")(explorer.Current())
        explorer.Next()
        if _st(BRep_Tool, "Degenerated")(edge):
            continue
        curve = BRepAdaptor_Curve(edge, face)
        for t in np.linspace(curve.FirstParameter(), curve.LastParameter(), n_per_edge):
            pts.append(curve.Value(float(t)).Coord())
    return np.asarray(pts, dtype=float).reshape(-1, 3)


def section_points(
    shape,
    faces: list[FaceSamples],
    origin: NDArray,
    normal: NDArray,
    half_size_mm: float,
) -> dict[int, NDArray]:
    """Exact planar section, points grouped by the solid face each edge lies on.

    The cutting tool is a bounded face: sectioning with an infinite ``gp_Pln``
    silently returns nothing for some planes.
    """
    plane = gp_Pln(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, normal)))
    h = half_size_mm
    tool = BRepBuilderAPI_MakeFace(plane, -h, h, -h, h).Face()
    section = BRepAlgoAPI_Section(shape, tool, False)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    section.Build()
    if not section.IsDone():
        raise RuntimeError("B-rep section failed")
    grouped: dict[int, list[NDArray]] = {}
    explorer = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    while explorer.More():
        edge = _st(TopoDS, "Edge")(explorer.Current())
        explorer.Next()
        ancestor = TopoDS_Face()
        if not section.HasAncestorFaceOn1(edge, ancestor) or ancestor.IsNull():
            continue
        index = next((f.index for f in faces if f.face.IsSame(ancestor)), -1)
        curve = BRepAdaptor_Curve(edge)
        t = np.linspace(curve.FirstParameter(), curve.LastParameter(), 300)
        grouped.setdefault(index, []).append(
            np.array([curve.Value(float(tt)).Coord() for tt in t])
        )
    return {k: np.vstack(v) for k, v in grouped.items()}


# ---------------------------------------------------------------------------
# Bore extraction
# ---------------------------------------------------------------------------


@dataclass
class ExtractConfig:
    part_frame: str = "ALS"
    sample_deflection_mm: float = 0.002
    max_draft_deg: float = 5.0
    max_normal_var: float = 2e-3
    min_span_deg: float = 150.0
    min_half_axis_mm: float = 0.1
    max_half_axis_mm: float = 1.5
    coaxial_angle_deg: float = 0.5
    coaxial_offset_mm: float = 0.05
    min_channel_length_mm: float = 0.25
    section_inset_mm: float = 0.02
    section_half_size_mm: float = 3.0
    wall_probe_mm: float = 0.012
    max_mouth_mm: float = 1.0
    row_gap_mm: float = 0.35


@dataclass
class OvalSection:
    depth_mm: float  # projection on the axis (part frame, absolute)
    center: NDArray
    a: float
    b: float
    major: NDArray  # unit major-axis direction
    fit_rms_mm: float


@dataclass
class Wall:
    face_index: int
    point: NDArray
    normal: NDArray  # from the bore into the solid
    plane_rms_mm: float
    tilt_deg: float  # angle between the plane and the bore axis
    dist_top_mm: float
    dist_bottom_mm: float


@dataclass
class Bore:
    axis: NDArray  # unit, points dorsal
    ref_point: NDArray  # axis point at the channel top
    sections: list[OvalSection]  # top first
    walls: list[Wall]
    mouth_point: NDArray
    face_indices: list[int]
    draft_deg: float
    length_mm: float
    span_deg: float
    id: int = 0
    notes: list[str] = field(default_factory=list)


def _channel_candidate(fs: FaceSamples, dorsal: NDArray, cfg: ExtractConfig):
    if fs.is_plane or len(fs.points) < 30:
        return None
    w, v = np.linalg.eigh(np.cov(fs.normals.T))
    if w[0] > cfg.max_normal_var:
        return None
    axis = v[:, 0] * (1.0 if v[:, 0] @ dorsal > 0 else -1.0)
    draft = math.degrees(math.asin(float(np.clip(np.mean(fs.normals @ axis), -1, 1))))
    if abs(draft) > cfg.max_draft_deg:
        return None
    e1, e2 = cap_basis(axis)
    xy = np.column_stack([fs.points @ e1, fs.points @ e2])
    try:
        centre, a, b, _ = _ellipse_fit(xy)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not (b >= cfg.min_half_axis_mm and a <= cfg.max_half_axis_mm):
        return None
    rel = xy - centre
    normals2d = np.column_stack([fs.normals @ e1, fs.normals @ e2])
    if np.mean(np.sum(normals2d * rel, axis=1)) >= 0:  # faces away from the axis
        return None
    span = _angular_span_deg(rel)
    if span < cfg.min_span_deg:
        return None
    point = centre[0] * e1 + centre[1] * e2
    return {"fs": fs, "axis": axis, "point": point, "span": span}


def _group_coaxial(cands: list[dict], cfg: ExtractConfig) -> list[list[dict]]:
    groups: list[list[dict]] = []
    cos_tol = math.cos(math.radians(cfg.coaxial_angle_deg))
    for c in sorted(cands, key=lambda c: -c["fs"].area):
        for g in groups:
            h = g[0]
            if (
                abs(float(c["axis"] @ h["axis"])) >= cos_tol
                and _point_line_distance(c["point"], h["point"], h["axis"])
                <= cfg.coaxial_offset_mm
            ):
                g.append(c)
                break
        else:
            groups.append([c])
    return groups


def _full_wall_depths(boundary: NDArray, point: NDArray, axis: NDArray, n_bins=72):
    """Depth range over which the wall covers every angle it covers at all:
    highest point of the lower boundary to lowest point of the upper one."""
    rel = boundary - point
    s = rel @ axis
    e1, e2 = cap_basis(axis)
    ang = np.arctan2(rel @ e2, rel @ e1)
    bins = np.floor((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    lows, highs = [], []
    for b in np.unique(bins):
        m = bins == b
        lows.append(s[m].min())
        highs.append(s[m].max())
    return float(max(lows)), float(min(highs))


def _axis_point(point: NDArray, axis: NDArray, depth: float) -> NDArray:
    return point + (depth - point @ axis) * axis


def _fit_section(pts: NDArray, origin: NDArray, axis: NDArray, depth: float):
    e1, e2 = cap_basis(axis)
    xy = np.column_stack([(pts - origin) @ e1, (pts - origin) @ e2])
    centre, a, b, theta = _ellipse_fit(xy)
    major = math.cos(theta) * e1 + math.sin(theta) * e2
    return OvalSection(
        depth_mm=depth,
        center=origin + centre[0] * e1 + centre[1] * e2,
        a=a,
        b=b,
        major=major,
        fit_rms_mm=_ellipse_rms(xy, centre, a, b, theta),
    )


def _classify(shape, p: NDArray) -> int:
    return BRepClass3d_SolidClassifier(shape, gp_Pnt(*map(float, p)), 1e-7).State()


def _find_walls(shape, faces, group_ids, cut_by_depth, sections, axis, cfg):
    """Faces cutting into the channel oval with void on the axis side."""
    e1, e2 = cap_basis(axis)
    hits: dict[int, list[NDArray]] = {}
    depth_count: dict[int, int] = {}
    for sec, cut in zip(sections, cut_by_depth):
        rel_major = sec.major
        rel_minor = np.cross(axis, rel_major)
        for idx, pts in cut.items():
            if idx in group_ids:
                continue
            rel = pts - sec.center
            u, v = rel @ rel_major, rel @ rel_minor
            inside = (u / sec.a) ** 2 + (v / sec.b) ** 2 < 0.995**2
            if inside.sum() < 10:
                continue
            q = np.column_stack([rel[inside] @ e1, rel[inside] @ e2])
            mean = q.mean(0)
            n2 = np.linalg.svd(q - mean, full_matrices=False)[2][1]
            if mean @ n2 < 0:
                n2 = -n2
            d = float(mean @ n2)
            n3 = n2[0] * e1 + n2[1] * e2
            near = _classify(shape, sec.center + (d - cfg.wall_probe_mm) * n3)
            far = _classify(shape, sec.center + (d + cfg.wall_probe_mm) * n3)
            if near == TopAbs_OUT and far == TopAbs_IN:
                hits.setdefault(idx, []).append(pts[inside])
                depth_count[idx] = depth_count.get(idx, 0) + 1
    walls: list[Wall] = []
    top, bottom = sections[0], sections[-1]
    for idx, chunks in hits.items():
        pts = np.vstack(chunks)
        centroid = pts.mean(0)
        if depth_count[idx] >= 2:
            _, sv, vt = np.linalg.svd(pts - centroid, full_matrices=False)
            normal = vt[2]
            rms = float(sv[2] / math.sqrt(len(pts)))
        else:  # one depth only: assume the wall runs parallel to the bore axis
            _, sv, vt = np.linalg.svd(pts - centroid, full_matrices=False)
            line_dir = vt[0]
            normal = np.cross(line_dir, axis)
            normal /= np.linalg.norm(normal)
            rms = float(sv[1] / math.sqrt(len(pts)))
        radial = centroid - _axis_point(top.center, axis, centroid @ axis)
        if radial @ normal < 0:
            normal = -normal
        walls.append(
            Wall(
                face_index=idx,
                point=centroid,
                normal=normal,
                plane_rms_mm=rms,
                tilt_deg=math.degrees(math.asin(min(1.0, abs(float(normal @ axis))))),
                dist_top_mm=float(normal @ (centroid - top.center)),
                dist_bottom_mm=float(normal @ (centroid - bottom.center)),
            )
        )
    return sorted(walls, key=lambda w: w.dist_bottom_mm)


def _mouth_top(faces, group_ids, top: OvalSection, axis, channel_top, cfg):
    """Highest depth of the countersink: conical faces above the channel that
    face the axis and open dorsally."""
    radius = 1.5 * top.a
    best, found = channel_top, False
    lo_up, hi_up = math.sin(math.radians(15.0)), math.sin(math.radians(80.0))
    for fs in faces:
        if fs.index in group_ids or fs.is_plane or len(fs.points) < 10:
            continue
        rel = fs.points - top.center
        s = rel @ axis + top.depth_mm
        radial = rel - np.outer(rel @ axis, axis)
        r = np.linalg.norm(radial, axis=1)
        m = (
            (r < radius)
            & (s > channel_top - 0.01)
            & (s < channel_top + cfg.max_mouth_mm)
        )
        if m.sum() < 10:
            continue
        n = fs.normals[m]
        up = float(np.median(n @ axis))
        inward = float(
            np.median(np.sum(n * radial[m], axis=1) / np.maximum(r[m], 1e-9))
        )
        if lo_up <= up <= hi_up and inward < -0.2:
            best, found = max(best, float(s[m].max())), True
    return best, found


def extract_bores(shape, cfg: ExtractConfig) -> list[Bore]:
    to_lps, _ = part_frame_matrices(cfg.part_frame)
    dorsal = to_lps.T @ np.array([0.0, 0.0, 1.0])
    faces = sample_faces(shape, cfg.sample_deflection_mm)
    cands = [c for fs in faces if (c := _channel_candidate(fs, dorsal, cfg))]
    bores: list[Bore] = []
    for group in _group_coaxial(cands, cfg):
        ids = [c["fs"].index for c in group]
        normals = np.vstack([c["fs"].normals for c in group])
        _w, v = np.linalg.eigh(np.cov(normals.T))
        axis = v[:, 0] * (1.0 if v[:, 0] @ dorsal > 0 else -1.0)
        draft = math.degrees(math.asin(float(np.clip(np.mean(normals @ axis), -1, 1))))
        point = group[0]["point"]
        boundary = np.vstack([_boundary_points(c["fs"].face) for c in group])
        s_lo, s_hi = _full_wall_depths(boundary, point, axis)
        if s_hi - s_lo < max(cfg.min_channel_length_mm, 2 * cfg.section_inset_mm):
            continue
        depths = (
            s_hi - cfg.section_inset_mm,
            0.5 * (s_lo + s_hi),
            s_lo + cfg.section_inset_mm,
        )
        cuts, sections = [], []
        for depth in depths:
            origin = _axis_point(point, axis, depth)
            cut = section_points(shape, faces, origin, axis, cfg.section_half_size_mm)
            wall_pts = [cut[i] for i in ids if i in cut]
            if not wall_pts or sum(len(p) for p in wall_pts) < 20:
                raise RuntimeError(
                    f"channel faces {ids} missing from section at {depth:.3f}"
                )
            sections.append(_fit_section(np.vstack(wall_pts), origin, axis, depth))
            cuts.append(cut)
        walls = _find_walls(shape, faces, set(ids), cuts, sections, axis, cfg)
        top, bottom = sections[0], sections[-1]
        ref_point = top.center + (s_hi - top.depth_mm) * axis
        mouth, found = _mouth_top(faces, set(ids), top, axis, s_hi, cfg)
        drift = _point_line_distance(bottom.center, top.center, axis)
        notes = [] if found else ["no countersink found; numbered at channel top"]
        bores.append(
            Bore(
                axis=axis,
                ref_point=ref_point,
                sections=[top, bottom],
                walls=walls,
                mouth_point=top.center + (mouth - top.depth_mm) * axis,
                face_indices=ids,
                draft_deg=draft,
                length_mm=s_hi - s_lo,
                span_deg=max(c["span"] for c in group),
                notes=notes
                + ([f"centre drift {1e3 * drift:.1f} um"] if drift > 5e-3 else []),
            )
        )
    return bores


# ---------------------------------------------------------------------------
# Numbering, output
# ---------------------------------------------------------------------------


def _reading_view(bore: Bore, to_lps: NDArray) -> NDArray:
    """Dorsal view, anterior up, mouse's left on the viewer's left."""
    lps = to_lps @ bore.mouth_point
    return np.array([-lps[0], -lps[1]])


def _order(bores: list[Bore], to_lps: NDArray, row_gap_mm: float) -> list[Bore]:
    view = {id(b): _reading_view(b, to_lps) for b in bores}
    rows: list[list[Bore]] = []
    prev = None
    for b in sorted(bores, key=lambda b: -view[id(b)][1]):
        if prev is None or prev - view[id(b)][1] > row_gap_mm:
            rows.append([])
        rows[-1].append(b)
        prev = view[id(b)][1]
    return [b for row in rows for b in sorted(row, key=lambda b: view[id(b)][0])]


def number_bores(
    bores: list[Bore], to_lps: NDArray, row_gap_mm: float
) -> tuple[list[Bore], tuple[float, float]]:
    """Assign ids from 1 and return the row-gap range giving the same order."""
    ordered = _order(bores, to_lps, row_gap_mm)
    for i, b in enumerate(ordered, 1):
        b.id = i
    key = [id(b) for b in ordered]
    same = [
        g
        for g in np.arange(0.01, 2.0, 0.01)
        if [id(b) for b in _order(bores, to_lps, float(g))] == key
    ]
    return ordered, (float(min(same)), float(max(same)))


def holes_to_yaml(bores: list[Bore], to_lps: NDArray) -> dict:
    def vec(v):
        return [round(float(x), 6) for x in v]

    def pos(p):
        return [round(float(x), 4) for x in p]

    out = []
    for b in bores:
        axis = to_lps @ b.axis
        ref = to_lps @ b.ref_point
        e1, e2 = cap_basis(axis)
        sections = []
        for sec in b.sections:
            center = to_lps @ sec.center
            major = to_lps @ sec.major
            theta = _wrap_half_pi(math.atan2(float(major @ e2), float(major @ e1)))
            sections.append(
                {
                    "s_mm": round(float((center - ref) @ axis), 4),
                    "center_LPS": pos(center),
                    "a_mm": round(sec.a, 4),
                    "b_mm": round(sec.b, 4),
                    "theta_rad": round(theta, 6),
                }
            )
        entry = {
            "id": b.id,
            "axis_LPS": vec(axis),
            "ref_point_LPS": pos(ref),
            "sections": sections,
        }
        if b.walls:
            entry["walls"] = [
                {
                    "point_LPS": pos(to_lps @ w.point),
                    "normal_LPS": vec(to_lps @ w.normal),
                }
                for w in b.walls
            ]
        out.append(entry)
    return {"holes": out}


def write_obj(
    shape, path: str | Path, to_asr: NDArray, linear_mm: float, angular_rad: float
) -> trimesh.Trimesh:
    """Tessellate a fresh copy of the solid and write it in ASR, mm."""
    copy = BRepBuilderAPI_Copy(shape).Shape()  # untessellated, so the params apply
    BRepMesh_IncrementalMesh(copy, linear_mm, False, angular_rad, True)
    vertices: list[NDArray] = []
    triangles: list[NDArray] = []
    base = 0
    explorer = TopExp_Explorer(copy, TopAbs_FACE)
    while explorer.More():
        face = _st(TopoDS, "Face")(explorer.Current())
        explorer.Next()
        loc = TopLoc_Location()
        tri = _st(BRep_Tool, "Triangulation")(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        nodes = np.array(
            [tri.Node(i).Transformed(trsf).Coord() for i in range(1, tri.NbNodes() + 1)]
        )
        tris = np.array(
            [
                [tri.Triangle(j).Value(k) for k in (1, 2, 3)]
                for j in range(1, tri.NbTriangles() + 1)
            ]
        )
        if face.Orientation() == TopAbs_REVERSED:
            tris = tris[:, [0, 2, 1]]
        vertices.append(nodes)
        triangles.append(tris - 1 + base)
        base += len(nodes)
    mesh = trimesh.Trimesh(
        vertices=np.vstack(vertices) @ to_asr.T,
        faces=np.vstack(triangles),
        process=True,
    )
    mesh.export(str(path), file_type="obj")
    return mesh


def _report(bores: list[Bore], to_lps: NDArray, gap_range, row_gap_mm: float) -> None:
    print(
        f"{len(bores)} bores; row gap {row_gap_mm:.2f} mm "
        f"(same order for {gap_range[0]:.2f}-{gap_range[1]:.2f} mm)"
    )
    print(
        " id  mouth part (x, y)   tilt  top 2a x 2b   bottom 2a x 2b  length  draft"
        "  fit rms  walls (distance from axis, bottom-top)"
    )
    dorsal = to_lps.T @ np.array([0.0, 0.0, 1.0])
    for b in bores:
        top, bot = b.sections
        tilt = math.degrees(math.acos(min(1.0, float(b.axis @ dorsal))))
        walls = ", ".join(
            f"face {w.face_index} {w.dist_bottom_mm:.3f}-{w.dist_top_mm:.3f} mm "
            f"({w.tilt_deg:.1f} deg, rms {1e3 * w.plane_rms_mm:.1f} um)"
            for w in b.walls
        )
        rms = max(top.fit_rms_mm, bot.fit_rms_mm)
        print(
            f"{b.id:3d}  ({b.mouth_point[0]:+.2f}, {b.mouth_point[1]:+.2f})"
            f"  {tilt:5.1f}"
            f"  {2 * top.a:.3f} x {2 * top.b:.3f}  {2 * bot.a:.3f} x {2 * bot.b:.3f}"
            f"   {b.length_mm:.3f}  {b.draft_deg:4.1f}  {1e3 * rms:5.1f} um"
            f"  {walls or '-'}" + ("".join(f"  [{n}]" for n in b.notes))
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("step", type=Path, help="Implant STEP file (mm)")
    p.add_argument("--holes-out", type=Path, help="Holes YAML (canonical LPS)")
    p.add_argument("--obj-out", type=Path, help="Implant mesh OBJ (ASR, mm)")
    p.add_argument(
        "--part-frame",
        default=ExtractConfig.part_frame,
        help="Anatomical direction of the part's +x, +y, +z (default ALS: "
        "x anterior, y left, z superior)",
    )
    p.add_argument("--row-gap-mm", type=float, default=ExtractConfig.row_gap_mm)
    p.add_argument("--max-draft-deg", type=float, default=ExtractConfig.max_draft_deg)
    p.add_argument("--min-span-deg", type=float, default=ExtractConfig.min_span_deg)
    p.add_argument(
        "--min-channel-length-mm",
        type=float,
        default=ExtractConfig.min_channel_length_mm,
    )
    p.add_argument(
        "--max-half-axis-mm", type=float, default=ExtractConfig.max_half_axis_mm
    )
    p.add_argument(
        "--section-inset-mm", type=float, default=ExtractConfig.section_inset_mm
    )
    p.add_argument("--obj-linear-mm", type=float, default=0.01)
    p.add_argument("--obj-angular-rad", type=float, default=0.2)
    args = p.parse_args(argv)

    cfg = ExtractConfig(
        part_frame=args.part_frame,
        row_gap_mm=args.row_gap_mm,
        max_draft_deg=args.max_draft_deg,
        min_span_deg=args.min_span_deg,
        min_channel_length_mm=args.min_channel_length_mm,
        max_half_axis_mm=args.max_half_axis_mm,
        section_inset_mm=args.section_inset_mm,
    )
    to_lps, to_asr = part_frame_matrices(cfg.part_frame)
    shape = read_step(args.step)
    bores, gap_range = number_bores(extract_bores(shape, cfg), to_lps, cfg.row_gap_mm)
    _report(bores, to_lps, gap_range, cfg.row_gap_mm)
    if args.holes_out:
        args.holes_out.write_text(
            yaml.safe_dump(holes_to_yaml(bores, to_lps), sort_keys=False)
        )
        print(f"wrote {args.holes_out}")
    if args.obj_out:
        mesh = write_obj(
            shape, args.obj_out, to_asr, args.obj_linear_mm, args.obj_angular_rad
        )
        print(
            f"wrote {args.obj_out}: {len(mesh.vertices)} vertices, {len(mesh.faces)} "
            f"triangles, watertight {mesh.is_watertight}"
        )


if __name__ == "__main__":
    main()
