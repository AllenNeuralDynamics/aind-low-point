"""Threading bores fitted to an implant's STEP solid: the holes file the solver reads.

Two passes. **Seeding** finds each bore on the B-rep: a channel wall is a curved
face whose exact normals all lie near one plane (the draft cone about the bore's
axis), face that axis, and wrap at least ``min_span_deg`` around it. CAD draws a
drafted oval as two ~190° half-faces whose normal-PCA axes tilt by half the
draft, so faces are grouped by centre line, not by axis, and the axis is taken
over the group. **Fitting** cuts a fine tessellation normal to the axis every
``step_mm`` (a B-rep plane section drops half-faces at some angles), fits an oval
to the nearest wall per cut, keeps the closed run through the seed whose
semi-major tapers slower than ``max_taper``, refits the axis through the run's
oval centres, and fits the oval's centre and semi-axes linear in depth.

Each bore is written twice: sections at the channel's top and exit (``exit``),
and at its top and the end of the partial wall below the exit, the oval
extrapolated there (``skirt``, the one runs use). A flat face cutting into the
oval (a D-shaped bore) is fitted as a plane over the run's cuts and written as a
wall; the oval is fitted without it. Ids number the bores in a dorsal view,
anterior up and the mouse's left on the viewer's left: rows top to bottom, then
left to right, by the channel top. Coordinates are the implant's
canonical LPS (``--part-frame`` names the STEP's axes, as the config's
``cad-part`` canonicalization does), placed per subject by ``implant_to_lps``.

Needs the ``step`` extra.

    rutter-bores implant.step implant.holes.yml --diagram implant.diagram.yml
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml
from numpy.typing import NDArray

from aind_rutter.optimization.geometry.primitives import cap_basis


@dataclass(frozen=True)
class FitSettings:
    part_frame: str = "ALS"
    tessellation_mm: float = 0.002
    tessellation_rad: float = 0.05
    # Seeding.
    max_draft_deg: float = 10.0  # a half-face's own axis reads up to twice the draft
    max_normal_var: float = 2e-3
    min_span_deg: float = 150.0
    min_half_axis_mm: float = 0.1
    max_half_axis_mm: float = 1.5
    same_bore_mm: float = 0.2  # centre lines closer than this are one bore
    # Fitting.
    step_mm: float = 0.01
    scan_mm: float = 0.5  # cut from +scan to -scan about the seed
    max_radius_mm: float = 1.2  # walls farther from the axis are not this bore's
    max_taper: float = 0.4  # mm of semi-major per mm of depth
    end_tol_mm: float = 0.003
    skirt_tol_mm: float = 0.02
    axis_tol_deg: float = 0.05  # the channel run gains or loses a cut: ~0.03° cycles
    max_refits: int = 20
    row_gap_mm: float = 0.35
    # Walls: points this far inside the oval fitted without them are a flat face's.
    # Refitting converges in a few rounds; a cut whose flat covers more than
    # max_flat_cover of the angle about the centre is a lip or rim, not a channel.
    wall_inside_mm: float = 0.01
    max_wall_splits: int = 20
    max_flat_cover: float = 0.6
    min_wall_cuts: int = 3
    wall_gap_deg: float = 30.0  # flat points split into walls at angular gaps this wide
    # A CAD flat can bow by ~0.01 mm; more than this is two faces, not one.
    max_wall_rms_mm: float = 0.02


@dataclass
class Seed:
    axis: NDArray  # part frame, unit, dorsal
    point: NDArray  # on the axis, mid-channel


@dataclass
class BoreFit:
    axis: NDArray  # part frame, unit, dorsal
    origin: NDArray  # axis point at s = 0
    fit_a: NDArray  # polyfit coefficients in s
    fit_b: NDArray
    fit_centre: NDArray  # (2, 3)
    major: NDArray
    s_top: float
    s_exit: float
    s_skirt: float
    oval_rms_mm: float
    # (point, normal into the solid) per flat face cutting into the oval
    walls: list[tuple[NDArray, NDArray]] = field(default_factory=list)
    id: int = 0

    def section(self, s: float) -> tuple[NDArray, float, float]:
        return (
            np.polyval(self.fit_centre, s),
            float(np.polyval(self.fit_a, s)),
            float(np.polyval(self.fit_b, s)),
        )


def part_to_lps(code: str) -> NDArray:
    """Rotation taking the STEP's part frame (orientation ``code``) to LPS."""
    from aind_anatomical_utils.coordinate_systems import convert_coordinate_system

    return convert_coordinate_system(np.eye(3), code, "LPS").T


def ellipse_fit(xy: NDArray) -> tuple[NDArray, float, float, float]:
    """Conic fit: ``(centre, a, b, theta)``, ``a >= b``, theta the major axis."""
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


def _ellipse_radius(phi: NDArray, a: float, b: float, theta: float) -> NDArray:
    return a * b / np.hypot(b * np.cos(phi - theta), a * np.sin(phi - theta))


def _oval_resid(xy: NDArray, c: NDArray, a: float, b: float, theta: float) -> NDArray:
    """Signed distance of each point outside the oval, along its ray from the centre."""
    phi = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    return np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) - _ellipse_radius(
        phi, a, b, theta
    )


def _wrap_half_pi(theta: float) -> float:
    """Ovals are symmetric under a half turn; keep theta in (-pi/2, pi/2]."""
    theta = math.remainder(theta, math.pi)
    return theta + math.pi if theta <= -math.pi / 2 else theta


# --- seeding on the B-rep ---------------------------------------------------------


def _face_samples(shape, deflection_mm: float):
    """Per curved face: tessellation nodes and exact unit normals (out of the solid)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, deflection_mm, False, 0.05, True)
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        explorer.Next()
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane:
            continue
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf, props = loc.Transformation(), BRepGProp_Face(face)
        pts, nrm = [], []
        for i in range(1, tri.NbNodes() + 1):
            uv = tri.UVNode(i)
            p, n = gp_Pnt(), gp_Vec()
            props.Normal(uv.X(), uv.Y(), p, n)
            if n.Magnitude() > 1e-12:
                pts.append(p.Transformed(trsf).Coord())
                nrm.append(np.array(n.Transformed(trsf).Coord()) / n.Magnitude())
        if len(pts) >= 30:
            yield np.asarray(pts), np.asarray(nrm)


def _channel_face(points: NDArray, normals: NDArray, dorsal: NDArray, cfg: FitSettings):
    """``(axis, centre point, normals)`` if the face is a bore wall, else None."""
    w, v = np.linalg.eigh(np.cov(normals.T))
    if w[0] > cfg.max_normal_var:
        return None
    axis = v[:, 0] if v[:, 0] @ dorsal > 0 else -v[:, 0]
    if (
        abs(math.degrees(math.asin(float(np.clip(np.mean(normals @ axis), -1, 1)))))
        > cfg.max_draft_deg
    ):
        return None
    e1, e2 = cap_basis(axis)
    xy = np.column_stack([points @ e1, points @ e2])
    try:
        centre, a, b, _ = ellipse_fit(xy)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not (b >= cfg.min_half_axis_mm and a <= cfg.max_half_axis_mm):
        return None
    rel = xy - centre
    if (
        np.mean(np.sum(np.column_stack([normals @ e1, normals @ e2]) * rel, axis=1))
        >= 0
    ):
        return None  # convex: a boss, not a bore
    ang = np.sort(np.mod(np.degrees(np.arctan2(rel[:, 1], rel[:, 0])), 360.0))
    if 360.0 - np.diff(np.r_[ang, ang[0] + 360.0]).max() < cfg.min_span_deg:
        return None
    point = centre[0] * e1 + centre[1] * e2 + float(np.mean(points @ axis)) * axis
    return axis, point, normals


def find_seeds(shape, cfg: FitSettings) -> list[Seed]:
    """One seed per bore, from the B-rep's channel faces."""
    dorsal = part_to_lps(cfg.part_frame).T @ np.array([0.0, 0.0, 1.0])
    faces = [
        c
        for pts, nrm in _face_samples(shape, cfg.tessellation_mm)
        if (c := _channel_face(pts, nrm, dorsal, cfg)) is not None
    ]
    groups: list[list[tuple]] = []
    for f in faces:
        for g in groups:
            ax, p = g[0][0], g[0][1]
            d = f[1] - p
            if np.linalg.norm(d - (d @ ax) * ax) <= cfg.same_bore_mm:
                g.append(f)
                break
        else:
            groups.append([f])
    seeds = []
    for g in groups:
        _, v = np.linalg.eigh(np.cov(np.vstack([f[2] for f in g]).T))
        axis = v[:, 0] if v[:, 0] @ dorsal > 0 else -v[:, 0]
        seeds.append(Seed(axis=axis, point=np.mean([f[1] for f in g], axis=0)))
    return seeds


# --- fitting on the tessellation ----------------------------------------------------


def _cut(
    mesh: trimesh.Trimesh, axis: NDArray, origin: NDArray, cfg: FitSettings, n_bins: int
):
    """Nearest wall radius per angle bin, and the wall's own points near that radius."""
    e1, e2 = cap_basis(axis)
    wall = np.full(n_bins, np.inf)
    near = np.zeros((0, 2))
    segs = trimesh.intersections.mesh_plane(mesh, axis, origin)
    if len(segs):
        # A flat face tessellates into long segments: sample them by length, or their
        # angle bins read open.
        n = np.maximum(
            6, np.ceil(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) / 0.005) + 1
        ).astype(int)
        seg = np.repeat(np.arange(len(segs)), n)
        f = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / (n[seg] - 1)
        f = f[:, None]
        pts = segs[seg, 0] * (1 - f) + segs[seg, 1] * f - origin
        x, y = pts @ e1, pts @ e2
        r = np.hypot(x, y)
        k = ((np.arctan2(y, x) + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
        np.minimum.at(wall, k, r)
        # Fit the nearest wall's points at their own angles: a bin's minimum placed at
        # the bin's centre pulls the oval inside the wall at its major ends.
        keep = (r < wall[k] + 0.02) & (r <= cfg.max_radius_mm)
        near = np.column_stack([x[keep], y[keep]])
    wall[wall > cfg.max_radius_mm] = np.inf
    return wall, near


def _split_flat(xy: NDArray, cfg: FitSettings):
    """The oval fitted without a flat face biting into it, and which points are the
    flat's: those ``wall_inside_mm`` inside the oval, refitted until they settle."""
    fit = ellipse_fit(xy)
    on_flat = _oval_resid(xy, *fit) <= -cfg.wall_inside_mm
    for _ in range(cfg.max_wall_splits):
        if not on_flat.any():
            break
        fit = ellipse_fit(xy[~on_flat])
        again = _oval_resid(xy, *fit) <= -cfg.wall_inside_mm
        if (again == on_flat).all():
            break
        on_flat = again
    return fit, on_flat


def _cover(rel: NDArray, n_bins: int = 72) -> float:
    """Fraction of angle bins about the origin that hold a point."""
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    bins = ((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    return len(np.unique(bins)) / n_bins


def _scan(mesh, axis, origin, cfg: FitSettings, s_grid: NDArray, n_bins: int):
    """Per cut, the wall and its oval where closed; and the channel run through 0."""
    e1, e2 = cap_basis(axis)
    rows = []
    for s in s_grid:
        wall, xy = _cut(mesh, axis, origin + s * axis, cfg, n_bins)
        row = {"s": s, "wall": wall, "closed": bool(np.isfinite(wall).all())}
        if row["closed"]:
            (c, a, b, th), on_flat = _split_flat(xy, cfg)
            if _cover(xy[on_flat] - c) > cfg.max_flat_cover:
                row["closed"] = False
                rows.append(row)
                continue
            flat, xy = xy[on_flat], xy[~on_flat]
            resid = _oval_resid(xy, c, a, b, th)
            row.update(
                a=a,
                b=b,
                rms=float(np.sqrt(np.mean(resid**2))),
                centre=origin + s * axis + c[0] * e1 + c[1] * e2,
                major=math.cos(th) * e1 + math.sin(th) * e2,
                flat=origin + s * axis + flat[:, :1] * e1 + flat[:, 1:] * e2,
            )
        rows.append(row)
    ok = [r["closed"] for r in rows]
    lag = max(
        1, round(0.04 / cfg.step_mm)
    )  # per-cut a is bin-noisy; slope over 0.04 mm
    for i in range(lag, len(rows)):
        if (
            ok[i]
            and ok[i - lag]
            and abs(rows[i]["a"] - rows[i - lag]["a"]) / (lag * cfg.step_mm)
            > cfg.max_taper
        ):
            ok[i - lag] = False
    i0 = int(np.argmin(np.abs(s_grid)))
    if not ok[i0]:
        return rows, []
    lo = hi = i0
    while hi + 1 < len(rows) and ok[hi + 1]:
        hi += 1
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    return rows, list(range(lo, hi + 1))


def _trim_ends(rows: list[dict], run: list[int], cfg: FitSettings) -> list[int]:
    """Drop end cuts off the interior's linear model: a cut on the countersink's lip
    or the exit's rim fits a closed oval that is not the channel's."""
    while len(run) > 5:
        s_in = np.array([rows[i]["s"] for i in run[1:-1]])
        fa = np.polyfit(s_in, [rows[i]["a"] for i in run[1:-1]], 1)
        fb = np.polyfit(s_in, [rows[i]["b"] for i in run[1:-1]], 1)
        off = [
            max(
                abs(rows[i]["a"] - np.polyval(fa, rows[i]["s"])),
                abs(rows[i]["b"] - np.polyval(fb, rows[i]["s"])),
            )
            for i in (run[0], run[-1])
        ]
        if max(off) <= cfg.end_tol_mm:
            break
        run = run[1:] if off[0] >= off[1] else run[:-1]
    return run


def _fit_walls(
    rows: list[dict], run: list[int], axis: NDArray, origin: NDArray, cfg: FitSettings
) -> list[tuple[NDArray, NDArray]]:
    """A plane per flat face the run's cuts found inside the oval, normal into the
    solid, through the face's point nearest the bore so none of it crosses the plane;
    flat points split into faces at angular gaps about the axis."""
    chunks = [(k, rows[i]["flat"]) for k, i in enumerate(run) if len(rows[i]["flat"])]
    if not chunks:
        return []
    pts = np.vstack([p for _, p in chunks])
    cut = np.concatenate([np.full(len(p), k) for k, p in chunks])
    e1, e2 = cap_basis(axis)
    rel = pts - origin
    ang = np.degrees(np.arctan2(rel @ e2, rel @ e1))
    # Sweep from just past the widest angular gap so no face straddles the wrap.
    order = np.argsort(ang)
    gaps = np.diff(np.r_[ang[order], ang[order[0]] + 360.0])
    order = np.roll(order, -(int(np.argmax(gaps)) + 1))
    gaps = np.mod(np.diff(ang[order]), 360.0)
    walls = []
    for g in np.split(order, np.flatnonzero(gaps > cfg.wall_gap_deg) + 1):
        if len(np.unique(cut[g])) < cfg.min_wall_cuts:
            continue
        centroid = pts[g].mean(0)
        _, sv, vt = np.linalg.svd(pts[g] - centroid, full_matrices=False)
        rms = float(sv[2] / math.sqrt(len(g)))
        if rms > cfg.max_wall_rms_mm:
            raise RuntimeError(
                f"bore at {np.round(origin, 3)}: flat points fit a plane to "
                f"{1e3 * rms:.1f} um"
            )
        normal = vt[2]
        radial = centroid - origin - ((centroid - origin) @ axis) * axis
        normal = normal if normal @ radial > 0 else -normal
        walls.append((centroid + ((pts[g] - centroid) @ normal).min() * normal, normal))
    return walls


def fit_bore(mesh: trimesh.Trimesh, seed: Seed, cfg: FitSettings) -> BoreFit:
    s_grid = np.arange(cfg.scan_mm, -cfg.scan_mm - 1e-9, -cfg.step_mm)
    n_bins = 72
    axis, origin = seed.axis, seed.point
    for _ in range(cfg.max_refits):
        rows, run = _scan(mesh, axis, origin, cfg, s_grid, n_bins)
        if len(run) < 5:
            raise RuntimeError(
                f"bore at {np.round(seed.point, 3)}: no channel through the seed"
            )
        run = _trim_ends(rows, run, cfg)
        centres = np.stack([rows[i]["centre"] for i in run])
        mean = centres.mean(0)
        d = np.linalg.svd(centres - mean)[2][0]
        d = d if d @ axis > 0 else -d
        moved = math.degrees(math.acos(min(1.0, float(d @ axis))))
        axis, origin = d, mean + ((origin - mean) @ d) * d
        if moved < cfg.axis_tol_deg:
            break
    else:
        raise RuntimeError(
            f"bore at {np.round(seed.point, 3)}: axis still moving {moved:.3f}° "
            f"after {cfg.max_refits} refits"
        )
    rows, run = _scan(mesh, axis, origin, cfg, s_grid, n_bins)
    if len(run) < 5:
        raise RuntimeError(
            f"bore at {np.round(seed.point, 3)}: no channel after the axis refit"
        )
    run = _trim_ends(rows, run, cfg)
    s_run = np.array([rows[i]["s"] for i in run])
    fit_a = np.polyfit(s_run, [rows[i]["a"] for i in run], 1)
    fit_b = np.polyfit(s_run, [rows[i]["b"] for i in run], 1)
    fit_centre = np.polyfit(s_run, np.stack([rows[i]["centre"] for i in run]), 1)
    major = rows[run[len(run) // 2]]["major"]
    s_exit = float(s_run.min())
    # Skirt: the lowest cut below the exit where some wall is within skirt_tol of the
    # extrapolated oval (its centre is µm from the axis, so rays from the axis).
    e1, e2 = cap_basis(axis)
    theta = math.atan2(major @ e2, major @ e1)
    mid = (np.arange(n_bins) + 0.5) / n_bins * 2 * np.pi - np.pi
    s_skirt = s_exit
    for r in rows:
        if r["s"] < s_exit:
            rr = _ellipse_radius(
                mid,
                float(np.polyval(fit_a, r["s"])),
                float(np.polyval(fit_b, r["s"])),
                theta,
            )
            if (np.isfinite(r["wall"]) & (r["wall"] - rr < cfg.skirt_tol_mm)).any():
                s_skirt = float(r["s"])
    return BoreFit(
        axis=axis,
        origin=origin,
        fit_a=fit_a,
        fit_b=fit_b,
        fit_centre=fit_centre,
        major=major,
        s_top=float(s_run.max()),
        s_exit=s_exit,
        s_skirt=s_skirt,
        oval_rms_mm=max(rows[i]["rms"] for i in run),
        walls=_fit_walls(rows, run, axis, origin, cfg),
    )


# --- numbering and output ------------------------------------------------------------


def number_bores(
    bores: list[BoreFit], to_lps: NDArray, row_gap_mm: float
) -> list[BoreFit]:
    """Ids from 1, in reading order of a dorsal view by channel top."""

    def view(b):
        lps = to_lps @ b.section(b.s_top)[0]
        return -lps[0], -lps[1]  # x to the mouse's right, y anterior

    rows: list[list[BoreFit]] = []
    prev = None
    for b in sorted(bores, key=lambda b: -view(b)[1]):
        if prev is None or prev - view(b)[1] > row_gap_mm:
            rows.append([])
        rows[-1].append(b)
        prev = view(b)[1]
    ordered = [b for row in rows for b in sorted(row, key=lambda b: view(b)[0])]
    for i, b in enumerate(ordered, 1):
        b.id = i
    return ordered


def holes_yaml(bores: list[BoreFit], to_lps: NDArray, bottom: str) -> dict:
    """The holes file: sections at the channel top and at ``bottom``, exit or skirt."""
    out = []
    for b in bores:
        axis = to_lps @ b.axis
        e1, e2 = cap_basis(axis)
        major = to_lps @ b.major
        theta = round(
            _wrap_half_pi(math.atan2(float(major @ e2), float(major @ e1))), 6
        )
        sections = []
        for s in (b.s_top, b.s_exit if bottom == "exit" else b.s_skirt):
            centre, a, bb = b.section(s)
            sections.append(
                {
                    "s_mm": round(s - b.s_top, 4),
                    "center_LPS": [round(float(x), 4) for x in to_lps @ centre],
                    "a_mm": round(a, 4),
                    "b_mm": round(bb, 4),
                    "theta_rad": theta,
                }
            )
        entry = {
            "id": b.id,
            "axis_LPS": [round(float(x), 6) for x in axis],
            "ref_point_LPS": [
                round(float(x), 4) for x in to_lps @ (b.origin + b.s_top * b.axis)
            ],
            "sections": sections,
        }
        if b.walls:
            entry["walls"] = [
                {
                    "point_LPS": [round(float(x), 4) for x in to_lps @ p],
                    "normal_LPS": [round(float(x), 6) for x in to_lps @ n],
                }
                for p, n in b.walls
            ]
        out.append(entry)
    return {"holes": out}


def number_by_diagram(
    bores: list[BoreFit], to_lps: NDArray, diagram: Path, max_offset_mm: float = 0.4
) -> list[BoreFit]:
    """Ids from the implant's diagram file (``bores``: ``id``, ``xy_LPS``, optional
    ``probe``), matched one to one by channel-top (x, y); bores the diagram marks
    ``probe: false`` are dropped."""
    from scipy.optimize import linear_sum_assignment

    ref = yaml.safe_load(Path(diagram).read_text())["bores"]
    if len(ref) != len(bores):
        raise ValueError(
            f"{diagram}: {len(ref)} bores in the diagram, {len(bores)} fitted"
        )
    xy = np.array([(to_lps @ b.section(b.s_top)[0])[:2] for b in bores])
    ref_xy = np.array([r["xy_LPS"] for r in ref], float)
    dist = np.linalg.norm(xy[:, None] - ref_xy[None], axis=2)
    rows, cols = linear_sum_assignment(dist)
    if dist[rows, cols].max() > max_offset_mm:
        raise ValueError(
            f"{diagram}: a fitted bore sits {dist[rows, cols].max():.2f} mm "
            "from its diagram position"
        )
    kept = []
    for i, j in zip(rows, cols):
        if ref[j].get("probe", True):
            bores[i].id = int(ref[j]["id"])
            kept.append(bores[i])
    return sorted(kept, key=lambda b: b.id)


def fit_step(
    path: str | Path, cfg: FitSettings = FitSettings(), diagram: Path | None = None
) -> list[BoreFit]:
    """Seed, fit and number every bore of the implant STEP at ``path``."""
    from aind_rutter.build.step import load_step, read_step_shape

    shape = read_step_shape(str(path))
    mesh = load_step(str(path), cfg.tessellation_mm, cfg.tessellation_rad)
    bores = [fit_bore(mesh, seed, cfg) for seed in find_seeds(shape, cfg)]
    to_lps = part_to_lps(cfg.part_frame)
    if diagram is not None:
        return number_by_diagram(bores, to_lps, diagram)
    return number_bores(bores, to_lps, cfg.row_gap_mm)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("step", type=Path, help="implant STEP file")
    p.add_argument("out", type=Path, help="holes YAML (skirt sections)")
    p.add_argument(
        "--exit-out", type=Path, help="also write the exit-section file here"
    )
    p.add_argument(
        "--diagram",
        type=Path,
        help="number by the manufacturer's diagram in this file, not reading order",
    )
    p.add_argument(
        "--part-frame",
        default=FitSettings.part_frame,
        help="anatomical direction of the STEP's +x, +y, +z (default ALS)",
    )
    p.add_argument("--row-gap-mm", type=float, default=FitSettings.row_gap_mm)
    args = p.parse_args(argv)
    cfg = FitSettings(part_frame=args.part_frame, row_gap_mm=args.row_gap_mm)
    bores = fit_step(args.step, cfg, args.diagram)
    to_lps = part_to_lps(cfg.part_frame)
    print(
        " id  exit    skirt   a top->exit    b top->exit    oval rms"
        "  walls, mm from axis top->exit   (s from channel top, mm)"
    )
    for b in bores:
        c0, a0, b0 = b.section(b.s_top)
        c1, a1, b1 = b.section(b.s_exit)
        walls = ", ".join(f"{n @ (p - c0):.3f}->{n @ (p - c1):.3f}" for p, n in b.walls)
        print(
            f"{b.id:3d}  {b.s_exit - b.s_top:+.3f}  {b.s_skirt - b.s_top:+.3f}  "
            f"{a0:.3f}->{a1:.3f}  {b0:.3f}->{b1:.3f}  {1e3 * b.oval_rms_mm:.1f} um"
            f"   {walls or '-'}"
        )
    args.out.write_text(
        yaml.safe_dump(holes_yaml(bores, to_lps, "skirt"), sort_keys=False)
    )
    print(f"wrote {len(bores)} bores to {args.out}")
    if args.exit_out:
        args.exit_out.write_text(
            yaml.safe_dump(holes_yaml(bores, to_lps, "exit"), sort_keys=False)
        )
        print(f"wrote {args.exit_out}")
