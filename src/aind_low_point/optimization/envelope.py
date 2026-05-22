"""α-wrap envelope construction for probe collision geometry.

Raw probe CAD is surface-modeled (sheets, open tubes, multi-holed
plates) — non-watertight at the component level, so libigl PSEUDONORMAL
SDF gives wrong signs and any algorithm that needs interior/exterior
classification fails. The fix is CGAL's α-wrap (Portaneri et al.,
SIGGRAPH 2022 — "Alpha Wrapping with an Offset"): given a soup of
triangles, produce a *watertight, manifold* envelope at user-specified
offset ``α``.

The envelope preserves concavities at scales larger than ``α`` and
smooths out features below it. For our 25 × 170 × 25 mm bodies,
``α = 0.5 mm, offset = 0.05 mm`` gives ~22k-vertex envelopes that match
raw FCL BVH clearance to within 1.1% FP and 0% FN.

Shanks are stripped before wrapping (long thin features that the
α-wrap would either dilate beyond their literature width or merge into
adjacent body via the gap-closing). Stripped shanks become analytic
OBB primitives in the dual-rep collision query.

Caches envelopes by (mesh-hash, α, offset) under
``$AIND_LOW_POINT_CACHE_DIR/envelopes`` or ``~/.cache/aind_low_point/envelopes``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import trimesh

# CGAL alpha-wrap via pymeshlab. Imported lazily so unrelated code
# paths don't pay the ~100 MB pymeshlab import cost.


SHANK_MAX_TIP_Z_MM = 1.0
SHANK_MAX_LENGTH_MM = 10.5
SHANK_MIN_LENGTH_MM = 5.0
SHANK_MIN_ASPECT = 30.0

# Anything below this z is "shank zone" — silicon shank plus the
# transitional junction caps that connect shank to body. The OBB grows
# to enclose the union; the body envelope is built from components with
# z_min ≥ this. Set to 10.5 mm to cover the NP 2.0 silicon (z ≤ 10) plus
# a 0.5 mm slop for the junction (typically at z ≈ 10.0–10.3 mm).
SHANK_ZONE_Z_MAX_MM = 10.5

# Lower z bound for the junction zone: components whose ``z_max`` is at
# least this far up are candidates for the transition-zone union OBB.
# Anything below this (e.g., tiny ``z_extent ≈ 0`` tip-face caps at
# z=0 that fail :func:`is_shank_component`'s length criterion) is left
# alone — bundling those into the transition OBB would extend it down
# to z=0 and balloon the xy extent to engulf the whole shank zone.
SHANK_JUNCTION_MIN_Z_MAX_MM = 9.0


def is_shank_component(comp: trimesh.Trimesh) -> bool:
    """True if ``comp`` looks like a silicon shank sheet.

    Criteria (literature NP 2.x):
      - extends through ``z ∈ [near 0, ≤ 10.5 mm]`` (tip near origin)
      - z-extent ≥ 5 mm
      - aspect ratio ``z / max(xy) ≥ 30`` (very elongated)
    """
    bounds = comp.bounds
    z_min, z_max = bounds[0, 2], bounds[1, 2]
    if z_max > SHANK_MAX_LENGTH_MM:
        return False
    if z_min > SHANK_MAX_TIP_Z_MM:
        return False
    extent_xy = max(bounds[1, 0] - bounds[0, 0], bounds[1, 1] - bounds[0, 1])
    extent_z = z_max - z_min
    if extent_z < SHANK_MIN_LENGTH_MM:
        return False
    return extent_z / max(extent_xy, 1e-6) > SHANK_MIN_ASPECT


def is_in_shank_zone(comp: trimesh.Trimesh) -> bool:
    """True if every part of ``comp`` lies in the shank zone ``z ≤
    SHANK_ZONE_Z_MAX_MM``.

    This catches both the silicon shank sheets *and* the transitional
    junction caps between shank and body — small flat shapes that fail
    :func:`is_shank_component`'s aspect/length criteria but sit in the
    same z range and would otherwise be smoothed away by alpha-wrap.
    Treating them as part of the shank zone lets the shank OBB cover
    them and the body envelope skip them cleanly.
    """
    z_max = comp.bounds[1, 2]
    return z_max <= SHANK_ZONE_Z_MAX_MM


def strip_shanks(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    """Return ``(body_mesh, n_stripped)``. ``body_mesh`` is the union of
    all components NOT in the shank zone (silicon shank + junction
    caps). The shank OBB built via :func:`extract_shank_obbs` covers
    the stripped geometry.
    """
    comps = mesh.split(only_watertight=False)
    body = [c for c in comps if not is_in_shank_zone(c)]
    n_stripped = len(comps) - len(body)
    if not body:
        raise ValueError("No body components after stripping shank zone")
    return trimesh.util.concatenate(body), n_stripped


def extract_shank_obbs(
    mesh: trimesh.Trimesh,
    *,
    dedup_xy_tol_mm: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract shank-zone OBBs from a probe mesh.

    Returns ``(centers, half_extents)`` of shape ``(n_obbs, 3)`` in the
    probe's canonical local frame. The shank zone is split into two
    kinds of OBB:

    1. **One OBB per silicon shank**, derived from the components that
       satisfy :func:`is_shank_component` (long, thin, near-origin in
       z). Duplicate shank sheets (front/back face) merged by XY
       centroid clustering (``dedup_xy_tol_mm``).
    2. **A single union OBB for the transition zone** — the bounding
       box over every component that's :func:`is_in_shank_zone` but
       NOT :func:`is_shank_component`. These are the small flat caps
       at the shank-PCB junction that would otherwise fall between
       the per-shank OBB (too thin in xy) and the body envelope (which
       starts above z=``SHANK_ZONE_Z_MAX_MM``).

    Total OBB count is ``n_shanks + (1 if any junctions else 0)``.

    Raw CAD often represents shanks as zero-thickness sheets; the
    bbox-derived half-extent is ``0`` on those axes. Use
    :func:`floor_shank_half_extents` to pad to literature dimensions
    before using for collision.
    """
    comps = mesh.split(only_watertight=False)
    silicon_centers: list[np.ndarray] = []
    silicon_halves: list[np.ndarray] = []
    junction_vertices: list[np.ndarray] = []
    for c in comps:
        if is_shank_component(c):
            bmin, bmax = c.bounds
            silicon_centers.append(0.5 * (bmin + bmax))
            silicon_halves.append(0.5 * (bmax - bmin))
        elif is_in_shank_zone(c) and c.bounds[1, 2] >= SHANK_JUNCTION_MIN_Z_MAX_MM:
            junction_vertices.append(np.asarray(c.vertices, dtype=np.float64))

    out_centers: list[np.ndarray] = []
    out_halves: list[np.ndarray] = []

    # Per-silicon-shank OBBs with XY-centroid dedup.
    if silicon_centers:
        sc = np.stack(silicon_centers, axis=0)
        sh = np.stack(silicon_halves, axis=0)
        if dedup_xy_tol_mm <= 0:
            out_centers.extend(sc)
            out_halves.extend(sh)
        else:
            cluster: list[list[int]] = []
            for i, c_xy in enumerate(sc[:, :2]):
                for members in cluster:
                    ref_xy = sc[members[0], :2]
                    if np.linalg.norm(c_xy - ref_xy) <= dedup_xy_tol_mm:
                        members.append(i)
                        break
                else:
                    cluster.append([i])
            for members in cluster:
                sub_min = (sc[members] - sh[members]).min(axis=0)
                sub_max = (sc[members] + sh[members]).max(axis=0)
                out_centers.append(0.5 * (sub_min + sub_max))
                out_halves.append(0.5 * (sub_max - sub_min))

    # Single union OBB over all junction-zone components.
    if junction_vertices:
        all_v = np.vstack(junction_vertices)
        bmin = all_v.min(axis=0)
        bmax = all_v.max(axis=0)
        out_centers.append(0.5 * (bmin + bmax))
        out_halves.append(0.5 * (bmax - bmin))

    if not out_centers:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )
    return (
        np.stack(out_centers, axis=0).astype(np.float64),
        np.stack(out_halves, axis=0).astype(np.float64),
    )


def floor_shank_half_extents(
    half_extents: np.ndarray,
    *,
    min_thick_mm: float = 0.012,
    min_width_mm: float = 0.035,
) -> np.ndarray:
    """Apply literature-spec floors to shank half-extents.

    NP 2.0 silicon: 24 µm thick × 70 µm wide × ~10 mm long. Raw CAD
    sometimes encodes shanks as zero-thickness sheets in one axis; this
    pads them to literature dimensions for collision purposes.

    Floors per row: half-thickness ``min_thick_mm`` to the smallest axis,
    half-width ``min_width_mm`` to the next smallest. Z (length) is
    left as-is — the mesh's z-extent is the actual shank length.
    """
    if half_extents.shape[0] == 0:
        return half_extents
    out = half_extents.copy()
    # Sort xy half-extents per row so we know which is "thick" vs "wide"
    xy = out[:, :2]
    order = np.argsort(xy, axis=1)  # (N, 2): index 0 → smaller
    for i in range(out.shape[0]):
        small_axis = order[i, 0]
        wide_axis = order[i, 1]
        out[i, small_axis] = max(out[i, small_axis], min_thick_mm)
        out[i, wide_axis] = max(out[i, wide_axis], min_width_mm)
    return out


def _envelope_cache_dir() -> Path:
    root = os.environ.get("AIND_LOW_POINT_CACHE_DIR")
    if root:
        return Path(root) / "envelopes"
    return Path.home() / ".cache" / "aind_low_point" / "envelopes"


def _mesh_hash(mesh: trimesh.Trimesh) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def build_alpha_wrap_envelope(
    mesh: trimesh.Trimesh,
    *,
    alpha_mm: float = 0.5,
    offset_mm: float = 0.2,
    strip_shanks_first: bool = True,
    use_cache: bool = True,
) -> trimesh.Trimesh:
    """Construct a watertight α-wrap envelope of ``mesh``.

    Parameters
    ----------
    mesh
        Raw probe mesh in canonical local frame.
    alpha_mm
        α-wrap characteristic size. Smaller preserves finer concavities
        but costs more compute (and produces more vertices). Default
        0.5 mm gives a 22k-vertex envelope on quadbase bodies in ~7 s.
    offset_mm
        Outward inflation distance. Must be ``< alpha_mm`` per CGAL
        guidelines. Default 0.2 mm fully contains the body surface
        (1M dense samples on probe:2.1, 100% inside envelope) once the
        shank-body junction caps are pulled into the transition OBB
        (see :func:`extract_shank_obbs`). Previously 0.05 left a 0.59 %
        residue of face midpoints outside the envelope near the
        junction; that residue is now in the OBB instead.
    strip_shanks_first
        When True (default), drop shank-like components before wrapping
        so the envelope contains only the body. The optimizer handles
        shanks as analytic OBBs.
    use_cache
        Cache by (mesh-hash, α, offset, strip-shanks) under the
        envelope cache directory.
    """
    cdir = _envelope_cache_dir()
    key = (
        f"{_mesh_hash(mesh)}_a{alpha_mm}_o{offset_mm}"
        f"_strip{int(strip_shanks_first)}"
    )
    cpath = cdir / f"{key}.npz"
    if use_cache and cpath.exists():
        with np.load(cpath) as data:
            return trimesh.Trimesh(
                vertices=data["vertices"].astype(np.float64),
                faces=data["faces"].astype(np.int64),
                process=False,
            )

    import pymeshlab

    body = strip_shanks(mesh)[0] if strip_shanks_first else mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.ascontiguousarray(body.vertices, dtype=np.float64),
            face_matrix=np.ascontiguousarray(body.faces, dtype=np.int32),
        ),
        "body",
    )
    ms.generate_alpha_wrap(
        alpha=pymeshlab.PureValue(alpha_mm),
        offset=pymeshlab.PureValue(offset_mm),
    )
    out = ms.current_mesh()
    env = trimesh.Trimesh(
        vertices=np.asarray(out.vertex_matrix(), dtype=np.float64),
        faces=np.asarray(out.face_matrix(), dtype=np.int64),
        process=False,
    )

    if use_cache:
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cpath,
            vertices=env.vertices.astype(np.float64),
            faces=env.faces.astype(np.int64),
        )
    return env
