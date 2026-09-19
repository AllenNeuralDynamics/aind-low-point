"""Phase-1 pipeline utilities shared across scripts.

This module exports the infrastructure helpers that production scripts need:

  - :func:`phase1_bounds`
  - :func:`build_fixture_sdf_data` / :func:`fixture_keys_from_runtime`
  - :func:`build_brain_sdf` / :func:`maybe_build_brain_sdf`
  - :func:`build_coverage_data`
"""

from __future__ import annotations

import os as _os
from dataclasses import replace

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from aind_rutter.build.queries import fixture_node_keys, world_geometry_for_node
from aind_rutter.domain.probe_kinds import RECORDING_GEOMETRY, RecordingGeometry
from aind_rutter.domain.rig import ML_LIMIT_DEG, reachable_ap_window_deg
from aind_rutter.optimization.clearance import (
    build_probe_sdf,
    build_probe_sdf_from_alpha_wrap,
)
from aind_rutter.optimization.objectives.coverage import (
    GaussianCoverageData,
    build_coverage_data_from_probe_context,
)
from aind_rutter.optimization.objectives.soft import BrainSDFData, FixtureSDFData


def phase1_bounds(n_arcs: int, n_probes: int, head_pitch_deg: float = 0.0):
    """Box bounds for Phase 1 x = (arc_aps, (ml, sx, sy, off_R, off_A, depth) × P)."""
    bounds = []
    ap_window = reachable_ap_window_deg(head_pitch_deg)
    for _ in range(n_arcs):
        bounds.append(ap_window)
    for _ in range(n_probes):
        bounds.append((-ML_LIMIT_DEG, +ML_LIMIT_DEG))  # ml
        # (sx, sy) ±1.1 — unit_circle_penalty pulls magnitude → 1.
        bounds.append((-1.1, +1.1))  # sx
        bounds.append((-1.1, +1.1))  # sy
        bounds.append((-3.0, +3.0))  # off_R (mm)
        bounds.append((-3.0, +3.0))  # off_A (mm)
        bounds.append((-2.0, +2.0))  # depth (mm past target)
    return bounds


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def fixture_keys_from_runtime(runtime) -> list[str]:
    """Identify static scene-fixture asset keys.

    Picks scene nodes tagged ``fixture``/``cone``/``well``/``headframe``
    but excludes any node also tagged ``implant`` — probes thread
    through the implant via holes, so it's not a body-collision
    obstacle. The 836656 config tags the implant with both ``fixture``
    AND ``implant`` so the explicit exclusion is required.
    """
    return list(fixture_node_keys(runtime))


_CONE_CROP_MARGIN_FRAC = 0.30


def _crop_fixture_to_box(
    fx: FixtureSDFData, box_min: np.ndarray, box_max: np.ndarray
) -> FixtureSDFData:
    """Crop a fixture's SDF grid to the world-LPS AABB ``[box_min, box_max]``."""
    origin = np.asarray(fx.origin, float)
    spacing = np.asarray(fx.spacing, float)
    grid = np.asarray(fx.grid)
    shape = np.asarray(grid.shape)
    i0 = np.clip(np.floor((box_min - origin) / spacing).astype(int), 0, shape)
    i1 = np.clip(np.ceil((box_max - origin) / spacing).astype(int) + 1, 0, shape)
    if np.any(i1 <= i0):
        return fx
    sub = grid[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]]
    new_origin = origin + i0 * spacing
    return replace(
        fx,
        grid=jnp.asarray(sub, fx.grid.dtype),
        origin=jnp.asarray(new_origin, jnp.float32),
    )


def _resample_envelope_surface_in_box(
    raw_mesh, *, offset_mm: float, box_min: np.ndarray, box_max: np.ndarray, n: int
) -> np.ndarray:
    """Area-uniform sample ``n`` points on the α-wrap envelope within the box.

    Seeded and cached on disk, so every process builds identical fixture constraints.
    """
    import hashlib

    import trimesh

    from aind_rutter.optimization.clearance.envelope import build_alpha_wrap_envelope
    from aind_rutter.optimization.clearance.voxel_sdf import (
        SURFACE_SAMPLE_SEED,
        _cache_dir,
        _mesh_hash,
    )

    box = np.round(np.concatenate([box_min, box_max]).astype(np.float64), 6)
    box_key = hashlib.sha256(box.tobytes()).hexdigest()[:12]
    cpath = _cache_dir() / (
        f"envcrop_{_mesh_hash(raw_mesh)}_a0.2_o{offset_mm}_n{n}"
        f"_seed{SURFACE_SAMPLE_SEED}_{box_key}.npz"
    )
    if cpath.exists():
        with np.load(cpath) as data:
            return data["surface"].astype(np.float32)

    env = build_alpha_wrap_envelope(
        raw_mesh, alpha_mm=0.2, offset_mm=offset_mm, strip_shanks_first=False
    )
    rng = np.random.default_rng(SURFACE_SAMPLE_SEED)
    kept: list[np.ndarray] = []
    have = 0
    for _ in range(60):
        pts, _ = trimesh.sample.sample_surface(env, max((n - have) * 6, 2000), seed=rng)
        pts = np.asarray(pts, np.float32)
        inb = pts[np.all((pts >= box_min) & (pts <= box_max), axis=1)]
        if len(inb):
            kept.append(inb)
            have += len(inb)
        if have >= n:
            break
    allp = np.concatenate(kept, axis=0) if kept else np.zeros((0, 3), np.float32)
    if len(allp) < n:
        reps = int(np.ceil(n / max(len(allp), 1)))
        allp = np.tile(allp, (reps, 1)) if len(allp) else np.zeros((n, 3), np.float32)
    out = allp[:n].astype(np.float32)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    # Workers start together; write-then-rename keeps a reader from seeing a
    # partial file.
    tmp = cpath.with_name(f"{cpath.stem}.{_os.getpid()}.tmp.npz")
    np.savez(tmp, surface=out)
    _os.replace(tmp, cpath)
    return out


def _crop_cone_to_well(
    fixtures: tuple[FixtureSDFData, ...], raw_meshes: dict
) -> tuple[FixtureSDFData, ...]:
    """Crop the cone SDF grid to the well neighbourhood."""
    well = next((f for f in fixtures if "well" in f.name.lower()), None)
    if well is None:
        return fixtures
    w_min = np.asarray(well.origin, float)
    w_max = w_min + (np.asarray(well.grid.shape) - 1) * np.asarray(well.spacing, float)
    margin = _CONE_CROP_MARGIN_FRAC * (w_max - w_min)
    box_min, box_max = w_min - margin, w_max + margin
    out = []
    for f in fixtures:
        if "cone" in f.name.lower():
            n_surf = int(np.asarray(f.surface).shape[0])
            f = _crop_fixture_to_box(f, box_min, box_max)
            mesh = raw_meshes.get(f.name)
            if mesh is not None:
                surf = _resample_envelope_surface_in_box(
                    mesh, offset_mm=0.15, box_min=box_min, box_max=box_max, n=n_surf
                )
                f = replace(f, surface=jnp.asarray(surf, jnp.float32))
        out.append(f)
    return tuple(out)


def build_fixture_sdf_data(runtime) -> tuple[FixtureSDFData, ...]:
    """Build α-wrap SDFs for static fixtures."""
    out: list[FixtureSDFData] = []
    raw_meshes: dict = {}
    for key in fixture_keys_from_runtime(runtime):
        geometry = world_geometry_for_node(runtime, key)
        if geometry is None:
            continue
        mesh = geometry.raw
        if mesh is None:
            continue
        raw_meshes[key] = mesh
        offset_mm = 0.07 if "well" in key.lower() else 0.15
        sdf = build_probe_sdf_from_alpha_wrap(
            mesh,
            offset_mm=offset_mm,
            spacing_mm=0.2,
            strip_shanks_first=False,
        )
        out.append(
            FixtureSDFData(
                name=key,
                grid=jnp.asarray(sdf.grid, dtype=jnp.float32),
                origin=jnp.asarray(sdf.origin, dtype=jnp.float32),
                spacing=jnp.asarray(sdf.spacing, dtype=jnp.float32),
                surface=jnp.asarray(sdf.surface_points, dtype=jnp.float32),
            )
        )
    return _crop_cone_to_well(tuple(out), raw_meshes)


def build_brain_sdf(
    runtime,
    compiled_transforms,
    *,
    asset_key: str = "brain",
    transform_key: str = "headframe_to_lps",
    spacing_mm: float = 0.3,
    pad_mm: float = 3.0,
) -> BrainSDFData:
    """Signed-distance grid (negative inside) for the world-frame brain mesh."""
    import trimesh

    geom = runtime.asset_catalog.get_geometry(asset_key)
    mesh = geom.raw
    R, t = compiled_transforms[transform_key].rotate_translate
    world = trimesh.Trimesh(
        np.asarray(mesh.vertices, np.float64) @ np.asarray(R).T + np.asarray(t),
        np.asarray(mesh.faces),
        process=False,
    )
    sdf = build_probe_sdf(
        world,
        spacing_mm=spacing_mm,
        pad_mm=pad_mm,
        n_surface_points=1,
        sign_type="fwn",
    )
    return BrainSDFData(
        grid=jnp.asarray(sdf.grid, dtype=jnp.float32),
        origin=jnp.asarray(sdf.origin, dtype=jnp.float32),
        spacing=jnp.asarray(sdf.spacing, dtype=jnp.float32),
    )


def maybe_build_brain_sdf(
    runtime, compiled_transforms, *, asset_key: str = "brain", **kw
) -> BrainSDFData | None:
    """:func:`build_brain_sdf` if the config has a brain asset, else ``None``."""
    try:
        runtime.asset_catalog.get_geometry(asset_key)
    except Exception:
        return None
    return build_brain_sdf(runtime, compiled_transforms, asset_key=asset_key, **kw)


# ---------------------------------------------------------------------------
# Coverage data per probe
# ---------------------------------------------------------------------------


def build_coverage_data(
    probes,
    statics,
) -> tuple[GaussianCoverageData, ...]:
    """One Gaussian-mode CoverageData per probe from target_LPS and sigma."""
    out = []
    fallback_geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))
    for st in statics:
        parent = next(p for p in probes if p.name == st.name)
        geom = parent.recording or RECORDING_GEOMETRY.get(parent.kind, fallback_geom)
        active_range = geom.active_ranges_mm[0]
        cd = build_coverage_data_from_probe_context(parent, active_range)
        out.append(cd)
    return tuple(out)
