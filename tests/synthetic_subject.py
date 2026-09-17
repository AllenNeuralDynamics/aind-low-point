"""A complete, machine-independent subject: meshes, points, holes and a config.

Every tracked example config points at meshes under ``/mnt``, so nothing tests
the path from a YAML file to a built runtime. This writes a subject small enough
to build in a second — a spherical brain under a conical well, an implant whose
bars leave a gap over each bore, and two probe meshes.

The geometry is anatomically meaningless; what it reproduces is the *shape* of a
real config: templates, a named transform, a derived target, scene tags that the
optimizer reads, and probe assets keyed ``probe:<kind>`` whose shank tips give
the kinematic pivot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml

# Probe kinds the recording-geometry table knows, so the build derives a pivot.
PROBE_KINDS = ("2.1", "quadbase-alpha")
HEAD_PITCH_DEG = 14.0
WELL_TOP_Z_MM = 10.0
HOLE_IDS = (1, 2)
HOLE_PITCH_MM = 3.0
HOLE_RADIUS_MM = 0.9
# The brain sits under hole 1 so a probe reaching it threads the implant gap.
BRAIN_CENTER_LPS = (HOLE_PITCH_MM, 0.0, -5.0)
SHANK_PITCH_MM = 0.25


def _probe_mesh(n_shanks: int, *, pitch_mm: float = SHANK_PITCH_MM) -> trimesh.Trimesh:
    """A probe pointing down -z: shank tips at z=0, body above them.

    Each shank is a thin box; the body is a wider box above. The tip detector
    clusters the lowest vertices, so the shanks must be separated by more than
    its cluster radius in y.
    """
    parts = []
    for i in range(n_shanks):
        shank = trimesh.creation.box(extents=(0.07, 0.07, 5.0))
        shank.apply_translation((0.0, i * pitch_mm, 2.5))
        parts.append(shank)
    body = trimesh.creation.box(extents=(2.0, 2.0 + n_shanks * pitch_mm, 6.0))
    body.apply_translation((0.0, (n_shanks - 1) * pitch_mm / 2, 8.0))
    parts.append(body)
    return trimesh.util.concatenate(parts)


def _well_mesh() -> trimesh.Trimesh:
    """An open funnel widening with z, revolved from its wall profile.

    Revolving a closed ``(radius, z)`` profile keeps this watertight without a
    boolean: trimesh has no boolean backend in this environment and a difference
    silently returns an empty mesh.
    """
    profile = np.array(
        [[4.8, 0.0], [6.3, 0.0], [10.5, WELL_TOP_Z_MM], [9.0, WELL_TOP_Z_MM]]
    )
    return trimesh.creation.revolve(profile, sections=48)


def _implant_mesh() -> trimesh.Trimesh:
    """Three bars with a gap over each hole, so probes thread between them."""
    bars = []
    for x in (-2.0 * HOLE_PITCH_MM, 0.0, 2.0 * HOLE_PITCH_MM):
        bar = trimesh.creation.box(extents=(2.0, 12.0, 2.0))
        bar.apply_translation((x, 0.0, 1.0))
        bars.append(bar)
    return trimesh.util.concatenate(bars)


def _hole_center(hole_id: int) -> tuple[float, float, float]:
    """Hole centres sit in the gaps between the implant bars."""
    return (HOLE_PITCH_MM * (1 if hole_id == 1 else -1), 0.0, 1.0)


def _write_holes(path: Path) -> None:
    """Hole geometry in implant-local mm, in the shape ``load_holes`` reads."""
    holes = []
    for hole_id in HOLE_IDS:
        cx, cy, _ = _hole_center(hole_id)
        holes.append(
            {
                "id": hole_id,
                "axis_LPS": [0.0, 0.0, 1.0],
                "ref_point_LPS": [cx, cy, 0.0],
                "sections": [
                    {
                        "center_LPS": [cx, cy, z],
                        "a_mm": HOLE_RADIUS_MM,
                        "b_mm": HOLE_RADIUS_MM,
                        "theta_rad": 0.0,
                    }
                    for z in (0.2, 1.8)
                ],
            }
        )
    path.write_text(yaml.safe_dump({"holes": holes}, sort_keys=False))


def _write_points(path: Path, rng: np.random.Generator) -> None:
    points = rng.normal(scale=1.5, size=(64, 3)) + np.array([0.0, 0.0, -4.0])
    lines = ["index,x,y,z"]
    lines += [f"{i},{x:.6f},{y:.6f},{z:.6f}" for i, (x, y, z) in enumerate(points)]
    path.write_text("\n".join(lines) + "\n")


def _config(meshes: Path) -> dict:
    """The config document, mirroring the structure of a real subject's."""
    return {
        "version": 1,
        "paths": {"mesh_path": str(meshes)},
        "transforms": {
            # A real subject gets this from a registration file; a rotation and
            # a shift exercise the same compile path.
            "headframe_to_lps": {
                "sequence": [
                    {
                        "kind": "rotate_euler_deg",
                        "order": "XYZ",
                        "angles_deg": [2.0, 0.0, 0.0],
                    },
                    {"kind": "translate_mm", "delta": [0.1, -0.2, 0.3]},
                ]
            },
            "implant_to_lps": {
                "sequence": [{"kind": "translate_mm", "delta": [0.0, 0.0, 0.5]}]
            },
        },
        "materials": {
            "structure": {"color": "#88CCFF", "opacity": 0.1},
            "implant": {"color": "#FF00FF", "opacity": 0.7},
        },
        "asset_templates": {
            "hardware": {
                "kind": "mesh",
                "role": "geometry",
                "loader": "trimesh",
                "material": {"color": "#E7DDFF", "opacity": 1.0},
                "caps": ["renderable", "collidable"],
                "collision": {"group": "fixture", "mask": ["probe"]},
            },
            "probe": {
                "kind": "mesh",
                "role": "geometry",
                "loader": "trimesh",
                "material": {"color": "#66DD66", "opacity": 1.0},
                "caps": [
                    "renderable",
                    "movable",
                    "collidable",
                    "selectable",
                    "savable",
                ],
                "collision": {"group": "probe", "mask": ["fixture", "probe"]},
            },
        },
        "target_templates": {
            "structure": {
                "reducer": "mesh_center_mass",
                "material_ref": "structure",
                "tags": ["structure", "target"],
                "collision": {"group": "static", "mask": []},
            }
        },
        "assets": [
            {
                "key": "brain",
                "kind": "mesh",
                "role": "anatomy",
                "loader": "trimesh",
                "src": "${paths.mesh_path}/brain.obj",
                "material_ref": "structure",
                "caps": ["renderable"],
                "transform": "headframe_to_lps",
                "scene_tags": ["static"],
            },
            {
                "key": "well",
                "src": "${paths.mesh_path}/well.obj",
                "templates": ["hardware"],
                "scene_tags": ["fixture", "well"],
            },
            {
                "key": "implant",
                "src": "${paths.mesh_path}/implant.obj",
                "templates": ["hardware"],
                "material_ref": "implant",
                "transform": "implant_to_lps",
                "scene_tags": ["fixture", "implant"],
            },
            {
                "key": "retro-targets",
                "kind": "points",
                "role": "anatomy",
                "loader": "csv_points",
                "src": "${paths.mesh_path}/retro_points.csv",
                "caps": ["renderable"],
                "material": {"color": "#FF4444", "opacity": 0.6, "point_size": 3.0},
                "transform": "headframe_to_lps",
                "scene_tags": ["static"],
            },
            *[
                {
                    "key": f"probe:{kind}",
                    "src": f"${{paths.mesh_path}}/probe-{kind}.obj",
                    "templates": ["probe"],
                }
                for kind in PROBE_KINDS
            ],
        ],
        "targets": [
            {
                "derive_from": ["brain"],
                "key_prefix": "target:",
                "templates": ["structure"],
                "transform": "headframe_to_lps",
                "scene_tags": ["static", "target"],
            }
        ],
        "plan": {
            "arcs": {"a": 10.0, "b": -20.0},
            "subject_from_rig": {
                "axis_LPS": [1.0, 0.0, 0.0],
                "angle_deg": HEAD_PITCH_DEG,
            },
            "probes": {
                "P1": {
                    "kind": "2.1",
                    "arc": "a",
                    "spin": 0.0,
                    "slider_ml": 6.0,
                    "past_target_mm": 0.0,
                    "offsets_RA": [0.0, 0.0],
                    "target": {"kind": "catalog", "key": "target:brain"},
                },
                "P2": {
                    "kind": "quadbase-alpha",
                    "arc": "b",
                    "spin": 90.0,
                    "slider_ml": -6.0,
                    "past_target_mm": 0.5,
                    "offsets_RA": [0.0, 0.0],
                    # Under hole 2: RAS R is -LPS x, so this is LPS (-3, 0, -5).
                    "target": {
                        "kind": "inline",
                        "point_RAS": [HOLE_PITCH_MM, 0.0, -5.0],
                    },
                },
            },
        },
    }


@dataclass(frozen=True)
class SyntheticSubject:
    """Where the written subject's files landed."""

    root: Path
    config: Path
    holes: Path


def write_subject(root: Path) -> SyntheticSubject:
    """Write meshes, points, holes and the config under ``root``."""
    meshes = root / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    brain = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    brain.apply_translation(BRAIN_CENTER_LPS)
    brain.export(meshes / "brain.obj")
    _well_mesh().export(meshes / "well.obj")
    _implant_mesh().export(meshes / "implant.obj")
    for kind in PROBE_KINDS:
        _probe_mesh(1 if kind == "2.1" else 4).export(meshes / f"probe-{kind}.obj")
    _write_points(meshes / "retro_points.csv", rng)

    holes_path = root / "implant.holes.yml"
    _write_holes(holes_path)

    config_path = root / "synthetic-config.yml"
    config_path.write_text(yaml.safe_dump(_config(meshes), sort_keys=False))
    return SyntheticSubject(root=root, config=config_path, holes=holes_path)
