"""The ``step`` loader: STEP solids to closed LPS meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

pytest.importorskip("OCP")

from aind_rutter.build.canonicalize import _apply_canonicalization_mesh
from aind_rutter.build.loaders import load_geometry
from aind_rutter.build.step import (
    _close_holes,
    _drop_degenerate_and_doubled,
)
from aind_rutter.config.models_catalog import extension_defaults

# Asymmetric on every axis, so a swapped or flipped axis moves the bounds.
BOX_MIN, BOX_MAX = (-1.0, -2.0, -3.0), (4.0, 5.0, 0.5)
BORE_CENTRE, BORE_RADIUS = (1.0, 1.0), 0.6


def _write_plate(path: Path) -> float:
    """A box with a vertical round bore; returns the solid's exact volume."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.GProp import GProp_GProps
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    box = BRepPrimAPI_MakeBox(gp_Pnt(*BOX_MIN), gp_Pnt(*BOX_MAX)).Shape()
    axis = gp_Ax2(gp_Pnt(*BORE_CENTRE, BOX_MIN[2] - 1), gp_Dir(0, 0, 1))
    bore = BRepPrimAPI_MakeCylinder(axis, BORE_RADIUS, 10.0).Shape()
    plate = BRepAlgoAPI_Cut(box, bore).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(plate, STEPControl_AsIs)
    writer.Write(str(path))
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(plate, props)
    return props.Mass()


@pytest.fixture(scope="module")
def plate(tmp_path_factory) -> tuple[Path, float]:
    path = tmp_path_factory.mktemp("step") / "plate.STEP"
    return path, _write_plate(path)


def test_closed_solid_meshes_watertight_and_outward(plate):
    path, volume = plate
    mesh = load_geometry(path, "step")
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(volume, rel=1e-3)
    np.testing.assert_allclose(mesh.bounds, [BOX_MIN, BOX_MAX], atol=1e-9)


def test_deflection_bounds_the_bore_error(plate):
    path, volume = plate
    height = BOX_MAX[2] - BOX_MIN[2]
    excess = {}
    for deflection in (0.05, 0.002):
        # A loose angular limit, which would otherwise govern so small a bore.
        mesh = load_geometry(
            path, "step", linear_deflection_mm=deflection, angular_deflection_rad=1.0
        )
        # Chords cut inside the bore, adding at most a band of sag depth.
        excess[deflection] = mesh.volume - volume
        bound = 2 * np.pi * BORE_RADIUS * deflection * height
        assert -1e-9 <= excess[deflection] <= bound
    assert excess[0.002] < excess[0.05]


def test_als_canonicalization_reaches_lps(plate):
    path, _ = plate
    mesh = _apply_canonicalization_mesh(load_geometry(path, "step"), "ALS", 1.0)
    # Part x is anterior, y left, z superior; LPS is (y, -x, z).
    lo, hi = np.array(BOX_MIN), np.array(BOX_MAX)
    expected = [[lo[1], -hi[0], lo[2]], [hi[1], -lo[0], hi[2]]]
    np.testing.assert_allclose(mesh.bounds, expected, atol=1e-9)


@pytest.mark.parametrize("name", ["part.step", "part.STEP", "part.stp", "PART.Stp"])
def test_extension_selects_the_step_loader(name):
    assert extension_defaults(name) == {"kind": "mesh", "loader": "step"}


def test_unreadable_file_is_refused(tmp_path):
    path = tmp_path / "bad.step"
    path.write_text("not a STEP file\n")
    with pytest.raises(ValueError, match="not a readable STEP file"):
        load_geometry(path, "step")


def test_repair_closes_an_irregular_gap_and_drops_a_fin():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    volume = mesh.volume
    # Two adjacent triangles out: a four-edge gap.
    shared = mesh.face_adjacency[0]
    faces = np.delete(mesh.faces, shared, axis=0)
    # A triangle doubled with opposite winding: a zero-volume fin.
    faces = np.vstack([faces, faces[:1, ::-1]])
    broken = trimesh.Trimesh(mesh.vertices, faces, process=False)
    assert not broken.is_watertight

    _drop_degenerate_and_doubled(broken)
    repaired = _close_holes(broken)
    assert repaired.is_watertight
    assert repaired.is_winding_consistent
    assert repaired.volume == pytest.approx(volume, rel=0.02)
