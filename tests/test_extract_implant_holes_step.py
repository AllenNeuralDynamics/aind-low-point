"""``scripts/extract_implant_holes_step.py`` on a synthetic STEP.

The part is a plate with a 15°-tilted elliptical bore and a slab fused through
the bore 0.2 mm from its axis, so the channel is cut by a wall with solid beyond
it — the same shape as the edge-cut bores of implant 0283-300-06.

Needs cadquery-ocp, which the project does not depend on::

    uv run --no-project --python 3.13 --with cadquery-ocp==8.0.1.0.0 --with pytest \\
        --with numpy --with scipy --with pyyaml --with trimesh \\
        pytest tests/test_extract_implant_holes_step.py -q
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OCP")
trimesh = pytest.importorskip("trimesh")
yaml = pytest.importorskip("yaml")

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "extract_implant_holes_step.py"
)
TILT_DEG = 15.0
SEMI_MAJOR, SEMI_MINOR = 0.5, 0.3
WALL_X = 0.2


@pytest.fixture(scope="module")
def ext():
    spec = importlib.util.spec_from_file_location("extract_implant_holes_step", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def step_path(tmp_path_factory) -> Path:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
    from OCP.gp import gp_Ax2, gp_Dir, gp_Elips, gp_Pnt, gp_Vec
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    # Asymmetric in y so the OBJ test can tell (x, z, -y) from (x, z, y).
    plate = BRepPrimAPI_MakeBox(gp_Pnt(-3, -2, -1), gp_Pnt(3, 3, 0)).Shape()
    t = math.radians(TILT_DEG)
    axis = np.array([0.0, math.sin(t), math.cos(t)])
    start = np.array([0.0, 0.0, -0.5]) - 1.5 * axis
    ellipse = gp_Elips(
        gp_Ax2(gp_Pnt(*start), gp_Dir(*axis), gp_Dir(1.0, 0.0, 0.0)),
        SEMI_MAJOR,
        SEMI_MINOR,
    )
    wire = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(ellipse).Edge()).Wire()
    bore = BRepPrimAPI_MakePrism(
        BRepBuilderAPI_MakeFace(wire).Face(), gp_Vec(*(3.0 * axis))
    ).Shape()
    body = BRepAlgoAPI_Cut(plate, bore).Shape()
    slab = BRepPrimAPI_MakeBox(
        gp_Pnt(WALL_X, -1.2, -1), gp_Pnt(WALL_X + 0.1, 1.2, 0)
    ).Shape()
    body = BRepAlgoAPI_Fuse(body, slab).Shape()

    path = tmp_path_factory.mktemp("step") / "plate.step"
    writer = STEPControl_Writer()
    writer.Transfer(body, STEPControl_AsIs)
    writer.Write(str(path))
    return path


@pytest.fixture(scope="module")
def extracted(ext, step_path):
    cfg = ext.ExtractConfig(part_frame="ALS")
    to_lps, _ = ext.part_frame_matrices(cfg.part_frame)
    shape = ext.read_step(step_path)
    bores, _ = ext.number_bores(ext.extract_bores(shape, cfg), to_lps, cfg.row_gap_mm)
    return shape, bores, ext.holes_to_yaml(bores, to_lps)


def test_frame_matrices(ext) -> None:
    to_lps, to_asr = ext.part_frame_matrices("ALS")
    x, y, _z = np.eye(3)
    assert np.allclose(to_lps @ x, [0, -1, 0])  # anterior = -P
    assert np.allclose(to_lps @ y, [1, 0, 0])  # left = +L
    assert np.allclose(to_asr @ y, [0, 0, -1])  # ASR z = R = -left
    with pytest.raises(ValueError):
        ext.part_frame_matrices("ARS")  # mirror image


def test_single_bore_axis_and_ovals(ext, extracted) -> None:
    _, bores, data = extracted
    assert len(bores) == 1
    hole = data["holes"][0]
    assert hole["id"] == 1
    t = math.radians(TILT_DEG)
    # part axis (0, sin t, cos t) -> LPS (sin t, 0, cos t)
    assert np.allclose(hole["axis_LPS"], [math.sin(t), 0.0, math.cos(t)], atol=1e-4)

    top, bottom = hole["sections"]
    axis = np.asarray(hole["axis_LPS"])
    assert (
        np.asarray(top["center_LPS"]) @ axis > np.asarray(bottom["center_LPS"]) @ axis
    )
    assert top["s_mm"] < 0 and bottom["s_mm"] < top["s_mm"]
    for sec in (top, bottom):
        assert sec["a_mm"] == pytest.approx(SEMI_MAJOR, abs=2e-3)
        assert sec["b_mm"] == pytest.approx(SEMI_MINOR, abs=2e-3)
        # major axis along part +x = LPS -P; theta is modulo a half turn
        e1, e2 = ext.cap_basis(axis)
        major = math.cos(sec["theta_rad"]) * e1 + math.sin(sec["theta_rad"]) * e2
        assert abs(major @ np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0, abs=1e-3)


def test_wall_plane(extracted) -> None:
    _, _, data = extracted
    hole = data["holes"][0]
    (wall,) = hole["walls"]
    normal = np.asarray(wall["normal_LPS"])
    assert np.allclose(normal, [0.0, -1.0, 0.0], atol=1e-3)  # part +x, into the slab
    for sec in hole["sections"]:
        dist = normal @ (np.asarray(wall["point_LPS"]) - np.asarray(sec["center_LPS"]))
        assert dist == pytest.approx(WALL_X, abs=2e-3)


def test_obj_is_asr(ext, extracted, tmp_path) -> None:
    shape, _, _ = extracted
    _, to_asr = ext.part_frame_matrices("ALS")
    path = tmp_path / "plate.obj"
    ext.write_obj(shape, path, to_asr, 0.01, 0.2)
    mesh = trimesh.load(path, force="mesh")
    # part box (-3,-2,-1)..(3,3,0) -> (x, z, -y)
    assert np.allclose(mesh.bounds, [[-3, -1, -3], [3, 0, 2]], atol=1e-6)
