"""``optimization/geometry/bore_fit.py`` on a synthetic STEP.

A plate with a 15°-tilted elliptical bore and a small straight one beside it.
Needs OpenCascade, which the dev group installs through the ``step`` extra.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("OCP")

from aind_rutter.optimization.geometry import bore_fit
from aind_rutter.optimization.geometry.holes import load_holes

TILT_DEG = 15.0
SEMI_MAJOR, SEMI_MINOR = 0.5, 0.3
SMALL_R = 0.2
SMALL_AT = (1.5, 0.0)  # part (x, y): 1.5 mm anterior of the tilted bore


@pytest.fixture(scope="module")
def step_path(tmp_path_factory) -> Path:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakePrism,
    )
    from OCP.gp import gp_Ax2, gp_Dir, gp_Elips, gp_Pnt, gp_Vec
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    body = BRepPrimAPI_MakeBox(gp_Pnt(-3, -2, -1), gp_Pnt(3, 3, 0)).Shape()
    t = math.radians(TILT_DEG)
    axis = np.array([0.0, math.sin(t), math.cos(t)])
    start = np.array([0.0, 0.0, -0.5]) - 1.5 * axis
    ellipse = gp_Elips(
        gp_Ax2(gp_Pnt(*start), gp_Dir(*axis), gp_Dir(1.0, 0.0, 0.0)),
        SEMI_MAJOR,
        SEMI_MINOR,
    )
    wire = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(ellipse).Edge()).Wire()
    tilted = BRepPrimAPI_MakePrism(
        BRepBuilderAPI_MakeFace(wire).Face(), gp_Vec(*(3.0 * axis))
    ).Shape()
    small = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(*SMALL_AT, -2.0), gp_Dir(0, 0, 1)), SMALL_R, 3.0
    ).Shape()
    body = BRepAlgoAPI_Cut(BRepAlgoAPI_Cut(body, tilted).Shape(), small).Shape()
    path = tmp_path_factory.mktemp("step") / "plate.step"
    writer = STEPControl_Writer()
    writer.Transfer(body, STEPControl_AsIs)
    writer.Write(str(path))
    return path


@pytest.fixture(scope="module")
def bores(step_path):
    return bore_fit.fit_step(step_path)


def test_finds_each_bore_once(bores) -> None:
    assert sorted(round(b.section(b.s_top)[1], 2) for b in bores) == [
        SMALL_R,
        SEMI_MAJOR,
    ]


def test_tilted_bore_axis_and_oval(bores) -> None:
    (b,) = [b for b in bores if b.section(b.s_top)[1] > 0.4]
    t = math.radians(TILT_DEG)
    assert np.allclose(b.axis, [0.0, math.sin(t), math.cos(t)], atol=1e-3)
    for s in (b.s_top, b.s_exit):
        _, a, bb = b.section(s)
        assert a == pytest.approx(SEMI_MAJOR, abs=2e-3)
        assert bb == pytest.approx(SEMI_MINOR, abs=2e-3)
    assert abs(b.major @ [1.0, 0.0, 0.0]) == pytest.approx(
        1.0, abs=1e-3
    )  # major along part x


def test_reading_order_ids(bores) -> None:
    # Dorsal view, anterior up: the small bore (part +x = anterior) is the first row.
    small = next(b for b in bores if b.section(b.s_top)[1] < 0.3)
    assert small.id == 1 and sorted(b.id for b in bores) == [1, 2]


def test_diagram_ids_drop_non_probe_bores(step_path, tmp_path) -> None:
    to_lps = bore_fit.part_to_lps("ALS")
    small_xy = (to_lps @ [*SMALL_AT, 0.0])[:2]
    diagram = tmp_path / "plate.yml"
    diagram.write_text(
        yaml.safe_dump(
            {
                "bores": [
                    {"id": 0, "xy_LPS": [float(x) for x in small_xy], "probe": False},
                    {"id": 7, "xy_LPS": [0.1, 0.0]},
                ]
            }
        )
    )
    out = bore_fit.fit_step(step_path, diagram=diagram)
    assert [b.id for b in out] == [7]
    assert out[0].section(out[0].s_top)[1] == pytest.approx(SEMI_MAJOR, abs=2e-3)


def test_written_file_loads(step_path, tmp_path) -> None:
    out = tmp_path / "plate.holes.yml"
    bore_fit.main([str(step_path), str(out)])
    holes = load_holes(out)
    assert sorted(h.id for h in holes) == [1, 2]
    t = math.radians(TILT_DEG)
    tilted = next(h for h in holes if h.sections[0].a > 0.4)
    assert np.allclose(
        tilted.axis, [math.sin(t), 0.0, math.cos(t)], atol=1e-3
    )  # part (0, sin, cos) in LPS
    assert (
        tilted.sections[0].center @ tilted.axis
        > tilted.sections[1].center @ tilted.axis
    )


def test_packaged_diagram_exists() -> None:
    ref = yaml.safe_load(bore_fit.diagram_path("0283-300-04").read_text())["bores"]
    assert [r["id"] for r in ref] == list(range(14))
    assert [r["id"] for r in ref if not r.get("probe", True)] == [0]
