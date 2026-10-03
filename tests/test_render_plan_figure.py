"""``scripts/render_plan_figure.py``: the sizing and image arithmetic.

Rendering needs a GL context, so these cover what can go wrong without one:
the physical size a panel is drawn at, line weights carried through
supersampling, and the PNG post-processing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_plan_figure.py"


@pytest.fixture(scope="module")
def fig():
    spec = importlib.util.spec_from_file_location("render_plan_figure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def _args(fig, tmp_path: Path, **kw):
    return fig.Args(
        config=tmp_path / "c.yml", plan=tmp_path / "p.yml", out=tmp_path / "o", **kw
    )


def test_a_raster_panel_is_its_printed_size_at_its_dpi(fig, tmp_path) -> None:
    args = _args(fig, tmp_path, width_mm=25.4, height_mm=12.7, dpi=300)
    assert fig.render_size(args) == (300, 150)


def test_a_vector_panel_is_sized_in_points(fig, tmp_path) -> None:
    args = _args(fig, tmp_path, width_mm=25.4, height_mm=12.7, format="pdf")
    assert fig.render_size(args) == (72, 36)


def test_width_without_height_is_refused(fig, tmp_path) -> None:
    with pytest.raises(SystemExit):
        fig.print_size(_args(fig, tmp_path, width_mm=50.0, height_mm=None))


def test_a_stroke_keeps_its_weight_through_supersampling(fig, tmp_path) -> None:
    one = fig.line_px(1.0, _args(fig, tmp_path, dpi=144))
    assert one == pytest.approx(2.0)
    assert fig.line_px(1.0, _args(fig, tmp_path, dpi=144, scale=3)) == 3 * one


def test_union_skips_point_nodes(fig) -> None:
    point = (1.0, 1.0, 2.0, 2.0, 3.0, 3.0)
    box = (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
    other = (-1.0, 0.5, 1.0, 4.0, 2.0, 2.5)
    assert fig._union([point]) is None
    assert fig._union([point, box, other]) == (-1.0, 1.0, 0.0, 4.0, 0.0, 3.0)


def test_downsampling_leaves_no_background_halo(fig, tmp_path) -> None:
    """Transparent pixels carry the background colour in their RGB; filtered
    straight, it bleeds into the edge of the content."""
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., :3] = 255  # white background, fully transparent
    rgba[:, :4] = (0, 0, 0, 255)  # opaque black on the left half
    path = tmp_path / "panel.png"
    Image.fromarray(rgba).save(path)
    fig.resample_to(path, (4, 4), dpi=300)
    with Image.open(path) as out:
        small = np.asarray(out)
        assert out.info["dpi"] == pytest.approx(
            (300, 300), rel=1e-4
        )  # PNG stores pixels per metre
    partial = (small[..., 3] > 0) & (small[..., 3] < 255)
    assert partial.any()
    assert small[partial][:, :3].max() <= 1


def test_frame_fill_reads_the_alpha(fig, tmp_path) -> None:
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[2:4, 2:7, 3] = 255
    path = tmp_path / "panel.png"
    Image.fromarray(rgba).save(path)
    assert fig.frame_fill(path) == pytest.approx((0.10, 0.10))


def test_the_border_keeps_the_panels_dpi(fig, tmp_path) -> None:
    path = tmp_path / "panel.png"
    Image.new("RGB", (40, 30), "white").save(path, dpi=(600, 600))
    fig.draw_border(path, _args(fig, tmp_path, dpi=600, border_pt=0.6))
    with Image.open(path) as out:
        assert out.info["dpi"] == pytest.approx((600, 600), rel=1e-4)
        assert out.getpixel((0, 0)) != (255, 255, 255)
        assert out.getpixel((20, 15)) == (255, 255, 255)


def test_a_geometry_report_is_not_loaded_as_a_plan(fig, tmp_path) -> None:
    path = tmp_path / "report.yml"
    path.write_text(yaml.safe_dump({"plan_export_version": 2}))
    with pytest.raises(SystemExit, match="geometry report"):
        fig.load_plan(path)


def test_a_region_colour_is_found_whatever_its_case(fig) -> None:
    assert fig.target_colour("vCA1") == fig.target_colour("VCA1") == "#2ca02c"
    assert fig.target_colour("pl") == fig.TARGET_COLOURS["PL"]
    assert fig.target_colour("SMT") is None
