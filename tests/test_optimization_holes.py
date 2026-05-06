"""Tests for ``aind_low_point.optimization.holes``."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from aind_low_point.optimization.holes import (
    Hole,
    find_hole_by_id,
    load_holes,
)


def _write_yaml(tmp_path, data):
    path = tmp_path / "holes.yml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_holes_basic(tmp_path):
    data = {
        "holes": [
            {
                "id": 0,
                "axis_LPS": [0.0, 0.0, 1.0],
                "ref_point_LPS": [1.0, 2.0, 3.0],
                "sections": [
                    {
                        "s_mm": 0.5,
                        "center_LPS": [1.0, 2.0, 3.5],
                        "a_mm": 0.6,
                        "b_mm": 0.4,
                        "theta_rad": 0.1,
                    },
                    {
                        "s_mm": 0.0,
                        "center_LPS": [1.0, 2.0, 3.0],
                        "a_mm": 0.55,
                        "b_mm": 0.35,
                        "theta_rad": 0.2,
                    },
                    {
                        "s_mm": -0.5,
                        "center_LPS": [1.0, 2.0, 2.5],
                        "a_mm": 0.55,
                        "b_mm": 0.35,
                        "theta_rad": 0.3,
                    },
                ],
            }
        ]
    }
    holes = load_holes(_write_yaml(tmp_path, data))
    assert len(holes) == 1
    h = holes[0]
    assert h.id == 0
    assert isinstance(h, Hole)
    assert np.allclose(h.axis, [0, 0, 1])
    assert np.allclose(h.ref_point, [1, 2, 3])
    assert len(h.sections) == 3
    assert h.sections[0].a == pytest.approx(0.6)
    assert h.sections[-1].b == pytest.approx(0.35)


def test_load_holes_multiple(tmp_path):
    data = {
        "holes": [
            {
                "id": 0,
                "axis_LPS": [0, 0, 1],
                "ref_point_LPS": [0, 0, 0],
                "sections": [
                    {
                        "s_mm": 0.0,
                        "center_LPS": [0, 0, 0],
                        "a_mm": 0.5,
                        "b_mm": 0.3,
                        "theta_rad": 0.0,
                    },
                    {
                        "s_mm": -0.5,
                        "center_LPS": [0, 0, -0.5],
                        "a_mm": 0.5,
                        "b_mm": 0.3,
                        "theta_rad": 0.0,
                    },
                ],
            },
            {
                "id": 1,
                "axis_LPS": [0.1, 0, 0.995],
                "ref_point_LPS": [2, 0, 0],
                "sections": [
                    {
                        "s_mm": 0.0,
                        "center_LPS": [2, 0, 0],
                        "a_mm": 0.6,
                        "b_mm": 0.35,
                        "theta_rad": 1.5,
                    },
                    {
                        "s_mm": -0.5,
                        "center_LPS": [1.95, 0, -0.5],
                        "a_mm": 0.6,
                        "b_mm": 0.35,
                        "theta_rad": 1.5,
                    },
                ],
            },
        ]
    }
    holes = load_holes(_write_yaml(tmp_path, data))
    assert [h.id for h in holes] == [0, 1]


def test_find_hole_by_id_hit(tmp_path):
    data = {
        "holes": [
            {
                "id": 7,
                "axis_LPS": [0, 0, 1],
                "ref_point_LPS": [0, 0, 0],
                "sections": [
                    {
                        "s_mm": 0,
                        "center_LPS": [0, 0, 0],
                        "a_mm": 0.5,
                        "b_mm": 0.3,
                        "theta_rad": 0,
                    },
                    {
                        "s_mm": -1,
                        "center_LPS": [0, 0, -1],
                        "a_mm": 0.5,
                        "b_mm": 0.3,
                        "theta_rad": 0,
                    },
                ],
            }
        ]
    }
    holes = load_holes(_write_yaml(tmp_path, data))
    h = find_hole_by_id(holes, 7)
    assert h.id == 7


def test_find_hole_by_id_miss_raises(tmp_path):
    data = {
        "holes": [
            {
                "id": 0,
                "axis_LPS": [0, 0, 1],
                "ref_point_LPS": [0, 0, 0],
                "sections": [
                    {
                        "s_mm": 0,
                        "center_LPS": [0, 0, 0],
                        "a_mm": 0.5,
                        "b_mm": 0.3,
                        "theta_rad": 0,
                    },
                    {
                        "s_mm": -1,
                        "center_LPS": [0, 0, -1],
                        "a_mm": 0.5,
                        "b_mm": 0.3,
                        "theta_rad": 0,
                    },
                ],
            }
        ]
    }
    holes = load_holes(_write_yaml(tmp_path, data))
    with pytest.raises(KeyError):
        find_hole_by_id(holes, 42)


def test_slot_theta_rad_from_bottom_section(tmp_path):
    """Bottom section's theta is what `slot_theta_rad` returns, since
    the straight bore is the binding constraint and its orientation
    is the canonical slot-major angle."""
    data = {
        "holes": [
            {
                "id": 0,
                "axis_LPS": [0, 0, 1],
                "ref_point_LPS": [0, 0, 0],
                "sections": [
                    {
                        "s_mm": 0.5,
                        "center_LPS": [0, 0, 0.5],
                        "a_mm": 0.7,
                        "b_mm": 0.45,
                        "theta_rad": 1.0,  # chamfer top — different
                    },
                    {
                        "s_mm": 0.0,
                        "center_LPS": [0, 0, 0],
                        "a_mm": 0.6,
                        "b_mm": 0.35,
                        "theta_rad": 1.4,
                    },
                    {
                        "s_mm": -0.5,
                        "center_LPS": [0, 0, -0.5],
                        "a_mm": 0.6,
                        "b_mm": 0.35,
                        "theta_rad": 1.5,  # bottom — canonical
                    },
                ],
            }
        ]
    }
    h = load_holes(_write_yaml(tmp_path, data))[0]
    assert h.slot_theta_rad == pytest.approx(1.5)


def test_slot_major_dir_axis_aligned(tmp_path):
    """For axis = +z, the cap basis is (e1, e2) = (+y, -x). With
    theta = 0 the slot major direction equals e1 = +y."""
    data = {
        "holes": [
            {
                "id": 0,
                "axis_LPS": [0, 0, 1],
                "ref_point_LPS": [0, 0, 0],
                "sections": [
                    {
                        "s_mm": 0,
                        "center_LPS": [0, 0, 0],
                        "a_mm": 0.6,
                        "b_mm": 0.35,
                        "theta_rad": 0.0,
                    },
                    {
                        "s_mm": -0.5,
                        "center_LPS": [0, 0, -0.5],
                        "a_mm": 0.6,
                        "b_mm": 0.35,
                        "theta_rad": 0.0,
                    },
                ],
            }
        ]
    }
    h = load_holes(_write_yaml(tmp_path, data))[0]
    direction = h.slot_major_dir()
    assert np.allclose(direction, [0, 1, 0], atol=1e-9)
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_slot_major_dir_rotated(tmp_path):
    """theta = pi/2 should rotate the slot major from e1 to e2."""
    data = {
        "holes": [
            {
                "id": 0,
                "axis_LPS": [0, 0, 1],
                "ref_point_LPS": [0, 0, 0],
                "sections": [
                    {
                        "s_mm": 0,
                        "center_LPS": [0, 0, 0],
                        "a_mm": 0.6,
                        "b_mm": 0.35,
                        "theta_rad": float(np.pi / 2),
                    },
                    {
                        "s_mm": -0.5,
                        "center_LPS": [0, 0, -0.5],
                        "a_mm": 0.6,
                        "b_mm": 0.35,
                        "theta_rad": float(np.pi / 2),
                    },
                ],
            }
        ]
    }
    h = load_holes(_write_yaml(tmp_path, data))[0]
    # cap_basis([0,0,1]) -> e1=+y, e2=-x. Rotating by pi/2 toward e2 -> -x.
    direction = h.slot_major_dir()
    assert np.allclose(direction, [-1, 0, 0], atol=1e-9)


def test_load_holes_missing_key_raises(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text("not_holes_key: 5\n")
    with pytest.raises(ValueError, match="missing 'holes' key"):
        load_holes(path)


def test_load_holes_real_extracted_yaml():
    """Smoke-test against the actual extracted build5 implant YAML, if
    present. Skips if not — keeps the test suite portable."""
    import os

    path = os.path.expanduser("~/Downloads/0274-P-001.holes.yml")
    if not os.path.exists(path):
        pytest.skip(f"build5 holes YAML not found at {path}")
    holes = load_holes(path)
    assert len(holes) >= 10  # extracted ~15
    for h in holes:
        assert h.axis.shape == (3,)
        assert np.linalg.norm(h.axis) == pytest.approx(1.0, abs=1e-3)
        assert len(h.sections) >= 2
        # Sections should be ordered top-to-bottom.
        s_mms = [sec.center @ h.axis for sec in h.sections]
        assert s_mms == sorted(s_mms, reverse=True)
