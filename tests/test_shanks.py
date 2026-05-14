"""Tests for runtime/shanks.py — shank-tip auto-detection."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from aind_low_point.runtime.shanks import detect_shank_tips_local


def _shank_mesh(
    shank_centers_xy: list[tuple[float, float]],
    shank_length_mm: float = 5.0,
    tip_half_width_mm: float = 0.035,
) -> trimesh.Trimesh:
    """Tiny synthetic probe mesh: each shank is a thin square column.

    Each shank's tip is approximated by 4 vertices at z=0 forming a
    square of side ``2 * tip_half_width_mm`` (matches the ~70 µm width
    of a real NP 2.0 tip), centered at *(cx, cy)*. Body extends up to
    ``z = shank_length_mm``. Faces are not necessary for the tip
    detector (it only reads ``mesh.vertices``); a stub triangle keeps
    trimesh happy.
    """
    verts: list[list[float]] = []
    for cx, cy in shank_centers_xy:
        h = tip_half_width_mm
        for dx, dy in [(-h, -h), (h, -h), (-h, h), (h, h)]:
            verts.append([cx + dx, cy + dy, 0.0])
            verts.append([cx + dx, cy + dy, shank_length_mm])
    v = np.array(verts, dtype=np.float64)
    # one stub triangle so trimesh has something to chew on
    f = np.array([[0, 1, 2]], dtype=np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


class TestDetectShankTips:
    def test_single_shank(self):
        m = _shank_mesh([(0.0, 0.0)])
        tips = detect_shank_tips_local(m)
        assert len(tips) == 1
        np.testing.assert_allclose(tips[0, :2], [0.0, 0.0], atol=1e-6)
        assert tips[0, 2] == pytest.approx(0.0, abs=1e-6)

    def test_np2_four_shank_pitch(self):
        # NP 2.0 four-shank: 250 µm pitch in y
        m = _shank_mesh([(0.0, y) for y in (0.0, 0.25, 0.5, 0.75)])
        tips = detect_shank_tips_local(m)
        assert len(tips) == 4
        # Sorted by y per the function's lexsort; check pitch.
        ys = sorted(tips[:, 1].tolist())
        np.testing.assert_allclose(ys, [0.0, 0.25, 0.5, 0.75], atol=1e-3)
        # Each tip is at z = 0
        np.testing.assert_allclose(tips[:, 2], 0.0, atol=1e-6)

    def test_does_not_split_within_shank_pair(self):
        # The real NP 2.0 mesh has each shank tip spanning ~70 µm in y
        # (two y-positions per shank). Default cluster_radius_mm=0.15
        # must group those within-shank pairs and keep adjacent shanks
        # separate (250 µm pitch).
        m = _shank_mesh(
            [(0.0, y) for y in (0.0, 0.25, 0.5, 0.75)],
            tip_half_width_mm=0.035,  # = 70 µm tip width
        )
        tips = detect_shank_tips_local(m)
        assert len(tips) == 4

    def test_empty_mesh_returns_origin(self):
        m = trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
        tips = detect_shank_tips_local(m)
        assert tips.shape == (1, 3)
        np.testing.assert_allclose(tips[0], [0.0, 0.0, 0.0], atol=1e-9)

    def test_shanks_offset_from_origin(self):
        # Shanks at x=2, y in {0, 0.25}; tip detector should find both.
        m = _shank_mesh([(2.0, 0.0), (2.0, 0.25)])
        tips = detect_shank_tips_local(m)
        assert len(tips) == 2
        np.testing.assert_allclose(sorted(tips[:, 1].tolist()), [0.0, 0.25], atol=1e-3)
        np.testing.assert_allclose(tips[:, 0], 2.0, atol=1e-3)

    def test_below_z_tolerance_collapses(self):
        # Two shanks at slightly different z values — within the
        # default 50 µm z_tolerance both should be picked up.
        verts = []
        for z in (0.0, 0.04):
            for dx, dy in [(-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03), (0.03, 0.03)]:
                verts.append([dx, dy, z])
                verts.append([dx + 0.5, dy, z])
            for dx, dy in [(-0.03, -0.03), (0.03, -0.03)]:
                verts.append([dx, dy, 5.0])
        m = trimesh.Trimesh(
            vertices=np.array(verts),
            faces=np.array([[0, 1, 2]]),
            process=False,
        )
        tips = detect_shank_tips_local(m)
        assert len(tips) == 2
