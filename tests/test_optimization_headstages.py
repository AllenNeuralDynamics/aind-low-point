"""Tests for ``aind_low_point.optimization.geometry.headstages``."""

from __future__ import annotations

import fcl
import numpy as np
import pytest
import trimesh

from aind_low_point.optimization.geometry.headstages import (
    build_headstage_hull,
    detect_body_region,
    make_fcl_convex,
)


def _make_probe_like_mesh(
    *,
    shank_radius: float = 0.05,
    shank_height: float = 5.0,
    body_half_width_x: float = 5.0,
    body_half_width_y: float = 2.0,
    body_thickness_z: float = 5.0,
    shank_n_stacks: int = 50,
    body_n_stacks: int = 20,
) -> trimesh.Trimesh:
    """Synthetic probe: a thin column (the shank) topped by a wide slab.

    The column z ∈ [0, shank_height] and slab z ∈ [shank_height,
    shank_height + body_thickness_z] are each populated with stacked
    point cloud rings so that vertices are spread across z bins (not
    clumped at face caps like a simple ``box``).
    """
    pieces: list[trimesh.Trimesh] = []
    # Shank: a stack of thin boxes; gives a dense vertex set along z.
    for k in range(shank_n_stacks):
        slab = trimesh.creation.box(
            extents=(2 * shank_radius, 2 * shank_radius, shank_height / shank_n_stacks)
        )
        slab.apply_translation((0.0, 0.0, (k + 0.5) * shank_height / shank_n_stacks))
        pieces.append(slab)
    # Body: a stack of wide thin slabs.
    for k in range(body_n_stacks):
        slab = trimesh.creation.box(
            extents=(
                2 * body_half_width_x,
                2 * body_half_width_y,
                body_thickness_z / body_n_stacks,
            )
        )
        slab.apply_translation(
            (
                0.0,
                0.0,
                shank_height + (k + 0.5) * body_thickness_z / body_n_stacks,
            )
        )
        pieces.append(slab)
    return trimesh.util.concatenate(pieces)


def test_detect_body_region_synthetic():
    """Synthetic mesh: shank 0-5, body 5-10 → body_start_z ≈ 5.0."""
    mesh = _make_probe_like_mesh(shank_height=5.0, body_thickness_z=5.0)
    body_start_z, body_verts = detect_body_region(mesh)
    # The bin boundary detection should land within half a bin of 5.0
    # (bin width is 10/40 = 0.25).
    assert 4.5 <= body_start_z <= 5.5
    # Body region has most of the slab vertices
    assert body_verts.shape[0] >= 4


def test_detect_body_region_returns_vertices_at_or_above_threshold():
    mesh = _make_probe_like_mesh()
    body_start_z, body_verts = detect_body_region(mesh)
    if body_verts.shape[0] > 0:
        assert float(body_verts[:, 2].min()) >= body_start_z - 1e-9


def test_build_headstage_hull_synthetic():
    """Synthetic shank-plus-body mesh produces a non-degenerate hull."""
    mesh = _make_probe_like_mesh(
        body_half_width_x=5.0, body_half_width_y=2.0, body_thickness_z=5.0
    )
    hull = build_headstage_hull(mesh)
    assert hull is not None
    assert len(hull.vertices) >= 4
    assert hull.volume > 0.0
    # The slab volume is 10 × 4 × 5 = 200; the hull should be at least
    # close to that (it may include parts of the shank-body transition).
    assert hull.volume > 150.0


def test_build_headstage_hull_degenerate_single_point():
    """A single-point mesh has no body to hull; return None."""
    mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0]]), faces=np.empty((0, 3), int)
    )
    assert build_headstage_hull(mesh) is None


def test_build_headstage_hull_degenerate_thin_tube():
    """A thin column with no slab returns None (no body detected)."""
    tube = trimesh.creation.cylinder(radius=0.05, height=5.0, sections=8)
    # tube is centered on the origin along +z; that's a degenerate "probe"
    # with no body region above the shanks.
    hull = build_headstage_hull(tube, min_body_verts=20)
    assert hull is None


def test_make_fcl_convex_constructs_collision_object():
    """A small synthetic hull wraps into an FCL CollisionObject."""
    mesh = _make_probe_like_mesh()
    hull = build_headstage_hull(mesh)
    assert hull is not None
    obj = make_fcl_convex(hull)
    assert isinstance(obj, fcl.CollisionObject)


def test_fcl_hull_distance_matches_intuition():
    """Two boxes 10 mm apart along x → FCL signed distance ≈ 6 mm.

    Each box is 4 × 4 × 4 (half-extent 2). When placed at x=0 and x=10,
    the gap between their nearest faces is 10 - 2 - 2 = 6.
    """
    box = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    obj_a = make_fcl_convex(box)
    obj_b = make_fcl_convex(box)
    obj_a.setTransform(fcl.Transform(np.eye(3), np.array([0.0, 0.0, 0.0])))
    obj_b.setTransform(fcl.Transform(np.eye(3), np.array([10.0, 0.0, 0.0])))
    req = fcl.DistanceRequest(enable_signed_distance=True)
    res = fcl.DistanceResult()
    fcl.distance(obj_a, obj_b, req, res)
    assert res.min_distance == pytest.approx(6.0, abs=1e-3)


def test_fcl_hull_distance_negative_when_overlapping():
    """Overlapping boxes give a non-positive signed distance."""
    box = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    obj_a = make_fcl_convex(box)
    obj_b = make_fcl_convex(box)
    obj_a.setTransform(fcl.Transform(np.eye(3), np.array([0.0, 0.0, 0.0])))
    obj_b.setTransform(fcl.Transform(np.eye(3), np.array([2.0, 0.0, 0.0])))
    req = fcl.DistanceRequest(enable_signed_distance=True)
    res = fcl.DistanceResult()
    fcl.distance(obj_a, obj_b, req, res)
    # Boxes overlap by 2 mm (each half-extent 2, centers 2 apart).
    # FCL signed distance should be non-positive.
    assert res.min_distance <= 0.0
