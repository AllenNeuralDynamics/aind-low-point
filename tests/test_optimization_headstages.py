"""Tests for ``aind_low_point.optimization.headstages``."""

from __future__ import annotations

import fcl
import numpy as np
import pytest
import trimesh

from aind_low_point.optimization.headstages import (
    build_headstage_hull,
    detect_body_region,
    make_fcl_convex,
)
from aind_low_point.optimization.objective import (
    OptimizerContext,
    ProbeEvaluation,
    VariableLayout,
    pairwise_headstage_clearances,
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
        slab.apply_translation(
            (0.0, 0.0, (k + 0.5) * shank_height / shank_n_stacks)
        )
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


def _dummy_eval(R: np.ndarray, t: np.ndarray) -> ProbeEvaluation:
    from aind_low_point.optimization.geometry import Capsule

    return ProbeEvaluation(
        R=R,
        pose_tip=t,
        shanks=[],
        # Capsule is unused under the hull path but required by the dataclass.
        headstage=Capsule(np.zeros(3), np.array([0, 0, 5.0]), 2.0),
        coverage=0.0,
        threading_gs=np.zeros(0),
    )


def test_pairwise_headstage_clearances_skips_no_hull_pairs():
    """When only one probe has a hull, no valid pair → empty result."""
    box = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    hull_obj = make_fcl_convex(box)
    layout = VariableLayout(arc_ids=("a",), probe_names=("p0", "p1"))
    ctx = OptimizerContext(
        layout=layout, probes=(), headstage_fcl_objs=(hull_obj, None)
    )
    evals = [
        _dummy_eval(np.eye(3), np.zeros(3)),
        _dummy_eval(np.eye(3), np.array([10.0, 0.0, 0.0])),
    ]
    out = pairwise_headstage_clearances(evals, ctx)
    assert out.shape == (0,)


def test_pairwise_headstage_clearances_hull_path_distance():
    """Two boxes via the hull path return a single distance entry."""
    box = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    obj_a = make_fcl_convex(box)
    obj_b = make_fcl_convex(box)
    layout = VariableLayout(arc_ids=("a",), probe_names=("p0", "p1"))
    ctx = OptimizerContext(
        layout=layout, probes=(), headstage_fcl_objs=(obj_a, obj_b)
    )
    evals = [
        _dummy_eval(np.eye(3), np.zeros(3)),
        _dummy_eval(np.eye(3), np.array([10.0, 0.0, 0.0])),
    ]
    out = pairwise_headstage_clearances(evals, ctx)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(6.0, abs=1e-3)


def test_pairwise_headstage_clearances_falls_back_without_ctx():
    """Without ``ctx`` (or with empty ``headstage_fcl_objs``) the legacy
    capsule path is used so existing tests still work."""
    from aind_low_point.optimization.geometry import Capsule

    cap_a = Capsule(np.array([0, 0, 10.0]), np.array([0, 0, 15.0]), 2.0)
    cap_b = Capsule(np.array([4, 0, 10.0]), np.array([4, 0, 15.0]), 2.0)
    evals = [
        ProbeEvaluation(
            R=np.eye(3),
            pose_tip=np.zeros(3),
            shanks=[],
            headstage=cap_a,
            coverage=0.0,
            threading_gs=np.zeros(0),
        ),
        ProbeEvaluation(
            R=np.eye(3),
            pose_tip=np.zeros(3),
            shanks=[],
            headstage=cap_b,
            coverage=0.0,
            threading_gs=np.zeros(0),
        ),
    ]
    out = pairwise_headstage_clearances(evals)
    assert out.shape == (1,)
    # Center-to-center xy distance 4, both radii 2 → clearance 0.
    assert out[0] == pytest.approx(0.0, abs=1e-9)
