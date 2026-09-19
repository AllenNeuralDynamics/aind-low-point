"""Coarse-to-fine clearance samples: geometry weights, cells and caching."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from aind_rutter.optimization.clearance.samples import (
    SampleParams,
    build_clearance_samples,
    convex_mean_curvature,
    local_hull_distance,
)

SMALL = SampleParams(fine_n=3000, coarse_n=120, coarse_even=60)


def _column(extents, bottom_z: float = 10.0) -> trimesh.Trimesh:
    """A box subdivided to probe-envelope triangle sizes, bottom face at bottom_z."""
    box = trimesh.creation.box(extents=extents)
    box.apply_translation((0.0, 0.0, bottom_z + extents[2] / 2))
    vertices, faces = trimesh.remesh.subdivide_to_size(box.vertices, box.faces, 0.4)
    return trimesh.Trimesh(vertices, faces)


def test_convex_mean_curvature_of_a_sphere_is_one_over_radius() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=2.0)
    probe = np.asarray(sphere.vertices[:: len(sphere.vertices) // 50])
    curvature = convex_mean_curvature(sphere, probe, radius=0.5)
    assert np.median(curvature) == pytest.approx(0.5, rel=0.15)

    inverted = sphere.copy()
    inverted.invert()
    assert convex_mean_curvature(inverted, probe, radius=0.5).max() == 0.0


def test_local_hull_distance_is_zero_on_the_hull_and_depth_inside() -> None:
    box = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    V = np.asarray(box.vertices)
    on_surface = np.array([[2.0, 0.0, 0.0], [0.0, -2.0, 1.0]])
    inside = np.array([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    kw = {"zmin": -2.0, "bin_mm": 10.0, "window_mm": 10.0}
    np.testing.assert_allclose(local_hull_distance(on_surface, V, **kw), 0.0, atol=1e-9)
    np.testing.assert_allclose(local_hull_distance(inside, V, **kw), [0.5, 2.0])


def test_cells_partition_the_fine_set_and_radii_bound_their_members() -> None:
    body = _column((3.0, 3.0, 40.0))
    s = build_clearance_samples(body, SMALL, use_cache=False)

    assert s.fine.shape == (SMALL.fine_n, 3)
    assert s.coarse.shape == (SMALL.coarse_n, 3)
    members = s.cells[s.cells < SMALL.fine_n]
    np.testing.assert_array_equal(np.sort(members), np.arange(SMALL.fine_n))
    for c in range(SMALL.coarse_n):
        row = s.cells[c][s.cells[c] < SMALL.fine_n]
        if row.size:
            far = np.linalg.norm(s.fine[row] - s.coarse[c], axis=1).max()
            assert s.radius[c] == pytest.approx(far, abs=1e-5)

    height = s.fine[:, 2] - body.vertices[:, 2].min()
    assert np.mean(height <= SMALL.band_mm) == pytest.approx(SMALL.band_share, abs=0.01)


def test_samples_are_deterministic_and_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIND_LOW_POINT_CACHE_DIR", str(tmp_path))
    body = _column((3.0, 3.0, 40.0))
    first = build_clearance_samples(body, SMALL)
    assert list(tmp_path.rglob("c2f_*.npz"))
    again = build_clearance_samples(body, SMALL, use_cache=False)
    cached = build_clearance_samples(body, SMALL)
    for field in ("fine", "coarse", "cells", "radius"):
        np.testing.assert_array_equal(getattr(first, field), getattr(again, field))
        np.testing.assert_array_equal(getattr(first, field), getattr(cached, field))


def test_edges_draw_more_samples_than_flat_faces() -> None:
    extents = np.array([4.0, 4.0, 8.0])
    body = _column(tuple(extents), bottom_z=0.0)
    centre = np.array([0.0, 0.0, extents[2] / 2])
    s = build_clearance_samples(
        body, SampleParams(fine_n=4000, coarse_n=100, coarse_even=50), use_cache=False
    )

    def near_an_edge(points):
        # On a box face one coordinate sits at its half extent; the distance to the
        # face's nearest edge is the smaller of the other two margins.
        margins = np.sort(extents / 2 - np.abs(points - centre), axis=1)
        return np.mean(margins[:, 1] < 0.5)

    uniform, _ = trimesh.sample.sample_surface(body, 4000, seed=0)
    assert near_an_edge(s.fine) > 2 * near_an_edge(uniform)
