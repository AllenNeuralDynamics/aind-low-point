"""Dense shank-box mid-plane samples in the other body's SDF."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from aind_rutter.optimization.clearance.boxes import box_face_fractions
from aind_rutter.optimization.clearance.pairs import (
    body_shank_box_sdf_clearance,
    body_shank_corners_pair_clearance,
)
from aind_rutter.optimization.clearance.voxel_sdf import build_probe_sdf

SHANK_CENTER = np.array([[0.0, 0.0, 4.98]])
SHANK_HALVES = np.array([[0.012, 0.035, 4.984]])
FRACTIONS = box_face_fractions(101, 5)
NO_SHANKS = jnp.zeros((0, 3), jnp.float32)


@pytest.fixture(scope="module")
def body():
    """A 3 x 3 x 20 mm column standing on z = 10, as probe b's body."""
    mesh = trimesh.creation.box(extents=(3.0, 3.0, 20.0))
    mesh.apply_translation((0.0, 0.0, 20.0))
    return build_probe_sdf(
        mesh, spacing_mm=0.1, n_surface_points=500, use_cache=False,
        sign_type="pseudonormal",
    )  # fmt: skip


def _shank_pose(normal, along, face_point, gap):
    """Pose putting the shank's broad face ``gap`` from ``face_point`` along
    ``normal``, its length along ``along`` and its width along world z."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    u = np.asarray(along, float) / np.linalg.norm(along)
    R = np.stack([n, np.cross(u, n), u], axis=1)  # local x, y, z in world
    mid = np.asarray(face_point, float) + (gap + SHANK_HALVES[0, 0]) * n
    return R, mid - R @ SHANK_CENTER[0]


def _clearances(body, R_a, t_a):
    grid = jnp.asarray(body.grid)
    args = (
        jnp.asarray(R_a, jnp.float32), jnp.asarray(t_a, jnp.float32),
        jnp.eye(3), jnp.zeros(3),
        grid, body.origin, body.spacing, grid, body.origin, body.spacing,
        jnp.asarray(SHANK_CENTER, jnp.float32), jnp.asarray(SHANK_HALVES, jnp.float32),
        NO_SHANKS, NO_SHANKS,
    )  # fmt: skip
    dense = body_shank_box_sdf_clearance(*args, FRACTIONS, top_k=16)
    corners = body_shank_corners_pair_clearance(*args, top_k=16)
    return float(dense[0]), float(corners[0])


@pytest.mark.parametrize("gap", [0.05, 0.3])
def test_parallel_face_reads_the_gap(body, gap) -> None:
    R, t = _shank_pose((1, 0, 0), (0, 1, 0), (1.5, 0.0, 20.0), gap)
    dense, _ = _clearances(body, R, t)
    assert dense == pytest.approx(gap, abs=0.02)


def test_contact_between_the_corners_is_seen(body) -> None:
    """Crossing the column's vertical edge mid-shank: the corners are millimetres
    away, the edge is ``gap`` from the shank's middle."""
    diagonal = np.array([1.0, 1.0, 0.0])
    R, t = _shank_pose(diagonal, (1, -1, 0), (1.5, 1.5, 20.0), 0.1)
    dense, corners = _clearances(body, R, t)
    assert dense == pytest.approx(0.1, abs=0.03)
    assert corners > 1.0


def test_overlap_reads_negative_with_finite_gradient(body) -> None:
    R, t = _shank_pose((1, 0, 0), (0, 1, 0), (1.5, 0.0, 20.0), -0.2)
    assert _clearances(body, R, t)[0] < -0.15

    grid = jnp.asarray(body.grid)

    def soft(t_a):
        return body_shank_box_sdf_clearance(
            jnp.asarray(R, jnp.float32), t_a, jnp.eye(3), jnp.zeros(3),
            grid, body.origin, body.spacing, grid, body.origin, body.spacing,
            jnp.asarray(SHANK_CENTER, jnp.float32),
            jnp.asarray(SHANK_HALVES, jnp.float32), NO_SHANKS, NO_SHANKS,
            FRACTIONS, top_k=16,
        )[1]  # fmt: skip

    g = jax.jit(jax.grad(soft))(jnp.asarray(t, jnp.float32))
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(g[0]) > 0.5  # moving away along the face normal clears it
