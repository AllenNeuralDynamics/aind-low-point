"""Coarse-to-fine body clearance kernel against brute-force lookups."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from aind_rutter.optimization.clearance.kernels import (
    body_body_pair_clearance_c2f,
    body_shank_box_clearance_c2f,
    trilinear_sdf,
    trilinear_sdf_stacked,
)
from aind_rutter.optimization.clearance.samples import (
    SampleParams,
    build_clearance_samples,
)
from aind_rutter.optimization.clearance.sweep import (
    build_padded_probe_tables,
    swept_pair_clearances,
)
from aind_rutter.optimization.clearance.voxel_sdf import build_probe_sdf

PAD_VOXELS = 3
OFF_GRID = 1e3


def _boundary_min(grid: np.ndarray) -> float:
    faces = [grid[0], grid[-1], grid[:, 0], grid[:, -1], grid[:, :, 0], grid[:, :, -1]]
    return float(min(f.min() for f in faces))


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def tables():
    """Two probe kinds (a column and a wider column), grids padded to one shape."""
    kinds = []
    for extents in ((3.0, 3.0, 20.0), (4.0, 2.0, 20.0)):
        body = trimesh.creation.box(extents=extents)
        body.apply_translation((0.0, 0.0, 20.0))
        sdf = build_probe_sdf(
            body, spacing_mm=0.25, n_surface_points=500, use_cache=False,
            sign_type="pseudonormal",
        )  # fmt: skip
        samples = build_clearance_samples(
            body,
            SampleParams(fine_n=4000, coarse_n=200, coarse_even=100),
            use_cache=False,
        )
        kinds.append((sdf, samples))
    shape = np.max([s.grid.shape for s, _ in kinds], axis=0) + PAD_VOXELS
    width = max(c.cells.shape[1] for _, c in kinds)
    n_fine = kinds[0][1].fine.shape[0]

    def pad_grid(g):
        return np.pad(g, [(0, n - m) for n, m in zip(shape, g.shape)], mode="edge")

    def pad_cells(c):
        return np.pad(c, ((0, 0), (0, width - c.shape[1])), constant_values=n_fine)

    return {
        "sdfs": [s for s, _ in kinds],
        "samples": [c for _, c in kinds],
        "grids": jnp.asarray(np.stack([pad_grid(s.grid) for s, _ in kinds])),
        "origins": jnp.asarray(np.stack([s.origin for s, _ in kinds]), jnp.float32),
        "spacings": jnp.asarray([s.spacing for s, _ in kinds], jnp.float32),
        "n_reals": jnp.asarray([s.grid.shape for s, _ in kinds], jnp.int32),
        "outside_min": jnp.asarray([_boundary_min(s.grid) for s, _ in kinds]),
        "coarse": jnp.asarray(np.stack([c.coarse for _, c in kinds])),
        "fine": jnp.asarray(np.stack([c.fine for _, c in kinds])),
        "cells": jnp.asarray(np.stack([pad_cells(c.cells) for _, c in kinds])),
        "radius": jnp.asarray(np.stack([c.radius for _, c in kinds])),
    }


def _lookup(sdf, points):
    return np.asarray(
        trilinear_sdf(
            jnp.asarray(sdf.grid), sdf.origin, sdf.spacing, jnp.asarray(points)
        )
    )


def test_stacked_lookup_matches_per_grid_lookup(tables) -> None:
    rng = np.random.default_rng(0)
    for kind, sdf in enumerate(tables["sdfs"]):
        lo, hi = sdf.bbox_min - 1.0, sdf.bbox_max + 1.0  # includes out-of-bounds points
        points = rng.uniform(lo, hi, size=(2000, 3)).astype(np.float32)
        stacked = trilinear_sdf_stacked(
            tables["grids"], tables["origins"], tables["spacings"], tables["n_reals"],
            jnp.asarray(kind), jnp.asarray(points),
        )  # fmt: skip
        np.testing.assert_allclose(np.asarray(stacked), _lookup(sdf, points), atol=1e-5)


def _pose_pair():
    R_a, t_a = np.eye(3), np.zeros(3)
    R_b, t_b = _rot_z(12.0), np.array([3.9, 0.4, 1.0])
    return R_a, t_a, R_b, t_b


def _brute(tables, R_a, t_a, R_b, t_b, kind_a=0, kind_b=1):
    """All coarse and fine values per direction, plus each cell's members' values."""
    sa, sb = tables["samples"][kind_a], tables["samples"][kind_b]
    da, db = tables["sdfs"][kind_a], tables["sdfs"][kind_b]

    def into(p, R1, t1, R2, t2):
        return (p @ R1.T + t1 - t2) @ R2

    coarse = np.concatenate(
        [_lookup(da, into(sb.coarse, R_b, t_b, R_a, t_a)),
         _lookup(db, into(sa.coarse, R_a, t_a, R_b, t_b))]
    )  # fmt: skip
    fine_b_in_a = _lookup(da, into(sb.fine, R_b, t_b, R_a, t_a))
    fine_a_in_b = _lookup(db, into(sa.fine, R_a, t_a, R_b, t_b))
    radius = np.concatenate([sb.radius, sa.radius])
    floor = np.concatenate(
        [np.full(len(sb.radius), _boundary_min(da.grid)),
         np.full(len(sa.radius), _boundary_min(db.grid))]
    )  # fmt: skip
    ranked = np.where(coarse >= OFF_GRID, floor, coarse) - radius
    cell_mins = []
    for c in range(len(radius)):
        side, row = (sb, c) if c < len(sb.radius) else (sa, c - len(sb.radius))
        values = fine_b_in_a if side is sb else fine_a_in_b
        members = side.cells[row][side.cells[row] < len(side.fine)]
        cell_mins.append(values[members].min() if members.size else np.inf)
    return (
        coarse,
        ranked,
        np.array(cell_mins),
        min(fine_b_in_a.min(), fine_a_in_b.min()),
    )


def _c2f(tables, pose, n_cells, kind_a=0, kind_b=1):
    R_a, t_a, R_b, t_b = (jnp.asarray(v, jnp.float32) for v in pose)
    return body_body_pair_clearance_c2f(
        R_a, t_a, R_b, t_b, jnp.asarray(kind_a), jnp.asarray(kind_b),
        tables["grids"], tables["origins"], tables["spacings"], tables["n_reals"],
        tables["outside_min"],
        tables["coarse"], tables["fine"], tables["cells"], tables["radius"],
        n_cells=n_cells,
    )  # fmt: skip


def test_refining_every_cell_reproduces_the_fine_minimum(tables) -> None:
    pose = _pose_pair()
    coarse, _, _, fine_min = _brute(tables, *pose)
    n_all = len(coarse)
    hard, soft = _c2f(tables, pose, n_all)
    assert float(hard) == pytest.approx(min(coarse.min(), fine_min), abs=1e-5)
    assert float(soft) <= float(hard) + 1e-6


def test_top_cells_follow_the_lipschitz_bound(tables) -> None:
    pose = _pose_pair()
    coarse, bound, cell_mins, fine_min = _brute(tables, *pose)
    chosen = np.argsort(bound)[:16]
    expected = min(coarse.min(), cell_mins[chosen].min())
    hard, _ = _c2f(tables, pose, 16)
    assert float(hard) == pytest.approx(expected, abs=1e-5)
    assert float(hard) >= min(coarse.min(), fine_min) - 1e-5
    # The bound holds, including cells whose coarse point is off the grid: no cell
    # member on the grid reads below its cell's bound.
    on_grid = cell_mins < OFF_GRID
    assert on_grid.any()
    assert np.all(cell_mins[on_grid] >= bound[on_grid] - 1e-4)


def test_kernel_jits_vmaps_and_has_finite_gradients(tables) -> None:
    R_a, t_a, R_b, t_b = (jnp.asarray(v, jnp.float32) for v in _pose_pair())

    def soft(tb, kb):
        return body_body_pair_clearance_c2f(
            R_a, t_a, R_b, tb, jnp.asarray(0), kb,
            tables["grids"], tables["origins"], tables["spacings"], tables["n_reals"],
            tables["outside_min"],
            tables["coarse"], tables["fine"], tables["cells"], tables["radius"],
        )[1]  # fmt: skip

    batch_t = jnp.stack([t_b, t_b + jnp.asarray([0.5, 0.0, 0.0])])
    batch_k = jnp.asarray([1, 0])
    values = jax.jit(jax.vmap(soft))(batch_t, batch_k)
    assert values.shape == (2,) and bool(jnp.all(jnp.isfinite(values)))
    grad = jax.jit(jax.grad(soft))(t_b, jnp.asarray(1))
    assert bool(jnp.all(jnp.isfinite(grad))) and float(jnp.abs(grad).sum()) > 0


def _probe_table(tables, clearance):
    sdfs = tables["sdfs"]
    obb = tuple(jnp.zeros((0, 3), jnp.float32) for _ in sdfs)
    return build_padded_probe_tables(
        tuple(jnp.asarray(s.grid) for s in sdfs),
        tuple(jnp.asarray(s.origin, jnp.float32) for s in sdfs),
        tuple(jnp.float32(s.spacing) for s in sdfs),
        tuple(jnp.asarray(s.surface_points, jnp.float32) for s in sdfs),
        obb,
        obb,
        clearance=clearance,
    )


def _sweep_body_body(table, pose):
    R_a, t_a, R_b, t_b = (np.asarray(v, np.float32) for v in pose)
    hard, soft = swept_pair_clearances(
        jnp.asarray(np.stack([R_a, R_b])),
        jnp.asarray(np.stack([t_a, t_b])),
        table,
        jnp.asarray([0]),
        jnp.asarray([1]),
        beta=20.0,
        top_k_body_body=16,
        top_k_body_shank=8,
        top_k_shank_shank=8,
    )
    return float(hard[0, 0]), float(soft[0, 0])


def test_pair_sweep_uses_coarse_to_fine_samples_when_the_table_has_them(tables) -> None:
    pose = _pose_pair()
    table = _probe_table(tables, clearance=tuple(tables["samples"]))
    assert "c2f_fine" in table
    hard, soft = _sweep_body_body(table, pose)
    ref_hard, ref_soft = _c2f(tables, pose, 16)
    assert hard == pytest.approx(float(ref_hard), abs=1e-5)
    assert soft == pytest.approx(float(ref_soft), abs=1e-5)

    uniform = _probe_table(tables, clearance=None)
    assert "c2f_fine" not in uniform
    assert np.isfinite(_sweep_body_body(uniform, pose)).all()


def test_table_rejects_a_real_probe_kind_without_samples(tables) -> None:
    with pytest.raises(ValueError, match="lacks coarse-to-fine samples"):
        _probe_table(tables, clearance=(tables["samples"][0], None))


SHANK_A = (np.array([[0.0, 0.0, 5.0]]), np.array([[0.5, 0.04, 5.0]]))
SHANK_B = (np.array([[0.0, 0.0, 5.0]]), np.array([[0.6, 0.04, 5.0]]))


def _shank_args(tables, kind_a=0, kind_b=1):
    return (
        tables["coarse"], tables["fine"], tables["cells"], tables["radius"],
        jnp.asarray(SHANK_A[0], jnp.float32), jnp.asarray(SHANK_A[1], jnp.float32),
        jnp.asarray(SHANK_B[0], jnp.float32), jnp.asarray(SHANK_B[1], jnp.float32),
    )  # fmt: skip


def _box_distance(points, R, t, centers, halves):
    """Distance to the nearest box, mirroring obb_sdf in numpy."""
    local = (points - t) @ R
    q = np.abs(local[:, None, :] - centers[None]) - halves[None]
    outside = np.sqrt((np.maximum(q, 0.0) ** 2).sum(-1) + 1e-12)
    return (outside + np.minimum(q.max(-1), 0.0)).min(axis=1)


def _brute_shank(tables, pose, kind_a=0, kind_b=1):
    R_a, t_a, R_b, t_b = pose
    sa, sb = tables["samples"][kind_a], tables["samples"][kind_b]
    coarse, cell_mins, radius = [], [], []
    for side, R_src, t_src, R_dst, t_dst, boxes in (
        (sb, R_b, t_b, R_a, t_a, SHANK_A),
        (sa, R_a, t_a, R_b, t_b, SHANK_B),
    ):
        coarse.append(
            _box_distance(side.coarse @ R_src.T + t_src, R_dst, t_dst, *boxes)
        )
        fine = _box_distance(side.fine @ R_src.T + t_src, R_dst, t_dst, *boxes)
        radius.append(side.radius)
        for row in side.cells:
            members = row[row < len(side.fine)]
            cell_mins.append(fine[members].min() if members.size else np.inf)
    return np.concatenate(coarse), np.concatenate(radius), np.array(cell_mins)


def _c2f_shank(tables, pose, n_cells):
    R_a, t_a, R_b, t_b = (jnp.asarray(v, jnp.float32) for v in pose)
    return body_shank_box_clearance_c2f(
        R_a, t_a, R_b, t_b, jnp.asarray(0), jnp.asarray(1),
        *_shank_args(tables), n_cells=n_cells,
    )  # fmt: skip


def test_shank_box_refining_every_cell_reproduces_the_fine_minimum(tables) -> None:
    pose = _pose_pair()
    coarse, _, cell_mins = _brute_shank(tables, pose)
    hard, soft = _c2f_shank(tables, pose, len(coarse))
    assert float(hard) == pytest.approx(min(coarse.min(), cell_mins.min()), abs=1e-4)
    assert float(soft) <= float(hard) + 1e-6


def test_shank_box_top_cells_follow_the_bound(tables) -> None:
    pose = _pose_pair()
    coarse, radius, cell_mins = _brute_shank(tables, pose)
    chosen = np.argsort(coarse - radius)[:16]
    hard, _ = _c2f_shank(tables, pose, 16)
    assert float(hard) == pytest.approx(
        min(coarse.min(), cell_mins[chosen].min()), abs=1e-4
    )
    finite = np.isfinite(cell_mins)
    assert np.all(cell_mins[finite] >= (coarse - radius)[finite] - 1e-4)


def test_shank_box_kernel_jits_and_has_finite_gradients(tables) -> None:
    R_a, t_a, R_b, t_b = (jnp.asarray(v, jnp.float32) for v in _pose_pair())

    def soft(tb):
        return body_shank_box_clearance_c2f(
            R_a, t_a, R_b, tb, jnp.asarray(0), jnp.asarray(1), *_shank_args(tables)
        )[1]

    grad = jax.jit(jax.grad(soft))(t_b)
    assert bool(jnp.all(jnp.isfinite(grad))) and float(jnp.abs(grad).sum()) > 0


def test_pair_sweep_switches_shank_categories_together(tables) -> None:
    pose = _pose_pair()
    table = _probe_table(tables, clearance=tuple(tables["samples"]))
    uniform = _probe_table(tables, clearance=None)
    R_a, t_a, R_b, t_b = (np.asarray(v, np.float32) for v in pose)
    kw = {
        "pair_a": jnp.asarray([0]), "pair_b": jnp.asarray([1]), "beta": 20.0,
        "top_k_body_body": 16, "top_k_body_shank": 8, "top_k_shank_shank": 8,
    }  # fmt: skip
    Rs, ts = jnp.asarray(np.stack([R_a, R_b])), jnp.asarray(np.stack([t_a, t_b]))
    c2f_hard, _ = swept_pair_clearances(Rs, ts, table, **kw)
    uniform_hard, _ = swept_pair_clearances(Rs, ts, uniform, **kw)
    # These probe kinds carry no shank boxes, so both shank categories read the
    # empty-pool sentinel either way; the body categories still differ.
    assert float(c2f_hard[0, 2]) == pytest.approx(float(uniform_hard[0, 2]))
    assert float(c2f_hard[0, 3]) == pytest.approx(float(uniform_hard[0, 3]))
    assert np.isfinite([float(c2f_hard[0, 0]), float(c2f_hard[0, 1])]).all()
