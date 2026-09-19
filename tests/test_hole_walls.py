"""Wall planes that cut into a bore: loading, transforms, packing, threading checks.

Geometry used throughout: a vertical bore (axis +z) centred at the origin whose
``cap_basis`` frame puts the oval's ``b`` half-axis along x, and a wall plane at
``x = WALL_X`` with solid on the ``+x`` side.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from aind_rutter.optimization.assignment.visibility_atlas import _shank_in_section_jax
from aind_rutter.optimization.geometry.holes import (
    MAX_WALLS_PAD,
    NO_WALL_OFFSET_MM,
    Hole,
    HoleWall,
    load_holes,
    pack_walls,
    threading_margin_mm,
)
from aind_rutter.optimization.geometry.primitives import HoleSection, cap_basis
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo
from aind_rutter.optimization.objectives.constrained import _threading_g_per_probe
from aind_rutter.optimization.objectives.packing import (
    build_batched_probe_static,
)
from aind_rutter.optimization.objectives.reduced import (
    make_batched_reduced_objective,
)
from aind_rutter.optimization.objectives.soft import (
    PACKED_ARG_ORDER,
    PACKED_PER_CAND_KEYS,
)
from aind_rutter.optimization.objectives.soft import (
    _pack_statics as pack_phase1,
)
from aind_rutter.optimization.objectives.statics import (
    JointWeights,
    _build_probe_static,
)
from aind_rutter.optimization.objectives.threading import threading_g_matrix
from aind_rutter.optimization.pipeline.probe_setup import _transform_holes

A_MM, B_MM = 0.6, 0.3
WALL_X = 0.2
AXIS = np.array([0.0, 0.0, 1.0])


def _hole(walls=()) -> Hole:
    sec = HoleSection(axis=AXIS, center=np.zeros(3), a=A_MM, b=B_MM, theta=0.0)
    return Hole(
        id=1, axis=AXIS, ref_point=np.zeros(3), sections=[sec], walls=tuple(walls)
    )


def _wall() -> HoleWall:
    return HoleWall(
        point=np.array([WALL_X, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0])
    )


def _g(tip_x: float, walls=None) -> float:
    """threading g for one vertical shank crossing the section plane at (tip_x, 0)."""
    e1, e2 = cap_basis(AXIS)
    wall_kw = {}
    if walls is not None:
        normals, offsets = pack_walls(walls, 0.0)
        wall_kw = {"w_normals": jnp.asarray(normals), "w_offsets": jnp.asarray(offsets)}
    g = threading_g_matrix(
        jnp.eye(3),
        jnp.zeros(3),
        jnp.asarray([[tip_x, 0.0, -0.5]], jnp.float32),
        jnp.asarray([AXIS], jnp.float32),
        jnp.zeros((1, 3), jnp.float32),
        jnp.asarray([e1], jnp.float32),
        jnp.asarray([e2], jnp.float32),
        jnp.ones(1, jnp.float32),
        jnp.zeros(1, jnp.float32),
        jnp.asarray([A_MM], jnp.float32),
        jnp.asarray([B_MM], jnp.float32),
        **wall_kw,
    )
    return float(np.asarray(g)[0, 0])


def _probe(target_x: float) -> ProbeStaticInfo:
    return ProbeStaticInfo(
        name="p",
        target_LPS=np.array([target_x, 0.0, -3.0]),
        kind="2.1",
        shank_tips_local=np.array([[0.0, 0.0, 0.0]]),
    )


_HA = SimpleNamespace(probe_to_hole={"p": 1})
_AA = SimpleNamespace(probe_to_arc_idx={"p": 0}, arc_centroids_deg=(0.0,))


def test_frame_puts_b_half_axis_along_x() -> None:
    # The expected values below rely on the oval edge along x sitting at b.
    assert _g(B_MM) == pytest.approx(0.0, abs=1e-5)


def test_wall_blocks_the_far_side_of_the_oval() -> None:
    # x = 0.25 is inside the oval (b = 0.3) but 0.05 mm into the wall.
    assert _g(0.25) < 0.0
    assert _g(0.25, [_wall()]) == pytest.approx(2 * 0.05 / B_MM, rel=1e-5)


def test_wall_leaves_points_far_on_the_open_side_unchanged() -> None:
    assert _g(-0.25, [_wall()]) == _g(-0.25)


def test_padded_walls_are_an_exact_no_op() -> None:
    for x in (-0.4, -0.1, 0.0, 0.15, 0.29, 0.5):
        assert _g(x, []) == _g(x)


def test_pack_walls_insets_by_margin_and_pads() -> None:
    normals, offsets = pack_walls([_wall()], margin_mm=0.07)
    assert normals.shape == (MAX_WALLS_PAD, 3)
    assert offsets[0] == pytest.approx(WALL_X - 0.07)
    assert not normals[1].any() and offsets[1] == NO_WALL_OFFSET_MM
    with pytest.raises(ValueError):
        pack_walls([_wall()] * (MAX_WALLS_PAD + 1), margin_mm=0.0)


def test_load_holes_reads_walls_and_normalises(tmp_path) -> None:
    sec = {
        "s_mm": 0.0,
        "center_LPS": [0, 0, 0],
        "a_mm": A_MM,
        "b_mm": B_MM,
        "theta_rad": 0.0,
    }
    doc = {
        "holes": [
            {
                "id": 1,
                "axis_LPS": [0, 0, 1],
                "ref_point_LPS": [0, 0, 0],
                "sections": [sec],
                "walls": [{"point_LPS": [WALL_X, 0, 0], "normal_LPS": [2.0, 0, 0]}],
            },
            {
                "id": 2,
                "axis_LPS": [0, 0, 1],
                "ref_point_LPS": [0, 0, 0],
                "sections": [sec],
            },
        ]
    }
    path = tmp_path / "holes.yml"
    path.write_text(yaml.safe_dump(doc))
    with_wall, without = load_holes(path)
    assert len(with_wall.walls) == 1
    np.testing.assert_allclose(with_wall.walls[0].normal, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(with_wall.walls[0].point, [WALL_X, 0.0, 0.0])
    assert without.walls == ()


def test_transform_holes_moves_walls_rigidly() -> None:
    c, s = np.cos(0.7), np.sin(0.7)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([1.0, -2.0, 0.5])
    (moved,) = _transform_holes([_hole([_wall()])], R, t)
    n0, d0 = pack_walls(_hole([_wall()]).walls, 0.0)
    n1, d1 = pack_walls(moved.walls, 0.0)
    p = np.array([0.13, -0.4, 0.2])
    # pack_walls stores float32, so compare to float32 resolution.
    assert float(n1[0] @ (R @ p + t) - d1[0]) == pytest.approx(
        float(n0[0] @ p - d0[0]), abs=1e-6
    )


def test_build_probe_static_carries_margin_inset_walls() -> None:
    st = _build_probe_static([_probe(0.0)], [_hole([_wall()])], _HA, _AA)[0]
    assert st.wall_normals.shape == (MAX_WALLS_PAD, 3)
    assert st.wall_offsets[0] == pytest.approx(WALL_X - threading_margin_mm())


def test_phase1_pack_passes_walls_per_candidate() -> None:
    assert {"w_normals", "w_offsets"} <= set(PACKED_ARG_ORDER)
    assert {"w_normals", "w_offsets"} <= PACKED_PER_CAND_KEYS
    st = _build_probe_static([_probe(0.0)], [_hole([_wall()])], _HA, _AA)[0]
    packed = pack_phase1([st], 1, build_sdf=False, build_table=False)
    assert packed["w_offsets"].shape == (1, MAX_WALLS_PAD)
    assert float(packed["w_offsets"][0, 0]) == pytest.approx(
        WALL_X - threading_margin_mm()
    )


def test_phase2_threading_sees_the_wall() -> None:
    # Shank at x = 0.15: inside the margin-inset oval (b' = 0.23) but past the
    # margin-inset wall (x = 0.13).
    m = threading_margin_mm()
    st = _build_probe_static([_probe(0.0)], [_hole([_wall()])], _HA, _AA)[0]
    packed = pack_phase1([st], 1, build_sdf=False, build_table=False)
    g, valid = _threading_g_per_probe(
        jnp.eye(3)[None],
        jnp.asarray([[0.15, 0.0, -1.0]], jnp.float32),
        packed["tips_local"],
        packed["s_axes"],
        packed["s_centers"],
        packed["s_e1"],
        packed["s_e2"],
        packed["s_cos"],
        packed["s_sin"],
        packed["s_a"],
        packed["s_b"],
        packed["section_mask"],
        packed["shank_mask"],
        packed["w_normals"],
        packed["w_offsets"],
        shaft_len=10.0,
    )
    expected = 2 * (0.15 - (WALL_X - m)) / (B_MM - m)
    assert float(g[0, 0, 0]) == pytest.approx(expected, rel=1e-4)
    assert float(valid[0, 0, 0]) == 1.0


def test_batched_reduced_objective_penalises_the_wall() -> None:
    # Vertical pose (ap = ml = 0) puts the single shank at the target's x.
    m = threading_margin_mm()
    weights = JointWeights()
    y = jnp.asarray([[0.0, 0.0, 1.0, 0.0]], jnp.float32)
    objs = []
    for walls in ((), (_wall(),)):
        bs = build_batched_probe_static(
            [(_HA, _AA)], [_probe(0.15)], [_hole(walls)], n_arcs=1
        )
        assert bs.wall_normals.shape == (1, 1, MAX_WALLS_PAD, 3)
        obj_batched, _ = make_batched_reduced_objective(bs, weights)
        objs.append(float(obj_batched(y, bs)[0]))
    g_wall = 2 * (0.15 - (WALL_X - m)) / (B_MM - m)
    assert objs[1] - objs[0] == pytest.approx(
        weights.lambda_thread * g_wall**2, rel=1e-3
    )


def test_atlas_shank_check_respects_walls() -> None:
    e1, e2 = cap_basis(AXIS)
    normals, offsets = pack_walls([_wall()], 0.0)

    def inside(x: float, with_wall: bool) -> bool:
        kw = (
            {"wall_normals": jnp.asarray(normals), "wall_offsets": jnp.asarray(offsets)}
            if with_wall
            else {}
        )
        return bool(
            _shank_in_section_jax(
                jnp.asarray([x, 0.0, -1.0]),
                jnp.asarray(AXIS, jnp.float32),
                jnp.zeros(3),
                jnp.asarray(AXIS, jnp.float32),
                jnp.asarray(e1, jnp.float32),
                jnp.asarray(e2, jnp.float32),
                A_MM,
                B_MM,
                0.0,
                oval_slack=0.0,
                **kw,
            )
        )

    assert inside(0.25, with_wall=False)
    assert not inside(0.25, with_wall=True)
    assert inside(-0.25, with_wall=True)
