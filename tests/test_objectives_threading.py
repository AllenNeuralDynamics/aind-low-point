"""Direct coverage for the live threading primitives that lost their test when
``test_optimization_joint_rerank.py`` was deleted in the package reorg:

  - ``objectives.statics._build_probe_static`` — builds the per-probe static
    cache, insetting each bore oval by the shank-radius threading margin.
  - ``objectives.threading.threading_g_matrix`` — the bore-fit metric
    (``g <= 0`` ⇔ shank centerline inside the inset oval).

These are central (threading_g_matrix is imported by 5 objectives modules and is
the metric behind the whole threading pipeline), so guard them in CI.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from aind_rutter.optimization.clearance.envelope import shank_half_extents_xy
from aind_rutter.optimization.geometry.holes import Hole, threading_margin_mm
from aind_rutter.optimization.geometry.primitives import HoleSection, cap_basis
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo
from aind_rutter.optimization.objectives.statics import _build_probe_static
from aind_rutter.optimization.objectives.threading import threading_g_matrix


def _vertical_hole(a: float = 0.6, b: float = 0.3) -> Hole:
    """A single-section vertical bore (axis +z, oval centred at the origin)."""
    sec = HoleSection(axis=(0.0, 0.0, 1.0), center=(0.0, 0.0, 0.0), a=a, b=b, theta=0.0)
    return Hole(id=0, axis=(0.0, 0.0, 1.0), ref_point=(0.0, 0.0, 0.0), sections=[sec])


def test_build_probe_static_insets_bore_by_threading_margin() -> None:
    hole = _vertical_hole(a=0.6, b=0.3)
    probe = ProbeStaticInfo(
        name="p",
        target_LPS=np.array([0.0, 0.0, -3.0]),
        kind="2.1",
        shank_tips_local=np.array([[0.0, 0.0, 0.0]]),
    )
    ha = SimpleNamespace(probe_to_hole={"p": 0})
    aa = SimpleNamespace(probe_to_arc_idx={"p": 0}, arc_centroids_deg=(0.0,))
    st = _build_probe_static([probe], [hole], ha, aa)[0]

    margin = threading_margin_mm()
    assert margin > 0.0
    # Oval inset by the shank-radius margin (centerline g<=0 ⇒ real shank clears).
    assert np.allclose(st.section_a, 0.6 - margin)
    assert np.allclose(st.section_b, 0.3 - margin)
    # cap_basis frame is orthonormal and perpendicular to the section axis.
    ax, e1, e2 = st.section_axes[0], st.section_e1[0], st.section_e2[0]
    assert abs(float(np.dot(e1, ax))) < 1e-9
    assert abs(float(np.dot(e2, ax))) < 1e-9
    assert abs(float(np.dot(e1, e2))) < 1e-9


def _g_for_tip(
    tip_xyz,
    a: float = 0.6,
    b: float = 0.3,
    theta: float = 0.0,
    shank_half_xy=None,
    t=(0.0, 0.0, 0.0),
) -> jnp.ndarray:
    """threading_g for a single vertical-bore section with R=I, pose_tip=0."""
    axis = np.array([[0.0, 0.0, 1.0]])
    e1v, e2v = cap_basis(axis[0])
    return threading_g_matrix(
        jnp.eye(3),
        jnp.asarray(t, dtype=jnp.float32),
        jnp.asarray(np.atleast_2d(tip_xyz), dtype=jnp.float32),
        jnp.asarray(axis, dtype=jnp.float32),
        jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),  # center
        jnp.asarray([e1v], dtype=jnp.float32),
        jnp.asarray([e2v], dtype=jnp.float32),
        jnp.asarray([np.cos(theta)], dtype=jnp.float32),
        jnp.asarray([np.sin(theta)], dtype=jnp.float32),
        jnp.asarray([a], dtype=jnp.float32),
        jnp.asarray([b], dtype=jnp.float32),
        shank_half_xy=None
        if shank_half_xy is None
        else jnp.asarray(shank_half_xy, dtype=jnp.float32),
    )


def test_threading_g_matrix_inside_vs_outside_bore() -> None:
    # Shank centerline through the bore centre (x=y=0) → inside the oval (g<=0).
    g_in = _g_for_tip([0.0, 0.0, 0.5])
    assert float(g_in.max()) <= 0.0
    # Shank shifted 2 mm laterally — far outside the 0.6 mm oval → g>0.
    g_out = _g_for_tip([2.0, 0.0, 0.5])
    assert float(g_out.min()) > 0.0


def test_corner_threading_on_a_circle_is_the_far_corner() -> None:
    # Offset dx along x: the far corner (dx + hx, hy) sets g on a round bore.
    r, dx, hx, hy = 0.1, 0.03, 0.012, 0.035
    g = _g_for_tip([dx, 0.0, 0.5], a=r, b=r, shank_half_xy=[hx, hy])
    expected = ((dx + hx) ** 2 + hy**2) / r**2 - 1.0
    assert float(g[0, 0]) == pytest.approx(expected, rel=1e-5)


def test_corner_threading_is_the_worst_corner_centreline() -> None:
    # On a rotated oval, corners equal the centreline check at each corner.
    half = np.array([0.012, 0.035])
    tips = np.array([[0.02, -0.01, 0.0], [0.27, 0.03, 0.0]])
    kw = dict(a=0.2, b=0.08, theta=0.6)
    g = np.asarray(_g_for_tip(tips, shank_half_xy=half, **kw))
    signs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    for k, tip in enumerate(tips):
        corners = [tip + np.r_[sg * half, 0.0] for sg in signs]
        ref = np.asarray(_g_for_tip(np.array(corners), **kw))
        assert g[0, k] == pytest.approx(ref.max(), rel=1e-5)


def test_zero_half_widths_reproduce_the_centreline() -> None:
    tips = [[0.04, -0.02, 0.0], [0.3, 0.05, 0.0]]
    kw = dict(a=0.2, b=0.08, theta=0.6)
    g_line = _g_for_tip(tips, **kw)
    g_zero = _g_for_tip(tips, shank_half_xy=[0.0, 0.0], **kw)
    assert np.array_equal(np.asarray(g_line), np.asarray(g_zero))

    def total(t, half):
        return jnp.sum(_g_for_tip(tips, shank_half_xy=half, t=t, **kw))

    t0 = jnp.asarray([0.01, 0.02, 0.0])
    grad_line = jax.grad(lambda t: jnp.sum(_g_for_tip(tips, t=t, **kw)))(t0)
    grad_zero = jax.grad(total)(t0, jnp.zeros(2))
    np.testing.assert_allclose(grad_zero, grad_line, rtol=1e-6)


def test_shank_half_extents_from_sheets() -> None:
    # Two 24 x 70 um shanks 0.25 mm apart, plus a body the detector ignores.
    shanks = [
        trimesh.creation.box(
            extents=(0.024, 0.07, 10.0),
            transform=trimesh.transformations.translation_matrix((x, 0.0, 5.0)),
        )
        for x in (0.0, 0.25)
    ]
    body = trimesh.creation.box(
        extents=(1.0, 3.0, 8.0),
        transform=trimesh.transformations.translation_matrix((0.1, 0.0, 15.0)),
    )
    mesh = trimesh.util.concatenate([*shanks, body])
    tips = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    np.testing.assert_allclose(shank_half_extents_xy(mesh, tips), [0.012, 0.035])
    assert shank_half_extents_xy(body, tips) is None


def test_corner_threading_insets_by_its_buffer(monkeypatch) -> None:
    monkeypatch.delenv("RUTTER_THREADING_MARGIN_MM", raising=False)
    buffer = threading_margin_mm(corners=True)
    assert 0.0 < buffer < threading_margin_mm()
    probe = ProbeStaticInfo(
        name="p",
        target_LPS=np.array([0.0, 0.0, -3.0]),
        kind="2.1",
        shank_tips_local=np.array([[0.0, 0.0, 0.0]]),
        shank_half_xy=np.array([0.012, 0.035]),
    )
    ha = SimpleNamespace(probe_to_hole={"p": 0})
    aa = SimpleNamespace(probe_to_arc_idx={"p": 0}, arc_centroids_deg=(0.0,))
    st = _build_probe_static([probe], [_vertical_hole(a=0.6, b=0.3)], ha, aa)[0]
    assert np.allclose(st.section_a, 0.6 - buffer)
    assert np.allclose(st.section_b, 0.3 - buffer)
    np.testing.assert_allclose(st.shank_half_xy, [0.012, 0.035])

    monkeypatch.setenv("RUTTER_THREADING_MARGIN_MM", "0.01")
    assert threading_margin_mm(corners=True) == pytest.approx(0.01)
