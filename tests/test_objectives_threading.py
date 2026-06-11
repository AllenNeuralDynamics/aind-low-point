"""Direct coverage for the live threading primitives that lost their test when
``test_optimization_joint_rerank.py`` was deleted in the package reorg:

  - ``objectives.probe_static._build_probe_static`` — builds the per-probe static
    cache, insetting each bore oval by the shank-radius threading margin.
  - ``objectives.reduced_jax.threading_g_matrix`` — the bore-fit metric
    (``g <= 0`` ⇔ shank centerline inside the inset oval).

These are central (threading_g_matrix is imported by 5 objectives modules and is
the metric behind the whole threading pipeline), so guard them in CI.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.geometry.holes import Hole, threading_margin_mm
from aind_low_point.optimization.geometry.primitives import HoleSection, cap_basis
from aind_low_point.optimization.geometry.probes import ProbeStaticInfo
from aind_low_point.optimization.objectives.probe_static import _build_probe_static
from aind_low_point.optimization.objectives.reduced_jax import threading_g_matrix


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


def _g_for_tip(tip_xyz, a: float = 0.6, b: float = 0.3) -> np.ndarray:
    """threading_g for a single vertical-bore section with R=I, pose_tip=0."""
    axis = np.array([[0.0, 0.0, 1.0]])
    e1v, e2v = cap_basis(axis[0])
    return np.asarray(
        threading_g_matrix(
            jnp.eye(3),
            jnp.zeros(3),
            jnp.asarray([tip_xyz], dtype=jnp.float32),
            jnp.asarray(axis, dtype=jnp.float32),
            jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),  # center
            jnp.asarray([e1v], dtype=jnp.float32),
            jnp.asarray([e2v], dtype=jnp.float32),
            jnp.asarray([1.0], dtype=jnp.float32),  # cos theta
            jnp.asarray([0.0], dtype=jnp.float32),  # sin theta
            jnp.asarray([a], dtype=jnp.float32),
            jnp.asarray([b], dtype=jnp.float32),
        )
    )


def test_threading_g_matrix_inside_vs_outside_bore() -> None:
    # Shank centerline through the bore centre (x=y=0) → inside the oval (g<=0).
    g_in = _g_for_tip([0.0, 0.0, 0.5])
    assert float(g_in.max()) <= 0.0
    # Shank shifted 2 mm laterally — far outside the 0.6 mm oval → g>0.
    g_out = _g_for_tip([2.0, 0.0, 0.5])
    assert float(g_out.min()) > 0.0
