"""Phase 2's clearance rows are continuous where a probe leaves another's SDF grid,
and the pair category the kernel never measures is constant and dropped."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from aind_rutter.optimization.clearance.voxel_sdf import (
    DEFAULT_PAD_MM,
    DEFAULT_SPACING_MM,
    build_probe_sdf,
)
from aind_rutter.optimization.geometry.holes import Hole
from aind_rutter.optimization.geometry.primitives import HoleSection
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo
from aind_rutter.optimization.objectives.constrained import (
    CLEARANCE_CAP_MM,
    Phase2Weights,
    make_phase2,
)
from aind_rutter.optimization.objectives.feasibility import FeasibilitySpec
from aind_rutter.optimization.objectives.objective_spec import ObjectiveSpec
from aind_rutter.optimization.objectives.slack_layout import PADDED_SLACK
from aind_rutter.optimization.objectives.statics import _build_probe_static

SPEC = FeasibilitySpec(min_clearance_mm=0.2, softmin_correction=True)
TARGETS_X = (0.0, 3.0)
# (arc AP, then (ml, sx, sy, off_R, off_A, depth) per probe); the second probe's
# off_R slides it along x.
POSE = [0.0] + [0.0, 1, 0, 0, 0, 0] + [0.0, 1, 0, 0, 0, 0]
OFF_R = 1 + 6 + 3


@pytest.fixture(scope="module")
def problem():
    """Two single-shank probes on one arc with 2 × 2 × 4 mm box bodies."""
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 4.0))
    mesh.apply_translation((0.0, 0.0, 5.0))
    body = replace(
        build_probe_sdf(
            mesh, spacing_mm=DEFAULT_SPACING_MM, n_surface_points=500,
            use_cache=False, sign_type="pseudonormal",
        ),
        shank_centers=np.array([[0.0, 0.0, 1.5]]),
        shank_halves=np.array([[0.035, 0.012, 1.5]]),
    )  # fmt: skip
    axis = np.array([0.0, 0.0, 1.0])
    names = ["p", "q"]
    probes = [
        ProbeStaticInfo(
            name=name,
            target_LPS=np.array([x, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=np.zeros((1, 3)),
        )
        for name, x in zip(names, TARGETS_X)
    ]
    holes = [
        Hole(
            id=i + 1,
            axis=axis,
            ref_point=np.array([x, 0.0, -3.0]),
            sections=[
                HoleSection(
                    axis=axis, center=np.array([x, 0.0, -3.0]), a=0.6, b=0.3, theta=0.0
                )
            ],
        )
        for i, x in enumerate(TARGETS_X)
    ]
    statics = _build_probe_static(
        probes,
        holes,
        SimpleNamespace(probe_to_hole={"p": 1, "q": 2}),
        SimpleNamespace(probe_to_arc_idx={"p": 0, "q": 0}, arc_centroids_deg=(0.0,)),
        sdf_by_name={n: body for n in names},
    )
    return make_phase2(
        statics, 1, weights=Phase2Weights(SPEC, ObjectiveSpec()),
        drop_padded_rows=False, grid_dtype=np.float32,
    )  # fmt: skip


def _pairs(p2, x) -> np.ndarray:
    return np.asarray(p2["slack_parts"](np.asarray(x, float))["probe_pair"], float)


def test_the_cap_sits_below_the_grid_padding() -> None:
    """An out-of-grid point is at least a padding from the body; the last in-grid
    sample reads within a voxel of that, and the soft-min reads at most
    log(k)/β under its minimum."""
    k = max(SPEC.top_k_body_body, SPEC.top_k_body_shank, SPEC.top_k_shank_shank)
    softmin_bias = np.log(k) / SPEC.softmin_beta
    assert CLEARANCE_CAP_MM < DEFAULT_PAD_MM - DEFAULT_SPACING_MM - softmin_bias


def test_rows_stay_continuous_as_a_probe_leaves_the_grid(problem) -> None:
    offsets = np.arange(0.0, 8.0, 0.05)
    rows = []
    for off in offsets:
        x = list(POSE)
        x[OFF_R] = off
        rows.append(_pairs(problem, x))
    rows = np.array(rows)  # (offsets, pairs, categories)
    assert (rows < PADDED_SLACK - 1.0).all(), "no row may read the sentinel"
    gains = np.asarray(problem["slack_labels"]["pair_gains"], float)
    steps_mm = np.abs(np.diff(rows, axis=0)) / gains
    # The probes slide 0.05 mm per step; a row leaving the grid would jump ~1e3.
    assert steps_mm.max() < 0.5, steps_mm.max()
    # Far apart, every category reads the cap and stops changing.
    assert (np.ptp(rows[-20:], axis=0) == 0.0).all()


def test_the_unmeasured_category_is_constant_and_dropped(problem) -> None:
    labels = problem["slack_labels"]
    categories = list(labels["pair_categories"])
    unmeasured = [categories.index(c) for c in labels["unmeasured_pair_categories"]]
    assert unmeasured, "the swept kernel leaves a category unmeasured"
    values = []
    for off in (0.0, 1.0, 4.0):
        x = list(POSE)
        x[OFF_R] = off
        values.append(_pairs(problem, x).reshape(-1, len(categories))[:, unmeasured])
    assert np.ptp(np.array(values)) == 0.0
    live = np.asarray(problem["live_rows"])
    thread = np.asarray(problem["slack_parts"](np.asarray(POSE, float))["thread"]).size
    pair_live = live[thread : thread + len(categories)]
    assert not pair_live[unmeasured].any()
    assert np.delete(pair_live, unmeasured).all()
