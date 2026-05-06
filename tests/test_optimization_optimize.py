"""Tests for ``aind_low_point.optimization.optimize`` (the three-level driver)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aind_low_point.optimization.geometry import HoleSection
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.optimize import (
    OptimizationResult,
    ProbeStaticInfo,
    optimize,
)


def _np24_tips() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.75, 0.0, 0.0],
        ]
    )


def _single_shank_tips() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0]])


def _make_hole(
    hole_id: int,
    *,
    center=(0, 0, 0),
    axis=(0, 0, 1),
    a=0.6,
    b=0.35,
    theta=np.pi / 2,
) -> Hole:
    axis_arr = np.asarray(axis, dtype=float)
    axis_arr /= np.linalg.norm(axis_arr)
    sec = HoleSection(
        axis=axis_arr,
        center=np.asarray(center, dtype=float),
        a=a, b=b, theta=theta,
    )
    return Hole(
        id=hole_id, axis=axis_arr,
        ref_point=np.asarray(center, dtype=float),
        sections=[sec, sec, sec],
    )


# -- single-probe smoke test ------------------------------------------------


def test_optimize_single_probe_returns_result():
    """1 probe, 1 hole, 1 arc — optimizer should produce a result with
    positive coverage and zero hard violations."""
    target = np.array([0.0, 0.0, -3.0])
    probes = [
        ProbeStaticInfo(
            name="p1", target_LPS=target, kind="2.4",
            shank_tips_local=_np24_tips(),
        )
    ]
    holes = [_make_hole(0, center=(0, 0, 0), axis=(0, 0, 1))]
    with warnings.catch_warnings():
        # CMA-ES likely missing; suppress the install-cma warning.
        warnings.simplefilter("ignore")
        result = optimize(
            probes, holes,
            max_num_arcs=1, k_holes=1, k_arcs=1,
            slsqp_max_iter=50,
        )
    assert result is not None
    assert isinstance(result, OptimizationResult)
    assert result.probe_to_hole == {"p1": 0}
    assert result.n_arcs == 1
    assert result.breakdown.coverage_total > 0.0
    assert result.breakdown.threading_penalty == pytest.approx(0.0, abs=1e-6)
    assert result.breakdown.kinematic_penalty == pytest.approx(0.0, abs=1e-6)


# -- multi-probe ------------------------------------------------------------


def test_optimize_multi_probe_distinct_holes():
    """3 probes, 5 holes, 2 arcs — optimizer should assign distinct
    holes to each probe."""
    holes = [
        _make_hole(j, center=(float(j) - 1, 0, 0), axis=(0, 0, 1))
        for j in range(5)
    ]
    probes = [
        ProbeStaticInfo(
            name=f"p{i}",
            target_LPS=np.array([float(i) - 1, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=_single_shank_tips(),
        )
        for i in range(3)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize(
            probes, holes,
            max_num_arcs=2, min_num_arcs=1,
            k_holes=3, k_arcs=2,
            slsqp_max_iter=50,
        )
    assert result is not None
    assert len(result.probe_to_hole) == 3
    assert len(set(result.probe_to_hole.values())) == 3  # distinct
    assert result.breakdown.coverage_total > 0.0


# -- arc-assignment passthroughs --------------------------------------------


def test_optimize_max_num_arcs_capped():
    """``max_num_arcs=2`` forces 2-arc partitioning even when more arcs
    would be feasible. Uses varied hole axes so required-AP differs
    enough to support arc clustering."""
    aps_deg = [-25, -24, +20, +21]
    holes = [
        _make_hole(
            j,
            center=(float(j), 0.0, 0.0),
            axis=(
                0.0,
                np.sin(np.deg2rad(ap)),
                np.cos(np.deg2rad(ap)),
            ),
        )
        for j, ap in enumerate(aps_deg)
    ]
    probes = [
        ProbeStaticInfo(
            name=f"p{i}",
            target_LPS=np.array([float(i), 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=_single_shank_tips(),
        )
        for i in range(4)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize(
            probes, holes,
            max_num_arcs=2, min_num_arcs=2,
            k_holes=2, k_arcs=2,
            arc_count_penalty_deg2=0.0,
            slsqp_max_iter=30,
        )
    assert result is not None
    assert result.n_arcs == 2


def test_optimize_arc_count_penalty_prefers_fewer_arcs():
    """With a positive arc penalty, the 2-arc result wins over 4-arc.
    Uses 4 probes whose hole axes give well-separated required APs at
    roughly -30, -10, +10, +30°."""
    aps_deg = [-30, -10, +10, +30]
    holes = [
        _make_hole(
            j,
            center=(float(j) * 5, 0.0, 0.0),
            axis=(
                0.0,
                np.sin(np.deg2rad(ap)),
                np.cos(np.deg2rad(ap)),
            ),
        )
        for j, ap in enumerate(aps_deg)
    ]
    probes = [
        ProbeStaticInfo(
            name=f"p{i}",
            target_LPS=np.array(
                [float(i) * 5, 0.0, -3.0], dtype=np.float64
            ),
            kind="2.1",
            shank_tips_local=_single_shank_tips(),
        )
        for i in range(4)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # No penalty → 4 separate arcs likely wins (each probe tightly
        # clustered alone).
        no_penalty = optimize(
            probes, holes,
            max_num_arcs=4, min_num_arcs=2, k_holes=1, k_arcs=5,
            arc_count_penalty_deg2=0.0,
            slsqp_max_iter=20,
        )
        # Big penalty → 2-arc result wins.
        with_penalty = optimize(
            probes, holes,
            max_num_arcs=4, min_num_arcs=2, k_holes=1, k_arcs=5,
            arc_count_penalty_deg2=10000.0,
            slsqp_max_iter=20,
        )
    assert no_penalty is not None
    assert with_penalty is not None
    assert with_penalty.n_arcs <= no_penalty.n_arcs
    assert with_penalty.n_arcs == 2


# -- robustness -------------------------------------------------------------


def test_optimize_no_probes_returns_none():
    holes = [_make_hole(0)]
    assert optimize([], holes, max_num_arcs=1, k_holes=1, k_arcs=1) is None


def test_optimize_no_feasible_holes_returns_none():
    """A 4-shank probe paired with a too-small slot → infeasible
    everywhere → optimize returns None."""
    holes = [
        _make_hole(0, a=0.3, b=0.2, theta=np.pi / 2),
    ]
    probes = [
        ProbeStaticInfo(
            name="p", target_LPS=np.zeros(3), kind="2.4",
            shank_tips_local=_np24_tips(),
        )
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize(
            probes, holes,
            max_num_arcs=1, k_holes=1, k_arcs=1,
            slsqp_max_iter=10,
        )
    assert result is None


def test_optimize_warm_start_shape():
    """Result.x has the correct flat shape: ``num_arcs + 5*K`` entries."""
    holes = [_make_hole(0, center=(0, 0, 0), axis=(0, 0, 1))]
    probes = [
        ProbeStaticInfo(
            name="p", target_LPS=np.array([0.0, 0.0, -3.0]),
            kind="2.1", shank_tips_local=_single_shank_tips(),
        )
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize(
            probes, holes,
            max_num_arcs=1, k_holes=1, k_arcs=1, slsqp_max_iter=20,
        )
    assert result is not None
    assert result.x.shape == (1 + 5 * 1,)


# -- end-to-end against real build5 holes ----------------------------------


def test_optimize_against_build5_holes():
    """Smoke test: load real extracted hole specs and run the full
    pipeline with 4 NP 2.0 four-shank probes. Expect a feasible result
    with all probes assigned to distinct holes."""
    import os
    from aind_low_point.optimization.holes import load_holes

    yaml_path = os.path.expanduser("~/Downloads/0274-P-001.holes.yml")
    if not os.path.exists(yaml_path):
        pytest.skip(f"build5 holes YAML not found at {yaml_path}")

    holes = load_holes(yaml_path)
    # Targets at varying xy 5 mm below the first 4 hole centers (in LPS,
    # +z is Superior; -z is into the brain).
    probes = [
        ProbeStaticInfo(
            name=f"P{i}",
            target_LPS=holes[i].sections[-1].center
            + np.array([0.0, 0.0, -5.0]),
            kind="2.4",
            shank_tips_local=_np24_tips(),
        )
        for i in range(4)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize(
            probes, holes,
            max_num_arcs=2, min_num_arcs=1,
            k_holes=3, k_arcs=2,
            slsqp_max_iter=30,
        )
    assert result is not None
    assert len(result.probe_to_hole) == 4
    assert len(set(result.probe_to_hole.values())) == 4
    assert result.n_arcs in (1, 2)
    # Coverage > 0 for every probe (Gaussian sees something on each shaft).
    assert result.breakdown.coverage_total > 0.0
