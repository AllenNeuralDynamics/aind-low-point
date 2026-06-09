"""Tests for ``aind_low_point.optimization.hole_assignment``."""

from __future__ import annotations

import numpy as np
import pytest

from aind_low_point.optimization.geometry import HoleSection
from aind_low_point.optimization.hole_assignment import (
    AssignmentProbe,
    CostWeights,
    angle_to_target_rad,
    build_cost_matrix,
    solve_optimal_assignment,
    solve_top_k_assignments,
    static_threading_max_g,
)
from aind_low_point.optimization.holes import Hole

# -- helpers ----------------------------------------------------------------


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
        a=a,
        b=b,
        theta=theta,
    )
    return Hole(
        id=hole_id,
        axis=axis_arr,
        ref_point=np.asarray(center, dtype=float),
        sections=[sec, sec, sec],
    )


def _np24_tips() -> np.ndarray:
    """4 shanks at 250 µm pitch along local +y (AIND shank-row convention)."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.75, 0.0],
        ]
    )


def _single_shank_tips() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0]])


# -- angle_to_target_rad ----------------------------------------------------


def test_angle_target_directly_below_zero():
    """Hole pointing +z, target directly below → angle 0."""
    hole = _make_hole(0, center=(0, 0, 0), axis=(0, 0, 1))
    angle = angle_to_target_rad([0, 0, -5], hole)
    assert angle == pytest.approx(0.0, abs=1e-9)


def test_angle_target_directly_above_pi():
    """Hole pointing +z, target directly above (wrong side) → angle π."""
    hole = _make_hole(0, center=(0, 0, 0), axis=(0, 0, 1))
    angle = angle_to_target_rad([0, 0, 5], hole)
    assert angle == pytest.approx(np.pi, abs=1e-9)


def test_angle_target_lateral_pi_over_2():
    """Hole pointing +z, target laterally offset → angle π/2."""
    hole = _make_hole(0, center=(0, 0, 0), axis=(0, 0, 1))
    angle = angle_to_target_rad([5, 0, 0], hole)
    assert angle == pytest.approx(np.pi / 2, abs=1e-9)


def test_angle_target_at_center_returns_zero():
    """Pathological case: target at hole center → returns 0 (no preferred direction)."""
    hole = _make_hole(0, center=(0, 0, 0), axis=(0, 0, 1))
    angle = angle_to_target_rad([0, 0, 0], hole)
    assert angle == 0.0


# -- static_threading_max_g -------------------------------------------------


def test_static_threading_max_g_4shank_fits_normal_slot():
    """4-shank probe (750 µm span) fits a 1.20 × 0.70 mm slot."""
    hole = _make_hole(0, a=0.6, b=0.35, theta=np.pi / 2)
    probe = AssignmentProbe(
        name="p",
        target_LPS=np.array([0, 0, -2]),
        shank_tips_local=_np24_tips(),
    )
    g = static_threading_max_g(probe, hole)
    assert g < 0.0
    # Outer shanks at ±0.375 along major (a=0.6): g = (0.375/0.6)² − 1 ≈ −0.609
    assert g == pytest.approx(-0.609, abs=1e-3)


def test_static_threading_max_g_4shank_too_wide_for_slot():
    """4-shank probe doesn't fit a 0.7 × 0.5 slot — outer shanks
    overflow the major axis."""
    # Slot major a=0.35 means outer shank at 0.375 → g > 0
    hole = _make_hole(0, a=0.35, b=0.25, theta=np.pi / 2)
    probe = AssignmentProbe(
        name="p",
        target_LPS=np.array([0, 0, -2]),
        shank_tips_local=_np24_tips(),
    )
    g = static_threading_max_g(probe, hole)
    assert g > 0.0


def test_static_threading_single_shank_fits_easily():
    """Single-shank probe through any reasonable hole has lots of clearance."""
    hole = _make_hole(0, a=0.6, b=0.35, theta=np.pi / 2)
    probe = AssignmentProbe(
        name="p",
        target_LPS=np.array([0, 0, -2]),
        shank_tips_local=_single_shank_tips(),
    )
    g = static_threading_max_g(probe, hole)
    # Single shank at slot center: g = -1
    assert g == pytest.approx(-1.0, abs=1e-9)


# -- build_cost_matrix ------------------------------------------------------


def test_cost_matrix_shape():
    probes = [
        AssignmentProbe(
            name=f"p{i}", target_LPS=np.zeros(3), shank_tips_local=_single_shank_tips()
        )
        for i in range(3)
    ]
    holes = [_make_hole(i) for i in range(5)]
    cost = build_cost_matrix(probes, holes)
    assert cost.shape == (3, 5)


def test_cost_matrix_rejects_infeasible_pairs():
    """Slot too small for the 4-shank probe gets a forbid-cost entry."""
    probes = [
        AssignmentProbe(
            name="big_probe",
            target_LPS=np.array([0, 0, -2]),
            shank_tips_local=_np24_tips(),
        ),
    ]
    holes = [
        _make_hole(0, a=0.35, b=0.25, theta=np.pi / 2),  # too small
        _make_hole(1, a=0.6, b=0.35, theta=np.pi / 2),  # OK
    ]
    cost = build_cost_matrix(probes, holes)
    assert cost[0, 0] >= CostWeights().forbid_cost
    assert cost[0, 1] < CostWeights().forbid_cost


def test_cost_matrix_prefers_aligned_target():
    """Hole pointing toward target gets lower cost than hole pointing away."""
    probes = [
        AssignmentProbe(
            name="p",
            target_LPS=np.array([0, 0, -5]),
            shank_tips_local=_single_shank_tips(),
        )
    ]
    # Hole 0: axis pointing +z (up) → target line aligns with -axis (0)
    # Hole 1: axis pointing +x (sideways) → target line at 90° from -axis
    holes = [
        _make_hole(0, center=(0, 0, 0), axis=(0, 0, 1)),
        _make_hole(1, center=(0, 0, 0), axis=(1, 0, 0)),
    ]
    cost = build_cost_matrix(probes, holes)
    assert cost[0, 0] < cost[0, 1]


# -- solve_optimal_assignment ----------------------------------------------


def test_optimal_assignment_picks_aligned_holes():
    """Two probes with targets directly below distinct holes → each
    gets the hole directly above its target."""
    probes = [
        AssignmentProbe(
            name="p_left",
            target_LPS=np.array([-5, 0, -5]),
            shank_tips_local=_single_shank_tips(),
        ),
        AssignmentProbe(
            name="p_right",
            target_LPS=np.array([+5, 0, -5]),
            shank_tips_local=_single_shank_tips(),
        ),
    ]
    holes = [
        _make_hole(0, center=(-5, 0, 0), axis=(0, 0, 1)),
        _make_hole(1, center=(+5, 0, 0), axis=(0, 0, 1)),
    ]
    result = solve_optimal_assignment(probes, holes)
    assert result.feasible
    assert result.probe_to_hole == {"p_left": 0, "p_right": 1}


def test_optimal_assignment_skips_infeasible_pair():
    """If hole 0 doesn't physically fit the only big probe, LSAP should
    route the big probe to a feasible hole."""
    probes = [
        AssignmentProbe(
            name="big",
            target_LPS=np.array([0, 0, -5]),
            shank_tips_local=_np24_tips(),
        ),
        AssignmentProbe(
            name="small",
            target_LPS=np.array([3, 0, -5]),
            shank_tips_local=_single_shank_tips(),
        ),
    ]
    # Hole 0 is too narrow for the 4-shank probe.
    holes = [
        _make_hole(0, center=(0, 0, 0), a=0.35, b=0.25, theta=np.pi / 2),
        _make_hole(1, center=(3, 0, 0), a=0.6, b=0.35, theta=np.pi / 2),
        _make_hole(2, center=(-3, 0, 0), a=0.6, b=0.35, theta=np.pi / 2),
    ]
    result = solve_optimal_assignment(probes, holes)
    assert result.feasible
    # big probe must NOT be routed to hole 0
    assert result.probe_to_hole["big"] != 0


def test_optimal_assignment_empty_inputs():
    assert not solve_optimal_assignment([], []).feasible


# -- Murty's k-best ---------------------------------------------------------


def test_murty_returns_k_results_ranked_by_cost():
    """Top-K assignments come back sorted by cost."""
    probes = [
        AssignmentProbe(
            name=f"p{i}",
            target_LPS=np.array([float(i), 0, -5]),
            shank_tips_local=_single_shank_tips(),
        )
        for i in range(3)
    ]
    holes = [_make_hole(j, center=(float(j), 0, 0), axis=(0, 0, 1)) for j in range(5)]
    results = solve_top_k_assignments(probes, holes, k=5)
    assert len(results) == 5
    costs = [r.cost for r in results]
    assert costs == sorted(costs)


def test_murty_first_matches_optimal():
    """The first (best) Murty result equals the LSAP-1 optimal."""
    probes = [
        AssignmentProbe(
            name=f"p{i}",
            target_LPS=np.array([float(i), 0, -5]),
            shank_tips_local=_single_shank_tips(),
        )
        for i in range(3)
    ]
    holes = [_make_hole(j, center=(float(j), 0, 0), axis=(0, 0, 1)) for j in range(5)]
    optimal = solve_optimal_assignment(probes, holes)
    top_k = solve_top_k_assignments(probes, holes, k=3)
    assert top_k[0].cost == pytest.approx(optimal.cost)
    assert top_k[0].probe_to_hole == optimal.probe_to_hole


def test_murty_yields_distinct_assignments():
    """Top-K Murty results all have different probe→hole mappings."""
    probes = [
        AssignmentProbe(
            name=f"p{i}",
            target_LPS=np.array([float(i), 0, -5]),
            shank_tips_local=_single_shank_tips(),
        )
        for i in range(3)
    ]
    holes = [_make_hole(j, center=(float(j), 0, 0), axis=(0, 0, 1)) for j in range(5)]
    results = solve_top_k_assignments(probes, holes, k=4)
    seen = set()
    for r in results:
        # Convert to hashable signature
        sig = tuple(sorted(r.probe_to_hole.items()))
        assert sig not in seen
        seen.add(sig)


def test_murty_k_zero_returns_empty():
    probes = [
        AssignmentProbe(
            name="p", target_LPS=np.zeros(3), shank_tips_local=_single_shank_tips()
        )
    ]
    holes = [_make_hole(0)]
    assert solve_top_k_assignments(probes, holes, k=0) == []


def test_murty_capped_by_total_distinct_assignments():
    """If we ask for more than exists, we get only the existing ones."""
    # 1 probe × 1 hole → only 1 distinct assignment
    probes = [
        AssignmentProbe(
            name="p",
            target_LPS=np.array([0, 0, -5]),
            shank_tips_local=_single_shank_tips(),
        )
    ]
    holes = [_make_hole(0, center=(0, 0, 0), axis=(0, 0, 1))]
    results = solve_top_k_assignments(probes, holes, k=10)
    assert len(results) == 1


# -- end-to-end on real extracted holes (skips if absent) ------------------


def test_optimal_assignment_against_build5_holes():
    """Smoke test: the assignment runs against real extracted hole specs
    and returns a feasible mapping for 4 probes."""
    import os

    from aind_low_point.optimization.holes import load_holes

    yaml_path = os.path.expanduser("~/Downloads/0274-P-001.holes.yml")
    if not os.path.exists(yaml_path):
        pytest.skip(f"build5 holes YAML not found at {yaml_path}")
    holes = load_holes(yaml_path)
    # Targets in LPS approximately below each of the first 4 hole centers.
    probes = [
        AssignmentProbe(
            name=f"P{i}",
            target_LPS=holes[i].sections[-1].center + np.array([0, 0, -5]),
            shank_tips_local=_np24_tips(),
        )
        for i in range(4)
    ]
    result = solve_optimal_assignment(probes, holes)
    assert result.feasible
    assert len(result.probe_to_hole) == 4
    assert len(set(result.probe_to_hole.values())) == 4  # distinct
