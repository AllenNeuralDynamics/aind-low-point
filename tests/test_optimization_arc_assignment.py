"""Tests for ``aind_low_point.optimization.arc_assignment``."""

from __future__ import annotations

import numpy as np
import pytest

from aind_low_point.optimization.arc_assignment import (
    enumerate_partitions,
    required_aps_deg_for_assignment,
    solve_top_k_arc_assignments,
)
from aind_low_point.optimization.geometry import HoleSection
from aind_low_point.optimization.holes import Hole


def _make_hole(hole_id: int, axis) -> Hole:
    axis_arr = np.asarray(axis, dtype=float)
    axis_arr /= np.linalg.norm(axis_arr)
    sec = HoleSection(
        axis=axis_arr,
        center=np.zeros(3),
        a=0.6,
        b=0.35,
        theta=np.pi / 2,
    )
    return Hole(id=hole_id, axis=axis_arr, ref_point=np.zeros(3), sections=[sec, sec])


# -- required_aps_deg_for_assignment --------------------------------------


def test_required_aps_for_assignment_basic():
    """Vertical hole → required AP ≈ 0; tilted hole → matches y-tilt."""
    holes = [
        _make_hole(0, axis=(0, 0, 1)),
        _make_hole(1, axis=(0, 0.5, np.sqrt(0.75))),
    ]
    probe_to_hole = {"vertical_probe": 0, "tilted_probe": 1}
    aps = required_aps_deg_for_assignment(probe_to_hole, holes)
    assert aps["vertical_probe"] == pytest.approx(0.0, abs=0.5)
    assert aps["tilted_probe"] == pytest.approx(30.0, abs=0.5)


def test_required_aps_unknown_hole_id_raises():
    holes = [_make_hole(0, axis=(0, 0, 1))]
    with pytest.raises(KeyError, match="hole id"):
        required_aps_deg_for_assignment({"p": 99}, holes)


# -- enumerate_partitions --------------------------------------------------


def test_enumerate_partitions_two_well_separated_clusters_best_is_tight():
    """Probes at AP {-20, -19, +20, +21} with 2 arcs: the lowest-cost
    partition splits {-20, -19} onto one arc and {+20, +21} onto the
    other. Other partitions may be feasible (centroids still ≥16° apart)
    but rank worse on within-cluster variance."""
    probe_names = ["p0", "p1", "p2", "p3"]
    aps = np.array([-20.0, -19.0, +20.0, +21.0])
    parts = enumerate_partitions(probe_names, aps, num_arcs=2)
    assert len(parts) >= 1
    best = parts[0]
    arc_for_p0 = best.probe_to_arc_idx["p0"]
    arc_for_p1 = best.probe_to_arc_idx["p1"]
    arc_for_p2 = best.probe_to_arc_idx["p2"]
    arc_for_p3 = best.probe_to_arc_idx["p3"]
    assert arc_for_p0 == arc_for_p1
    assert arc_for_p2 == arc_for_p3
    assert arc_for_p0 != arc_for_p2
    assert best.cost == pytest.approx(1.0)
    # Subsequent partitions cost strictly more.
    for p in parts[1:]:
        assert p.cost > best.cost


def test_enumerate_partitions_arc_sep_is_soft():
    """With probes all clustered near 0°, splitting into 2 arcs gives
    centroids closer than 16° apart — that's an inner-loop concern, not
    a hard middle-layer reject. The partition still surfaces, ranked
    high (bad) by ``arc_sep_shortfall_weight × shortfall²``.

    Hard ``+inf`` weight recovers the legacy filter-out behaviour.
    """
    probe_names = ["p0", "p1", "p2", "p3", "p4"]
    aps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # all near 0
    parts = enumerate_partitions(probe_names, aps, num_arcs=2)
    # Soft default: partition surfaces, but its cost is dominated by
    # the AP-sep shortfall (16² weight × ~1).
    assert len(parts) >= 1
    # Cost should be dominated by shortfall ≫ within-cluster variance.
    assert parts[0].cost > 100.0  # 16° shortfall → 256 × weight = 2560
    parts_hard = enumerate_partitions(
        probe_names,
        aps,
        num_arcs=2,
        arc_sep_shortfall_weight=float("inf"),
    )
    assert parts_hard == []  # legacy hard-filter behaviour


def test_enumerate_partitions_capacity_filter_hard():
    """Capacity stays a hard reject (it's a hardware ceiling)."""
    probe_names = [f"p{i}" for i in range(5)]
    aps = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # 5 probes into 1 arc with capacity 4 → infeasible regardless of
    # the arc-sep weight.
    parts = enumerate_partitions(
        probe_names,
        aps,
        num_arcs=1,
        max_per_arc=4,
    )
    assert parts == []


def test_enumerate_partitions_canonical_ordering():
    """Arcs are labelled 0, 1, ... in ascending AP centroid — the
    best (lowest-cost) partition has c, d on arc 0 (-10°) and a, b
    on arc 1 (+30°)."""
    probe_names = ["a", "b", "c", "d"]
    aps = np.array([+30.0, +30.0, -10.0, -10.0])
    parts = enumerate_partitions(probe_names, aps, num_arcs=2)
    best = parts[0]
    assert best.probe_to_arc_idx["c"] == 0
    assert best.probe_to_arc_idx["d"] == 0
    assert best.probe_to_arc_idx["a"] == 1
    assert best.probe_to_arc_idx["b"] == 1
    assert best.arc_centroids_deg[0] < best.arc_centroids_deg[1]
    # Tightly clustered probes → zero within-cluster cost
    assert best.cost == pytest.approx(0.0, abs=1e-9)


def test_enumerate_partitions_three_arcs():
    """3 arcs at -20, 0, +20 → exactly one valid partition."""
    probe_names = ["p0", "p1", "p2"]
    aps = np.array([-20.0, 0.0, +20.0])
    parts = enumerate_partitions(probe_names, aps, num_arcs=3)
    assert len(parts) == 1
    p = parts[0]
    # Each probe gets its own arc (in canonical order: -20→0, 0→1, +20→2)
    assert p.probe_to_arc_idx["p0"] == 0
    assert p.probe_to_arc_idx["p1"] == 1
    assert p.probe_to_arc_idx["p2"] == 2


def test_enumerate_partitions_per_arc_capacity_4():
    """5 probes on the same AP → infeasible for 1 arc (capacity 4)."""
    probe_names = [f"p{i}" for i in range(5)]
    aps = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # Single arc with capacity 4 → infeasible
    parts = enumerate_partitions(probe_names, aps, num_arcs=1, max_per_arc=4)
    assert parts == []


def test_enumerate_partitions_within_cluster_cost_ranks():
    """Lowest within-cluster cost ranks first; cost = sum of squared
    deviations from each arc's centroid."""
    probe_names = ["a", "b", "c", "d"]
    aps = np.array([-20.0, -19.0, +20.0, +21.0])
    parts = enumerate_partitions(probe_names, aps, num_arcs=2)
    # Lowest-cost partition: {-20, -19} centroid=-19.5 (devs² 0.25 each = 0.5)
    #                       {+20, +21} centroid=+20.5 (devs² 0.25 each = 0.5)
    # Total: 1.0
    assert parts[0].cost == pytest.approx(1.0, abs=1e-9)


def test_enumerate_partitions_empty_probes():
    parts = enumerate_partitions([], np.zeros(0), num_arcs=2)
    assert parts == []


def test_enumerate_partitions_shape_validation():
    with pytest.raises(ValueError, match="shape"):
        enumerate_partitions(["a", "b"], np.array([1.0]), num_arcs=2)


# -- solve_top_k_arc_assignments ------------------------------------------


def test_solve_top_k_e2e():
    """End-to-end: hole assignment → required APs → top-K arc partitions.

    The *best* partition pairs the low-AP probes (p0, p1) onto one arc
    and the high-AP probes (p2, p3) onto the other. Worse-ranked
    partitions may split differently but pay a within-cluster cost
    penalty."""
    holes = [
        # Two holes pointing roughly +z (small AP), two tilted toward +y.
        _make_hole(0, axis=(0, 0, 1)),
        _make_hole(1, axis=(0, 0.05, np.sqrt(1 - 0.0025))),
        _make_hole(2, axis=(0, 0.6, np.sqrt(1 - 0.36))),
        _make_hole(3, axis=(0, 0.65, np.sqrt(1 - 0.4225))),
    ]
    probe_to_hole = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
    parts = solve_top_k_arc_assignments(
        probe_to_hole,
        holes,
        max_num_arcs=2,
        k=5,
    )
    assert len(parts) >= 1
    best = parts[0]
    assert best.probe_to_arc_idx["p0"] == best.probe_to_arc_idx["p1"]
    assert best.probe_to_arc_idx["p2"] == best.probe_to_arc_idx["p3"]
    assert best.probe_to_arc_idx["p0"] != best.probe_to_arc_idx["p2"]
    # And the best partition has the lowest cost in the returned list.
    for p in parts[1:]:
        assert p.cost >= best.cost


def test_solve_top_k_zero_returns_empty():
    holes = [_make_hole(0, axis=(0, 0, 1))]
    out = solve_top_k_arc_assignments({"p": 0}, holes, max_num_arcs=1, k=0)
    assert out == []


def test_solve_top_k_returns_sorted_by_cost():
    """If multiple partitions survive, they're ranked ascending by cost."""
    # 6 probes spread across 3 well-separated clusters
    holes = [
        _make_hole(i, axis=(0, np.sin(np.deg2rad(ap)), np.cos(np.deg2rad(ap))))
        for i, ap in enumerate([-30, -29, -10, -9, +20, +21])
    ]
    probe_to_hole = {f"p{i}": i for i in range(6)}
    parts = solve_top_k_arc_assignments(
        probe_to_hole,
        holes,
        max_num_arcs=3,
        k=10,
    )
    if len(parts) > 1:
        costs = [p.cost for p in parts]
        assert costs == sorted(costs)


# -- arc-count penalty + range -------------------------------------------


def test_max_num_arcs_caps_partition_count():
    """A problem feasible with up to 4 arcs gets only 2-arc partitions
    when ``max_num_arcs=2``. Required APs are well-separated so all
    arc-counts in [1, 4] yield feasible partitions."""
    holes = [
        _make_hole(i, axis=(0, np.sin(np.deg2rad(ap)), np.cos(np.deg2rad(ap))))
        for i, ap in enumerate([-30, -10, +10, +30])
    ]
    probe_to_hole = {f"p{i}": i for i in range(4)}
    parts_max2 = solve_top_k_arc_assignments(
        probe_to_hole,
        holes,
        max_num_arcs=2,
        min_num_arcs=2,
        k=20,
        arc_count_penalty_deg2=0.0,
    )
    parts_max4 = solve_top_k_arc_assignments(
        probe_to_hole,
        holes,
        max_num_arcs=4,
        min_num_arcs=2,
        k=20,
        arc_count_penalty_deg2=0.0,
    )
    n_arcs_used = lambda p: len(set(p.probe_to_arc_idx.values()))  # noqa: E731
    # max_num_arcs=2 cuts the search space to only 2-arc partitions.
    assert all(n_arcs_used(p) == 2 for p in parts_max2)
    # max_num_arcs=4 admits partitions with more arcs.
    arc_counts_present = {n_arcs_used(p) for p in parts_max4}
    assert 2 in arc_counts_present
    assert max(arc_counts_present) > 2


def test_arc_count_penalty_prefers_fewer_arcs():
    """With penalty=0, tightest cluster (4 arcs for 4 well-separated
    probes) wins. With a positive penalty, the 2-arc partition wins."""
    holes = [
        _make_hole(i, axis=(0, np.sin(np.deg2rad(ap)), np.cos(np.deg2rad(ap))))
        for i, ap in enumerate([-30, -10, +10, +30])
    ]
    probe_to_hole = {f"p{i}": i for i in range(4)}
    n_arcs_used = lambda p: len(set(p.probe_to_arc_idx.values()))  # noqa: E731

    # Penalty 0 → 4-arc partition (each probe its own arc) has zero
    # within-cluster cost and wins.
    no_penalty = solve_top_k_arc_assignments(
        probe_to_hole,
        holes,
        max_num_arcs=4,
        min_num_arcs=2,
        k=1,
        arc_count_penalty_deg2=0.0,
    )
    assert no_penalty[0].cost == pytest.approx(0.0, abs=1e-9)
    assert n_arcs_used(no_penalty[0]) == 4

    # 2-arc tightest partition has cost = 200 (each arc has two probes
    # ±10° from centroid, devs² = 100 + 100 per arc; 2 arcs ⇒ 200).
    # Penalty per arc beyond min_num_arcs=2 must be > 100 to push 4 arcs
    # (penalty 200, cost 0) above 2 arcs (penalty 0, cost 200).
    with_penalty = solve_top_k_arc_assignments(
        probe_to_hole,
        holes,
        max_num_arcs=4,
        min_num_arcs=2,
        k=1,
        arc_count_penalty_deg2=200.0,
    )
    assert n_arcs_used(with_penalty[0]) == 2


def test_arc_count_penalty_default_is_nonzero():
    """The default ``arc_count_penalty_deg2`` is a positive value so
    fewer-arc partitions are preferred unless explicitly disabled."""
    holes = [
        _make_hole(i, axis=(0, np.sin(np.deg2rad(ap)), np.cos(np.deg2rad(ap))))
        for i, ap in enumerate([-25, -24, +20, +21])
    ]
    probe_to_hole = {f"p{i}": i for i in range(4)}
    parts = solve_top_k_arc_assignments(
        probe_to_hole,
        holes,
        max_num_arcs=4,
        k=5,
    )
    # The first (best) partition should be 2 arcs (the tight 4-probe
    # split with default penalty).
    n_arcs_used = lambda p: len(set(p.probe_to_arc_idx.values()))  # noqa: E731
    assert n_arcs_used(parts[0]) == 2


def test_min_num_arcs_validation():
    holes = [_make_hole(0, axis=(0, 0, 1))]
    with pytest.raises(ValueError, match="min_num_arcs"):
        solve_top_k_arc_assignments(
            {"p": 0},
            holes,
            max_num_arcs=1,
            k=1,
            min_num_arcs=2,
        )
