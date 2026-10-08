"""Seeding each probe's lone solve, measuring recorded share, and handing the
shares alone to Phase 2."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aind_rutter.optimization.pipeline.coverage_floor import (
    recorded_shares,
    spread_seeds,
)
from aind_rutter.optimization.pipeline.handoff import best_alone_by_probe
from aind_rutter.optimization.pipeline.settings import Phase2Settings


def _entry(spins):
    anchors = [
        SimpleNamespace(ap_deg=float(i % 30) - 15.0, ml_deg=0.0, spin_deg=float(s))
        for i, s in enumerate(spins)
    ]
    return SimpleNamespace(anchors=anchors)


def test_seeds_spread_over_spin_inside_the_ap_window() -> None:
    entry = _entry(range(-180, 180, 5))
    seeds = spread_seeds(entry, -10.0, 10.0, 4)
    assert len(seeds) == 4
    assert all(-10.0 <= ap <= 10.0 for ap, _, _ in seeds)
    spins = [spin for _, _, spin in seeds]
    assert spins == sorted(spins) and spins[-1] - spins[0] > 300


def test_a_bore_with_no_anchor_in_the_window_gives_no_seed() -> None:
    assert spread_seeds(_entry(range(0, 50, 5)), 40.0, 50.0, 4) == []


def test_fewer_anchors_than_seeds_gives_each_anchor_once() -> None:
    assert len(spread_seeds(_entry([0.0, 90.0]), -20.0, 20.0, 8)) == 2


PROBES = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]


def test_phase2_reads_the_pools_shares_alone_in_probe_order() -> None:
    settings = Phase2Settings(min_coverage_fraction=0.1)
    assert best_alone_by_probe(settings, PROBES, {"b": 2.0, "a": 1.0}) == (1.0, 2.0)
    assert best_alone_by_probe(Phase2Settings(), PROBES, None) is None


@pytest.mark.parametrize(
    ("ceilings", "match"), [(None, "records no share"), ({"a": 1.0}, r"\['b'\]")]
)
def test_phase2_refuses_a_floor_the_pool_cannot_state(ceilings, match) -> None:
    with pytest.raises(ValueError, match=match):
        best_alone_by_probe(Phase2Settings(min_coverage_fraction=0.1), PROBES, ceilings)


def test_phase2_refuses_a_floor_with_no_radius() -> None:
    settings = Phase2Settings(min_coverage_fraction=0.1, recorded_radius_mm=None)
    with pytest.raises(ValueError, match="RECORDED_RADIUS"):
        best_alone_by_probe(settings, PROBES, {"a": 1.0, "b": 1.0})


def _lone_probe(name, points):
    return SimpleNamespace(
        name=name,
        recording=SimpleNamespace(active_ranges_mm=((0.0, 1.0),)),
        shank_tips_local=[np.zeros(3)],
        recorded_points=None if points is None else np.asarray(points, float),
    )


def test_a_share_counts_points_near_any_probes_active_stretch() -> None:
    # Two probes along z at x=0 and x=5; a's region has one point by each probe
    # and one past both active stretches, so another probe's sites count too.
    a = _lone_probe("a", [[0.1, 0, 0.5], [5.1, 0, 0.5], [0, 0, 3.0]])
    b = _lone_probe("b", [[9.0, 0, 0.5]])
    Rs = np.stack([np.eye(3)] * 2)
    ts = np.array([[0.0, 0, 0], [5.0, 0, 0]])
    assert recorded_shares([a, b], Rs, ts, 0.2) == pytest.approx([2 / 3, 0.0])


def test_a_share_needs_the_target_regions_cloud() -> None:
    probe = _lone_probe("a", None)
    with pytest.raises(ValueError, match="RUTTER_RETRO_DENSITY"):
        recorded_shares([probe], np.eye(3)[None], np.zeros((1, 3)), 0.2)
