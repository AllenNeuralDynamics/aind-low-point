"""Seeding each probe's threaded best, and handing the bests to Phase 2."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aind_rutter.optimization.pipeline.coverage_floor import spread_seeds
from aind_rutter.optimization.pipeline.handoff import floor_ceilings_by_probe
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


def test_phase2_reads_the_pools_bests_in_probe_order() -> None:
    settings = Phase2Settings(min_coverage_fraction=0.1)
    assert floor_ceilings_by_probe(settings, PROBES, {"b": 2.0, "a": 1.0}) == (1.0, 2.0)
    assert floor_ceilings_by_probe(Phase2Settings(), PROBES, None) is None


@pytest.mark.parametrize(
    ("ceilings", "match"), [(None, "records no threaded"), ({"a": 1.0}, r"\['b'\]")]
)
def test_phase2_refuses_a_floor_the_pool_cannot_state(ceilings, match) -> None:
    with pytest.raises(ValueError, match=match):
        floor_ceilings_by_probe(
            Phase2Settings(min_coverage_fraction=0.1), PROBES, ceilings
        )
