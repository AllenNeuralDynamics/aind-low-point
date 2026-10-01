"""The allocation proposer's geometry and search, on a bore and clouds made here."""

from __future__ import annotations

import numpy as np
import pytest
from aind_mri_utils.arc_angles import arc_angles_to_affine

from aind_rutter.optimization.assignment.allocation import (
    Cover,
    Line,
    ProbeShape,
    aim_points,
    best_per_count,
    bore_lines,
    covers,
    find_units,
    partitions,
    seat,
    shaft_down_LPS,
    threads,
)
from aind_rutter.optimization.geometry.holes import Hole, HoleWall
from aind_rutter.optimization.geometry.primitives import HoleSection

Z = np.array([0.0, 0.0, 1.0])
DOWN = -Z
RADIUS_MM = 0.2
AP_WINDOW = (-30.0, 30.0)
PITCH_MM = 0.25


def _hole(hole_id: int = 1, x: float = 0.0, radius: float = 1.0, walls=()) -> Hole:
    """A straight round bore down ``-z``, its top at ``z = 0``."""
    centre = np.array([x, 0.0, 0.0])
    sections = [
        HoleSection(Z, centre + z * Z, radius, radius, 0.0) for z in (0.0, -1.0)
    ]
    return Hole(hole_id, Z, centre, sections, tuple(walls))


def _shape(kind: str, n_shanks: int, count: int = 1) -> ProbeShape:
    across = (np.arange(n_shanks) - (n_shanks - 1) / 2) * PITCH_MM
    offsets = np.stack([across, np.zeros(n_shanks), np.zeros(n_shanks)], axis=1)
    return ProbeShape(kind, offsets, (0.2, 3.0), count)


def _column(depths: tuple[float, float], x: float = 0.0, y: float = 0.0):
    """Points every 0.1 mm down a vertical at ``(x, y)`` between two depths."""
    z = -np.arange(depths[0], depths[1] + 1e-9, 0.1)
    return np.stack([np.full_like(z, x), np.full_like(z, y), z], axis=1)


SINGLE, QUAD = _shape("single", 1, count=2), _shape("quad", 4)


def _line(kind: str, hole: int, share: dict[int, float]) -> Line:
    return Line(kind, hole, share, np.zeros(3), DOWN, 0.0, 0.0, 0.0, 0.0)


def _units(*lines: Line):
    units: dict = {}
    for line in lines:
        units.setdefault(frozenset(line.share), {}).setdefault(line.hole, {})[
            line.kind
        ] = line
    return units


class TestLines:
    def test_the_shaft_points_down_the_probe_frames_z(self):
        rng = np.random.default_rng(0)
        for ap, ml, spin in rng.uniform(-40, 40, (20, 3)):
            want = -np.asarray(arc_angles_to_affine(ap, ml, spin))[:3, 2]
            assert shaft_down_LPS(np.array(ap), np.array(ml)) == pytest.approx(want)

    def test_a_line_threads_inside_every_oval_less_the_margin(self):
        hole = _hole()
        p = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [0.85, 0.0, 0.0]])
        d = np.tile(DOWN, (3, 1))
        assert threads(hole, p, d, 0.1).tolist() == [True, False, True]
        # Inside the top oval and outside the bottom one.
        tilted = np.array([[1.2, 0.0, -1.0]]) / np.hypot(1.2, 1.0)
        assert threads(hole, p[:1], tilted, 0.0).tolist() == [False]

    def test_a_wall_cuts_the_bore(self):
        wall = HoleWall(np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        hole = _hole(walls=[wall])
        p = np.array([[0.0, 0.0, 0.0], [0.45, 0.0, 0.0], [0.6, 0.0, 0.0]])
        d = np.tile(DOWN, (3, 1))
        assert threads(hole, p, d, 0.1).tolist() == [True, False, False]

    def test_bore_lines_thread_and_stay_in_the_rigs_angles(self):
        entry, direction, ap, ml = bore_lines(_hole(), AP_WINDOW, 3.0, 0.07)
        assert len(entry) and threads(_hole(), entry, direction, 0.07).all()
        assert ap.min() >= AP_WINDOW[0] and ap.max() <= AP_WINDOW[1]
        assert direction == pytest.approx(shaft_down_LPS(ap, ml))
        vertical = (ap == 0) & (ml == 0)
        assert vertical.sum() == 7  # the centre and the ring of six


class TestUnits:
    def test_regions_one_line_records_are_a_unit(self):
        clouds = [_column((3.0, 5.0)), _column((5.5, 6.5)), _column((3.0, 5.0), x=6.0)]
        units = find_units(clouds, [_hole()], AP_WINDOW, [SINGLE], radius_mm=RADIUS_MM)
        assert set(units) == {frozenset({0}), frozenset({1}), frozenset({0, 1})}
        alone = units[frozenset({0})][1]["single"]
        assert alone.share == {0: pytest.approx(1.0)}
        assert (alone.ap_deg, alone.ml_deg) == (0.0, 0.0)
        # A 2.8 mm stretch holds neither column whole beside the other.
        both = units[frozenset({0, 1})][1]["single"]
        assert 0.5 < both.share[0] < 1.0 and both.share[1] == pytest.approx(1.0)
        assert both.total < 2.0

    def test_a_region_kept_too_little_of_does_not_share(self):
        clouds = [_column((3.0, 5.0)), _column((5.5, 6.5))]
        units = find_units(
            clouds, [_hole()], AP_WINDOW, [SINGLE], radius_mm=RADIUS_MM, min_kept=0.9
        )
        assert frozenset({0, 1}) not in units

    def test_a_row_of_shanks_records_what_one_shank_cannot(self):
        offsets = (np.arange(4) - 1.5) * PITCH_MM
        cloud = np.concatenate([_column((3.0, 5.0), y=y) for y in offsets])
        units = find_units([cloud], [_hole()], AP_WINDOW, [SINGLE, QUAD], radius_mm=0.1)
        by_kind = units[frozenset({0})][1]
        # One column at most, and less where no entry point sits over it.
        assert 0.15 < by_kind["single"].share[0] <= 0.25
        assert by_kind["quad"].share[0] == pytest.approx(1.0)
        assert abs(by_kind["quad"].spin_deg - 90.0) <= 15.0  # the row turned onto y

    def test_a_row_too_wide_for_the_bore_records_nothing(self):
        narrow = _hole(radius=0.3)
        units = find_units(
            [_column((3.0, 5.0))], [narrow], AP_WINDOW, [SINGLE, QUAD], radius_mm=0.2
        )
        assert set(units[frozenset({0})][1]) == {"single"}

    def test_nothing_past_the_shanks_length_is_recorded(self):
        deep = [_column((10.5, 11.5))]
        assert not find_units(deep, [_hole()], AP_WINDOW, [SINGLE], radius_mm=0.2)

    def test_the_aim_is_midway_through_what_the_line_records(self):
        clouds = [_column((3.0, 5.0)), _column((5.5, 6.5))]
        units = find_units(clouds, [_hole()], AP_WINDOW, [SINGLE], radius_mm=RADIUS_MM)
        line = units[frozenset({0, 1})][1]["single"]
        at = aim_points(line, dict(enumerate(clouds)), SINGLE, RADIUS_MM)
        assert at[1][2] == pytest.approx(-6.0, abs=0.1)
        assert -5.0 <= at[0][2] <= -4.0  # the lower part of the upper column


class TestCovers:
    def test_partitions_are_single_regions_and_the_given_groups(self):
        groups = [frozenset({0, 1}), frozenset({1, 2})]
        got = {frozenset(p) for p in partitions([0, 1, 2], groups)}
        assert got == {
            frozenset({frozenset({0}), frozenset({1}), frozenset({2})}),
            frozenset({frozenset({0, 1}), frozenset({2})}),
            frozenset({frozenset({0}), frozenset({1, 2})}),
        }

    def test_each_probe_takes_a_bore_of_its_own(self):
        a, b = frozenset({0}), frozenset({1})
        units = _units(
            _line("single", 1, {0: 0.5}),
            _line("single", 2, {0: 0.4}),
            _line("single", 1, {1: 0.5}),
            _line("single", 2, {1: 0.1}),
        )
        seated = seat([a, b], units, [SINGLE])
        assert {u: line.hole for u, line in seated.items()} == {a: 2, b: 1}
        one_bore = _units(_line("single", 1, {0: 0.5}), _line("single", 1, {1: 0.5}))
        assert seat([a, b], one_bore, [SINGLE]) is None

    def test_kinds_go_where_they_record_most_within_the_inventory(self):
        a, b = frozenset({0}), frozenset({1})
        units = _units(
            _line("single", 1, {0: 0.2}),
            _line("quad", 1, {0: 0.5}),
            _line("single", 2, {1: 0.1}),
            _line("quad", 2, {1: 0.8}),
        )
        seated = seat([a, b], units, [SINGLE, QUAD])
        assert {u: line.kind for u, line in seated.items()} == {a: "single", b: "quad"}
        only_quads = _units(_line("quad", 1, {0: 0.5}), _line("quad", 2, {1: 0.8}))
        assert seat([a, b], only_quads, [SINGLE, QUAD]) is None

    def test_covers_come_fewest_probes_first_then_most_recorded(self):
        units = _units(
            _line("single", 1, {0: 0.6}),
            _line("single", 2, {0: 0.5}),
            _line("single", 2, {1: 0.4}),
            _line("single", 1, {0: 0.3, 1: 0.2}),
        )
        found = covers(2, units, [SINGLE])
        assert [(c.n_probes, c.mean_share) for c in found] == [
            (1, pytest.approx(0.25)),
            (2, pytest.approx(0.5)),
        ]
        assert found[1].share == {0: 0.6, 1: 0.4}

    def test_a_region_no_line_records_leaves_no_cover(self):
        assert covers(2, _units(_line("single", 1, {0: 0.6})), [SINGLE]) == []

    def test_best_per_count_keeps_the_first_of_each(self):
        def cover(n: int, share: float) -> Cover:
            lines = {frozenset({i}): _line("single", i, {i: share}) for i in range(n)}
            return Cover(lines, n)

        found = [cover(1, 0.5), cover(1, 0.4), cover(1, 0.3), cover(2, 0.2)]
        assert best_per_count(found) == [found[0], found[3]]
        assert best_per_count(found, 2) == [found[0], found[1], found[3]]
