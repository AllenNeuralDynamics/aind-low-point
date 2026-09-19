"""Tests for ``aind_rutter.domain.probe_kinds``."""

from __future__ import annotations

import pytest

from aind_rutter.domain.probe_kinds import RECORDING_GEOMETRY, get_recording_geometry

# -- recording geometry table ----------------------------------------------


def test_recording_geometry_2_1_single_shank():
    geom = get_recording_geometry("2.1")
    assert geom.n_shanks == 1
    assert geom.active_ranges_mm == ((0.200, 3.065),)


def test_recording_geometry_2_4_four_shank_short_bank():
    geom = get_recording_geometry("2.4")
    assert geom.n_shanks == 4
    assert all(r == (0.200, 0.905) for r in geom.active_ranges_mm)


def test_recording_geometry_quadbase_four_shank_full_bank():
    geom = get_recording_geometry("quadbase")
    assert geom.n_shanks == 4
    assert all(r == (0.200, 3.065) for r in geom.active_ranges_mm)


def test_recording_geometry_active_center_mm():
    """Active center = (start + end) / 2 averaged across shanks."""
    geom = get_recording_geometry("2.1")
    # Single shank: center = (0.200 + 3.065) / 2 = 1.6325
    assert geom.active_center_mm == pytest.approx(1.6325)

    geom = get_recording_geometry("2.4")
    # Four shanks all (0.200, 0.905): center = 0.5525
    assert geom.active_center_mm == pytest.approx(0.5525)


def test_recording_geometry_unknown_kind_raises():
    with pytest.raises(KeyError, match="probe kind"):
        get_recording_geometry("not-a-real-kind")


def test_recording_geometry_table_keys():
    """Sanity check: known kinds present in the registry."""
    expected_kinds = {"2.1", "2.4", "quadbase"}
    assert expected_kinds <= set(RECORDING_GEOMETRY.keys())
