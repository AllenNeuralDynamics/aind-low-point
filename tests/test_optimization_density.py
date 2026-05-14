"""Tests for ``aind_low_point.optimization.density`` and
``aind_low_point.optimization.recording``."""

from __future__ import annotations

import numpy as np
import pytest

from aind_low_point.optimization.density import (
    coverage,
    gaussian_density,
    integrate_density_along_shank,
)
from aind_low_point.optimization.geometry import Capsule
from aind_low_point.optimization.recording import (
    RECORDING_GEOMETRY,
    RecordingGeometry,
    get_recording_geometry,
)

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


# -- gaussian density factory ---------------------------------------------


def test_gaussian_density_at_center_is_one():
    fn = gaussian_density([1.0, 2.0, 3.0], sigma_mm=0.5)
    assert fn([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_gaussian_density_at_one_sigma_is_exp_minus_half():
    fn = gaussian_density([0, 0, 0], sigma_mm=1.0)
    # Distance 1 mm from center along z; sigma=1 → exp(-0.5)
    assert fn([0, 0, 1.0]) == pytest.approx(np.exp(-0.5))


def test_gaussian_density_batched_input():
    fn = gaussian_density([0, 0, 0], sigma_mm=0.5)
    pts = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]])
    vals = fn(pts)
    assert vals.shape == (3,)
    assert vals[0] == pytest.approx(1.0)
    # 0.5 mm = 1 sigma → exp(-0.5)
    assert vals[1] == pytest.approx(np.exp(-0.5))


# -- per-shank line integral ----------------------------------------------


def test_integrate_density_uniform_density_equals_span():
    """A density that's identically 1 integrates to the span length."""
    fn = lambda p: np.ones(p.shape[:-1]) if p.ndim > 1 else 1.0  # noqa: E731
    shank = Capsule(p0=np.array([0, 0, 0]), p1=np.array([0, 0, 1]), radius=0.0)
    span = integrate_density_along_shank(
        fn, shank, start_mm=0.0, end_mm=2.0, n_samples=21
    )
    assert span == pytest.approx(2.0)


def test_integrate_density_zero_span_is_zero():
    fn = gaussian_density([0, 0, 0], sigma_mm=1.0)
    shank = Capsule(p0=np.array([0, 0, 0]), p1=np.array([0, 0, 1]), radius=0.0)
    assert (
        integrate_density_along_shank(fn, shank, start_mm=0.5, end_mm=0.5, n_samples=21)
        == 0.0
    )


def test_integrate_density_gaussian_axis_aligned():
    """Integrating an isotropic Gaussian along its axis from -∞ to +∞
    gives σ·√(2π). Truncating to ±5σ recovers nearly all of it."""
    sigma = 0.3
    fn = gaussian_density([0, 0, 0], sigma_mm=sigma)
    # Shank along +z passing through origin; integrate from -2 to +2 mm
    shank = Capsule(p0=np.array([0, 0, -2.0]), p1=np.array([0, 0, 1.0]), radius=0.0)
    # With p0 at -2 along z, "start_mm" measured from p0 along shank
    # direction (+z), so to integrate symmetrically around z=0 we use
    # start_mm=0 (at z=-2) and end_mm=4 (at z=+2).
    integral = integrate_density_along_shank(
        fn, shank, start_mm=0.0, end_mm=4.0, n_samples=201
    )
    expected = sigma * np.sqrt(2 * np.pi)
    assert integral == pytest.approx(expected, rel=1e-4)


def test_integrate_density_centered_perpendicular_offset():
    """A shank parallel to the Gaussian's axis but offset by d
    perpendicular returns ``exp(-d²/2σ²) · σ·√(2π)``."""
    sigma = 0.4
    fn = gaussian_density([0, 0, 0], sigma_mm=sigma)
    d = 0.6  # perpendicular offset along x
    shank = Capsule(p0=np.array([d, 0, -3.0]), p1=np.array([d, 0, 0.0]), radius=0.0)
    integral = integrate_density_along_shank(
        fn, shank, start_mm=0.0, end_mm=6.0, n_samples=301
    )
    expected = np.exp(-(d**2) / (2 * sigma**2)) * sigma * np.sqrt(2 * np.pi)
    assert integral == pytest.approx(expected, rel=1e-3)


# -- multi-shank coverage --------------------------------------------------


def test_coverage_sum_across_shanks():
    """Coverage equals sum of per-shank integrals."""
    fn = gaussian_density([0, 0, 0], sigma_mm=0.5)
    # Two parallel shanks offset by 0.5 mm in x; both crossing through
    # the target at z=0.
    shanks = [
        Capsule(np.array([-0.25, 0, -2]), np.array([-0.25, 0, 1]), 0.0),
        Capsule(np.array([+0.25, 0, -2]), np.array([+0.25, 0, 1]), 0.0),
    ]
    geom = RecordingGeometry(active_ranges_mm=((0.0, 3.0), (0.0, 3.0)))
    total = coverage(fn, shanks, geom, n_samples=201)
    # Sum equals 2 × (single-shank integral at offset 0.25)
    single = integrate_density_along_shank(
        fn, shanks[0], start_mm=0.0, end_mm=3.0, n_samples=201
    )
    assert total == pytest.approx(2 * single, rel=1e-6)


def test_coverage_shank_count_mismatch_raises():
    fn = gaussian_density([0, 0, 0], sigma_mm=0.5)
    shanks = [
        Capsule(np.zeros(3), np.array([0, 0, 1]), 0.0),
    ]
    geom = RecordingGeometry(active_ranges_mm=((0.0, 1.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="shank count mismatch"):
        coverage(fn, shanks, geom)


def test_coverage_optimum_at_active_center():
    """The optimal probe placement (per the coverage gradient w.r.t.
    depth) puts the active region's center at the target."""
    sigma = 0.3
    fn = gaussian_density([0, 0, 0], sigma_mm=sigma)
    geom = get_recording_geometry("2.4")  # active 0.200–0.905 per shank
    active_center = geom.active_center_mm  # ≈ 0.5525

    # Probe shaft along +z, single shank for simplicity (use modified geom).
    geom1 = RecordingGeometry(active_ranges_mm=((0.200, 0.905),))

    def cov_at_depth(depth_mm: float) -> float:
        # Tip below the target by ``depth_mm`` along -z (so positive
        # depth puts tip past the target). Shaft direction is +z.
        tip = np.array([0.0, 0.0, -depth_mm])
        shank = Capsule(tip, tip + np.array([0, 0, 5.0]), 0.0)
        return coverage(fn, [shank], geom1, n_samples=201)

    # Sample around the predicted optimum and confirm it's the max.
    depths = np.linspace(0.0, 1.5, 31)
    covs = np.array([cov_at_depth(d) for d in depths])
    best_depth = depths[int(np.argmax(covs))]
    assert best_depth == pytest.approx(active_center, abs=0.05)
