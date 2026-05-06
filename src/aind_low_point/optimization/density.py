"""Target density model and per-probe coverage integral.

For v1 the density is a Gaussian on the target centroid:

    density(p) = exp(-||p − target||² / (2·σ²))

The optimizer's coverage objective integrates this density along each
shank's active recording range, summed over all shanks listed in the
probe kind's :class:`RecordingGeometry`. This naturally encourages
the optimizer to:

- Aim the shaft so the *shank row* is centered on the target (small
  perpendicular distance for every shank), not just the
  position-bearing shank.
- Set ``past_target_mm`` so the active recording region is centered
  on the target's projection along the shaft.

Numerics
--------
The integral is approximated by Simpson's rule on a fixed sample
count along each active range. Sample count is configurable; default
of 41 covers the typical 0.7–3 mm ranges with ~20–75 µm step — much
finer than the spatial scale of the density (σ ≈ 0.5 mm). A trapezoid
fallback is used when the sample count is even.

The implementation is pure-numpy and JAX-traceable as long as the
density callable is too. The default :func:`gaussian_density` factory
is JAX-friendly (uses ``np.exp`` only).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aind_low_point.optimization.geometry import Capsule
from aind_low_point.optimization.recording import RecordingGeometry


DensityFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]


def gaussian_density(
    target_LPS: ArrayLike, sigma_mm: float = 0.5
) -> DensityFn:
    """Return a Gaussian density centered on ``target_LPS`` with
    isotropic standard deviation ``sigma_mm``.

    ``density(points)`` accepts a single ``(3,)`` point or a batched
    ``(N, 3)`` array and returns a scalar or ``(N,)`` array.
    """
    target = np.asarray(target_LPS, dtype=np.float64)
    inv_two_sigma_sq = 1.0 / (2.0 * float(sigma_mm) ** 2)

    def fn(points):
        p = np.asarray(points, dtype=np.float64)
        d2 = np.sum((p - target) ** 2, axis=-1)
        return np.exp(-d2 * inv_two_sigma_sq)

    return fn


def _simpson_weights(n: int) -> NDArray[np.floating]:
    """Composite Simpson 1/3 weights for an odd ``n``. For even ``n``
    falls back to trapezoidal."""
    if n < 2:
        return np.ones(max(n, 1), dtype=np.float64)
    if n % 2 == 0:
        # Trapezoidal for even sample count.
        w = np.ones(n, dtype=np.float64)
        w[0] = w[-1] = 0.5
        return w
    # Simpson 1/3: 1, 4, 2, 4, 2, ..., 4, 1, divided by 3.
    w = np.ones(n, dtype=np.float64)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w / 3.0


def integrate_density_along_shank(
    density_fn: DensityFn,
    shank: Capsule,
    *,
    start_mm: float,
    end_mm: float,
    n_samples: int = 41,
) -> float:
    """Line-integral of ``density_fn`` along the shank's axis between
    ``start_mm`` and ``end_mm`` measured from ``shank.p0`` (the tip).

    The integration variable is *physical arc length along the
    shank*, not capsule parameter — so the result is in
    ``density × mm`` units.
    """
    p0 = np.asarray(shank.p0, dtype=np.float64)
    p1 = np.asarray(shank.p1, dtype=np.float64)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return 0.0
    unit = direction / length
    s = np.linspace(start_mm, end_mm, n_samples)
    points = p0[None, :] + s[:, None] * unit[None, :]
    values = density_fn(points)
    weights = _simpson_weights(n_samples)
    span = float(end_mm - start_mm)
    if span <= 0.0:
        return 0.0
    h = span / (n_samples - 1) if n_samples > 1 else span
    return float(np.sum(values * weights) * h)


def coverage(
    density_fn: DensityFn,
    shank_capsules: list[Capsule],
    recording_geom: RecordingGeometry,
    *,
    n_samples: int = 41,
) -> float:
    """Total coverage = sum of per-shank line integrals over each
    shank's active recording range.

    The number of capsules and the number of active ranges in
    ``recording_geom`` must match. Mismatches raise ``ValueError``
    with a clear message — defensive because the optimizer surfaces
    silent zero-coverage as 'no good poses found,' which is the worst
    failure mode.
    """
    if len(shank_capsules) != recording_geom.n_shanks:
        raise ValueError(
            f"shank count mismatch: got {len(shank_capsules)} capsules vs. "
            f"{recording_geom.n_shanks} ranges in recording geometry"
        )
    total = 0.0
    for shank, (start_mm, end_mm) in zip(
        shank_capsules, recording_geom.active_ranges_mm
    ):
        total += integrate_density_along_shank(
            density_fn, shank,
            start_mm=start_mm, end_mm=end_mm, n_samples=n_samples,
        )
    return total
