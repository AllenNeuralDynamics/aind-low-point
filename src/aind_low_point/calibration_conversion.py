"""NewScale machine-coordinate ↔ subject LPS conversions.

The NewScale manipulator reports the probe tip in its own "probe"
machine frame. Reticle-based calibration (loaded from a manual or
parallax file via :mod:`aind_mri_utils.reticle_calibrations`) gives a
rigid affine ``(R, t)`` that maps **bregma RAS** ↔ **probe machine
frame**. The trame app's world is **subject LPS**, related to bregma
RAS by the standard sign flip on x and y (bregma is at LPS origin by
construction of the AIND MRI registration).

Two converters in this module:

- :func:`newscale_to_lps` — NewScale ``(x, y, z)`` → subject LPS ``(x, y, z)``
- :func:`lps_to_newscale` — inverse

Both accept ``(3,)`` or ``(N, 3)`` and return the same shape.
"""

from __future__ import annotations

import numpy as np
from aind_mri_utils.reticle_calibrations import (
    transform_bregma_to_probe,
    transform_probe_to_bregma,
)
from numpy.typing import ArrayLike, NDArray

from aind_low_point.core import AffineTransform


def _bregma_RAS_to_subject_LPS(p: NDArray[np.floating]) -> NDArray[np.floating]:
    """Flip x and y signs: bregma RAS ↔ subject LPS."""
    out = np.asarray(p, dtype=np.float64).copy()
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


# Symmetric: same flip in both directions.
_subject_LPS_to_bregma_RAS = _bregma_RAS_to_subject_LPS


def newscale_to_lps(
    xyz_newscale: ArrayLike, cal: AffineTransform
) -> NDArray[np.floating]:
    """Convert one or more NewScale machine-frame points to subject LPS.

    ``cal`` is the per-probe calibration ``(R, t)`` loaded by
    :func:`runtime.calibration._get_calibration_rt`.
    """
    pts = np.asarray(xyz_newscale, dtype=np.float64)
    single = pts.ndim == 1
    pts2 = pts.reshape(-1, 3)
    bregma_ras = transform_probe_to_bregma(pts2, cal.rotation, cal.translation)
    lps = _bregma_RAS_to_subject_LPS(bregma_ras)
    return lps.reshape(3) if single else lps


def lps_to_newscale(xyz_lps: ArrayLike, cal: AffineTransform) -> NDArray[np.floating]:
    """Convert one or more subject-LPS points to NewScale machine frame.

    Inverse of :func:`newscale_to_lps`.
    """
    pts = np.asarray(xyz_lps, dtype=np.float64)
    single = pts.ndim == 1
    pts2 = pts.reshape(-1, 3)
    bregma_ras = _subject_LPS_to_bregma_RAS(pts2)
    newscale = transform_bregma_to_probe(bregma_ras, cal.rotation, cal.translation)
    return newscale.reshape(3) if single else newscale
