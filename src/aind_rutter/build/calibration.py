"""Probe calibration: loading a bank, and the NewScale frame it is read in."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from aind_mri_utils.reticle_calibrations import (
    fit_rotation_params_from_manual_calibration,
    fit_rotation_params_from_parallax,
    transform_bregma_to_probe,
    transform_probe_to_bregma,
)
from numpy.typing import ArrayLike, NDArray

from aind_rutter.config import (
    CalibrationReticleModel,
    CalibrationsModel,
    CalibrationSourceModel,
)
from aind_rutter.domain.transforms import AffineTransform


def _load_calibration_bank(
    cal_file: CalibrationSourceModel, reticles: dict[str, CalibrationReticleModel]
) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a calibration file that contains multiple probe entries.
    Return a dict mapping probe_code (string) -> (R,t).
    """
    if cal_file.directory:
        if cal_file.reticle is None:
            raise ValueError("Reticle model is required for directory calibration")
        reticle = reticles.get(cal_file.reticle)
        offset = np.array(reticle.offset_RAS, dtype=float)
        rotation = reticle.rotation_z
        cal_by_probe = fit_rotation_params_from_parallax(
            cal_file.directory, offset, rotation
        )[0]
    else:
        cal_by_probe = fit_rotation_params_from_manual_calibration(cal_file.file)[0]
    return {str(k): v for k, v in cal_by_probe.items()}


def _merge_stacked_sources(
    sources: list[CalibrationSourceModel],
    reticles: dict[str, CalibrationReticleModel],
) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load each ``CalibrationSourceModel`` in order and merge their
    probe banks with **last source wins** on per-code conflicts.

    Returns a single ``probe_code → (R, t)`` mapping.
    """
    merged: dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for src in sources:
        bank = _load_calibration_bank(src, reticles)
        merged.update(bank)  # later sources overwrite earlier ones
    return merged


def _get_calibration_rt(
    calibrations: CalibrationsModel,
    reticles: dict[str, CalibrationReticleModel] = {},
) -> dict[str, "AffineTransform"]:
    """
    For each domain probe name, resolve to ``(R, t)``.

    Two config shapes are supported:

    - **Stacked mode** (``sources`` + ``probe_to_code``): merge all
      sources into one bank (last source wins per probe code), then
      look up each probe's code.
    - **Legacy mode** (``files`` + ``probe_to_ref``): per-probe choice
      of ``(cal_id, probe_code)``; each file is loaded once and cached.
    """
    out: dict[str, AffineTransform] = {}

    if calibrations.sources:
        merged = _merge_stacked_sources(calibrations.sources, reticles)
        for probe_name, code in calibrations.probe_to_code.items():
            code = str(code)
            if code not in merged:
                avail = ", ".join(sorted(merged.keys())[:8])
                raise KeyError(
                    f"Calibration probe_code '{code}' for probe '{probe_name}' "
                    f"not found in any of the {len(calibrations.sources)} "
                    f"stacked source(s). Examples available: {avail}"
                    f"{' …' if len(merged) > 8 else ''}"
                )
            R, t = merged[code]
            out[probe_name] = AffineTransform(
                rotation=np.asarray(R, float), translation=np.asarray(t, float)
            )
        return out

    # Legacy mode.
    cal_files = calibrations.files
    probe_to_ref = calibrations.probe_to_ref
    cache: dict[str, dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    for probe_name, ref in probe_to_ref.items():
        # load or reuse the bank
        if ref.cal_id not in cache:
            cal_file = cal_files[ref.cal_id]
            bank = _load_calibration_bank(cal_file, reticles)
            cache[ref.cal_id] = bank
        else:
            bank = cache[ref.cal_id]

        code = str(ref.probe_code)
        if code not in bank:
            # Clear error message showing available keys
            avail = ", ".join(sorted(bank.keys())[:8])
            raise KeyError(
                f"Calibration probe_code '{code}' not found in cal_id '{ref.cal_id}'. "
                f"Examples available: {avail}{' …' if len(bank) > 8 else ''}"
            )

        R, t = bank[code]
        out[probe_name] = AffineTransform(
            rotation=np.asarray(R, float), translation=np.asarray(t, float)
        )

    return out


# --------------------------------------------------------------------------
# NewScale stage frame
# --------------------------------------------------------------------------
# The manipulator reports the probe tip in its own machine frame. Reticle
# calibration gives a rigid (R, t) from bregma RAS to that frame, and the app's
# world is subject LPS — bregma RAS with x and y flipped, bregma being at the
# LPS origin by construction of the AIND MRI registration. Both converters take
# (3,) or (N, 3) and return the same shape.


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
