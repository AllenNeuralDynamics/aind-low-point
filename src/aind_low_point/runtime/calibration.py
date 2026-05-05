"""Probe calibration bank loading."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from aind_mri_utils.reticle_calibrations import (
    fit_rotation_params_from_manual_calibration,
    fit_rotation_params_from_parallax,
)

from aind_low_point.config import (
    CalibrationReticleModel,
    CalibrationsModel,
    CalibrationSourceModel,
)
from aind_low_point.core import AffineTransform


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


def _get_calibration_rt(
    calibrations: CalibrationsModel,
    reticles: dict[str, CalibrationReticleModel] = {},
) -> dict[str, "AffineTransform"]:
    """
    For each domain probe name, resolve (cal_id → path) then (probe_code → R,t).
    Cache each file load so it's read once.
    """
    cal_files = calibrations.files
    probe_to_ref = calibrations.probe_to_ref
    cache: dict[str, dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    out: dict[str, AffineTransform] = {}

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
