"""MRI chemical-shift correction context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import SimpleITK as sitk
from aind_mri_utils.chemical_shift import (
    chemical_shift_transform,
    compute_chemical_shift,
)

from aind_rutter.common import MRSignal
from aind_rutter.config import BaseSpecModel, ConfigModel
from aind_rutter.core import AffineTransform


@dataclass(frozen=True)
class ChemShiftContext:
    enabled: bool
    magnet_MHz: float
    default_ppm: float = 3.7
    # transforms to apply to geometry in image/LPS space
    image: Optional[sitk.Image] = None
    # lazy cache: ppm -> AffineTransform (observed → corrected)
    _cache: dict[float, AffineTransform] = field(
        default_factory=dict, repr=False, compare=False
    )

    def pt_transform_for_ppm(self, ppm: Optional[float] = None) -> "AffineTransform":
        """
        Return the transform that moves points from observed (chem-shifted)
        positions to corrected positions for the given ppm, in LPS mm.
        """
        if self.image:
            if ppm is None:
                ppm = self.default_ppm
            if ppm in self._cache:
                return self._cache[ppm]
            chem_shift_pt_R, chem_shift_pt_t = chemical_shift_transform(
                compute_chemical_shift(self.image, ppm=ppm)
            )
            tf = AffineTransform(chem_shift_pt_R, chem_shift_pt_t)
            self._cache[ppm] = tf
        else:
            tf = AffineTransform.identity()

        return tf

    @classmethod
    def from_config(cls, cfg: ConfigModel) -> ChemShiftContext:
        im = cfg.imaging
        if im is None:
            return ChemShiftContext(False, 0.0, 0.0)
        # Build correction using your existing aind_mri_utils helpers.
        # If your `compute_chemical_shift` accepts only ppm, scale ppm if you want
        # frequency-awareness; otherwise pass ppm through (common in practice).
        if im.image_path:
            brain_image = sitk.ReadImage(str(im.image_path))
        else:
            brain_image = None
        return ChemShiftContext(
            enabled=True,
            magnet_MHz=im.magnet_frequency_MHz,
            default_ppm=im.chem_shift_ppm_default,
            image=brain_image,
        )


def _should_apply_chem(asset_model: BaseSpecModel, chem: ChemShiftContext) -> bool:
    """Whether this feature must be translated into the headframe frame.

    The correction moves water-localized geometry into the frame the vaseline
    fiducials define; fat-localized geometry already sits in it, and geometry the
    image never saw has nothing to correct.
    """
    return chem.enabled and asset_model.mr_signal is MRSignal.WATER
