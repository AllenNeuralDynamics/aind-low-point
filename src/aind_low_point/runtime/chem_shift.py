"""MRI chemical-shift correction context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import SimpleITK as sitk
from aind_mri_utils.chemical_shift import (
    chemical_shift_transform,
    compute_chemical_shift,
)

from aind_low_point.common import Role
from aind_low_point.config import BaseSpecModel, ConfigModel
from aind_low_point.core import AffineTransform


@dataclass(frozen=True)
class ChemShiftContext:
    enabled: bool
    magnet_MHz: float
    default_ppm: float = 3.7
    apply_by_role: set[Role] = field(default_factory=set)
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
            apply_by_role=set(im.chem_shift_apply_by_role),
            image=brain_image,
        )


def _should_apply_chem(asset_model: BaseSpecModel, chem: ChemShiftContext) -> bool:
    if not chem.enabled:
        return False
    mode = asset_model.chem_shift_policy  # "on"|"off"|"auto"
    if mode == "on":
        return True
    if mode == "off":
        return False
    # "auto": follow role defaults
    return asset_model.role in chem.apply_by_role
