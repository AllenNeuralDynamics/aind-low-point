"""Which resonance localized a feature, and when a config must say.

Chemical shift moves geometry by millimetres with nothing downstream to catch
it, so `mr_signal` is required per feature — but only when there is an image to
be shifted in. An atlas-based plan has no `imaging` block and says nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aind_rutter.common import MRSignal
from aind_rutter.config import ConfigModel
from aind_rutter.runtime.chem_shift import ChemShiftContext, _should_apply_chem

IMAGING = {"magnet_frequency_MHz": 599.0, "chem_shift_ppm_default": 3.9}
ASSET: dict[str, Any] = {"key": "brain", "kind": "mesh", "src": "brain.obj"}


def _config(*, imaging: bool, **asset_over: Any) -> dict[str, Any]:
    asset = {**ASSET, **asset_over}
    doc: dict[str, Any] = {"version": 1, "assets": [asset]}
    if imaging:
        doc["imaging"] = dict(IMAGING)
    return doc


def _context(cfg: ConfigModel) -> ChemShiftContext:
    im = cfg.imaging
    if im is None:
        return ChemShiftContext(False, 0.0, 0.0)
    return ChemShiftContext(
        enabled=True,
        magnet_MHz=im.magnet_frequency_MHz,
        default_ppm=im.chem_shift_ppm_default,
        image=None,
    )


def test_an_atlas_config_needs_no_mr_signal() -> None:
    """No image, nothing to correct — and nothing to declare."""
    cfg = ConfigModel.model_validate(_config(imaging=False))
    assert cfg.assets[0].mr_signal is None


def test_an_imaged_config_must_declare_it() -> None:
    with pytest.raises(ValidationError, match="mr_signal is required"):
        ConfigModel.model_validate(_config(imaging=True))


def test_the_error_names_the_offending_asset() -> None:
    with pytest.raises(ValidationError, match="'brain'"):
        ConfigModel.model_validate(_config(imaging=True))


@pytest.mark.parametrize("signal", [m.value for m in MRSignal])
def test_every_member_satisfies_the_requirement(signal: str) -> None:
    cfg = ConfigModel.model_validate(_config(imaging=True, mr_signal=signal))
    assert cfg.assets[0].mr_signal is MRSignal(signal)


def test_a_misspelled_signal_is_refused() -> None:
    """The point of an enum over a tag: this cannot load and go wrong later."""
    with pytest.raises(ValidationError):
        ConfigModel.model_validate(_config(imaging=True, mr_signal="wter"))


@pytest.mark.parametrize(
    ("signal", "shifted"),
    [("water", True), ("fat", False), ("none", False)],
)
def test_only_water_is_translated_into_the_headframe_frame(
    signal: str, shifted: bool
) -> None:
    cfg = ConfigModel.model_validate(_config(imaging=True, mr_signal=signal))
    assert _should_apply_chem(cfg.assets[0], _context(cfg)) is shifted


def test_without_an_image_even_water_stays_put() -> None:
    cfg = ConfigModel.model_validate(_config(imaging=False, mr_signal="water"))
    assert _should_apply_chem(cfg.assets[0], _context(cfg)) is False


def test_a_template_may_declare_it_for_every_instance() -> None:
    """Where the real configs put it: once per template, not once per asset."""
    cfg = ConfigModel.model_validate(
        {
            "version": 1,
            "imaging": dict(IMAGING),
            "asset_templates": {"structure": {"kind": "mesh", "mr_signal": "water"}},
            "assets": [
                {"key": "structure:MD", "src": "md.obj", "templates": ["structure"]},
                {"key": "structure:PL", "src": "pl.obj", "templates": ["structure"]},
            ],
        }
    )
    assert [a.mr_signal for a in cfg.assets] == [MRSignal.WATER, MRSignal.WATER]


def test_an_asset_overrides_its_template() -> None:
    """The headframe is hardware, but it is the one piece found from vaseline."""
    cfg = ConfigModel.model_validate(
        {
            "version": 1,
            "imaging": dict(IMAGING),
            "asset_templates": {"hardware": {"kind": "mesh", "mr_signal": "none"}},
            "assets": [
                {"key": "cone", "src": "c.obj", "templates": ["hardware"]},
                {
                    "key": "headframe",
                    "src": "h.obj",
                    "templates": ["hardware"],
                    "mr_signal": "fat",
                },
            ],
        }
    )
    assert [a.mr_signal for a in cfg.assets] == [MRSignal.NONE, MRSignal.FAT]
