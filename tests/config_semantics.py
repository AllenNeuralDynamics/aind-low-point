"""What a config resolves to, independent of how the config spells it.

The flag vocabularies — `Role`, `Capability`, asset tags, scene tags and the
collision group/mask labels — are being collapsed onto scene tags. Every one of
them feeds a decision the pipeline makes per asset, and those decisions must not
move. This module computes them from a validated `ConfigModel`; the golden file
beside it pins today's answers.

Chemical shift is the one that has to be exactly right: a flipped decision
displaces geometry by the fat/water offset and nothing downstream notices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aind_rutter.common import Capability
from aind_rutter.config import ConfigModel
from aind_rutter.runtime.chem_shift import ChemShiftContext, _should_apply_chem

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).with_name("config_semantics.json")

# Every tracked subject config that validates. The plan-only file has its own
# schema and carries no assets.
CONFIGS = (
    "examples/786864-config.yml",
    "examples/836656-config-T12.yml",
    "examples/836656-config.yml",
    "examples/837229-config.yml",
    "examples/build5-template-config.yml",
)


def _chem_context(cfg: ConfigModel) -> ChemShiftContext:
    """`ChemShiftContext.from_config` without reading the MRI off disk.

    The decision and the ppm do not depend on the image; only the transform
    does, and the image lives on a share a test cannot reach.
    """
    im = cfg.imaging
    if im is None:
        return ChemShiftContext(False, 0.0, 0.0)
    return ChemShiftContext(
        enabled=True,
        magnet_MHz=im.magnet_frequency_MHz,
        default_ppm=im.chem_shift_ppm_default,
        apply_by_role=set(im.chem_shift_apply_by_role),
        image=None,
    )


def _spec_semantics(spec: Any, chem: ChemShiftContext) -> dict[str, Any]:
    caps = Capability(0)
    for cap in spec.caps or ():
        caps |= cap
    shifted = _should_apply_chem(spec, chem)
    ppm = spec.chem_shift_ppm if spec.chem_shift_ppm is not None else chem.default_ppm
    return {
        "kind": str(getattr(spec.kind, "value", spec.kind)),
        "chem_shift": shifted,
        "chem_ppm": ppm if shifted else None,
        "collidable": bool(caps & Capability.COLLIDABLE),
        "renderable": bool(caps & Capability.RENDERABLE),
        "scene_tags": sorted(spec.scene_tags or ()),
        "asset_tags": sorted(spec.tags or ()),
        "collision_group": spec.collision.group,
        "collision_mask": sorted(spec.collision.mask or ()),
    }


def semantics_for(path: str) -> dict[str, Any]:
    """Per-key resolved decisions for one config, as the golden file stores them."""
    cfg = ConfigModel.from_yaml(ROOT / path)
    chem = _chem_context(cfg)
    out: dict[str, Any] = {}
    for spec in [*cfg.assets, *cfg.targets]:
        out[str(spec.key)] = _spec_semantics(spec, chem)
    return out


def scene_for(path: str) -> dict[str, dict[str, Any]]:
    """The scene nodes a config produces, keyed by node id.

    Node creation currently keys off ``transform or scene_tags``, so tagging an
    asset that has neither — every probe *kind* asset — would invent a node.
    Pinning the node set is what makes the tag migration checkable.
    """
    cfg = ConfigModel.from_yaml(ROOT / path)
    return {
        node.key: {
            "asset": node.asset,
            "tags": sorted(node.tags or ()),
            "pose_source_probe": node.pose_source_probe,
        }
        for node in cfg.scene.nodes
    }


def all_semantics() -> dict[str, dict[str, Any]]:
    return {
        path: {"specs": semantics_for(path), "scene": scene_for(path)}
        for path in CONFIGS
    }


def write_golden() -> None:
    GOLDEN.write_text(json.dumps(all_semantics(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    write_golden()
    print(f"wrote {GOLDEN}")
