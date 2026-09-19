"""What a config resolves to, independent of how the config spells it.

Every field a config states — `mr_signal`, `role`, `collidable`, asset tags and
scene tags — feeds a decision the pipeline makes per asset, and those decisions
must not move when the config layer is rearranged. This module computes them
from a validated `ConfigModel`; the golden file beside it pins today's answers.

Chemical shift is the one that has to be exactly right: a flipped decision
displaces geometry by the fat/water offset and nothing downstream notices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aind_rutter.build.assemble import resolve_collidable
from aind_rutter.build.chem_shift import ChemShiftContext, _should_apply_chem
from aind_rutter.build.queries import FIXTURE_EXCLUDED_TAGS, FIXTURE_TAGS
from aind_rutter.collisions import pair_bits
from aind_rutter.config import ConfigModel

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
        image=None,
    )


def _spec_semantics(spec: Any, chem: ChemShiftContext) -> dict[str, Any]:
    shifted = _should_apply_chem(spec, chem)
    ppm = spec.chem_shift_ppm if spec.chem_shift_ppm is not None else chem.default_ppm
    return {
        "kind": str(getattr(spec.kind, "value", spec.kind)),
        "chem_shift": shifted,
        "chem_ppm": ppm if shifted else None,
        "collidable": resolve_collidable(spec),
        "role": None if spec.role is None else spec.role.value,
        "scene_tags": sorted(spec.scene_tags or ()),
        "asset_tags": sorted(spec.tags or ()),
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


def fixtures_for(path: str) -> list[str]:
    """The optimizer's static-obstacle set, by the rule `fixture_node_keys` uses.

    Probes thread *through* the implant via its bores, so an implant-tagged node
    is excluded however else it is tagged. This is the consequence that must not
    move when node tags gain the asset's tags.
    """
    include = FIXTURE_TAGS
    exclude = FIXTURE_EXCLUDED_TAGS
    return sorted(
        node.key
        for node in ConfigModel.from_yaml(ROOT / path).scene.nodes
        if (set(node.tags or ()) & include) and not (set(node.tags or ()) & exclude)
    )


class _PairSpec:
    """`pair_bits` reads a runtime spec; a config model leaves `collidable` unset."""

    def __init__(self, spec: Any) -> None:
        self.collidable = resolve_collidable(spec)
        self.role = spec.role


def pairs_for(path: str) -> list[list[str]]:
    """The pairs the collision backend tests, read through the rule itself."""
    specs = [*(cfg := ConfigModel.from_yaml(ROOT / path)).assets, *cfg.targets]
    bits = {str(s.key): pair_bits(_PairSpec(s)) for s in specs}
    out = set()
    for i, a in enumerate(specs):
        for b in specs[i + 1 :]:
            (ga, ma), (gb, mb) = bits[str(a.key)], bits[str(b.key)]
            if (ma & gb) and (mb & ga):
                out.add(tuple(sorted((str(a.key), str(b.key)))))
    return [list(pair) for pair in sorted(out)]


def all_semantics() -> dict[str, dict[str, Any]]:
    return {
        path: {
            "specs": semantics_for(path),
            "scene": scene_for(path),
            "fixtures": fixtures_for(path),
            "pairs": pairs_for(path),
        }
        for path in CONFIGS
    }


def write_golden() -> None:
    GOLDEN.write_text(json.dumps(all_semantics(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    write_golden()
    print(f"wrote {GOLDEN}")
