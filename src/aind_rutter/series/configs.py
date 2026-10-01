"""A series spec to one planner config per subject, set and allocation.

A subject's base config supplies everything but the probes: assets, transforms,
the arcs and the rig pitch. Each generated config adds the set's regions as
structure assets, a target at each region's interior point, and the
allocation's probes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aind_rutter.config import ConfigModel
from aind_rutter.series.lock import write_lock
from aind_rutter.series.spec import ProbeSpec, Score, SeriesSpec

# Retro scoring and the recorded share both look a probe's regions up here.
STRUCTURE_KEY = "structure:{label}"
TARGET_PREFIX = "target:"
# The cell cloud retro scoring reads; without it a solve quietly scores centroids.
RETRO_KEY = "retro-targets"
REGION_LOADER = "ccf_annotation_region"
REGION_ASSET_DEFAULTS: dict[str, Any] = {
    "src": "${paths.ccf_annotation_path}",
    "role": "anatomy",
    "tags": ["structure"],
    "scene_tags": ["static", "structure"],
}
REGION_TARGET_DEFAULTS: dict[str, Any] = {"reducer": "points_interior"}


@dataclass(frozen=True)
class SeriesConfig:
    """One generated config and what it is for."""

    subject: str
    set_name: str
    allocation: str
    path: Path


def config_path(run_dir: Path, subject: str, set_name: str, allocation: str) -> Path:
    return run_dir / subject / "configs" / set_name / f"{allocation}.yml"


def region_asset(spec: SeriesSpec, subject: str, label: str) -> dict[str, Any]:
    """The structure asset that draws, masks and measures ``label``."""
    region = spec.region(label)
    kwargs: dict[str, Any] = {
        "acronym": region.ccf[0] if len(region.ccf) == 1 else list(region.ccf),
        "hemisphere": spec.hemisphere.value,
    }
    if region.keep:
        kwargs["keep"] = [half.value for half in region.keep]
    return {
        "key": STRUCTURE_KEY.format(label=label),
        "loader": REGION_LOADER,
        "loader_kwargs": kwargs,
        **copy.deepcopy(REGION_ASSET_DEFAULTS | spec.subjects[subject].region_asset),
    }


def _target_ref(probe: ProbeSpec, label: str) -> dict[str, Any]:
    point = probe.points_RAS.get(label)
    if point is not None:
        return {"kind": "inline", "point_RAS": list(point)}
    return {"kind": "catalog", "key": f"{TARGET_PREFIX}{label}"}


def _probe_decl(probe: ProbeSpec, arc: str) -> dict[str, Any]:
    anchor, *others = probe.regions
    decl: dict[str, Any] = {
        "kind": probe.kind,
        "arc": arc,
        "target": _target_ref(probe, anchor),
    }
    if others:
        decl["also_record"] = [_target_ref(probe, label) for label in others]
    decl["regions"] = list(probe.regions)
    return decl


def allocation_config(
    spec: SeriesSpec,
    subject: str,
    set_name: str,
    allocation: str,
    base: dict[str, Any],
) -> dict[str, Any]:
    """``base`` with the set's regions and the allocation's probes written in.

    The base's own probes are replaced. Probes start on its arcs in turn; the
    solve reassigns them.
    """
    regions = spec.sets[set_name].regions
    probes = spec.sets[set_name].allocations[allocation]
    cfg = copy.deepcopy(base)
    arcs = list((cfg.get("plan") or {}).get("arcs") or {})
    if not arcs:
        raise ValueError("the base config states no plan.arcs")

    assets = [region_asset(spec, subject, label) for label in regions]
    existing = {a["key"] for a in cfg.get("assets") or [] if "key" in a}
    clash = sorted(existing & {a["key"] for a in assets})
    if clash:
        raise ValueError(f"the base config already defines {clash}")
    if spec.sets[set_name].score is Score.RETRO and RETRO_KEY not in existing:
        raise ValueError(f"the set scores retro and the base config has no {RETRO_KEY}")
    cfg["assets"] = [*(cfg.get("assets") or []), *assets]
    cfg["targets"] = [
        *(cfg.get("targets") or []),
        {
            "derive_from": [a["key"] for a in assets],
            "key_prefix": TARGET_PREFIX,
            "hemisphere": spec.hemisphere.value,
            **copy.deepcopy(
                REGION_TARGET_DEFAULTS | spec.subjects[subject].region_target
            ),
        },
    ]
    cfg["plan"]["probes"] = {
        probe.probe_name: _probe_decl(probe, arcs[i % len(arcs)])
        for i, probe in enumerate(probes)
    }
    return cfg


def write_configs(
    spec: SeriesSpec, run_dir: str | Path, *, spec_path: str | Path | None = None
) -> list[SeriesConfig]:
    """Write every subject's configs under ``run_dir``, then the lock file.

    Each config is loaded back before the next is written, so a kind with no
    probe asset or a target the base does not support is refused here and not
    hours into a solve.
    """
    run_dir = Path(run_dir)
    written = []
    for subject, subject_spec in spec.subjects.items():
        base = yaml.safe_load(subject_spec.base_config.read_text())
        for set_name in spec.sets_of(subject):
            for allocation, probes in spec.sets[set_name].allocations.items():
                where = f"{subject} / {set_name} / {allocation}"
                try:
                    cfg = allocation_config(spec, subject, set_name, allocation, base)
                except ValueError as e:
                    raise ValueError(f"{where}: {e}") from None
                path = config_path(run_dir, subject, set_name, allocation)
                path.parent.mkdir(parents=True, exist_ok=True)
                header = (
                    f"# {spec.name}: subject {subject}, set {set_name}, allocation "
                    f"{allocation} ({len(probes)} probes).\n"
                    f"# Generated from {subject_spec.base_config}; edit the series "
                    f"spec, not this file.\n"
                )
                path.write_text(
                    header + yaml.safe_dump(cfg, sort_keys=False, width=100)
                )
                try:
                    ConfigModel.from_yaml(path)
                except Exception as e:
                    raise ValueError(f"{where}: {path} does not load: {e}") from e
                written.append(SeriesConfig(subject, set_name, allocation, path))
    write_lock(spec, run_dir, [c.path for c in written], spec_path=spec_path)
    return written
