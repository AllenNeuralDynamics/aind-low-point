"""The one file that states a series: its mice, regions, sets and allocations.

Regions, sets and allocations hold for every subject; a subject brings its
anatomy, its implant and which sets it runs. An allocation says which probe
records which regions of a set, so a decision made once is a line here and not
a hand-edited config.
"""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aind_rutter.ccf.ontology import CCFOntology
from aind_rutter.ccf.regions import Half, parse_halves

# Names become file names and catalog keys, which split on ":".
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
# What a subject may not restate: the series fills these in per region.
REGION_ASSET_FILLED = frozenset({"key", "loader", "loader_kwargs"})
REGION_TARGET_FILLED = frozenset({"derive_from", "key_prefix", "hemisphere"})


def _name_errors(what: str, names: Any) -> list[str]:
    bad = sorted(n for n in names if not _NAME.fullmatch(n))
    if not bad:
        return []
    return [
        f"{what} {bad}: a name is letters, digits and _ . + - , starting with a "
        f"letter or digit"
    ]


class Hemisphere(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegionDef(_Closed):
    """A label's voxels: one or more CCF structures, cut to the halves kept."""

    ccf: tuple[str, ...] = Field(min_length=1)
    keep: tuple[Half, ...] = ()

    @field_validator("ccf", mode="before")
    @classmethod
    def _one_or_several(cls, v: Any) -> Any:
        return (v,) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _known_and_consistent(self) -> RegionDef:
        ontology = CCFOntology.from_bundled()
        unknown = [a for a in self.ccf if ontology.find_by_acronym(a) is None]
        if unknown:
            raise ValueError(f"not CCF acronyms: {unknown}")
        parse_halves(self.keep)
        return self


class ProbeSpec(_Closed):
    """One probe of an allocation and the regions it records."""

    kind: str
    # The first anchors the pose; the rest are recorded beside it.
    regions: tuple[str, ...] = Field(min_length=1)
    name: str | None = None
    # Where a region is aimed, when not at its interior point (subject RAS mm).
    points_RAS: dict[str, tuple[float, float, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _points_are_its_regions(self) -> ProbeSpec:
        foreign = sorted(set(self.points_RAS) - set(self.regions))
        if foreign:
            raise ValueError(f"points_RAS names regions not on this probe: {foreign}")
        return self

    @property
    def probe_name(self) -> str:
        return self.name or "_".join(self.regions)


class SetSpec(_Closed):
    """The regions of one session and the ways to put probes on them."""

    regions: tuple[str, ...] = Field(min_length=1)
    allocations: dict[str, tuple[ProbeSpec, ...]] = Field(min_length=1)


class SubjectSpec(_Closed):
    """One mouse: a config with everything but probes, and its implant's bores."""

    base_config: Path
    holes: Path
    sets: tuple[str, ...] | Literal["all"] = "all"
    # Fields every region's structure asset carries in this subject's config,
    # and every region's derived target, beyond what the series fills in.
    region_asset: dict[str, Any] = Field(default_factory=dict)
    region_target: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _leaves_the_filled_fields(self) -> SubjectSpec:
        for name, filled in (
            ("region_asset", REGION_ASSET_FILLED),
            ("region_target", REGION_TARGET_FILLED),
        ):
            stated = sorted(set(getattr(self, name)) & filled)
            if stated:
                raise ValueError(f"{name} may not state {stated}: the series does")
        return self


class SeriesSpec(_Closed):
    name: str
    hemisphere: Hemisphere
    subjects: dict[str, SubjectSpec] = Field(min_length=1)
    # How many probes of each kind exist to mount at once.
    inventory: dict[str, int] = Field(min_length=1)
    # label -> definition; a label that is a CCF acronym needs no entry.
    regions: dict[str, RegionDef] = Field(default_factory=dict)
    sets: dict[str, SetSpec] = Field(min_length=1)

    def region(self, label: str) -> RegionDef:
        return self.regions.get(label) or RegionDef(ccf=(label,))

    def sets_of(self, subject: str) -> tuple[str, ...]:
        stated = self.subjects[subject].sets
        return tuple(self.sets) if stated == "all" else stated

    @model_validator(mode="after")
    def _consistent(self) -> SeriesSpec:
        errors: list[str] = []
        errors += _name_errors("series name", [self.name])
        errors += _name_errors("subject names", self.subjects)
        errors += _name_errors("region labels", self.regions)
        errors += _name_errors("set names", self.sets)
        for kind, count in self.inventory.items():
            if count < 1:
                errors.append(f"inventory: {kind} has {count}")
        for subject, s in self.subjects.items():
            if s.sets != "all":
                unknown = sorted(set(s.sets) - set(self.sets))
                if unknown:
                    errors.append(f"subject {subject}: unknown sets {unknown}")
        for set_name, s in self.sets.items():
            errors += [f"set {set_name}: {e}" for e in self._set_errors(s)]
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def _set_errors(self, s: SetSpec) -> list[str]:
        errors = []
        ontology = CCFOntology.from_bundled()
        undefined = [
            r
            for r in s.regions
            if r not in self.regions and ontology.find_by_acronym(r) is None
        ]
        if undefined:
            errors.append(f"regions {undefined} are neither defined nor CCF acronyms")
        repeated = sorted(r for r, n in Counter(s.regions).items() if n > 1)
        if repeated:
            errors.append(f"regions {repeated} are listed twice")
        for name, probes in s.allocations.items():
            errors += [
                f"allocation {name}: {e}" for e in self._allocation_errors(s, probes)
            ]
        return errors + _name_errors("allocation names", s.allocations)

    def _allocation_errors(
        self, s: SetSpec, probes: tuple[ProbeSpec, ...]
    ) -> list[str]:
        errors = []
        recorded = Counter(r for p in probes for r in p.regions)
        missing = [r for r in s.regions if r not in recorded]
        foreign = sorted(set(recorded) - set(s.regions))
        twice = sorted(r for r, n in recorded.items() if n > 1)
        if missing:
            errors.append(f"records none of {missing}")
        if foreign:
            errors.append(f"records {foreign}, which the set does not list")
        if twice:
            errors.append(f"records {twice} on more than one probe")
        names = Counter(p.probe_name for p in probes)
        errors += _name_errors("probe names", names)
        shared = sorted(n for n, k in names.items() if k > 1)
        if shared:
            errors.append(f"probe names {shared} are used twice")
        for kind, used in Counter(p.kind for p in probes).items():
            have = self.inventory.get(kind)
            if have is None:
                errors.append(f"kind {kind} is not in the inventory")
            elif used > have:
                errors.append(f"uses {used} {kind}, the inventory has {have}")
        return errors


def load_spec(path: str | Path) -> SeriesSpec:
    """Read a series spec; a subject's relative paths are relative to the file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a series spec is a mapping")
    for subject in (raw.get("subjects") or {}).values():
        if not isinstance(subject, dict):
            continue
        for field in ("base_config", "holes"):
            if isinstance(subject.get(field), str):
                subject[field] = str(path.parent / Path(subject[field]).expanduser())
    return SeriesSpec.model_validate(raw)
