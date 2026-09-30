"""A set of solved plans: emit writes its index, the planner browses it.

A plan set is an emit output directory: ``plans/*.plan.yml`` beside a
``manifest.json`` that indexes them. A directory without the index still
loads, with the rank and coverage its filenames carry and nothing else.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
_FILENAME = re.compile(r"plan-(\d+)-cov([\d.]+)-")


@dataclass(frozen=True)
class PlanEntry:
    """One plan of a set; ``None`` where the set does not record it."""

    file: str
    rank: int  # the plan-file number: this entry is plan-{rank:02d}
    coverage: float | None = None
    fcl_mm: float | None = None
    n_arcs: int | None = None
    candidate: int | None = None
    holes: dict[str, int] = field(default_factory=dict)
    arcs: dict[str, str | None] = field(default_factory=dict)

    def holes_str(self, probes: tuple[str, ...]) -> str:
        return " ".join(f"{p} {self.holes[p]}" for p in probes if p in self.holes)


@dataclass(frozen=True)
class PlanSet:
    plans_dir: Path
    entries: tuple[PlanEntry, ...]
    probes: tuple[str, ...] = ()
    config: str | None = None
    indexed: bool = True  # False: read from filenames, no manifest

    def path(self, entry: PlanEntry) -> Path:
        return self.plans_dir / entry.file


def write_manifest(
    out_dir: Path,
    entries: list[PlanEntry],
    *,
    probes: list[str],
    config: str | None,
) -> Path:
    path = Path(out_dir) / MANIFEST_NAME
    payload = {
        "version": MANIFEST_VERSION,
        "config": config,
        "probes": list(probes),
        "plans": [asdict(e) for e in entries],
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def load_plan_set(path: Path) -> PlanSet:
    """Load an emit output directory, or the ``plans/`` folder inside one."""
    path = Path(path)
    out_dir, plans_dir = (
        (path.parent, path) if path.name == "plans" else (path, path / "plans")
    )
    if not plans_dir.is_dir():
        plans_dir = out_dir  # plan files directly in the given directory
    manifest = out_dir / MANIFEST_NAME
    if manifest.exists():
        raw = json.loads(manifest.read_text())
        entries = tuple(PlanEntry(**e) for e in raw["plans"])
        return PlanSet(
            plans_dir, entries, tuple(raw.get("probes", ())), raw.get("config")
        )
    found: list[PlanEntry] = []
    for f in sorted(plans_dir.glob("*.plan.yml")):
        m = _FILENAME.match(f.name)
        found.append(
            PlanEntry(
                f.name,
                int(m.group(1)) if m else len(found) + 1,
                float(m.group(2)) if m else None,
            )
        )
    if not found:
        raise FileNotFoundError(f"no *.plan.yml in {plans_dir}")
    return PlanSet(plans_dir, tuple(found), indexed=False)
