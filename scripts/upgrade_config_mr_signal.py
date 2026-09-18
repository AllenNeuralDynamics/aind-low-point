"""Add ``mr_signal`` to configs written before the field existed.

Chemical shift used to be decided by ``role`` plus a per-asset
``chem_shift_policy``. That pair could not distinguish an annotation centroid
from a bore centre — both are targets, only the first is localized from water —
so it has been replaced by an explicit ``mr_signal: water|fat|none``.

This upgrades a config in place. ``water`` is derived from what the old rule
resolves to for that very file, so the decision cannot move; the run re-reads
the result and restores the original if any key's answer changed. ``fat`` is
assigned by key, and is documentation only: ``fat`` and ``none`` behave
identically, differing solely in whether the coordinates came from a vaseline
fiducial or from outside the image.

    uv run --python 3.13 python scripts/upgrade_config_mr_signal.py \
        --dry-run examples/*_out.yml

A config with no ``imaging`` block — an atlas-based plan — has no image to be
shifted in, declares no ``mr_signal``, and is skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from aind_rutter.common import MRSignal
from aind_rutter.config import ConfigModel
from aind_rutter.runtime.chem_shift import ChemShiftContext, _should_apply_chem

# Localized from a vaseline fiducial rather than from tissue. Behaviourally the
# same as ``none`` today; recorded so the frame convention stays reversible.
FAT_KEYS = ("headframe", "implant")
FAT_PREFIXES = ("target:hole:",)

_SECTIONS = ("assets", "targets")
_ITEM = re.compile(r"^(\s*)- ")
_BODY_KEY = re.compile(r"^(\s*)[A-Za-z_][A-Za-z0-9_]*:")
_INTERPOLATION = re.compile(r"\$\{[^}]*\}")


class Unchanged(Exception):
    """The file needs no edit."""


def _context(cfg: ConfigModel) -> ChemShiftContext:
    """`ChemShiftContext.from_config` without reading the MRI off a share."""
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


def decisions(cfg: ConfigModel) -> dict[str, bool]:
    """Per expanded key, whether the config shifts it today."""
    chem = _context(cfg)
    return {
        str(spec.key): _should_apply_chem(spec, chem)
        for spec in [*cfg.assets, *cfg.targets]
    }


def signal_for(keys: list[str], shifted: dict[str, bool]) -> MRSignal:
    """The one signal that covers every key a declaration expands to."""
    known = [k for k in keys if k in shifted]
    if not known:
        raise KeyError(f"no expanded key of {keys} appears in the config")
    if len({shifted[k] for k in known}) != 1:
        raise ValueError(f"declaration {keys} resolves both ways: cannot upgrade")
    if shifted[known[0]]:
        return MRSignal.WATER
    if any(k in FAT_KEYS or k.startswith(FAT_PREFIXES) for k in known):
        return MRSignal.FAT
    return MRSignal.NONE


def _raw_items(text: str, section: str) -> list[dict[str, Any]]:
    """The section's declarations as written, before template expansion."""
    doc = yaml.safe_load(_INTERPOLATION.sub("X", text)) or {}
    return [item for item in (doc.get(section) or []) if isinstance(item, dict)]


def _section_lines(lines: list[str], section: str) -> tuple[int, int] | None:
    try:
        start = next(i for i, ln in enumerate(lines) if ln.rstrip() == f"{section}:")
    except StopIteration:
        return None
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not lines[i][0].isspace() and not lines[i].lstrip().startswith("- "):
            return start, i
    return start, len(lines)


def _item_starts(lines: list[str], lo: int, hi: int) -> list[tuple[int, str]]:
    """Line index and body indent of each top-level list item in a section."""
    indent: str | None = None
    out: list[tuple[int, str]] = []
    for i in range(lo + 1, hi):
        m = _ITEM.match(lines[i])
        if not m:
            continue
        if indent is None:
            indent = m.group(1)
        if m.group(1) == indent:
            out.append((i, indent + "  "))
    return out


def _insert_at(lines: list[str], start: int, body: str, limit: int) -> int:
    """Where a new mapping entry may go: before the item's next top-level key.

    Never inside a flow sequence: a ``keys: [a,\\n b]`` continuation sits deeper
    than the body indent, so it is not a candidate.
    """
    for i in range(start + 1, limit):
        m = _BODY_KEY.match(lines[i])
        if m and m.group(1) == body:
            return i
    return limit


def upgrade_text(text: str, shifted: dict[str, bool]) -> str:
    lines = text.splitlines()
    insertions: dict[int, str] = {}
    for section in _SECTIONS:
        bounds = _section_lines(lines, section)
        if bounds is None:
            continue
        lo, hi = bounds
        items = _item_starts(lines, lo, hi)
        raw = _raw_items(text, section)
        if len(items) != len(raw):
            raise ValueError(
                f"{section}: {len(items)} list items in the text but "
                f"{len(raw)} parsed — refusing to guess which is which"
            )
        for n, ((start, body), item) in enumerate(zip(items, raw)):
            if "mr_signal" in item:
                continue
            keys = item.get("keys") or ([item["key"]] if item.get("key") else [])
            signal = signal_for([str(k) for k in keys], shifted)
            stop = items[n + 1][0] if n + 1 < len(items) else hi
            insertions[_insert_at(lines, start, body, stop)] = (
                f"{body}mr_signal: {signal.value}"
            )

    out: list[str] = []
    for i, line in enumerate(lines):
        if i in insertions:
            out.append(insertions[i])
        # both are ignored once mr_signal is present, and `extra="forbid"`
        # will reject them once the fields are gone
        if "chem_shift_policy:" in line:
            continue
        out.append(line)
    if len(lines) in insertions:
        out.append(insertions[len(lines)])
    return "\n".join(_drop_apply_by_role(out)) + "\n"


def _drop_apply_by_role(lines: list[str]) -> list[str]:
    """Remove ``chem_shift_apply_by_role`` and, if block style, its items."""
    out: list[str] = []
    skipping_items = False
    for line in lines:
        if skipping_items:
            if line.lstrip().startswith("- "):
                continue
            skipping_items = False
        if "chem_shift_apply_by_role:" in line:
            skipping_items = line.rstrip().endswith(":")
            continue
        out.append(line)
    return out


def upgrade(path: Path, *, dry_run: bool) -> str:
    cfg = ConfigModel.from_yaml(path, require_mr_signal=False)
    if cfg.imaging is None:
        raise Unchanged("no imaging block — an atlas plan shifts nothing")
    before = decisions(cfg)
    original = path.read_text()
    updated = upgrade_text(original, before)
    if updated == original:
        raise Unchanged("already declares mr_signal")
    if dry_run:
        return _summary(before, updated, path)
    path.write_text(updated)
    try:
        after = decisions(ConfigModel.from_yaml(path))  # now strict
    except Exception:
        path.write_text(original)
        raise
    if after != before:
        path.write_text(original)
        moved = sorted(k for k in before if before.get(k) != after.get(k))
        raise ValueError(f"{path}: decision moved for {moved} — reverted")
    return _summary(before, updated, path)


def _summary(before: dict[str, bool], updated: str, path: Path) -> str:
    counts = {m.value: updated.count(f"mr_signal: {m.value}") for m in MRSignal}
    shifted = sum(before.values())
    return f"{path}: {shifted}/{len(before)} keys shift; wrote " + ", ".join(
        f"{n}×{name}" for name, n in counts.items() if n
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument(
        "--dry-run", action="store_true", help="report without writing anything"
    )
    args = ap.parse_args(argv)

    failed = 0
    for path in args.paths:
        try:
            print(upgrade(path, dry_run=args.dry_run))
        except Unchanged as skip:
            print(f"{path}: skipped — {skip}")
        except Exception as exc:  # a bad file must not stop the batch
            failed += 1
            print(f"{path}: FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
