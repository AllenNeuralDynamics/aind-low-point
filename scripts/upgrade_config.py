"""Bring a config up to the current schema, one migration at a time.

Each migration rewrites the YAML text — preserving comments and OmegaConf
interpolations — and the run is accepted only if the config's *behaviour* is
unchanged: the same chemical-shift decision and ppm per key, the same
collidability, and the same set of colliding pairs. Anything else and the
original is restored.

    uv run --python 3.13 python scripts/upgrade_config.py --dry-run examples/*.yml
    uv run --python 3.13 python scripts/upgrade_config.py --list

Run this *before* taking a version that deletes the superseded fields: the
migrations read the old fields to derive the new ones, so a config can only be
upgraded while the code still understands it.

A config with no ``imaging`` block — an atlas-based plan — has no image to be
shifted in, so the chemical-shift migration skips it.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aind_rutter.common import MRSignal, Role
from aind_rutter.config import ConfigModel
from aind_rutter.runtime.build import resolve_collidable
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


def _is_probe(spec: Any) -> bool:
    """Probe-ness, by whichever mechanism this config still uses."""
    if spec.role is Role.PROBE:
        return True
    if spec.collision.group == "probe":
        return True
    return str(spec.key or "").startswith("probe:")


def colliding_pairs(cfg: ConfigModel) -> set[tuple[str, str]]:
    """The pairs `FCLBackend` would test.

    Read through the rule rather than the labels, because a migration removes
    the labels — `tests/test_collision_pairs.py` is what proves the two agree
    before one replaces the other.
    """
    specs = [*cfg.assets, *cfg.targets]
    state = {str(s.key): (resolve_collidable(s), _is_probe(s)) for s in specs}
    out: set[tuple[str, str]] = set()
    for i, a in enumerate(specs):
        for b in specs[i + 1 :]:
            (ca, pa), (cb, pb) = state[str(a.key)], state[str(b.key)]
            if ca and cb and (pa or pb):
                out.add(tuple(sorted((str(a.key), str(b.key)))))  # type: ignore[arg-type]
    return out


def behaviour(cfg: ConfigModel) -> dict[str, Any]:
    """Everything a migration must leave alone.

    A migration changes how a config says things. If any of these moves, it
    changed what the config *means*, and the run is rejected.
    """
    chem = _context(cfg)
    return {
        "chem": {
            str(s.key): (
                _should_apply_chem(s, chem),
                (s.chem_shift_ppm if s.chem_shift_ppm is not None else chem.default_ppm)
                if _should_apply_chem(s, chem)
                else None,
            )
            for s in [*cfg.assets, *cfg.targets]
        },
        "collidable": {
            str(s.key): resolve_collidable(s) for s in [*cfg.assets, *cfg.targets]
        },
        "pairs": sorted(colliding_pairs(cfg)),
    }


def _diff(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Which keys a migration moved, named well enough to chase."""
    out: list[str] = []
    for field in ("chem", "collidable"):
        for key in sorted(set(before[field]) | set(after[field])):
            if before[field].get(key) != after[field].get(key):
                out.append(
                    f"{field}[{key}]: {before[field].get(key)} -> "
                    f"{after[field].get(key)}"
                )
    if before["pairs"] != after["pairs"]:
        gained = set(map(tuple, after["pairs"])) - set(map(tuple, before["pairs"]))
        lost = set(map(tuple, before["pairs"])) - set(map(tuple, after["pairs"]))
        if gained:
            out.append(f"pairs gained: {sorted(gained)}")
        if lost:
            out.append(f"pairs lost: {sorted(lost)}")
    return out


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


def rewrite_declarations(
    text: str,
    field: str,
    value_for: Callable[[list[str]], str | None],
    *,
    drop_keys: tuple[str, ...] = (),
    drop_blocks: tuple[str, ...] = (),
) -> str:
    """Insert ``field: <value>`` into every declaration in assets/targets.

    Works on the text so comments and ``${...}`` interpolations survive. The
    value is chosen per declaration from the keys it expands to, and a
    declaration that already states the field is left alone.

    ``drop_keys`` removes single lines; ``drop_blocks`` removes a key and, when
    it is block style, the list items under it.
    """
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
            if field in item:
                continue
            keys = item.get("keys") or ([item["key"]] if item.get("key") else [])
            value = value_for([str(k) for k in keys])
            if value is None:
                continue
            stop = items[n + 1][0] if n + 1 < len(items) else hi
            insertions[_insert_at(lines, start, body, stop)] = f"{body}{field}: {value}"

    out: list[str] = []
    for i, line in enumerate(lines):
        if i in insertions:
            out.append(insertions[i])
        if any(f"{key}:" in line for key in drop_keys):
            continue
        out.append(line)
    if len(lines) in insertions:
        out.append(insertions[len(lines)])
    for key in drop_blocks:
        out = _drop_block(out, key)
    return "\n".join(out) + "\n"


def _drop_block(lines: list[str], key: str) -> list[str]:
    """Remove ``key:`` and, when it is block style, everything nested under it."""
    out: list[str] = []
    depth: int | None = None
    for line in lines:
        if depth is not None:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            # a block sequence sits at the key's own indent, a nested mapping
            # deeper; both belong to the key being removed
            nested = indent > depth
            sequence = indent == depth and stripped.startswith("-")
            if stripped and (nested or sequence):
                continue
            depth = None
        if re.match(rf"^\s*{re.escape(key)}:", line):
            depth = len(line) - len(line.lstrip())
            continue
        out.append(line)
    return out


@dataclass(frozen=True)
class Migration:
    """One schema step: what it is called, when it applies, how it rewrites."""

    name: str
    summary: str
    applies: Callable[[ConfigModel, str], bool]
    rewrite: Callable[[str, ConfigModel], str]


def _mr_signal_applies(cfg: ConfigModel, text: str) -> bool:
    if cfg.imaging is None:
        return False  # an atlas plan has no image to be shifted in
    return any(s.mr_signal is None for s in [*cfg.assets, *cfg.targets])


def _mr_signal_rewrite(text: str, cfg: ConfigModel) -> str:
    shifted = decisions(cfg)
    return rewrite_declarations(
        text,
        "mr_signal",
        lambda keys: signal_for(keys, shifted).value,
        drop_keys=("chem_shift_policy",),
        drop_blocks=("chem_shift_apply_by_role",),
    )


def _collidable_applies(cfg: ConfigModel, text: str) -> bool:
    return any(
        s.collidable is None and (s.caps or s.collision.group or s.collision.mask)
        for s in [*cfg.assets, *cfg.targets]
    )


def _collidable_rewrite(text: str, cfg: ConfigModel) -> str:
    """State collidability and probe-ness; drop the labels they replace.

    The pair filter is a rule — both sides collidable, at least one a probe —
    so `collision.group` and `collision.mask` carry nothing the two say.
    """
    by_key = {str(s.key): s for s in [*cfg.assets, *cfg.targets]}

    def collidable_for(keys: list[str]) -> str | None:
        """``true`` where it is, nothing where it is not.

        ``False`` is the model default, and no template can be setting ``True``
        — the field did not exist before this migration — so writing it would
        be noise on every target in the file.
        """
        known = [k for k in keys if k in by_key]
        if not known:
            return None
        values = {resolve_collidable(by_key[k]) for k in known}
        if len(values) != 1:
            raise ValueError(f"declaration {keys} is collidable both ways")
        return "true" if values.pop() else None

    def role_for(keys: list[str]) -> str | None:
        known = [k for k in keys if k in by_key]
        if not known:
            return None
        roles = {
            "probe"
            if _is_probe(by_key[k])
            else ("fixture" if by_key[k].collision.group == "fixture" else None)
            for k in known
        }
        if len(roles) != 1:
            raise ValueError(f"declaration {keys} spans roles {roles}")
        return roles.pop()

    text = rewrite_declarations(text, "collidable", collidable_for)
    # Both go as blocks: a generated config writes `caps` as the IntFlag's
    # integers rather than its names, so dropping the key alone would strand
    # the list items under it.
    return rewrite_declarations(
        text, "role", role_for, drop_blocks=("caps", "collision")
    )


def _options_applies(cfg: ConfigModel, text: str) -> bool:
    """Read from the text: the model has retired the field, so a parsed config
    never shows it."""
    return any(re.match(r"^options:", ln) for ln in text.splitlines())


def _options_rewrite(text: str, cfg: ConfigModel) -> str:
    """`color_map` and `remove_last_color` were the reference notebook's palette
    knobs; nothing has read them since."""
    return "\n".join(_drop_block(text.splitlines(), "options")) + "\n"


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        name="mr_signal",
        summary="chemical shift moves from role + policy to an explicit signal",
        applies=_mr_signal_applies,
        rewrite=_mr_signal_rewrite,
    ),
    Migration(
        name="collidable",
        summary="caps and collision labels become `collidable` plus a role",
        applies=_collidable_applies,
        rewrite=_collidable_rewrite,
    ),
    Migration(
        name="options",
        summary="drop the `options` block, which nothing has read",
        applies=_options_applies,
        rewrite=_options_rewrite,
    ),
)


def upgrade(path: Path, *, dry_run: bool) -> str:
    """Apply every applicable migration, or leave the file exactly as it was."""
    cfg = ConfigModel.from_yaml(path, legacy=True)
    before = behaviour(cfg)

    original = path.read_text()
    text = original
    applied: list[str] = []
    for migration in MIGRATIONS:
        if not migration.applies(cfg, text):
            continue
        text = migration.rewrite(text, cfg)
        applied.append(migration.name)

    if not applied:
        # Worth naming: run over a directory and this is the line that explains
        # why a file was passed over.
        if cfg.imaging is None:
            raise Unchanged("no imaging block — an atlas plan shifts nothing")
        raise Unchanged("already current")
    if text == original:
        raise Unchanged(f"{', '.join(applied)} had nothing to write")
    if dry_run:
        return f"{path}: would apply {', '.join(applied)}"

    path.write_text(text)
    try:
        after = behaviour(ConfigModel.from_yaml(path))
    except Exception:
        path.write_text(original)
        raise
    moved = _diff(before, after)
    if moved:
        path.write_text(original)
        raise ValueError(f"{path}: behaviour moved, reverted — " + "; ".join(moved))
    return f"{path}: applied {', '.join(applied)}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument(
        "--dry-run", action="store_true", help="report without writing anything"
    )
    ap.add_argument("--list", action="store_true", help="print the migrations and exit")
    args = ap.parse_args(argv)

    if args.list:
        for migration in MIGRATIONS:
            print(f"{migration.name:12s} {migration.summary}")
        return 0
    if not args.paths:
        ap.error("give at least one config path, or --list")

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
