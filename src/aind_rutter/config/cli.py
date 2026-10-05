"""``rutter-paths``: where a config's ``paths:`` names are on this machine.

Examples
--------
Each name, its folder and what set it::

    rutter-paths examples/build5-template-config.yml

The paths files this machine reads, without a config::

    rutter-paths
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from aind_rutter.config.locations import (
    CONFIG_SOURCE,
    Locator,
    load_resolved,
    locate,
    parse_overrides,
    read_paths_file,
)


@dataclass
class Args:
    """Command-line arguments for ``rutter-paths``."""

    config: tyro.conf.Positional[Path | None] = None
    """Config YAML whose ``paths:`` names to locate."""

    paths: Annotated[tuple[Path, ...], tyro.conf.UseAppendAction] = ()
    """Paths file read ahead of ``RUTTER_PATHS`` and the files found by
    search; repeatable, first wins."""

    path: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """``NAME=DIR`` for one name, ahead of every paths file; repeatable."""


def main() -> None:
    args = tyro.cli(Args)
    try:
        locator = Locator.from_environment(args.paths, parse_overrides(args.path))
        if args.config is None:
            show_files(locator.files + _machine(locator))
        else:
            show_config(args.config, locator)
    except (OSError, ValueError, LookupError) as e:
        print(f"rutter-paths: {e}", file=sys.stderr)
        sys.exit(1)


def _machine(locator: Locator) -> tuple[Path, ...]:
    return () if locator.machine_file is None else (locator.machine_file,)


def show_config(config: Path, locator: Locator) -> None:
    from omegaconf import OmegaConf

    raw = OmegaConf.load(config)
    written = OmegaConf.to_container(raw.get("paths") or {}, resolve=False)
    assert isinstance(written, dict)
    located = locate(config, {str(k): v for k, v in written.items()}, locator)
    values = load_resolved(config, locator).get("paths") or {}
    print(f"config: {config}\n")
    rows = [
        (
            loc.name,
            str(values.get(loc.name, loc.value)),
            loc.source,
            Path(str(values.get(loc.name, loc.value))).exists(),
        )
        for loc in located.values()
    ]
    if not rows:
        print("no paths: block; every input is written as a location")
    width = max((len(r[0]) for r in rows), default=0)
    for name, value, source, exists in rows:
        mark = "" if exists else "  [missing]"
        print(f"  {name:<{width}}  {value}{mark}")
        set_by = "the config" if source == CONFIG_SOURCE else source
        print(f"  {'':<{width}}  set by {set_by}")
    print()
    show_files(tuple(locator.searched(config)))


def show_files(files: tuple[Path, ...]) -> None:
    print("paths files, first wins:")
    for path in files:
        if not path.is_file():
            print(f"  -  {path}  (absent)")
            continue
        names = read_paths_file(path)
        print(f"  *  {path}  ({len(names)} name{'s' * (len(names) != 1)})")
        for name, folder in names.items():
            print(f"       {name}: {folder}")


if __name__ == "__main__":
    main()
