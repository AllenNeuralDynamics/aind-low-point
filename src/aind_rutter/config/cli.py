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
    CACHE_SOURCE,
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

    offline: bool = False
    """Fetch nothing: packs are read from the cache, and a missing file fails
    naming it. ``RUTTER_OFFLINE=1`` does the same."""


def main() -> None:
    args = tyro.cli(Args)
    try:
        locator = Locator.from_environment(
            args.paths, parse_overrides(args.path), offline=args.offline
        )
        if args.config is not None:
            show_config(args.config, locator)
            searched = locator.searched(args.config)
        else:
            searched = [*locator.files, *_machine(locator)]
        show_files(searched, locator.machine_file)
        show_cache(locator)
    except (OSError, ValueError, LookupError) as e:
        print(f"rutter-paths: {e}", file=sys.stderr)
        sys.exit(1)


def _machine(locator: Locator) -> list[Path]:
    return [] if locator.machine_file is None else [locator.machine_file]


def show_config(config: Path, locator: Locator) -> None:
    from omegaconf import OmegaConf

    raw = OmegaConf.load(config)
    written = OmegaConf.to_container(raw.get("paths") or {}, resolve=False)
    assert isinstance(written, dict)
    located = locate(config, {str(k): v for k, v in written.items()}, locator)
    values = load_resolved(config, locator).get("paths") or {}
    print(f"config: {config}\n")
    if not located:
        print("  no paths: block; every input is written as a location\n")
        return
    width = max(len(name) for name in located)
    for loc in located.values():
        value = str(values.get(loc.name, loc.value))
        mark = "" if Path(value).exists() else "  [missing]"
        print(f"  {loc.name:<{width}}  {value}{mark}")
        set_by = "the config" if loc.source == CONFIG_SOURCE else loc.source
        if loc.source == CACHE_SOURCE:
            set_by = f"the pack cache, from {locator.pack_cache().store}"
        pack = f"pack {loc.pack}, " if loc.pack is not None else ""
        print(f"  {'':<{width}}  {pack}set by {set_by}")
    print()


def show_files(files: list[Path], machine_file: Path | None) -> None:
    print("paths files, first wins:")
    for path in files:
        if not path.is_file():
            print(f"  -  {path}  (absent)")
            continue
        found = read_paths_file(path, machine=path == machine_file)
        print(f"  *  {path}")
        for name, folder in found.names.items():
            print(f"       {name}: {folder}")
        for pack, folder in found.packs.items():
            print(f"       packs.{pack}: {folder}")
        for key, value in found.settings.items():
            print(f"       {key}: {value}")
    print()


def show_cache(locator: Locator) -> None:
    cache = locator.pack_cache()
    s = cache.settings
    versions = cache.versions() if cache.root.is_dir() else []
    total = sum(v.size for v in versions)
    print(f"pack cache: {cache.top}")
    print(f"  store {cache.store}{'  (offline)' if s.offline else ''}")
    print(
        f"  {total / 2**30:.2f} of {s.cache_max_gb:g} GiB, versions unused for "
        f"{s.cache_max_age_days:g} days removed"
    )


if __name__ == "__main__":
    main()
