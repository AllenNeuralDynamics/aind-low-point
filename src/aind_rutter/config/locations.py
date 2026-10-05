"""Where each name in a config's ``paths:`` block is on this machine.

A config names its input folders through ``${paths.<name>}``; the folders
themselves differ by machine. Each name takes the first of: a ``--path``
override, a paths file given to the command or in ``RUTTER_PATHS``, a
``rutter.paths.yml`` in the config's folder or one above it, the machine's
``paths.yml``, for a pack the pack cache, and last the value the config writes.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TextIO

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from aind_rutter.packs import CachedPack, PackCache, PackRef, PackSettings

log = logging.getLogger(__name__)

PATHS_FILE_NAME = "rutter.paths.yml"
CONFIG_SOURCE = "config"
CACHE_SOURCE = "cache"
OVERRIDE_SOURCE = "--path"

PathsFiles = Annotated[tuple[Path, ...], NoDecode]
"""Paths files from the environment, separated by ``os.pathsep`` (``:`` on
POSIX, ``;`` on Windows, where ``:`` is part of a drive)."""


def split_paths_files(value: Any) -> Any:
    if isinstance(value, str):
        return tuple(Path(p) for p in value.split(os.pathsep) if p.strip())
    return value


class LocationSettings(BaseSettings):
    """``RUTTER_PATHS`` and ``RUTTER_CONFIG_DIR``, read by any command that
    loads a config."""

    model_config = SettingsConfigDict(env_prefix="RUTTER_", frozen=True)

    paths: PathsFiles = ()
    # Replaces the platform's config folder, which holds the machine's paths.yml.
    config_dir: Path | None = None

    _split = field_validator("paths", mode="before")(split_paths_files)


def machine_paths_file() -> Path:
    """The machine's ``paths.yml``: in ``RUTTER_CONFIG_DIR``, else the
    platform's config folder for rutter."""
    config_dir = LocationSettings().config_dir
    if config_dir is None:
        import platformdirs

        config_dir = platformdirs.user_config_path("rutter", appauthor=False)
    return config_dir / "paths.yml"


@dataclass(frozen=True)
class Location:
    """What one ``paths:`` name resolved to, and what set it: ``--path``, a
    paths file, ``cache`` or ``config``; ``pack`` is the pack the config names
    for it, if any."""

    name: str
    value: str
    source: str
    pack: PackRef | None = None
    cached: CachedPack | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Locator:
    """The places a config's ``paths:`` names are looked up, highest first."""

    overrides: Mapping[str, str] = field(default_factory=dict)
    files: tuple[Path, ...] = ()
    search_upward: bool = True
    machine_file: Path | None = field(default_factory=machine_paths_file)
    cache: PackCache | None = None
    offline: bool = False

    def pack_cache(self) -> PackCache:
        """The cache packs no paths file locates come from, set by the
        environment and the machine's paths file; ``offline`` forbids fetching
        whatever they say."""
        cache = self.cache
        if cache is None:
            machine = self.machine_file
            settings = (
                read_paths_file(machine, machine=True).settings
                if machine is not None and machine.is_file()
                else {}
            )
            cache = PackCache(PackSettings.with_machine_file(settings))
        if self.offline and not cache.settings.offline:
            cache = PackCache(cache.settings.model_copy(update={"offline": True}))
        if cache is not self.cache:
            # A frozen dataclass memoizes through object.__setattr__.
            object.__setattr__(self, "cache", cache)
        return cache

    @classmethod
    def from_environment(
        cls,
        files: Iterable[Path] = (),
        overrides: Mapping[str, str] | None = None,
        *,
        offline: bool = False,
    ) -> Locator:
        """``files`` from the command line ahead of ``RUTTER_PATHS``."""
        return cls(
            overrides=dict(overrides or {}),
            files=(*files, *LocationSettings().paths),
            offline=offline,
        )

    def searched(self, config_path: Path) -> list[Path]:
        """Every paths file consulted for ``config_path``, highest first,
        whether or not it exists."""
        out = list(self.files)
        if self.search_upward:
            folder = Path(os.path.abspath(config_path)).parent
            out += [d / PATHS_FILE_NAME for d in (folder, *folder.parents)]
        if self.machine_file is not None:
            out.append(self.machine_file)
        return out


def parse_overrides(items: Iterable[str]) -> dict[str, str]:
    """``NAME=DIR`` pairs from the command line."""
    out: dict[str, str] = {}
    for item in items:
        name, sep, folder = item.partition("=")
        if not sep or not name.strip() or not folder.strip():
            raise ValueError(f"--path {item!r}: expected NAME=DIR")
        out[name.strip()] = _expand(folder.strip(), Path.cwd())
    return out


@dataclass(frozen=True)
class PathsFile:
    """One paths file: folders by name, folders by pack, and the settings only
    the machine's file may hold (``pack_store``, ``cache``)."""

    names: Mapping[str, str]
    packs: Mapping[str, str]
    settings: Mapping[str, Any]


MACHINE_ONLY = ("pack_store", "cache")


def read_paths_file(path: Path, *, machine: bool = False) -> PathsFile:
    """A paths file, relative folders taken against the file's own folder."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return PathsFile({}, {}, {})
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a paths file maps names to folders")
    settings = {k: data.pop(k) for k in MACHINE_ONLY if k in data}
    if settings and not machine:
        raise ValueError(
            f"{path}: only the machine's paths.yml sets {', '.join(settings)}"
        )
    packs = data.pop("packs", None) or {}
    if not isinstance(packs, dict):
        raise ValueError(f"{path}: `packs` maps pack versions to folders")
    for pack in packs:
        PackRef.parse(str(pack))
    return PathsFile(
        names={str(k): _folder(path, k, v) for k, v in data.items()},
        packs={str(k): _folder(path, f"packs.{k}", v) for k, v in packs.items()},
        settings=settings,
    )


def _folder(path: Path, name: object, folder: object) -> str:
    if not isinstance(folder, str) or not folder.strip():
        raise ValueError(f"{path}: `{name}` must be a folder, got {folder!r}")
    if "${" in folder:
        raise ValueError(
            f"{path}: `{name}` is an interpolation; a paths file holds folders"
        )
    return _expand(folder.strip(), path.parent)


def _expand(folder: str, base: Path) -> str:
    folder = os.path.expanduser(folder)
    return folder if is_absolute_location(folder) else os.path.normpath(base / folder)


def is_absolute_location(folder: str) -> bool:
    """POSIX, drive-letter and UNC locations alike, whatever the platform."""
    # ``Z:/data`` is absolute on Windows only; read it as absolute everywhere
    # so a paths file means the same thing on every machine that reads it.
    return (
        folder.startswith(("/", "\\"))
        or re.match(r"[A-Za-z]:[\\/]", folder) is not None
    )


def written_pack(name: str, value: Any) -> PackRef | None:
    """The pack a ``paths:`` value names (``{pack: hardware/v1}``), if any."""
    if not isinstance(value, Mapping):
        return None
    if set(value) != {"pack"}:
        raise ValueError(
            f"paths.{name}: a mapping value names a pack as {{pack: name/version}}"
        )
    return PackRef.parse(str(value["pack"]))


def locate(
    config_path: Path,
    written: Mapping[str, Any],
    locator: Locator,
) -> dict[str, Location]:
    """The winning location for every name the config's ``paths:`` declares.

    Within each paths file a name's own line beats its pack's line. A pack no
    file locates comes from the cache. Names a paths file sets that the config
    does not declare are ignored, so one file serves configs that each use
    part of it; an override naming nothing the config declares is refused as
    a misspelling.
    """
    unknown = sorted(set(locator.overrides) - set(written))
    if unknown:
        raise ValueError(
            f"--path {', '.join(unknown)}: not in {config_path}'s paths "
            f"(declared: {', '.join(sorted(written)) or 'none'})"
        )
    packs = {name: written_pack(name, value) for name, value in written.items()}
    found = {
        name: Location(name, value, OVERRIDE_SOURCE, packs[name])
        for name, value in locator.overrides.items()
    }
    for path in locator.searched(config_path):
        if not path.is_file():
            continue
        paths_file = read_paths_file(path, machine=path == locator.machine_file)
        for name in written:
            if name in found:
                continue
            pack = packs[name]
            if name in paths_file.names:
                found[name] = Location(name, paths_file.names[name], str(path), pack)
            elif pack is not None and str(pack) in paths_file.packs:
                folder = paths_file.packs[str(pack)]
                found[name] = Location(name, folder, f"{path} (packs)", pack)
    for name, value in written.items():
        if name in found:
            continue
        pack = packs[name]
        if pack is None:
            found[name] = Location(name, str(value), CONFIG_SOURCE)
            continue
        cached = locator.pack_cache().open(pack)
        found[name] = Location(
            name, str(cached.folder), CACHE_SOURCE, pack, cached=cached
        )
    return {name: found[name] for name in written}


@dataclass(frozen=True)
class Loaded:
    """A config's YAML resolved, where each ``paths:`` name was found, and the
    cached packs it reads from."""

    data: Any
    packs: tuple[CachedPack, ...]
    locations: Mapping[str, Location] = field(default_factory=dict)


def load_resolved(path: str | Path, locator: Locator | None = None) -> Any:
    """A config's YAML with its ``paths:`` located and every interpolation
    resolved."""
    return load_located(path, locator).data


def load_located(path: str | Path, locator: Locator | None = None) -> Loaded:
    """``load_resolved``, keeping the cached packs so the caller can fetch what
    the validated config names (``fetch_referenced``)."""
    from omegaconf import DictConfig, OmegaConf

    path = Path(path)
    raw = OmegaConf.load(path)
    if locator is None:
        locator = Locator.from_environment()
    cached: list[CachedPack] = []
    located: dict[str, Location] = {}
    block = raw.get("paths") if isinstance(raw, DictConfig) else None
    if isinstance(block, DictConfig):
        # Unresolved, so a name derived from another (``${.template_dir}/x``)
        # is still a reference and follows wherever that name is located.
        written = OmegaConf.to_container(block, resolve=False)
        assert isinstance(written, dict)
        located = locate(path, {str(k): v for k, v in written.items()}, locator)
        missing = []
        for loc in located.values():
            report(loc)
            if loc.source != CONFIG_SOURCE:
                block[loc.name] = loc.value
            elif OmegaConf.is_missing(block, loc.name):
                missing.append(loc.name)
            if loc.cached is not None:
                cached.append(loc.cached)
        if missing:
            tried = "\n  ".join(str(p) for p in locator.searched(path))
            raise LookupError(
                f"{path}: nothing sets paths {', '.join(missing)}. Searched:\n"
                f"  {tried}\nAdd a line `{missing[0]}: <folder>` to one of "
                f"these, or pass --path {missing[0]}=<folder>."
            )
    data = OmegaConf.to_container(raw, resolve=True)
    packs = tuple(cached)
    # Files the config names exactly are fetched now, because validation
    # checks that some of them exist; a folder is only created until the
    # config is expanded.
    fetch_referenced(packs, references(data), folders=False)
    return Loaded(data, packs, located)


_reported: set[Location] = set()


def report(loc: Location) -> None:
    """Log where a name was found, once per process however often it is loaded."""
    if loc in _reported:
        return
    _reported.add(loc)
    where = "the config" if loc.source == CONFIG_SOURCE else loc.source
    via = f"pack {loc.pack}, " if loc.pack is not None else ""
    log.info("paths.%s = %s (%sfrom %s)", loc.name, loc.value, via, where)


def report_to(stream: TextIO) -> None:
    """Write each location's line to ``stream``, for a caller that configures
    no logging of its own; repeated calls add nothing."""
    if not any(getattr(h, "rutter_report", False) for h in log.handlers):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.rutter_report = True  # type: ignore[attr-defined]
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def fetch_referenced(
    packs: Iterable[CachedPack], values: Iterable[Any], *, folders: bool
) -> None:
    """Fetch every pack file ``values`` name, with its companions; a named
    folder whole when ``folders``."""
    paths = [v for v in values if isinstance(v, (str, Path)) and str(v)]
    for pack in packs:
        pack.ensure_referenced(paths, folders=folders)


def references(config: Any) -> Iterator[Any]:
    """Every location a config names outside ``paths:``, whose values are the
    pack roots and would otherwise fetch every pack whole."""
    if isinstance(config, Mapping):
        yield from strings_in({k: v for k, v in config.items() if k != "paths"})


def strings_in(node: Any) -> Iterator[Any]:
    """Every string or path anywhere in a nested mapping or list."""
    if isinstance(node, Mapping):
        for value in node.values():
            yield from strings_in(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from strings_in(value)
    elif isinstance(node, (str, Path)):
        yield node


def record_pack_folder(paths_file: Path, pack: PackRef, folder: Path) -> None:
    """Add (or replace) ``packs: {pack: folder}`` in a paths file, leaving every
    other line as written; checked by reading the file back."""
    import json

    line = f"  {pack}: {json.dumps(str(folder))}\n"
    lines = (
        paths_file.read_text(encoding="utf-8").splitlines(keepends=True)
        if paths_file.is_file()
        else []
    )
    before = read_paths_file(paths_file, machine=True) if lines else None
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines = [x for x in lines if not re.match(rf"\s+{re.escape(str(pack))}\s*:", x)]
    at = next((i for i, x in enumerate(lines) if re.match(r"packs\s*:\s*$", x)), None)
    if at is None:
        lines += ["packs:\n", line]
    else:
        lines.insert(at + 1, line)
    text = "".join(lines)
    paths_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths_file.with_name(paths_file.name + ".new")
    tmp.write_text(text, encoding="utf-8")
    after = read_paths_file(tmp, machine=True)
    unchanged = before is None or (
        after.names == before.names
        and after.settings == before.settings
        and {k: v for k, v in after.packs.items() if k != str(pack)}
        == {k: v for k, v in before.packs.items() if k != str(pack)}
    )
    if not unchanged or after.packs.get(str(pack)) != str(folder):
        tmp.unlink()
        raise ValueError(
            f"{paths_file}: could not add the pack's line in place; add "
            f"`{line.strip()}` under `packs:` by hand"
        )
    os.replace(tmp, paths_file)
