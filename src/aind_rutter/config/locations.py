"""Where each name in a config's ``paths:`` block is on this machine.

A config names its input folders through ``${paths.<name>}``; the folders
themselves differ by machine. Each name takes the first of: a ``--path``
override, a paths file given to the command or in ``RUTTER_PATHS``, a
``rutter.paths.yml`` in the config's folder or one above it, the machine's
``paths.yml``, and last the value the config writes.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

log = logging.getLogger(__name__)

PATHS_FILE_NAME = "rutter.paths.yml"
CONFIG_SOURCE = "config"
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
    paths file, or ``config``."""

    name: str
    value: str
    source: str


@dataclass(frozen=True)
class Locator:
    """The places a config's ``paths:`` names are looked up, highest first."""

    overrides: Mapping[str, str] = field(default_factory=dict)
    files: tuple[Path, ...] = ()
    search_upward: bool = True
    machine_file: Path | None = field(default_factory=machine_paths_file)

    @classmethod
    def from_environment(
        cls,
        files: Iterable[Path] = (),
        overrides: Mapping[str, str] | None = None,
    ) -> Locator:
        """``files`` from the command line ahead of ``RUTTER_PATHS``."""
        return cls(
            overrides=dict(overrides or {}),
            files=(*files, *LocationSettings().paths),
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


def read_paths_file(path: Path) -> dict[str, str]:
    """A paths file's names and folders, relative folders taken against the
    file's own folder."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a paths file maps names to folders")
    out: dict[str, str] = {}
    for name, folder in data.items():
        if not isinstance(folder, str) or not folder.strip():
            raise ValueError(f"{path}: `{name}` must be a folder, got {folder!r}")
        if "${" in folder:
            raise ValueError(
                f"{path}: `{name}` is an interpolation; a paths file holds folders"
            )
        out[str(name)] = _expand(folder.strip(), path.parent)
    return out


def _expand(folder: str, base: Path) -> str:
    folder = os.path.expanduser(folder)
    return folder if _is_absolute(folder) else os.path.normpath(base / folder)


def _is_absolute(folder: str) -> bool:
    # ``Z:/data`` is absolute on Windows only; read it as absolute everywhere
    # so a paths file means the same thing on every machine that reads it.
    return (
        folder.startswith(("/", "\\"))
        or re.match(r"[A-Za-z]:[\\/]", folder) is not None
    )


def locate(
    config_path: Path,
    written: Mapping[str, Any],
    locator: Locator,
) -> dict[str, Location]:
    """The winning location for every name the config's ``paths:`` declares.

    Names a paths file sets that the config does not declare are ignored, so
    one file serves configs that each use part of it; an override naming
    nothing the config declares is refused as a misspelling.
    """
    unknown = sorted(set(locator.overrides) - set(written))
    if unknown:
        raise ValueError(
            f"--path {', '.join(unknown)}: not in {config_path}'s paths "
            f"(declared: {', '.join(sorted(written)) or 'none'})"
        )
    found = {
        name: Location(name, value, OVERRIDE_SOURCE)
        for name, value in locator.overrides.items()
    }
    for path in locator.searched(config_path):
        if not path.is_file():
            continue
        for name, value in read_paths_file(path).items():
            if name in written and name not in found:
                found[name] = Location(name, value, str(path))
    for name, value in written.items():
        if name not in found:
            found[name] = Location(name, str(value), CONFIG_SOURCE)
    return {name: found[name] for name in written}


def load_resolved(path: str | Path, locator: Locator | None = None) -> Any:
    """A config's YAML with its ``paths:`` located and every interpolation
    resolved."""
    from omegaconf import DictConfig, OmegaConf

    path = Path(path)
    raw = OmegaConf.load(path)
    if locator is None:
        locator = Locator.from_environment()
    block = raw.get("paths") if isinstance(raw, DictConfig) else None
    if isinstance(block, DictConfig):
        # Unresolved, so a name derived from another (``${.template_dir}/x``)
        # is still a reference and follows wherever that name is located.
        written = OmegaConf.to_container(block, resolve=False)
        assert isinstance(written, dict)
        located = locate(path, {str(k): v for k, v in written.items()}, locator)
        missing = []
        for loc in located.values():
            if loc.source != CONFIG_SOURCE:
                block[loc.name] = loc.value
                log.info("paths.%s = %s (from %s)", loc.name, loc.value, loc.source)
            elif OmegaConf.is_missing(block, loc.name):
                missing.append(loc.name)
        if missing:
            tried = "\n  ".join(str(p) for p in locator.searched(path))
            raise LookupError(
                f"{path}: nothing sets paths {', '.join(missing)}. Searched:\n"
                f"  {tried}\nAdd a line `{missing[0]}: <folder>` to one of "
                f"these, or pass --path {missing[0]}=<folder>."
            )
    return OmegaConf.to_container(raw, resolve=True)
