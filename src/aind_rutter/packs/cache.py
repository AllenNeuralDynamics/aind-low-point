"""The pack cache: where fetched pack files live, and how it stays small.

Everything here is immutable and can be fetched again, so removing the wrong
thing costs a download and nothing else. The unit of removal is a pack
version; last use is a marker file's mtime; a sweep runs at most once a day,
claimed by creating a dated stamp file exclusively, so no lock or index exists
to go wrong.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from aind_rutter.packs import store as store_io
from aind_rutter.packs.manifest import Manifest, PackRef

log = logging.getLogger(__name__)

LAYOUT = "v1"
USED = ".used"
DAY_S = 86400.0
TOUCH_EVERY_S = 3600.0
CACHEDIR_TAG = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by rutter (aind-rutter).\n"
    "# For information about cache directory tags see https://bford.info/cachedir/\n"
)


class PackSettings(BaseSettings):
    """``RUTTER_PACK_STORE``, ``RUTTER_CACHE``, ``RUTTER_CACHE_MAX_GB``,
    ``RUTTER_CACHE_MAX_AGE_DAYS``, ``RUTTER_OFFLINE`` and
    ``RUTTER_VERIFY_DOWNLOADS``. The machine's paths file may set the store and
    the two limits; the environment outranks it."""

    model_config = SettingsConfigDict(env_prefix="RUTTER_", frozen=True)

    pack_store: str = store_io.DEFAULT_STORE
    cache: Path | None = None
    cache_max_gb: float = 10.0
    cache_max_age_days: float = 60.0
    offline: bool = False
    verify_downloads: bool = True

    @classmethod
    def with_machine_file(cls, machine: Mapping[str, Any]) -> PackSettings:
        """Values from the machine's paths file beneath the environment's."""
        from_env = cls()
        given = {
            "pack_store": machine.get("pack_store"),
            "cache_max_gb": (machine.get("cache") or {}).get("max_gb"),
            "cache_max_age_days": (machine.get("cache") or {}).get("max_age_days"),
        }
        return cls(
            **{
                k: v
                for k, v in given.items()
                if v is not None and k not in from_env.model_fields_set
            }
        )

    def cache_dir(self) -> Path:
        if self.cache is not None:
            return self.cache
        import platformdirs

        return platformdirs.user_cache_path("rutter", appauthor=False)


@dataclass(frozen=True)
class CachedVersion:
    """One pack version's folder in the cache, as the sweep sees it."""

    folder: Path
    label: str
    size: int
    last_used: float


@dataclass
class PackCache:
    settings: PackSettings = field(default_factory=PackSettings)

    @property
    def top(self) -> Path:
        return self.settings.cache_dir()

    @property
    def root(self) -> Path:
        return self.top / LAYOUT

    @property
    def store(self) -> str:
        return store_io.store_url(self.settings.pack_store)

    def store_root(self) -> Path:
        """The default store's packs share one root; any other store has its
        own, so two stores publishing the same version cannot mix."""
        if self.store == store_io.DEFAULT_STORE:
            return self.root / "packs"
        tag = hashlib.sha256(self.store.encode()).hexdigest()[:8]
        return self.root / "stores" / tag / "packs"

    def version_dir(self, pack: PackRef) -> Path:
        return self.store_root() / pack.name / pack.version

    def open(self, pack: PackRef) -> CachedPack:
        """The pack's cache folder, its manifest fetched on first use. Marks
        the version used and sweeps the cache if no sweep has run today."""
        vdir = self.version_dir(pack)
        manifest_path = vdir / "manifest.json"
        if manifest_path.is_file():
            manifest = Manifest.from_json(manifest_path.read_text("utf-8"), pack)
        elif self.settings.offline:
            raise FileNotFoundError(
                f"{pack} is not in the cache at {vdir} and RUTTER_OFFLINE is set; "
                f"run `rutter-data pull {pack}` while online"
            )
        else:
            manifest = store_io.read_manifest(self.store, pack)
            self._prepare_top()
            vdir.mkdir(parents=True, exist_ok=True)
            tmp = vdir / f".manifest.{os.getpid()}-{uuid.uuid4().hex[:8]}.part"
            tmp.write_text(manifest.to_json(), encoding="utf-8")
            store_io.place(tmp, manifest_path, tmp.stat().st_size)
        (vdir / "files").mkdir(parents=True, exist_ok=True)
        touch_used(vdir)
        if not self.settings.offline:
            self.sweep_daily()
        return CachedPack(self, pack, vdir, manifest)

    def _prepare_top(self) -> None:
        self.top.mkdir(parents=True, exist_ok=True)
        tag = self.top / "CACHEDIR.TAG"
        if not tag.exists():
            tag.write_text(CACHEDIR_TAG, encoding="utf-8")

    # ---- keeping it small ----

    def versions(self) -> list[CachedVersion]:
        out = []
        roots = [self.root / "packs", *sorted((self.root / "stores").glob("*/packs"))]
        for root in roots:
            for vdir in sorted(root.glob("*/*")):
                if not vdir.is_dir():
                    continue
                label = f"{vdir.parent.name}/{vdir.name}"
                if root != self.root / "packs":
                    label += f" ({root.parent.name})"
                out.append(
                    CachedVersion(vdir, label, _tree_bytes(vdir), last_used(vdir))
                )
        return out

    def sweep_daily(self, now: float | None = None) -> list[CachedVersion]:
        """Sweep if this process is the first today to claim the stamp."""
        today = date.fromtimestamp(time.time() if now is None else now).isoformat()
        stamp = self.root / f".sweep-{today}"
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(stamp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return []
        os.close(fd)
        for old in self.root.glob(".sweep-*"):
            if old != stamp:
                old.unlink(missing_ok=True)
        return self.sweep(now)

    def sweep(self, now: float | None = None) -> list[CachedVersion]:
        now = time.time() if now is None else now
        self._remove_leftovers(now)
        versions = self.versions()
        doomed = select_removals(
            versions,
            now=now,
            max_bytes=int(self.settings.cache_max_gb * 2**30),
            max_age_s=self.settings.cache_max_age_days * DAY_S,
        )
        removed = [v for v in doomed if self._remove(v.folder)]
        if removed:
            log.warning(
                "pack cache: removed %s (%.2f GiB freed)",
                ", ".join(v.label for v in removed),
                sum(v.size for v in removed) / 2**30,
            )
        kept = sum(v.size for v in versions) - sum(v.size for v in removed)
        if kept > self.settings.cache_max_gb * 2**30:
            log.warning(
                "pack cache holds %.2f GiB, over its %.0f GiB limit, all of it "
                "used in the last day",
                kept / 2**30,
                self.settings.cache_max_gb,
            )
        return removed

    def clear(self, which: str | None = None) -> list[CachedVersion]:
        """Remove every version, every version of one pack, or one version."""
        removed = []
        for v in self.versions():
            name_ver = v.label.split(" ", 1)[0]
            if which is None or which in (name_ver, name_ver.split("/")[0]):
                if self._remove(v.folder):
                    removed.append(v)
        self._remove_leftovers(time.time())
        return removed

    def _remove(self, folder: Path) -> bool:
        """Rename into ``trash/`` (atomic; fails on Windows while a file inside
        is open, which marks the version in use), then delete."""
        trash = self.root / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        dest = trash / uuid.uuid4().hex
        try:
            os.replace(folder, dest)
        except OSError as e:
            log.info("pack cache: %s is in use, kept (%s)", folder, e)
            return False
        shutil.rmtree(dest, ignore_errors=True)
        return True

    def _remove_leftovers(self, now: float) -> None:
        # Another layout's root is removed only from a folder rutter tagged,
        # so a RUTTER_CACHE pointed at a data folder never loses a "v2".
        if (self.top / "CACHEDIR.TAG").is_file():
            for old in self.top.glob("v*"):
                if old.is_dir() and old.name != LAYOUT and old.name[1:].isdigit():
                    shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(self.root / "trash", ignore_errors=True)
        for part in self.root.rglob("*.part"):
            try:
                if now - part.stat().st_mtime > DAY_S:
                    part.unlink()
            except OSError:
                pass


@dataclass
class CachedPack:
    """One pack version in the cache: ``folder`` is what a config's
    ``${paths.<name>}`` resolves to."""

    cache: PackCache
    pack: PackRef
    vdir: Path
    manifest: Manifest

    @property
    def folder(self) -> Path:
        return self.vdir / "files"

    def relative(self, path: str | Path) -> str | None:
        """``path`` as a POSIX path inside the pack, or None if outside it."""
        try:
            rel = Path(os.path.abspath(path)).relative_to(self.folder)
        except ValueError:
            return None
        return rel.as_posix() if rel.parts else ""

    def ensure(self, paths: Iterable[str]) -> list[str]:
        """Fetch each listed file that is absent; return those fetched."""
        missing = [p for p in dict.fromkeys(paths) if not (self.folder / p).is_file()]
        if missing and self.cache.settings.offline:
            raise FileNotFoundError(
                f"{self.pack}: {len(missing)} file(s) not in the cache, e.g. "
                f"{missing[0]}, and RUTTER_OFFLINE is set; run `rutter-data pull "
                f"{self.pack}` while online"
            )
        for i, p in enumerate(missing, 1):
            entry = self.manifest.files[p]
            log.info(
                "fetching %s %s (%d of %d, %.1f MB)",
                self.pack,
                p,
                i,
                len(missing),
                entry.size / 1e6,
            )
            store_io.download(
                store_io.file_url(self.cache.store, self.pack, f"files/{p}"),
                self.folder / p,
                entry,
                check_sha=self.cache.settings.verify_downloads,
            )
        return missing

    def ensure_referenced(self, values: Iterable[str | Path], *, folders: bool) -> None:
        """Fetch what each value names inside the pack: a file with its
        companions, and, when ``folders``, every file of a named folder.
        Without ``folders`` a named folder is only created, so a check that it
        exists passes before the config is expanded."""
        want: list[str] = []
        for value in values:
            rel = self.relative(value)
            if rel is None:
                continue
            if rel in self.manifest.files:
                want += [rel, *self.manifest.companions(rel)]
            elif self.manifest.is_folder(rel):
                if folders:
                    want += self.manifest.under(rel)
                else:
                    (self.folder / rel).mkdir(parents=True, exist_ok=True)
        self.ensure(want)

    def ensure_all(self) -> list[str]:
        return self.ensure(self.manifest.files)


def select_removals(
    versions: Sequence[CachedVersion],
    *,
    now: float,
    max_bytes: int,
    max_age_s: float,
    grace_s: float = DAY_S,
) -> list[CachedVersion]:
    """Versions unused for ``max_age_s``, then least recently used first while
    the rest exceed ``max_bytes``. Nothing used within ``grace_s`` is chosen,
    so a running job keeps its inputs and the size limit is soft."""
    doomed = [v for v in versions if now - v.last_used > max(max_age_s, grace_s)]
    left = sorted(
        (v for v in versions if v not in doomed), key=lambda v: (v.last_used, v.label)
    )
    total = sum(v.size for v in left)
    for v in left:
        if total <= max_bytes:
            break
        if now - v.last_used <= grace_s:
            break
        doomed.append(v)
        total -= v.size
    return doomed


def touch_used(vdir: Path) -> None:
    marker = vdir / USED
    try:
        if time.time() - marker.stat().st_mtime < TOUCH_EVERY_S:
            return
    except FileNotFoundError:
        marker.touch()
        return
    os.utime(marker)


def last_used(vdir: Path) -> float:
    for name in (USED, "manifest.json"):
        try:
            return (vdir / name).stat().st_mtime
        except FileNotFoundError:
            continue
    return vdir.stat().st_mtime


def _tree_bytes(folder: Path) -> int:
    total = 0
    for dirpath, _, names in os.walk(folder):
        for name in names:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total
