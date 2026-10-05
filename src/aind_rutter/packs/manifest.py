"""A pack's name and version, and the manifest listing every file it holds."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MANIFEST_VERSION = 1
# Lower case only, so no two packs differ by case alone on macOS or Windows.
_SEGMENT = r"[a-z0-9][a-z0-9._-]*"
_REF = re.compile(rf"({_SEGMENT})/({_SEGMENT})")

# Left behind by file managers; never part of a pack.
JUNK_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
JUNK_DIRS = frozenset({"__MACOSX", ".git"})

_WINDOWS_RESERVED = re.compile(r"(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", re.I)
_WINDOWS_INVALID = set('<>:"|?*') | {chr(c) for c in range(32)}
LONG_PATH = 150


@dataclass(frozen=True, order=True)
class PackRef:
    """``name/version``, e.g. ``hardware/v1``. A published version never changes."""

    name: str
    version: str

    @classmethod
    def parse(cls, text: str) -> PackRef:
        m = _REF.fullmatch(text.strip())
        if m is None:
            raise ValueError(
                f"{text!r} is not a pack: expected name/version in lower case, "
                "e.g. hardware/v1"
            )
        return cls(m.group(1), m.group(2))

    def __str__(self) -> str:
        return f"{self.name}/{self.version}"


@dataclass(frozen=True)
class FileEntry:
    size: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    """Every file of one pack version by its POSIX path inside the pack."""

    pack: PackRef
    files: Mapping[str, FileEntry]

    def to_json(self) -> str:
        return json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "pack": str(self.pack),
                "files": {
                    p: {"size": e.size, "sha256": e.sha256}
                    for p, e in sorted(self.files.items())
                },
            },
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str, expected: PackRef) -> Manifest:
        data = json.loads(text)
        if data.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(
                f"{expected}: manifest version {data.get('manifest_version')!r} "
                f"is not {MANIFEST_VERSION}; a newer rutter wrote it"
            )
        if data.get("pack") != str(expected):
            raise ValueError(f"{expected}: the manifest is for {data.get('pack')!r}")
        files = {}
        for path, entry in data["files"].items():
            check_pack_path(path)
            files[path] = FileEntry(int(entry["size"]), str(entry["sha256"]))
        return cls(expected, files)

    def total_bytes(self) -> int:
        return sum(e.size for e in self.files.values())

    def under(self, prefix: str) -> list[str]:
        """Files inside folder ``prefix`` (``""`` for the whole pack)."""
        if not prefix:
            return sorted(self.files)
        lead = prefix.rstrip("/") + "/"
        return sorted(p for p in self.files if p.startswith(lead))

    def is_folder(self, path: str) -> bool:
        return bool(self.under(path)) if path else True

    def companions(self, path: str) -> list[str]:
        """Files beside ``path`` sharing its stem, which a loader may open with
        it: ``x.obj``'s ``x.mtl``, a ``.nhdr``'s ``.raw.gz``."""
        p = PurePosixPath(path)
        stem = p.name.split(".", 1)[0]
        return sorted(
            q
            for q in self.files
            if q != path
            and PurePosixPath(q).parent == p.parent
            and PurePosixPath(q).name.split(".", 1)[0] == stem
        )


def check_pack_path(path: str) -> None:
    """Refuse a manifest path that could write outside the pack's folder."""
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or re.match(r"[A-Za-z]:", path)
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"manifest path {path!r} is not a relative POSIX path")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def pack_files(folder: Path) -> Iterator[tuple[str, Path]]:
    """Every file under ``folder`` by POSIX path, junk left by file managers
    skipped."""
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder)
        if any(part in JUNK_DIRS for part in rel.parts) or rel.name in JUNK_NAMES:
            continue
        if path.is_file():
            yield rel.as_posix(), path


def portability_problems(paths: list[str]) -> tuple[list[str], list[str]]:
    """Errors that stop a pack being read on macOS or Windows, and warnings
    about paths long enough to break Windows' 260-character limit."""
    errors, warnings = [], []
    by_folded: dict[str, str] = {}
    for path in paths:
        folded = path.casefold()
        if folded in by_folded:
            errors.append(f"{by_folded[folded]} and {path} differ only in case")
        by_folded[folded] = path
        for part in path.split("/"):
            bad = sorted(set(part) & _WINDOWS_INVALID)
            if bad:
                errors.append(f"{path}: Windows cannot store {''.join(bad)!r}")
            if part.endswith((".", " ")):
                errors.append(f"{path}: Windows drops a trailing dot or space")
            if _WINDOWS_RESERVED.fullmatch(part):
                errors.append(f"{path}: {part!r} is a reserved name on Windows")
        if len(path) > LONG_PATH:
            warnings.append(f"{path}: {len(path)} characters long")
    return errors, warnings


def build_manifest(folder: Path, pack: PackRef) -> Manifest:
    files = {
        rel: FileEntry(path.stat().st_size, file_sha256(path))
        for rel, path in pack_files(folder)
    }
    if not files:
        raise ValueError(f"{folder}: no files to publish")
    return Manifest(pack, files)
