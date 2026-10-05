"""What a run was generated from: the spec, the code and every input's hash."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from aind_rutter.config.locations import Locator, load_resolved

if TYPE_CHECKING:
    from aind_rutter.series.spec import SeriesSpec

LOCK_NAME = "series.lock.yml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def referenced_files(config: Path, locator: Locator | None = None) -> set[Path]:
    """Every file a config names, after interpolation: meshes, volumes, transforms."""
    found: set[Path] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and "/" in node and Path(node).is_file():
            found.add(Path(node).resolve())

    walk(load_resolved(config, locator))
    return found


def checkout_commit() -> str | None:
    """The package checkout's commit, marked when tracked files differ from it."""
    here = Path(__file__).parent

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(here), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return f"{commit}-dirty" if dirty else commit


def write_lock(
    spec: SeriesSpec,
    run_dir: Path,
    configs: Sequence[Path],
    *,
    spec_path: str | Path | None = None,
) -> Path:
    """Record the resolved spec, the code version and the hash of every input."""
    inputs: set[Path] = set()
    if spec_path is not None:
        inputs.add(Path(spec_path).resolve())
    for subject in spec.subjects.values():
        inputs |= {subject.base_config.resolve(), subject.holes.resolve()}
    for config in configs:
        inputs |= referenced_files(config)
    lock = {
        "series": spec.name,
        "spec_file": None if spec_path is None else str(Path(spec_path).resolve()),
        "aind_rutter": {"version": version("aind-rutter"), "commit": checkout_commit()},
        "spec": spec.model_dump(mode="json"),
        "configs": {str(c.relative_to(run_dir)): file_sha256(c) for c in configs},
        # A file that is not there yet has no hash.
        "inputs": {
            str(p): file_sha256(p) if p.is_file() else None for p in sorted(inputs)
        },
    }
    path = run_dir / LOCK_NAME
    path.write_text(yaml.safe_dump(lock, sort_keys=False, width=100))
    return path
