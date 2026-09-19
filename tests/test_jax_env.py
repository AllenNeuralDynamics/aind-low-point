"""The compile-cache directory must not depend on which module imported first.

Two modules used to configure it at import with different directories, so a run
landed wherever the later import put it — which sent Phase 2 somewhere other than
the directory it had just asked for, and changed which cached kernels loaded.

The cache is process-global state, so the checks that touch jax run in a clean
interpreter; the resolution rules are pure and run in process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aind_rutter.optimization.jax_env import (
    CACHE_DIR_ENV,
    DEFAULT_CACHE_DIR,
    compile_cache_dir,
)

# Importing this pulls in the whole objectives chain that used to reconfigure
# the cache on the way past.
OBJECTIVES_MODULE = "aind_rutter.optimization.objectives.threading"


def _run(code: str) -> str:
    """Run `code` in a clean interpreter on CPU; return its stdout."""
    import os

    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        # Neither name may be set, or it would decide the answer.
        **dict.fromkeys(CACHE_DIR_ENV, ""),
    }
    for name in CACHE_DIR_ENV:
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_argument_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CACHE_DIR_ENV[0], str(tmp_path / "from_env"))
    assert compile_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_the_environment_names_are_tried_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first, second = CACHE_DIR_ENV
    monkeypatch.setenv(second, str(tmp_path / "second"))
    assert compile_cache_dir() == tmp_path / "second"
    monkeypatch.setenv(first, str(tmp_path / "first"))
    assert compile_cache_dir() == tmp_path / "first"


def test_an_empty_variable_does_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CACHE_DIR_ENV:
        monkeypatch.setenv(name, "")
    assert compile_cache_dir() == DEFAULT_CACHE_DIR


def test_importing_the_objectives_leaves_the_cache_alone() -> None:
    """The defect itself: an import used to overwrite the configured directory."""
    before_after = _run(
        "import jax;"
        "before = jax.config.jax_compilation_cache_dir;"
        f"import {OBJECTIVES_MODULE};"
        "print(before);"
        "print(jax.config.jax_compilation_cache_dir)"
    ).splitlines()
    assert before_after[0] == before_after[1]


def test_a_configured_directory_survives_a_later_import(tmp_path: Path) -> None:
    """Configure, then import the objectives: the directory must still hold."""
    target = tmp_path / "pinned"
    result = _run(
        "from aind_rutter.optimization.jax_env import configure_compile_cache;"
        f"configure_compile_cache({str(target)!r});"
        f"import {OBJECTIVES_MODULE};"
        "import jax;"
        "print(jax.config.jax_compilation_cache_dir)"
    )
    assert result == str(target)
    assert target.is_dir()


def test_configuring_reports_where_it_went(tmp_path: Path) -> None:
    target = tmp_path / "reported"
    result = _run(
        "from aind_rutter.optimization.jax_env import configure_compile_cache;"
        f"print(configure_compile_cache({str(target)!r}))"
    )
    assert result == str(target)


def test_an_unusable_directory_disables_the_cache_without_raising(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "a_file"
    blocker.write_text("not a directory")
    result = _run(
        "from aind_rutter.optimization.jax_env import configure_compile_cache;"
        f"print(configure_compile_cache({str(blocker / 'under_a_file')!r}))"
    )
    assert result == "None"
