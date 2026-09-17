"""The five installed commands import and, where they parse arguments, take --help.

Nothing else covers the stage drivers, so a rename that breaks one is otherwise
invisible until a run fails hours later. Each check runs in a clean interpreter:
importing a stage driver sets JAX environment variables and would otherwise
change the platform of every later test in the process.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.architecture.graph import ROOT, console_scripts, module_graph

# Scripts with an argument parser, so `--help` is meaningful. The stage drivers
# take their configuration from the environment and parse nothing.
PARSED_ARGUMENTS = {"rutter-plan", "rutter-plan-csv"}

# Importing these must not choose a GPU: the default platform is cuda.
CPU_ENV = {"JAX_PLATFORMS": "cpu", "PLATFORM": "cpu"}


def _targets() -> list[tuple[str, str, str]]:
    out = []
    for script, target in sorted(console_scripts().items()):
        module, _, attr = target.partition(":")
        out.append((script, module, attr))
    return out


def _run(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    import os

    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **CPU_ENV},
        cwd=ROOT,
        timeout=600,
    )


def test_pyproject_declares_the_expected_commands() -> None:
    assert set(console_scripts()) == {
        "rutter-plan",
        "rutter-plan-csv",
        "rutter-phase1",
        "rutter-phase2",
        "rutter-emit",
    }


@pytest.mark.parametrize(("script", "module", "attr"), _targets())
def test_entry_point_target_is_defined(script: str, module: str, attr: str) -> None:
    """Catch a moved module or a renamed `main` without paying for an import."""
    facts = module_graph().modules.get(module)
    assert facts is not None, f"{script} points at missing module {module}"
    assert attr in facts.definitions, f"{script}: {module} defines no {attr}"


@pytest.mark.parametrize(("script", "module", "attr"), _targets())
def test_entry_point_imports(script: str, module: str, attr: str) -> None:
    result = _run(
        "import importlib, sys;"
        f"m = importlib.import_module({module!r});"
        f"sys.exit(0 if callable(getattr(m, {attr!r})) else 1)"
    )
    assert result.returncode == 0, f"{script} failed to import:\n{result.stderr}"


@pytest.mark.parametrize(
    "script", sorted(s for s in console_scripts() if s in PARSED_ARGUMENTS)
)
def test_help_exits_zero(script: str) -> None:
    module, _, attr = console_scripts()[script].partition(":")
    result = _run(f"from {module} import {attr}; {attr}()", "--help")
    assert result.returncode == 0, f"{script} --help failed:\n{result.stderr}"
    assert result.stdout.strip(), f"{script} --help printed nothing"
