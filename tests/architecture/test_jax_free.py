"""Modules that must import without loading jax.

A stage's inputs and outputs have to be readable on a machine with no GPU
backend, and the settings have to resolve before anything chooses a platform.
Each name below is imported in a clean interpreter, which is the only way to
tell: once one test has loaded jax, every later in-process import finds it
already in ``sys.modules``.
"""

from __future__ import annotations

import pytest

from tests.architecture.graph import import_in_subprocess, module_graph

JAX_FREE_MODULES = [
    "aind_rutter.config",
    "aind_rutter.domain.commands",
    "aind_rutter.domain.plan",
    "aind_rutter.domain.pose",
    "aind_rutter.domain.probe_kinds",
    "aind_rutter.domain.transforms",
    "aind_rutter.optimization.pipeline.records",
    "aind_rutter.optimization.pipeline.handoff",
    "aind_rutter.optimization.pipeline.payloads",
    "aind_rutter.optimization.pipeline.phase2_diagnostics",
    "aind_rutter.optimization.pipeline.selection",
    "aind_rutter.optimization.pipeline.settings",
]


def test_the_listed_modules_exist() -> None:
    missing = set(JAX_FREE_MODULES) - set(module_graph().modules)
    assert not missing, f"unknown module(s) in JAX_FREE_MODULES: {sorted(missing)}"


@pytest.mark.parametrize("module", JAX_FREE_MODULES)
def test_imports_without_jax(module: str) -> None:
    loaded = import_in_subprocess(module).split()
    assert "jax" not in loaded, f"{module} loads jax at import"
