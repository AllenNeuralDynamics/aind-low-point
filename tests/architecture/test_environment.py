"""Where the package reads the environment.

The target layout confines environment access to the settings models, a single
`jax_env` module and the CLI shims, and forbids reading it at import below the
CLI. Both properties are far off today, so the rule is a baseline that may only
shrink: each entry is `module:NAME@import` or `module:NAME@function`, and moving
a variable into settings (or from import time into a function) deletes its line.

The rule reads the AST, so a value a settings model resolves is invisible to it.
An entry leaving this list therefore means the variable is parsed in one place
under one set of rules, not that nothing reads the environment for it — which is
the property worth holding, since the phases once parsed the same names
differently.
"""

from __future__ import annotations

from tests.architecture.graph import module_graph

ENV_READ_BASELINE = {
    "aind_rutter.optimization.geometry.holes:RUTTER_THREADING_MARGIN_MM@function",
    "aind_rutter.optimization.jax_env:AIND_JAX_CACHE_DIR@function",
    "aind_rutter.optimization.jax_env:JAX_CACHE_DIR@function",
    "aind_rutter.optimization.pipeline.enumeration:JAX_PLATFORMS@import",
    "aind_rutter.optimization.pipeline.enumeration:XLA_PYTHON_CLIENT_PREALLOCATE@import",
    "aind_rutter.optimization.pipeline.phase1_build:JAX_PLATFORMS@import",
    "aind_rutter.optimization.pipeline.phase1_build:XLA_PYTHON_CLIENT_PREALLOCATE@import",
    "aind_rutter.optimization.pipeline.phase1_geometry:JAX_PLATFORMS@import",
    "aind_rutter.optimization.pipeline.phase1_geometry:XLA_PYTHON_CLIENT_PREALLOCATE@import",
    "aind_rutter.optimization.pipeline.phase1_pool:JAX_PLATFORMS@import",
    "aind_rutter.optimization.pipeline.phase1_pool:XLA_PYTHON_CLIENT_PREALLOCATE@import",
    "aind_rutter.optimization.pipeline.phase2_ipopt:<dynamic>@import",
    "aind_rutter.optimization.pipeline.phase2_ipopt:RUTTER_GPU_MEM_FRACTION@import",
    "aind_rutter.optimization.pipeline.phase2_ipopt:JAX_PLATFORMS@import",
    "aind_rutter.optimization.pipeline.phase2_ipopt:RUTTER_P2_HEADROOM_GB@function",
    "aind_rutter.optimization.pipeline.phase2_ipopt:RUTTER_P2_PER_WORKER_GB@function",
    "aind_rutter.optimization.pipeline.phase2_ipopt:RUTTER_PLATFORM@import",
    "aind_rutter.optimization.pipeline.phase2_ipopt:RUTTER_POOL@import",
    "aind_rutter.optimization.pipeline.phase2_ipopt:RUTTER_THREADS@import",
    "aind_rutter.optimization.pipeline.phase2_ipopt:XLA_PYTHON_CLIENT_MEM_FRACTION@import",
    "aind_rutter.optimization.pipeline.phase2_ipopt:XLA_PYTHON_CLIENT_PREALLOCATE@import",
    "aind_rutter.optimization.pipeline.probe_setup:RUTTER_RETRO_DENSITY@function",
    "aind_rutter.optimization.pipeline.restore:JAX_PLATFORMS@import",
    "aind_rutter.optimization.pipeline.restore:XLA_PYTHON_CLIENT_PREALLOCATE@import",
    "aind_rutter.optimization.pipeline.thick_well:JAX_PLATFORMS@import",
    "aind_rutter.optimization.pipeline.thick_well:XLA_PYTHON_CLIENT_PREALLOCATE@import",
    "aind_rutter.optimization.sdf.build:AIND_LOW_POINT_CACHE_DIR@function",
    "aind_rutter.optimization.sdf.envelope:AIND_LOW_POINT_CACHE_DIR@function",
}


def _env_reads() -> set[str]:
    return {
        f"{e.module}:{e.name}@{'import' if e.at_import_time else 'function'}"
        for e in module_graph().env_reads
    }


def test_no_new_environment_reads() -> None:
    new = _env_reads() - ENV_READ_BASELINE
    assert not new, (
        "new environment read(s); settings own configuration, not module "
        f"globals: {sorted(new)}"
    )


def test_environment_baseline_has_no_stale_entries() -> None:
    gone = ENV_READ_BASELINE - _env_reads()
    assert not gone, (
        f"{sorted(gone)} no longer read the environment — delete them from "
        f"ENV_READ_BASELINE in {__file__}"
    )
