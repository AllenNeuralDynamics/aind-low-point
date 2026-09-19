"""Dependency-direction rules, each with a baseline of today's exceptions.

A baseline may only shrink: every rule asserts both that no new violation appears
and that each listed exception is still a real violation, so fixing one forces
its line out of the list instead of leaving a rule that quietly passes.
"""

from __future__ import annotations

import pytest

from tests.architecture.graph import PACKAGE, Graph, module_graph

# aind_rutter.optimization.sdf.build imports surface_samples inside a function and
# surface_samples imports build's cache helpers at import time.
CYCLES_BASELINE = {
    (
        "aind_rutter.optimization.sdf.build",
        "aind_rutter.optimization.sdf.surface_samples",
    ),
}

# Objectives must not depend on the pipeline that drives them. Both exceptions
# reach for the stage record types, which the target layout moves under
# `optimization/problem`.
OBJECTIVES_TO_PIPELINE_BASELINE = {
    (
        "aind_rutter.optimization.objectives.phase2",
        "aind_rutter.optimization.pipeline.contracts",
    ),
    (
        "aind_rutter.optimization.objectives.spin_restore",
        "aind_rutter.optimization.pipeline.contracts",
    ),
}

# Modules reaching into another module's private names. The `runtime` and
# `build_runtime` entries are re-export shims that the target layout deletes.
PRIVATE_IMPORT_BASELINE = {
    ("aind_rutter.domain.commands", "aind_rutter.domain.pose"),
    (
        "aind_rutter.optimization.objectives.phase1",
        "aind_rutter.optimization.objectives.reduced_jax",
    ),
    (
        "aind_rutter.optimization.objectives.phase2",
        "aind_rutter.optimization.objectives.phase1",
    ),
    (
        "aind_rutter.optimization.objectives.phase2",
        "aind_rutter.optimization.objectives.reduced_jax",
    ),
    (
        "aind_rutter.optimization.objectives.spin_restore",
        "aind_rutter.optimization.objectives.batched_reduced",
    ),
    (
        "aind_rutter.optimization.pipeline.emit",
        "aind_rutter.optimization.objectives.variables",
    ),
    (
        "aind_rutter.optimization.pipeline.enumeration",
        "aind_rutter.optimization.enumeration.seed_emission",
    ),
    (
        "aind_rutter.optimization.pipeline.phase1_build",
        "aind_rutter.optimization.objectives.phase1",
    ),
    (
        "aind_rutter.optimization.pipeline.phase1_geometry",
        "aind_rutter.optimization.sdf.build",
    ),
    (
        "aind_rutter.optimization.pipeline.phase1_pool",
        "aind_rutter.optimization.objectives.probe_static",
    ),
    (
        "aind_rutter.optimization.pipeline.phase2_ipopt",
        "aind_rutter.optimization.objectives.probe_static",
    ),
    (
        "aind_rutter.optimization.pipeline.phase2_ipopt",
        "aind_rutter.optimization.objectives.variables",
    ),
    (
        "aind_rutter.optimization.pipeline.runtime_adapter",
        "aind_rutter.optimization.pipeline.probe_setup",
    ),
    (
        "aind_rutter.optimization.sdf.surface_samples",
        "aind_rutter.optimization.sdf.build",
    ),
    ("aind_rutter.build.assemble", "aind_rutter.build.calibration"),
    ("aind_rutter.build.assemble", "aind_rutter.build.canonicalize"),
    ("aind_rutter.build.assemble", "aind_rutter.build.chem_shift"),
    ("aind_rutter.build.assemble", "aind_rutter.build.reducers"),
}

# Modules whose job is drawing or driving a UI. The optimizer never imports one.
FRONTEND_MODULES = {
    "aind_rutter.web.app",
    "aind_rutter.web.cli",
    "aind_rutter.web.ccf_regions",
    "aind_rutter.render.pyvista",
    "aind_rutter.render.adapter",
    "aind_rutter.render.overlays",
    "aind_rutter.web.controller",
}


@pytest.fixture(scope="session")
def graph() -> Graph:
    return module_graph()


def _assert_baseline(violations: set, baseline: set, rule: str) -> None:
    """Fail on a new violation, and on a stale baseline entry."""
    new = violations - baseline
    assert not new, f"{rule}: new violation(s) {sorted(new)}"
    fixed = baseline - violations
    assert not fixed, (
        f"{rule}: {sorted(fixed)} no longer violate the rule — "
        f"delete them from the baseline in {__file__}"
    )


def test_every_module_parses(graph: Graph) -> None:
    assert len(graph.modules) > 50
    assert f"{PACKAGE}.config" in graph.modules


def test_no_import_cycles(graph: Graph) -> None:
    cycles = {tuple(c) for c in graph.cycles()}
    _assert_baseline(cycles, CYCLES_BASELINE, "import cycle")


def test_objectives_do_not_import_the_pipeline(graph: Graph) -> None:
    violations = set()
    for imp in graph.imports:
        target = graph.owner(imp.target)
        if target is None:
            continue
        if imp.importer.startswith(
            f"{PACKAGE}.optimization.objectives"
        ) and target.startswith(f"{PACKAGE}.optimization.pipeline"):
            violations.add((imp.importer, target))
    _assert_baseline(
        violations, OBJECTIVES_TO_PIPELINE_BASELINE, "objectives → pipeline"
    )


def test_optimization_does_not_import_a_frontend(graph: Graph) -> None:
    violations = set()
    for imp in graph.imports:
        target = graph.owner(imp.target)
        if (
            imp.importer.startswith(f"{PACKAGE}.optimization")
            and target in FRONTEND_MODULES
        ):
            violations.add((imp.importer, target))
    assert not violations, f"optimization → frontend: {sorted(violations)}"


def test_no_cross_module_private_imports(graph: Graph) -> None:
    violations = set()
    for imp in graph.imports:
        target = graph.owner(imp.target)
        if target is None:
            continue
        for name in imp.names:
            if name.startswith("_") and not name.startswith("__"):
                violations.add((imp.importer, target))
    _assert_baseline(violations, PRIVATE_IMPORT_BASELINE, "private-name import")


def test_frontend_modules_exist(graph: Graph) -> None:
    """The frontend rule is vacuous if its module names go stale."""
    missing = FRONTEND_MODULES - set(graph.modules)
    assert not missing, f"unknown module(s) in FRONTEND_MODULES: {sorted(missing)}"
