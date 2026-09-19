"""Dependency-direction rules, each with a baseline of today's exceptions.

A baseline may only shrink: every rule asserts both that no new violation appears
and that each listed exception is still a real violation, so fixing one forces
its line out of the list instead of leaving a rule that quietly passes.
"""

from __future__ import annotations

import pytest

from tests.architecture.graph import PACKAGE, Graph, module_graph

# `clearance.voxel_sdf` imports `samples` inside a function, and `samples`
# imports voxel_sdf's cache helpers at import time.
CYCLES_BASELINE = {
    (
        "aind_rutter.optimization.clearance.samples",
        "aind_rutter.optimization.clearance.voxel_sdf",
    ),
}

# The solver must not depend on the pipeline that drives it. Every exception
# reaches for the stage record types, which the target layout moves under
# `optimization/problem`.
SOLVER_TO_PIPELINE_BASELINE = {
    (
        "aind_rutter.optimization.objectives.constrained",
        "aind_rutter.optimization.pipeline.records",
    ),
    (
        "aind_rutter.optimization.search.minimizers",
        "aind_rutter.optimization.pipeline.records",
    ),
    (
        "aind_rutter.optimization.search.spin_restore",
        "aind_rutter.optimization.pipeline.records",
    ),
}

# Modules reaching into another module's private names.
PRIVATE_IMPORT_BASELINE = {
    ("aind_rutter.domain.commands", "aind_rutter.domain.pose"),
    (
        "aind_rutter.optimization.objectives.soft",
        "aind_rutter.optimization.objectives.threading",
    ),
    (
        "aind_rutter.optimization.objectives.constrained",
        "aind_rutter.optimization.objectives.threading",
    ),
    (
        "aind_rutter.optimization.search.spin_restore",
        "aind_rutter.optimization.objectives.reduced",
    ),
    (
        "aind_rutter.optimization.pipeline.emit",
        "aind_rutter.optimization.objectives.variables",
    ),
    (
        "aind_rutter.optimization.pipeline.candidates",
        "aind_rutter.optimization.assignment.seed_emission",
    ),
    (
        "aind_rutter.optimization.search.minimizers",
        "aind_rutter.optimization.objectives.soft",
    ),
    (
        "aind_rutter.optimization.pipeline.fixtures",
        "aind_rutter.optimization.clearance.voxel_sdf",
    ),
    (
        "aind_rutter.optimization.pipeline.phase1",
        "aind_rutter.optimization.objectives.statics",
    ),
    (
        "aind_rutter.optimization.pipeline.phase2",
        "aind_rutter.optimization.objectives.statics",
    ),
    (
        "aind_rutter.optimization.pipeline.phase2",
        "aind_rutter.optimization.objectives.variables",
    ),
    (
        "aind_rutter.optimization.pipeline.subject",
        "aind_rutter.optimization.pipeline.probe_setup",
    ),
    (
        "aind_rutter.optimization.clearance.samples",
        "aind_rutter.optimization.clearance.voxel_sdf",
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


SOLVER_SUBPACKAGES = (
    "assignment",
    "clearance",
    "geometry",
    "objectives",
    "search",
    "validation",
)


def test_the_solver_does_not_import_the_pipeline(graph: Graph) -> None:
    """The pipeline drives the solver, so the edge only runs one way."""
    solver = tuple(f"{PACKAGE}.optimization.{s}" for s in SOLVER_SUBPACKAGES)
    violations = set()
    for imp in graph.imports:
        target = graph.owner(imp.target)
        if target is None:
            continue
        if imp.importer.startswith(solver) and target.startswith(
            f"{PACKAGE}.optimization.pipeline"
        ):
            violations.add((imp.importer, target))
    _assert_baseline(violations, SOLVER_TO_PIPELINE_BASELINE, "solver → pipeline")


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
