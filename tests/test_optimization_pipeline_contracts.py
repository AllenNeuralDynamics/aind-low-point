"""Rules about how the pipeline stages call into the rest of the optimizer.

Each rule names identifiers it forbids or requires. Those names are asserted to
exist first: a rule that silently stops matching after a rename would pass while
checking nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.architecture.graph import module_graph

ROOT = Path(__file__).resolve().parents[1]

# Legacy setup helpers the stages must reach through the runtime adapter instead.
BANNED_SETUP_HELPERS = {
    "_probe_static_info": "aind_rutter.optimization.pipeline.probe_setup",
    "_transform_holes": "aind_rutter.optimization.pipeline.probe_setup",
    "build_fixture_sdf_data": "aind_rutter.optimization.pipeline.fixtures",
}

REQUIRED_NAMES = {
    "Enumerator": "aind_rutter.optimization.pipeline.candidates",
    "build_or_load_atlas": "aind_rutter.optimization.pipeline.candidates",
    "make_phase1_pool_record": "aind_rutter.optimization.pipeline.phase1",
}


@pytest.mark.parametrize(
    ("name", "module"), sorted({**BANNED_SETUP_HELPERS, **REQUIRED_NAMES}.items())
)
def test_the_name_a_rule_matches_on_still_exists(name: str, module: str) -> None:
    facts = module_graph().modules.get(module)
    assert facts is not None, f"{module} is gone; the rule naming {name} is vacuous"
    assert name in facts.definitions, f"{module} no longer defines {name}"


def test_build_or_load_atlas_is_not_splatted_into_enumerator() -> None:
    offenders: list[tuple[Path, int]] = []
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py")]:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "Enumerator":
                continue
            for arg in node.args:
                if not isinstance(arg, ast.Starred):
                    continue
                value = arg.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "build_or_load_atlas"
                ):
                    offenders.append((path.relative_to(ROOT), node.lineno))

    assert offenders == []


def test_phase1_pool_records_are_built_through_helper() -> None:
    path = ROOT / "src/aind_rutter/optimization/pipeline/phase1.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_phase1_pool_record"
    ]

    assert helper_calls


def test_active_pipeline_entrypoints_use_runtime_adapter_for_setup() -> None:
    # Pipeline stages must build setup via the runtime adapter, not the banned
    # legacy helpers.
    stages = ("phase1", "phase2", "emit")
    paths = [ROOT / f"src/aind_rutter/optimization/pipeline/{s}.py" for s in stages]
    assert all(path.exists() for path in paths), "a named stage module has moved"
    banned_names = set(BANNED_SETUP_HELPERS)
    offenders: list[tuple[Path, int, str]] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in banned_names:
                        offenders.append(
                            (path.relative_to(ROOT), node.lineno, alias.name)
                        )
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in banned_names:
                    offenders.append((path.relative_to(ROOT), node.lineno, func.id))

    assert offenders == []
