from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    path = ROOT / "src/aind_rutter/optimization/pipeline/phase1_pool.py"
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
    stages = ("phase1_pool", "phase1_build", "restore", "phase2_ipopt", "emit")
    paths = [ROOT / f"src/aind_rutter/optimization/pipeline/{s}.py" for s in stages]
    banned_names = {
        "_probe_static_info",
        "_transform_holes",
        "build_fixture_sdf_data",
    }
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
