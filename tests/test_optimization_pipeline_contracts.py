from __future__ import annotations

import ast
from pathlib import Path

from aind_low_point.optimization.atlas import Atlas
from aind_low_point.optimization.pipeline.contracts import AtlasCachePayload
from aind_low_point.optimization.pipeline.enumeration import _normalize_atlas_payload

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_atlas_tuple_normalizes_to_payload() -> None:
    atlas = Atlas(entries={}, probe_names=("probe-a",), hole_ids=(1,))

    payload = _normalize_atlas_payload((atlas, ["probe-a"], 12.5))

    assert payload == AtlasCachePayload(
        atlas=atlas,
        probe_names=("probe-a",),
        head_pitch_deg=12.5,
    )


def test_legacy_two_tuple_atlas_payload_is_rejected() -> None:
    atlas = Atlas(entries={}, probe_names=("probe-a",), hole_ids=(1,))

    assert _normalize_atlas_payload((atlas, ["probe-a"])) is None


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
    path = ROOT / "src/aind_low_point/optimization/pipeline/phase1_pool.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_phase1_pool_record"
    ]

    assert helper_calls
