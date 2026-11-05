"""Solving an earlier Phase 2 again: which records, from which pose, and under
which settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from aind_rutter.optimization.pipeline.payloads import payload_kind
from aind_rutter.optimization.pipeline.selection import (
    candidate_ids,
    from_handoff_record,
    select_from_handoff,
    select_records,
)
from aind_rutter.optimization.pipeline.settings import (
    Phase2Settings,
    resolve_settings,
)
from aind_rutter.plan_io.plan_set import PlanEntry, write_manifest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in Phase2Settings.model_fields.values():
        for name in getattr(field.validation_alias, "choices", ()) or ():
            monkeypatch.delenv(str(name), raising=False)


def _solved(idx: int) -> dict[str, Any]:
    return {
        "idx": idx,
        "n_arcs": 2,
        "pose": [float(idx), 1.0],
        "pose_in": [float(idx), 0.0],
        "hole": {"p": idx},
        "partition": frozenset({frozenset({"p"})}),
        "probe_to_arc_idx": {"p": 0},
        "arc_centroids_deg": [0.0],
        "min_clear": 0.2,
        "objective_p1": -float(idx),
    }


def _handoff(ids: list[int], ranked: list[int]) -> Any:
    recs = {i: _solved(i) for i in ids}
    return cast(Any, {"all": list(recs.values()), "ranked": [recs[i] for i in ranked]})


def test_a_handoff_record_starts_from_its_solved_or_phase1_pose() -> None:
    rec = cast(Any, _solved(4))
    solved = from_handoff_record(rec, 0, "solved")
    assert solved["pose"] == [4.0, 1.0]
    assert solved["probe_to_hole"] == {"p": 4} and solved["objective"] == -4.0
    assert from_handoff_record(rec, 0, "phase1")["pose"] == [4.0, 0.0]


def test_without_names_a_handoff_gives_its_first_ranked_records() -> None:
    out = select_from_handoff(_handoff([0, 1, 2], [2, 0, 1]), Phase2Settings(topk=2))
    assert [r["idx"] for r in out] == [2, 0]
    assert [r["rank"] for r in out] == [0, 1]


def test_named_candidates_come_in_the_order_named() -> None:
    out = select_from_handoff(
        _handoff([0, 1, 2], [0]), Phase2Settings(candidates="2,0", topk=1)
    )
    assert [r["idx"] for r in out] == [2, 0], "names outrank topk"


def test_an_unknown_or_repeated_candidate_is_refused() -> None:
    with pytest.raises(ValueError, match="no candidates"):
        select_from_handoff(_handoff([0, 1], [0]), Phase2Settings(candidates="5"))
    with pytest.raises(ValueError, match="named twice"):
        select_from_handoff(_handoff([0, 1], [0]), Phase2Settings(candidates="1,1"))


def test_pool_offsets_are_refused_for_a_handoff() -> None:
    with pytest.raises(ValueError, match="offsets"):
        select_from_handoff(_handoff([0], [0]), Phase2Settings(ranks="0"))


def test_only_one_way_of_choosing_candidates_is_accepted() -> None:
    with pytest.raises(ValueError, match="one way"):
        Phase2Settings(ranks="0", candidates="1")


def test_named_candidates_select_from_a_pool_by_id() -> None:
    recs = [
        {
            "idx": i,
            "n_arcs": 2,
            "x": [float(i)],
            "probe_to_hole": {"p": i},
            "partition": frozenset({frozenset({"p"})}),
            "probe_to_arc_idx": {"p": 0},
            "arc_centroids_deg": [0.0],
        }
        for i in (10, 11, 12)
    ]
    out = select_records(cast(Any, recs), Phase2Settings(candidates="12,10"))
    assert [r["idx"] for r in out] == [12, 10]


def _manifest(tmp_path: Path) -> Path:
    entries = [
        PlanEntry("plan-01.plan.yml", 1, candidate=7, recorded={"A": 0.1, "B": 0.1}),
        PlanEntry("plan-02.plan.yml", 2, candidate=3, recorded={"A": 0.5, "B": 0.3}),
        PlanEntry("plan-03.plan.yml", 3, candidate=9, recorded={"A": 0.2, "B": 0.2}),
    ]
    (tmp_path / "plans").mkdir()
    write_manifest(tmp_path, entries, probes=["p"], config="cfg.yml")
    return tmp_path / "manifest.json"


def test_a_manifest_names_its_first_plans_in_either_order(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    listed = Phase2Settings(candidates_file=path, topk=2)
    assert candidate_ids(listed) == [7, 3]
    by_share = Phase2Settings(candidates_file=path, topk=2, candidates_by="recorded")
    assert candidate_ids(by_share) == [3, 9]


def test_any_other_candidates_file_is_a_list_of_ids(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("4, 8  # best two\n2\n")
    assert candidate_ids(Phase2Settings(candidates_file=path)) == [4, 8, 2]


def test_recorded_settings_fill_what_the_run_leaves_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = Phase2Settings(fcl_tol=0.3, g_tol=0.2, topk=5).model_dump(mode="json")
    monkeypatch.setenv("RUTTER_G_TOL", "0.05")
    settings, changed = resolve_settings(recorded)
    assert settings.fcl_tol == 0.3, "inherited"
    assert settings.g_tol == 0.05, "the environment outranks the recording"
    assert settings.topk == Phase2Settings().topk, "run fields are not inherited"
    assert changed == {"g_tol": (0.2, 0.05)}


def test_given_settings_outrank_the_recording() -> None:
    recorded = Phase2Settings(fcl_tol=0.3).model_dump(mode="json")
    settings, changed = resolve_settings(recorded, environ={}, fcl_tol=0.1)
    assert settings.fcl_tol == 0.1 and changed == {"fcl_tol": (0.3, 0.1)}


def test_a_payload_names_its_kind(tmp_path: Path) -> None:
    path = tmp_path / "phase2_handoff.json"
    path.write_text(json.dumps({"kind": "phase2_handoff", "schema_version": 1}))
    assert payload_kind(path) == "phase2_handoff"
    path.write_text("{}")
    with pytest.raises(ValueError, match="not a pipeline payload"):
        payload_kind(path)
