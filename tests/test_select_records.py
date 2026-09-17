"""Which Phase-1 candidates reach Phase 2, and how they arrive."""

from __future__ import annotations

import pytest

from aind_rutter.optimization.pipeline.selection import (
    rank_order,
    select_records,
)
from aind_rutter.optimization.pipeline.settings import Phase2Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # TOPK, SELECT_BY, RANKS and the payload paths all have environment aliases.
    for field in Phase2Settings.model_fields.values():
        for name in getattr(field.validation_alias, "choices", ()) or ():
            monkeypatch.delenv(str(name), raising=False)


def _rec(idx: int, **fields) -> dict:
    base = {
        "idx": idx,
        "n_arcs": 2,
        "x": [float(idx)],
        "probe_to_hole": {"p": idx},
        "partition": frozenset({frozenset({"p"})}),
        "probe_to_arc_idx": {"p": 0},
        "arc_centroids_deg": [0.0],
    }
    base.update(fields)
    return base


def test_clearance_style_fields_rank_highest_first() -> None:
    recs = [_rec(0, min_clear=0.1), _rec(1, min_clear=0.5), _rec(2, min_clear=0.3)]
    assert rank_order(recs, "min_clear") == [1, 2, 0]


def test_objective_ranks_lowest_first() -> None:
    recs = [_rec(0, objective=9.0), _rec(1, objective=-3.0), _rec(2, objective=1.0)]
    assert rank_order(recs, "objective") == [1, 2, 0]


def test_records_missing_the_ranking_field_sort_last_either_way() -> None:
    recs = [_rec(0), _rec(1, min_clear=0.2)]
    assert rank_order(recs, "min_clear") == [1, 0]
    recs = [_rec(0), _rec(1, objective=5.0)]
    assert rank_order(recs, "objective") == [1, 0]


def test_topk_truncates_and_rank_records_the_position() -> None:
    recs = [_rec(i, min_clear=i / 10) for i in range(5)]
    out = select_records(recs, Phase2Settings(select_by="min_clear", topk=2))
    assert [r["idx"] for r in out] == [4, 3], "best first"
    assert [r["rank"] for r in out] == [0, 1]


def test_explicit_ranks_override_topk_and_drop_out_of_range() -> None:
    recs = [_rec(i, min_clear=i / 10) for i in range(4)]
    out = select_records(
        recs, Phase2Settings(select_by="min_clear", topk=99, ranks="0,2,17")
    )
    # Ranked order is 3,2,1,0; offsets 0 and 2 select candidates 3 and 1, and the
    # offset past the end is dropped rather than raising.
    assert [r["idx"] for r in out] == [3, 1]
    assert [r["rank"] for r in out] == [0, 2]


def test_pose_falls_back_to_the_phase_one_variable_name() -> None:
    out = select_records(
        [_rec(0, min_clear=1.0, pose=[7.0])], Phase2Settings(select_by="min_clear")
    )
    assert out[0]["pose"] == [7.0]
    out = select_records(
        [_rec(0, min_clear=1.0)], Phase2Settings(select_by="min_clear")
    )
    assert out[0]["pose"] == [0.0], "falls back to x when pose is absent"


def test_the_arc_assignment_is_carried_through_verbatim() -> None:
    rec = _rec(3, min_clear=1.0)
    out = select_records([rec], Phase2Settings(select_by="min_clear"))[0]
    assert out["probe_to_hole"] == rec["probe_to_hole"]
    assert out["probe_to_arc_idx"] == rec["probe_to_arc_idx"]
    assert out["arc_centroids_deg"] == rec["arc_centroids_deg"]
    assert out["n_arcs"] == rec["n_arcs"]
