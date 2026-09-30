"""Phase 2's learned intake order, without measuring a pool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aind_rutter.optimization.pipeline import learned_rank as lr
from aind_rutter.optimization.pipeline.selection import select_records
from aind_rutter.optimization.pipeline.settings import Phase2Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in Phase2Settings.model_fields.values():
        for name in getattr(field.validation_alias, "choices", ()) or ():
            monkeypatch.delenv(str(name), raising=False)


def _rec(idx: int, n_arcs: int, objective: float) -> dict:
    aps = [10.0 * a - 5.0 for a in range(n_arcs)]
    probes = [[3.0 * p, 1.0, 0.0, 0.3, 0.4, 0.5 + p] for p in range(3)]
    return {
        "idx": idx,
        "n_arcs": n_arcs,
        "x": np.array(aps + [v for p in probes for v in p]),
        "objective": objective,
        "min_clear": 0.1,
        "min_clear_reduced": 0.2,
        "min_ml_gap": 2.0,
        "probe_to_hole": {"a": 1, "b": 2, "c": 3},
        "partition": frozenset(),
        "probe_to_arc_idx": {"a": 0, "b": 0, "c": n_arcs - 1},
        "arc_centroids_deg": aps,
    }


def _measured(n: int) -> dict[str, np.ndarray]:
    return {k: np.linspace(0.0, 1.0, n) for k in lr.MEASURED}


def test_hybrid_takes_the_objective_head_then_the_score() -> None:
    order = lr.hybrid_order(
        [4, 3, 2, 1, 0], np.array([0.0, 0.9, 0.5, 0.1, 0.2]), head=2
    )
    assert order == [4, 3, 1, 2, 0]


def test_feature_matrix_columns() -> None:
    recs = [_rec(0, 1, 5.0), _rec(1, 2, 1.0), _rec(2, 2, 3.0)]
    X = lr.feature_matrix(recs, _measured(3))
    col = dict(zip(lr.FEATURES, X.T))
    assert X.shape == (3, len(lr.FEATURES))
    assert np.allclose(col["objective_pct"], [5 / 6, 1 / 6, 1 / 2])
    assert np.isnan(col["min_ap_gap"][0]) and col["min_ap_gap"][1] == 10.0
    assert np.allclose(col["ml_span"], 6.0) and np.allclose(col["offset_max"], 0.5)
    assert list(col["max_per_arc"]) == [3.0, 2.0, 2.0]


def test_a_written_table_fits_to_probabilities_of_percentiles(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    recs = [_rec(i, 2 + i % 2, float(rng.normal())) for i in range(n)]
    kept = rng.random(n) < 0.5
    path = tmp_path / "table.npz"
    lr.write_training_table(path, recs, _measured(n), kept, rng.random(n), "test")
    t = lr.training_table(path)
    assert t["X"].shape == (n, len(lr.FEATURES)) and t["kept"].sum() == kept.sum()
    score = lr.fit(t["X"], t["kept"], t["target"])(t["X"])
    assert np.all((score > -0.1) & (score < 1.1))


def test_a_cut_past_the_head_needs_a_table() -> None:
    recs = [_rec(0, 2, 5.0), _rec(1, 2, 1.0), _rec(2, 2, 3.0)]
    with pytest.raises(ValueError, match="RUTTER_RANKER_TABLE"):
        select_records(recs, Phase2Settings(topk=2, rank_head=1))


def test_learned_is_the_default_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake(recs, settings, objective_order, log=None):
        seen["objective_order"] = objective_order
        return [0, 2, 1]

    monkeypatch.setattr(lr, "learned_order", fake)
    recs = [_rec(0, 2, 5.0), _rec(1, 2, 1.0), _rec(2, 2, 3.0)]
    settings = Phase2Settings(topk=2, rank_head=1, ranker_table=Path("table.npz"))
    out = select_records(recs, settings)
    assert seen["objective_order"] == [1, 2, 0]
    assert [r["idx"] for r in out] == [0, 2] and [r["rank"] for r in out] == [0, 1]


def test_a_cut_inside_the_head_measures_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("measured a pool whose cut is the objective's")

    monkeypatch.setattr(lr, "learned_order", fail)
    recs = [_rec(0, 2, 5.0), _rec(1, 2, 1.0), _rec(2, 2, 3.0)]
    out = select_records(recs, Phase2Settings(topk=2, rank_head=2))
    assert [r["idx"] for r in out] == [1, 2]
