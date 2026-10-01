"""A solved plan set: the index emit writes, and the order the planner walks it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aind_rutter.plan_io.plan_set import (
    PlanEntry,
    RecordedMeasure,
    load_plan_set,
    write_manifest,
)
from aind_rutter.web.plan_sets import ordered_files, plan_rows

ENTRIES = [
    PlanEntry(
        "plan-01-cov08.00-md1_pl2.plan.yml", 1, 8.0, 0.05, 3, 11, {"MD": 1, "PL": 2}
    ),
    PlanEntry(
        "plan-02-cov09.00-md3_pl2.plan.yml", 2, 9.0, 0.01, 2, 12, {"MD": 3, "PL": 2}
    ),
    PlanEntry(
        "plan-03-cov07.00-md12_pl4.plan.yml", 3, 7.0, 0.10, 3, 13, {"MD": 12, "PL": 4}
    ),
]


def _emit_dir(tmp_path: Path, *, manifest: bool) -> Path:
    plans = tmp_path / "plans"
    plans.mkdir()
    for e in ENTRIES:
        (plans / e.file).write_text("arcs: {}\n")
    if manifest:
        write_manifest(tmp_path, ENTRIES, probes=["MD", "PL"], config="cfg.yml")
    return tmp_path


def test_the_manifest_round_trips(tmp_path: Path) -> None:
    ps = load_plan_set(_emit_dir(tmp_path, manifest=True))
    assert ps.entries == tuple(ENTRIES)
    assert ps.probes == ("MD", "PL") and ps.config == "cfg.yml" and ps.indexed
    assert ps.path(ps.entries[0]).is_file()


def test_the_plans_folder_opens_the_same_set(tmp_path: Path) -> None:
    out = _emit_dir(tmp_path, manifest=True)
    assert load_plan_set(out / "plans").entries == load_plan_set(out).entries


def test_a_set_without_a_manifest_reads_rank_and_coverage_from_filenames(
    tmp_path: Path,
) -> None:
    ps = load_plan_set(_emit_dir(tmp_path, manifest=False))
    assert not ps.indexed
    assert [(e.rank, e.coverage, e.fcl_mm) for e in ps.entries] == [
        (1, 8.0, None),
        (2, 9.0, None),
        (3, 7.0, None),
    ]


def test_a_directory_with_no_plans_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_plan_set(tmp_path)


def _rows(tmp_path: Path) -> list[dict]:
    return plan_rows(load_plan_set(_emit_dir(tmp_path, manifest=True)))


def test_rows_walk_in_the_table_sort(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    by_cov = ordered_files(rows, [{"key": "coverage", "order": "desc"}], "")
    assert by_cov == [ENTRIES[i].file for i in (1, 0, 2)]
    by_arcs_then_fcl = ordered_files(
        rows,
        [{"key": "n_arcs", "order": "asc"}, {"key": "fcl_mm", "order": "desc"}],
        "",
    )
    assert by_arcs_then_fcl == [ENTRIES[i].file for i in (1, 2, 0)]


def test_the_filter_keeps_rows_any_column_matches(tmp_path: Path) -> None:
    assert ordered_files(_rows(tmp_path), [], "md 12") == [ENTRIES[2].file]


def test_blank_values_sort_last_both_ways(tmp_path: Path) -> None:
    rows = plan_rows(load_plan_set(_emit_dir(tmp_path, manifest=False)))
    for order in ("asc", "desc"):
        assert ordered_files(rows, [{"key": "fcl_mm", "order": order}], "") == [
            e.file for e in ENTRIES
        ]


def test_the_recorded_share_round_trips_and_averages_over_regions(
    tmp_path: Path,
) -> None:
    entry = PlanEntry(
        "plan-01-cov08.00-md1.plan.yml", 1, recorded={"MD": 0.4, "PL": 0.1}
    )
    measure = RecordedMeasure("retro", 0.2)
    write_manifest(tmp_path, [entry], probes=["MD"], config=None, measure=measure)
    ps = load_plan_set(tmp_path)
    assert ps.measure == measure and ps.entries == (entry,)
    assert ps.entries[0].mean_recorded() == pytest.approx(0.25)


def test_a_manifest_written_before_the_measure_still_loads(tmp_path: Path) -> None:
    out = _emit_dir(tmp_path, manifest=True)
    raw = json.loads((out / "manifest.json").read_text())
    raw["version"] = 1
    del raw["measure"]
    for plan in raw["plans"]:
        del plan["recorded"]
    (out / "manifest.json").write_text(json.dumps(raw))
    ps = load_plan_set(out)
    assert ps.measure is None and ps.entries == tuple(ENTRIES)
    assert ps.entries[0].mean_recorded() is None
