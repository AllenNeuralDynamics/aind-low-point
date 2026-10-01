"""The series solve driver, against a stand-in for the pipeline wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from aind_rutter.series.solve import (
    RECORD_NAME,
    default_script,
    run_environment,
    solve,
    solve_runs,
)
from aind_rutter.series.spec import SeriesSpec

S = "2.1"
# Writes FAKE_PLANS_<arc cap> plans (none when unset), exits FAKE_EXIT, and logs
# the variables a test asks about.
SCRIPT = """#!/usr/bin/env bash
set -eu
echo "arcs=$RUTTER_MAX_ARCS fresh=${FRESH:-0} retro=$RUTTER_RETRO_DENSITY \\
topk=$RUTTER_TOPK workers=${RUTTER_WORKERS:-} atlas=${RUTTER_ATLAS_CACHE:-}" \\
  >> "$RUTTER_RUN_DIR/calls.log"
[ "${FAKE_EXIT:-0}" = 0 ] || exit "$FAKE_EXIT"
n_var="FAKE_PLANS_$RUTTER_MAX_ARCS"; n="${!n_var:-0}"
mkdir -p "$RUTTER_RUN_DIR/plans"
plans=""
for i in $(seq 1 "$n"); do
  touch "$RUTTER_RUN_DIR/plans/plan-$RUTTER_MAX_ARCS-$i.plan.yml"
  plans="$plans{\\"rank\\": $i},"
done
echo "{\\"plans\\": [${plans%,}]}" > "$RUTTER_RUN_DIR/manifest.json"
"""


@pytest.fixture
def series(tmp_path: Path):
    """A two-set series whose configs are stand-in files, and the fake wrapper."""
    for name in ("base.yml", "holes.yml"):
        (tmp_path / name).write_text(f"{name}\n")
    data: dict[str, Any] = {
        "name": "pair",
        "hemisphere": "left",
        "subjects": {
            "m1": {
                "base_config": str(tmp_path / "base.yml"),
                "holes": str(tmp_path / "holes.yml"),
            }
        },
        "inventory": {S: 2},
        "solve": {"max_arcs": [3, 4], "topk": 50},
        "sets": {
            "one": {
                "regions": ["MD"],
                "score": "retro",
                "allocations": {"1p": [{"kind": S, "regions": ["MD"]}]},
            },
            "two": {
                "regions": ["VM"],
                "solve": {"max_arcs": [4]},
                "allocations": {"1p": [{"kind": S, "regions": ["VM"]}]},
            },
        },
    }
    spec = SeriesSpec.model_validate(data)
    run_dir = tmp_path / "run"
    for run in solve_runs(spec, run_dir):
        run.config.parent.mkdir(parents=True)
        run.config.write_text(f"{run.label}\n")
    script = tmp_path / "wrapper.sh"
    script.write_text(SCRIPT)
    script.chmod(0o755)
    return spec, run_dir, script, data


def _calls(run_dir: Path, set_name: str) -> list[str]:
    log = run_dir / "m1" / "solve" / set_name / "1p" / "calls.log"
    return log.read_text().splitlines() if log.is_file() else []


def _solve(series, **environ: str):
    spec, run_dir, script, _ = series
    said: list[str] = []
    env = {"PATH": "/usr/bin:/bin", **environ}
    outcomes = solve(spec, run_dir, script=script, environ=env, report=said.append)
    return outcomes, said


class TestRuns:
    def test_one_run_per_config_with_its_own_directory(self, series):
        spec, run_dir, _, _ = series
        runs = solve_runs(spec, run_dir)
        assert [r.label for r in runs] == ["m1 / one / 1p", "m1 / two / 1p"]
        assert runs[0].out_dir == run_dir / "m1" / "solve" / "one" / "1p"
        assert runs[0].config == run_dir / "m1" / "configs" / "one" / "1p.yml"

    def test_the_spec_decides_the_result_changing_variables(self, series):
        spec, run_dir, _, _ = series
        one, two = solve_runs(spec, run_dir)
        env = run_environment(spec, one, 3)
        assert env == {
            "RUTTER_CONFIG": str(one.config),
            "RUTTER_HOLES": str(spec.subjects["m1"].holes),
            "RUTTER_RUN_DIR": str(one.out_dir),
            "RUTTER_MAX_ARCS": "3",
            "RUTTER_TOPK": "50",
            "RUTTER_SELECT_BY": "objective",
            "RUTTER_RETRO_DENSITY": "1",
            "RUTTER_RECORDED_RADIUS_MM": "0.2",
        }
        assert run_environment(spec, two, 4)["RUTTER_RETRO_DENSITY"] == "0"

    def test_the_default_wrapper_is_the_checkouts(self):
        assert default_script().name == "run_subject_overnight.sh"


class TestSolve:
    def test_the_arc_cap_rises_only_when_a_run_finds_nothing(self, series):
        _, run_dir, _, _ = series
        outcomes, said = _solve(series, FAKE_PLANS_4="2")
        assert [(o.ok, o.max_arcs, o.n_plans) for o in outcomes] == [
            (True, 4, 2),
            (True, 4, 2),
        ]
        assert [c.split()[0] for c in _calls(run_dir, "one")] == ["arcs=3", "arcs=4"]
        assert [c.split()[0] for c in _calls(run_dir, "two")] == ["arcs=4"]
        assert "m1 / one / 1p: done (arcs<=3): 0 plans" in said

    def test_plans_at_the_first_cap_end_the_run(self, series):
        _, run_dir, _, _ = series
        outcomes, _ = _solve(series, FAKE_PLANS_3="5", FAKE_PLANS_4="1")
        assert (outcomes[0].max_arcs, outcomes[0].n_plans) == (3, 5)
        assert len(_calls(run_dir, "one")) == 1

    def test_the_machines_variables_pass_and_one_runs_paths_do_not(self, series):
        _, run_dir, _, _ = series
        _solve(
            series,
            FAKE_PLANS_3="1",
            RUTTER_WORKERS="2",
            RUTTER_ATLAS_CACHE="/elsewhere/atlas.json.gz",
        )
        call = _calls(run_dir, "one")[0]
        assert "workers=2" in call and call.endswith("atlas=")
        assert "retro=1" in call and "topk=50" in call

    def test_a_solved_run_is_skipped(self, series):
        _, run_dir, _, _ = series
        _solve(series, FAKE_PLANS_3="1", FAKE_PLANS_4="1")
        outcomes, said = _solve(series, FAKE_PLANS_3="9", FAKE_PLANS_4="9")
        assert [(o.skipped, o.n_plans) for o in outcomes] == [(True, 1), (True, 1)]
        assert len(_calls(run_dir, "one")) == 1
        assert said == ["m1 / one / 1p: done, 1 plans", "m1 / two / 1p: done, 1 plans"]

    def test_a_run_that_found_nothing_at_every_cap_is_done(self, series):
        _, run_dir, _, _ = series
        _solve(series)
        outcomes, _ = _solve(series)
        assert [(o.skipped, o.ok, o.n_plans) for o in outcomes] == [(True, True, 0)] * 2
        assert len(_calls(run_dir, "one")) == 2

    def test_a_failure_ends_that_run_and_is_retried_next_time(self, series):
        _, run_dir, _, _ = series
        outcomes, said = _solve(series, FAKE_EXIT="3")
        assert [(o.ok, o.max_arcs) for o in outcomes] == [(False, 3), (False, 4)]
        assert "m1 / one / 1p: FAILED (arcs<=3): 0 plans" in said
        outcomes, _ = _solve(series, FAKE_PLANS_3="1", FAKE_PLANS_4="1")
        assert [(o.skipped, o.ok, o.n_plans) for o in outcomes] == [
            (False, True, 1)
        ] * 2
        calls = _calls(run_dir, "one")
        assert [c.split()[:2] for c in calls] == [["arcs=3", "fresh=0"]] * 2

    def test_a_changed_input_is_redone_from_nothing(self, series):
        spec, run_dir, script, data = series
        _solve(series, FAKE_PLANS_3="2", FAKE_PLANS_4="2")
        (run_dir / "m1" / "configs" / "one" / "1p.yml").write_text("edited\n")
        outcomes, _ = _solve(series, FAKE_PLANS_3="1", FAKE_PLANS_4="1")
        assert [(o.skipped, o.n_plans) for o in outcomes] == [(False, 1), (True, 2)]
        assert _calls(run_dir, "one")[-1].split()[:2] == ["arcs=3", "fresh=1"]
        # The earlier attempt's plans are gone, not added to.
        plans = run_dir / "m1" / "solve" / "one" / "1p" / "plans"
        assert [p.name for p in plans.iterdir()] == ["plan-3-1.plan.yml"]

        data["solve"]["topk"] = 60
        changed = SeriesSpec.model_validate(data)
        env = {"PATH": "/usr/bin:/bin", "FAKE_PLANS_3": "1", "FAKE_PLANS_4": "1"}
        outcomes = solve(changed, run_dir, script=script, environ=env, report=str)
        assert [o.skipped for o in outcomes] == [False, False]

    def test_the_record_says_what_was_asked_and_what_ran(self, series):
        spec, run_dir, _, _ = series
        _solve(series, FAKE_PLANS_4="2", RUTTER_WORKERS="2")
        out_dir = run_dir / "m1" / "solve" / "one" / "1p"
        record = json.loads((out_dir / RECORD_NAME).read_text())
        assert record["request"]["score"] == "retro"
        assert record["request"]["solve"]["max_arcs"] == [3, 4]
        assert [
            (a["max_arcs"], a["returncode"], a["n_plans"]) for a in record["attempts"]
        ] == [
            (3, 0, 0),
            (4, 0, 2),
        ]
        assert record["environment"]["RUTTER_WORKERS"] == "2"
        assert record["environment"]["RUTTER_MAX_ARCS"] == "4"
        assert (out_dir / "solve.arcs3.log").is_file()

    def test_select_narrows_the_runs_and_an_unwritten_config_is_refused(self, series):
        spec, run_dir, script, _ = series
        env = {"PATH": "/usr/bin:/bin", "FAKE_PLANS_4": "1"}
        outcomes = solve(
            spec,
            run_dir,
            script=script,
            environ=env,
            select=lambda run: run.set_name == "two",
            report=str,
        )
        assert [o.run.set_name for o in outcomes] == ["two"]
        (run_dir / "m1" / "configs" / "one" / "1p.yml").unlink()
        with pytest.raises(FileNotFoundError, match="m1 / one / 1p"):
            solve(spec, run_dir, script=script, environ=env, report=str)


class TestSpec:
    def _data(self, series) -> dict[str, Any]:
        return yaml.safe_load(yaml.safe_dump(series[3]))

    def test_a_set_restates_only_the_fields_it_names(self, series):
        spec = series[0]
        assert spec.solve_for("one").max_arcs == (3, 4)
        assert spec.solve_for("two").max_arcs == (4,)
        assert spec.solve_for("two").topk == 50

    @pytest.mark.parametrize(
        "solve, says",
        [
            ({"max_arcs": [4, 3]}, "rising"),
            ({"max_arcs": []}, "at least 1"),
            ({"topk": 0}, "greater than or equal to 1"),
            ({"select_by": "learned"}, "ranker_table"),
            ({"select_by": "coverage"}, "objective"),
            ({"workers": 3}, "Extra inputs"),
        ],
    )
    def test_refuses(self, series, solve: dict[str, Any], says: str):
        data = self._data(series)
        data["solve"] = solve
        with pytest.raises(ValueError, match=says):
            SeriesSpec.model_validate(data)
        data = self._data(series)
        data["sets"]["two"]["solve"] = solve
        with pytest.raises(ValueError, match=says):
            SeriesSpec.model_validate(data)
