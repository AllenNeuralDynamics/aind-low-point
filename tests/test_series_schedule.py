"""The series schedule: the search against brute force, then a run directory."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from aind_rutter.domain.rig_setup import SetupChanges
from aind_rutter.plan_io.plan_set import PlanEntry, write_manifest
from aind_rutter.series import cli
from aind_rutter.series.schedule import (
    cheapest,
    plan_setup,
    schedule_sets,
    schedule_subject,
    subject_pools,
    write_schedule,
)
from aind_rutter.series.solve import RECORD_NAME, RunState, request, solve_runs
from aind_rutter.series.spec import ScheduleSpec, SeriesSpec

Q, S = "quadbase-alpha", "2.1"
Probe = tuple[str, str, float, float]  # kind, arc, slider ML, spin


def _brute(cost, score, first=frozenset(), latest=None):
    """Every order and every plan of each set: the least cost, then the most score."""
    n, latest = len(score), latest or {}
    best = None
    for order in itertools.permutations(range(n)):
        if set(order[: len(first)]) != set(first):
            continue
        if any(order.index(s) + 1 > last for s, last in latest.items()):
            continue
        for picks in itertools.product(*(range(len(score[s])) for s in order)):
            if any(not np.isfinite(score[s][p]) for s, p in zip(order, picks)):
                continue
            steps = list(zip(order, picks))
            total = sum(cost[a, b][p, q] for (a, p), (b, q) in zip(steps, steps[1:]))
            got = sum(score[s][p] for s, p in steps)
            if best is None or (total, -got) < best:
                best = (total, -got)
    return best


def _instance(rng: np.random.Generator, n: int):
    sizes = rng.integers(1, 4, n)
    score = [rng.integers(1, 5, k).astype(float) for k in sizes]
    for s in score:  # some plans below the floor, never all of a set
        s[1:][rng.random(len(s) - 1) < 0.3] = -np.inf
    cost = {
        (a, b): rng.integers(0, 4, (sizes[a], sizes[b])).astype(float)
        for a in range(n)
        for b in range(n)
        if a != b
    }
    return cost, score


class TestCheapest:
    @pytest.mark.parametrize("seed", range(40))
    def test_matches_brute_force(self, seed: int):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 6))
        cost, score = _instance(rng, n)
        first = frozenset(rng.choice(n, int(rng.integers(0, n)), replace=False))
        first = frozenset(int(s) for s in first)
        latest = {int(rng.integers(0, n)): int(rng.integers(1, n + 1))}
        for constraints in ({}, {"first": first}, {"first": first, "latest": latest}):
            want = _brute(cost, score, **constraints)
            found = cheapest(cost, score, **constraints)
            if want is None:
                assert found is None
                continue
            total, picks = found
            assert sorted(s for s, _ in picks) == list(range(n))
            paid = sum(cost[a, b][p, q] for (a, p), (b, q) in zip(picks, picks[1:]))
            assert (paid, total) == (want[0], want[0])
            assert -sum(score[s][p] for s, p in picks) == want[1]
            order = [s for s, _ in picks]
            assert set(order[: len(constraints.get("first", ()))]) == set(
                constraints.get("first", ())
            )
            for s, last in constraints.get("latest", {}).items():
                assert order.index(s) + 1 <= last

    def test_one_set_is_its_best_plan(self):
        assert cheapest({}, [np.array([0.2, 0.5, -np.inf])]) == (0.0, [(0, 1)])

    def test_orders_that_cannot_be_are_none(self):
        cost, score = _instance(np.random.default_rng(0), 3)
        assert cheapest(cost, score, latest={0: 1, 1: 1}) is None
        assert cheapest(cost, score, frozenset({0, 1}), {2: 2}) is None


def _write_plan(path: Path, arcs: dict[str, float], probes: list[Probe]) -> None:
    plan = {
        "arcs": arcs,
        "probes": {
            f"P{i}": {
                "kind": kind,
                "arc": arc,
                "slider_ml": ml,
                "spin": spin,
                "target": [0.0, 0.0, 0.0],
            }
            for i, (kind, arc, ml, spin) in enumerate(probes, 1)
        },
    }
    path.write_text(yaml.safe_dump(plan))


def _write_plans(
    out_dir: Path, plans: list[tuple[float, dict[str, float], list[Probe]]]
) -> None:
    """A solve directory whose plans record the given shares, best first or not."""
    (out_dir / "plans").mkdir(parents=True, exist_ok=True)
    entries = []
    for rank, (share, arcs, probes) in enumerate(plans, 1):
        file = f"plan-{rank:02d}-cov01.00-x.plan.yml"
        _write_plan(out_dir / "plans" / file, arcs, probes)
        # Two regions, so the score is their mean.
        entries.append(
            PlanEntry(file, rank, recorded={"MD": share + 0.1, "VM": share - 0.1})
        )
    write_manifest(out_dir, entries, probes=[], config=None)


ONE_SINGLE = ({"a": 0.0}, [(S, "a", 0.0, 0.0)])
SINGLE_SLID = ({"a": 0.0}, [(S, "a", 20.0, 0.0)])
ONE_QUAD = ({"a": 0.0}, [(Q, "a", 0.0, 0.0)])


@pytest.fixture
def series(tmp_path: Path):
    """Three sets, the first with two allocations, and nothing solved."""
    for name in ("base.yml", "holes.yml"):
        (tmp_path / name).write_text(f"{name}\n")
    one_probe = [{"kind": S, "regions": ["MD", "VM"]}]
    data: dict[str, Any] = {
        "name": "trio",
        "hemisphere": "left",
        "subjects": {
            "m1": {
                "base_config": str(tmp_path / "base.yml"),
                "holes": str(tmp_path / "holes.yml"),
            }
        },
        "inventory": {S: 2, Q: 1},
        "sets": {
            name: {"regions": ["MD", "VM"], "allocations": {"s": one_probe}}
            for name in ("one", "two", "three")
        },
        "schedule": {"floors": [1.0, 0.9]},
    }
    data["sets"]["one"]["allocations"]["q"] = [{"kind": Q, "regions": ["MD", "VM"]}]
    spec = SeriesSpec.model_validate(data)
    run_dir = tmp_path / "run"
    for run in solve_runs(spec, run_dir):
        run.config.parent.mkdir(parents=True, exist_ok=True)
        run.config.write_text(f"{run.label}\n")
    return spec, run_dir, data


def _solved(spec: SeriesSpec, run_dir: Path, label: str, plans: list) -> None:
    run = next(r for r in solve_runs(spec, run_dir) if r.label == label)
    _write_plans(run.out_dir, plans)
    record = {
        "request": request(spec, run),
        "attempts": [{"max_arcs": 3, "returncode": 0, "n_plans": len(plans)}],
    }
    (run.out_dir / RECORD_NAME).write_text(json.dumps(record))


def _solve_all(spec: SeriesSpec, run_dir: Path) -> None:
    _solved(spec, run_dir, "m1 / one / s", [(0.50, *ONE_SINGLE), (0.46, *SINGLE_SLID)])
    _solved(spec, run_dir, "m1 / two / s", [(0.30, *SINGLE_SLID)])
    _solved(spec, run_dir, "m1 / three / s", [(0.20, *SINGLE_SLID)])


class TestPlanSetup:
    def test_reads_the_arcs_and_the_probes_on_them(self, tmp_path: Path):
        path = tmp_path / "x.plan.yml"
        _write_plan(
            path,
            {"a": 10.0, "b": -20.0, "c": 0.0},
            [(S, "a", 5.0, -90.0), (Q, "a", -5.0, 0.0), (S, "b", 0.0, 30.0)],
        )
        setup = plan_setup(path)
        assert [arc.ap_deg for arc in setup] == [-20.0, 10.0]
        assert [(s.kind, s.ml_deg, s.spin_deg) for s in setup[1].slots] == [
            (Q, -5.0, 0.0),
            (S, 5.0, 270.0),
        ]


class TestSchedule:
    def test_a_lower_floor_buys_less_rework(self, series):
        spec, run_dir, _ = series
        _solve_all(spec, run_dir)
        strict, loose = schedule_subject(spec, run_dir, "m1").schedules
        assert (strict.floor, loose.floor) == (1.0, 0.9)
        # The best plan of set one sits at another slider angle than the rest.
        assert strict.cost == 1.0 and loose.cost == 0.0
        ranks = {x.plan.set_name: x.plan.entry.rank for x in loose.sessions}
        assert ranks == {"one": 2, "two": 1, "three": 1}
        # The dear session goes to an end, where it is paid once.
        assert strict.sessions[0].plan.set_name == "one" or (
            strict.sessions[-1].plan.set_name == "one"
        )
        assert strict.sessions[0].changes is None
        assert [x.cost for x in strict.sessions].count(1.0) == 1
        slid = next(x.changes for x in strict.sessions if x.cost)
        assert slid == SetupChanges(slider_moves=1)

    def test_a_sets_floor_is_its_best_plan_of_any_allocation(self, series):
        spec, run_dir, _ = series
        _solve_all(spec, run_dir)
        _solved(spec, run_dir, "m1 / one / q", [(0.55, *ONE_QUAD)])
        result = schedule_subject(spec, run_dir, "m1")
        assert sorted(c.allocation for c in result.pools["one"]) == ["q", "s", "s"]
        strict, loose = result.schedules
        one = next(x for x in strict.sessions if x.plan.set_name == "one")
        assert (one.plan.allocation, one.of_best) == ("q", 1.0)
        # 0.46 is below nine tenths of 0.55, so the slid single no longer counts.
        one = next(x for x in loose.sessions if x.plan.set_name == "one")
        assert (one.plan.allocation, one.plan.entry.rank) == ("s", 1)
        assert one.of_best == pytest.approx(0.50 / 0.55)

    def test_the_order_constraints_hold(self, series):
        spec, run_dir, data = series
        data["schedule"] = {"floors": [1.0], "first": ["two"], "within": {"one": 2}}
        spec = SeriesSpec.model_validate(data)
        _solve_all(spec, run_dir)
        (only,) = schedule_subject(spec, run_dir, "m1").schedules
        assert [x.plan.set_name for x in only.sessions] == ["two", "one", "three"]
        assert only.cost == 2.0

    def test_constraints_no_order_meets_are_refused(self, series):
        spec, run_dir, data = series
        data["schedule"] = {"within": {"one": 1, "two": 1}}
        spec = SeriesSpec.model_validate(data)
        _solve_all(spec, run_dir)
        with pytest.raises(ValueError, match="no order"):
            schedule_subject(spec, run_dir, "m1")

    def test_only_runs_solved_under_the_spec_are_pooled(self, series):
        spec, run_dir, data = series
        _solve_all(spec, run_dir)
        pools, left_out = subject_pools(spec, run_dir, "m1")
        assert left_out == {"m1 / one / q": RunState.PENDING}
        data["solve"] = {"topk": 7}
        pools, left_out = subject_pools(SeriesSpec.model_validate(data), run_dir, "m1")
        assert set(left_out.values()) == {RunState.STALE, RunState.PENDING}
        assert not any(pools.values())

    def test_a_set_with_no_plans_is_refused(self, series):
        spec, run_dir, _ = series
        _solved(spec, run_dir, "m1 / one / s", [(0.5, *ONE_SINGLE)])
        with pytest.raises(ValueError, match=r"no solved plans: \['two', 'three'\]"):
            schedule_subject(spec, run_dir, "m1")

    def test_a_plan_with_no_recorded_share_is_refused(self, series, tmp_path: Path):
        spec, run_dir, _ = series
        _solve_all(spec, run_dir)
        out = run_dir / "m1" / "solve" / "two" / "s"
        write_manifest(out, [PlanEntry("p.plan.yml", 1)], probes=[], config=None)
        with pytest.raises(ValueError, match="no recorded share"):
            schedule_subject(spec, run_dir, "m1")


class TestWritten:
    def test_the_json_the_report_and_a_copy_of_each_plan(self, series):
        spec, run_dir, _ = series
        _solve_all(spec, run_dir)
        out = run_dir / "m1"
        result = schedule_subject(spec, run_dir, "m1")
        report = write_schedule(result, spec.schedule, out)
        data = json.loads((out / "schedule.json").read_text())
        assert [s["floor"] for s in data["schedules"]] == [1.0, 0.9]
        assert data["sets"]["one"] == {
            "plans": 2,
            "best": 0.5,
            "allocations": {"s": {"plans": 2, "best": 0.5}},
        }
        assert data["left_out"] == {"m1 / one / q": "pending"}
        sessions = data["schedules"][1]["sessions"]
        assert [s["session"] for s in sessions] == [1, 2, 3]
        for s in sessions:
            assert (out / s["plan"]).read_bytes() == (out / s["copy"]).read_bytes()
            assert (out / s["config"]).is_file()
            assert s["copy"].startswith(f"schedule/floor-0.90/0{s['session']}-")
        text = report.read_text()
        assert "at least 90% of their set's best: rework 0" in text
        assert "1 probe slid" in text
        assert (
            f"rutter-plan {sessions[0]['config']} --plan {sessions[0]['copy']}" in text
        )

    def test_a_copy_someone_changed_is_kept(self, series):
        spec, run_dir, _ = series
        _solve_all(spec, run_dir)
        out = run_dir / "m1"
        result = schedule_subject(spec, run_dir, "m1")
        write_schedule(result, spec.schedule, out)
        copy = (
            out
            / json.loads((out / "schedule.json").read_text())["schedules"][0][
                "sessions"
            ][0]["copy"]
        )
        copy.write_text("adjusted\n")
        report = write_schedule(result, spec.schedule, out)
        assert copy.read_text() == "adjusted\n"
        assert json.loads((out / "schedule.json").read_text())["kept"] == [
            str(copy.relative_to(out))
        ]
        assert "Kept as found" in report.read_text()


class TestCommand:
    def _run(self, monkeypatch, capsys, *argv: str) -> tuple[int, str]:
        monkeypatch.setattr("sys.argv", ["rutter-series", *argv])
        with pytest.raises(SystemExit) as stop:
            cli.main()
        return stop.value.code, capsys.readouterr().out

    def test_status_then_schedule(self, series, tmp_path, monkeypatch, capsys):
        spec, run_dir, data = series
        _solve_all(spec, run_dir)
        path = tmp_path / "trio.series.yml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        code, said = self._run(monkeypatch, capsys, "status", str(path), str(run_dir))
        assert code == 0
        assert said.splitlines() == [
            "m1 / one / s: done, 2 plans",
            "m1 / one / q: pending",
            "m1 / two / s: done, 1 plans",
            "m1 / three / s: done, 1 plans",
        ]
        code, said = self._run(monkeypatch, capsys, "schedule", str(path), str(run_dir))
        assert code == 0 and "# Schedule: m1" in said
        assert (run_dir / "m1" / "schedule.md").is_file()

    def test_a_name_not_in_the_spec_is_refused(
        self, series, tmp_path, monkeypatch, capsys
    ):
        _, run_dir, data = series
        path = tmp_path / "trio.series.yml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        code, _ = self._run(
            monkeypatch, capsys, "schedule", str(path), str(run_dir), "--subject", "m9"
        )
        assert code == 2


@pytest.mark.parametrize(
    "stated, says",
    [
        ({"floors": [1.2]}, "at most 1"),
        ({"floors": [0.0]}, "above 0"),
        ({"floors": []}, "at least 1"),
        ({"floors": [0.9, 0.9]}, "repeat"),
        ({"first": ["a", "a"]}, "repeats a set"),
        ({"within": {"a": 0}}, "from 1"),
        ({"costs": {"mount": -1}}, "greater than or equal to 0"),
        ({"costs": {"unmount": 1}}, "Extra inputs"),
    ],
)
def test_the_schedule_section_refuses(stated: dict, says: str):
    with pytest.raises(ValueError, match=says):
        ScheduleSpec.model_validate(stated)


def test_schedule_sets_takes_plan_folders_directly(tmp_path: Path):
    """No run directory needed: any solved plan sets can be scheduled."""
    from aind_rutter.series.schedule import plan_candidates

    _write_plans(tmp_path / "a", [(0.5, *ONE_SINGLE)])
    _write_plans(tmp_path / "b", [(0.4, *ONE_QUAD), (0.39, *ONE_SINGLE)])
    pools = {n: plan_candidates(n, "x", tmp_path / n) for n in ("a", "b")}
    strict, loose = schedule_sets(pools, ScheduleSpec(floors=(1.0, 0.9)))
    assert (strict.cost, loose.cost) == (20.0, 0.0)
