"""Solve every config of a series, one pipeline run each, resumably.

The pipeline wrapper owns the per-arc-count processes and the MPS daemon, so a
run is one call to it with the environment the spec implies. Each run's
directory records what it was asked for, which is what makes a rerun skip the
runs already done and redo the ones whose inputs changed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from aind_rutter.series.configs import config_path
from aind_rutter.series.lock import file_sha256
from aind_rutter.series.spec import Score, SeriesSpec

RECORD_NAME = "solve.json"
# Set by a shell for one run's outputs; inherited, they would point every run of
# a series at the same files.
PER_RUN_VARIABLES = (
    "TAG",
    "FRESH",
    "RUTTER_ATLAS_CACHE",
    "RUTTER_SEED_CACHE",
    "RUTTER_OUT",
    "RUTTER_POSES",
    "RUTTER_HANDOFF",
    "RUTTER_OUTDIR",
    "RUTTER_ONLY_NARCS",
    "RUTTER_RANKS",
    "RUTTER_RANKS_FILE",
)


def default_script() -> Path:
    """The pipeline wrapper of the checkout this package runs from."""
    script = Path(__file__).resolve().parents[3] / "scripts/run_subject_overnight.sh"
    if not script.is_file():
        raise FileNotFoundError(
            f"{script} is not there: the package is not running from a checkout, "
            f"so say where the pipeline wrapper is"
        )
    return script


@dataclass(frozen=True)
class SolveRun:
    """One config to solve and where its results go."""

    subject: str
    set_name: str
    allocation: str
    config: Path
    out_dir: Path

    @property
    def label(self) -> str:
        return f"{self.subject} / {self.set_name} / {self.allocation}"


@dataclass(frozen=True)
class SolveOutcome:
    run: SolveRun
    skipped: bool  # already done for this request
    ok: bool
    max_arcs: int | None  # the cap of the last attempt
    n_plans: int


def solve_dir(run_dir: Path, subject: str, set_name: str, allocation: str) -> Path:
    return run_dir / subject / "solve" / set_name / allocation


def solve_runs(spec: SeriesSpec, run_dir: str | Path) -> list[SolveRun]:
    """Every run of the series, in the order the spec states them."""
    run_dir = Path(run_dir).resolve()
    return [
        SolveRun(
            subject,
            set_name,
            allocation,
            config_path(run_dir, subject, set_name, allocation),
            solve_dir(run_dir, subject, set_name, allocation),
        )
        for subject in spec.subjects
        for set_name in spec.sets_of(subject)
        for allocation in spec.sets[set_name].allocations
    ]


def request(spec: SeriesSpec, run: SolveRun) -> dict[str, Any]:
    """What decides the run's result; a record made under another is stale."""
    holes = spec.subjects[run.subject].holes
    return {
        "config_sha256": file_sha256(run.config),
        "holes_sha256": file_sha256(holes),
        "score": spec.sets[run.set_name].score.value,
        "solve": spec.solve_for(run.set_name).model_dump(mode="json"),
    }


def run_environment(spec: SeriesSpec, run: SolveRun, max_arcs: int) -> dict[str, str]:
    """The variables the spec decides for one attempt."""
    solve = spec.solve_for(run.set_name)
    env = {
        "RUTTER_CONFIG": str(run.config),
        "RUTTER_HOLES": str(spec.subjects[run.subject].holes.resolve()),
        "RUTTER_RUN_DIR": str(run.out_dir),
        "RUTTER_MAX_ARCS": str(max_arcs),
        "RUTTER_TOPK": str(solve.topk),
        "RUTTER_SELECT_BY": solve.select_by,
        "RUTTER_RETRO_DENSITY": str(int(spec.sets[run.set_name].score is Score.RETRO)),
        "RUTTER_RECORDED_RADIUS_MM": str(solve.recorded_radius_mm),
    }
    if solve.ranker_table is not None:
        env["RUTTER_RANKER_TABLE"] = str(solve.ranker_table.resolve())
    return env


def read_record(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / RECORD_NAME
    return json.loads(path.read_text()) if path.is_file() else None


def n_plans(out_dir: Path) -> int:
    manifest = out_dir / "manifest.json"
    if not manifest.is_file():
        return 0
    return len(json.loads(manifest.read_text()).get("plans", []))


def _finished(record: dict[str, Any], caps: tuple[int, ...]) -> bool:
    """Whether the record's attempts leave nothing to run."""
    attempts = record.get("attempts", [])
    if not attempts or attempts[-1]["returncode"] != 0:
        return False
    return attempts[-1]["n_plans"] > 0 or attempts[-1]["max_arcs"] == caps[-1]


class RunState(StrEnum):
    NO_CONFIG = "no config"
    PENDING = "pending"  # not run, or arc caps left to try
    STALE = "stale"  # solved under inputs the spec no longer states
    FAILED = "failed"
    DONE = "done"


def run_state(spec: SeriesSpec, run: SolveRun) -> RunState:
    """Where a run stands against what the spec now asks of it."""
    if not run.config.is_file():
        return RunState.NO_CONFIG
    record = read_record(run.out_dir)
    if record is None:
        return RunState.PENDING
    if record.get("request") != request(spec, run):
        return RunState.STALE
    if _finished(record, spec.solve_for(run.set_name).max_arcs):
        return RunState.DONE
    attempts = record.get("attempts", [])
    failed = bool(attempts) and attempts[-1]["returncode"] != 0
    return RunState.FAILED if failed else RunState.PENDING


def _attempt(
    script: Path, env: Mapping[str, str], out_dir: Path, log: Path
) -> subprocess.CompletedProcess[bytes]:
    # Emit adds to a plans folder and never empties it.
    shutil.rmtree(out_dir / "plans", ignore_errors=True)
    (out_dir / "manifest.json").unlink(missing_ok=True)
    with log.open("wb") as out:
        return subprocess.run(
            [str(script)], env=dict(env), stdout=out, stderr=subprocess.STDOUT
        )


def solve(
    spec: SeriesSpec,
    run_dir: str | Path,
    *,
    script: Path | None = None,
    environ: Mapping[str, str] | None = None,
    select: Callable[[SolveRun], bool] | None = None,
    report: Callable[[str], None] = print,
) -> list[SolveOutcome]:
    """Run the pipeline on each config that is not already solved.

    A run whose record matches what the spec now asks is skipped; one made
    under different inputs is redone from nothing. A failed attempt ends that
    run and the series moves on, so one bad config does not cost the night.
    ``environ`` is the machine's side of the environment (workers, memory,
    chunk sizes), by default this process's.
    """
    script = default_script() if script is None else Path(script).resolve()
    inherited = {
        k: v
        for k, v in (os.environ if environ is None else environ).items()
        if k not in PER_RUN_VARIABLES
    }
    outcomes = []
    for run in solve_runs(spec, run_dir):
        if select is not None and not select(run):
            continue
        if not run.config.is_file():
            raise FileNotFoundError(f"{run.label}: {run.config} is not written yet")
        caps = spec.solve_for(run.set_name).max_arcs
        asked = request(spec, run)
        record = read_record(run.out_dir)
        stale = record is not None and record.get("request") != asked
        if record is None or stale:
            record = {"request": asked, "attempts": []}
        elif _finished(record, caps):
            last = record["attempts"][-1]
            report(f"{run.label}: done, {last['n_plans']} plans")
            outcomes.append(
                SolveOutcome(run, True, True, last["max_arcs"], last["n_plans"])
            )
            continue
        run.out_dir.mkdir(parents=True, exist_ok=True)
        ok, cap, plans = True, None, 0
        # A failed attempt is repeated; one that ran and found nothing is not.
        tried = {a["max_arcs"] for a in record["attempts"] if a["returncode"] == 0}
        for cap in (c for c in caps if c not in tried):
            env = inherited | run_environment(spec, run, cap)
            if stale:
                env["FRESH"] = "1"
                stale = False
            record["environment"] = {
                k: v for k, v in sorted(env.items()) if k.startswith("RUTTER_")
            }
            _write_record(run.out_dir, record)
            report(f"{run.label}: start (arcs<={cap})")
            began = time.monotonic()
            done = _attempt(
                script, env, run.out_dir, run.out_dir / f"solve.arcs{cap}.log"
            )
            plans = n_plans(run.out_dir) if done.returncode == 0 else 0
            record["attempts"].append(
                {
                    "max_arcs": cap,
                    "returncode": done.returncode,
                    "n_plans": plans,
                    "seconds": round(time.monotonic() - began, 1),
                }
            )
            _write_record(run.out_dir, record)
            ok = done.returncode == 0
            report(
                f"{run.label}: {'done' if ok else 'FAILED'} (arcs<={cap}): "
                f"{plans} plans"
            )
            if not ok or plans:
                break
        outcomes.append(SolveOutcome(run, False, ok, cap, plans))
    return outcomes


def _write_record(out_dir: Path, record: dict[str, Any]) -> None:
    (out_dir / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")
