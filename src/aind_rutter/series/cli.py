"""The ``rutter-series`` command: a series spec through to its schedule.

rutter-series propose  dhc.series.yml runs/dhc [--subject M] [--set S]
rutter-series configs  dhc.series.yml runs/dhc
rutter-series solve    dhc.series.yml runs/dhc [--subject M] [--set S]
rutter-series schedule dhc.series.yml runs/dhc [--subject M]
rutter-series status   dhc.series.yml runs/dhc
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from aind_rutter.series.configs import write_configs
from aind_rutter.series.schedule import schedule_subject, write_schedule
from aind_rutter.series.solve import (
    SolveRun,
    n_plans,
    run_state,
    solve,
    solve_runs,
)
from aind_rutter.series.spec import SeriesSpec, load_spec


def _configs(spec: SeriesSpec, args: argparse.Namespace) -> int:
    written = write_configs(spec, args.run, spec_path=args.spec)
    print(f"{len(written)} configs under {args.run}")
    return 0


def _propose(spec: SeriesSpec, args: argparse.Namespace) -> int:
    # The proposer needs no card, and importing the pipeline would claim one.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from aind_rutter.series.propose import write_proposals

    write_proposals(spec, args.run, subjects=args.subject, sets=args.set)
    return 0


def _solve(spec: SeriesSpec, args: argparse.Namespace) -> int:
    def wanted(run: SolveRun) -> bool:
        return (not args.subject or run.subject in args.subject) and (
            not args.set or run.set_name in args.set
        )

    outcomes = solve(spec, args.run, script=args.script, select=wanted)
    failed = [o.run.label for o in outcomes if not o.ok]
    for label in failed:
        print(f"FAILED {label}", file=sys.stderr)
    return 1 if failed else 0


def _schedule(spec: SeriesSpec, args: argparse.Namespace) -> int:
    for subject in args.subject or spec.subjects:
        result = schedule_subject(spec, args.run, subject)
        report = write_schedule(result, spec.schedule, args.run / subject)
        print(report.read_text())
        print(f"written to {report}")
    return 0


def _status(spec: SeriesSpec, args: argparse.Namespace) -> int:
    for run in solve_runs(spec, args.run):
        state = run_state(spec, run)
        plans = f", {n_plans(run.out_dir)} plans" if state == "done" else ""
        print(f"{run.label}: {state}{plans}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rutter-series",
        description="Plan a series of insertion sessions from one spec.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, run, text in (
        ("propose", _propose, "propose allocations for each set, to paste in"),
        ("configs", _configs, "write one config per subject, set and allocation"),
        ("solve", _solve, "solve every config not already solved"),
        ("schedule", _schedule, "pick a plan per set and the order of the sets"),
        ("status", _status, "where each run stands"),
    ):
        one = commands.add_parser(name, help=text, description=text)
        one.add_argument("spec", type=Path, help="the series spec")
        one.add_argument("run", type=Path, help="the run directory")
        one.set_defaults(run_command=run)
        if name in ("propose", "solve", "schedule"):
            one.add_argument(
                "--subject", action="append", help="only this subject; repeatable"
            )
        if name in ("propose", "solve"):
            one.add_argument("--set", action="append", help="only this set; repeatable")
        if name == "solve":
            one.add_argument(
                "--script",
                type=Path,
                help="the pipeline wrapper (default: the checkout's "
                "scripts/run_subject_overnight.sh)",
            )
    args = parser.parse_args()
    spec = load_spec(args.spec)
    unknown = sorted(set(getattr(args, "subject", None) or ()) - set(spec.subjects))
    unknown += sorted(set(getattr(args, "set", None) or ()) - set(spec.sets))
    if unknown:
        parser.error(f"not in the spec: {unknown}")
    sys.exit(args.run_command(spec, args))
