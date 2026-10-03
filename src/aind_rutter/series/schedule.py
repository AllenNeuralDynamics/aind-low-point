"""Pick one solved plan per set, and the order of the sets, with the least rig rework.

A set's plans are pooled over every allocation solved for it and scored by the
mean recorded share, the one figure that compares across probe kinds and probe
counts. A plan that reaches less than ``min_region_mm3`` of any region, or
than ``min_region_fraction`` of a region too small to give that, is dropped
first: by volume so that a large region is not held to more tissue than a small
one, and by fraction so that a small one is not held to more than it has. For
each floor, a plan qualifies when it scores that share of its
set's best. Among the qualifying plans the cheapest choice and order is found
exactly; of equally cheap ones, the one that records most. Building the first
session's rig is not charged.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from aind_rutter.config import PlanningModel
from aind_rutter.domain.plan import PlanningState, ProbePlan
from aind_rutter.domain.rig import Kinematics
from aind_rutter.domain.rig_setup import (
    RigSetup,
    SetupChanges,
    SetupCosts,
    rig_setup,
    setup_changes,
)
from aind_rutter.plan_io.plan_set import PlanEntry, load_plan_set
from aind_rutter.series.solve import RunState, run_state, solve_runs
from aind_rutter.series.spec import ScheduleSpec, SeriesSpec

JSON_NAME = "schedule.json"
REPORT_NAME = "schedule.md"
PLANS_DIR = "schedule"


@dataclass(frozen=True)
class Candidate:
    """One solved plan of a set."""

    set_name: str
    allocation: str
    entry: PlanEntry
    path: Path
    score: float  # mean recorded share
    config: Path | None = None
    # Region -> volume of all its voxels; see PlanSet.region_mm3.
    region_mm3: dict[str, float] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class Session:
    plan: Candidate
    of_best: float  # its score over its set's best
    setup: RigSetup
    changes: SetupChanges | None  # from the session before; None for the first
    cost: float


@dataclass(frozen=True)
class Schedule:
    floor: float
    cost: float
    sessions: tuple[Session, ...]


@dataclass(frozen=True)
class SubjectSchedule:
    subject: str
    pools: dict[str, list[Candidate]]
    left_out: dict[str, RunState]  # run label -> why its plans are not pooled
    schedules: tuple[Schedule, ...]


def plan_setup(path: Path) -> RigSetup:
    """What a plan file puts on the rig."""
    plan = PlanningModel(**yaml.safe_load(path.read_text()))
    state = PlanningState(
        kinematics=Kinematics(arc_angles=dict(plan.arcs)),
        probes={
            name: ProbePlan(
                kind=str(decl.kind),
                arc_id=decl.arc,
                ml_local=decl.slider_ml,
                spin=decl.spin,
            )
            for name, decl in plan.probes.items()
        },
    )
    return rig_setup(state)


def plan_candidates(
    set_name: str, allocation: str, out_dir: Path, config: Path | None = None
) -> list[Candidate]:
    """The plans of one solve directory, scored."""
    plan_set = load_plan_set(out_dir)
    out = []
    for entry in plan_set.entries:
        score = entry.mean_recorded()
        if score is None:
            raise ValueError(
                f"{plan_set.path(entry)} carries no recorded share, which is what "
                f"a schedule compares; emit the plans again"
            )
        out.append(
            Candidate(
                set_name,
                allocation,
                entry,
                plan_set.path(entry),
                score,
                config,
                plan_set.region_mm3,
            )
        )
    return out


def cheapest(
    cost: Mapping[tuple[int, int], NDArray[np.float64]],
    score: Sequence[NDArray[np.float64]],
    first: frozenset[int] = frozenset(),
    latest: Mapping[int, int] | None = None,
) -> tuple[float, list[tuple[int, int]]] | None:
    """The cheapest order of the sets and plan of each, as ``(set, plan)`` pairs.

    ``cost[a, b]`` prices going from each plan of set ``a`` to each of set
    ``b``; a plan that may not be chosen scores ``-inf``. ``first`` holds the
    sets that take the first sessions and ``latest[s]`` the last session set
    ``s`` may be, counted from 1. ``None`` when no order satisfies both.

    Exact, by the best way to have run each subset of the sets ending on each
    plan, which grows with ``2**n`` and not with ``n!``.
    """
    n = len(score)
    latest = latest or {}

    def allowed(s: int, position: int) -> bool:
        return (s in first) == (position <= len(first)) and position <= latest.get(s, n)

    # (sets run, last set) -> per plan of the last set: cost, score, and the
    # set and plan before it.
    best: dict[tuple[int, int], tuple[NDArray, NDArray, NDArray, NDArray]] = {}
    for s in range(n):
        if allowed(s, 1):
            none = np.full(len(score[s]), -1)
            start = np.where(np.isfinite(score[s]), 0.0, np.inf)
            best[1 << s, s] = (start, score[s], none, none)
    for done in range(1, 1 << n):
        position = done.bit_count() + 1
        for s in range(n):
            if (done, s) not in best:
                continue
            c_s, v_s, _, _ = best[done, s]
            for t in range(n):
                if done >> t & 1 or not allowed(t, position):
                    continue
                through = c_s[:, None] + cost[s, t]
                c = through.min(axis=0)
                tied = np.where(through == c, v_s[:, None], -np.inf)
                via = tied.argmax(axis=0)
                v = tied.max(axis=0) + score[t]
                c = np.where(np.isfinite(score[t]), c, np.inf)
                key = (done | 1 << t, t)
                if key in best:
                    c0, v0, s0, p0 = best[key]
                    better = (c < c0) | ((c == c0) & (v > v0))
                    best[key] = (
                        np.where(better, c, c0),
                        np.where(better, v, v0),
                        np.where(better, s, s0),
                        np.where(better, via, p0),
                    )
                else:
                    best[key] = (c, v, np.full(len(c), s), via)

    everything = (1 << n) - 1
    ends = [
        (float(c[p]), -float(v[p]), s, p)
        for s in range(n)
        if (everything, s) in best
        for c, v in [best[everything, s][:2]]
        for p in range(len(c))
        if np.isfinite(c[p])
    ]
    if not ends:
        return None
    total, _, s, p = min(ends)
    picks, done = [(s, p)], everything
    while done.bit_count() > 1:
        _, _, before_set, before_plan = best[done, s]
        done ^= 1 << s
        s, p = int(before_set[p]), int(before_plan[p])
        picks.append((s, p))
    return total, picks[::-1]


def schedule_sets(
    pools: Mapping[str, Sequence[Candidate]],
    schedule: ScheduleSpec = ScheduleSpec(),
) -> list[Schedule]:
    """One schedule per floor over the given sets' plans."""
    empty = [name for name, plans in pools.items() if not plans]
    if empty:
        raise ValueError(f"sets with no solved plans: {empty}")
    if schedule.min_region_mm3 > 0:
        pools = {
            name: _reaching(
                name, plans, schedule.min_region_mm3, schedule.min_region_fraction
            )
            for name, plans in pools.items()
        }
    costs = schedule.costs.costs()
    names = list(pools)
    top = {name: max(c.score for c in pools[name]) for name in names}
    # Only plans some floor admits are ever chosen, so only they are priced.
    plans = [
        [c for c in pools[name] if c.score >= min(schedule.floors) * top[name]]
        for name in names
    ]
    setups = [[plan_setup(c.path) for c in group] for group in plans]
    cost = _transition_costs(setups, costs)
    first = frozenset(names.index(s) for s in schedule.first if s in pools)
    latest = {names.index(s): n for s, n in schedule.within.items() if s in pools}

    out = []
    for floor in schedule.floors:
        score = [
            np.array(
                [c.score if c.score >= floor * top[name] else -np.inf for c in group]
            )
            for name, group in zip(names, plans)
        ]
        found = cheapest(cost, score, first, latest)
        if found is None:
            raise ValueError(
                f"no order of {names} satisfies first {list(schedule.first)} and "
                f"within {dict(schedule.within)}"
            )
        total, picks = found
        sessions: list[Session] = []
        for s, p in picks:
            changes = (
                setup_changes(sessions[-1].setup, setups[s][p], costs)
                if sessions
                else None
            )
            sessions.append(
                Session(
                    plans[s][p],
                    plans[s][p].score / top[names[s]] if top[names[s]] else 1.0,
                    setups[s][p],
                    changes,
                    0.0 if changes is None else costs.of(changes),
                )
            )
        out.append(Schedule(floor, total, tuple(sessions)))
    return out


def _reaching(
    name: str, plans: Sequence[Candidate], min_mm3: float, fraction: float
) -> list[Candidate]:
    unmeasured = [
        c for c in plans if not c.entry.recorded_mm3 or (fraction and not c.region_mm3)
    ]
    if unmeasured:
        raise ValueError(
            f"{unmeasured[0].path} carries no recorded volume, which a region "
            "minimum needs; emit the plans again"
        )

    def needed(c: Candidate, region: str) -> float:
        if not fraction:
            return min_mm3
        return min(min_mm3, fraction * c.region_mm3[region])

    def short(c: Candidate) -> dict[str, float]:
        """Per region, how far the plan falls below what it must reach."""
        gaps = {r: needed(c, r) - v for r, v in c.entry.recorded_mm3.items()}
        return {r: gap for r, gap in gaps.items() if gap > 0}

    kept = [c for c in plans if not short(c)]
    if not kept:
        closest = min(plans, key=lambda c: max(short(c).values()))
        gaps = short(closest)
        region = max(gaps, key=gaps.__getitem__)
        raise ValueError(
            f"no plan of {name} reaches every region's minimum; the closest, "
            f"{closest.path.name}, reaches "
            f"{closest.entry.recorded_mm3[region]:.3f} mm³ of {region}, "
            f"{gaps[region]:.3f} mm³ short"
        )
    return kept


def region_standing(
    plan: Candidate, pool: Sequence[Candidate]
) -> dict[str, tuple[float, float]]:
    """Per region, the volume ``plan`` reaches and the share of ``pool``'s plans
    that reach less, which says whether a low figure is the plan or the region."""
    out = {}
    for region, mm3 in plan.entry.recorded_mm3.items():
        others = [c.entry.recorded_mm3.get(region, 0.0) for c in pool]
        out[region] = (mm3, sum(v < mm3 for v in others) / len(others))
    return out


def _transition_costs(
    setups: Sequence[Sequence[RigSetup]], costs: SetupCosts
) -> dict[tuple[int, int], NDArray[np.float64]]:
    out = {}
    for a in range(len(setups)):
        for b in range(a + 1, len(setups)):
            there = np.array(
                [
                    [costs.of(setup_changes(x, y, costs)) for y in setups[b]]
                    for x in setups[a]
                ]
            ).reshape(len(setups[a]), len(setups[b]))
            # Taking a probe or an arc off costs what putting it on does, so
            # the way back costs the same.
            out[a, b], out[b, a] = there, there.T
    return out


def subject_pools(
    spec: SeriesSpec, run_dir: str | Path, subject: str
) -> tuple[dict[str, list[Candidate]], dict[str, RunState]]:
    """Each of the subject's sets' plans over its solved allocations.

    Only a run solved under what the spec now states is pooled; the rest are
    returned with their state.
    """
    pools: dict[str, list[Candidate]] = {s: [] for s in spec.sets_of(subject)}
    left_out: dict[str, RunState] = {}
    for run in solve_runs(spec, run_dir):
        if run.subject != subject:
            continue
        state = run_state(spec, run)
        if state is not RunState.DONE:
            left_out[run.label] = state
        elif (run.out_dir / "manifest.json").is_file():
            found = plan_candidates(
                run.set_name, run.allocation, run.out_dir, run.config
            )
            regions = set(spec.sets[run.set_name].regions)
            partial = [c for c in found if set(c.entry.recorded) != regions]
            if partial:
                raise ValueError(
                    f"{run.label}: {partial[0].path} records "
                    f"{sorted(partial[0].entry.recorded)}, not the set's "
                    f"{sorted(regions)}, so its mean is not the set's"
                )
            pools[run.set_name] += found
    return pools, left_out


def schedule_subject(
    spec: SeriesSpec, run_dir: str | Path, subject: str
) -> SubjectSchedule:
    pools, left_out = subject_pools(spec, run_dir, subject)
    return SubjectSchedule(
        subject, pools, left_out, tuple(schedule_sets(pools, spec.schedule))
    )


def _least_text(entry: PlanEntry) -> str:
    volumes = entry.recorded_mm3
    if not volumes:
        return ""
    region = min(volumes, key=volumes.__getitem__)
    return f"{region} {volumes[region]:.2f} mm³"


def _setup_text(setup: RigSetup) -> str:
    return "; ".join(
        f"{arc.ap_deg:+.1f}°: " + ", ".join(slot.kind for slot in arc.slots)
        for arc in setup
    )


def _changes_text(changes: SetupChanges | None) -> str:
    if changes is None:
        return ""
    names = {
        "arc_moves": "arc{s} slid",
        "slider_moves": "probe{s} slid",
        "spins": "probe{s} turned",
        "mounts": "probe{s} on or off",
        "arc_mounts": "arc{s} on or off",
    }
    parts = [
        f"{n} {names[k].format(s='s' if n > 1 else '')}"
        for k, n in asdict(changes).items()
        if n
    ]
    return ", ".join(parts) or "none"


def _floor_dir(floor: float) -> str:
    return f"floor-{floor:.2f}"


def _session_file(position: int, session: Session) -> str:
    plan = session.plan
    return (
        f"{position:02d}-{plan.set_name}-{plan.allocation}"
        f"-plan-{plan.entry.rank:02d}.plan.yml"
    )


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def schedule_json(
    result: SubjectSchedule, schedule: ScheduleSpec, root: Path
) -> dict[str, Any]:
    """The schedules as plain data; paths under ``root`` are relative to it."""
    sets = {}
    for name, plans in result.pools.items():
        by_allocation: dict[str, list[float]] = {}
        for c in plans:
            by_allocation.setdefault(c.allocation, []).append(c.score)
        sets[name] = {
            "plans": len(plans),
            "best": max(c.score for c in plans),
            "allocations": {
                a: {"plans": len(scores), "best": max(scores)}
                for a, scores in by_allocation.items()
            },
        }
    return {
        "subject": result.subject,
        "schedule": schedule.model_dump(mode="json"),
        "sets": sets,
        "left_out": {label: str(state) for label, state in result.left_out.items()},
        "schedules": [
            {
                "floor": s.floor,
                "cost": s.cost,
                "sessions": [
                    {
                        "session": position,
                        "set": x.plan.set_name,
                        "allocation": x.plan.allocation,
                        "rank": x.plan.entry.rank,
                        "plan": _relative(x.plan.path, root),
                        "config": _relative(x.plan.config, root),
                        "copy": f"{PLANS_DIR}/{_floor_dir(s.floor)}/"
                        f"{_session_file(position, x)}",
                        "recorded": x.plan.score,
                        "regions": {
                            region: {"mm3": mm3, "beats": beats}
                            for region, (mm3, beats) in region_standing(
                                x.plan, result.pools[x.plan.set_name]
                            ).items()
                        },
                        "of_best": x.of_best,
                        "setup": [asdict(arc) for arc in x.setup],
                        "changes": None if x.changes is None else asdict(x.changes),
                        "cost": x.cost,
                    }
                    for position, x in enumerate(s.sessions, 1)
                ],
            }
            for s in result.schedules
        ],
    }


def schedule_report(result: SubjectSchedule, schedule: ScheduleSpec, root: Path) -> str:
    """The schedules as a Markdown page."""
    costs = schedule.costs.model_dump()
    lines = [
        f"# Schedule: {result.subject}",
        "",
        "Rework prices: "
        + ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in costs.items())
        + ". The first session's setup is not charged.",
    ]
    if schedule.first:
        lines.append(f"First, in either order: {', '.join(schedule.first)}.")
    for name, last in schedule.within.items():
        lines.append(f"{name} is among the first {last} sessions.")
    if schedule.min_region_mm3 > 0:
        lines.append(
            f"Every plan reaches at least {schedule.min_region_mm3:g} mm³ of "
            "every region"
            + (
                f", or {schedule.min_region_fraction:.0%} of one too small to give it."
                if schedule.min_region_fraction
                else "."
            )
        )
    lines += [
        "",
        "| Set | Allocation | Plans | Best recorded share |",
        "|---|---|---:|---:|",
    ]
    data = schedule_json(result, schedule, root)
    for name, one in data["sets"].items():
        for allocation, a in one["allocations"].items():
            lines.append(f"| {name} | {allocation} | {a['plans']} | {a['best']:.3f} |")
    if result.left_out:
        lines += ["", "Not pooled:", ""]
        lines += [f"- {label}: {state}" for label, state in result.left_out.items()]
    for s in result.schedules:
        lines += [
            "",
            f"## Plans recording at least {s.floor:.0%} of their set's best: "
            f"rework {s.cost:g}",
            "",
            "| Session | Set | Allocation | Plan | Recorded | Of best | Least region "
            "| Rework | Changes | Rig (arc AP: probes by ML) |",
            "|---:|---|---|---:|---:|---:|---|---:|---|---|",
        ]
        for position, x in enumerate(s.sessions, 1):
            lines.append(
                f"| {position} | {x.plan.set_name} | {x.plan.allocation} "
                f"| {x.plan.entry.rank} | {x.plan.score:.3f} | {x.of_best:.0%} "
                f"| {_least_text(x.plan.entry)} "
                f"| {'' if x.changes is None else format(x.cost, 'g')} "
                f"| {_changes_text(x.changes)} | {_setup_text(x.setup)} |"
            )
        lines += [
            "",
            "Volume of each region reached, and the share of the set's plans that "
            "reach less:",
            "",
        ]
        for position, x in enumerate(s.sessions, 1):
            standing = region_standing(x.plan, result.pools[x.plan.set_name])
            if not standing:
                continue
            lines.append(
                f"{position}. {x.plan.set_name}: "
                + ", ".join(
                    f"{region} {mm3:.2f} mm³ ({beats:.0%})"
                    for region, (mm3, beats) in standing.items()
                )
            )
        lines += ["", "```text"]
        for position, x in enumerate(s.sessions, 1):
            config = _relative(x.plan.config, root) or "<config>"
            copy = f"{PLANS_DIR}/{_floor_dir(s.floor)}/{_session_file(position, x)}"
            lines.append(f"rutter-plan {config} --plan {copy}")
        lines.append("```")
    return "\n".join(lines) + "\n"


def write_schedule(
    result: SubjectSchedule, schedule: ScheduleSpec, out_dir: str | Path
) -> Path:
    """Write the JSON, the report and a copy of each chosen plan, by session.

    The copies are what the report's commands open, so a plan adjusted in the
    planner is saved beside the schedule and the solved one is left alone. A
    copy that already differs from its solved plan is therefore kept, and the
    report names it.
    """
    out_dir = Path(out_dir).resolve()
    kept = []
    for s in result.schedules:
        folder = out_dir / PLANS_DIR / _floor_dir(s.floor)
        folder.mkdir(parents=True, exist_ok=True)
        for position, x in enumerate(s.sessions, 1):
            copy = folder / _session_file(position, x)
            if copy.is_file() and copy.read_bytes() != x.plan.path.read_bytes():
                kept.append(str(copy.relative_to(out_dir)))
            else:
                shutil.copyfile(x.plan.path, copy)
    data = schedule_json(result, schedule, out_dir) | {"kept": kept}
    (out_dir / JSON_NAME).write_text(json.dumps(data, indent=2) + "\n")
    text = schedule_report(result, schedule, out_dir)
    if kept:
        text += "\nKept as found, since they differ from the solved plan:\n\n"
        text += "".join(f"- {name}\n" for name in kept)
    report = out_dir / REPORT_NAME
    report.write_text(text)
    return report
