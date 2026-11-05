"""Choosing which Phase-1 candidates Phase 2 solves.

Kept free of jax so it can be imported and tested without initialising a GPU
backend: the stage module sets JAX platform and memory options at import, which a
unit test has no business triggering.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from aind_rutter.optimization.pipeline.payloads import pinned_spec_differences
from aind_rutter.optimization.pipeline.records import (
    Phase1PoolPayload,
    Phase1PoolRecord,
    Phase2HandoffPayload,
    Phase2InputRecord,
    Phase2ResultRecord,
)
from aind_rutter.optimization.pipeline.settings import Phase2Settings
from aind_rutter.plan_io.plan_set import MANIFEST_NAME, load_plan_set

# Sentinels for records missing the ranking field, chosen so they sort last in
# whichever direction the field is ranked.
_WORST_ASCENDING = 1e18
_WORST_DESCENDING = -1e9


def check_pool_specs(pool: Phase1PoolPayload, settings: Phase2Settings) -> str | None:
    """Refuse a pool built under a feasibility spec or objective other than
    ``settings``'s; return a warning for one that records neither.

    Raises
    ------
    ValueError
        If the pool records a spec that differs.
    """
    differences = pinned_spec_differences(pool, settings.pinned_specs())
    if differences is None:
        return f"{settings.poses} records no specs; its Phase 1 is taken on trust"
    if differences:
        raise ValueError(
            f"{settings.poses} was built under other specs: " + "; ".join(differences)
        )
    return None


def rank_order(all_recs: list[Phase1PoolRecord], select_by: str) -> list[int]:
    """Indices of ``all_recs``, best first, by the named Phase-1 field.

    ``objective`` is Phase 1's, lower-is-better, so it sorts ascending; the
    clearance-style metrics such as ``min_clear`` are higher-is-better and sort
    descending.
    """
    ascending = select_by == "objective"
    worst = _WORST_ASCENDING if ascending else _WORST_DESCENDING
    return sorted(
        range(len(all_recs)),
        key=lambda i: float(cast(Any, all_recs[i]).get(select_by, worst)),
        reverse=not ascending,
    )


def normalize_record(
    rec: Phase1PoolRecord, source_index: int, rank: int
) -> Phase2InputRecord:
    """A Phase-1 pool record as Phase 2 consumes it.

    The pose is whichever of ``pose`` or ``x`` the producing stage wrote, and the
    arc assignment is carried through verbatim so the geometry matches what the
    pose was optimized against.
    """
    return {
        "idx": rec.get("idx", source_index),
        "n_arcs": rec["n_arcs"],
        "pose": cast(Any, rec).get("pose", rec["x"]),
        "probe_to_hole": rec["probe_to_hole"],
        "partition": rec["partition"],
        "probe_to_arc_idx": rec["probe_to_arc_idx"],
        "arc_centroids_deg": rec["arc_centroids_deg"],
        "min_clear": rec.get("min_clear"),
        "objective": rec.get("objective"),
        "rank": rank,
    }


def select_records(
    all_recs: list[Phase1PoolRecord],
    settings: Phase2Settings,
    log: Callable[[str], None] | None = None,
) -> list[Phase2InputRecord]:
    """The candidates Phase 2 will solve, in rank order.

    Nothing is culled by FCL between the phases — the pool is ranked and the best
    ``topk`` handed on, with FCL applied once at the end as the ground-truth gate.
    ``ranks`` or ``ranks_file`` replace that cut with explicit offsets into the
    ranked order, which is how a run probes where good candidates stop appearing
    instead of guessing a cutoff; offsets past the end are dropped. ``select_by``
    'learned' measures the whole pool first, which loads jax, unless the cut
    falls inside the objective's head.
    """
    ids = candidate_ids(settings)
    if ids is not None:
        by_idx = {rec.get("idx", i): (i, rec) for i, rec in enumerate(all_recs)}
        _check_known(ids, by_idx, str(settings.poses))
        return [
            normalize_record(by_idx[c][1], by_idx[c][0], rank)
            for rank, c in enumerate(ids)
        ]
    explicit = bool(settings.ranks or settings.ranks_file)
    head_only = not explicit and settings.topk <= settings.rank_head
    if settings.select_by == "learned" and not head_only:
        if settings.ranker_table is None:
            raise ValueError(
                f"select_by 'learned' past the objective's first {settings.rank_head} "
                "needs RUTTER_RANKER_TABLE, a solved pool's training table; or set "
                "RUTTER_SELECT_BY=objective"
            )
        from aind_rutter.optimization.pipeline.learned_rank import learned_order

        order = learned_order(
            all_recs, settings, rank_order(all_recs, "objective"), log=log
        )
    elif settings.select_by == "learned":
        order = rank_order(all_recs, "objective")
    else:
        order = rank_order(all_recs, settings.select_by)
    if explicit:
        from aind_rutter.optimization.pipeline.phase2_diagnostics import read_ranks

        chosen = (
            read_ranks(str(settings.ranks_file))
            if settings.ranks_file
            else [int(x) for x in settings.ranks.split(",") if x.strip()]
        )
        chosen = [r for r in chosen if r < len(order)]
    else:
        chosen = list(range(min(settings.topk, len(order))))
    return [normalize_record(all_recs[order[r]], order[r], r) for r in chosen]


def candidate_ids(settings: Phase2Settings) -> list[int] | None:
    """The candidate ids ``settings`` name, in solve order, or None if none.

    A plan set's manifest gives its first ``topk`` plans' candidates, ordered as
    listed or by mean recorded share; any other file is a list of ids.
    """
    if settings.candidates:
        return [int(x) for x in settings.candidates.split(",") if x.strip()]
    path = settings.candidates_file
    if path is None:
        return None
    if Path(path).name != MANIFEST_NAME:
        from aind_rutter.optimization.pipeline.phase2_diagnostics import read_ranks

        return read_ranks(str(path))
    entries = list(load_plan_set(Path(path).parent).entries)
    if settings.candidates_by == "recorded":
        unmeasured = [e.file for e in entries if e.mean_recorded() is None]
        if unmeasured:
            raise ValueError(f"{path} records no share for {unmeasured[0]}")
        entries.sort(key=lambda e: -cast(float, e.mean_recorded()))
    entries = entries[: settings.topk]
    unknown = [e.file for e in entries if e.candidate is None]
    if unknown:
        raise ValueError(f"{path} records no candidate for {unknown[0]}")
    return [cast(int, e.candidate) for e in entries]


def _check_known(ids: list[int], known: Mapping[int, Any], source: str) -> None:
    missing = [c for c in ids if c not in known]
    if missing:
        raise ValueError(f"{source} has no candidates {missing}")
    repeated = sorted({c for c in ids if ids.count(c) > 1})
    if repeated:
        raise ValueError(f"candidates {repeated} are named twice")


def from_handoff_record(
    rec: Phase2ResultRecord, rank: int, start: str
) -> Phase2InputRecord:
    """A solved record as Phase 2 consumes it, starting from its solved pose or
    the Phase-1 pose it started from. The arc assignment is the one both poses
    were solved against."""
    pose = rec["pose"] if start == "solved" else rec.get("pose_in")
    if pose is None:
        raise ValueError(f"candidate {rec['idx']} records no Phase-1 pose")
    return {
        "idx": rec["idx"],
        "n_arcs": rec["n_arcs"],
        "pose": pose,
        "probe_to_hole": rec["hole"],
        "partition": rec["partition"],
        "probe_to_arc_idx": rec["probe_to_arc_idx"],
        "arc_centroids_deg": rec["arc_centroids_deg"],
        "min_clear": rec.get("min_clear"),
        "objective": rec.get("objective_p1"),
        "rank": rank,
    }


def select_from_handoff(
    handoff: Phase2HandoffPayload, settings: Phase2Settings
) -> list[Phase2InputRecord]:
    """The records of an earlier Phase 2 to solve again: the named candidates,
    or else the first ``topk`` it kept, in its ranking."""
    if settings.ranks or settings.ranks_file:
        raise ValueError(
            "ranks are offsets into a Phase-1 pool's order; name a handoff's "
            "records with RUTTER_CANDIDATES or RUTTER_CANDIDATES_FILE"
        )
    by_idx = {rec["idx"]: rec for rec in handoff["all"]}
    ids = candidate_ids(settings)
    if ids is None:
        ids = [rec["idx"] for rec in handoff["ranked"]][: settings.topk]
    _check_known(ids, by_idx, str(settings.poses))
    return [
        from_handoff_record(by_idx[c], rank, settings.handoff_start)
        for rank, c in enumerate(ids)
    ]
