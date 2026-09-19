"""Choosing which Phase-1 candidates Phase 2 solves.

Kept free of jax so it can be imported and tested without initialising a GPU
backend: the stage module sets JAX platform and memory options at import, which a
unit test has no business triggering.
"""

from __future__ import annotations

from typing import Any, cast

from aind_rutter.optimization.pipeline.records import (
    Phase1PoolRecord,
    Phase2InputRecord,
)
from aind_rutter.optimization.pipeline.settings import Phase2Settings

# Sentinels for records missing the ranking field, chosen so they sort last in
# whichever direction the field is ranked.
_WORST_ASCENDING = 1e18
_WORST_DESCENDING = -1e9


def rank_order(all_recs: list[Phase1PoolRecord], select_by: str) -> list[int]:
    """Indices of ``all_recs``, best first, by the named Phase-1 field.

    ``objective`` blends clearance, normalized coverage and the threading penalty
    and is lower-is-better, so it sorts ascending; the clearance-style metrics
    such as ``min_clear`` are higher-is-better and sort descending.
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
    all_recs: list[Phase1PoolRecord], settings: Phase2Settings
) -> list[Phase2InputRecord]:
    """The candidates Phase 2 will solve, in rank order.

    Nothing is culled by FCL between the phases — the pool is ranked and the best
    ``topk`` handed on, with FCL applied once at the end as the ground-truth gate.
    ``ranks`` or ``ranks_file`` replace that cut with explicit offsets into the
    ranked order, which is how a run probes where good candidates stop appearing
    instead of guessing a cutoff; offsets past the end are dropped.
    """
    order = rank_order(all_recs, settings.select_by)
    if settings.ranks or settings.ranks_file:
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
