"""Arc-first candidate generation for the optimizer's Stage 1.

Replaces ``solve_top_k_assignments`` (LSAP) + ``solve_top_k_arc_assignments``
(arc enumeration) when the visibility atlas is available. The natural
search unit is ``(partition, AP triple, local arc configs)``:

  for each unordered partition of K probes into ≤ max_arcs groups:
      for each arc group G and AP bin a:
          LocalConfigs[G, a] = all probe → (hole, anchor) assignments
              such that anchors exist at AP within tolerance of a,
              holes are distinct in the arc,
              and intra-arc ML separation ≥ min_ml_sep_deg.
      for each arc-AP triple with pairwise ≥ min_arc_ap_sep:
          for one local config per arc (cross-arc hole unique):
              emit Cell(partition, AP_tuple, per-arc local config)

Cells are scored by a pluggable function (default: mean AP centeredness)
and the top-K are kept via a min-heap during enumeration so the full
cartesian product is never materialised.

Each top-K cell converts to a ``(HoleAssignment, ArcAssignment)`` for
``score_joint`` to consume — Stage 2 reduced SLSQP and Stage 3 full
polish are unchanged.

See ``dev/minlp_assignment_brief.md`` and
``dev/target_valid_atlas_design.md`` for the broader strategy.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from aind_low_point.optimization.arc_assignment import ArcAssignment
from aind_low_point.optimization.atlas import Atlas
from aind_low_point.optimization.hole_assignment import HoleAssignment


# ---------------------------------------------------------------------------
# Cell type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One arc-first candidate after streaming top-K selection.

    ``partition`` is the ordered tuple of probe-index tuples per arc,
    sorted into canonical (ascending arc AP) order. ``ap_triple`` is
    the per-arc chosen AP, parallel to ``partition``. ``picks`` maps
    probe index → ``(hole_id, ml_deg, spin_deg)`` (the local arc
    config representative for that cell).
    """

    partition: tuple[tuple[int, ...], ...]
    ap_triple: tuple[float, ...]
    picks: dict[int, tuple[int, float, float, float]]
    score: float
    score_components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Partition enumeration
# ---------------------------------------------------------------------------


def unordered_partitions(
    K: int, max_arcs: int, *, prefer_more_arcs: bool = True
) -> Iterator[list[tuple[int, ...]]]:
    """Yield all unordered set-partitions of ``range(K)`` into
    ``1..max_arcs`` non-empty groups. Each group is a tuple of probe
    indices in ascending order; partitions are emitted with groups
    sorted by their lowest probe index (canonical form).

    When ``prefer_more_arcs`` (default), the enumeration prefers
    starting a new arc over packing into an existing one, so the
    ``max_arcs``-arc partitions appear FIRST. Practical configs almost
    always use ``max_arcs`` distinct arcs; this keeps the manual-quality
    partitions early in the stream so a bounded budget reaches them.
    """

    def helper(idx: int, groups: list[list[int]]):
        if idx == K:
            if 1 <= len(groups) <= max_arcs:
                yield [tuple(sorted(g)) for g in groups]
            return
        if prefer_more_arcs:
            if len(groups) < max_arcs:
                groups.append([idx])
                yield from helper(idx + 1, groups)
                groups.pop()
            for g in groups:
                g.append(idx)
                yield from helper(idx + 1, groups)
                g.pop()
        else:
            for g in groups:
                g.append(idx)
                yield from helper(idx + 1, groups)
                g.pop()
            if len(groups) < max_arcs:
                groups.append([idx])
                yield from helper(idx + 1, groups)
                groups.pop()

    yield from helper(0, [])


# ---------------------------------------------------------------------------
# Per-probe atlas choices at a chosen arc AP
# ---------------------------------------------------------------------------


_GATHER_CACHE: dict[tuple, list[tuple[int, float, float, float]]] = {}


def clear_arc_first_caches() -> None:
    """Drop cached per-(probe, arc_ap) anchor lists and per-arc-group
    local-config lists. Call at the start of a search if the atlas
    changed since last invocation."""
    _GATHER_CACHE.clear()
    _LOCAL_ARC_CACHE.clear()


def gather_probe_choices(
    atlas: Atlas,
    probe_name: str,
    arc_ap: float,
    ap_tol: float,
    *,
    ml_bin_deg: float = 0.5,
    restrict_holes: Iterable[int] | None = None,
) -> list[tuple[int, float, float, float]]:
    """Return list of ``(hole_id, ml_deg, spin_deg, anchor_ap_deg)``
    over atlas anchors at ``arc_ap ± ap_tol``, deduplicated by
    ``(hole, ML-bin)``.

    Two anchors at the same hole within ``ml_bin_deg`` of each other
    are interchangeable for the ML-separation constraint, so we keep
    one representative per bin. This collapses ~7k anchors / pair to
    tens of MLs / pair and keeps the backtracker tractable without
    losing distinct ML basins (per the
    ``diagnose_arc_first.py`` post-mortem).

    ``restrict_holes`` filters to the listed hole set when set.
    ``anchor_ap_deg`` is kept so scorers can weight by AP centeredness.
    """
    # Cache hot path (no restrict_holes): keyed by (atlas_id, probe,
    # ap-quantized, ap_tol, ml_bin). Many partitions share probes at
    # the same AP grid point, so this trims ~95% of the work.
    cache_key = None
    if restrict_holes is None:
        cache_key = (
            id(atlas), probe_name,
            int(round(arc_ap * 100)),
            int(round(ap_tol * 100)),
            int(round(ml_bin_deg * 100)),
        )
        cached = _GATHER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    restrict = set(restrict_holes) if restrict_holes is not None else None
    out: list[tuple[int, float, float, float]] = []
    for hid in atlas.hole_ids:
        if restrict is not None and hid not in restrict:
            continue
        e = atlas.entries[(probe_name, hid)]
        if e.ap_min is None or e.ap_max is None:
            continue
        if arc_ap + ap_tol < e.ap_min or arc_ap - ap_tol > e.ap_max:
            continue
        # Per-hole dedup by ML bin (keep the anchor closest in AP to arc_ap)
        seen: dict[int, tuple[float, float, float]] = {}
        for a in e.anchors:
            if abs(a.ap_deg - arc_ap) > ap_tol:
                continue
            bin_key = int(round(float(a.ml_deg) / ml_bin_deg))
            cand = (float(a.ml_deg), float(a.spin_deg), float(a.ap_deg))
            cur = seen.get(bin_key)
            if cur is None or abs(cand[2] - arc_ap) < abs(cur[2] - arc_ap):
                seen[bin_key] = cand
        for ml, spin, ap in seen.values():
            out.append((hid, ml, spin, ap))

    if cache_key is not None:
        _GATHER_CACHE[cache_key] = out
    return out


# ---------------------------------------------------------------------------
# Local arc configs (probe → (hole, ml, spin) for one arc group)
# ---------------------------------------------------------------------------


def score_local_config(
    config: dict[int, tuple[int, float, float, float]],
    arc_ap: float,
    *,
    min_ml_sep: float,
    ap_tol: float,
    boundary_sigma_deg: float = 6.0,
) -> float:
    """Per-arc score for one local config.

    Composite of:
      - boundary kernel: ``exp(-(ml_slack)^2 / σ²)``, peaks AT the
        ml-sep boundary (slack ≈ 0). Manual-quality configs sit there.
      - AP centeredness: ``mean(1 − |anchor_ap − arc_ap| / ap_tol)``.

    Higher = better; used to pick the top-N local configs per arc
    when caller wants to bound combinatorics. With this score, the
    manual-like (tight ml-sep, well-centered AP) configs rank near the
    top of the per-arc list instead of being pruned by hole-order.
    """
    if not config:
        return 0.0
    mls = [v[1] for v in config.values()]
    if len(mls) >= 2:
        pair_diffs = [
            abs(mls[a] - mls[b])
            for a in range(len(mls)) for b in range(a + 1, len(mls))
        ]
        slack = min(pair_diffs) - min_ml_sep
        boundary = float(np.exp(-(slack ** 2) / (boundary_sigma_deg ** 2)))
    else:
        boundary = 1.0
    cent = float(np.mean([
        1.0 - abs(v[3] - arc_ap) / ap_tol for v in config.values()
    ]))
    return 0.5 * boundary + 0.5 * cent


_LOCAL_ARC_CACHE: dict[tuple, list[dict]] = {}


def _best_ml_combo_vec(
    per_probe_anchors: list[list[tuple[float, float, float]]],
    *,
    min_ml_sep: float,
    arc_ap: float,
    ap_tol: float,
    boundary_sigma_deg: float,
) -> tuple[float, list[tuple[float, float, float]]] | None:
    """Vectorized n=2 / n=3 fast-path. Builds an outer-product grid of
    per-probe anchors and computes pairwise ml-sep + score in numpy,
    then argmax over feasible cells. ~100× faster than Python recursion
    for typical 5–30-anchor probes."""
    n = len(per_probe_anchors)
    if n == 2:
        mls0 = np.array([a[0] for a in per_probe_anchors[0]])
        mls1 = np.array([a[0] for a in per_probe_anchors[1]])
        aps0 = np.array([a[2] for a in per_probe_anchors[0]])
        aps1 = np.array([a[2] for a in per_probe_anchors[1]])
        diff = np.abs(mls0[:, None] - mls1[None, :])
        feasible = diff >= min_ml_sep
        if not feasible.any():
            return None
        slack = diff - min_ml_sep
        sigma_sq = boundary_sigma_deg * boundary_sigma_deg
        boundary = np.exp(-(slack * slack) / sigma_sq)
        cent0 = 1.0 - np.abs(aps0 - arc_ap) / ap_tol
        cent1 = 1.0 - np.abs(aps1 - arc_ap) / ap_tol
        cent = (cent0[:, None] + cent1[None, :]) * 0.5
        score = 0.5 * boundary + 0.5 * cent
        score = np.where(feasible, score, -np.inf)
        idx_flat = int(np.argmax(score))
        i, j = np.unravel_index(idx_flat, score.shape)
        return float(score[i, j]), [
            per_probe_anchors[0][i], per_probe_anchors[1][j]
        ]
    # n == 3
    mls0 = np.array([a[0] for a in per_probe_anchors[0]])
    mls1 = np.array([a[0] for a in per_probe_anchors[1]])
    mls2 = np.array([a[0] for a in per_probe_anchors[2]])
    aps0 = np.array([a[2] for a in per_probe_anchors[0]])
    aps1 = np.array([a[2] for a in per_probe_anchors[1]])
    aps2 = np.array([a[2] for a in per_probe_anchors[2]])
    d01 = np.abs(mls0[:, None, None] - mls1[None, :, None])
    d02 = np.abs(mls0[:, None, None] - mls2[None, None, :])
    d12 = np.abs(mls1[None, :, None] - mls2[None, None, :])
    feasible = (d01 >= min_ml_sep) & (d02 >= min_ml_sep) & (d12 >= min_ml_sep)
    if not feasible.any():
        return None
    min_pair = np.minimum(np.minimum(d01, d02), d12)
    slack = min_pair - min_ml_sep
    sigma_sq = boundary_sigma_deg * boundary_sigma_deg
    boundary = np.exp(-(slack * slack) / sigma_sq)
    cent0 = 1.0 - np.abs(aps0 - arc_ap) / ap_tol
    cent1 = 1.0 - np.abs(aps1 - arc_ap) / ap_tol
    cent2 = 1.0 - np.abs(aps2 - arc_ap) / ap_tol
    cent = (
        cent0[:, None, None] + cent1[None, :, None] + cent2[None, None, :]
    ) / 3.0
    score = 0.5 * boundary + 0.5 * cent
    score = np.where(feasible, score, -np.inf)
    idx_flat = int(np.argmax(score))
    i, j, k = np.unravel_index(idx_flat, score.shape)
    return float(score[i, j, k]), [
        per_probe_anchors[0][i], per_probe_anchors[1][j],
        per_probe_anchors[2][k],
    ]


def _best_ml_combo_for_hole_tuple(
    per_probe_anchors: list[list[tuple[float, float, float]]],
    *,
    min_ml_sep: float,
    arc_ap: float,
    ap_tol: float,
) -> tuple[float, list[tuple[float, float, float]]] | None:
    """Among all ml/spin combinations for a fixed hole-tuple, find the
    highest-scoring one that satisfies the ml-sep constraint.

    ``per_probe_anchors[k]`` is the (ml, spin, anchor_ap) anchor list for
    probe k at its fixed (probe, hole). Each probe contributes one
    anchor; the combination must satisfy pairwise |ml_a − ml_b| ≥
    ``min_ml_sep``.

    Returns ``(best_score, [anchor_per_probe, ...])`` or ``None``
    if no combination satisfies ml-sep.

    Score is the per-arc composite used by :func:`score_local_config`,
    inlined here for speed.
    """
    n = len(per_probe_anchors)
    if n == 0 or any(not lst for lst in per_probe_anchors):
        return None

    # Single-probe arc — no ml-sep constraint; score is pure AP centeredness
    if n == 1:
        best_anchor = min(
            per_probe_anchors[0], key=lambda a: abs(a[2] - arc_ap)
        )
        score = 1.0 - abs(best_anchor[2] - arc_ap) / ap_tol
        return score, [best_anchor]

    # Python tight-loop fast-path. Earlier prototype used a numpy
    # vmap-over-anchors variant, but for the tiny per-probe anchor
    # counts we actually have (~5-30 per probe), the Python recursion
    # below is FASTER than numpy due to per-call array-creation
    # overhead. Numpy/JAX would only win at much larger sizes.
    #
    # Score is pure AP centeredness (boundary kernel removed — it
    # incorrectly penalized comfortable ml-sep on arcs whose geometry
    # didn't force tight packing).
    inv_ap_tol = 1.0 / ap_tol

    best_score = -1.0
    best_combo: list[tuple[float, float, float]] | None = None

    def search(idx: int, current: list[tuple[float, float, float]]) -> None:
        nonlocal best_score, best_combo
        if idx == n:
            # Pure AP centeredness across probes
            s = sum(
                1.0 - abs(c[2] - arc_ap) * inv_ap_tol for c in current
            ) / n
            if s > best_score:
                best_score = s
                best_combo = list(current)
            return
        for cand in per_probe_anchors[idx]:
            ok = True
            cand_ml = cand[0]
            for prev in current:
                if abs(cand_ml - prev[0]) < min_ml_sep:
                    ok = False
                    break
            if not ok:
                continue
            current.append(cand)
            search(idx + 1, current)
            current.pop()

    search(0, [])
    if best_combo is None:
        return None
    return best_score, best_combo


def local_arc_configs(
    probe_indices: tuple[int, ...],
    arc_ap: float,
    atlas: Atlas,
    probe_names: list[str],
    *,
    min_ml_sep: float,
    ap_tol: float,
    ml_bin_deg: float = 0.5,
    cap: int | None = None,        # kept for backwards-compat; ignored
    top_n: int | None = None,
    max_backtrack_calls: int = 200_000,  # kept for backwards-compat; ignored
) -> list[dict[int, tuple[int, float, float, float]]]:
    """Enumerate distinct **hole-tuples** for one arc group; for each,
    pick the best ml/spin combination satisfying the ml-sep constraint.

    The discrete decision is the hole-tuple; ml/spin are continuous
    parameters that Stage 2 SLSQP polishes downstream. Earlier versions
    enumerated all ml/spin permutations and deduped at the end, which
    blew up the search tree (3839 configs / arc → 6 hole-tuples).

    ``top_n`` keeps the highest-scoring N hole-tuples; ``None`` returns
    them all. ``cap`` and ``max_backtrack_calls`` are accepted for API
    stability but ignored (the new path doesn't backtrack on ml).

    Result is cached by ``(atlas_id, probe_indices, arc_ap_q, ap_tol_q,
    ml_bin_q, min_ml_sep_q, top_n)``. Many partitions share an arc
    group at the same AP grid point; cache hits eliminate the redundant
    work.
    """
    # Cache key — quantize floats so near-duplicate calls hit
    cache_key = (
        id(atlas), tuple(probe_indices),
        int(round(arc_ap * 100)),
        int(round(ap_tol * 100)),
        int(round(ml_bin_deg * 100)),
        int(round(min_ml_sep * 100)),
        top_n,
    )
    cached = _LOCAL_ARC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Per-probe: (probe, hole) → list of (ml, spin, anchor_ap) anchors
    per_probe_holes: list[dict[int, list[tuple[float, float, float]]]] = []
    for i in probe_indices:
        choices = gather_probe_choices(
            atlas, probe_names[i], arc_ap, ap_tol, ml_bin_deg=ml_bin_deg
        )
        by_hole: dict[int, list[tuple[float, float, float]]] = {}
        for hid, ml, spin, anchor_ap in choices:
            by_hole.setdefault(hid, []).append((ml, spin, anchor_ap))
        per_probe_holes.append(by_hole)

    if any(not h for h in per_probe_holes):
        _LOCAL_ARC_CACHE[cache_key] = []
        return []

    valid_holes_per_probe = [list(h.keys()) for h in per_probe_holes]

    scored: list[tuple[float, dict[int, tuple[int, float, float, float]]]] = []
    for hole_tuple in itertools.product(*valid_holes_per_probe):
        if len(set(hole_tuple)) != len(hole_tuple):
            continue
        per_probe_anchors = [
            per_probe_holes[k][hid] for k, hid in enumerate(hole_tuple)
        ]
        result = _best_ml_combo_for_hole_tuple(
            per_probe_anchors,
            min_ml_sep=min_ml_sep,
            arc_ap=arc_ap,
            ap_tol=ap_tol,
        )
        if result is None:
            continue
        score, anchor_combo = result
        cfg = {
            probe_indices[k]: (
                hole_tuple[k], anchor_combo[k][0], anchor_combo[k][1],
                anchor_combo[k][2],
            )
            for k in range(len(probe_indices))
        }
        scored.append((score, cfg))

    if top_n is not None and len(scored) > top_n:
        scored.sort(key=lambda x: -x[0])
        scored = scored[:top_n]

    out = [cfg for _, cfg in scored]
    _LOCAL_ARC_CACHE[cache_key] = out
    return out


# ---------------------------------------------------------------------------
# Cell scoring
# ---------------------------------------------------------------------------


CellScorer = Callable[
    [
        tuple[tuple[int, ...], ...],            # partition (canonical, asc AP)
        tuple[float, ...],                       # ap_triple (asc AP)
        dict[int, tuple[int, float, float, float]],  # picks: i → (hid, ml, spin, anchor_ap)
        dict,                                    # context kwargs
    ],
    tuple[float, dict[str, float]],
]


def ap_centeredness_score(
    partition: tuple[tuple[int, ...], ...],
    ap_triple: tuple[float, ...],
    picks: dict[int, tuple[int, float, float, float]],
    ctx: dict,
) -> tuple[float, dict[str, float]]:
    """Default cell scorer: mean AP centeredness across probes.

    ``centeredness_i = 1 − |anchor_ap_i − arc_ap_for_i| / ap_tol``.
    Higher = atlas anchors close to the chosen arc AP = comfortable
    polish seed.

    Earlier versions composited this with a boundary kernel on
    ``min_arc_ml_slack``. That was removed: the boundary kernel
    incorrectly punished arcs whose geometry didn't force tight ml-sep
    (e.g., manual T12 arc 1 has 7.5° slack and is correct, but the
    kernel dropped its score). Without the kernel the score becomes a
    coarse pre-filter — discrimination between top cells should happen
    at Stage 2 polish, not at the cell scorer.

    Side outputs (in components dict) carry the ml-slack signal for
    diagnostics and downstream rankers that may want it.
    """
    ap_tol = float(ctx.get("ap_tol", 1.0))
    arc_ap_by_probe: dict[int, float] = {}
    for arc_idx, group in enumerate(partition):
        for i in group:
            arc_ap_by_probe[i] = ap_triple[arc_idx]
    centeredness: list[float] = []
    for i, (_, _, _, anchor_ap) in picks.items():
        centeredness.append(1.0 - abs(anchor_ap - arc_ap_by_probe[i]) / ap_tol)
    mean_cent = float(np.mean(centeredness)) if centeredness else 0.0

    # Carry the ml-slack signal as a component for diagnostics
    min_ml_sep = float(ctx.get("min_ml_sep", 16.0))
    arc_slacks: list[float] = []
    for group in partition:
        mls = [picks[i][1] for i in group if i in picks]
        if len(mls) >= 2:
            pair_diffs = [
                abs(mls[a] - mls[b])
                for a in range(len(mls)) for b in range(a + 1, len(mls))
            ]
            arc_slacks.append(min(pair_diffs) - min_ml_sep)
    min_arc_slack = float(min(arc_slacks)) if arc_slacks else float("inf")

    components = {
        "mean_ap_centeredness": mean_cent,
        "min_arc_ml_slack_deg": min_arc_slack,
    }
    return mean_cent, components


# ---------------------------------------------------------------------------
# Streaming top-K enumeration
# ---------------------------------------------------------------------------


def _per_probe_envelope(
    atlas: Atlas, probe_names: list[str]
) -> list[tuple[float, float] | None]:
    """Per-probe atlas AP envelope (min, max) across all valid holes."""
    out: list[tuple[float, float] | None] = []
    for name in probe_names:
        los: list[float] = []
        his: list[float] = []
        for hid in atlas.hole_ids:
            e = atlas.entries[(name, hid)]
            if e.ap_min is not None and e.ap_max is not None:
                los.append(e.ap_min)
                his.append(e.ap_max)
        out.append((min(los), max(his)) if los else None)
    return out


def arc_first_top_k(
    probes: list,
    atlas: Atlas,
    *,
    top_k: int = 50,
    max_arcs: int = 3,
    min_arc_ap_sep_deg: float = 16.0,
    min_ml_sep_deg: float = 16.0,
    ap_step_deg: float = 2.0,
    ap_tol_deg: float = 3.0,
    ml_bin_deg: float = 0.5,
    arc_cfg_cap_per_cell: int = 10,
    max_cells_per_partition_ap: int = 500,
    max_cells_per_partition: int = 10_000,
    global_cell_budget: int = 2_000_000,
    scorer: CellScorer = ap_centeredness_score,
    track_targets: list[dict[str, int]] | None = None,
    verbose: bool = False,
    progress_every: int = 50_000,
) -> tuple[list[Cell], list[int | None]]:
    """Stream arc-first cells, return the top-``top_k`` by ``scorer``.

    Parameters
    ----------
    probes : list[ProbeStaticInfo]
        Probes (named, kind-tagged). Only ``.name`` is used here; the
        full info is downstream when cells convert to ``ScoreJoint``
        inputs.
    atlas : Atlas
        Visibility atlas; must contain entries for all
        ``(probe.name, hole_id)`` pairs.
    top_k : int
        Number of best-scoring cells to retain.
    max_arcs : int
        Upper bound on arc count per partition. Default 3.
    min_arc_ap_sep_deg : float
        Required pairwise AP separation between arc APs.
    min_ml_sep_deg : float
        Required intra-arc ML separation between probe anchors.
    ap_step_deg : float
        Per-arc AP grid resolution within the arc's envelope.
    ap_tol_deg : float
        Anchor AP must lie within ``ap_tol_deg`` of the chosen arc AP.
    ml_bin_deg : float
        ML dedup bin width per (probe, hole) atlas entry.
    arc_cfg_cap_per_cell : int
        Per-arc top-N local configs kept (by per-arc composite score).
        Replaces the older first-N-by-hole-order truncation. Default 50.
        With score-based selection, boundary-tight (manual-like)
        configs survive the per-arc cap even when they're not in the
        first 50 in hole-iteration order.
    max_cells_per_partition_ap : int
        Bound on cross-arc cells emitted per ``(partition, AP triple)``.
        Default 2000 — prevents the 50³ Cartesian explosion when every
        arc cap is near its limit.
    max_cells_per_partition : int
        Bound on total cells emitted per partition. Default 100k.
        Prevents 2-arc partitions (lots of AP pairs) from hogging the
        global budget before 3-arc partitions get a turn.
    global_cell_budget : int
        Hard ceiling on total cells streamed. Default 1M. Search breaks
        out early on exceedance.
    progress_every : int
        Print streaming counters every ``progress_every`` cells when
        ``verbose=True``.
    scorer : CellScorer
        Pluggable cell scorer. Default :func:`ap_centeredness_score`.
    track_targets : optional list of probe → hole maps
        Each entry is a ``{probe_name: hole_id}`` dict; for every match
        the rank in streaming order is recorded in the returned ranks
        list. Used by diagnostics to ask "where in the stream is the
        manual cell?".
    verbose : bool
        If True, print streaming counters.

    Returns
    -------
    cells : list[Cell]
        Top-K cells sorted by descending score.
    target_ranks : list[Optional[int]]
        For each entry in ``track_targets``, the 0-based rank of its
        first occurrence in the streaming sequence, or ``None`` if it
        never appeared.
    """
    K = len(probes)
    probe_names = [p.name for p in probes]
    envelope = _per_probe_envelope(atlas, probe_names)
    clear_arc_first_caches()

    scorer_ctx = {
        "ap_tol": ap_tol_deg,
        "min_ml_sep": min_ml_sep_deg,
    }

    # Min-heap by (score, tiebreaker, cell) — we pop smallest when over capacity.
    heap: list[tuple[float, int, Cell]] = []
    tie = 0  # stable insertion counter

    # Tracking
    target_signatures: list[frozenset[tuple[str, int]]] = [
        frozenset(t.items()) for t in (track_targets or [])
    ]
    target_ranks: list[int | None] = [None] * len(target_signatures)
    stream_idx = 0  # counts emitted cells

    def cell_signature(picks: dict[int, tuple[int, float, float, float]]) -> frozenset:
        return frozenset(
            (probe_names[i], hid) for i, (hid, _, _, _) in picks.items()
        )

    n_partitions = 0
    n_partition_ap_triples = 0
    n_cells_with_configs = 0
    last_progress_print = 0
    budget_exceeded = False

    for partition in unordered_partitions(K, max_arcs):
        if budget_exceeded:
            break
        partition_cell_count = 0
        if verbose and n_partitions % 20 == 0:
            print(f"[arc_first_top_k] starting partition {n_partitions + 1}: "
                  f"{[len(g) for g in partition]} probes/arc; "
                  f"streamed {stream_idx} cells so far, heap@{len(heap)}",
                  flush=True)
        n_partitions += 1
        # Per-arc AP envelope: intersection of probe envelopes within arc
        arc_envs: list[tuple[float, float]] = []
        ok = True
        for group in partition:
            lo, hi = -1e6, +1e6
            for i in group:
                env = envelope[i]
                if env is None:
                    ok = False
                    break
                lo = max(lo, env[0])
                hi = min(hi, env[1])
            if not ok or lo > hi:
                ok = False
                break
            arc_envs.append((lo, hi))
        if not ok:
            continue

        # Per-arc AP grid
        per_arc_aps: list[np.ndarray] = []
        for lo, hi in arc_envs:
            n_steps = max(1, int(np.ceil((hi - lo) / ap_step_deg)) + 1)
            per_arc_aps.append(np.linspace(lo, hi, n_steps))

        # AP-triple enumeration with pairwise separation
        for ap_combo in itertools.product(*per_arc_aps):
            if budget_exceeded or partition_cell_count >= max_cells_per_partition:
                break
            if len(ap_combo) > 1:
                ok2 = True
                for i in range(len(ap_combo)):
                    for j in range(i + 1, len(ap_combo)):
                        if abs(ap_combo[i] - ap_combo[j]) < min_arc_ap_sep_deg:
                            ok2 = False
                            break
                    if not ok2:
                        break
                if not ok2:
                    continue
            n_partition_ap_triples += 1

            # Per-arc local configs at the chosen AP. Use top_n (score-
            # based) instead of cap (hole-order) so boundary-tight configs
            # survive the per-arc bound.
            cfgs_per_arc: list[list[dict[int, tuple[int, float, float, float]]]] = []
            arc_ok = True
            for group, ap in zip(partition, ap_combo):
                cfgs = local_arc_configs(
                    group, ap, atlas, probe_names,
                    min_ml_sep=min_ml_sep_deg,
                    ap_tol=ap_tol_deg,
                    ml_bin_deg=ml_bin_deg,
                    top_n=arc_cfg_cap_per_cell,
                )
                if not cfgs:
                    arc_ok = False
                    break
                cfgs_per_arc.append(cfgs)
            if not arc_ok:
                continue
            n_cells_with_configs += 1

            # Canonicalize partition + ap_triple to ascending AP order so
            # downstream consumers see a stable arc indexing.
            asc_order = sorted(range(len(ap_combo)), key=lambda i: ap_combo[i])
            canonical_partition = tuple(partition[i] for i in asc_order)
            canonical_ap = tuple(ap_combo[i] for i in asc_order)
            canonical_cfgs = [cfgs_per_arc[i] for i in asc_order]

            # Cross-arc assembly with global hole uniqueness
            cells_this_combo = 0
            for combo in itertools.product(*canonical_cfgs):
                if cells_this_combo >= max_cells_per_partition_ap:
                    break
                if stream_idx >= global_cell_budget:
                    budget_exceeded = True
                    break
                used: set[int] = set()
                conflict = False
                for arc_cfg in combo:
                    for _, (hid, _, _, _) in arc_cfg.items():
                        if hid in used:
                            conflict = True
                            break
                        used.add(hid)
                    if conflict:
                        break
                if conflict:
                    continue

                # Merge per-arc dicts into one picks map (probe_idx → tuple)
                picks: dict[int, tuple[int, float, float, float]] = {}
                for arc_cfg in combo:
                    picks.update(arc_cfg)

                stream_idx += 1
                cells_this_combo += 1
                partition_cell_count += 1
                if verbose and stream_idx - last_progress_print >= progress_every:
                    last_progress_print = stream_idx
                    print(f"[arc_first_top_k] streamed {stream_idx:>9} cells "
                          f"(part {n_partitions}, ap_triples {n_partition_ap_triples}, "
                          f"heap@{len(heap)})",
                          flush=True)
                sig = cell_signature(picks)
                for ti, ts in enumerate(target_signatures):
                    if target_ranks[ti] is None and sig == ts:
                        target_ranks[ti] = stream_idx - 1

                score, components = scorer(
                    canonical_partition, canonical_ap, picks, scorer_ctx
                )
                cell = Cell(
                    partition=canonical_partition,
                    ap_triple=canonical_ap,
                    picks=picks,
                    score=score,
                    score_components=components,
                )
                tie += 1
                if len(heap) < top_k:
                    heapq.heappush(heap, (score, tie, cell))
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, (score, tie, cell))

    if verbose:
        print(f"[arc_first_top_k] partitions explored:        {n_partitions}")
        print(f"[arc_first_top_k] (partition, AP-triple):     {n_partition_ap_triples}")
        print(f"[arc_first_top_k] cells with arc configs:     {n_cells_with_configs}")
        print(f"[arc_first_top_k] total cells streamed:       {stream_idx}")
        if budget_exceeded:
            print(f"[arc_first_top_k] HIT global_cell_budget ({global_cell_budget})")
        if heap:
            print(f"[arc_first_top_k] top-{top_k} kept; score range: "
                  f"{min(c[0] for c in heap):+.4f} .. {max(c[0] for c in heap):+.4f}")
        else:
            print("[arc_first_top_k] (no cells produced)")

    # Sort heap descending by score
    cells = sorted([c for _, _, c in heap], key=lambda c: -c.score)
    return cells, target_ranks


# ---------------------------------------------------------------------------
# Cell → (HoleAssignment, ArcAssignment) conversion
# ---------------------------------------------------------------------------


def cell_to_ha_aa(
    cell: Cell, probes: list
) -> tuple[HoleAssignment, ArcAssignment]:
    """Convert a ``Cell`` to a ``(HoleAssignment, ArcAssignment)`` pair
    consumable by :func:`score_joint`. Both inputs are canonical
    (ascending AP per arc); the arc indexing follows that order.

    ``cost`` fields are set to ``-cell.score`` so legacy "lower is
    better" callers don't break, but downstream consumers should rely on
    the joint reranker's own ranking; this cost is a placeholder.
    """
    probe_names = [p.name for p in probes]
    probe_to_hole: dict[str, int] = {}
    probe_to_arc_idx: dict[str, int] = {}
    for arc_idx, group in enumerate(cell.partition):
        for i in group:
            probe_to_hole[probe_names[i]] = cell.picks[i][0]
            probe_to_arc_idx[probe_names[i]] = arc_idx
    ha = HoleAssignment(probe_to_hole=probe_to_hole, cost=-cell.score)
    aa = ArcAssignment(
        probe_to_arc_idx=probe_to_arc_idx,
        arc_centroids_deg=tuple(cell.ap_triple),
        cost=-cell.score,
    )
    return ha, aa
