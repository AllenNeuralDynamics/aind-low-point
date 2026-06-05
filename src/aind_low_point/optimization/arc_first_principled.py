"""Arc-first search with principled per-(partition, hole-tuple) emission.

Replaces the grid-enumerated ``arc_first_top_k`` cell stream (which
generated ~2M cells per run by enumerating an AP grid × hole-tuples)
with a deduplicated emission at the level of `(partition, hole-tuple)`.

For each `(partition, hole-tuple)`:
  - Per arc: the per-probe atlas AP envelope intersection gives the
    feasible arc-AP interval. Empty intersection → reject the
    candidate.
  - Principled arc AP = midpoint of intersection. SLSQP polishes
    from there.
  - Per probe: pick the atlas anchor closest to the principled AP
    as the ml/spin warm-start.
  - Check pairwise AP separation ≥ 16°.
  - Check intra-arc ml-sep at the chosen anchors.

Each emitted candidate carries cheap pre-polish signals (intersection
widths, ml-sep slack, anchor density) so the caller can order the
polish queue. The signals are NOT used to filter — they only influence
order.

Designed in conversation 2026-05-20 in response to the realization that
AP-triple grid enumeration was redundant (SLSQP optimizes arc APs;
multiple AP seeds within a (partition, hole-tuple) family converge to
roughly the same local minimum). Discrete decision unit is
`(partition, hole-tuple)`; AP triple is a single principled seed.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from aind_low_point.optimization.arc_assignment import (
    ArcAssignment,
    bounded_isotonic_arc_aps,
)
from aind_low_point.optimization.atlas import Atlas
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.planning import PoseLimits

# Arc / per-arc caps are KINEMATIC (16 deg angular exclusion over the AP/ML
# range), not hardware counts — see PoseLimits. Default to the rig limits;
# the realized counts are further bounded by the probe count.
_POSE_LIMITS = PoseLimits()


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


@dataclass(frozen=True)
class _AtlasArrays:
    """Per-(probe, hole) numpy arrays of anchor data, pre-sorted by ``ap_deg``.

    Built once per atlas to amortise the cost of finding the nearest-AP
    anchor across thousands of envelope-feasibility checks. Without this,
    ``min(anchors, key=lambda a: abs(a.ap_deg - target_ap))`` does an
    O(N) Python scan per call (N up to ~9000 with n_top=128, n_spin=72);
    with sorted arrays + ``np.searchsorted`` it's O(log N) in C.
    """

    # Each key is (probe_idx, hole_id).
    ap_sorted: dict
    ml_sorted: dict        # parallel to ap_sorted
    spin_sorted: dict      # parallel to ap_sorted
    ap_min_max: dict       # (probe_idx, hole_id) -> (ap_min, ap_max)


def _build_atlas_arrays(
    atlas: Atlas, probe_names: list[str]
) -> _AtlasArrays:
    """Pre-pack atlas anchors into numpy arrays per (probe, hole).

    Anchors are sorted by ``ap_deg`` so ``np.searchsorted`` finds the
    nearest-AP anchor in O(log N).
    """
    ap_sorted: dict = {}
    ml_sorted: dict = {}
    spin_sorted: dict = {}
    ap_min_max: dict = {}
    for probe_idx, name in enumerate(probe_names):
        for hid in atlas.hole_ids:
            e = atlas.entries[(name, hid)]
            if not e.anchors or e.ap_min is None or e.ap_max is None:
                continue
            aps = np.array([a.ap_deg for a in e.anchors], dtype=np.float32)
            mls = np.array([a.ml_deg for a in e.anchors], dtype=np.float32)
            spins = np.array([a.spin_deg for a in e.anchors], dtype=np.float32)
            order = np.argsort(aps)
            key = (probe_idx, hid)
            ap_sorted[key] = aps[order]
            ml_sorted[key] = mls[order]
            spin_sorted[key] = spins[order]
            ap_min_max[key] = (float(e.ap_min), float(e.ap_max))
    return _AtlasArrays(
        ap_sorted=ap_sorted,
        ml_sorted=ml_sorted,
        spin_sorted=spin_sorted,
        ap_min_max=ap_min_max,
    )


def _nearest_ap_index(ap_sorted_arr: np.ndarray, target_ap: float) -> int:
    """Index into sorted ``ap_sorted_arr`` of the anchor closest to
    ``target_ap``. O(log N) via ``np.searchsorted``.
    """
    pos = int(np.searchsorted(ap_sorted_arr, target_ap))
    n = len(ap_sorted_arr)
    if pos == 0:
        return 0
    if pos == n:
        return n - 1
    # Compare neighbors
    if abs(ap_sorted_arr[pos] - target_ap) < abs(ap_sorted_arr[pos - 1] - target_ap):
        return pos
    return pos - 1


@dataclass(frozen=True)
class ArcFirstCandidate:
    """One discrete decision from principled arc-first search.

    Polishable directly by Stage 2 (``score_joint`` or batched path):
    ``ha`` and ``aa`` are the discrete decisions; ``ml_seed`` /
    ``spin_seed`` are atlas-anchor warm starts (the polish will refine
    them).

    Cheap signals (no polish) are attached for adaptive ordering. They
    are NOT for filtering — the polish itself is the only reliable
    quality discriminator. Signals just bias the polish order so that
    likely-feasible candidates run first.
    """

    ha: HoleAssignment
    aa: ArcAssignment
    ml_seed: dict[str, float]              # probe_name → ml warm-start
    spin_seed: dict[str, float]            # probe_name → spin warm-start

    # Cheap pre-polish signals
    ap_intersection_min_width_deg: float   # min over arcs of (hi - lo) of envelope intersection
    min_intra_arc_ml_slack_deg: float      # min over arcs of (min pairwise ml diff − min_ml_sep)
    total_atlas_anchors: int               # Σ over (probe, hole) of anchor counts
    ap_centeredness_sum: float             # Σ over probes of (1 − |anchor_ap − arc_ap|/ap_tol)
    arc_ap_pairwise_min_sep_deg: float     # min over arc pairs of |AP_i - AP_j|
    composite_order_score: float           # weighted sum of the above
    components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-arc principled hole-tuple enumeration
# ---------------------------------------------------------------------------


def _enumerate_arc_hole_tuples(
    probe_indices: tuple[int, ...],
    aa: _AtlasArrays,
    *,
    min_ml_sep_deg: float,
) -> list[dict]:
    """Enumerate envelope-feasible hole-tuples for one arc group.

    Each tuple's per-probe AP envelopes must intersect (non-empty
    interval), and at the midpoint AP the per-probe atlas anchors must
    admit a combination satisfying ml-sep.

    Returns dicts with:
        holes: tuple of hole_ids parallel to ``probe_indices``
        ap_lo, ap_hi, ap_mid: envelope intersection + midpoint
        seed_ml, seed_spin, seed_ap: per-probe seed anchor data (parallel to probe_indices)
        min_ml_diff: min pairwise ml diff after anchor selection
        total_anchors: Σ anchor counts at the chosen holes
    """
    # Pre-collect per-probe (hole, ap_min, ap_max) lists
    per_probe_holes: list[list[tuple[int, float, float]]] = []
    for pi in probe_indices:
        hits: list[tuple[int, float, float]] = []
        for (probe_idx, hid), (ap_min, ap_max) in aa.ap_min_max.items():
            if probe_idx != pi:
                continue
            hits.append((hid, ap_min, ap_max))
        per_probe_holes.append(hits)
    if any(not lst for lst in per_probe_holes):
        return []

    out: list[dict] = []
    for combo in itertools.product(*per_probe_holes):
        holes = tuple(c[0] for c in combo)
        if len(set(holes)) != len(holes):
            continue
        # Envelope intersection
        lo = max(c[1] for c in combo)
        hi = min(c[2] for c in combo)
        if lo > hi:
            continue
        ap_mid = 0.5 * (lo + hi)

        # Pick anchor per probe near midpoint via O(log N) searchsorted
        # over precomputed arrays. Replaces the ~ms-per-call
        # ``min(anchors, key=lambda a: ...)`` Python scan.
        seed_idx = []
        for k, pi in enumerate(probe_indices):
            hid = holes[k]
            aps_arr = aa.ap_sorted[(pi, hid)]
            seed_idx.append(_nearest_ap_index(aps_arr, ap_mid))

        seed_ml = [
            float(aa.ml_sorted[(probe_indices[k], holes[k])][seed_idx[k]])
            for k in range(len(probe_indices))
        ]
        seed_spin = [
            float(aa.spin_sorted[(probe_indices[k], holes[k])][seed_idx[k]])
            for k in range(len(probe_indices))
        ]
        seed_ap = [
            float(aa.ap_sorted[(probe_indices[k], holes[k])][seed_idx[k]])
            for k in range(len(probe_indices))
        ]

        # ml-sep feasibility check. The atlas envelope is conservative —
        # SLSQP can move ml beyond atlas anchors (threading is soft).
        # So the right check is "max-possible pairwise diff ≥ min_ml_sep",
        # not "found atlas-anchor combo satisfying ml-sep". The latter
        # rejects manual-quality hole-tuples that sit at the boundary
        # (manual T12 arc c has max diff = 16.33° at atlas envelopes but
        # the polish moves ml outside to satisfy more comfortably).
        if len(seed_ml) >= 2:
            # Compute per-probe ml range from precomputed sorted arrays
            ml_ranges = []
            for k in range(len(probe_indices)):
                mls_arr = aa.ml_sorted[(probe_indices[k], holes[k])]
                # Restrict to anchors within the envelope intersection
                aps_arr = aa.ap_sorted[(probe_indices[k], holes[k])]
                mask = (aps_arr >= lo) & (aps_arr <= hi)
                if not mask.any():
                    # Fall back to full range if envelope filter empties
                    sel = mls_arr
                else:
                    sel = mls_arr[mask]
                ml_ranges.append((float(sel.min()), float(sel.max())))

            # Max-possible pairwise diff per (a, b)
            max_possible_min_diff = float("inf")
            for a in range(len(ml_ranges)):
                for b in range(a + 1, len(ml_ranges)):
                    la, ha_ = ml_ranges[a]
                    lb, hb = ml_ranges[b]
                    # Max |x - y| over x ∈ [la, ha_], y ∈ [lb, hb]
                    max_pair = max(abs(ha_ - lb), abs(hb - la))
                    if max_pair < max_possible_min_diff:
                        max_possible_min_diff = max_pair

            if max_possible_min_diff < min_ml_sep_deg:
                continue  # truly infeasible — no ml combo can satisfy

            # Track best-case diff at the seed anchors (informational)
            min_diff = min(
                abs(seed_ml[a] - seed_ml[b])
                for a in range(len(seed_ml))
                for b in range(a + 1, len(seed_ml))
            )
        else:
            min_diff = float("inf")

        total_anchors = sum(
            len(aa.ap_sorted[(probe_indices[k], holes[k])])
            for k in range(len(probe_indices))
        )
        out.append({
            "holes": holes,
            "ap_lo": lo,
            "ap_hi": hi,
            "ap_mid": ap_mid,
            "seed_ml": seed_ml,
            "seed_spin": seed_spin,
            "seed_ap": seed_ap,
            "min_ml_diff": min_diff,
            "total_anchors": total_anchors,
        })
    return out


def _find_ml_sep_anchors_arr(
    probe_indices: tuple[int, ...],
    holes: tuple[int, ...],
    target_ap: float,
    min_ml_sep_deg: float,
    aa: _AtlasArrays,
    max_calls: int = 5000,
    max_anchors_per_probe: int = 200,
) -> tuple[list[float], list[float], list[float]] | None:
    """Backtrack to find a per-probe anchor combination satisfying
    pairwise ml-sep, preferring anchors close to ``target_ap``.

    Bounded by ``max_calls`` to prevent worst-case 9K^n explosion on
    infeasible hole-tuples (treats unbounded-search as infeasible).
    Also caps per-probe anchor pool at ``max_anchors_per_probe`` (the
    closest-in-AP subset); for typical (probe, hole) entries with ~9K
    anchors, only the top-200 closest to target_ap are meaningful
    seeds — the rest are far in AP and would polish to the same basin.
    """
    n = len(probe_indices)
    if n == 0:
        return ([], [], [])
    if n == 1:
        aps_arr = aa.ap_sorted[(probe_indices[0], holes[0])]
        idx = _nearest_ap_index(aps_arr, target_ap)
        return (
            [float(aa.ml_sorted[(probe_indices[0], holes[0])][idx])],
            [float(aa.spin_sorted[(probe_indices[0], holes[0])][idx])],
            [float(aps_arr[idx])],
        )

    # For each probe, take the top-K closest-AP anchors as candidates.
    sorted_per_probe = []
    for k in range(n):
        key = (probe_indices[k], holes[k])
        aps_arr = aa.ap_sorted[key]
        mls_arr = aa.ml_sorted[key]
        spins_arr = aa.spin_sorted[key]
        # Closest-AP subset, capped at max_anchors_per_probe
        order = np.argsort(np.abs(aps_arr - target_ap))[:max_anchors_per_probe]
        sorted_per_probe.append((
            mls_arr[order], spins_arr[order], aps_arr[order]
        ))

    best: list = [None]
    call_count = [0]

    def search(idx: int, mls: list[float], spins: list[float], aps: list[float]) -> bool:
        call_count[0] += 1
        if call_count[0] >= max_calls:
            return False
        if idx == n:
            best[0] = (mls.copy(), spins.copy(), aps.copy())
            return True
        mls_arr, spins_arr, aps_arr = sorted_per_probe[idx]
        for j in range(len(mls_arr)):
            if call_count[0] >= max_calls:
                return False
            cand_ml = float(mls_arr[j])
            ok = True
            for prev_ml in mls:
                if abs(cand_ml - prev_ml) < min_ml_sep_deg:
                    ok = False
                    break
            if not ok:
                continue
            mls.append(cand_ml)
            spins.append(float(spins_arr[j]))
            aps.append(float(aps_arr[j]))
            if search(idx + 1, mls, spins, aps):
                return True
            mls.pop()
            spins.pop()
            aps.pop()
        return False

    found = search(0, [], [], [])
    if not found:
        return None
    return best[0]  # type: ignore


def ml_anchors_mrv(  # noqa: C901
    anchor_sets: "list[tuple[np.ndarray, np.ndarray, np.ndarray]]",
    target_ap: float,
    min_ml_sep_deg: float,
    *,
    max_calls: int = 5000,
    max_anchors_per_probe: int = 200,
) -> "tuple[list[tuple[float, float, float]], float] | None":
    """MRV/CSP pick of one ``(ml, spin, ap)`` anchor per probe, MLs as far apart
    as the atlas allows (target ≥``min_ml_sep_deg``), each anchor as close in AP
    to ``target_ap`` as possible.

    ``anchor_sets[p] = (mls, spins, aps)`` are the candidate atlas anchors for
    probe ``p`` (one (probe, hole)'s anchors).

    Returns ``(assignment, min_gap)`` — the chosen ``(ml, spin, ap)`` per probe
    and the achieved minimum pairwise ML gap — or ``None`` only if some probe
    has NO anchors at all. **Best-effort, not strict:** if a ≥``min_ml_sep_deg``
    combination exists it returns the closest-AP one with ``min_gap ≥
    min_ml_sep_deg``; otherwise (the atlas, sampled at offset=0, can't separate
    these probes) it returns the **max-min-gap** combination with ``min_gap <
    min_ml_sep_deg``. The downstream optimizer still enforces the hard 16°
    separation (and reaches the true config via offsets the atlas doesn't
    sample, as the manual plan does); ``min_gap`` is a ranking/quality flag, not
    a reject. The FCL gate is the real feasibility check.

    Generalizes :func:`_find_ml_sep_anchors_arr` with **dynamic MRV** variable
    ordering (assign the probe with the fewest viable anchors first) +
    closest-AP value ordering + backtracking (bounded by ``max_calls``). Emits
    ``spin`` alongside ``ml``.
    """
    n = len(anchor_sets)
    if n == 0:
        return [], float("inf")
    cand: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for mls, spins, aps in anchor_sets:
        mls = np.asarray(mls, dtype=np.float64)
        spins = np.asarray(spins, dtype=np.float64)
        aps = np.asarray(aps, dtype=np.float64)
        order = np.argsort(np.abs(aps - target_ap))[:max_anchors_per_probe]
        cand.append((mls[order], spins[order], aps[order]))
    if any(c[0].size == 0 for c in cand):
        return None  # a probe has no anchors → truly impossible

    def search(sep: float) -> "list[tuple[float, float, float]] | None":
        chosen: dict[int, tuple[float, float, float]] = {}
        calls = [0]

        def rec(placed_mls: list[float], remaining: frozenset) -> bool:
            calls[0] += 1
            if calls[0] > max_calls:
                return False
            if not remaining:
                return True
            viab: dict[int, np.ndarray] = {}
            for p in remaining:
                mls = cand[p][0]
                mask = np.ones(mls.size, dtype=bool)
                for pm in placed_mls:
                    mask &= np.abs(mls - pm) >= sep
                viab[p] = mask
            p = min(remaining, key=lambda q: int(viab[q].sum()))
            if not viab[p].any():
                return False
            mls, spins, aps = cand[p]
            rest = remaining - {p}
            for j in np.nonzero(viab[p])[0]:        # closest-AP order
                chosen[p] = (float(mls[j]), float(spins[j]), float(aps[j]))
                if rec(placed_mls + [float(mls[j])], rest):
                    return True
                del chosen[p]
            return False

        if rec([], frozenset(range(n))):
            return [chosen[p] for p in range(n)]
        return None

    def min_gap(combo: "list[tuple[float, float, float]]") -> float:
        if n < 2:
            return float("inf")
        m = [c[0] for c in combo]
        return min(abs(m[i] - m[j]) for i in range(n) for j in range(i + 1, n))

    combo = search(min_ml_sep_deg)              # strict first (common path)
    if combo is not None:
        return combo, min_gap(combo)
    # Soft fallback: binary-search the largest achievable separation, return
    # the max-min-gap combo. sep=0 always succeeds (nearest-AP anchors).
    best = search(0.0)
    lo, hi = 0.0, min_ml_sep_deg
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        c = search(mid)
        if c is not None:
            best, lo = c, mid
        else:
            hi = mid
    return best, min_gap(best)  # type: ignore[arg-type]


def emit_seed(
    arcs: "list[dict]",
    aa,
    *,
    min_arc_ap_sep_deg: float = 16.0,
    min_ml_sep_deg: float = 16.0,
) -> "tuple[list[float], dict[str, float], dict[str, float], float] | None":
    """Joint AP+ML+spin seed for a fixed (probe→hole, probe→arc) assignment.

    The single source of truth for seed emission — replaces both the
    midpoint arc-AP and the band-edge ML packing. AP layer is a convex
    projection (:func:`bounded_isotonic_arc_aps`); ML layer is the MRV/CSP
    anchor pick (:func:`ml_anchors_mrv`) at each arc's placed AP, which also
    yields spin.

    Parameters
    ----------
    arcs
        One dict per arc, with keys: ``members`` (list of
        ``(probe_idx, hole_id, probe_name)``), ``ap_lo`` / ``ap_hi`` (the
        arc's feasible AP window = member AP-envelope intersection), and
        ``ap_desired`` (the arc's preferred AP, e.g. the member required-AP
        centroid).
    aa
        Atlas arrays exposing ``ml_sorted`` / ``spin_sorted`` / ``ap_sorted``
        dicts keyed by ``(probe_idx, hole_id)``.

    Returns
    -------
    ``(arc_aps, ml_seed, spin_seed, min_ml_gap)`` — separated arc APs (one per
    arc, input order), per-probe-name ml / spin from real anchors, and the
    smallest achieved within-arc ML gap across all arcs. ``min_ml_gap <
    min_ml_sep_deg`` flags a best-effort (atlas-limited) seed — NOT a reject;
    the optimizer enforces the hard separation downstream. Returns ``None``
    only if some probe has no atlas anchors at all (degenerate).
    """
    if not arcs:
        return [], {}, {}, float("inf")
    desired = np.array([a["ap_desired"] for a in arcs], dtype=np.float64)
    lows = np.array([a["ap_lo"] for a in arcs], dtype=np.float64)
    highs = np.array([a["ap_hi"] for a in arcs], dtype=np.float64)
    arc_aps = bounded_isotonic_arc_aps(desired, lows, highs, min_arc_ap_sep_deg)

    ml_seed: dict[str, float] = {}
    spin_seed: dict[str, float] = {}
    min_ml_gap = float("inf")
    for j, arc in enumerate(arcs):
        anchor_sets = [
            (aa.ml_sorted[(p, h)], aa.spin_sorted[(p, h)], aa.ap_sorted[(p, h)])
            for (p, h, _name) in arc["members"]
        ]
        res = ml_anchors_mrv(anchor_sets, float(arc_aps[j]), min_ml_sep_deg)
        if res is None:
            return None  # a probe has no anchors at all
        combo, gap = res
        min_ml_gap = min(min_ml_gap, gap)
        for (_p, _h, name), (ml, spin, _ap) in zip(arc["members"], combo):
            ml_seed[name] = float(ml)
            spin_seed[name] = float(spin)
    return [float(x) for x in arc_aps], ml_seed, spin_seed, min_ml_gap


# ---------------------------------------------------------------------------
# Main entry: enumerate (partition, hole-tuple) candidates
# ---------------------------------------------------------------------------


def enumerate_arc_first_candidates(
    probes,
    atlas: Atlas,
    *,
    max_arcs: int = _POSE_LIMITS.max_arcs(),
    max_probes_per_arc: int = _POSE_LIMITS.max_probes_per_arc(),
    min_arc_ap_sep_deg: float = 16.0,
    min_ml_sep_deg: float = 16.0,
    ap_tol_for_centeredness_deg: float = 5.0,
    per_arc_max_hole_tuples: int | None = 50,
    global_max_candidates: int | None = 100_000,
    composite_weights: dict[str, float] | None = None,
    verbose: bool = False,
) -> list[ArcFirstCandidate]:
    """Enumerate all deduplicated `(partition, hole-tuple)` candidates.

    Output is a list of :class:`ArcFirstCandidate`, ordered by
    ``composite_order_score`` (descending — best first). The score is
    NOT a filter; the full list is returned.

    Parameters
    ----------
    probes : list[ProbeStaticInfo]
    atlas : Atlas
    max_arcs : int
        Upper bound on arcs per partition. Default 3.
    min_arc_ap_sep_deg, min_ml_sep_deg : float
        Constraint thresholds.
    ap_tol_for_centeredness_deg : float
        Used only for the AP centeredness signal: `1 - |anchor_ap -
        arc_ap| / ap_tol`. Doesn't enter the dedup decision.
    per_arc_max_hole_tuples : int or None
        Cap per arc on hole-tuple count (by total_anchors desc). None =
        no cap.
    composite_weights : dict or None
        Optional override of the default weighting for the composite
        order score. Default weights are equal across normalized signals.
    verbose : bool
        Print enumeration progress.
    """
    K = len(probes)
    probe_names = [p.name for p in probes]

    # Precompute per-(probe, hole) anchor arrays once (sorted by ap_deg).
    # Replaces per-call min(anchors, key=lambda) which was the hot spot.
    if verbose:
        import time as _time
        _t = _time.perf_counter()
    atlas_arr = _build_atlas_arrays(atlas, probe_names)
    if verbose:
        import time as _time
        print(f"[arc-first] atlas arrays built in {_time.perf_counter() - _t:.2f}s")

    if composite_weights is None:
        composite_weights = {
            "ap_intersection_min_width_deg": 1.0,
            "min_intra_arc_ml_slack_deg": 1.0,
            "total_atlas_anchors": 0.5,
            "ap_centeredness_sum": 0.5,
            "arc_ap_pairwise_min_sep_deg": 0.5,
        }

    # Cache per-arc-group hole-tuple enumeration keyed by group
    # (the SAME (probe_indices, atlas) returns the same hole-tuples
    # regardless of partition siblings).
    _arc_cache: dict[tuple[int, ...], list[dict]] = {}

    def get_arc_hole_tuples(probe_indices):
        key = tuple(probe_indices)
        cached = _arc_cache.get(key)
        if cached is not None:
            return cached
        if verbose:
            import time as _time
            _arc_t = _time.perf_counter()
        out = _enumerate_arc_hole_tuples(
            key, atlas_arr, min_ml_sep_deg=min_ml_sep_deg
        )
        if verbose:
            import time as _time
            _dt = _time.perf_counter() - _arc_t
            if _dt > 0.1:
                print(f"[arc-first] arc-group {key} (size {len(key)}): "
                      f"enumerated {len(out)} hole-tuples in {_dt:.2f}s",
                      flush=True)
        if per_arc_max_hole_tuples is not None and len(out) > per_arc_max_hole_tuples:
            # Sort by AP-intersection width (wider = polish-friendlier) and keep
            # top-N. NB: this is a hard discrete cap; if a hole-tuple's
            # intersection is narrow it's still emit-able when feasible, but we
            # drop it when there are too many alternatives. Used in
            # diagnostic context where exhaustive enumeration would explode.
            out.sort(key=lambda d: -(d["ap_hi"] - d["ap_lo"]))
            out = out[:per_arc_max_hole_tuples]
        _arc_cache[key] = out
        return out

    candidates: list[ArcFirstCandidate] = []
    n_partitions = 0
    n_partitions_kept = 0
    budget_exceeded = False

    for partition in unordered_partitions(K, max_arcs):
        if budget_exceeded:
            break
        n_partitions += 1
        # Skip partitions with very large arcs — practically infeasible
        # (n probes need (n-1)*16° ml-spread). Manual uses ≤3 probes/arc.
        if any(len(g) > max_probes_per_arc for g in partition):
            continue
        # Get per-arc hole-tuples
        per_arc_tuples = [get_arc_hole_tuples(g) for g in partition]
        if any(not lst for lst in per_arc_tuples):
            continue
        n_partitions_kept += 1
        if verbose and n_partitions % 50 == 0:
            print(f"[arc-first] partition {n_partitions}: "
                  f"per-arc tuple counts = {[len(t) for t in per_arc_tuples]}, "
                  f"cumulative candidates = {len(candidates)}",
                  flush=True)

        # Cross-arc Cartesian
        for combo in itertools.product(*per_arc_tuples):
            if (
                global_max_candidates is not None
                and len(candidates) >= global_max_candidates
            ):
                budget_exceeded = True
                break
            # Global hole uniqueness across arcs
            all_holes: set[int] = set()
            conflict = False
            for arc_pick in combo:
                for hid in arc_pick["holes"]:
                    if hid in all_holes:
                        conflict = True
                        break
                    all_holes.add(hid)
                if conflict:
                    break
            if conflict:
                continue

            # Principled arc APs from per-arc midpoints
            arc_aps_raw = [arc_pick["ap_mid"] for arc_pick in combo]

            # Pairwise AP separation check
            sep_ok = True
            for i in range(len(arc_aps_raw)):
                for j in range(i + 1, len(arc_aps_raw)):
                    if abs(arc_aps_raw[i] - arc_aps_raw[j]) < min_arc_ap_sep_deg:
                        sep_ok = False
                        break
                if not sep_ok:
                    break
            if not sep_ok:
                # Could perturb to nearest separated triple; for now
                # reject. (Most rejections come from impossible
                # geometric configurations; perturbing wouldn't help.)
                continue

            # Canonical (ascending AP) ordering for AA arc_centroids_deg
            order = sorted(range(len(arc_aps_raw)), key=lambda i: arc_aps_raw[i])
            canonical_partition = tuple(partition[i] for i in order)
            canonical_aps = tuple(arc_aps_raw[i] for i in order)
            canonical_combo = [combo[i] for i in order]

            # Build HA, AA, seeds
            probe_to_hole: dict[str, int] = {}
            probe_to_arc_idx: dict[str, int] = {}
            ml_seed: dict[str, float] = {}
            spin_seed: dict[str, float] = {}
            for arc_idx, (group, arc_pick) in enumerate(
                zip(canonical_partition, canonical_combo)
            ):
                for k, probe_idx in enumerate(group):
                    name = probe_names[probe_idx]
                    hid = arc_pick["holes"][k]
                    probe_to_hole[name] = hid
                    probe_to_arc_idx[name] = arc_idx
                    ml_seed[name] = arc_pick["seed_ml"][k]
                    spin_seed[name] = arc_pick["seed_spin"][k]

            # Cheap signals
            ap_widths = [
                arc_pick["ap_hi"] - arc_pick["ap_lo"] for arc_pick in canonical_combo
            ]
            ap_intersection_min_width_deg = min(ap_widths)

            arc_ml_slacks: list[float] = []
            for arc_pick in canonical_combo:
                if arc_pick["min_ml_diff"] == float("inf"):
                    continue
                arc_ml_slacks.append(arc_pick["min_ml_diff"] - min_ml_sep_deg)
            min_intra_arc_ml_slack_deg = (
                min(arc_ml_slacks) if arc_ml_slacks else float("inf")
            )

            total_atlas_anchors = sum(c["total_anchors"] for c in canonical_combo)

            ap_centeredness_sum = 0.0
            for arc_idx, arc_pick in enumerate(canonical_combo):
                arc_ap = canonical_aps[arc_idx]
                for anchor_ap in arc_pick["seed_ap"]:
                    ap_centeredness_sum += max(
                        0.0,
                        1.0 - abs(anchor_ap - arc_ap) / ap_tol_for_centeredness_deg,
                    )

            arc_ap_pairwise_min_sep_deg = (
                min(
                    abs(canonical_aps[i] - canonical_aps[j])
                    for i in range(len(canonical_aps))
                    for j in range(i + 1, len(canonical_aps))
                )
                if len(canonical_aps) >= 2 else float("inf")
            )

            # Composite (un-normalized — caller can re-rank if desired)
            ml_slack_for_score = (
                10.0 if min_intra_arc_ml_slack_deg == float("inf")
                else min(min_intra_arc_ml_slack_deg, 30.0)
            )
            arc_sep_for_score = (
                30.0 if arc_ap_pairwise_min_sep_deg == float("inf")
                else min(arc_ap_pairwise_min_sep_deg, 60.0)
            )
            comp = (
                composite_weights["ap_intersection_min_width_deg"] * ap_intersection_min_width_deg
                + composite_weights["min_intra_arc_ml_slack_deg"] * ml_slack_for_score
                + composite_weights["total_atlas_anchors"] * np.log1p(total_atlas_anchors)
                + composite_weights["ap_centeredness_sum"] * ap_centeredness_sum
                + composite_weights["arc_ap_pairwise_min_sep_deg"] * arc_sep_for_score
            )

            components = {
                "ap_intersection_min_width_deg": float(ap_intersection_min_width_deg),
                "min_intra_arc_ml_slack_deg": float(min_intra_arc_ml_slack_deg),
                "total_atlas_anchors": int(total_atlas_anchors),
                "ap_centeredness_sum": float(ap_centeredness_sum),
                "arc_ap_pairwise_min_sep_deg": float(arc_ap_pairwise_min_sep_deg),
            }

            ha = HoleAssignment(probe_to_hole=probe_to_hole, cost=-comp)
            aa = ArcAssignment(
                probe_to_arc_idx=probe_to_arc_idx,
                arc_centroids_deg=tuple(float(a) for a in canonical_aps),
                cost=-comp,
            )
            candidates.append(ArcFirstCandidate(
                ha=ha, aa=aa,
                ml_seed=ml_seed, spin_seed=spin_seed,
                ap_intersection_min_width_deg=float(ap_intersection_min_width_deg),
                min_intra_arc_ml_slack_deg=float(min_intra_arc_ml_slack_deg),
                total_atlas_anchors=int(total_atlas_anchors),
                ap_centeredness_sum=float(ap_centeredness_sum),
                arc_ap_pairwise_min_sep_deg=float(arc_ap_pairwise_min_sep_deg),
                composite_order_score=float(comp),
                components=components,
            ))

    candidates.sort(key=lambda c: -c.composite_order_score)
    if verbose:
        print(f"[arc-first] enumerated {n_partitions} partitions "
              f"({n_partitions_kept} with valid arcs); "
              f"{len(candidates)} candidates")
    return candidates


def find_target_in_candidates(
    candidates: list[ArcFirstCandidate],
    target_ha: dict[str, int],
) -> int | None:
    """Return the rank (by composite order) of the first candidate
    matching ``target_ha`` (probe_name → hole_id), or ``None``.
    """
    target = frozenset(target_ha.items())
    for rank, c in enumerate(candidates):
        if frozenset(c.ha.probe_to_hole.items()) == target:
            return rank
    return None
