"""Shared atlas packing and AP/ML/spin seed emission for enumeration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aind_low_point.optimization.enumeration.arc_placement import (
    bounded_isotonic_arc_aps,
)
from aind_low_point.optimization.enumeration.atlas import Atlas


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
    ml_sorted: dict  # parallel to ap_sorted
    spin_sorted: dict  # parallel to ap_sorted
    ap_min_max: dict  # (probe_idx, hole_id) -> (ap_min, ap_max)


def _build_atlas_arrays(atlas: Atlas, probe_names: list[str]) -> _AtlasArrays:
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

    Uses **dynamic MRV** variable ordering (assign the probe with the fewest
    viable anchors first), closest-AP value ordering, and bounded
    backtracking. Emits ``spin`` alongside ``ml``.
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
            for j in np.nonzero(viab[p])[0]:  # closest-AP order
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

    combo = search(min_ml_sep_deg)  # strict first (common path)
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
