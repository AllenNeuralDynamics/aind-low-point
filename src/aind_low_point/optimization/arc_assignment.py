"""Middle-layer probe→arc assignment.

Given a probe→hole assignment from the outer layer, every probe has an
extracted hole axis. We project that axis onto the rig's AP plane to
get a **required-AP** angle per probe; arc assignment is then a 1-D
constrained partition of those K required-APs into ``num_arcs``
labelled groups.

Constraints / costs:

- **Per-arc capacity (hard).** Kinematic only: with the ML range
  (±60° ⇒ 120° span) and 16° pairwise minimum, an arc holds at most
  ``floor(120/16) + 1 = 8`` probes. There is NO hardware mount-count
  limit (the rig takes more than 4 per arc); see
  ``PoseLimits.max_probes_per_arc()``.
- **Inter-arc AP separation (soft).** Cluster centroids ideally sit
  ≥16° apart pairwise. The hard rig constraint (`ap_arc_{σ(j+1)} ≥
  ap_arc_{σ(j)} + 16°`) applies to the inner loop's continuous
  ``ap_arc`` variables, *not* the cluster centroids — those are warm-
  starts. So we score centroid-spread shortfall as a soft penalty
  (Σ over arc pairs of ``max(0, 16 − |Δc|)²``) and let the inner loop
  decide whether the threading + coverage + clearance cost of pushing
  centroids apart is worth it.

Symmetry quotient: arc *labels* are interchangeable; canonicalize by
ordering arcs ascending in their AP centroid. Reduces the assignment
count by ``num_arcs!`` and removes label-permutation duplicates from
the top-K_a output.

For K = 7 probes into 2-4 arcs, ``num_arcs ** K`` is at most 16,384.
The capacity filter culls aggressively; after symmetry quotient the
surviving count is typically a few hundred per hole assignment, which
``solve_top_k_arc_assignments`` truncates to ``k`` (default 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import isotonic_regression

from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.kinematics import required_ap_deg
from aind_low_point.planning import PoseLimits

# Per-arc capacity is KINEMATIC (16 deg ML exclusion over the ML range), not a
# hardware mount count — see PoseLimits.max_probes_per_arc().
_POSE_LIMITS = PoseLimits()

# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArcAssignment:
    """One probe→arc assignment after capacity / AP-separation filtering.

    Attributes
    ----------
    probe_to_arc_idx : dict
        ``probe_name -> arc_index`` (0-based, in canonical AP order).
    arc_centroids_deg : tuple
        Required-AP centroid per arc (sorted ascending). Suitable as
        the warm-start for the inner loop's ``ap_arc`` variable.
    cost : float
        Within-cluster sum of squared deviations from the centroid.
        Smaller = more tightly clustered probes per arc = warmer
        start for SLSQP.
    """

    probe_to_arc_idx: dict[str, int]
    arc_centroids_deg: tuple[float, ...]
    cost: float


# ---------------------------------------------------------------------------
# Required-AP per probe under a hole assignment
# ---------------------------------------------------------------------------


def required_aps_deg_for_assignment(
    probe_to_hole: dict[str, int],
    holes: Iterable[Hole],
) -> dict[str, float]:
    """For each probe, look up the assigned hole and compute its
    subject-frame required-AP angle (degrees).

    Returns a dict ``probe_name -> required_ap_deg`` matching the
    input dict's keys.
    """
    holes_by_id = {h.id: h for h in holes}
    out: dict[str, float] = {}
    for probe_name, hole_id in probe_to_hole.items():
        hole = holes_by_id.get(hole_id)
        if hole is None:
            raise KeyError(f"hole id={hole_id} not in provided holes list")
        out[probe_name] = required_ap_deg(hole.axis)
    return out


# ---------------------------------------------------------------------------
# Partition validation
# ---------------------------------------------------------------------------


def _arc_centroids(aps: NDArray, arc_idxs: NDArray, num_arcs: int) -> NDArray:
    """Per-arc mean of the AP angles assigned to that arc. Empty arcs
    return NaN for that slot — callers detect via ``isnan``."""
    out = np.full(num_arcs, np.nan, dtype=np.float64)
    for k in range(num_arcs):
        members = aps[arc_idxs == k]
        if len(members) > 0:
            out[k] = float(np.mean(members))
    return out


def _within_cluster_cost(aps: NDArray, arc_idxs: NDArray, num_arcs: int) -> float:
    """Sum of squared deviations from each arc's centroid."""
    total = 0.0
    for k in range(num_arcs):
        members = aps[arc_idxs == k]
        if len(members) == 0:
            continue
        c = float(np.mean(members))
        total += float(np.sum((members - c) ** 2))
    return total


def _is_valid_partition(
    arc_idxs: NDArray,
    centroids: NDArray,
    *,
    max_per_arc: int,
) -> bool:
    """Hard capacity + non-empty-arc check.

    AP-separation is intentionally **not** a hard gate here: it's a
    constraint on the inner loop's continuous ``ap_arc`` variables, not
    on the warm-start centroid. See ``_arc_sep_shortfall_sq``.
    """
    num_arcs = len(centroids)
    # Per-arc capacity is kinematic (16° ML exclusion over the ML range),
    # NOT a hardware count — the rig takes more than 4 per arc.
    counts = np.bincount(arc_idxs, minlength=num_arcs)
    if counts.max() > max_per_arc:
        return False
    # Require every arc to have at least one probe (we use the same
    # num_arcs the rig supplies; un-occupied arcs are OK in principle
    # but redundant — emit only fully-populated partitions for now).
    if (counts == 0).any():
        return False
    return True


def project_centroids_min_sep(centroids: NDArray, min_sep_deg: float) -> NDArray:
    """L2-optimal projection of arc centroids onto the chained constraint
    ``c_{σ(k+1)} − c_{σ(k)} ≥ min_sep`` (in sorted order).

    The rig's hard 16° AP-separation acts on the *continuous* arc-AP
    variables, not the cluster mean. So when the natural cluster
    centroids are too close, the inner loop has to push them apart
    anyway — at the cost of tilting probes away from their preferred
    bore-axis APs. This function precomputes the projected centroids
    so the partitioner can rank partitions by the resulting *tilt
    cost* (sum of squared probe-vs-arc deviations) rather than the
    natural within-cluster variance.

    Reduction: substitute ``d_k = c'_{σ(k)} − k · min_sep``; the
    constraint becomes ``d_k ≥ d_{k-1}`` — isotonic regression on
    ``(c_{σ(k)} − k · min_sep)``. Solve with PAV.
    """
    c = np.asarray(centroids, dtype=np.float64)
    n = c.size
    if n <= 1:
        return c.copy()
    order = np.argsort(c)
    inv = np.empty(n, dtype=np.int64)
    for i, j in enumerate(order):
        inv[j] = i
    c_sorted = c[order]
    # Reduce to isotonic regression on (c_k - k*min_sep).
    y = c_sorted - min_sep_deg * np.arange(n, dtype=np.float64)
    # Pool-adjacent-violators.
    levels = list(y.tolist())
    counts = [1] * n
    k = 0
    while k < len(levels) - 1:
        if levels[k] > levels[k + 1]:
            new_count = counts[k] + counts[k + 1]
            new_level = (
                levels[k] * counts[k] + levels[k + 1] * counts[k + 1]
            ) / new_count
            levels[k : k + 2] = [new_level]
            counts[k : k + 2] = [new_count]
            if k > 0:
                k -= 1
        else:
            k += 1
    # Expand back to length n.
    d = np.empty(n, dtype=np.float64)
    pos = 0
    for level, count in zip(levels, counts):
        d[pos : pos + count] = level
        pos += count
    c_proj_sorted = d + min_sep_deg * np.arange(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[order] = c_proj_sorted
    return out


def _qp_chain_box(c: NDArray, lo: NDArray, hi: NDArray, sep: float) -> NDArray:
    """Exact L2 projection onto (ascending chain ≥``sep``) ∩ (box ``[lo,hi]``)
    for a SORTED-by-centroid input. Small dense QP via SLSQP — only invoked on
    the rare path where the box-free isotonic optimum leaves a window."""
    from scipy.optimize import minimize

    n = len(c)
    cons = [
        {"type": "ineq", "fun": (lambda a, i=i: a[i + 1] - a[i] - sep)}
        for i in range(n - 1)
    ]
    x0 = np.clip(c, lo, hi)
    res = minimize(
        lambda a: float(np.sum((a - c) ** 2)), x0, method="SLSQP",
        bounds=list(zip(lo.tolist(), hi.tolist())), constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    return np.asarray(res.x, dtype=np.float64)


def bounded_isotonic_arc_aps(
    centroids: NDArray,
    lows: NDArray,
    highs: NDArray,
    min_sep_deg: float,
) -> NDArray:
    """Place arc APs as close as possible (L2) to their preferred
    ``centroids`` subject to BOTH the chained ≥``min_sep_deg`` AP separation
    AND each arc's feasible window ``[lows[k], highs[k]]``.

    Unlike a greedy stab, this keeps slack where it exists (arcs stay at
    their centroids when already ≥min_sep apart) and packs optimally where
    the separation binds (the deficit is split across the squeezed arcs).
    Generalizes :func:`project_centroids_min_sep` with per-arc box bounds.

    Method: substitute ``d_k = ap_{σ(k)} − k·sep`` (σ = ascending-centroid
    order) so the chain constraint becomes monotone ``d_0 ≤ d_1 ≤ …``; the
    box-FREE optimum is then ``scipy.optimize.isotonic_regression`` (exact
    PAVA). If that already lands inside every window the box constraint is
    inactive and it IS the constrained optimum; otherwise (rare) a window
    clips and we solve the small exact QP. The caller guarantees the
    feasible region is non-empty (the enumerator's AP-sep gate).

    Returns arc APs in the ORIGINAL input order.
    """
    c = np.asarray(centroids, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    hi = np.asarray(highs, dtype=np.float64)
    n = c.size
    if n == 0:
        return c.copy()
    if n == 1:
        return np.clip(c, lo, hi)

    order = np.argsort(c, kind="stable")
    k = np.arange(n, dtype=np.float64)
    cs, los, his = c[order], lo[order], hi[order]
    # Box-free monotone optimum in d-space, then undo the substitution.
    d = isotonic_regression(cs - k * min_sep_deg, increasing=True).x
    aps_sorted = np.asarray(d) + k * min_sep_deg

    if not ((aps_sorted >= los - 1e-9).all() and (aps_sorted <= his + 1e-9).all()):
        aps_sorted = _qp_chain_box(cs, los, his, min_sep_deg)  # box active

    out = np.empty(n, dtype=np.float64)
    out[order] = aps_sorted
    return out


def _arc_sep_shortfall_sq(centroids: NDArray, min_arc_ap_sep_deg: float) -> float:
    """``Σ_{i<j} max(0, min_sep − |c_i − c_j|)²`` — soft AP-separation
    cost on cluster centroids.

    Zero when every pair of arc centroids is already ≥``min_sep`` apart
    (i.e. the warm-start respects the hard rig constraint without any
    inner-loop AP-pushing). Strictly positive otherwise; the inner
    loop's chained ``ap_arc`` constraint will spread the centroids by
    roughly the missing degrees, paying a coverage / threading cost
    we capture here as a ranking signal.
    """
    n = len(centroids)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(float(centroids[i]) - float(centroids[j]))
            short = max(0.0, min_arc_ap_sep_deg - d)
            total += short * short
    return total


def _canonical_arc_relabel(
    arc_idxs: NDArray, centroids: NDArray
) -> tuple[NDArray, NDArray]:
    """Reorder arc labels ascending in centroid AP. Returns
    ``(canonical_arc_idxs, canonical_centroids)``."""
    order = np.argsort(centroids)
    label_map = np.empty_like(order)
    for new_label, old_label in enumerate(order):
        label_map[old_label] = new_label
    return label_map[arc_idxs], centroids[order]


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_partitions(
    probe_names: list[str],
    required_aps_deg: NDArray,
    num_arcs: int,
    *,
    max_per_arc: int = _POSE_LIMITS.max_probes_per_arc(),
    min_arc_ap_sep_deg: float = 16.0,
    arc_sep_shortfall_weight: float = 10.0,
) -> list[ArcAssignment]:
    """Brute-force enumerate every probe→arc partition, hard-filter by
    capacity, score with ``within-cluster variance + arc_sep_shortfall``,
    deduplicate by canonical arc ordering, return survivors ranked by
    cost (best first).

    For K probes and ``num_arcs`` ≤ 4, the search space is
    ``num_arcs ** K`` (≤ 16,384 for the AIND rig). The capacity filter
    cuts most candidates; arc-separation enters as a soft cost so the
    inner loop sees marginally-spaced centroids too (its chained
    ``ap_arc`` constraint can push them apart at the cost of coverage
    and threading violations — a tradeoff the caller surfaces by
    inspecting the inner-loop result).

    Parameters
    ----------
    arc_sep_shortfall_weight
        Weight on ``Σ_{i<j} max(0, min_sep − |Δc_ij|)²`` (degrees²).
        Default 10.0 — empirically large enough that a fully feasible
        partition (zero shortfall) ranks above a partition needing 5°
        of centroid-spreading (cost = 250) for typical 7-probe within-
        cluster spreads (~10–30 deg²), but small enough that the inner
        loop still gets a chance to optimize a marginal partition. Set
        to ``+inf`` to recover the legacy hard-AP-sep filter.
    """
    K = len(probe_names)
    if K == 0 or num_arcs == 0:
        return []
    aps = np.asarray(required_aps_deg, dtype=np.float64)
    if aps.shape != (K,):
        raise ValueError(f"required_aps_deg has shape {aps.shape}; expected ({K},)")

    seen_canonical: set[tuple[int, ...]] = set()
    candidates: list[ArcAssignment] = []

    for assignment in product(range(num_arcs), repeat=K):
        arc_idxs = np.asarray(assignment, dtype=np.int64)
        centroids = _arc_centroids(aps, arc_idxs, num_arcs)
        if np.isnan(centroids).any():
            continue
        if not _is_valid_partition(
            arc_idxs,
            centroids,
            max_per_arc=max_per_arc,
        ):
            continue
        canonical_idxs, canonical_centroids = _canonical_arc_relabel(
            arc_idxs, centroids
        )
        sig = tuple(int(i) for i in canonical_idxs)
        if sig in seen_canonical:
            continue
        seen_canonical.add(sig)

        # Project centroids onto the rig's ≥min_sep constraint and rank
        # by the resulting tilt cost (sum of squared probe-vs-arc
        # deviations using the *projected* centroids). This surfaces
        # partitions whose probes can be deliberately tilted to give the
        # inner loop a kinematically feasible warm start, even when the
        # natural means cluster too tightly.
        projected_centroids = project_centroids_min_sep(
            canonical_centroids, min_arc_ap_sep_deg
        )
        tilt_cost = float(np.sum((aps - projected_centroids[canonical_idxs]) ** 2))
        shortfall_sq = _arc_sep_shortfall_sq(canonical_centroids, min_arc_ap_sep_deg)
        if not np.isfinite(arc_sep_shortfall_weight) and shortfall_sq > 0.0:
            # Treat ``+inf`` weight as a hard reject — convenience knob.
            continue
        # Tilt cost dominates ranking; the soft shortfall penalty stays
        # as a small additional preference for partitions where natural
        # means already respect the constraint.
        cost = tilt_cost + arc_sep_shortfall_weight * shortfall_sq
        candidates.append(
            ArcAssignment(
                probe_to_arc_idx={
                    probe_names[i]: int(canonical_idxs[i]) for i in range(K)
                },
                # Hand the inner loop the *projected* centroids as the
                # warm start so SLSQP doesn't waste Stage A pushing them
                # apart from a tightly clustered seed.
                arc_centroids_deg=tuple(float(c) for c in projected_centroids),
                cost=cost,
            )
        )

    candidates.sort(key=lambda a: a.cost)
    return candidates


def solve_top_k_arc_assignments(
    probe_to_hole: dict[str, int],
    holes: Iterable[Hole],
    *,
    max_num_arcs: int,
    k: int,
    min_num_arcs: int = 1,
    max_per_arc: int = _POSE_LIMITS.max_probes_per_arc(),
    min_arc_ap_sep_deg: float = 16.0,
    arc_sep_shortfall_weight: float = 10.0,
    arc_count_penalty_deg2: float = 25.0,
) -> list[ArcAssignment]:
    """End-to-end: from a hole assignment, enumerate arc partitions for
    every ``num_arcs ∈ [min_num_arcs, max_num_arcs]``, apply the per-arc
    penalty, return the top-``k`` ranked by *penalised* cost.

    Parameters
    ----------
    max_num_arcs
        Rig hardware limit. Some rigs are 2-arc, others 3-arc, others
        4-arc — surface this from the rig's config.
    min_num_arcs
        Floor on arc count, default 1. Useful when a rig physically
        requires at least one arc; rarely needs raising.
    arc_count_penalty_deg2
        Cost added per arc beyond ``min_num_arcs``, in degrees².
        Default 25.0 ≈ "≤ 5° per-probe cluster spread is worth one
        extra arc" — encodes the "fewer arcs is better" preference
        without overruling cases where extra arcs really help cluster
        tightness. Set to 0.0 to remove the preference and rank by
        within-cluster variance only.
    arc_sep_shortfall_weight
        Soft penalty per ``deg²`` on AP-centroid pairs closer than
        ``min_arc_ap_sep_deg``. Forwarded to :func:`enumerate_partitions`;
        see that function for the rationale.
    """
    if k <= 0:
        return []
    if min_num_arcs < 1 or max_num_arcs < min_num_arcs:
        raise ValueError(
            f"need 1 <= min_num_arcs <= max_num_arcs; "
            f"got min={min_num_arcs}, max={max_num_arcs}"
        )

    probe_names = list(probe_to_hole.keys())
    aps_dict = required_aps_deg_for_assignment(probe_to_hole, holes)
    aps = np.array([aps_dict[name] for name in probe_names], dtype=np.float64)

    all_candidates: list[ArcAssignment] = []
    for n_arcs in range(min_num_arcs, max_num_arcs + 1):
        candidates = enumerate_partitions(
            probe_names,
            aps,
            n_arcs,
            max_per_arc=max_per_arc,
            min_arc_ap_sep_deg=min_arc_ap_sep_deg,
            arc_sep_shortfall_weight=arc_sep_shortfall_weight,
        )
        penalty = arc_count_penalty_deg2 * (n_arcs - min_num_arcs)
        if penalty != 0.0:
            candidates = [
                ArcAssignment(
                    probe_to_arc_idx=c.probe_to_arc_idx,
                    arc_centroids_deg=c.arc_centroids_deg,
                    cost=c.cost + penalty,
                )
                for c in candidates
            ]
        all_candidates.extend(candidates)

    all_candidates.sort(key=lambda a: a.cost)
    return all_candidates[:k]
