"""Middle-layer probe→arc assignment.

Given a probe→hole assignment from the outer layer, every probe has an
extracted hole axis. We project that axis onto the rig's AP plane to
get a **required-AP** angle per probe; arc assignment is then a 1-D
constrained partition of those K required-APs into ``num_arcs``
labelled groups.

Constraints:

- **Per-arc capacity.** With ML range ±30° and 16° pairwise minimum,
  an arc holds at most ``floor(60/16) + 1 = 4`` probes.
- **Inter-arc AP separation.** Cluster centroids must be ≥16° apart
  pairwise (default ``min_arc_ap_sep_deg``).

Symmetry quotient: arc *labels* are interchangeable; canonicalize by
ordering arcs ascending in their AP centroid. Reduces the assignment
count by ``num_arcs!`` and removes label-permutation duplicates from
the top-K_a output.

For K = 7 probes into 2-4 arcs, ``num_arcs ** K`` is at most 16,384.
After capacity + AP-sep filters and symmetry quotient, the surviving
count is typically 5-50 partitions per hole assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.kinematics import required_ap_deg


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
    probe_to_hole: dict[str, int], holes: Iterable[Hole]
) -> dict[str, float]:
    """For each probe, look up the assigned hole and compute its
    required-AP angle (degrees).

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


def _arc_centroids(
    aps: NDArray, arc_idxs: NDArray, num_arcs: int
) -> NDArray:
    """Per-arc mean of the AP angles assigned to that arc. Empty arcs
    return NaN for that slot — callers detect via ``isnan``."""
    out = np.full(num_arcs, np.nan, dtype=np.float64)
    for k in range(num_arcs):
        members = aps[arc_idxs == k]
        if len(members) > 0:
            out[k] = float(np.mean(members))
    return out


def _within_cluster_cost(
    aps: NDArray, arc_idxs: NDArray, num_arcs: int
) -> float:
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
    min_arc_ap_sep_deg: float,
) -> bool:
    """Capacity + AP-separation feasibility check."""
    num_arcs = len(centroids)
    # Capacity per arc
    counts = np.bincount(arc_idxs, minlength=num_arcs)
    if counts.max() > max_per_arc:
        return False
    # Require every arc to have at least one probe (we use the same
    # num_arcs the rig supplies; un-occupied arcs are OK in principle
    # but redundant — emit only fully-populated partitions for now).
    if (counts == 0).any():
        return False
    # AP separation: centroids must be pairwise ≥ threshold apart
    for i in range(num_arcs):
        for j in range(i + 1, num_arcs):
            if abs(centroids[i] - centroids[j]) < min_arc_ap_sep_deg:
                return False
    return True


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
    max_per_arc: int = 4,
    min_arc_ap_sep_deg: float = 16.0,
) -> list[ArcAssignment]:
    """Brute-force enumerate every probe→arc partition, filter by
    capacity + AP-separation, deduplicate by canonical arc ordering,
    return the survivors ranked by within-cluster cost (best first).

    For K probes and ``num_arcs`` ≤ 4, the search space is
    ``num_arcs ** K`` (≤ 16,384 for the AIND rig). This is small
    enough to enumerate exhaustively without smarter clustering — the
    capacity + AP-sep filters reject most candidates immediately.
    """
    K = len(probe_names)
    if K == 0 or num_arcs == 0:
        return []
    aps = np.asarray(required_aps_deg, dtype=np.float64)
    if aps.shape != (K,):
        raise ValueError(
            f"required_aps_deg has shape {aps.shape}; expected ({K},)"
        )

    seen_canonical: set[tuple[int, ...]] = set()
    candidates: list[ArcAssignment] = []

    for assignment in product(range(num_arcs), repeat=K):
        arc_idxs = np.asarray(assignment, dtype=np.int64)
        centroids = _arc_centroids(aps, arc_idxs, num_arcs)
        if np.isnan(centroids).any():
            continue
        if not _is_valid_partition(
            arc_idxs, centroids,
            max_per_arc=max_per_arc,
            min_arc_ap_sep_deg=min_arc_ap_sep_deg,
        ):
            continue
        canonical_idxs, canonical_centroids = _canonical_arc_relabel(
            arc_idxs, centroids
        )
        sig = tuple(int(i) for i in canonical_idxs)
        if sig in seen_canonical:
            continue
        seen_canonical.add(sig)

        cost = _within_cluster_cost(aps, canonical_idxs, num_arcs)
        candidates.append(
            ArcAssignment(
                probe_to_arc_idx={
                    probe_names[i]: int(canonical_idxs[i]) for i in range(K)
                },
                arc_centroids_deg=tuple(float(c) for c in canonical_centroids),
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
    max_per_arc: int = 4,
    min_arc_ap_sep_deg: float = 16.0,
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
            probe_names, aps, n_arcs,
            max_per_arc=max_per_arc,
            min_arc_ap_sep_deg=min_arc_ap_sep_deg,
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
