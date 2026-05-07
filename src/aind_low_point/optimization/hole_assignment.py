"""Outer-layer probe→hole assignment.

For each (probe, hole) pair we compute a static cost composed of:

1. **Target-line alignment** — angle between the hole's bore axis and
   the line from the hole center to the probe's target. Small =
   probe shaft can be aimed at the target without much tilt away
   from the bore axis.
2. **Static threading clearance** — ``max_g`` across (shanks ×
   sections) at the geometric "best-fit pose" (shaft along bore
   axis, shank-row aligned with slot major). ``max_g > 0`` means the
   probe physically can't fit through this slot — that pair is hard
   rejected from the LSAP. Among feasible pairs, more-negative
   ``max_g`` is better.
3. **Pairwise interference** — soft penalty if two probes' valid-angle
   cones overlap significantly. Catches obvious joint infeasibility
   before the inner loop runs. Computed pairwise across probes and
   added to each pair's row symmetrically (a heuristic; the inner
   loop handles real interference exactly).

The combined cost goes into a (K × N) matrix. ``scipy.optimize.linear_sum_assignment``
gives the optimal assignment in O(K·N²). Murty's k-best variant enumerates
the top-``K_h`` ranked assignments.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from aind_low_point.optimization.density import coverage, gaussian_density
from aind_low_point.optimization.geometry import shaft_section_oval_value
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.kinematics import (
    pose_at_hole_best_fit,
    shank_capsules_from_pose,
)
from aind_low_point.optimization.recording import (
    RecordingGeometry,
    get_recording_geometry,
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssignmentProbe:
    """Slim per-probe info the hole-assignment cost needs.

    Avoids pulling in ``ProbeContext``'s assignment-dependent fields
    (``assigned_hole``, ``density_fn``) since those don't exist yet
    when this layer runs.

    ``kind`` is used to look up :class:`RecordingGeometry` for the
    coverage-cost term (LSAP-time coverage estimate at the geometric
    best-fit pose). ``density_sigma_mm`` controls the Gaussian density
    width (default 0.5 mm matches the inner loop).
    """

    name: str
    target_LPS: NDArray[np.floating]
    shank_tips_local: NDArray[np.floating]
    kind: str = "2.1"
    density_sigma_mm: float = 0.5


# ---------------------------------------------------------------------------
# Cost components
# ---------------------------------------------------------------------------


def angle_to_target_rad(
    target_LPS: ArrayLike, hole: Hole
) -> float:
    """Angle between the hole's bore axis (going into the brain) and
    the line from the hole's bottom-section center to the target.

    The bore enters from above; the brain is below. The optimizer's
    preferred shaft direction points roughly in ``-hole.axis`` (down
    the bore). The target should ideally lie along that direction.
    """
    target = np.asarray(target_LPS, dtype=np.float64)
    center = np.asarray(hole.sections[-1].center, dtype=np.float64)
    to_target = target - center
    n = float(np.linalg.norm(to_target))
    if n < 1e-12:
        return 0.0
    to_target /= n
    axis_into_brain = -np.asarray(hole.axis, dtype=np.float64)
    axis_into_brain /= np.linalg.norm(axis_into_brain)
    # Clamp the dot product for numerical safety before arccos.
    cos = float(np.clip(np.dot(to_target, axis_into_brain), -1.0, 1.0))
    return float(np.arccos(cos))


def static_threading_max_g(
    probe: AssignmentProbe,
    hole: Hole,
    *,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
) -> float:
    """Max threading-constraint value at the geometric best-fit pose.

    Used as the LSAP-time clearance signal:
    - ``max_g > 0``: probe physically doesn't fit through this slot →
      hard reject (caller assigns ``+∞`` cost).
    - ``max_g <= 0``: probe fits; smaller (more negative) = more clearance.

    Convention note: ``pose_at_hole_best_fit`` places the probe's local
    origin at the slot center. For probes whose canonical mesh has shank-0
    at the local origin (NP 2.0 convention), this leaves the *entire
    shank row* offset from center by the shank-row centroid. Here we
    shift ``pose_tip`` so the *centroid* of the shank tips lands at the
    slot center — that's what the geometric "best-fit" should mean for
    a multi-shank probe.
    """
    R, pose_tip = pose_at_hole_best_fit(hole)
    centroid_local = np.asarray(probe.shank_tips_local, dtype=np.float64).mean(
        axis=0
    )
    # Shift pose_tip so shank-row centroid (in world) sits at slot center.
    pose_tip = pose_tip - R @ centroid_local
    capsules = shank_capsules_from_pose(
        R, pose_tip, probe.shank_tips_local,
        shaft_length_mm=shaft_length_mm,
        shank_radius_mm=shank_radius_mm,
    )
    gs = [
        shaft_section_oval_value(cap, sec)
        for cap in capsules
        for sec in hole.sections
    ]
    return float(max(gs)) if gs else 0.0


def pairwise_interference_penalty(
    probes: list[AssignmentProbe],
    holes: list[Hole],
    cone_radius_deg: float = 16.0,
) -> NDArray[np.floating]:
    """Soft penalty for pairs of probes whose valid-angle cones overlap.

    For each (i, j) probe pair, we compute the angle between the lines
    ``hole_center_i → target_i`` and ``hole_center_j → target_j``. If
    that angle is small (< ``cone_radius_deg``) the cones overlap →
    likely AP/ML conflict for any (hole_i, hole_j) assignment.

    Returns a (K × N) "interference matrix" — for each (probe, hole),
    the *worst* interference with any *other* probe's most likely
    hole. The result is a heuristic and gets added with a small
    weight; the real test is in the inner loop.
    """
    K = len(probes)
    N = len(holes)
    out = np.zeros((K, N), dtype=np.float64)
    if K < 2 or N < 1:
        return out
    # Direction from each probe's *first preferred hole* (the one with
    # smallest target-line angle) toward its target — used as a proxy
    # for the cone direction.
    preferred = []
    for probe in probes:
        angles = [angle_to_target_rad(probe.target_LPS, h) for h in holes]
        best_h = holes[int(np.argmin(angles))]
        d = probe.target_LPS - best_h.sections[-1].center
        d = d / max(np.linalg.norm(d), 1e-12)
        preferred.append(d)

    cone_rad = float(np.deg2rad(cone_radius_deg))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            cos = float(np.clip(np.dot(preferred[i], preferred[j]), -1.0, 1.0))
            angle = float(np.arccos(cos))
            if angle < cone_rad:
                # Smaller angle ⇒ bigger overlap penalty
                penalty = (cone_rad - angle) / cone_rad
                out[i, :] += penalty
    return out


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostWeights:
    """Weights for the LSAP cost components."""

    alpha_target_angle: float = 1.0      # primary
    beta_clearance: float = 0.3          # tiebreaker
    gamma_interference: float = 0.5      # soft pairwise penalty
    delta_coverage: float = 5.0          # subtract (= maximise) coverage
    forbid_cost: float = 1.0e9           # used in place of +∞


def static_coverage(
    probe: "AssignmentProbe",
    hole: Hole,
    *,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
    n_samples: int = 41,
) -> float:
    """Coverage of the probe's target volume at the geometric best-fit
    pose for ``hole``. Used in the LSAP cost so the outer layer prefers
    holes whose recording bank actually overlaps the target.

    Replicates :func:`evaluate_probe`'s pose computation for the
    best-fit pose: shank-row centroid lands at the slot bottom, row
    aligned with slot major. Then integrates a Gaussian density
    (centered on ``probe.target_LPS``) along each shank's active
    recording range using the kind's :class:`RecordingGeometry`.
    """
    R, pose_tip = pose_at_hole_best_fit(hole)
    centroid_local = np.asarray(probe.shank_tips_local, dtype=np.float64).mean(
        axis=0
    )
    pose_tip = pose_tip - R @ centroid_local
    capsules = shank_capsules_from_pose(
        R, pose_tip, probe.shank_tips_local,
        shaft_length_mm=shaft_length_mm,
        shank_radius_mm=shank_radius_mm,
    )
    try:
        geom = get_recording_geometry(probe.kind)
    except Exception:
        geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))
    if len(capsules) != geom.n_shanks:
        # Mismatch — silent zero coverage would bias the LSAP toward
        # this hole. Better to bail loudly.
        return 0.0
    density_fn = gaussian_density(probe.target_LPS, sigma_mm=probe.density_sigma_mm)
    return coverage(density_fn, capsules, geom, n_samples=n_samples)


def build_cost_matrix(
    probes: list[AssignmentProbe],
    holes: list[Hole],
    *,
    weights: CostWeights = CostWeights(),
) -> NDArray[np.floating]:
    """Return the (K × N) probe→hole cost matrix.

    Hard rejects (``max_g > 0`` ⇒ probe physically doesn't fit the
    slot) get ``weights.forbid_cost`` rather than ``np.inf`` because
    SciPy's ``linear_sum_assignment`` doesn't accept ``inf``.
    """
    K = len(probes)
    N = len(holes)
    if K == 0 or N == 0:
        return np.zeros((K, N), dtype=np.float64)

    # Target-line angle (radians) and static clearance per pair.
    angle_mat = np.zeros((K, N), dtype=np.float64)
    max_g_mat = np.zeros((K, N), dtype=np.float64)
    coverage_mat = np.zeros((K, N), dtype=np.float64)
    for i, probe in enumerate(probes):
        for j, hole in enumerate(holes):
            angle_mat[i, j] = angle_to_target_rad(probe.target_LPS, hole)
            max_g_mat[i, j] = static_threading_max_g(probe, hole)
            coverage_mat[i, j] = static_coverage(probe, hole)

    interference_mat = pairwise_interference_penalty(probes, holes)

    cost = (
        weights.alpha_target_angle * angle_mat
        + weights.beta_clearance * max_g_mat
        + weights.gamma_interference * interference_mat
        - weights.delta_coverage * coverage_mat
    )
    # Hard reject infeasible pairs.
    cost[max_g_mat > 0.0] = weights.forbid_cost
    return cost


# ---------------------------------------------------------------------------
# LSAP + Murty's k-best
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoleAssignment:
    """One probe→hole assignment with its total LSAP cost."""

    probe_to_hole: dict[str, int]
    cost: float
    feasible: bool = True

    @classmethod
    def infeasible(cls) -> HoleAssignment:
        return cls(probe_to_hole={}, cost=float("inf"), feasible=False)


def _solve_constrained(
    cost: NDArray,
    forced_edges: list[tuple[int, int]],
    forbidden_edges: list[tuple[int, int]],
    *,
    forbid_cost: float = 1.0e9,
) -> tuple[float, list[tuple[int, int]] | None]:
    """LSAP with forced-in / forced-out edges. Returns (total_cost, edges)
    or ``(inf, None)`` if infeasible.

    Forced edges are removed from the problem (their rows/cols dropped)
    and added back at the end with their original cost. Forbidden edges
    have their cell set to ``forbid_cost`` so the solver avoids them.
    """
    K, N = cost.shape
    forced_rows = {r for r, _ in forced_edges}
    forced_cols = {c for _, c in forced_edges}
    if len(forced_rows) != len(forced_edges) or len(forced_cols) != len(forced_edges):
        # Forced edges share a row or column → impossible.
        return float("inf"), None

    free_rows = [r for r in range(K) if r not in forced_rows]
    free_cols = [c for c in range(N) if c not in forced_cols]
    if not free_rows:
        # All probes forced; total cost is sum of forced edge costs.
        return float(sum(cost[r, c] for r, c in forced_edges)), list(forced_edges)

    sub = cost[np.ix_(free_rows, free_cols)].copy()
    forbidden_set = set(forbidden_edges)
    for fi, r in enumerate(free_rows):
        for fj, c in enumerate(free_cols):
            if (r, c) in forbidden_set:
                sub[fi, fj] = forbid_cost
    try:
        row_idx, col_idx = linear_sum_assignment(sub)
    except ValueError:
        return float("inf"), None
    sub_cost = float(sub[row_idx, col_idx].sum())
    if sub_cost >= forbid_cost:
        return float("inf"), None
    edges = list(forced_edges) + [
        (free_rows[i], free_cols[j]) for i, j in zip(row_idx, col_idx)
    ]
    total = sub_cost + float(sum(cost[r, c] for r, c in forced_edges))
    return total, edges


def solve_optimal_assignment(
    probes: list[AssignmentProbe],
    holes: list[Hole],
    *,
    weights: CostWeights = CostWeights(),
) -> HoleAssignment:
    """Best-only probe→hole assignment via LSAP."""
    cost = build_cost_matrix(probes, holes, weights=weights)
    if cost.size == 0:
        return HoleAssignment.infeasible()
    total, edges = _solve_constrained(cost, [], [], forbid_cost=weights.forbid_cost)
    if edges is None:
        return HoleAssignment.infeasible()
    mapping = {probes[r].name: holes[c].id for r, c in edges}
    return HoleAssignment(probe_to_hole=mapping, cost=total)


def solve_top_k_assignments(
    probes: list[AssignmentProbe],
    holes: list[Hole],
    k: int,
    *,
    weights: CostWeights = CostWeights(),
) -> list[HoleAssignment]:
    """Murty's algorithm: top-``k`` probe→hole assignments ranked by cost.

    Implementation sketch (the standard version):

    1. Solve LSAP for the optimal assignment ``A_1`` → push to a
       priority queue keyed by cost, along with empty
       (forced, forbidden) edge sets.
    2. Repeat: pop the minimum-cost queue entry. That's the next
       best assignment. Then *partition* the search space relative
       to it by walking through the popped assignment's edges and,
       for each edge ``e``, generating a sub-problem with ``e``
       forbidden (and earlier edges of the same partition forced).
       Solve each sub-problem and push its result on the queue.
    3. Stop after ``k`` results or when the queue is empty.
    """
    if k <= 0:
        return []
    cost_mat = build_cost_matrix(probes, holes, weights=weights)
    if cost_mat.size == 0:
        return []

    def make_assignment(edges, total) -> HoleAssignment:
        mapping = {probes[r].name: holes[c].id for r, c in edges}
        return HoleAssignment(probe_to_hole=mapping, cost=total)

    # Priority queue: (cost, tiebreaker, forced_edges, forbidden_edges, edges)
    counter = 0
    heap: list[tuple[float, int, list, list, list]] = []
    total, edges = _solve_constrained(
        cost_mat, [], [], forbid_cost=weights.forbid_cost
    )
    if edges is None:
        return []
    heapq.heappush(heap, (total, counter, [], [], edges))
    results: list[HoleAssignment] = []

    while heap and len(results) < k:
        cost_val, _, forced, forbidden, edges = heapq.heappop(heap)
        results.append(make_assignment(edges, cost_val))
        if len(results) == k:
            break
        # Partition: for each edge in `edges` not already forced,
        # generate a child by forbidding it (with earlier edges in
        # the partition forced).
        forced_extension: list[tuple[int, int]] = []
        for e in edges:
            if e in forced:
                continue
            new_forbidden = forbidden + [e]
            new_forced = forced + forced_extension
            sub_cost, sub_edges = _solve_constrained(
                cost_mat, new_forced, new_forbidden,
                forbid_cost=weights.forbid_cost,
            )
            if sub_edges is not None:
                counter += 1
                heapq.heappush(
                    heap,
                    (sub_cost, counter, new_forced, new_forbidden, sub_edges),
                )
            forced_extension.append(e)
    return results
