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


def angle_to_target_rad(target_LPS: ArrayLike, hole: Hole) -> float:
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
    centroid_local = np.asarray(probe.shank_tips_local, dtype=np.float64).mean(axis=0)
    # Shift pose_tip so shank-row centroid (in world) sits at slot center.
    pose_tip = pose_tip - R @ centroid_local
    capsules = shank_capsules_from_pose(
        R,
        pose_tip,
        probe.shank_tips_local,
        shaft_length_mm=shaft_length_mm,
        shank_radius_mm=shank_radius_mm,
    )
    gs = [
        shaft_section_oval_value(cap, sec) for cap in capsules for sec in hole.sections
    ]
    return float(max(gs)) if gs else 0.0


def _rotation_about_axis(axis: NDArray, angle_rad: float) -> NDArray:
    """3×3 rotation matrix about ``axis`` (unit) by ``angle_rad``
    (Rodrigues). Axis assumed already normalised by the caller."""
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def _rotate_to(from_dir: NDArray, to_dir: NDArray) -> NDArray:
    """Smallest rotation matrix taking ``from_dir`` to ``to_dir``
    (both unit). Identity if they're already (anti-)parallel."""
    f = np.asarray(from_dir, dtype=np.float64)
    t = np.asarray(to_dir, dtype=np.float64)
    cos = float(np.clip(np.dot(f, t), -1.0, 1.0))
    if cos > 1.0 - 1e-12:
        return np.eye(3)
    if cos < -1.0 + 1e-12:
        # 180° flip; pick any axis perpendicular to f.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, f)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        ax = np.cross(f, helper)
        ax /= np.linalg.norm(ax)
        return _rotation_about_axis(ax, np.pi)
    ax = np.cross(f, t)
    ax /= np.linalg.norm(ax)
    return _rotation_about_axis(ax, float(np.arccos(cos)))


def _build_pose_bank(
    probe: AssignmentProbe,
    hole: Hole,
    pivot_local: NDArray,
    *,
    tilt_deg: float = 2.0,
) -> list[tuple[NDArray, NDArray]]:
    """Construct the candidate-pose bank for LSAP feasibility scoring.

    The bank samples poses *around the target-oriented insertion* — the
    physically realistic regime for a real plan, where the probe points
    at the target rather than down the bore axis. Bore-aligned and
    halfway poses are intentionally excluded: they typically have
    excellent threading but near-zero coverage at off-axis targets, so
    crediting a pair with their min-clearance was over-optimistic about
    how feasible the eventual plan actually is.

    Bank composition (5 poses):

    1. Target-aligned (shaft along ``target − slot_center``).
    2-3. Target-aligned ± ``tilt_deg`` around world ``+x_LPS`` (rig AP).
    4-5. Target-aligned ± ``tilt_deg`` around world ``+y_LPS`` (rig ML).

    ``pose_tip`` is anchored the way the inner loop does: ``pose_tip =
    target_LPS − R @ pivot_local`` for each pose's own ``R``. That puts
    the recording-array centre on the target, so the LSAP's coverage
    integral reflects what a real plan would actually record. Threading
    values are insensitive to the parallel-to-shaft shift this implies
    (the section ``(u, v)`` projections of the shaft line don't change
    when ``pose_tip`` slides along ``R[:, 2]``), so this anchoring
    affects coverage scoring but not the threading penalties.

    Spin±90° (row across slot minor) is omitted: for the multi-shank
    AIND probes the row span (~0.75 mm) exceeds the slot minor (~0.7
    mm) by construction, so that orientation is trivially infeasible
    and adds no signal. Depth perturbations along the bore axis are
    also omitted: for straight bores aligned with the shaft, depth
    shifts the intersection along the shaft but leaves the projected
    ``(u, v)`` unchanged, so they don't relax max_g.
    """
    R_base, _ = pose_at_hole_best_fit(hole)
    target_LPS = np.asarray(probe.target_LPS, dtype=np.float64)
    pivot_local = np.asarray(pivot_local, dtype=np.float64)

    bore_dir = -np.asarray(hole.axis, dtype=np.float64)
    bore_dir /= np.linalg.norm(bore_dir)
    to_target = target_LPS - np.asarray(hole.sections[-1].center, dtype=np.float64)
    n = float(np.linalg.norm(to_target))
    target_dir = to_target / n if n >= 1e-9 else bore_dir

    R_target = _rotate_to(bore_dir, target_dir) @ R_base

    def _anchor(R: NDArray) -> NDArray:
        return target_LPS - R @ pivot_local

    poses: list[tuple[NDArray, NDArray]] = [(R_target, _anchor(R_target))]

    tilt_rad = float(np.deg2rad(tilt_deg))
    e_x = np.array([1.0, 0.0, 0.0])  # AP-rotation axis (LPS +x)
    e_y = np.array([0.0, 1.0, 0.0])  # ML-rotation axis (LPS +y)
    for sign in (+1.0, -1.0):
        for axis in (e_x, e_y):
            R = _rotation_about_axis(axis, sign * tilt_rad) @ R_target
            poses.append((R, _anchor(R)))
    return poses


@dataclass(frozen=True)
class MultiPoseScore:
    """Aggregates over the multi-pose bank for a single (probe, hole) pair.

    All three are taken across the bank — feasibility over poses is a
    "best-of" question (``min`` violations, ``max`` coverage):

    - ``min_violation_sq``: ``min_m Σ_j ReLU(g_j(x_m))²``. This is the
      Stage-A scalar from the inner loop, evaluated at each candidate
      pose. Zero ⇒ at least one bank pose is fully feasible.
    - ``min_max_g``: ``min_m max_j g_j(x_m)``. Single-pose worst-section
      reading at the most-feasible pose; tiebreaker among feasible
      pairs (more-negative = more clearance).
    - ``max_coverage``: ``max_m coverage(x_m)`` — coverage at the
      best-coverage pose (``static_coverage`` is the slot-aligned one,
      but a target-aimed pose can score higher for off-bore targets).
    """

    min_violation_sq: float
    min_max_g: float
    max_coverage: float


def multi_pose_threading_max_g(
    probe: AssignmentProbe,
    hole: Hole,
    *,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
    tilt_deg: float = 2.0,
) -> float:
    """Backwards-compatible scalar — returns ``MultiPoseScore.min_max_g``."""
    return multi_pose_evaluate(
        probe,
        hole,
        shaft_length_mm=shaft_length_mm,
        shank_radius_mm=shank_radius_mm,
        tilt_deg=tilt_deg,
    ).min_max_g


def multi_pose_evaluate(
    probe: AssignmentProbe,
    hole: Hole,
    *,
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
    tilt_deg: float = 2.0,
    coverage_n_samples: int = 41,
) -> MultiPoseScore:
    """Evaluate the (probe, hole) pair over the multi-pose bank.

    Builds the 5-pose bank from :func:`_build_pose_bank`, computes the
    threading-constraint vector and coverage at each, and returns the
    best-of-bank aggregates as a :class:`MultiPoseScore`. Used by
    :func:`build_cost_matrix` to set both the soft cost contribution
    and the (relaxed) hard-reject criterion.
    """
    try:
        geom = get_recording_geometry(probe.kind)
    except Exception:
        geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))
    # ``pivot_local`` matches what the inner loop uses to translate
    # pose_tip: shank-row centroid in xy + active recording centre in z.
    tips = np.asarray(probe.shank_tips_local, dtype=np.float64)
    if tips.shape[0] > 0:
        pivot_local = np.array(
            [
                float(tips[:, 0].mean()),
                float(tips[:, 1].mean()),
                float(geom.active_center_mm),
            ],
            dtype=np.float64,
        )
    else:
        pivot_local = np.array([0.0, 0.0, geom.active_center_mm], dtype=np.float64)
    poses = _build_pose_bank(probe, hole, pivot_local, tilt_deg=tilt_deg)

    density_fn = gaussian_density(probe.target_LPS, sigma_mm=probe.density_sigma_mm)

    min_vio_sq = float("inf")
    min_max_g = float("inf")
    max_cov = 0.0
    for R, tp in poses:
        capsules = shank_capsules_from_pose(
            R,
            tp,
            probe.shank_tips_local,
            shaft_length_mm=shaft_length_mm,
            shank_radius_mm=shank_radius_mm,
        )
        if not capsules:
            continue
        gs = np.array(
            [
                shaft_section_oval_value(cap, sec)
                for cap in capsules
                for sec in hole.sections
            ],
            dtype=np.float64,
        )
        if gs.size > 0:
            vio = float(np.sum(np.maximum(0.0, gs) ** 2))
            min_vio_sq = min(min_vio_sq, vio)
            min_max_g = min(min_max_g, float(np.max(gs)))
        if len(capsules) == geom.n_shanks:
            cov = coverage(density_fn, capsules, geom, n_samples=coverage_n_samples)
            max_cov = max(max_cov, float(cov))

    if min_vio_sq == float("inf"):
        min_vio_sq = 0.0
    if min_max_g == float("inf"):
        min_max_g = 0.0
    return MultiPoseScore(
        min_violation_sq=min_vio_sq,
        min_max_g=min_max_g,
        max_coverage=max_cov,
    )


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
    """Weights for the LSAP cost components.

    The bore-vs-target *angle* term is disabled by default (alpha=0.0):
    once the pose bank evaluates target-oriented feasibility directly
    (via ``min_max_g`` and ``min_violation_sq``) and coverage is
    anchored on the target, the bore-line angle is the wrong axis to
    penalise — it punishes off-bore targets even when the probe can
    in fact tilt to thread them, and ignores the rig's actual arc/AP/ML
    coordinate system. A proper "extreme rig-frame angle" penalty
    (accounting for the subject's mounted head pitch) is the right
    formulation; left as a follow-up.
    """

    alpha_target_angle: float = 0.0  # disabled — see class docstring
    beta_clearance: float = 0.3  # tiebreaker (more-negative max_g better)
    gamma_interference: float = 0.5  # soft pairwise penalty
    delta_coverage: float = 5.0  # subtract (= maximise) coverage
    eta_violation: float = 2.0  # ``min_m Σ ReLU(g)²`` term — soft
    # feasibility cost; only a hint, the
    # hard reject below is the gate
    violation_reject_threshold: float = 1.0
    """Pairs whose ``min_m Σ ReLU(g_j)²`` exceeds this are hard rejected
    (i.e. *every* sampled pose is so badly infeasible that no nearby
    inner-loop solve will recover). 1.0 corresponds roughly to one
    section-shank entry with ``g ≈ 1`` (~one full oval-radius worth of
    overlap) — anything more is hopeless. Set lower for stricter LSAP."""
    forbid_cost: float = 1.0e9  # used in place of +∞


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
    centroid_local = np.asarray(probe.shank_tips_local, dtype=np.float64).mean(axis=0)
    pose_tip = pose_tip - R @ centroid_local
    capsules = shank_capsules_from_pose(
        R,
        pose_tip,
        probe.shank_tips_local,
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

    # Target-line angle, multi-pose feasibility/clearance/coverage.
    # Per (probe, hole), evaluate the 7-pose bank (slot-aligned + target-
    # aligned + halfway + ±AP/ML wobbles) and aggregate:
    #   - min_violation_sq: best-of-bank Σ ReLU(g_j)² — feasibility
    #   - min_max_g:        best-of-bank max(g_j)    — clearance signal
    #   - max_coverage:     best-of-bank coverage    — coverage bonus
    angle_mat = np.zeros((K, N), dtype=np.float64)
    max_g_mat = np.zeros((K, N), dtype=np.float64)
    violation_mat = np.zeros((K, N), dtype=np.float64)
    coverage_mat = np.zeros((K, N), dtype=np.float64)
    for i, probe in enumerate(probes):
        for j, hole in enumerate(holes):
            angle_mat[i, j] = angle_to_target_rad(probe.target_LPS, hole)
            score = multi_pose_evaluate(probe, hole)
            max_g_mat[i, j] = score.min_max_g
            violation_mat[i, j] = score.min_violation_sq
            coverage_mat[i, j] = score.max_coverage

    interference_mat = pairwise_interference_penalty(probes, holes)

    cost = (
        weights.alpha_target_angle * angle_mat
        + weights.beta_clearance * max_g_mat
        + weights.eta_violation * violation_mat
        + weights.gamma_interference * interference_mat
        - weights.delta_coverage * coverage_mat
    )
    # Hard reject only when every sampled pose is so badly infeasible
    # that no nearby inner-loop solve will recover. Marginal "just
    # barely outside the slot" pairs survive — the inner loop's
    # two-stage solve can pull them inside.
    cost[violation_mat > weights.violation_reject_threshold] = weights.forbid_cost
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
    min_hamming_distance: int = 0,
    explore_multiplier: int = 8,
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

    Parameters
    ----------
    min_hamming_distance
        When > 0, *diversified* Murty: reject popped candidates whose
        Hamming distance to all already-accepted results is below this
        threshold (number of probes that map to different holes).
        Trivial 1-edge swap variants are filtered out, opening the
        enumeration to structurally distinct configurations. Rejected
        candidates still partition the search space so we don't lose
        the branch — they're just not returned. With probe count
        ``K``, ``min_hamming_distance=2`` filters single-probe swaps;
        ``3+`` filters 2-cycle (a↔b) swaps too.
    explore_multiplier
        When ``min_hamming_distance > 0``, expand the candidate budget
        to ``k × explore_multiplier`` to compensate for rejections.
        Stops early once ``k`` diverse results are found.
    """
    if k <= 0:
        return []
    cost_mat = build_cost_matrix(probes, holes, weights=weights)
    if cost_mat.size == 0:
        return []

    def make_assignment(edges, total) -> HoleAssignment:
        mapping = {probes[r].name: holes[c].id for r, c in edges}
        return HoleAssignment(probe_to_hole=mapping, cost=total)

    def hamming(a: dict[str, int], b: dict[str, int]) -> int:
        return sum(1 for n in a if a[n] != b.get(n))

    diverse = min_hamming_distance > 0
    max_explore = k * explore_multiplier if diverse else k

    # Priority queue: (cost, tiebreaker, forced_edges, forbidden_edges, edges)
    counter = 0
    heap: list[tuple[float, int, list, list, list]] = []
    total, edges = _solve_constrained(cost_mat, [], [], forbid_cost=weights.forbid_cost)
    if edges is None:
        return []
    heapq.heappush(heap, (total, counter, [], [], edges))
    results: list[HoleAssignment] = []
    explored = 0

    while heap and len(results) < k and explored < max_explore:
        cost_val, _, forced, forbidden, edges = heapq.heappop(heap)
        explored += 1
        candidate = make_assignment(edges, cost_val)

        # Diversity filter: reject if too close to any accepted result.
        accept = True
        if diverse and results:
            min_d = min(
                hamming(candidate.probe_to_hole, r.probe_to_hole) for r in results
            )
            if min_d < min_hamming_distance:
                accept = False
        if accept:
            results.append(candidate)
            if len(results) == k:
                break
        # Always partition — even rejected candidates contribute their
        # branch's children, so we don't lose access to deeper diverse
        # variants reachable through this branch.
        forced_extension: list[tuple[int, int]] = []
        for e in edges:
            if e in forced:
                continue
            new_forbidden = forbidden + [e]
            new_forced = forced + forced_extension
            sub_cost, sub_edges = _solve_constrained(
                cost_mat,
                new_forced,
                new_forbidden,
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
