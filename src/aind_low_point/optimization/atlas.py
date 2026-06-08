"""Target-aligned pose-feasibility atlas for the optimizer's Stage 1.

For each (probe, hole) pair, builds an achievable arc-AP **interval**
``[ap_min, ap_max]`` where the probe can thread the hole *and* place
its recording bank near its target. The seed AP is target-aligned —
the natural direction from probe.target_LPS out through the hole —
not the hole bore axis. This is the discrete-layer abstraction the
manual planner uses.

Replaces LSAP + Murty in the Stage 1 → Stage 2 handoff. With K=7, N=14
on 836656/T12 this returns ~63 H assignments vs LSAP's 1000+, all of
them arc-cover-feasible (every probe has a non-empty interval that
admits a 3-arc cover with ≥ 16° AP separation).

See ``scripts/diagnose_atlas_pass1.py`` for the diagnostic that
validated the approach.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from aind_mri_utils.arc_angles import vector_to_arc_angles
from numpy.typing import NDArray
from scipy.optimize import minimize

from aind_low_point.optimization.geometry import shaft_section_oval_value
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.kinematics import (
    pose_from_optimizer_vars,
    shank_capsules_from_pose,
)
from aind_low_point.optimization.recording import get_recording_geometry

# ---------------------------------------------------------------------------
# Atlas data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseAnchor:
    """One target-valid pose for (probe, hole) at a specific arc-AP."""

    ap_deg: float
    ml_deg: float
    spin_deg: float
    off_R_mm: float
    off_A_mm: float
    depth_mm: float
    threading_max_g: float
    target_miss_mm: float


@dataclass(frozen=True)
class AtlasEntry:
    """Atlas entry for one (probe, hole). Empty interval (None) means
    "no target-valid pose exists at any arc AP for this pair"."""

    probe_name: str
    hole_id: int
    ap_min: float | None
    ap_max: float | None
    anchors: tuple[PoseAnchor, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Atlas:
    """Built atlas — all (probe, hole) entries for one optimizer run."""

    entries: dict[tuple[str, int], AtlasEntry]
    probe_names: tuple[str, ...]
    hole_ids: tuple[int, ...]


# ---------------------------------------------------------------------------
# Pose evaluation (self-contained — no OptimizerContext needed)
# ---------------------------------------------------------------------------


def _pose_score(
    probe_static,
    hole: Hole,
    *,
    ap_deg: float,
    ml_deg: float,
    spin_deg: float,
    off_R_mm: float,
    off_A_mm: float,
    depth_mm: float,
    shaft_length_mm: float,
    shank_radius_mm: float,
) -> tuple[float, float]:
    """Return ``(threading_max_g, target_miss_mm)`` at the given pose."""
    geom = get_recording_geometry(probe_static.kind)
    tips = np.asarray(probe_static.shank_tips_local, dtype=np.float64)
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
    R, pose_tip = pose_from_optimizer_vars(
        target_LPS=probe_static.target_LPS,
        ap_deg=ap_deg,
        ml_deg=ml_deg,
        spin_deg=spin_deg,
        offset_R_mm=off_R_mm,
        offset_A_mm=off_A_mm,
        past_target_mm=depth_mm,
        recording_center_local=pivot_local,
    )
    shanks = shank_capsules_from_pose(
        R,
        pose_tip,
        probe_static.shank_tips_local,
        shaft_length_mm=shaft_length_mm,
        shank_radius_mm=shank_radius_mm,
    )
    if not shanks:
        return 0.0, 1e3
    max_g = -np.inf
    for sh in shanks:
        for sec in hole.sections:
            g = shaft_section_oval_value(sh, sec)
            if g > max_g:
                max_g = g
    centroid_world = R @ pivot_local + pose_tip
    target_miss = float(np.linalg.norm(centroid_world - probe_static.target_LPS))
    return float(max_g), target_miss


# ---------------------------------------------------------------------------
# Local anchor search at fixed AP
# ---------------------------------------------------------------------------


def _find_anchor_at_ap(
    probe_static,
    hole: Hole,
    ap_deg: float,
    *,
    max_target_miss_mm: float,
    threading_tol: float,
    starts: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (15.0, 0.0),
        (-15.0, 0.0),
    ),
    shaft_length_mm: float = 10.0,
    shank_radius_mm: float = 0.05,
) -> PoseAnchor | None:
    """Find a target-valid pose at fixed ``ap_deg`` via SLSQP from a few
    (ml, spin) starts. Returns the best anchor (lowest threading max_g)
    that passes both lex-feasibility checks, or ``None``."""
    bounds = [
        (-45.0, 45.0),
        (-180.0, 180.0),
        (-2.0, 2.0),
        (-2.0, 2.0),
        (-3.0, 3.0),
    ]

    def f(x):
        ml, spin, off_R, off_A, depth = x
        max_g, miss = _pose_score(
            probe_static,
            hole,
            ap_deg=ap_deg,
            ml_deg=ml,
            spin_deg=spin,
            off_R_mm=off_R,
            off_A_mm=off_A,
            depth_mm=depth,
            shaft_length_mm=shaft_length_mm,
            shank_radius_mm=shank_radius_mm,
        )
        thread_pen = max(0.0, max_g - threading_tol)
        return (
            100.0 * thread_pen * thread_pen
            + 1.0 * miss * miss
            + 0.01 * (off_R * off_R + off_A * off_A + depth * depth)
        )

    best: PoseAnchor | None = None
    for ml0, spin0 in starts:
        x0 = np.array([ml0, spin0, 0.0, 0.0, 0.0], dtype=np.float64)
        try:
            res = minimize(
                f,
                x0,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 40, "ftol": 1e-6, "disp": False},
            )
        except Exception:
            continue
        x = res.x
        max_g, miss = _pose_score(
            probe_static,
            hole,
            ap_deg=ap_deg,
            ml_deg=x[0],
            spin_deg=x[1],
            off_R_mm=x[2],
            off_A_mm=x[3],
            depth_mm=x[4],
            shaft_length_mm=shaft_length_mm,
            shank_radius_mm=shank_radius_mm,
        )
        if max_g > threading_tol:
            continue
        if miss > max_target_miss_mm:
            continue
        rec = PoseAnchor(
            ap_deg=ap_deg,
            ml_deg=float(x[0]),
            spin_deg=float(x[1]),
            off_R_mm=float(x[2]),
            off_A_mm=float(x[3]),
            depth_mm=float(x[4]),
            threading_max_g=max_g,
            target_miss_mm=miss,
        )
        if best is None or rec.threading_max_g < best.threading_max_g:
            best = rec
    return best


# ---------------------------------------------------------------------------
# AP-interval sweep
# ---------------------------------------------------------------------------


def _find_interval(
    probe_static,
    hole: Hole,
    *,
    ap_seed_deg: float,
    ap_step_deg: float,
    ap_max_excursion_deg: float,
    abandon_after_failures: int,
    max_target_miss_mm: float,
    threading_tol: float,
    seed_retry_offsets_deg: tuple[float, ...] = (
        0.0,
        -2.0,
        +2.0,
        -4.0,
        +4.0,
        -6.0,
        +6.0,
    ),
) -> tuple[float | None, float | None, list[PoseAnchor]]:
    """Sweep arc-AP outward from ``ap_seed_deg`` to find the interval
    where target-valid anchors exist.

    Narrow multi-seed retry: target_aligned is approximate (the natural
    "point at target" direction), and the per-AP SLSQP can fail at the
    exact seed even when a nearby AP succeeds. So we try seed ± a few
    small offsets before giving up.
    """
    seed: PoseAnchor | None = None
    effective_seed = ap_seed_deg
    for off in seed_retry_offsets_deg:
        candidate = ap_seed_deg + off
        a = _find_anchor_at_ap(
            probe_static,
            hole,
            candidate,
            max_target_miss_mm=max_target_miss_mm,
            threading_tol=threading_tol,
        )
        if a is not None:
            seed = a
            effective_seed = candidate
            break
    if seed is None:
        return None, None, []
    anchors: list[PoseAnchor] = [seed]
    ap_min = effective_seed
    ap_max = effective_seed
    ap_seed_deg = effective_seed

    # Sweep upward
    ap = ap_seed_deg
    failures = 0
    while ap - ap_seed_deg < ap_max_excursion_deg and failures < abandon_after_failures:
        ap += ap_step_deg
        a = _find_anchor_at_ap(
            probe_static,
            hole,
            ap,
            max_target_miss_mm=max_target_miss_mm,
            threading_tol=threading_tol,
        )
        if a is None:
            failures += 1
            continue
        failures = 0
        anchors.append(a)
        ap_max = ap

    # Sweep downward
    ap = ap_seed_deg
    failures = 0
    while ap_seed_deg - ap < ap_max_excursion_deg and failures < abandon_after_failures:
        ap -= ap_step_deg
        a = _find_anchor_at_ap(
            probe_static,
            hole,
            ap,
            max_target_miss_mm=max_target_miss_mm,
            threading_tol=threading_tol,
        )
        if a is None:
            failures += 1
            continue
        failures = 0
        anchors.append(a)
        ap_min = ap

    return ap_min, ap_max, sorted(anchors, key=lambda x: x.ap_deg)


# ---------------------------------------------------------------------------
# Atlas builder
# ---------------------------------------------------------------------------


def _target_aligned_ap(probe_static, hole: Hole) -> float:
    """Seed AP from the insertion direction (probe entering from above
    implant, pointing down toward target). ``vector_to_arc_angles``
    is direction-insensitive (flips downward vectors automatically)
    and **expects RAS coordinates**, so we convert from LPS by sign-
    flipping the L and P components.
    """
    target_lps = np.asarray(probe_static.target_LPS, dtype=np.float64)
    hole_centre_lps = np.asarray(hole.sections[0].center, dtype=np.float64)
    shaft_dir_lps = target_lps - hole_centre_lps  # tip - base, downward
    n = float(np.linalg.norm(shaft_dir_lps))
    if n < 1e-9:
        return 0.0
    shaft_dir_lps = shaft_dir_lps / n
    # LPS → RAS: (-L, -P, +S)
    shaft_dir_ras = np.array(
        [-shaft_dir_lps[0], -shaft_dir_lps[1], shaft_dir_lps[2]],
        dtype=np.float64,
    )
    try:
        ap, _ml = vector_to_arc_angles(shaft_dir_ras, degrees=True, invert_AP=True)
        return float(ap)
    except Exception:
        return 0.0


def build_atlas(
    probes,
    holes: list[Hole],
    *,
    ap_step_deg: float = 2.0,
    ap_max_excursion_deg: float = 60.0,
    abandon_after_failures: int = 2,
    max_target_miss_mm: float = 2.0,
    threading_tol: float = 0.0,
    verbose: bool = False,
) -> Atlas:
    """Build target-aligned atlas across all (probe, hole) pairs.

    For each (probe, hole):
      1. Compute target-aligned seed AP from
         ``(hole.sections[0].center - probe.target_LPS)``.
      2. Sweep arc-AP outward in ``±ap_step_deg`` steps, finding the
         range where a target-valid pose exists.
      3. Store the interval + per-AP anchors.

    Pairs where no target-valid pose exists at the seed AP receive
    empty atlas entries (ap_min=ap_max=None, anchors=()).

    ~30-50 s for K=7, N=14 at 2° step. Linear in K × N.
    """
    entries: dict[tuple[str, int], AtlasEntry] = {}
    probe_names = tuple(p.name for p in probes)
    hole_ids = tuple(h.id for h in holes)
    for p_idx, probe in enumerate(probes):
        for h_idx, hole in enumerate(holes):
            seed = _target_aligned_ap(probe, hole)
            ap_min, ap_max, anchors = _find_interval(
                probe,
                hole,
                ap_seed_deg=seed,
                ap_step_deg=ap_step_deg,
                ap_max_excursion_deg=ap_max_excursion_deg,
                abandon_after_failures=abandon_after_failures,
                max_target_miss_mm=max_target_miss_mm,
                threading_tol=threading_tol,
            )
            entries[(probe.name, hole.id)] = AtlasEntry(
                probe_name=probe.name,
                hole_id=hole.id,
                ap_min=ap_min,
                ap_max=ap_max,
                anchors=tuple(anchors),
            )
        if verbose:
            valid = [
                h_id
                for h_id in hole_ids
                if entries[(probe.name, h_id)].ap_min is not None
            ]
            print(
                f"  [atlas] {probe.name:>5}: {len(valid)}/{len(hole_ids)} "
                f"valid holes → {sorted(valid)}"
            )
    return Atlas(entries=entries, probe_names=probe_names, hole_ids=hole_ids)


# ---------------------------------------------------------------------------
# H-enumeration with atlas filter
# ---------------------------------------------------------------------------


def _admits_arc_cover(  # noqa: C901
    intervals: list[tuple[float, float]],
    max_arcs: int = 3,
    min_sep_deg: float = 16.0,
) -> bool:
    """True iff the K intervals admit a ``≤ max_arcs`` cover with each
    arc-AP centre inside its assigned interval and centres ≥ ``min_sep_deg``
    apart. Enumerates all ordered partitions of K probes into 1..max_arcs
    non-empty groups; per partition checks intersection then greedy
    centre placement."""
    K = len(intervals)
    if K == 0:
        return True

    def partitions(idx: int, groups: list[list[int]]):
        if idx == K:
            if len(groups) <= max_arcs:
                yield [list(g) for g in groups]
            return
        for g in groups:
            g.append(idx)
            yield from partitions(idx + 1, groups)
            g.pop()
        if len(groups) < max_arcs:
            groups.append([idx])
            yield from partitions(idx + 1, groups)
            groups.pop()

    for partition in partitions(0, []):
        # Per group: intersect intervals
        g_mins = []
        g_maxs = []
        empty = False
        for group in partition:
            gm = max(intervals[i][0] for i in group)
            gx = min(intervals[i][1] for i in group)
            if gm > gx:
                empty = True
                break
            g_mins.append(gm)
            g_maxs.append(gx)
        if empty:
            continue
        if len(partition) <= 1:
            return True
        # Sort by g_min and greedy place centres
        order = sorted(range(len(partition)), key=lambda k: g_mins[k])
        c_prev = g_mins[order[0]]
        ok = True
        for k_idx in order[1:]:
            c = max(g_mins[k_idx], c_prev + min_sep_deg)
            if c > g_maxs[k_idx]:
                ok = False
                break
            c_prev = c
        if ok:
            return True
    return False


def enumerate_hole_assignments(
    atlas: Atlas,
    probes,
    holes: list[Hole],
    *,
    viol_mat: NDArray | None = None,
    viol_threshold: float = 1.0,
    cost_for_ordering: NDArray | None = None,
    arc_cover: bool = True,
    max_arcs: int = 3,
    min_arc_sep_deg: float = 16.0,
    cap: int | None = None,
) -> list[HoleAssignment]:
    """Enumerate all probe→hole permutations where every (probe, hole)
    in the assignment has a non-empty atlas interval, (optionally)
    passes a per-cell violation filter, and (optionally) admits a
    ≤ ``max_arcs`` arc cover with ≥ ``min_arc_sep_deg`` separation.

    Returns a list of :class:`HoleAssignment` sorted by ``cost_for_ordering``
    (if provided — typically the LSAP cost matrix as a weak ordering),
    else by enumeration order with ``cost=0.0``. ``cap`` truncates the
    returned list to the lowest-cost ``cap`` entries.
    """
    K = len(probes)
    probe_names = [p.name for p in probes]
    valid_holes_per_probe: list[list[int]] = []
    for probe in probes:
        valid = [
            j
            for j, h in enumerate(holes)
            if atlas.entries[(probe.name, h.id)].ap_min is not None
        ]
        valid_holes_per_probe.append(valid)

    results: list[HoleAssignment] = []
    n_dropped_arc = 0
    for perm in itertools.permutations(range(len(holes)), K):
        # Atlas filter: every probe's assigned hole must be valid for it
        if any(perm[i] not in valid_holes_per_probe[i] for i in range(K)):
            continue
        # Per-cell viol filter (optional)
        if viol_mat is not None:
            if any(viol_mat[i, perm[i]] > viol_threshold for i in range(K)):
                continue
        # Arc-cover filter (optional but on by default)
        if arc_cover:
            ints = [
                (
                    atlas.entries[(probe_names[i], holes[perm[i]].id)].ap_min,
                    atlas.entries[(probe_names[i], holes[perm[i]].id)].ap_max,
                )
                for i in range(K)
            ]
            if not _admits_arc_cover(
                ints, max_arcs=max_arcs, min_sep_deg=min_arc_sep_deg
            ):
                n_dropped_arc += 1
                continue
        cost = 0.0
        if cost_for_ordering is not None:
            cost = float(sum(cost_for_ordering[i, perm[i]] for i in range(K)))
        mapping = {probe_names[i]: holes[perm[i]].id for i in range(K)}
        results.append(HoleAssignment(probe_to_hole=mapping, cost=cost))

    results.sort(key=lambda ha: ha.cost)
    if cap is not None and len(results) > cap:
        results = results[:cap]
    return results


def atlas_stage1(
    probes,
    holes: list[Hole],
    *,
    viol_mat: NDArray | None = None,
    cost_for_ordering: NDArray | None = None,
    cap_hole_assignments: int | None = 200,
    min_arc_sep_deg: float = 16.0,
    max_arcs: int = 3,
    verbose: bool = False,
    **atlas_kwargs,
) -> tuple[Atlas, list[HoleAssignment]]:
    """End-to-end Stage 1 via atlas: build atlas, enumerate H assignments.

    Returns ``(atlas, hole_assignments)``. ``cap_hole_assignments``
    truncates the returned list (sorted by ``cost_for_ordering``) so
    Stage 2's per-candidate cost doesn't blow up.
    """
    atlas = build_atlas(probes, holes, verbose=verbose, **atlas_kwargs)
    his = enumerate_hole_assignments(
        atlas,
        probes,
        holes,
        viol_mat=viol_mat,
        cost_for_ordering=cost_for_ordering,
        arc_cover=True,
        max_arcs=max_arcs,
        min_arc_sep_deg=min_arc_sep_deg,
        cap=cap_hole_assignments,
    )
    if verbose:
        print(
            f"  [atlas] enumerated {len(his)} feasible H assignments "
            f"(capped at {cap_hole_assignments})"
        )
    return atlas, his
