"""Which regions share a probe, and which kind each probe is.

A line through a bore, at angles the rig reaches, is scored the way a plan is:
the share of each region's points within a radius of the active stretch of the
shanks on it. Regions are a unit when one line records enough of each; a cover
partitions the regions into units, each on a bore of its own, with the kinds
that record the most within the inventory.

The test is loose on purpose. It knows the implant, the rig's angles and the
probes' shanks, and nothing of fixtures or crowding, which are the solver's:
a proposal that cannot be laid out costs solver time, a wrong refusal costs a
plan.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from aind_mri_utils.arc_angles import arc_angles_to_affine
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from aind_rutter.domain.rig import DEFAULT_ML_RANGE_DEG
from aind_rutter.optimization.geometry.holes import DEFAULT_THREADING_MARGIN_MM, Hole
from aind_rutter.optimization.geometry.primitives import HoleSection, cap_basis

SHANK_LENGTH_MM = 10.0  # tip to base; the base stays above the bore
LINE_DEPTH_MM = 12.0  # how far below the bore top a line is followed
LINES_PER_CHUNK = 200  # bounds the lines x points arrays


@dataclass(frozen=True)
class ProbeShape:
    """What the proposer needs of a probe kind."""

    kind: str
    # (S, 3) shank tips about the row's centre in the probe frame, z zeroed.
    shank_offsets: NDArray[np.float64]
    active_mm: tuple[float, float]  # the recorded stretch, from the tip up
    count: int  # how many the inventory holds

    @property
    def window_mm(self) -> float:
        return self.active_mm[1] - self.active_mm[0]


@dataclass(frozen=True)
class Line:
    """Where a probe records a unit from, and how much of each region."""

    kind: str
    hole: int
    share: dict[int, float]  # region index -> share recorded
    entry: NDArray[np.float64]  # on the bore's top section
    direction: NDArray[np.float64]  # unit, into the brain
    ap_deg: float
    ml_deg: float
    spin_deg: float
    start_mm: float  # along the line, where the active stretch begins

    @property
    def total(self) -> float:
        return sum(self.share.values())


Unit = frozenset[int]
# unit -> bore -> kind -> the best line found
Units = dict[Unit, dict[int, dict[str, Line]]]


@dataclass(frozen=True)
class Cover:
    """A partition of the regions into probes, each seated on its own bore."""

    lines: dict[Unit, Line]
    n_regions: int = field(repr=False)

    @property
    def n_probes(self) -> int:
        return len(self.lines)

    @property
    def share(self) -> dict[int, float]:
        return {r: s for line in self.lines.values() for r, s in line.share.items()}

    @property
    def mean_share(self) -> float:
        return sum(self.share.values()) / self.n_regions


def shaft_down_LPS(ap_deg: NDArray, ml_deg: NDArray) -> NDArray[np.float64]:
    """Shaft direction into the brain: ``-arc_angles_to_affine(ap, ml, ·)[:, 2]``."""
    ap, ml = np.deg2rad(ap_deg), np.deg2rad(ml_deg)
    return np.stack([np.sin(ml), np.sin(ap) * np.cos(ml), -np.cos(ap) * np.cos(ml)], -1)


def section_points(
    section: HoleSection, n_ring: int = 6, radius: float = 0.5
) -> NDArray[np.float64]:
    """The section's centre and a ring at ``radius`` of its semi-axes."""
    e1, e2 = cap_basis(np.asarray(section.axis, float))
    c, s = np.cos(section.theta), np.sin(section.theta)
    centre = np.asarray(section.center, float)
    points = [centre]
    for t in np.linspace(0, 2 * np.pi, n_ring, endpoint=False):
        u, v = radius * section.a * np.cos(t), radius * section.b * np.sin(t)
        points.append(centre + (c * u - s * v) * e1 + (s * u + c * v) * e2)
    return np.array(points)


def threads(hole: Hole, p: NDArray, d: NDArray, margin_mm: float) -> NDArray[np.bool_]:
    """(N,) whether each line ``p + t d`` clears every section oval and wall."""
    ok = np.ones(len(p), bool)
    for section in hole.sections:
        axis = np.asarray(section.axis, float)
        e1, e2 = cap_basis(axis)
        t = ((section.center - p) @ axis) / (d @ axis)
        at = p + t[:, None] * d
        u, v = (at - section.center) @ e1, (at - section.center) @ e2
        c, s = np.cos(section.theta), np.sin(section.theta)
        a, b = section.a - margin_mm, section.b - margin_mm
        ok &= ((c * u + s * v) / a) ** 2 + ((-s * u + c * v) / b) ** 2 <= 1.0
        for wall in hole.walls:
            ok &= (at - wall.point) @ wall.normal <= -margin_mm
    return ok


def bore_lines(
    hole: Hole,
    ap_window_deg: tuple[float, float],
    step_deg: float,
    margin_mm: float,
    ml_range_deg: tuple[float, float] = DEFAULT_ML_RANGE_DEG,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """``(entry, direction, ap, ml)`` of the lines that thread ``hole``."""
    aps = np.arange(ap_window_deg[0], ap_window_deg[1] + 1e-9, step_deg)
    mls = np.arange(ml_range_deg[0], ml_range_deg[1] + 1e-9, step_deg)
    ap, ml = (g.ravel() for g in np.meshgrid(aps, mls, indexing="ij"))
    d = shaft_down_LPS(ap, ml)
    tops = section_points(hole.sections[0])
    P, D = np.repeat(tops, len(d), axis=0), np.tile(d, (len(tops), 1))
    AP, ML = np.tile(ap, len(tops)), np.tile(ml, len(tops))
    ok = threads(hole, P, D, margin_mm)
    return P[ok], D[ok], AP[ok], ML[ok]


def _window_share(
    along: NDArray,
    hit: NDArray,
    region_of: NDArray,
    counts: NDArray,
    window_mm: float,
    step_mm: float,
) -> NDArray[np.float64]:
    """(L, starts, R): each region's share inside the stretch starting at each depth."""
    n_bins = int(round(LINE_DEPTH_MM / step_mm))
    w = int(round(window_mm / step_mm))
    L, R = len(along), len(counts)
    bins = np.floor(along / step_mm).astype(int)
    li, ni = np.nonzero(hit & (bins >= 0) & (bins < n_bins))
    flat = (li * n_bins + bins[li, ni]) * R + region_of[ni]
    hist = np.bincount(flat, minlength=L * n_bins * R).reshape(L, n_bins, R)
    csum = np.concatenate([np.zeros((L, 1, R)), np.cumsum(hist, axis=1)], axis=1)
    return (csum[:, w:] - csum[:, :-w]) / counts


@dataclass
class _Rows:
    """Stretches that record something, for one bore, kind and spin."""

    share: NDArray  # (K, R)
    line: NDArray  # (K,) index into the bore's lines
    start_mm: NDArray  # (K,)
    spin_deg: float


def _bore_rows(
    hole: Hole,
    lines: tuple[NDArray, NDArray, NDArray, NDArray],
    shape: ProbeShape,
    points: NDArray,
    region_of: NDArray,
    counts: NDArray,
    *,
    radius_mm: float,
    margin_mm: float,
    depth_step_mm: float,
    spin_step_deg: float,
) -> Iterator[_Rows]:
    P, D, AP, ML = lines
    offsets = np.asarray(shape.shank_offsets, float)
    one_shank = len(offsets) == 1 and not offsets.any()
    # A row of shanks centred on the line repeats every half turn.
    spins = [0.0] if one_shank else np.arange(0.0, 180.0, spin_step_deg)
    reach = radius_mm + float(np.linalg.norm(offsets, axis=1).max())
    starts = np.arange(0.0, LINE_DEPTH_MM, depth_step_mm)
    for lo in range(0, len(P), LINES_PER_CHUNK):
        sl = slice(lo, lo + LINES_PER_CHUNK)
        p, d = P[sl], D[sl]
        rel = points[None, :, :] - p[:, None, :]
        along = np.einsum("lnk,lk->ln", rel, d)
        across = rel - along[..., None] * d[:, None, :]
        li, ni = np.nonzero(np.einsum("lnk,lnk->ln", across, across) <= reach**2)
        near = across[li, ni][:, None, :]
        for spin in spins:
            # (L, S, 3): each line's shank row in LPS. The offsets are across
            # the shaft, so a shank's line is the centre line moved by one.
            rows = np.stack(
                [
                    (np.asarray(arc_angles_to_affine(a, m, spin))[:3, :3] @ offsets.T).T
                    for a, m in zip(AP[sl], ML[sl])
                ]
            )
            fits = np.ones(len(p), bool)
            if not one_shank:
                for k in range(rows.shape[1]):
                    fits &= threads(hole, p + rows[:, k], d, margin_mm)
            if not fits.any():
                continue
            gap = near - rows[li]
            hit = np.zeros(along.shape, bool)
            hit[li, ni] = (np.einsum("ksj,ksj->ks", gap, gap) <= radius_mm**2).any(1)
            hit[~fits] = False
            share = _window_share(
                along, hit, region_of, counts, shape.window_mm, depth_step_mm
            )
            n_starts = share.shape[1]
            # The stretch ends ``active_mm[0]`` short of the tip.
            tip = starts[:n_starts] + shape.window_mm + shape.active_mm[0]
            share[:, tip > SHANK_LENGTH_MM] = 0.0
            line, start = np.nonzero(share.any(axis=-1))
            if len(line):
                yield _Rows(share[line, start], lo + line, starts[start], float(spin))


def find_units(
    clouds: Sequence[NDArray],
    holes: Sequence[Hole],
    ap_window_deg: tuple[float, float],
    shapes: Sequence[ProbeShape],
    *,
    radius_mm: float,
    max_group: int = 3,
    min_kept: float = 0.1,
    angle_step_deg: float = 3.0,
    spin_step_deg: float = 15.0,
    depth_step_mm: float = 0.1,
    margin_mm: float = DEFAULT_THREADING_MARGIN_MM,
    ml_range_deg: tuple[float, float] = DEFAULT_ML_RANGE_DEG,
) -> Units:
    """Per group of up to ``max_group`` regions, bore and kind, its best line.

    A group's line is the one recording the most summed over its regions, among
    those that keep, of every region, at least ``min_kept`` of what the best
    line for that region alone records on the same kind.
    """
    R = len(clouds)
    points = np.concatenate([np.asarray(c, float).reshape(-1, 3) for c in clouds])
    region_of = np.repeat(np.arange(R), [len(c) for c in clouds])
    counts = np.maximum(np.bincount(region_of, minlength=R), 1).astype(float)

    found: list[tuple[Hole, ProbeShape, tuple, _Rows]] = []
    alone = {shape.kind: np.zeros(R) for shape in shapes}
    for hole in holes:
        lines = bore_lines(hole, ap_window_deg, angle_step_deg, margin_mm, ml_range_deg)
        for shape in shapes:
            for rows in _bore_rows(
                hole,
                lines,
                shape,
                points,
                region_of,
                counts,
                radius_mm=radius_mm,
                margin_mm=margin_mm,
                depth_step_mm=depth_step_mm,
                spin_step_deg=spin_step_deg,
            ):
                found.append((hole, shape, lines, rows))
                alone[shape.kind] = np.maximum(
                    alone[shape.kind], rows.share.max(axis=0)
                )

    groups = [
        frozenset(c)
        for k in range(1, max_group + 1)
        for c in itertools.combinations(range(R), k)
    ]
    units: Units = {}
    for hole, shape, (P, D, AP, ML), rows in found:
        enough = (rows.share > 0) & (rows.share >= min_kept * alone[shape.kind])
        for group in groups:
            idx = sorted(group)
            valid = enough[:, idx].all(axis=1)
            if not valid.any():
                continue
            total = np.where(valid, rows.share[:, idx].sum(axis=1), -1.0)
            i = int(total.argmax())
            current = units.get(group, {}).get(hole.id, {}).get(shape.kind)
            if current is not None and current.total >= total[i]:
                continue
            line = int(rows.line[i])
            units.setdefault(group, {}).setdefault(hole.id, {})[shape.kind] = Line(
                shape.kind,
                hole.id,
                {r: float(rows.share[i, r]) for r in idx},
                P[line],
                D[line],
                float(AP[line]),
                float(ML[line]),
                rows.spin_deg,
                float(rows.start_mm[i]),
            )
    return units


def partitions(items: list[int], groups: Sequence[Unit]) -> Iterator[list[Unit]]:
    """Every partition of ``items`` into single regions and the given groups."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    yield from ([frozenset([first]), *p] for p in partitions(rest, groups))
    for g in groups:
        if min(g) == first and g <= set(items):
            others = [i for i in rest if i not in g]
            yield from ([g, *p] for p in partitions(others, groups))


def seat(
    blocks: Sequence[Unit], units: Units, shapes: Sequence[ProbeShape]
) -> dict[Unit, Line] | None:
    """Each block on a bore of its own, at the kinds recording the most in all.

    ``None`` when no choice of kinds within the inventory seats every block.
    """
    hole_ids = sorted({h for b in blocks for h in units.get(b, {})})
    if len(hole_ids) < len(blocks):
        return None
    total = {
        shape.kind: np.array(
            [
                [
                    line.total if line is not None else -np.inf
                    for h in hole_ids
                    for line in [units.get(b, {}).get(h, {}).get(shape.kind)]
                ]
                for b in blocks
            ]
        )
        for shape in shapes
    }
    options = [
        [s.kind for s in shapes if np.isfinite(total[s.kind][i]).any()]
        for i in range(len(blocks))
    ]
    best: tuple[float, dict[Unit, Line]] | None = None
    for kinds in itertools.product(*options):
        if any(kinds.count(s.kind) > s.count for s in shapes):
            continue
        score = np.array([total[kind][i] for i, kind in enumerate(kinds)])
        # A bore the block cannot use must lose to any that it can.
        rows, cols = linear_sum_assignment(np.where(np.isfinite(score), -score, 1e6))
        chosen = score[rows, cols]
        if not np.isfinite(chosen).all():
            continue
        if best is None or chosen.sum() > best[0]:
            best = (
                float(chosen.sum()),
                {
                    blocks[i]: units[blocks[i]][hole_ids[j]][kinds[i]]
                    for i, j in zip(rows, cols)
                },
            )
    return None if best is None else best[1]


def covers(n_regions: int, units: Units, shapes: Sequence[ProbeShape]) -> list[Cover]:
    """Every cover that can be seated, fewest probes first, then most recorded."""
    groups = [g for g in units if len(g) > 1]
    out = []
    for blocks in partitions(list(range(n_regions)), groups):
        lines = seat(blocks, units, shapes)
        if lines is not None:
            out.append(Cover(lines, n_regions))
    out.sort(key=lambda c: (c.n_probes, -c.mean_share))
    return out


def best_per_count(found: Sequence[Cover], per_count: int = 1) -> list[Cover]:
    """The first ``per_count`` covers at each probe count."""
    kept: dict[int, list[Cover]] = {}
    for cover in found:
        at = kept.setdefault(cover.n_probes, [])
        if len(at) < per_count:
            at.append(cover)
    return [c for at in kept.values() for c in at]


def aim_points(
    line: Line, clouds: Mapping[int, NDArray], shape: ProbeShape, radius_mm: float
) -> dict[int, NDArray[np.float64]]:
    """Per region of the line, the point on it midway through what it records there."""
    R = np.asarray(
        arc_angles_to_affine(line.ap_deg, line.ml_deg, line.spin_deg), float
    )[:3, :3]
    rows = (R @ np.asarray(shape.shank_offsets, float).T).T
    out = {}
    for region in line.share:
        rel = np.asarray(clouds[region], float) - line.entry
        along = rel @ line.direction
        across = rel - along[:, None] * line.direction
        gap = across[:, None, :] - rows[None, :, :]
        hit = (np.einsum("nsj,nsj->ns", gap, gap) <= radius_mm**2).any(axis=1)
        hit &= (along >= line.start_mm) & (along <= line.start_mm + shape.window_mm)
        middle = (
            float(np.median(along[hit]))
            if hit.any()
            else (line.start_mm + shape.window_mm / 2)
        )
        out[region] = line.entry + middle * line.direction
    return out
