"""Each probe's threaded best: the coverage it reaches alone through a bore it
threads, which the coverage floor is a fraction of.

Every probe runs Phase 1's full stage by itself, seeded from the visibility
atlas's anchors in each bore it passes, and keeps the best coverage among the
results that meet the feasibility spec's threading, well and brain rows. Other
probes are absent, so the best is an upper bound on what a layout can give that
probe.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np

from aind_rutter.optimization.objectives.constrained import Phase2Weights, make_phase2
from aind_rutter.optimization.objectives.coverage import (
    coverage_per_probe_over_probes,
)
from aind_rutter.optimization.objectives.evaluate import poses_from_x
from aind_rutter.optimization.objectives.soft import pack_statics
from aind_rutter.optimization.objectives.spin_angle import (
    from_angle_layout,
    to_angle_layout,
)
from aind_rutter.optimization.objectives.variables import build_y
from aind_rutter.optimization.pipeline.fixtures import build_coverage_data
from aind_rutter.optimization.pipeline.settings import Phase1Settings

# The rows a lone probe can fail; the others are empty for one probe on one arc.
_JUDGED_GROUPS = ("thread", "probe_fixture", "brain")
# Slack a judged row may fall short by; Phase 2's keep band is no tighter.
FEASIBLE_TOL = 1e-3


def spread_seeds(entry, lo_ap: float, hi_ap: float, n_seeds: int) -> list[tuple]:
    """Up to ``n_seeds`` of an atlas entry's ``(ap, ml, spin)`` anchors inside the
    reachable AP window, evenly spaced in spin so each spin basin gets a start."""
    anchors = [
        (a.ap_deg, a.ml_deg, a.spin_deg)
        for a in entry.anchors
        if lo_ap <= a.ap_deg <= hi_ap
    ]
    if not anchors:
        return []
    anchors.sort(key=lambda a: a[2])
    picks = np.unique(np.linspace(0, len(anchors) - 1, n_seeds).round().astype(int))
    return [anchors[i] for i in picks]


def _seed_rows(p, holes, atlas, lo_ap, hi_ap, n_seeds, build_static):
    """Statics, start pose and bore id of every seed of probe ``p``."""
    statics, x0_rows, row_hole = [], [], []
    z = np.zeros(1)
    for h in holes:
        entry = atlas.entries.get((p.name, h.id))
        if entry is None or entry.ap_min is None:
            continue
        seeds = spread_seeds(entry, lo_ap, hi_ap, n_seeds)
        if not seeds:
            continue
        st = build_static(
            [p],
            SimpleNamespace(probe_to_hole={p.name: h.id}),
            SimpleNamespace(probe_to_arc_idx={p.name: 0}),
        )
        for ap, ml, spin in seeds:
            statics.append(st)
            row_hole.append(h.id)
            x0_rows.append(
                build_y(np.array([ap]), 1, np.array([ml]), np.array([spin]), z, z, z)
            )
    return statics, x0_rows, row_hole


def _best_feasible_by_hole(
    statics, x_out, row_hole, cov, well_soft, brain_sdf, weights, grid_dtype
) -> dict[int, float]:
    """Best coverage per bore among the solved poses that meet the judged rows."""
    per_hole: dict[int, float] = {}
    problems: dict[int, Any] = {}
    for st, x, hid in zip(statics, x_out, row_hole):
        if hid not in problems:
            problems[hid] = make_phase2(
                st,
                1,
                coverage_data=cov,
                fixtures=(well_soft,),
                weights=weights,
                brain_sdf=brain_sdf,
                grid_dtype=grid_dtype,
            )
        parts = problems[hid]["slack_parts"](x)
        worst = min(
            (float(parts[g].min()) for g in _JUDGED_GROUPS if parts[g].size),
            default=0.0,
        )
        if worst < -FEASIBLE_TOL:
            continue
        packed = pack_statics(st, 1)
        Rs, ts = poses_from_x(
            jnp.asarray(x, jnp.float32),
            1,
            1,
            packed["target_LPS"],
            packed["pivot_local"],
            packed["arc_idx"],
        )
        c = float(
            coverage_per_probe_over_probes(
                Rs, ts, packed["tips_local"], packed["shank_mask"], cov
            )[0]
        )
        per_hole[hid] = max(per_hole.get(hid, -np.inf), c)
    return per_hole


def threaded_coverage_ceilings(
    *,
    settings: Phase1Settings,
    atlas,
    probes,
    holes,
    build_static: Callable,
    well_soft,
    brain_sdf,
    make_kernel: Callable[..., tuple[Callable, Callable, Callable]],
    bounds: tuple[np.ndarray, np.ndarray],
    ap_window: tuple[float, float],
    grid_dtype,
    n_seeds: int = 8,
) -> tuple[dict[str, float], dict[str, dict[int, float]]]:
    """Each probe's best feasible coverage over the bores the atlas says it
    passes, and the best per bore (absent where no seed came out feasible).

    ``build_static(probes, hole_assignment, arc_assignment)`` builds a probe's
    statics, ``make_kernel(statics, coverage, settings)`` is Phase 1's one-arc
    kernel, and
    ``bounds`` its box in the layout it steps; ``ap_window`` limits the seeds.

    Raises
    ------
    ValueError
        If a probe has no feasible pose through any bore, which no layout could
        then lift past a floor.
    """
    # Coverage normalization and the floor both read these ceilings, so neither
    # may shape the solves that produce them.
    lone = settings.model_copy(update={"cov_norm": False, "min_coverage_fraction": 0.0})
    weights = Phase2Weights(settings.feasibility_spec(), lone.objective_spec())
    spin_angle = settings.spin_param == "angle"
    lo, hi = bounds
    lo_ap, hi_ap = ap_window
    chunk = settings.chunk

    best: dict[str, float] = {}
    by_hole: dict[str, dict[int, float]] = {}
    for p in probes:
        statics, x0_rows, row_hole = _seed_rows(
            p, holes, atlas, lo_ap, hi_ap, n_seeds, build_static
        )
        if not x0_rows:
            raise ValueError(f"{p.name}: the atlas finds no bore it passes")
        x0 = np.stack(x0_rows).astype(np.float32)
        if spin_angle:
            x0 = to_angle_layout(x0, 1)
        cov = build_coverage_data([p], statics[0])
        _vobj, barg, rprop = make_kernel(statics[0], cov, lone)
        n_rows = x0.shape[0]
        n_pad = (-n_rows) % chunk
        padded = statics + [statics[-1]] * n_pad
        x_all = np.concatenate([x0, np.repeat(x0[-1:], n_pad, 0)], 0)
        x_out = np.empty_like(x_all)
        for s in range(0, x_all.shape[0], chunk):
            x = jnp.asarray(x_all[s : s + chunk])
            x = rprop(x, barg(padded[s : s + chunk]), lo, hi, 1.0, settings.stage2)
            x_out[s : s + chunk] = np.asarray(x)
        x_out = x_out[:n_rows]
        if spin_angle:
            x_out = np.asarray(from_angle_layout(x_out, 1))
        per_hole = _best_feasible_by_hole(
            statics, x_out, row_hole, cov, well_soft, brain_sdf, weights, grid_dtype
        )
        if not per_hole or max(per_hole.values()) <= 0.0:
            raise ValueError(
                f"{p.name}: no feasible pose through any bore covers its target, so "
                "no layout meets a coverage floor; unset RUTTER_MIN_COVERAGE_FRACTION"
            )
        best[p.name] = max(per_hole.values())
        by_hole[p.name] = per_hole
    return best, by_hole
