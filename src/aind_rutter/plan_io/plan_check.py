"""How a plan's probes thread their holes on a located implant, and how far each
would have to move to thread again (``rutter-tracks check``).

A calibrated probe is driven to its target along the angle the calibration
reports, so an implant off its design position, or an angle off its plan,
moves where the probe crosses its hole rather than where it ends. A probe that
no longer threads is put back by its target offset: the horizontal shift that
restores the margin, the least of them, is what the planner's offsets would
have to change by.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from aind_rutter.plan_io.insertion_tracks import (
    Line,
    best_hole,
    lines_clearance,
    shank_tips_local,
)


def probe_lines(
    state: Any, catalog: Any, name: str, d_ap: float = 0.0, d_ml: float = 0.0
) -> list[Line]:
    """Each shank's line for a probe whose AP and ML land ``d_ap``/``d_ml``
    off its plan, still driven to its target."""
    from aind_rutter.domain.pose import ProbePose, resolved_angles
    from aind_rutter.domain.rig import probe_rotation

    if d_ap or d_ml:
        ap, ml, _ = resolved_angles(name, state)
        probe = replace(
            state.probes[name],
            bind_ap_to_arc=False,
            ap_local=ap + d_ap,
            ml_local=ml + d_ml,
        )
        state = replace(state, probes={**state.probes, name: probe})
    pose = ProbePose.from_planning_state(state, name, catalog=catalog)
    R = probe_rotation(pose.ap, pose.ml, pose.spin)
    down = -R[:, 2]
    return [
        (pose.tip + R @ tip, down) for tip in shank_tips_local(state.probes[name].kind)
    ]


def rethread(
    lines: Sequence[Line], hole: Any, shift: NDArray[np.floating], margin_mm: float
) -> NDArray[np.floating]:
    """The least horizontal move (LPS mm) of a probe that clears ``hole`` by
    ``margin_mm``; zero when it already does."""

    def clear(v: NDArray[np.floating]) -> float:
        moved = [(p + np.r_[v, 0.0], d) for p, d in lines]
        return lines_clearance(moved, hole, shift) - margin_mm

    if clear(np.zeros(2)) >= 0:
        return np.zeros(2)
    sol = minimize(
        lambda v: v @ v,
        np.zeros(2),
        constraints=[{"type": "ineq", "fun": clear}],
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-12},
    )
    return np.asarray(sol.x)


@dataclass(frozen=True)
class ProbeCheck:
    probe: str
    hole: int
    clearance: float
    worst_off_angle: float
    rethread_RA: tuple[float, float]


def check_plan(
    state: Any,
    catalog: Any,
    holes: Mapping[int, Any],
    shifts: Mapping[str, NDArray[np.floating]],
    angle_step_deg: float = 1.5,
    margin_mm: float = 0.04,
    landing: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, int], dict[str, list[ProbeCheck]]]:
    """Per implant placement (``shifts``: label → LPS shift, or per-probe shifts
    as label → {probe: shift}), each probe's clearance in its planned hole, its
    worst clearance with AP and ML each up to ``angle_step_deg`` off, and the
    target move that threads it again by ``margin_mm``.

    ``landing`` gives each probe's expected AP and ML off plan once dialed;
    the off-angle steps are taken about it. The planned hole is the one each
    probe clears best at the design position and planned angles.
    """
    names = list(state.probes)
    land = {n: (landing or {}).get(n, (0.0, 0.0)) for n in names}
    hole_of = {n: best_hole(probe_lines(state, catalog, n), holes) for n in names}
    nominal = {n: probe_lines(state, catalog, n, *land[n]) for n in names}
    steps = (-angle_step_deg, 0.0, angle_step_deg)
    off = {
        n: [
            probe_lines(state, catalog, n, land[n][0] + a, land[n][1] + m)
            for a in steps
            for m in steps
            if a or m
        ]
        for n in names
    }
    out: dict[str, list[ProbeCheck]] = {}
    for label, shift in shifts.items():
        rows = []
        for n in names:
            t = np.asarray(shift[n] if isinstance(shift, Mapping) else shift, float)
            hole = holes[hole_of[n]]
            c = lines_clearance(nominal[n], hole, t)
            worst = min(lines_clearance(ls, hole, t) for ls in off[n])
            v = rethread(nominal[n], hole, t, margin_mm)
            rows.append(
                ProbeCheck(n, hole_of[n], c, worst, (float(-v[0]), float(-v[1])))
            )
        out[label] = rows
    return hole_of, out


def rethreaded_plan(
    plan: Mapping[str, Any], rows: Sequence[ProbeCheck]
) -> dict[str, Any]:
    """``plan`` with each probe's ``offsets_RA`` moved by its re-thread; angles
    are kept, so each probe's line moves sideways by exactly that much."""
    out = {**plan, "probes": {n: dict(p) for n, p in plan["probes"].items()}}
    for r in rows:
        p = out["probes"][r.probe]
        r0, a0 = p.get("offsets_RA") or (0.0, 0.0)
        p["offsets_RA"] = [
            round(float(r0) + r.rethread_RA[0], 4),
            round(float(a0) + r.rethread_RA[1], 4),
        ]
    return out


def dialed_plan(plan: Mapping[str, Any], sheet: Any) -> dict[str, Any]:
    """``plan`` with its arcs and sliders at ``sheet``'s dials and each probe
    carrying its manipulator's bias, so each probe lands where ``plan`` put it
    up to what a shared arc cannot dial out."""
    arcs = dict(plan.get("arcs") or {})
    for arc, a in sheet.arcs.items():
        arcs[arc] = round(a["dial_rig_ap"] - sheet.head_pitch_deg, 4)
    probes = {n: dict(p) for n, p in plan["probes"].items()}
    for n, d in sheet.probes.items():
        probes[n].update(
            slider_ml=d["dial_ml"], ap_bias=d["ap_bias"], ml_bias=d["ml_bias"]
        )
    return {**plan, "arcs": arcs, "probes": probes}


def probe_gaps(cfg: Any, plan: Mapping[str, Any]) -> list[Any]:
    """FCL gaps, closest first, of every pair the collision check tests for
    ``plan`` applied to the loaded config ``cfg``."""
    from aind_rutter.build.assemble import build_runtime_from_config
    from aind_rutter.collision.adapter import CollisionAdapter
    from aind_rutter.collision.fcl import FCLBackend
    from aind_rutter.config import PlanningModel
    from aind_rutter.plan_io.replay import apply_plan_model_to_state
    from aind_rutter.session.store import PlanStore

    runtime = build_runtime_from_config(cfg)
    store = PlanStore(replace(runtime.plan_state, calibrations={}))
    apply_plan_model_to_state(PlanningModel.model_validate(plan), store)
    backend = FCLBackend()
    CollisionAdapter(
        backend=backend, scene=runtime.scene, assets=runtime.asset_catalog
    ).rebuild(store.state)
    return backend.gaps()


def report_checks(
    checks: Mapping[str, Sequence[ProbeCheck]], angle_step_deg: float
) -> list[str]:
    lines = []
    for label, rows in checks.items():
        lines.append(f"{label}:")
        lines.append(
            f"  {'probe':<12} {'hole':>4} {'clearance':>9} "
            f"{f'±{angle_step_deg:g}° worst':>11}  re-thread R, A (mm)"
        )
        for r in rows:
            move = (
                ""
                if not any(r.rethread_RA)
                else f"{r.rethread_RA[0]:+.2f}, {r.rethread_RA[1]:+.2f}"
            )
            lines.append(
                f"  {r.probe:<12} {r.hole:>4} {r.clearance:+9.3f} "
                f"{r.worst_off_angle:+11.3f}  {move}"
            )
    return lines
