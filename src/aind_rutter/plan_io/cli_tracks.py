"""The ``rutter-tracks`` command: what past insertions say about the implant and
the rig, and what that means for the next plan.

rutter-tracks locate tracks.csv --holes H.holes.yml --config C.yml --calibrations ROOT
rutter-tracks bias   sessions.yml
rutter-tracks check  C.yml --plan P.plan.yml --holes H.holes.yml --fit implants.yml \\
                     --biases bias.yml --assign PROBE=MANIPULATOR ...

``locate`` fits each subject's implant from its insertion tracks
(:mod:`~aind_rutter.plan_io.insertion_tracks`); ``bias`` measures each
manipulator's angle error against the plans it was dialed from
(:mod:`~aind_rutter.plan_io.manipulator_bias`); ``check`` threads a plan
through each fitted implant and writes the dial sheet
(:mod:`~aind_rutter.plan_io.plan_check`).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import tyro
import yaml

from aind_rutter.config.locations import Locator, parse_overrides
from aind_rutter.plan_io.insertion_tracks import ErrorModel


@dataclass
class Locate:
    """Fit each subject's implant from its insertion tracks."""

    tracks: tyro.conf.Positional[Path]
    """CSV of tracks, one row per insertion (see ``insertion_tracks.TRACK_COLUMNS``)."""

    holes: Path
    """Bores file of the implant, in the implant's own frame."""

    config: Path
    """A config declaring the calibrations' reticle; only its reticles are read."""

    calibrations: Path
    """Folder holding the calibration directories the tracks name."""

    reticle: str | None = None
    """Reticle the calibrations were taken on; needed when the config has several."""

    errors: ErrorModel = "manipulator"
    """What each error is shared by: a manipulator across days, one calibration,
    or nothing."""

    sigma_mm: float = 0.15
    """Prior scale of each error."""

    out: Path | None = None
    """Where to write the fit as YAML."""


@dataclass
class Bias:
    """Measure each manipulator's angle error against the plans it was dialed from."""

    sessions: tyro.conf.Positional[Path]
    """YAML: ``calibrations_root`` and ``sessions``, each a ``label``, ``config``,
    optional ``plan`` and its ``calibrations`` directories in order."""

    out: Path | None = None
    """Where to write the biases as YAML."""


@dataclass
class Check:
    """Thread a plan through each fitted implant and write the dial sheet."""

    config: tyro.conf.Positional[Path]
    """The plan's config."""

    plan: Path
    """Plan file to check."""

    holes: Path
    """Bores file of the implant, in the implant's own frame."""

    fit: Path | None = None
    """``rutter-tracks locate`` output; without it, only the design position."""

    biases: Path | None = None
    """``rutter-tracks bias`` output, for the dial sheet."""

    assign: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """``PROBE=MANIPULATOR``: which manipulator carries each probe; repeatable."""

    with_errors: bool = True
    """Add each assigned manipulator's fitted error to its implant shift."""

    angle_step_deg: float = 1.5
    """How far off plan AP and ML may each land after the dial corrections."""

    margin_mm: float = 0.04
    """Clearance a re-threaded probe should keep inside its bore."""

    out: Path | None = None
    """Where to write the dial sheet and checks as YAML."""

    configs: Path | None = None
    """Folder to write, per fitted subject, the config with its implant moved
    to the fit and each probe's offsets re-threaded, its plan, and the
    collision gaps of that plan."""

    paths: Annotated[tuple[Path, ...], tyro.conf.UseAppendAction] = ()
    """Paths file locating the config's ``paths:`` names; repeatable."""

    path: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """``NAME=DIR`` for one ``paths:`` name; repeatable."""

    offline: bool = False
    """Fetch nothing while loading the config."""


def _load_holes(path: Path) -> dict[int, Any]:
    from aind_rutter.optimization.geometry.holes import load_holes

    return {h.id: h for h in load_holes(path)}


def _write(out: Path | None, record: dict[str, Any]) -> None:
    if out is not None:
        out.write_text(yaml.safe_dump(record, sort_keys=False))
        print(f"Wrote {out}")


def locate(args: Locate) -> None:
    from aind_rutter.plan_io.insertion_tracks import (
        CalibrationBanks,
        fit_implants,
        fit_record,
        load_tracks,
        place_tracks,
        report_fit,
        reticles_of,
    )

    reticles = reticles_of(args.config)
    reticle = args.reticle or (next(iter(reticles)) if len(reticles) == 1 else None)
    if reticle is None:
        raise ValueError(f"name the --reticle; {args.config} has {sorted(reticles)}")
    holes = _load_holes(args.holes)
    banks = CalibrationBanks(args.calibrations, reticles, reticle)
    placed = place_tracks(load_tracks(args.tracks), banks, holes)
    fit = fit_implants(placed, holes, args.errors, args.sigma_mm)
    print("\n".join(report_fit(fit)))
    _write(args.out, fit_record(fit, args.holes, args.tracks))


def bias(args: Bias) -> None:
    from aind_rutter.plan_io.manipulator_bias import (
        bias_record,
        load_sessions,
        report_samples,
        session_samples,
    )

    root, sessions = load_sessions(args.sessions)
    samples = [s for session in sessions for s in session_samples(session, root)]
    print("\n".join(report_samples(samples)))
    _write(args.out, bias_record(samples, args.sessions))


def _assignment(items: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for item in items:
        probe, sep, manipulator = item.partition("=")
        if not sep:
            raise ValueError(f"--assign {item!r}: expected PROBE=MANIPULATOR")
        out[probe.strip()] = manipulator.strip()
    return out


def check(args: Check) -> None:
    from aind_rutter.build.assemble import build_runtime_from_config
    from aind_rutter.config import ConfigModel, PlanningModel
    from aind_rutter.plan_io.insertion_tracks import from_ras
    from aind_rutter.plan_io.manipulator_bias import dial_sheet, load_biases
    from aind_rutter.plan_io.plan_check import check_plan, report_checks
    from aind_rutter.plan_io.replay import apply_plan_model_to_state
    from aind_rutter.session.store import PlanStore

    locator = Locator.from_environment(
        args.paths, parse_overrides(args.path), offline=args.offline
    )
    cfg = ConfigModel.from_yaml(args.config, locator)
    runtime = build_runtime_from_config(cfg)
    plan_raw = yaml.safe_load(args.plan.read_text())
    # Planned angles, not a calibration's: the check is of what the plan asks for.
    store = PlanStore(replace(runtime.plan_state, calibrations={}))
    apply_plan_model_to_state(PlanningModel.model_validate(plan_raw), store)
    state = store.state
    assignment = _assignment(args.assign)
    unknown = sorted(set(assignment) - set(state.probes))
    if unknown:
        raise ValueError(f"--assign names probes the plan lacks: {unknown}")

    record: dict[str, Any] = {"config": str(args.config), "plan": str(args.plan)}
    landing: dict[str, tuple[float, float]] = {}
    sheet = None
    if args.biases is not None:
        if not assignment:
            raise ValueError("a dial sheet needs --assign for every probe")
        sheet = dial_sheet(
            plan_raw,
            assignment,
            load_biases(args.biases),
            state.kinematics.head_pitch_deg,
        )
        print(f"Dial sheet (rig AP = subject AP + {sheet.head_pitch_deg:g}°):")
        for arc, a in sheet.arcs.items():
            print(
                f"  arc {arc}: plan {a['planned_rig_ap']:+.1f}°, "
                f"dial {a['dial_rig_ap']:+.1f}° ({-a['correction']:+.1f}) "
                f"for {', '.join(a['probes'])}"
            )
        for p, d in sheet.probes.items():
            note = "" if d["known"] else " (no record; fleet AP bias assumed)"
            print(
                f"  {p:<12} on {d['manipulator']}: ML plan {d['planned_ml']:+.2f}°, "
                f"dial {d['dial_ml']:+.2f}°{note}"
            )
        landing = {p: (d["landing_ap_offset"], 0.0) for p, d in sheet.probes.items()}
        record["dial_sheet"] = {
            "head_pitch_deg": sheet.head_pitch_deg,
            "arcs": sheet.arcs,
            "probes": sheet.probes,
        }

    shifts: dict[str, Any] = {"design position": np.zeros(3)}
    located: dict[str, Any] = {}
    if args.fit is not None:
        fit = yaml.safe_load(args.fit.read_text())
        errors = fit.get(f"{fit['errors']}_errors_RAS_mm") or {}
        for subject, s in fit["subjects"].items():
            base = from_ras(s["implant_shift_RAS_mm"])
            shifts[subject] = base
            with_errors = None
            if args.with_errors and fit["errors"] == "manipulator" and assignment:
                with_errors = f"{subject} (with manipulator errors)"
                shifts[with_errors] = {
                    n: base + from_ras(errors.get(assignment.get(n, ""), [0, 0, 0]))
                    for n in state.probes
                }
            located[subject] = (base, with_errors)
    holes = _load_holes(args.holes)
    hole_of, checks = check_plan(
        state,
        runtime.asset_catalog,
        holes,
        shifts,
        args.angle_step_deg,
        args.margin_mm,
        landing,
    )
    if landing:
        print("Probes land at the dialed arc plus their own AP bias:")
        for n, (d_ap, _) in landing.items():
            print(f"  {n:<12} {d_ap:+.2f}° AP off plan")
    print("\n".join(report_checks(checks, args.angle_step_deg)))
    record["holes"] = hole_of
    record["checks"] = {
        label: [
            {
                "probe": r.probe,
                "hole": r.hole,
                "clearance_mm": round(r.clearance, 3),
                f"worst_at_{args.angle_step_deg:g}deg_mm": round(r.worst_off_angle, 3),
                "rethread_RA_mm": [round(v, 3) for v in r.rethread_RA],
            }
            for r in rows
        ]
        for label, rows in checks.items()
    }
    if args.configs is not None:
        record["configs"] = _write_located(
            args, plan_raw, checks, located, sheet, locator
        )
    _write(args.out, record)


def _write_located(
    args: Check,
    plan_raw: dict[str, Any],
    checks: dict[str, Any],
    located: dict[str, tuple[Any, str | None]],
    sheet: Any,
    locator: Locator,
) -> dict[str, Any]:
    """Per subject: the config with its implant moved to the fit, the plan with
    each probe re-threaded through it at its landing angle, and that plan's
    collision gaps. With a dial sheet, the plan's arcs and sliders are the dials
    and each probe carries its manipulator's bias.

    A manipulator's fitted error is where its probe really is against where the
    calibration says, which one config cannot draw, so it is kept out of the
    plan and reported as the extra target move to enter at the rig."""
    from aind_rutter.config import ConfigModel
    from aind_rutter.plan_io.locate_implant import ImplantFit, located_config
    from aind_rutter.plan_io.plan_check import (
        dialed_plan,
        probe_gaps,
        rethreaded_plan,
    )
    from aind_rutter.plan_io.roundtrip import replace_top_level_block

    if not located:
        raise ValueError("--configs needs --fit")
    assert args.configs is not None
    args.configs.mkdir(parents=True, exist_ok=True)
    written = args.config.read_text()
    out: dict[str, Any] = {}
    for subject, (shift, with_errors) in located.items():
        plan = rethreaded_plan(plan_raw, checks[subject])
        if sheet is not None:
            plan = dialed_plan(plan, sheet)
        text = located_config(written, ImplantFit(np.eye(3), shift, False), "implant")
        text = replace_top_level_block(text, "plan", plan)
        stem = f"{args.plan.name.removesuffix('.plan.yml')}-{subject}"
        config_path = args.configs / f"{stem}.yml"
        plan_path = args.configs / f"{stem}.plan.yml"
        config_path.write_text(text)
        plan_path.write_text(yaml.safe_dump(plan, sort_keys=False))
        gaps = probe_gaps(ConfigModel.from_yaml(config_path, locator), plan)
        hits = [g for g in gaps if g.distance_mm <= 0]
        print(f"{subject}: wrote {config_path} and {plan_path}")
        for g in gaps[:8]:
            print(f"  {g.distance_mm:+7.3f} mm  {g.id1} – {g.id2}")
        out[subject] = {
            "config": str(config_path),
            "plan": str(plan_path),
            "offsets_RA": {n: p["offsets_RA"] for n, p in plan["probes"].items()},
            "collisions": [f"{g.id1} – {g.id2}" for g in hits],
            "closest_gaps_mm": {
                f"{g.id1} – {g.id2}": round(g.distance_mm, 3) for g in gaps[:8]
            },
        }
        if with_errors is not None:
            out[subject]["rig_extra_offset_RA_mm"] = {
                a.probe: [round(x - y, 3) for x, y in zip(b.rethread_RA, a.rethread_RA)]
                for a, b in zip(checks[subject], checks[with_errors])
            }
            print("  extra target move at the rig for each manipulator's error (R, A):")
            for p, v in out[subject]["rig_extra_offset_RA_mm"].items():
                print(f"    {p:<12} {v[0]:+.2f}, {v[1]:+.2f}")
        if sheet is not None:
            out[subject]["manipulators"] = {
                p: d["manipulator"] for p, d in sheet.probes.items()
            }
    return out


def main() -> None:
    cmd = tyro.extras.subcommand_cli_from_dict(
        {"locate": Locate, "bias": Bias, "check": Check}, prog="rutter-tracks"
    )
    try:
        if isinstance(cmd, Locate):
            locate(cmd)
        elif isinstance(cmd, Bias):
            bias(cmd)
        else:
            check(cmd)
    except (OSError, ValueError, LookupError) as e:
        print(f"rutter-tracks: {e}", file=sys.stderr)
        sys.exit(1)
