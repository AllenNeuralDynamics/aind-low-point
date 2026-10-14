"""Add probe calibrations to a config (``rutter-calibrate``).

Appends calibration sources, in the order given, to the config's stacked
sources: parallax directories and manual-calibration workbooks (.xlsx), merged
per probe code with the last source winning. Declares the reticle the parallax
directories were taken against (a workbook states its own), maps probes to
probe codes and marks those probes calibrated. Only the ``plan:`` block is
rewritten; comments, ``paths:`` and every ``${…}`` reference elsewhere
survive. Each source is written through the config's own ``paths:`` names
where one contains it, so the result runs on any machine that locates them.

Examples
--------
Two parallax runs, reticle from a workbook, codes matched to a plan's angles::

    rutter-calibrate config.yml log_20261012_162041 log_20261013_141319 \\
        --reticle-from calibration_info_sws2_2026_07_03T08_08_00.xlsx \\
        --match-plan plans/plan-06-....plan.yml --out config-cal.yml

A manual calibration on top, codes given::

    rutter-calibrate config.yml calibration_info_sws2_2026_07_03T08_08_00.xlsx \\
        --probe CLA=50213 --probe VM=46810 --in-place
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Annotated, Any, Mapping

import numpy as np
import tyro
import yaml

from aind_rutter.config.locations import Locator, load_resolved, parse_overrides


@dataclass(frozen=True)
class Reticle:
    name: str
    offset_RAS: tuple[float, float, float]
    rotation_z: float

    def as_yaml(self) -> dict[str, Any]:
        return {"offset_RAS": list(self.offset_RAS), "rotation_z": self.rotation_z}


def reticle_from_workbook(path: Path) -> Reticle:
    """The reticle a manual-calibration workbook states in its metadata."""
    from aind_mri_utils.reticle_calibrations import read_manual_reticle_calibration

    _, offset, rotation, name = read_manual_reticle_calibration(str(path))
    return Reticle(
        name=str(name),
        offset_RAS=tuple(round(float(v), 9) for v in offset),  # type: ignore[arg-type]
        rotation_z=float(rotation),
    )


def fitted_angles(
    source: Path, reticle: Reticle | None
) -> dict[str, tuple[float, float]]:
    """Probe code → the (AP, ML) a source's fit places the probe at (deg): a
    parallax directory, read against ``reticle``, or a manual workbook."""
    from aind_mri_utils.reticle_calibrations import (
        find_probe_angle,
        fit_rotation_params_from_manual_calibration,
        fit_rotation_params_from_parallax,
    )

    if source.is_dir():
        if reticle is None:
            raise ValueError(f"{source} is a parallax directory and needs a reticle")
        bank = fit_rotation_params_from_parallax(
            str(source), np.asarray(reticle.offset_RAS, float), reticle.rotation_z
        )[0]
    else:
        bank = fit_rotation_params_from_manual_calibration(str(source))[0]
    out: dict[str, tuple[float, float]] = {}
    for code, (R, _t) in sorted(bank.items()):
        ap, ml = find_probe_angle(R)
        out[str(code)] = (float(ap), float(ml))
    return out


def planned_angles(plan: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """Probe → the subject (AP, ML) a plan sets it to: its arc's AP when bound."""
    arcs = plan.get("arcs") or {}
    out: dict[str, tuple[float, float]] = {}
    for name, decl in (plan.get("probes") or {}).items():
        arc = decl.get("arc")
        if arc is not None and decl.get("bind_ap_to_arc", True) and arc in arcs:
            ap = float(arcs[arc])
        else:
            ap = float(decl.get("ap_local") or 0.0)
        out[name] = (ap, float(decl.get("slider_ml") or 0.0))
    return out


def match_codes(
    fitted: Mapping[str, tuple[float, float]],
    planned: Mapping[str, tuple[float, float]],
) -> dict[str, str]:
    """Probe → code, pairing each planned probe with a fitted code so the
    summed (AP, ML) distance is least. Sound only when the probes sat at
    roughly the planned angles while they were calibrated."""
    from scipy.optimize import linear_sum_assignment

    probes, codes = list(planned), list(fitted)
    if not probes or not codes:
        return {}
    cost = np.array([[math.dist(planned[p], fitted[c]) for c in codes] for p in probes])
    rows, cols = linear_sum_assignment(cost)
    return {probes[r]: codes[c] for r, c in zip(rows, cols)}


def written_path(source: Path, located: Mapping[str, Any]) -> str:
    """``source`` as the config should write it: through the declared
    ``paths:`` name whose location most narrowly contains it, else absolute.
    A name located at the source itself is taken only when none contains it,
    since a calibration run is one among many under its root."""
    target = PurePath(source.resolve())
    best: tuple[tuple[bool, int], str] | None = None
    for name, value in located.items():
        if not isinstance(value, str):
            continue
        try:
            rel = target.relative_to(PurePath(Path(value).resolve()))
        except ValueError:
            continue
        depth = (not rel.parts, len(rel.parts))
        if best is None or depth < best[0]:
            ref = "${paths." + name + "}"
            best = (depth, ref if not rel.parts else f"{ref}/{rel.as_posix()}")
    return best[1] if best is not None else target.as_posix()


def source_entry(written: str, is_directory: bool, reticle: str | None) -> dict:
    """A stacked-source entry: a directory names its reticle, a file does not."""
    if not is_directory:
        return {"file": written}
    if reticle is None:
        raise ValueError(f"{written} is a parallax directory and needs a reticle")
    return {"directory": written, "reticle": reticle}


def add_calibration(
    plan: dict[str, Any],
    sources: list[dict[str, str]],
    reticle: Reticle | None,
    reticle_name: str | None,
    probe_to_code: Mapping[str, str],
) -> dict[str, Any]:
    """``plan`` with ``sources`` appended, in order, to its stacked sources,
    ``reticle`` declared, ``probe_to_code`` merged in and each mapped probe
    calibrated. A source already listed moves to the end, where it wins."""
    plan = dict(plan)
    cals = dict(plan.get("calibrations") or {})
    if cals.get("files") or cals.get("probe_to_ref"):
        raise ValueError(
            "plan.calibrations uses files/probe_to_ref; this tool adds to the "
            "stacked form (sources/probe_to_code)"
        )
    reticles = dict(plan.get("reticles") or {})
    if reticle is not None and reticle_name is not None:
        prior = reticles.get(reticle_name)
        if prior is not None and _reticle_differs(prior, reticle):
            raise ValueError(
                f"reticle {reticle_name!r} is already declared with other "
                f"values ({prior}); name the new one differently"
            )
        reticles[reticle_name] = reticle.as_yaml()
    elif reticle_name is not None and reticle_name not in reticles:
        raise ValueError(
            f"reticle {reticle_name!r} is not declared in plan.reticles; give "
            "--reticle-from or --reticle-offset"
        )
    unknown = sorted(set(probe_to_code) - set(plan.get("probes") or {}))
    if unknown:
        raise ValueError(f"no such probe in plan.probes: {', '.join(unknown)}")

    def where(entry: Mapping[str, Any]) -> Any:
        return entry.get("directory") or entry.get("file")

    added = {where(e) for e in sources}
    stacked = [e for e in (cals.get("sources") or []) if where(e) not in added]
    stacked.extend(sources)
    cals.pop("files", None)
    cals.pop("probe_to_ref", None)
    cals.update(
        sources=stacked,
        probe_to_code={
            **(cals.get("probe_to_code") or {}),
            **{p: str(c) for p, c in probe_to_code.items()},
        },
    )
    probes = {name: dict(decl) for name, decl in plan["probes"].items()}
    for name in probe_to_code:
        probes[name]["calibrated"] = True
    plan.update(reticles=reticles, calibrations=cals, probes=probes)
    return plan


def _reticle_differs(prior: Mapping[str, Any], reticle: Reticle) -> bool:
    offset = [float(v) for v in prior.get("offset_RAS") or []]
    return not (
        np.allclose(offset, reticle.offset_RAS)
        and math.isclose(float(prior.get("rotation_z", 0.0)), reticle.rotation_z)
    )


def write_config(
    config: Path, plan: Mapping[str, Any], out: Path, locator: Locator
) -> None:
    """Write ``config`` with its plan block replaced to ``out``, refusing to
    if the result does not load and resolve every calibration."""
    from aind_rutter.build.calibration import load_calibrations
    from aind_rutter.config import ConfigModel
    from aind_rutter.plan_io.roundtrip import replace_top_level_block

    text = replace_top_level_block(config.read_text(), "plan", dict(plan))
    # Beside ``out``, so relative paths in the config resolve as they will.
    staged = out.with_name(f".{out.name}.staged")
    staged.write_text(text)
    try:
        cfg = ConfigModel.from_yaml(staged, locator)
        load_calibrations(cfg.plan.calibrations, cfg.plan.reticles)
    except Exception:
        staged.unlink()
        raise
    staged.replace(out)


@dataclass
class Args:
    """Command-line arguments for ``rutter-calibrate``."""

    config: tyro.conf.Positional[Path]
    """Config YAML to add the calibration to."""

    sources: tyro.conf.Positional[tuple[Path, ...]]
    """Calibration sources, oldest first: parallax directories
    (``points_SN*.csv``, ``transM_SN*.csv``) and manual workbooks (.xlsx)."""

    out: Path | None = None
    """Where to write the calibrated config."""

    in_place: bool = False
    """Overwrite ``config`` instead of writing ``--out``."""

    reticle: str | None = None
    """Reticle the parallax directories were taken against; with neither
    ``--reticle-from`` nor ``--reticle-offset`` it must already be declared in
    the config."""

    reticle_from: Path | None = None
    """Manual-calibration workbook (.xlsx) whose metadata states the reticle's
    name, offset and rotation."""

    reticle_offset: tuple[float, float, float] | None = None
    """Reticle offset, RAS mm, to declare it with."""

    reticle_rotation: float = 0.0
    """Reticle rotation about z (deg), with ``--reticle-offset``."""

    probe: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """``PROBE=CODE`` for one probe; repeatable. Wins over ``--match-plan``."""

    match_plan: Path | None = None
    """Plan YAML whose angles the remaining probes' codes are matched to."""

    paths: Annotated[tuple[Path, ...], tyro.conf.UseAppendAction] = ()
    """Paths file locating the config's ``paths:`` names; repeatable."""

    path: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """``NAME=DIR`` for one ``paths:`` name; repeatable."""

    offline: bool = False
    """Fetch nothing while checking the result."""


def main() -> None:
    args = tyro.cli(Args)
    try:
        run(args)
    except (OSError, ValueError, LookupError, KeyError) as e:
        print(f"rutter-calibrate: {e}", file=sys.stderr)
        sys.exit(1)


def run(args: Args) -> None:
    if args.in_place == (args.out is not None):
        raise ValueError("give exactly one of --out or --in-place")
    out = args.config if args.in_place else args.out
    assert out is not None
    if not args.sources:
        raise ValueError("give at least one calibration directory or workbook")
    for source in args.sources:
        if not (source.is_dir() or source.suffix.lower() == ".xlsx"):
            raise ValueError(f"not a parallax directory or .xlsx workbook: {source}")
    locator = Locator.from_environment(
        args.paths, parse_overrides(args.path), offline=args.offline
    )
    raw = yaml.safe_load(args.config.read_text()) or {}
    plan = raw.get("plan") or {}

    declare, reticle = None, None
    if any(source.is_dir() for source in args.sources):
        declare = _reticle(args, plan)
        reticle = declare or _declared(plan, args.reticle)
    elif args.reticle or args.reticle_from or args.reticle_offset:
        raise ValueError("a reticle applies to parallax directories, and none is given")

    # Code → (AP, ML) and the source it comes from; later sources win.
    fitted: dict[str, tuple[float, float]] = {}
    origin: dict[str, str] = {}
    for source in args.sources:
        for code, angles in fitted_angles(source, reticle).items():
            fitted[code], origin[code] = angles, source.name
    given = dict(_pairs(args.probe))
    missing = sorted(set(given.values()) - set(fitted))
    if missing:
        raise ValueError(
            f"codes in no source given: {', '.join(missing)} "
            f"(they hold {', '.join(fitted)})"
        )
    planned: dict[str, tuple[float, float]] = {}
    mapping = dict(given)
    if args.match_plan is not None:
        planned = planned_angles(yaml.safe_load(args.match_plan.read_text()))
        free = {c: a for c, a in fitted.items() if c not in given.values()}
        rest = {p: a for p, a in planned.items() if p not in given}
        mapping.update(match_codes(free, rest))
    if not mapping:
        raise ValueError(
            f"no probe mapped; give --probe PROBE=CODE or --match-plan "
            f"(codes in these sources: {', '.join(fitted)})"
        )

    located = load_resolved(args.config, locator).get("paths") or {}
    name = reticle.name if reticle is not None else None
    entries = [
        source_entry(written_path(source, located), source.is_dir(), name)
        for source in args.sources
    ]
    new_plan = add_calibration(plan, entries, declare, name, mapping)
    write_config(args.config, new_plan, out, locator)
    _report(out, entries, mapping, fitted, origin, planned, given)


def _declared(plan: Mapping[str, Any], name: str | None) -> Reticle:
    assert name is not None
    declared = plan["reticles"][name]
    return Reticle(
        name,
        tuple(float(v) for v in declared["offset_RAS"]),  # type: ignore[arg-type]
        float(declared.get("rotation_z", 0.0)),
    )


def _reticle(args: Args, plan: Mapping[str, Any]) -> Reticle | None:
    if args.reticle_from is not None and args.reticle_offset is not None:
        raise ValueError("give --reticle-from or --reticle-offset, not both")
    if args.reticle_from is not None:
        found = reticle_from_workbook(args.reticle_from)
        return Reticle(args.reticle or found.name, found.offset_RAS, found.rotation_z)
    if args.reticle_offset is not None:
        if args.reticle is None:
            raise ValueError("--reticle-offset needs --reticle to name it")
        return Reticle(args.reticle, args.reticle_offset, args.reticle_rotation)
    if args.reticle is None:
        raise ValueError("give --reticle, --reticle-from or --reticle-offset")
    if args.reticle not in (plan.get("reticles") or {}):
        raise ValueError(
            f"reticle {args.reticle!r} is not declared in plan.reticles; give "
            "--reticle-from or --reticle-offset"
        )
    return None


def _pairs(items: tuple[str, ...]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        probe, sep, code = item.partition("=")
        if not sep or not probe.strip() or not code.strip():
            raise ValueError(f"--probe {item!r}: expected PROBE=CODE")
        out.append((probe.strip(), code.strip()))
    return out


def _report(
    out: Path,
    entries: list[dict[str, str]],
    mapping: Mapping[str, str],
    fitted: Mapping[str, tuple[float, float]],
    origin: Mapping[str, str],
    planned: Mapping[str, tuple[float, float]],
    given: Mapping[str, str],
) -> None:
    print(f"Wrote {out}")
    for entry in entries:
        where = entry.get("directory") or entry["file"]
        reticle = f" (reticle {entry['reticle']})" if "reticle" in entry else ""
        print(f"  source: {where}{reticle}")
    head = f"  {'probe':<8}{'code':<8}{'fit AP':>8}{'fit ML':>8}"
    if planned:
        head += f"{'plan AP':>9}{'plan ML':>9}{'off':>7}"
    print(head + "  mapped    fit from")
    for probe, code in mapping.items():
        ap, ml = fitted[code]
        row = f"  {probe:<8}{code:<8}{ap:8.1f}{ml:8.1f}"
        if probe in planned:
            pap, pml = planned[probe]
            row += f"{pap:9.1f}{pml:9.1f}{math.dist((ap, ml), (pap, pml)):7.1f}"
        elif planned:
            row += " " * 25
        how = "--probe" if probe in given else "matched"
        print(f"{row}  {how:<9} {origin[code]}")
    unused = [c for c in fitted if c not in mapping.values()]
    if unused:
        print(f"  codes left unmapped: {', '.join(unused)}")
    if any(p not in given for p in mapping):
        print(
            "  Matched codes assume the probes sat near the plan's angles while "
            "calibrated; check them against the rig."
        )


if __name__ == "__main__":
    main()
