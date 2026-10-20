"""Locate an implant from calibrated-probe readings of its holes
(``rutter-locate-implant``).

Each reading is a NewScale tip position of a calibrated probe at one hole:
``in`` the hole, where the tip lies on the bore's axis within its channel, or
``pointing`` at it, where the probe's line passes through the channel's middle.
The readings fit a rigid move of the implant from where the config puts it: a
translation, and each rotation the readings' geometry fixes well enough. What
they leave undetermined is reported and held at the config's placement: one
probe's readings all look along one line, so the shift along it is free until
a second probe at another angle reads a hole. The move is written as the
config's ``implant_to_lps``, after any steps it already has, so the bores and
the implant follow it and the brain stays where it was.

Examples
--------
One reading in hole 2, NewScale microns, reported only::

    rutter-locate-implant config-cal.yml --holes 0283-300-06.holes.yml \\
        --in-hole 50720:2:5964.5,3006.5,12402.5

Two probes aimed at the same holes, written to a new config::

    rutter-locate-implant config-cal.yml --holes 0283-300-06.holes.yml \\
        --pointing-at 46810:2:... --pointing-at 50213:2:... \\
        --pointing-at 46810:12:... --pointing-at 50213:12:... --out config-located.yml
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import tyro
import yaml
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aind_rutter.config.locations import Locator, parse_overrides

Mode = Literal["in", "pointing"]
# Holds a direction no reading constrains at the config's placement.
PRIOR_WEIGHT = 1e-3
# A rotation the readings fix more loosely than this is left unfitted.
MAX_ROTATION_SE_DEG = 1.5
# A translation direction seen more weakly than this (readings moving under a
# fifth as far as the implant) counts as undetermined.
MIN_SHIFT_GAIN = 0.2
# Rotations looser than this are reported as unseen rather than by their error.
UNSEEN_ROTATION_DEG = 45.0
TRANSFORM_KEY = "implant_to_lps"


@dataclass(frozen=True)
class HoleReading:
    """One NewScale tip position, in mm, of probe ``probe`` (a calibration code
    or a planned probe's name) at hole ``hole``."""

    probe: str
    hole: int
    newscale_mm: tuple[float, float, float]
    mode: Mode


def parse_reading(text: str, mode: Mode, scale_to_mm: float) -> HoleReading:
    """``PROBE:HOLE:X,Y,Z`` as a reading, NewScale coordinates scaled to mm."""
    try:
        probe, hole, xyz = text.split(":")
        x, y, z = (float(v) * scale_to_mm for v in xyz.split(","))
        return HoleReading(probe.strip(), int(hole), (x, y, z), mode)
    except ValueError:
        raise ValueError(f"not PROBE:HOLE:X,Y,Z: {text!r}") from None


@dataclass(frozen=True)
class Bore:
    """A hole as the fit uses it: its axis, channel top, and the channel's
    extent along the axis from the top (``bottom_s`` is negative)."""

    id: int
    axis: NDArray[np.floating]
    top: NDArray[np.floating]
    bottom_s: float

    @property
    def middle(self) -> NDArray[np.floating]:
        return self.top + 0.5 * self.bottom_s * self.axis

    @classmethod
    def from_hole(cls, hole: Any) -> "Bore":
        axis = np.asarray(hole.axis, float)
        axis = axis / np.linalg.norm(axis)
        centres = np.array([np.asarray(s.center, float) for s in hole.sections])
        s = (centres - centres[0]) @ axis
        top = centres[int(np.argmax(s))]
        return cls(int(hole.id), axis, top, float(s.min() - s.max()))

    def moved(self, R: NDArray[np.floating], t: NDArray[np.floating]) -> "Bore":
        return Bore(self.id, R @ self.axis, R @ self.top + t, self.bottom_s)


@dataclass(frozen=True)
class Placed:
    """A reading in subject LPS: the tip and the probe's insertion direction."""

    reading: HoleReading
    code: str
    tip: NDArray[np.floating]
    direction: NDArray[np.floating]


def place_readings(
    readings: Sequence[HoleReading],
    bank: Mapping[str, Any],
    probe_to_code: Mapping[str, str],
) -> list[Placed]:
    """Each reading's tip and insertion direction in subject LPS, through the
    calibration of its probe's code."""
    from aind_mri_utils.reticle_calibrations import find_probe_insertion_vector

    from aind_rutter.build.calibration import newscale_to_lps

    out = []
    for r in readings:
        code = r.probe if r.probe in bank else probe_to_code.get(r.probe)
        if code is None or code not in bank:
            raise ValueError(
                f"{r.probe} is neither a calibrated code nor a mapped probe "
                f"(codes: {', '.join(sorted(bank))})"
            )
        cal = bank[code]
        down = find_probe_insertion_vector(cal.rotation) * np.array([-1.0, -1.0, 1.0])
        out.append(
            Placed(
                r,
                code,
                newscale_to_lps(np.asarray(r.newscale_mm), cal),
                down / np.linalg.norm(down),
            )
        )
    return out


def misfit(p: Placed, bore: Bore) -> NDArray[np.floating]:
    """How far a reading is from agreeing with ``bore``, mm: the tip's distance
    off the axis and outside the channel (``in``), or the channel middle's
    distance off the probe's line and behind the tip (``pointing``)."""
    if p.reading.mode == "in":
        d = p.tip - bore.top
        s = float(d @ bore.axis)
        outside = max(0.0, s) + max(0.0, bore.bottom_s - s)
        return np.append(d - s * bore.axis, outside)
    v = bore.middle - p.tip
    ahead = float(v @ p.direction)
    return np.append(v - ahead * p.direction, max(0.0, -ahead))


@dataclass(frozen=True)
class ImplantFit:
    """The move ``x → R x + t`` of the implant from its configured placement."""

    rotation: NDArray[np.floating]
    translation: NDArray[np.floating]
    fit_rotation: bool

    def apply(self, pts: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.asarray(pts) @ self.rotation.T + self.translation

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(Rotation.from_matrix(self.rotation).magnitude()))

    def as_ops(self) -> list[dict[str, Any]]:
        """The move as config transform steps, applied in order."""
        ops: list[dict[str, Any]] = []
        if self.fit_rotation:
            angles = Rotation.from_matrix(self.rotation).as_euler("xyz", degrees=True)
            ops.append(
                {
                    "kind": "rotate_euler_deg",
                    "order": "xyz",
                    "angles_deg": [round(float(a), 6) for a in angles],
                }
            )
        ops.append(
            {
                "kind": "translate_mm",
                "delta": [round(float(v), 6) for v in self.translation],
            }
        )
        return ops


def _frame(
    placed: Sequence[Placed], bores: Mapping[int, Bore]
) -> tuple[list[Bore], NDArray[np.floating], float]:
    """The bores read, the pivot rotations turn about (their middles' mean),
    and the middles' RMS spread, which turns an angle into mm at the holes."""
    used = [bores[p.reading.hole] for p in placed]
    middles = np.array([b.middle for b in used])
    pivot = middles.mean(axis=0)
    spread = float(np.sqrt(((middles - pivot) ** 2).sum(axis=1).mean()))
    return used, pivot, max(spread, 1.0)


def _move(
    shift: NDArray[np.floating],
    rotvec: NDArray[np.floating],
    pivot: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return R, pivot + shift - R @ pivot


@dataclass(frozen=True)
class Observability:
    """What a set of readings can determine about the implant's move.

    ``free_shifts`` are translation directions no reading sees; ``axes`` are
    the rotation axes, best seen first, and ``rotation_se_deg`` each one's
    standard error once translation is fitted, at the assumed aim error."""

    pivot: NDArray[np.floating]
    spread: float
    jacobian: NDArray[np.floating]
    free_shifts: list[NDArray[np.floating]]
    axes: NDArray[np.floating]
    rotation_se_deg: NDArray[np.floating]

    def fitted_axes(self, policy: Literal["auto", "on", "off"]) -> NDArray:
        if policy == "on":
            return self.axes
        if policy == "off":
            return self.axes[:0]
        return self.axes[self.rotation_se_deg <= MAX_ROTATION_SE_DEG]


def observe(
    placed: Sequence[Placed], bores: Mapping[int, Bore], aim_error_mm: float
) -> Observability:
    """How well the readings' geometry fixes each part of the move, from the
    misfits' sensitivity at the config's placement. The channel-depth bound
    of an ``in`` reading is left out: it bounds a move, it does not fix one."""
    used, pivot, spread = _frame(placed, bores)

    def rows(x: NDArray[np.floating]) -> NDArray[np.floating]:
        R, t = _move(x[:3], x[3:] / spread, pivot)
        return np.concatenate(
            [misfit(p, b.moved(R, t))[:3] for p, b in zip(placed, used)]
        )

    h = 1e-6
    J = np.array([(rows(h * e) - rows(-h * e)) / (2 * h) for e in np.eye(6)]).T
    Jt, Jr = J[:, :3], J[:, 3:]
    _, gain, shifts = np.linalg.svd(Jt)
    free = [shifts[i] for i in range(3) if gain[i] < MIN_SHIFT_GAIN]
    beside_shift = np.eye(len(J)) - Jt @ np.linalg.pinv(Jt)
    _, turn_gain, axes = np.linalg.svd(beside_shift @ Jr)
    se = np.degrees(aim_error_mm / np.maximum(turn_gain, 1e-12) / spread)
    return Observability(pivot, spread, J, free, axes, se)


def fit_implant(
    placed: Sequence[Placed],
    bores: Mapping[int, Bore],
    rotation_axes: NDArray[np.floating] | None = None,
) -> ImplantFit:
    """The rigid move of the implant that best agrees with every reading: a
    translation, plus rotations about ``rotation_axes`` (rows) turning about
    the read holes' middle so the two parts stay separate. Directions no
    reading constrains stay at the config's placement."""
    used, pivot, spread = _frame(placed, bores)
    axes = np.zeros((0, 3)) if rotation_axes is None else np.asarray(rotation_axes)

    def move(x: NDArray[np.floating]) -> tuple[NDArray, NDArray]:
        return _move(x[:3], x[3:] @ axes / spread, pivot)

    def residuals(x: NDArray[np.floating]) -> NDArray[np.floating]:
        R, t = move(x)
        rows = [misfit(p, b.moved(R, t)) for p, b in zip(placed, used)]
        return np.concatenate([*rows, PRIOR_WEIGHT * x])

    solved = least_squares(residuals, np.zeros(3 + len(axes)), xtol=1e-12, ftol=1e-12)
    R, t = move(solved.x)
    return ImplantFit(R, t, len(axes) > 0)


def hole_se_mm(
    obs: Observability,
    rotation_axes: NDArray[np.floating],
    middle: NDArray,
    aim_error_mm: float,
) -> float:
    """Standard error, mm, of a hole's fitted position along its worst
    determined direction; a free shift is not in it."""
    Jp = np.hstack([obs.jacobian[:, :3], obs.jacobian[:, 3:] @ rotation_axes.T])
    cov = aim_error_mm**2 * np.linalg.pinv(Jp.T @ Jp)
    arm = middle - obs.pivot
    cross = np.array([[0, arm[2], -arm[1]], [-arm[2], 0, arm[0]], [arm[1], -arm[0], 0]])
    H = np.hstack([np.eye(3), cross @ rotation_axes.T / obs.spread])
    return float(np.sqrt(np.linalg.eigvalsh(H @ cov @ H.T).max()))


def line_gaps(placed: Sequence[Placed]) -> list[tuple[Placed, Placed, float, float]]:
    """Each pair of readings of one hole by two probes, with the angle
    between the probes and how far apart their lines pass (an ``in`` tip
    counts as a point), which at a shared hole should be nothing whatever
    the implant's placement: it is calibration plus aiming error."""
    out = []
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            if a.reading.hole != b.reading.hole or a.code == b.code:
                continue
            if a.reading.mode == "in" and b.reading.mode == "in":
                continue
            if a.reading.mode == "in":
                a, b = b, a
            angle = float(
                np.degrees(np.arccos(np.clip(abs(a.direction @ b.direction), 0, 1)))
            )
            v = b.tip - a.tip
            if b.reading.mode == "in":
                gap = float(np.linalg.norm(v - (v @ a.direction) * a.direction))
            else:
                n = np.cross(a.direction, b.direction)
                if np.linalg.norm(n) < 1e-6:
                    continue
                gap = float(abs(v @ n) / np.linalg.norm(n))
            out.append((a, b, angle, gap))
    return out


def configured_bores(holes_path: Path, cfg: Any) -> dict[int, Bore]:
    """The holes where the config's ``implant_to_lps`` puts them."""
    from aind_rutter.build.transforms import compile_all_transforms
    from aind_rutter.optimization.geometry.holes import load_holes

    bores = {h.id: Bore.from_hole(h) for h in load_holes(holes_path)}
    compiled = compile_all_transforms(cfg.transforms or {})
    if TRANSFORM_KEY not in compiled:
        return bores
    R, t = compiled[TRANSFORM_KEY].rotate_translate
    return {k: b.moved(np.asarray(R), np.asarray(t)) for k, b in bores.items()}


def located_config(written: str, fit: ImplantFit, implant: str) -> str:
    """``written`` with ``fit`` appended to ``implant_to_lps`` and the implant
    asset placed by it; every other block kept as written."""
    from aind_rutter.plan_io.roundtrip import replace_top_level_block

    raw = yaml.safe_load(written) or {}
    transforms = dict(raw.get("transforms") or {})
    prior = transforms.get(TRANSFORM_KEY) or {}
    steps = prior.get("sequence", [prior] if "kind" in prior else [])
    transforms[TRANSFORM_KEY] = {"sequence": [*steps, *fit.as_ops()]}
    text = replace_top_level_block(written, "transforms", transforms)
    found = [a for a in raw.get("assets") or [] if a.get("key") == implant]
    if not found:
        raise ValueError(f"no asset {implant!r}; give --implant-asset")
    placed_by = found[0].get("transform")
    if placed_by is None:
        text = _place_asset(text, implant)
    elif placed_by != TRANSFORM_KEY:
        raise ValueError(
            f"asset {implant!r} is placed by {placed_by!r}, not {TRANSFORM_KEY}"
        )
    return text


def _place_asset(written: str, key: str) -> str:
    """``written`` with ``transform: implant_to_lps`` added to asset ``key``,
    one line inserted so the assets list keeps its layout."""
    lines = written.splitlines(keepends=True)
    pattern = re.compile(rf"^(\s*)- key:\s*['\"]?{re.escape(key)}['\"]?\s*$")
    at = next((i for i, line in enumerate(lines) if pattern.match(line)), None)
    if at is None:
        raise ValueError(f"asset {key!r} is not written as '- key: {key}'")
    indent = pattern.match(lines[at]).group(1) + "  "  # type: ignore[union-attr]
    lines.insert(at + 1, f"{indent}transform: {TRANSFORM_KEY}\n")
    out = "".join(lines)
    expected = yaml.safe_load(written)
    next(a for a in expected["assets"] if a.get("key") == key)["transform"] = (
        TRANSFORM_KEY
    )
    if yaml.safe_load(out) != expected:
        raise ValueError(f"could not place asset {key!r} by {TRANSFORM_KEY}")
    return out


@dataclass
class Args:
    """Command-line arguments for ``rutter-locate-implant``."""

    config: tyro.conf.Positional[Path]
    """Calibrated config whose implant the readings locate."""

    holes: Path
    """Bores file of the config's implant, in the implant's own frame."""

    in_hole: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """``PROBE:HOLE:X,Y,Z`` for a tip inside a hole; repeatable. PROBE is a
    calibration code or a planned probe's name."""

    pointing_at: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """``PROBE:HOLE:X,Y,Z`` for a tip above a hole, the probe aimed at its
    middle; repeatable."""

    units: Literal["um", "mm"] = "um"
    """Units of the NewScale coordinates."""

    rotation: Literal["auto", "on", "off"] = "auto"
    """Fit rotations: ``auto`` fits those the readings fix to within
    1.5 deg at the aim error, ``on`` all three, ``off`` none."""

    aim_error_um: float = 50.0
    """How far a reading may be off its hole, the scale of every
    uncertainty reported."""

    out: Path | None = None
    """Where to write the located config; without it, or ``--in-place``, the
    fit is only reported."""

    in_place: bool = False
    """Overwrite ``config`` with the located config."""

    implant_asset: str = "implant"
    """Key of the implant's asset, which ``implant_to_lps`` places."""

    paths: Annotated[tuple[Path, ...], tyro.conf.UseAppendAction] = ()
    """Paths file locating the config's ``paths:`` names; repeatable."""

    path: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """``NAME=DIR`` for one ``paths:`` name; repeatable."""

    offline: bool = False
    """Fetch nothing while loading the config."""


def main() -> None:
    args = tyro.cli(Args)
    try:
        run(args)
    except (OSError, ValueError, LookupError, KeyError) as e:
        print(f"rutter-locate-implant: {e}", file=sys.stderr)
        sys.exit(1)


def run(args: Args) -> None:
    from aind_rutter.build.calibration import load_calibration_bank
    from aind_rutter.config import ConfigModel

    if args.in_place and args.out is not None:
        raise ValueError("give --out or --in-place, not both")
    scale = 1e-3 if args.units == "um" else 1.0
    readings = [parse_reading(t, "in", scale) for t in args.in_hole] + [
        parse_reading(t, "pointing", scale) for t in args.pointing_at
    ]
    if not readings:
        raise ValueError("give at least one --in-hole or --pointing-at reading")
    locator = Locator.from_environment(
        args.paths, parse_overrides(args.path), offline=args.offline
    )
    cfg = ConfigModel.from_yaml(args.config, locator)
    bank = load_calibration_bank(cfg.plan.calibrations, cfg.plan.reticles)
    placed = place_readings(readings, bank, cfg.plan.calibrations.probe_to_code)
    bores = configured_bores(args.holes, cfg)
    unknown = sorted({r.hole for r in readings} - set(bores))
    if unknown:
        raise ValueError(f"no hole {unknown} in {args.holes}")
    aim = args.aim_error_um * 1e-3
    obs = observe(placed, bores, aim)
    axes = obs.fitted_axes(args.rotation)
    fit = fit_implant(placed, bores, axes)
    report(placed, bores, fit, obs, axes, aim, sys.stdout)

    out = args.config if args.in_place else args.out
    if out is None:
        return
    text = located_config(args.config.read_text(), fit, args.implant_asset)
    staged = out.with_name(f".{out.name}.staged")
    staged.write_text(text)
    try:
        ConfigModel.from_yaml(staged, locator)
    except Exception:
        staged.unlink()
        raise
    staged.replace(out)
    print(f"Wrote {out} ({TRANSFORM_KEY} moves the implant and its bores)")


def _anatomical(v_lps: NDArray[np.floating]) -> str:
    """An LPS displacement in words: anterior, right and dorsal positive."""
    x, y, z = v_lps
    parts = [
        f"{abs(y):.3f} {'anterior' if y < 0 else 'posterior'}",
        f"{abs(x):.3f} {'right' if x < 0 else 'left'}",
        f"{abs(z):.3f} {'dorsal' if z > 0 else 'ventral'}",
    ]
    return ", ".join(parts)


def _line(u_lps: NDArray[np.floating]) -> str:
    """A direction without sign, as RAS components, superior made positive."""
    ras = np.asarray(u_lps) * np.array([-1.0, -1.0, 1.0])
    big = int(np.argmax(np.abs(ras)))
    ras = ras * (
        1.0 if ras[2] > 1e-9 or (abs(ras[2]) <= 1e-9 and ras[big] > 0) else -1.0
    )
    return "(" + ", ".join(f"{n} {v:+.2f}" for n, v in zip("RAS", ras)) + ")"


def _along(
    u: NDArray[np.floating], v: NDArray[np.floating], within_deg: float = 10.0
) -> bool:
    return abs(float(u @ v)) >= np.cos(np.radians(within_deg))


def report(
    placed: Sequence[Placed],
    bores: Mapping[int, Bore],
    fit: ImplantFit,
    obs: Observability,
    rotation_axes: NDArray[np.floating],
    aim_error_mm: float,
    stream: Any,
) -> None:
    lines = [
        *_readings_lines(placed, bores, fit),
        *_gap_lines(placed),
        *_move_lines(placed, bores, fit, obs, rotation_axes, aim_error_mm),
        *_undetermined_lines(placed, bores, fit, obs, rotation_axes),
    ]
    print("\n".join(lines), file=stream)


def _readings_lines(
    placed: Sequence[Placed], bores: Mapping[int, Bore], fit: ImplantFit
) -> list[str]:
    out = [
        "Readings (mm, subject RAS; misfit is how far off the hole agrees):",
        f"  {'probe':<8}{'code':<8}{'hole':>5}  {'mode':<9}"
        f"{'tip R':>8}{'tip A':>8}{'tip S':>8}{'misfit':>9}{'fitted':>8}",
    ]
    for p in placed:
        bore = bores[p.reading.hole]
        before = float(np.linalg.norm(misfit(p, bore)))
        after = float(
            np.linalg.norm(misfit(p, bore.moved(fit.rotation, fit.translation)))
        )
        r, a, s = p.tip * np.array([-1.0, -1.0, 1.0])
        out.append(
            f"  {p.reading.probe:<8}{p.code:<8}{p.reading.hole:>5}  "
            f"{p.reading.mode:<9}{r:8.3f}{a:8.3f}{s:8.3f}{before:9.3f}{after:8.3f}"
        )
        if p.reading.mode == "in":
            depth = float((p.tip - bore.top) @ bore.axis)
            out.append(
                f"    tip {-depth:.3f} below the hole's channel top "
                f"(channel {-bore.bottom_s:.3f} deep)"
            )
    return out


def _gap_lines(placed: Sequence[Placed]) -> list[str]:
    gaps = line_gaps(placed)
    if not gaps:
        return []
    return [
        "",
        "Two probes at one hole (their lines should meet; the gap is",
        "calibration plus aiming error, whatever the implant's placement):",
        *(
            f"  hole {a.reading.hole:>2}: {a.code} and {b.code}, "
            f"{angle:.0f} deg apart, gap {gap:.3f} mm"
            for a, b, angle, gap in gaps
        ),
    ]


def _is_fitted(axis: NDArray[np.floating], rotation_axes: NDArray) -> bool:
    return any(_along(axis, f, 1.0) for f in rotation_axes)


def _move_lines(
    placed: Sequence[Placed],
    bores: Mapping[int, Bore],
    fit: ImplantFit,
    obs: Observability,
    rotation_axes: NDArray[np.floating],
    aim_error_mm: float,
) -> list[str]:
    n_turns = len(rotation_axes)
    kind = (
        "translation only"
        if n_turns == 0
        else f"translation and {n_turns} of 3 rotations"
    )
    out = [
        "",
        f"Implant move from the config's placement ({kind}; ± is one standard "
        f"error at an aim error of {aim_error_mm * 1e3:.0f} um):",
    ]
    for hid in sorted({p.reading.hole for p in placed}):
        m = bores[hid].middle
        shift = fit.apply(m) - m
        se = hole_se_mm(obs, rotation_axes, m, aim_error_mm)
        out.append(
            f"  hole {hid:>2}: {np.linalg.norm(shift):.3f} mm "
            f"({_anatomical(shift)}) ±{se:.3f}"
        )
    rotvec = Rotation.from_matrix(fit.rotation).as_rotvec()
    out += [
        f"  rotation about {_line(axis)}: "
        f"{np.degrees(rotvec @ axis):+.2f} ± {se_deg:.2f} deg"
        for axis, se_deg in zip(obs.axes, obs.rotation_se_deg)
        if _is_fitted(axis, rotation_axes)
    ]
    return out


def _undetermined_lines(
    placed: Sequence[Placed],
    bores: Mapping[int, Bore],
    fit: ImplantFit,
    obs: Observability,
    rotation_axes: NDArray[np.floating],
) -> list[str]:
    turns = [
        (a, se)
        for a, se in zip(obs.axes, obs.rotation_se_deg)
        if not _is_fitted(a, rotation_axes)
    ]
    if not obs.free_shifts and not turns:
        return []
    out = ["", "Not determined by these readings (held at the config's placement):"]
    for u in obs.free_shifts:
        out += _free_shift_lines(u, placed, bores, fit, obs)
    if turns and all(se > UNSEEN_ROTATION_DEG for _, se in turns):
        out.append("  rotation: no reading sees it")
    else:
        out += [
            f"  rotation about {_line(a)}: ± {se:.1f} deg from these readings"
            for a, se in turns
        ]
    out += ["", "To determine it:"]
    if obs.free_shifts:
        out.append(
            "  aim a second probe at one of these holes from at least 40 deg away"
        )
    if turns:
        out.append(
            "  read three holes spread across the implant"
            if len({p.reading.hole for p in placed}) < 3
            else "  read the same holes with a second probe at a different angle"
        )
    return out


def _free_shift_lines(
    u: NDArray[np.floating],
    placed: Sequence[Placed],
    bores: Mapping[int, Bore],
    fit: ImplantFit,
    obs: Observability,
) -> list[str]:
    bounded = sorted(
        {
            p.reading.hole
            for p in placed
            if p.reading.mode == "in" and _along(u, bores[p.reading.hole].axis)
        }
    )
    if bounded:
        return [
            f"  shift along {_line(u)}, the bore of hole "
            f"{', '.join(map(str, bounded))}: the channel depth bounds it, and the "
            f"fit takes the least move that puts the tip inside"
        ]
    sighted = sorted(
        {
            p.code
            for p in placed
            if p.reading.mode == "pointing" and _along(u, p.direction)
        }
    )
    where = f", the line of sight of {', '.join(sighted)}" if sighted else ""
    out = [f"  shift along {_line(u)}{where}: any amount fits as well"]
    if len(obs.free_shifts) == 1 and abs(u[2]) >= 0.5:
        rise = fit.apply(obs.pivot)[2] - obs.pivot[2]
        level = ImplantFit(
            fit.rotation, fit.translation - rise / u[2] * u, fit.fit_rotation
        )
        out.append(
            "    with no dorsal-ventral move at the holes read, the same fit is:"
        )
        for hid in sorted({p.reading.hole for p in placed}):
            m = bores[hid].middle
            out.append(f"      hole {hid:>2}: {_anatomical(level.apply(m) - m)}")
    return out
