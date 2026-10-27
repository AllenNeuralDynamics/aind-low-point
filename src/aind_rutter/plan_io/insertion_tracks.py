"""Locate implants from the tracks of past insertions (``rutter-tracks locate``).

A calibrated probe that went through a hole leaves a track: its NewScale tip
reading and the calibration it was read through put each of its shanks on a
line in subject LPS, and every one of those lines passed inside its bore. One
implant pose per subject has to admit every track of that subject.

Tracks read through one calibration disagree with tracks read through another
by a few tenths of a millimetre, and so do two manipulators read through the
same calibration, so a pose that admits every track rarely exists. The fit
therefore allows each manipulator (or each calibration) its own error, an
apparent shift of the implant shared by every track it read, held small by a
prior, and minimises how far tracks fall outside their bores.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

# Half an NP 2.0 shank's width: a shank's centreline clears a bore by this less
# than its edge does.
SHANK_HALF_WIDTH_MM = 0.035

ErrorModel = Literal["manipulator", "calibration", "none"]
Line = tuple[NDArray[np.floating], NDArray[np.floating]]

TRACK_COLUMNS = (
    "subject",
    "session",
    "calibration",
    "manipulator",
    "hole",
    "x_um",
    "y_um",
    "z_um",
    "kind",
    "spin_deg",
)


@dataclass(frozen=True)
class Track:
    """One insertion: a manipulator's NewScale tip reading through a hole.

    ``hole`` is ``None`` when the notes do not name it; the hole the track
    clears best at the design position is taken. ``spin_deg`` is the spin the
    probe was set to; NewScale reads shank 1's tip, and the others follow at
    the kind's pitch along the spun local y.
    """

    subject: str
    session: str
    calibration: str
    manipulator: str
    hole: int | None
    newscale_um: tuple[float, float, float]
    kind: str
    spin_deg: float


def load_tracks(path: Path) -> list[Track]:
    """Tracks from a CSV with :data:`TRACK_COLUMNS`; ``#`` starts a comment row."""
    with path.open(newline="") as f:
        rows = [
            r for r in csv.DictReader(line for line in f if not line.startswith("#"))
        ]
    missing = [c for c in TRACK_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return [
        Track(
            subject=r["subject"].strip(),
            session=r["session"].strip(),
            calibration=r["calibration"].strip(),
            manipulator=r["manipulator"].strip(),
            hole=int(r["hole"]) if r["hole"].strip() else None,
            newscale_um=(float(r["x_um"]), float(r["y_um"]), float(r["z_um"])),
            kind=r["kind"].strip(),
            spin_deg=float(r["spin_deg"] or 0.0),
        )
        for r in rows
    ]


def shank_tips_local(kind: str) -> NDArray[np.floating]:
    """Each shank's tip in the probe frame, shank 1 at the origin."""
    from aind_rutter.domain.probe_kinds import RECORDING_GEOMETRY

    geom = RECORDING_GEOMETRY.get(kind)
    if geom is None:
        raise ValueError(f"no recording geometry for probe kind {kind!r}")
    return np.array([[0.0, i * geom.shank_pitch_mm, 0.0] for i in range(geom.n_shanks)])


def reticles_of(config: Path) -> dict[str, Any]:
    """The reticles a config declares, read without resolving its paths."""
    import yaml

    from aind_rutter.config.models_plan import CalibrationReticleModel

    raw = yaml.safe_load(config.read_text())
    return {
        str(k): CalibrationReticleModel(**v)
        for k, v in ((raw.get("plan") or {}).get("reticles") or {}).items()
    }


class CalibrationBanks:
    """Calibration banks loaded one directory at a time, on first use."""

    def __init__(self, root: Path, reticles: Mapping[str, Any], reticle: str):
        if reticle not in reticles:
            raise ValueError(f"no reticle {reticle!r}; have {sorted(reticles)}")
        self.root, self.reticles, self.reticle = root, dict(reticles), reticle
        self._banks: dict[str, dict[str, Any]] = {}

    def __getitem__(self, directories: str | Sequence[str]) -> dict[str, Any]:
        """The bank of one directory, or of several with the last winning."""
        from aind_rutter.build.calibration import load_calibration_bank
        from aind_rutter.config.models_plan import CalibrationsModel

        dirs = (directories,) if isinstance(directories, str) else tuple(directories)
        key = "\n".join(dirs)
        if key not in self._banks:
            sources = [
                {"directory": str(self.root / d), "reticle": self.reticle} for d in dirs
            ]
            # A stacked source needs a probe map to validate; the bank ignores it.
            model = CalibrationsModel(sources=sources, probe_to_code={"_": "_"})
            self._banks[key] = load_calibration_bank(model, self.reticles)
        return self._banks[key]


def track_lines(track: Track, bank: Mapping[str, Any]) -> list[Line]:
    """Each shank's line in subject LPS: a point on it and its direction."""
    from aind_mri_utils.reticle_calibrations import find_probe_angle

    from aind_rutter.domain.rig import probe_rotation
    from aind_rutter.plan_io.locate_implant import HoleReading, place_readings

    if track.manipulator not in bank:
        raise ValueError(
            f"{track.subject} {track.session}: manipulator {track.manipulator} "
            f"is not in calibration {track.calibration}"
        )
    x, y, z = track.newscale_um
    mm = (x * 1e-3, y * 1e-3, z * 1e-3)
    (placed,) = place_readings(
        [HoleReading(track.manipulator, 0, mm, "pointing")], bank, {}
    )
    ap, ml = find_probe_angle(bank[track.manipulator].rotation)
    R = probe_rotation(ap, ml, track.spin_deg)
    return [
        (placed.tip + R @ tip, placed.direction) for tip in shank_tips_local(track.kind)
    ]


def bore_clearance(
    point: NDArray[np.floating],
    direction: NDArray[np.floating],
    hole: Any,
    shift: NDArray[np.floating] | None = None,
) -> float:
    """How far (mm) a line stays inside a bore moved by ``shift``, at its
    tightest section or wall; negative once the line leaves the bore.

    The gap at a section is measured radially to its ellipse, which is exact on
    the axes and close enough elsewhere for sections this round.
    """
    from aind_rutter.optimization.geometry.primitives import cap_basis

    t = np.zeros(3) if shift is None else np.asarray(shift, float)
    axis = np.asarray(hole.axis, float)
    e1, e2 = cap_basis(axis)
    out = []
    for sec in hole.sections:
        centre = np.asarray(sec.center, float) + t
        at = point + ((centre - point) @ axis) / (direction @ axis) * direction
        v = at - centre
        u1 = np.cos(sec.theta) * e1 + np.sin(sec.theta) * e2
        u2 = -np.sin(sec.theta) * e1 + np.cos(sec.theta) * e2
        u, w = v @ u1, v @ u2
        r = np.hypot(u, w)
        if r < 1e-9:
            out.append(min(sec.a, sec.b))
        else:
            out.append(r / np.hypot(u / sec.a, w / sec.b) - r)
        for wall in hole.walls:
            out.append(
                -float(np.asarray(wall.normal) @ (at - np.asarray(wall.point) - t))
            )
    return float(min(out))


def lines_clearance(
    lines: Iterable[Line], hole: Any, shift: NDArray[np.floating] | None = None
) -> float:
    """The tightest shank's edge clearance in ``hole``."""
    return (
        min(bore_clearance(p, d, hole, shift) for p, d in lines) - SHANK_HALF_WIDTH_MM
    )


def best_hole(lines: Sequence[Line], holes: Mapping[int, Any]) -> int:
    """The hole the lines clear best with the implant where designed."""
    return max(holes, key=lambda k: lines_clearance(lines, holes[k]))


@dataclass(frozen=True)
class PlacedTrack:
    track: Track
    hole: int
    lines: list[Line]


def place_tracks(
    tracks: Sequence[Track], banks: CalibrationBanks, holes: Mapping[int, Any]
) -> list[PlacedTrack]:
    out = []
    for t in tracks:
        lines = track_lines(t, banks[t.calibration])
        hole = t.hole if t.hole is not None else best_hole(lines, holes)
        if hole not in holes:
            raise ValueError(f"{t.subject} {t.session} {t.manipulator}: no hole {hole}")
        out.append(PlacedTrack(t, hole, lines))
    return out


def error_key(model: ErrorModel) -> Callable[[Track], str | None]:
    return {
        "manipulator": lambda t: t.manipulator,
        "calibration": lambda t: t.calibration,
        "none": lambda t: None,
    }[model]


@dataclass(frozen=True)
class TrackFit:
    """Implant shifts per subject and apparent shifts per error group, LPS mm."""

    errors: ErrorModel
    sigma_mm: float
    shifts: dict[str, NDArray[np.floating]]
    group_errors: dict[str, NDArray[np.floating]]
    clearance: list[float]
    placed: list[PlacedTrack] = field(repr=False)

    @property
    def rms_violation(self) -> float:
        return float(np.sqrt(np.mean(np.minimum(self.clearance, 0.0) ** 2)))


def fit_implants(
    placed: Sequence[PlacedTrack],
    holes: Mapping[int, Any],
    errors: ErrorModel = "manipulator",
    sigma_mm: float = 0.15,
    margin_mm: float = 0.02,
    tolerance_mm: float = 0.02,
) -> TrackFit:
    """One implant shift per subject, and one apparent shift per error group.

    Each track costs how far its tightest shank falls short of ``margin_mm``
    inside its bore, in units of ``tolerance_mm``; each group's error costs its
    length in units of ``sigma_mm``. Least squares from a few starts, since the
    clearance is only piecewise smooth.
    """
    subjects = sorted({p.track.subject for p in placed})
    key = error_key(errors)
    groups = sorted({g for p in placed if (g := key(p.track)) is not None})
    ns = 3 * len(subjects)

    def unpack(z):
        shifts = {s: z[3 * i : 3 * i + 3] for i, s in enumerate(subjects)}
        errs = {g: z[ns + 3 * j : ns + 3 * j + 3] for j, g in enumerate(groups)}
        return shifts, errs

    def clearances(z):
        shifts, errs = unpack(z)
        return np.array(
            [
                lines_clearance(
                    p.lines,
                    holes[p.hole],
                    shifts[p.track.subject] + _error(errs, key(p.track)),
                )
                for p in placed
            ]
        )

    def residuals(z):
        _, errs = unpack(z)
        prior = np.concatenate([e for e in errs.values()] or [np.zeros(0)])
        return np.r_[
            np.minimum(clearances(z) - margin_mm, 0.0) / tolerance_mm, prior / sigma_mm
        ]

    best = None
    # Starts span the shifts tracks have shown: up to ~0.7 mm, mostly A and S.
    for start in ([0, 0, 0], [0, -0.4, 0.2], [-0.2, -0.2, 0.3], [0.1, -0.7, 0.1]):
        z0 = np.r_[np.tile(start, len(subjects)), np.zeros(3 * len(groups))]
        sol = least_squares(residuals, z0, diff_step=1e-3)
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None
    shifts, errs = unpack(best.x)
    return TrackFit(
        errors,
        sigma_mm,
        dict(shifts),
        dict(errs),
        clearances(best.x).tolist(),
        list(placed),
    )


def _error(errs: Mapping[str, NDArray[np.floating]], group: str | None) -> Any:
    return 0.0 if group is None else errs[group]


def to_ras(v_lps: NDArray[np.floating]) -> list[float]:
    return [
        round(float(-v_lps[0]), 3),
        round(float(-v_lps[1]), 3),
        round(float(v_lps[2]), 3),
    ]


def from_ras(v_ras: Sequence[float]) -> NDArray[np.floating]:
    return np.array([-v_ras[0], -v_ras[1], v_ras[2]], float)


def fit_record(fit: TrackFit, holes_path: Path, tracks_path: Path) -> dict[str, Any]:
    """The fit as YAML-ready data; shifts in RAS mm, as the planner shows them."""
    clear = np.array(fit.clearance)
    return {
        "tracks_file": str(tracks_path),
        "holes": str(holes_path),
        "errors": fit.errors,
        "sigma_mm": fit.sigma_mm,
        "rms_violation_mm": round(fit.rms_violation, 3),
        "worst_clearance_mm": round(float(clear.min()), 3),
        "tracks_outside": int((clear < 0).sum()),
        "subjects": {
            s: {
                "implant_shift_RAS_mm": to_ras(v),
                "tracks": sum(p.track.subject == s for p in fit.placed),
            }
            for s, v in fit.shifts.items()
        },
        f"{fit.errors}_errors_RAS_mm": {
            g: to_ras(v) for g, v in fit.group_errors.items()
        },
        "tracks": [
            {
                "subject": p.track.subject,
                "session": p.track.session,
                "manipulator": p.track.manipulator,
                "calibration": p.track.calibration,
                "hole": p.hole,
                "clearance_mm": round(c, 3),
            }
            for p, c in zip(fit.placed, fit.clearance)
        ],
    }


def report_fit(fit: TrackFit) -> list[str]:
    clear = np.array(fit.clearance)
    lines = [
        f"errors per {fit.errors} (prior {fit.sigma_mm:g} mm): rms shortfall "
        f"{fit.rms_violation:.3f} mm, worst {clear.min():+.3f} mm, "
        f"{(clear < 0).sum()}/{len(clear)} tracks outside their bores",
    ]
    for s, v in fit.shifts.items():
        r, a, sup = to_ras(v)
        lines.append(f"  implant {s}: R {r:+.2f}  A {a:+.2f}  S {sup:+.2f} mm")
    for g, v in fit.group_errors.items():
        r, a, sup = to_ras(v)
        lines.append(f"  {fit.errors} {g}: R {r:+.2f}  A {a:+.2f}  S {sup:+.2f} mm")
    for p, c in zip(fit.placed, fit.clearance):
        t = p.track
        lines.append(
            f"    {t.subject} {t.session:<10} {t.manipulator:>6} "
            f"hole {p.hole:>2}: {c:+.3f}"
        )
    return lines
