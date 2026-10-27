"""How far each manipulator lands from the angles it was dialed to
(``rutter-tracks bias``), and the dial settings that allow for it.

A calibration measures where a probe actually points. Against the plan it was
dialed from, that gives the manipulator's error in AP and ML for the session.
Arcs set AP for every probe they carry, so an arc is dialed off by the mean
bias of its manipulators; sliders are per probe, so ML is corrected where a
manipulator's bias keeps its sign. Every AP error measured so far has been
positive, so a manipulator's mean AP error is its correction once it has two
sessions. ML errors change sign, so an ML correction needs its mean to be at
least twice its spread across sessions: below that it is noise, and dialing
against noise moves the probe as far as it corrects.

Angles are subject angles, as the plan stores them and the calibration reports
them; the rig's AP is the subject's plus the head pitch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aind_rutter.plan_io.insertion_tracks import CalibrationBanks, reticles_of

# An ML bias applies when |mean| is at least this many spreads.
CONSISTENT_SPREADS = 2.0


@dataclass(frozen=True)
class Sample:
    manipulator: str
    session: str
    calibration: str
    probe: str
    arc: str | None
    planned_ap: float
    planned_ml: float
    calibrated_ap: float
    calibrated_ml: float

    @property
    def ap_error(self) -> float:
        return self.calibrated_ap - self.planned_ap

    @property
    def ml_error(self) -> float:
        return self.calibrated_ml - self.planned_ml


def planned_angles(
    plan: Mapping[str, Any],
) -> dict[str, tuple[str | None, float, float]]:
    """Per probe: its arc and the subject AP and ML the plan dials."""
    arcs = plan.get("arcs") or {}
    out = {}
    for name, p in (plan.get("probes") or {}).items():
        arc = p.get("arc")
        bound = p.get("bind_ap_to_arc", True) and arc is not None
        ap = arcs[arc] if bound else p.get("ap_local")
        if ap is None or p.get("slider_ml") is None:
            continue
        out[name] = (arc, float(ap), float(p["slider_ml"]))
    return out


def session_samples(session: Mapping[str, Any], root: Path) -> list[Sample]:
    """Calibrated minus planned angles for every mapped probe of one session.

    ``session`` names a ``config`` (for its probe map and reticle), an optional
    ``plan`` file when the config's own plan is not what was dialed, and the
    ``calibrations`` directories in the order they were taken; a later
    directory's calibration of a manipulator replaces an earlier one's.
    """
    from aind_mri_utils.reticle_calibrations import find_probe_angle

    config = Path(session["config"])
    raw = yaml.safe_load(config.read_text())
    cal = (raw.get("plan") or {}).get("calibrations") or {}
    plan = (
        yaml.safe_load(Path(session["plan"]).read_text())
        if session.get("plan")
        else raw.get("plan") or {}
    )
    reticle = session.get("reticle") or next(
        (s.get("reticle") for s in cal.get("sources") or [] if s.get("reticle")), None
    )
    if reticle is None:
        raise ValueError(f"{config}: name the session's reticle")
    banks = CalibrationBanks(root, reticles_of(config), str(reticle))
    dirs = list(session["calibrations"])
    planned = planned_angles(plan)
    out = []
    for probe, code in (cal.get("probe_to_code") or {}).items():
        code = str(code)
        if probe not in planned:
            continue
        source = next((d for d in reversed(dirs) if code in banks[d]), None)
        if source is None:
            continue
        ap, ml = find_probe_angle(banks[source][code].rotation)
        arc, pap, pml = planned[probe]
        out.append(
            Sample(
                code,
                session["label"],
                source,
                probe,
                arc,
                pap,
                pml,
                float(ap),
                float(ml),
            )
        )
    return out


def load_sessions(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    """A sessions file: ``calibrations_root`` and a list of ``sessions``.

    Relative ``config`` and ``plan`` paths resolve against the file's folder.
    """
    raw = yaml.safe_load(path.read_text())
    here = path.parent
    sessions = []
    for s in raw["sessions"]:
        s = dict(s)
        for k in ("config", "plan"):
            if s.get(k):
                s[k] = str((here / s[k]).expanduser())
        sessions.append(s)
    return Path(raw["calibrations_root"]).expanduser(), sessions


@dataclass(frozen=True)
class Bias:
    n: int
    ap_mean: float
    ap_sd: float
    ml_mean: float
    ml_sd: float

    @staticmethod
    def consistent(mean: float, sd: float, n: int) -> bool:
        return n >= 2 and abs(mean) >= CONSISTENT_SPREADS * sd

    @property
    def ap_correction(self) -> float:
        return self.ap_mean if self.n >= 2 else 0.0

    @property
    def ml_correction(self) -> float:
        return (
            self.ml_mean if self.consistent(self.ml_mean, self.ml_sd, self.n) else 0.0
        )


def summarize(samples: Sequence[Sample]) -> dict[str, Bias]:
    out = {}
    for m in sorted({s.manipulator for s in samples}):
        ap = np.array([s.ap_error for s in samples if s.manipulator == m])
        ml = np.array([s.ml_error for s in samples if s.manipulator == m])
        out[m] = Bias(len(ap), float(ap.mean()), _sd(ap), float(ml.mean()), _sd(ml))
    return out


def bias_record(samples: Sequence[Sample], sessions_file: Path) -> dict[str, Any]:
    r = lambda v: round(float(v), 2)  # noqa: E731
    return {
        "sessions_file": str(sessions_file),
        "angles": "subject degrees; error is calibrated minus planned",
        "consistent_spreads": CONSISTENT_SPREADS,
        "manipulators": {
            m: {
                "n": b.n,
                "ap_error_deg": {"mean": r(b.ap_mean), "sd": r(b.ap_sd)},
                "ml_error_deg": {"mean": r(b.ml_mean), "sd": r(b.ml_sd)},
                "ap_correction_deg": r(b.ap_correction),
                "ml_correction_deg": r(b.ml_correction),
            }
            for m, b in summarize(samples).items()
        },
        "samples": [
            {
                "manipulator": s.manipulator,
                "session": s.session,
                "calibration": s.calibration,
                "probe": s.probe,
                "arc": s.arc,
                "planned_ap": r(s.planned_ap),
                "planned_ml": r(s.planned_ml),
                "ap_error": r(s.ap_error),
                "ml_error": r(s.ml_error),
            }
            for s in samples
        ],
    }


def load_biases(path: Path) -> dict[str, Bias]:
    raw = yaml.safe_load(path.read_text())
    return {
        str(m): Bias(
            v["n"],
            v["ap_error_deg"]["mean"],
            v["ap_error_deg"]["sd"],
            v["ml_error_deg"]["mean"],
            v["ml_error_deg"]["sd"],
        )
        for m, v in raw["manipulators"].items()
    }


def _sd(v: np.ndarray) -> float:
    return float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def report_samples(samples: Sequence[Sample]) -> list[str]:
    lines = [
        f"{'manip':>6} {'session':<14} {'probe':<12} {'arc':>3} "
        f"{'plan AP':>8} {'plan ML':>8} {'dAP':>6} {'dML':>6}"
    ]
    for s in sorted(samples, key=lambda s: (s.manipulator, s.session)):
        lines.append(
            f"{s.manipulator:>6} {s.session:<14} {s.probe:<12} {s.arc or '-':>3} "
            f"{s.planned_ap:8.2f} {s.planned_ml:8.2f} "
            f"{s.ap_error:+6.2f} {s.ml_error:+6.2f}"
        )
    lines.append("")
    lines.append(
        f"{'manip':>6} {'n':>2} {'AP mean±sd':>12} {'ML mean±sd':>12}  "
        "corrections AP, ML"
    )
    for m, b in summarize(samples).items():
        lines.append(
            f"{m:>6} {b.n:>2} {b.ap_mean:+6.2f}±{b.ap_sd:4.2f} "
            f"{b.ml_mean:+6.2f}±{b.ml_sd:4.2f}  "
            f"{b.ap_correction:+.2f}, {b.ml_correction:+.2f}"
        )
    return lines


@dataclass(frozen=True)
class DialSheet:
    """What to set on the rig so the probes land on the plan."""

    head_pitch_deg: float
    arcs: dict[str, dict[str, Any]]
    probes: dict[str, dict[str, Any]]


def dial_sheet(
    plan: Mapping[str, Any],
    assignment: Mapping[str, str],
    biases: Mapping[str, Bias],
    head_pitch_deg: float,
) -> DialSheet:
    """Each arc dialed back by the mean AP correction of its manipulators, and
    each slider by its manipulator's ML correction.

    A manipulator with fewer than two sessions takes the mean AP correction of
    those with more, since every measured manipulator has read high.
    """
    planned = planned_angles(plan)
    unknown = sorted(set(assignment) - set(planned))
    if unknown:
        raise ValueError(f"the plan has no probes {unknown}")
    measured = [b.ap_mean for b in biases.values() if b.n >= 2]
    fleet_ap = float(np.mean(measured)) if measured else 0.0

    def ap_corr(m: str) -> float:
        return biases[m].ap_mean if m in biases and biases[m].n >= 2 else fleet_ap

    arcs: dict[str, dict[str, Any]] = {}
    for arc, value in (plan.get("arcs") or {}).items():
        members = [
            p for p, (a, _, _) in planned.items() if a == arc and p in assignment
        ]
        if not members:
            continue
        corr = float(np.mean([ap_corr(assignment[p]) for p in members]))
        arcs[arc] = {
            "planned_rig_ap": round(float(value) + head_pitch_deg, 2),
            "dial_rig_ap": round(float(value) + head_pitch_deg - corr, 2),
            "correction": round(corr, 2),
            "probes": members,
        }
    probes = {}
    for p, m in assignment.items():
        arc, _, ml = planned[p]
        corr = biases[m].ml_correction if m in biases else 0.0
        arc_corr = arcs[arc]["correction"] if arc in arcs else 0.0
        probes[p] = {
            "manipulator": m,
            "arc": arc,
            "planned_ml": round(ml, 2),
            "dial_ml": round(ml - corr, 2),
            "ml_correction": round(corr, 2),
            # The arc is dialed for its manipulators' mean bias, so each lands
            # off plan by the difference between its own and that mean.
            "landing_ap_offset": round(ap_corr(m) - arc_corr, 2),
            "ap_bias": round(ap_corr(m), 2),
            "ml_bias": round(corr, 2),
            "known": m in biases,
        }
    return DialSheet(head_pitch_deg, arcs, probes)
