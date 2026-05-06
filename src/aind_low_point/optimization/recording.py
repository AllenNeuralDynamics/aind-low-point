"""Recording-electrode geometry per probe kind.

The optimizer's coverage objective integrates the target density along
each probe shank's *active recording region* — the portion of the
shank that has electrodes for the configured bank. This file is the
single source of truth for those ranges.

Conventions
-----------
- Distances are in mm from the shank tip, measured along the shank's
  axis (probe local +z direction).
- ``active_ranges_mm`` has one ``(start_mm, end_mm)`` tuple per shank
  the optimizer should sum coverage across. For multi-shank probes
  recording from the bottom bank of every shank, all four entries are
  identical; if a particular configuration only records from a subset,
  shorten the list to those shanks.
- The shank order matches what
  :func:`runtime.shanks.detect_shank_tips_local` returns (lex-sorted
  by xy in canonical probe frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RecordingGeometry:
    """Per-shank active recording ranges for one probe kind.

    ``shank_pitch_mm`` is informational; the kinematic pivot (= the
    recording-array center in local frame) is *not* derived from it.
    Pivots come from the actual canonicalized mesh — see
    ``runtime.build._default_probe_pivot_local`` and the
    ``AssetSpec.pivot_LPS`` field. This avoids hard-coding a row
    direction (``+x`` vs ``+y``) that may not match the canonical
    convention for every probe vendor / mesh.
    """

    active_ranges_mm: tuple[tuple[float, float], ...]
    shank_pitch_mm: float = 0.25  # NP 2.0 default; informational only

    @property
    def n_shanks(self) -> int:
        return len(self.active_ranges_mm)

    @property
    def active_center_mm(self) -> float:
        """Average of (start + end)/2 across shanks. Reasonable default
        for ``past_target_mm`` initialization — puts the active region
        center at the target along the shaft axis."""
        centres = [(s + e) / 2 for s, e in self.active_ranges_mm]
        return sum(centres) / len(centres)

    @property
    def array_center_local(self) -> NDArray[np.floating]:
        """*Mesh-agnostic* fallback pivot: ``(0, 0, active_center_mm)``.

        Used only when the actual canonicalized mesh isn't available
        (e.g. legacy code paths, synthetic test contexts). The real
        path is to compute pivot from the mesh's shank tips at runtime
        build and store on ``AssetSpec.pivot_LPS``; runtime callers
        with catalog access should read that.
        """
        return np.array([0.0, 0.0, self.active_center_mm], dtype=np.float64)


# Default per-kind active-bank ranges. Overridable at optimizer-run
# time if a different bank configuration is used.
RECORDING_GEOMETRY: dict[str, RecordingGeometry] = {
    # NP 2.0 single-shank, bottom bank
    "2.1": RecordingGeometry(
        active_ranges_mm=((0.200, 3.065),),
    ),
    # NP 2.0 four-shank, bottom bank of each shank (705 µm span/shank)
    "2.4": RecordingGeometry(
        active_ranges_mm=tuple([(0.200, 0.905)] * 4),
        shank_pitch_mm=0.25,
    ),
    # Quad-base (four shanks, full bank per shank — 2864 µm span)
    "quadbase": RecordingGeometry(
        active_ranges_mm=tuple([(0.200, 3.065)] * 4),
        shank_pitch_mm=0.25,
    ),
}


def recording_center_local_for_kind(kind: str) -> NDArray[np.floating]:
    """Return the recording-array center in local frame for ``kind``,
    or ``(0, 0, 0)`` for unknown kinds (preserves the legacy
    "tip-on-target" semantics for un-registered probes)."""
    geom = RECORDING_GEOMETRY.get(kind)
    if geom is None:
        return np.zeros(3, dtype=np.float64)
    return geom.array_center_local


def get_recording_geometry(probe_kind: str) -> RecordingGeometry:
    """Return the default recording geometry for ``probe_kind``.

    Raises ``KeyError`` with the available kinds if not found, so
    misconfigured probe kinds surface immediately rather than running
    the optimizer with silent zero coverage.
    """
    if probe_kind not in RECORDING_GEOMETRY:
        raise KeyError(
            f"No recording geometry registered for probe kind {probe_kind!r}; "
            f"have {sorted(RECORDING_GEOMETRY)}"
        )
    return RECORDING_GEOMETRY[probe_kind]
