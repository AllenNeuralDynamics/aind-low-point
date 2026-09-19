"""Runtime probe-static container."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from numpy.typing import NDArray

from aind_rutter.domain.probe_kinds import RecordingGeometry

# ---------------------------------------------------------------------------
# Per-probe static info input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeStaticInfo:
    """Per-probe info the optimizer needs from the caller.

    Combines what the outer + inner layers each need: target, kind,
    detected shank tips. ``density_sigma_mm`` controls the coverage
    objective's Gaussian width / mixture bandwidth. ``collision_mesh``
    (optional) is the probe's canonical-local mesh used for the
    inner-loop pairwise clearance constraint via FCL BVH / GJK; pass
    the full probe mesh (not just the headstage region) so the silicon
    body and connector regions are included.

    ``target_points`` (optional) holds an ``(N, 3)`` point cloud (in
    world LPS mm) that, when set, switches the coverage density from a
    single-point Gaussian on ``target_LPS`` to an equally-weighted
    Gaussian mixture over the cloud. ``target_LPS`` is still used for
    target-anchored pose-bank construction and should be the
    cloud's centroid in that case.
    """

    name: str
    target_LPS: NDArray[np.floating]
    kind: str
    shank_tips_local: NDArray[np.floating]
    # Resolved by the config, not looked up from the built-in table: a subject
    # may use a probe the table has never heard of. ``None`` means no array.
    recording: "RecordingGeometry | None" = None
    density_sigma_mm: float = 0.5
    collision_mesh: trimesh.Trimesh | None = field(default=None, compare=False)
    target_points: NDArray[np.floating] | None = field(default=None, compare=False)
    # Per-target priority weight applied (after normalization) to this probe's
    # coverage in the normalized objective's weighted SUM (the fairness floor
    # stays unweighted). 1.0 ⇒ no preference.
    coverage_weight: float = 1.0
    # Kinematic pivot in the probe's local frame. None means "derive it from the
    # shank tips", which is what a caller building this by hand gets.
    pivot_local: NDArray[np.floating] | None = field(default=None, compare=False)
