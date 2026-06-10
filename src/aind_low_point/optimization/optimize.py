"""Runtime probe-static container."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from numpy.typing import NDArray

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
    LSAP target-anchored pose-bank construction and should be the
    cloud's centroid in that case.
    """

    name: str
    target_LPS: NDArray[np.floating]
    kind: str
    shank_tips_local: NDArray[np.floating]
    density_sigma_mm: float = 0.5
    collision_mesh: trimesh.Trimesh | None = field(default=None, compare=False)
    target_points: NDArray[np.floating] | None = field(default=None, compare=False)
    # Per-target priority weight applied (after normalization) to this probe's
    # coverage in the normalized objective's weighted SUM (the fairness floor
    # stays unweighted). 1.0 ⇒ no preference.
    coverage_weight: float = 1.0
