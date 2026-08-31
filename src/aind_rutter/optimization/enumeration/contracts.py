"""Small assignment carrier types used by the optimizer pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HoleAssignment:
    """One probe-to-hole discrete assignment."""

    probe_to_hole: dict[str, int]
    cost: float = 0.0
    feasible: bool = True

    @classmethod
    def infeasible(cls) -> HoleAssignment:
        return cls(probe_to_hole={}, cost=float("inf"), feasible=False)


@dataclass(frozen=True)
class ArcAssignment:
    """One probe-to-arc assignment with per-arc AP seed values."""

    probe_to_arc_idx: dict[str, int]
    arc_centroids_deg: tuple[float, ...]
    cost: float = 0.0
