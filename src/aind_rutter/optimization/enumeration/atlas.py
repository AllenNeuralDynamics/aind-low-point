"""Target-aligned pose-feasibility atlas data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Atlas data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseAnchor:
    """One target-valid pose for (probe, hole) at a specific arc-AP."""

    ap_deg: float
    ml_deg: float
    spin_deg: float
    off_R_mm: float
    off_A_mm: float
    depth_mm: float
    threading_max_g: float
    target_miss_mm: float


@dataclass(frozen=True)
class AtlasEntry:
    """Atlas entry for one (probe, hole). Empty interval (None) means
    "no target-valid pose exists at any arc AP for this pair"."""

    probe_name: str
    hole_id: int
    ap_min: float | None
    ap_max: float | None
    anchors: tuple[PoseAnchor, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Atlas:
    """Built atlas — all (probe, hole) entries for one optimizer run."""

    entries: dict[tuple[str, int], AtlasEntry]
    probe_names: tuple[str, ...]
    hole_ids: tuple[int, ...]
