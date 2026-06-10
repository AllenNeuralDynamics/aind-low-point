"""Geometry, kinematics, holes, and static probe data."""

from aind_low_point.optimization.geometry.holes import (
    DEFAULT_THREADING_MARGIN_MM,
    Hole,
    find_hole_by_id,
    load_holes,
    threading_margin_mm,
)
from aind_low_point.optimization.geometry.kinematics import (
    pose_at_hole_best_fit,
    pose_from_optimizer_vars,
    required_ap_deg,
    shank_capsules_from_pose,
)
from aind_low_point.optimization.geometry.primitives import (
    Capsule,
    HoleSection,
    cap_basis,
    capsule_capsule_dist,
    line_plane_intersect,
    point_to_segment_dist,
    section_oval_value,
    segment_to_segment_dist,
    shaft_section_oval_value,
)
from aind_low_point.optimization.geometry.probes import ProbeStaticInfo
from aind_low_point.optimization.geometry.recording import (
    RECORDING_GEOMETRY,
    RecordingGeometry,
    get_recording_geometry,
    recording_center_local_for_kind,
)

__all__ = [
    "DEFAULT_THREADING_MARGIN_MM",
    "Capsule",
    "Hole",
    "HoleSection",
    "ProbeStaticInfo",
    "RECORDING_GEOMETRY",
    "RecordingGeometry",
    "cap_basis",
    "capsule_capsule_dist",
    "find_hole_by_id",
    "get_recording_geometry",
    "line_plane_intersect",
    "load_holes",
    "point_to_segment_dist",
    "pose_at_hole_best_fit",
    "pose_from_optimizer_vars",
    "recording_center_local_for_kind",
    "required_ap_deg",
    "section_oval_value",
    "segment_to_segment_dist",
    "shaft_section_oval_value",
    "shank_capsules_from_pose",
    "threading_margin_mm",
]
