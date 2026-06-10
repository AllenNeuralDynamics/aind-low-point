"""Placement-optimizer subpackage.

The current production optimizer is the offline ``optimization.pipeline`` flow:
``alp-phase1`` builds the MRV/RProp pool, ``alp-phase2`` polishes and ranks the
handoff, and ``alp-emit`` writes plan-only YAML files. This package also exports
the lower-level geometry, assignment, density, and objective helpers used by
the pipeline and by diagnostic scripts.
"""

from aind_low_point.optimization.arc_placement import bounded_isotonic_arc_aps
from aind_low_point.optimization.assignment_contracts import (
    ArcAssignment,
    HoleAssignment,
)
from aind_low_point.optimization.density import (
    DensityFn,
    coverage,
    gaussian_density,
    gaussian_mixture_density,
    integrate_density_along_shank,
    voxel_kde_density,
)
from aind_low_point.optimization.geometry import (
    Capsule,
    HoleSection,
    cap_basis,
    capsule_capsule_dist,
    point_to_segment_dist,
    section_oval_value,
    segment_to_segment_dist,
    shaft_section_oval_value,
)
from aind_low_point.optimization.headstages import (
    build_headstage_hull,
    detect_body_region,
    make_fcl_bvh,
    make_fcl_convex,
)
from aind_low_point.optimization.holes import (
    Hole,
    find_hole_by_id,
    load_holes,
)
from aind_low_point.optimization.kinematics import (
    pose_at_hole_best_fit,
    pose_from_optimizer_vars,
    required_ap_deg,
    shank_capsules_from_pose,
)
from aind_low_point.optimization.objective import (
    ObjectiveBreakdown,
    ObjectiveWeights,
    OptimizerContext,
    ProbeContext,
    ProbeEvaluation,
    VariableLayout,
    evaluate_objective,
    evaluate_probe,
    headstage_capsule,
    kinematic_separations,
    make_objective,
    pairwise_headstage_clearances,
    scalar_objective,
)
from aind_low_point.optimization.optimize import ProbeStaticInfo
from aind_low_point.optimization.pose_features import (
    PoseFeatures,
    precompute_pose_features,
    required_ap_ml_for_target,
)
from aind_low_point.optimization.probe_static import JointWeights
from aind_low_point.optimization.recording import (
    RECORDING_GEOMETRY,
    RecordingGeometry,
    get_recording_geometry,
)

__all__ = [
    "ArcAssignment",
    "bounded_isotonic_arc_aps",
    "Capsule",
    "DensityFn",
    "Hole",
    "HoleAssignment",
    "HoleSection",
    "JointWeights",
    "ObjectiveBreakdown",
    "ObjectiveWeights",
    "OptimizerContext",
    "PoseFeatures",
    "ProbeContext",
    "ProbeEvaluation",
    "ProbeStaticInfo",
    "RECORDING_GEOMETRY",
    "RecordingGeometry",
    "VariableLayout",
    "build_headstage_hull",
    "cap_basis",
    "capsule_capsule_dist",
    "coverage",
    "detect_body_region",
    "evaluate_objective",
    "evaluate_probe",
    "find_hole_by_id",
    "gaussian_density",
    "gaussian_mixture_density",
    "get_recording_geometry",
    "headstage_capsule",
    "integrate_density_along_shank",
    "kinematic_separations",
    "load_holes",
    "make_fcl_bvh",
    "make_fcl_convex",
    "make_objective",
    "pairwise_headstage_clearances",
    "point_to_segment_dist",
    "pose_at_hole_best_fit",
    "pose_from_optimizer_vars",
    "precompute_pose_features",
    "required_ap_deg",
    "required_ap_ml_for_target",
    "scalar_objective",
    "section_oval_value",
    "segment_to_segment_dist",
    "shaft_section_oval_value",
    "shank_capsules_from_pose",
    "voxel_kde_density",
]
