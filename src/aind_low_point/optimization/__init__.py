"""Placement-optimizer subpackage.

Currently exposes geometric primitives only; the optimization driver
(CMA-ES + SLSQP, JAX inner loop) is not yet wired. See ``dev/optimizer_plan.md``
for the design.
"""

from aind_low_point.optimization.arc_assignment import (
    ArcAssignment,
    enumerate_partitions,
    required_aps_deg_for_assignment,
    solve_top_k_arc_assignments,
)
from aind_low_point.optimization.density import (
    DensityFn,
    coverage,
    gaussian_density,
    integrate_density_along_shank,
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
    make_fcl_convex,
)
from aind_low_point.optimization.hole_assignment import (
    AssignmentProbe,
    CostWeights,
    HoleAssignment,
    angle_to_target_rad,
    build_cost_matrix,
    pairwise_interference_penalty,
    solve_optimal_assignment,
    solve_top_k_assignments,
    static_threading_max_g,
)
from aind_low_point.optimization.holes import (
    Hole,
    find_hole_by_id,
    load_holes,
)
from aind_low_point.optimization.joint_rerank import (
    JointCandidate,
    JointRerankMetrics,
    JointWeights,
    expand_reduced_solution_to_full_x,
    optimize_joint,
    score_joint,
)
from aind_low_point.optimization.kinematics import (
    pose_at_hole_best_fit,
    pose_from_optimizer_vars,
    required_ap_deg,
    shank_capsules_from_pose,
)
from aind_low_point.optimization.pose_features import (
    PoseFeatures,
    precompute_pose_features,
    required_ap_ml_for_target,
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
from aind_low_point.optimization.optimize import (
    OptimizationResult,
    PlanCandidate,
    ProbeStaticInfo,
    best_fit_hole_id_at_pose,
    format_plan_table,
    optimize,
    polish_seed,
)
from aind_low_point.optimization.recording import (
    RECORDING_GEOMETRY,
    RecordingGeometry,
    get_recording_geometry,
)

__all__ = [
    "ArcAssignment",
    "AssignmentProbe",
    "Capsule",
    "CostWeights",
    "DensityFn",
    "Hole",
    "HoleAssignment",
    "HoleSection",
    "JointCandidate",
    "JointRerankMetrics",
    "JointWeights",
    "ObjectiveBreakdown",
    "ObjectiveWeights",
    "OptimizationResult",
    "OptimizerContext",
    "PlanCandidate",
    "PoseFeatures",
    "ProbeContext",
    "ProbeEvaluation",
    "ProbeStaticInfo",
    "RECORDING_GEOMETRY",
    "RecordingGeometry",
    "VariableLayout",
    "angle_to_target_rad",
    "best_fit_hole_id_at_pose",
    "build_cost_matrix",
    "build_headstage_hull",
    "cap_basis",
    "capsule_capsule_dist",
    "coverage",
    "detect_body_region",
    "enumerate_partitions",
    "evaluate_objective",
    "evaluate_probe",
    "expand_reduced_solution_to_full_x",
    "find_hole_by_id",
    "format_plan_table",
    "gaussian_density",
    "get_recording_geometry",
    "headstage_capsule",
    "integrate_density_along_shank",
    "kinematic_separations",
    "load_holes",
    "make_fcl_convex",
    "make_objective",
    "optimize",
    "optimize_joint",
    "pairwise_headstage_clearances",
    "pairwise_interference_penalty",
    "point_to_segment_dist",
    "polish_seed",
    "pose_at_hole_best_fit",
    "pose_from_optimizer_vars",
    "precompute_pose_features",
    "required_ap_deg",
    "required_ap_ml_for_target",
    "required_aps_deg_for_assignment",
    "scalar_objective",
    "score_joint",
    "section_oval_value",
    "segment_to_segment_dist",
    "shaft_section_oval_value",
    "shank_capsules_from_pose",
    "solve_optimal_assignment",
    "solve_top_k_arc_assignments",
    "solve_top_k_assignments",
    "static_threading_max_g",
]
