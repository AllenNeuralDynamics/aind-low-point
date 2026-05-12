"""Placement-optimizer subpackage.

Currently exposes geometric primitives only; the optimization driver
(CMA-ES + SLSQP, JAX inner loop) is not yet wired. See ``dev/optimizer_plan.md``
for the design.
"""

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
from aind_low_point.optimization.density import (
    DensityFn,
    coverage,
    gaussian_density,
    integrate_density_along_shank,
)
from aind_low_point.optimization.recording import (
    RECORDING_GEOMETRY,
    RecordingGeometry,
    get_recording_geometry,
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
from aind_low_point.optimization.arc_assignment import (
    ArcAssignment,
    enumerate_partitions,
    required_aps_deg_for_assignment,
    solve_top_k_arc_assignments,
)
from aind_low_point.optimization.optimize import (
    OptimizationResult,
    PlanCandidate,
    ProbeStaticInfo,
    format_plan_table,
    optimize,
)

__all__ = [
    "Capsule",
    "DensityFn",
    "Hole",
    "HoleSection",
    "ObjectiveBreakdown",
    "ObjectiveWeights",
    "OptimizerContext",
    "ProbeContext",
    "ProbeEvaluation",
    "RECORDING_GEOMETRY",
    "RecordingGeometry",
    "VariableLayout",
    "cap_basis",
    "capsule_capsule_dist",
    "coverage",
    "evaluate_objective",
    "evaluate_probe",
    "find_hole_by_id",
    "gaussian_density",
    "get_recording_geometry",
    "headstage_capsule",
    "integrate_density_along_shank",
    "kinematic_separations",
    "load_holes",
    "make_objective",
    "pairwise_headstage_clearances",
    "point_to_segment_dist",
    "pose_at_hole_best_fit",
    "pose_from_optimizer_vars",
    "required_ap_deg",
    "scalar_objective",
    "section_oval_value",
    "segment_to_segment_dist",
    "shaft_section_oval_value",
    "shank_capsules_from_pose",
]
