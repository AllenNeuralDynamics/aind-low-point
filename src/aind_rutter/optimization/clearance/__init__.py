"""Signed-distance field builders and JAX clearance kernels."""

from aind_rutter.optimization.clearance.envelope import (
    build_alpha_wrap_envelope,
    extract_shank_obbs,
    floor_shank_half_extents,
    strip_shanks,
)
from aind_rutter.optimization.clearance.kernels import (
    FixtureClearance,
    PairClearance,
    arc_angles_to_rotation,
    dual_rep_fixture_clearance,
    dual_rep_pair_clearance,
    pairwise_signed_clearance_dual,
    pairwise_signed_clearance_dual_world,
    pose_from_optimizer_vars,
    spin_deg_from_sxy,
    unit_circle_penalty,
)
from aind_rutter.optimization.clearance.voxel_sdf import (
    DEFAULT_PAD_MM,
    DEFAULT_SPACING_MM,
    ProbeSDF,
    build_probe_sdf,
    build_probe_sdf_from_alpha_wrap,
    build_sdf_by_name,
)

__all__ = [
    "DEFAULT_PAD_MM",
    "DEFAULT_SPACING_MM",
    "FixtureClearance",
    "PairClearance",
    "ProbeSDF",
    "arc_angles_to_rotation",
    "build_alpha_wrap_envelope",
    "build_probe_sdf",
    "build_probe_sdf_from_alpha_wrap",
    "build_sdf_by_name",
    "dual_rep_fixture_clearance",
    "dual_rep_pair_clearance",
    "extract_shank_obbs",
    "floor_shank_half_extents",
    "pairwise_signed_clearance_dual",
    "pairwise_signed_clearance_dual_world",
    "pose_from_optimizer_vars",
    "spin_deg_from_sxy",
    "strip_shanks",
    "unit_circle_penalty",
]
