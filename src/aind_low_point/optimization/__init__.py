"""Placement-optimizer subpackage.

The current production optimizer is the offline ``optimization.pipeline`` flow:
``alp-phase1`` builds the MRV/RProp pool, ``alp-phase2`` polishes and ranks the
handoff, and ``alp-emit`` writes plan-only YAML files. This package also exports
the lower-level geometry, assignment, density, and objective helpers used by
the pipeline and by diagnostic scripts.
"""

import importlib
from typing import TYPE_CHECKING

# Lazy re-exports: map each public symbol to the submodule that defines it.
# Importing this package must NOT eagerly pull in the JAX-heavy ``objectives``
# modules — the trame app only needs the (JAX-free) ``geometry`` helpers, and
# loading JAX here drags in the GPU/CUDA backend just to launch the viewer.
# Symbols are imported on first attribute access via PEP 562 ``__getattr__``.
_LAZY_EXPORTS = {
    "bounded_isotonic_arc_aps": "aind_low_point.optimization.enumeration.arc_placement",
    "ArcAssignment": "aind_low_point.optimization.enumeration.contracts",
    "HoleAssignment": "aind_low_point.optimization.enumeration.contracts",
    "Capsule": "aind_low_point.optimization.geometry",
    "HoleSection": "aind_low_point.optimization.geometry",
    "cap_basis": "aind_low_point.optimization.geometry",
    "capsule_capsule_dist": "aind_low_point.optimization.geometry",
    "point_to_segment_dist": "aind_low_point.optimization.geometry",
    "section_oval_value": "aind_low_point.optimization.geometry",
    "segment_to_segment_dist": "aind_low_point.optimization.geometry",
    "shaft_section_oval_value": "aind_low_point.optimization.geometry",
    "build_headstage_hull": "aind_low_point.optimization.geometry.headstages",
    "detect_body_region": "aind_low_point.optimization.geometry.headstages",
    "make_fcl_bvh": "aind_low_point.optimization.geometry.headstages",
    "make_fcl_convex": "aind_low_point.optimization.geometry.headstages",
    "Hole": "aind_low_point.optimization.geometry.holes",
    "find_hole_by_id": "aind_low_point.optimization.geometry.holes",
    "load_holes": "aind_low_point.optimization.geometry.holes",
    "pose_at_hole_best_fit": "aind_low_point.optimization.geometry.kinematics",
    "pose_from_optimizer_vars": "aind_low_point.optimization.geometry.kinematics",
    "required_ap_deg": "aind_low_point.optimization.geometry.kinematics",
    "shank_capsules_from_pose": "aind_low_point.optimization.geometry.kinematics",
    "ProbeStaticInfo": "aind_low_point.optimization.geometry.probes",
    "RECORDING_GEOMETRY": "aind_low_point.optimization.geometry.recording",
    "RecordingGeometry": "aind_low_point.optimization.geometry.recording",
    "get_recording_geometry": "aind_low_point.optimization.geometry.recording",
    "DensityFn": "aind_low_point.optimization.objectives.density",
    "coverage": "aind_low_point.optimization.objectives.density",
    "gaussian_density": "aind_low_point.optimization.objectives.density",
    "gaussian_mixture_density": "aind_low_point.optimization.objectives.density",
    "integrate_density_along_shank": "aind_low_point.optimization.objectives.density",
    "voxel_kde_density": "aind_low_point.optimization.objectives.density",
    "JointWeights": "aind_low_point.optimization.objectives.probe_static",
}


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:  # static type-checkers see the real symbols
    from aind_low_point.optimization.enumeration.arc_placement import (
        bounded_isotonic_arc_aps,
    )
    from aind_low_point.optimization.enumeration.contracts import (
        ArcAssignment,
        HoleAssignment,
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
    from aind_low_point.optimization.geometry.headstages import (
        build_headstage_hull,
        detect_body_region,
        make_fcl_bvh,
        make_fcl_convex,
    )
    from aind_low_point.optimization.geometry.holes import (
        Hole,
        find_hole_by_id,
        load_holes,
    )
    from aind_low_point.optimization.geometry.kinematics import (
        pose_at_hole_best_fit,
        pose_from_optimizer_vars,
        required_ap_deg,
        shank_capsules_from_pose,
    )
    from aind_low_point.optimization.geometry.probes import ProbeStaticInfo
    from aind_low_point.optimization.geometry.recording import (
        RECORDING_GEOMETRY,
        RecordingGeometry,
        get_recording_geometry,
    )
    from aind_low_point.optimization.objectives.density import (
        DensityFn,
        coverage,
        gaussian_density,
        gaussian_mixture_density,
        integrate_density_along_shank,
        voxel_kde_density,
    )
    from aind_low_point.optimization.objectives.probe_static import JointWeights

__all__ = [
    "ArcAssignment",
    "bounded_isotonic_arc_aps",
    "Capsule",
    "DensityFn",
    "Hole",
    "HoleAssignment",
    "HoleSection",
    "JointWeights",
    "ProbeStaticInfo",
    "RECORDING_GEOMETRY",
    "RecordingGeometry",
    "build_headstage_hull",
    "cap_basis",
    "capsule_capsule_dist",
    "coverage",
    "detect_body_region",
    "find_hole_by_id",
    "gaussian_density",
    "gaussian_mixture_density",
    "get_recording_geometry",
    "integrate_density_along_shank",
    "load_holes",
    "make_fcl_bvh",
    "make_fcl_convex",
    "point_to_segment_dist",
    "pose_at_hole_best_fit",
    "pose_from_optimizer_vars",
    "required_ap_deg",
    "section_oval_value",
    "segment_to_segment_dist",
    "shaft_section_oval_value",
    "shank_capsules_from_pose",
    "voxel_kde_density",
]
