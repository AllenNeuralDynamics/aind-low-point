"""Placement-optimizer subpackage.

The current production optimizer is the offline ``optimization.pipeline`` flow:
``rutter-phase1`` builds the MRV/RProp pool, ``rutter-phase2`` polishes and ranks the
handoff, and ``rutter-emit`` writes plan-only YAML files. This package also exports
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
    "bounded_isotonic_arc_aps": "aind_rutter.optimization.enumeration.arc_placement",
    "ArcAssignment": "aind_rutter.optimization.enumeration.contracts",
    "HoleAssignment": "aind_rutter.optimization.enumeration.contracts",
    "Capsule": "aind_rutter.optimization.geometry",
    "HoleSection": "aind_rutter.optimization.geometry",
    "cap_basis": "aind_rutter.optimization.geometry",
    "capsule_capsule_dist": "aind_rutter.optimization.geometry",
    "point_to_segment_dist": "aind_rutter.optimization.geometry",
    "section_oval_value": "aind_rutter.optimization.geometry",
    "segment_to_segment_dist": "aind_rutter.optimization.geometry",
    "shaft_section_oval_value": "aind_rutter.optimization.geometry",
    "make_fcl_bvh": "aind_rutter.optimization.geometry.headstages",
    "Hole": "aind_rutter.optimization.geometry.holes",
    "find_hole_by_id": "aind_rutter.optimization.geometry.holes",
    "load_holes": "aind_rutter.optimization.geometry.holes",
    "pose_at_hole_best_fit": "aind_rutter.optimization.geometry.kinematics",
    "pose_from_optimizer_vars": "aind_rutter.optimization.geometry.kinematics",
    "required_ap_deg": "aind_rutter.optimization.geometry.kinematics",
    "shank_capsules_from_pose": "aind_rutter.optimization.geometry.kinematics",
    "ProbeStaticInfo": "aind_rutter.optimization.geometry.probes",
    "RECORDING_GEOMETRY": "aind_rutter.optimization.geometry.recording",
    "RecordingGeometry": "aind_rutter.optimization.geometry.recording",
    "get_recording_geometry": "aind_rutter.optimization.geometry.recording",
    "DensityFn": "aind_rutter.optimization.objectives.density",
    "coverage": "aind_rutter.optimization.objectives.density",
    "gaussian_density": "aind_rutter.optimization.objectives.density",
    "gaussian_mixture_density": "aind_rutter.optimization.objectives.density",
    "integrate_density_along_shank": "aind_rutter.optimization.objectives.density",
    "voxel_kde_density": "aind_rutter.optimization.objectives.density",
    "JointWeights": "aind_rutter.optimization.objectives.probe_static",
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
    from aind_rutter.optimization.enumeration.arc_placement import (
        bounded_isotonic_arc_aps,
    )
    from aind_rutter.optimization.enumeration.contracts import (
        ArcAssignment,
        HoleAssignment,
    )
    from aind_rutter.optimization.geometry import (
        Capsule,
        HoleSection,
        cap_basis,
        capsule_capsule_dist,
        point_to_segment_dist,
        section_oval_value,
        segment_to_segment_dist,
        shaft_section_oval_value,
    )
    from aind_rutter.optimization.geometry.headstages import make_fcl_bvh
    from aind_rutter.optimization.geometry.holes import (
        Hole,
        find_hole_by_id,
        load_holes,
    )
    from aind_rutter.optimization.geometry.kinematics import (
        pose_at_hole_best_fit,
        pose_from_optimizer_vars,
        required_ap_deg,
        shank_capsules_from_pose,
    )
    from aind_rutter.optimization.geometry.probes import ProbeStaticInfo
    from aind_rutter.optimization.geometry.recording import (
        RECORDING_GEOMETRY,
        RecordingGeometry,
        get_recording_geometry,
    )
    from aind_rutter.optimization.objectives.density import (
        DensityFn,
        coverage,
        gaussian_density,
        gaussian_mixture_density,
        integrate_density_along_shank,
        voxel_kde_density,
    )
    from aind_rutter.optimization.objectives.probe_static import JointWeights

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
    "cap_basis",
    "capsule_capsule_dist",
    "coverage",
    "find_hole_by_id",
    "gaussian_density",
    "gaussian_mixture_density",
    "get_recording_geometry",
    "integrate_density_along_shank",
    "load_holes",
    "make_fcl_bvh",
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
