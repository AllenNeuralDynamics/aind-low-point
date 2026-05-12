"""Runtime package — turns a validated ConfigModel into a RuntimeBundle.

Split into focused submodules:

- transforms : compile config TransformRecipes / TransformRefs to
  AffineTransforms and chains.
- loaders    : registry-driven file loaders (trimesh, sitk_volume,
  csv_points, …) and the GeometryOut type.
- reducers   : registry-driven point reducers
  (mesh_centroid / mesh_center_mass / hemisphere_center_mass).
- canonicalize : apply orientation flip + scale + optional transform to
  bring a loaded asset into canonical LPS mm.
- chem_shift : MRI chemical-shift correction context.
- calibration: probe calibration bank loading.
- build      : the main ``build_runtime_from_config`` orchestrator and
  the ``RuntimeBundle`` container.
- export     : ``planning_state_to_plan_model`` + ``save_plan_to_config``
  (full round-trip) and ``export_plan_geometry`` (slim per-probe summary
  for experimenters); also ``_depth_along_probe_axis`` shared with the
  trame readout code.

The flat ``aind_low_point.build_runtime`` module re-exports everything
here for backward compatibility.
"""

from aind_low_point.runtime.build import (
    CollisionLabelIndex,
    RuntimeBundle,
    _base_spec_kwargs_from_model,
    _capabilities_from_list,
    _collision_bits,
    _compile_collision_labels,
    _material_from_model,
    _resolve_scene_node_transform,
    build_asset_spec,
    build_runtime_from_config,
    build_target_spec,
    load_resource,
    resolve_material_for_spec,
)
from aind_low_point.runtime.calibration import (
    _get_calibration_rt,
    _load_calibration_bank,
)
from aind_low_point.runtime.canonicalize import (
    CanonicalizationRuntime,
    _apply_canonicalization_mesh,
    _apply_canonicalization_points,
    _canon_runtime_from_model,
    _resolve_canon_model_to_runtime,
    _resolve_canonicalization_model,
)
from aind_low_point.runtime.chem_shift import ChemShiftContext, _should_apply_chem
from aind_low_point.runtime.export import (
    _depth_along_probe_axis,
    _reconstruct_target_ref,
    apply_plan_model_to_state,
    export_plan_geometry,
    planning_state_to_plan_model,
    save_plan_to_config,
)
from aind_low_point.runtime.loaders import (
    _GEOMETRY_LOADER_REGISTRY,
    GeometryOut,
    csv_points,
    load_geometry,
    load_trimesh_lps,
    register_loader,
    register_loader_fn,
    sitk_volume,
    trimesh_from_sitk_mask,
)
from aind_low_point.runtime.shanks import detect_shank_tips_local
from aind_low_point.runtime.reducers import (
    _REDUCER_REGISTRY,
    ReduceOut,
    SourceGeo,
    hemisphere_center_mass,
    mesh_center_mass,
    mesh_centroid,
    reduce_target,
    register_reducer,
    register_reducer_fn,
)
from aind_low_point.runtime.transforms import (
    CompiledTransforms,
    _op_to_affine,
    compile_all_transforms,
    compile_recipe_to_chain,
    resolve_transform_key_cached,
    resolve_transform_ref_cached,
)

__all__ = [
    # build
    "CollisionLabelIndex",
    "RuntimeBundle",
    "build_asset_spec",
    "build_runtime_from_config",
    "build_target_spec",
    "load_resource",
    "resolve_material_for_spec",
    # transforms
    "CompiledTransforms",
    "compile_all_transforms",
    "compile_recipe_to_chain",
    "resolve_transform_key_cached",
    "resolve_transform_ref_cached",
    # loaders
    "GeometryOut",
    "csv_points",
    "load_geometry",
    "load_trimesh_lps",
    "register_loader",
    "register_loader_fn",
    "sitk_volume",
    "trimesh_from_sitk_mask",
    # shanks
    "detect_shank_tips_local",
    # reducers
    "ReduceOut",
    "SourceGeo",
    "hemisphere_center_mass",
    "mesh_center_mass",
    "mesh_centroid",
    "reduce_target",
    "register_reducer",
    "register_reducer_fn",
    # canonicalize
    "CanonicalizationRuntime",
    # chem_shift
    "ChemShiftContext",
    # export
    "apply_plan_model_to_state",
    "export_plan_geometry",
    "planning_state_to_plan_model",
    "save_plan_to_config",
]
