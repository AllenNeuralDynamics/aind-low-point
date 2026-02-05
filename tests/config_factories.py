"""Factory classes for generating test configuration data."""

from pathlib import Path
from typing import Any, Dict, List

from aind_low_point.common import Capability, Kind, Role


class ConfigFactory:
    """Factory for creating test configuration data."""

    @staticmethod
    def minimal_config(**overrides) -> Dict[str, Any]:
        """Create minimal valid configuration."""
        config = {
            "version": 1,
            "assets": [],
            "targets": [],
            "scene": {"nodes": []},
            "plan": {
                "arcs": {},
                "probes": {},
                "reticles": {},
                "calibrations": {"files": {}, "probe_to_ref": {}},
            },
            "transforms": {},
            "canonicalizations": {},
        }
        config.update(overrides)
        return config

    @staticmethod
    def config_with_basic_assets(**overrides) -> Dict[str, Any]:
        """Create config with basic assets and targets."""
        config = ConfigFactory.minimal_config()
        config.update(
            {
                "assets": [
                    AssetFactory.mesh_asset(key="brain_mesh"),
                    AssetFactory.points_asset(key="landmarks"),
                ],
                "targets": [
                    TargetFactory.explicit_target(key="target1"),
                ],
                "scene": {
                    "nodes": [
                        {"key": "brain_node", "asset": "brain_mesh"},
                    ]
                },
            }
        )
        config.update(overrides)
        return config

    @staticmethod
    def config_with_probes(**overrides) -> Dict[str, Any]:
        """Create config with probes and planning setup."""
        config = ConfigFactory.config_with_basic_assets()
        config.update(
            {
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "target1",
                            "slider_ml": 5.0,
                            "spin": 0.0,
                            "past_target_mm": 2.0,
                            "offsets_RA": [0.0, 0.0],
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                }
            }
        )
        config.update(overrides)
        return config

    @staticmethod
    def config_with_materials(**overrides) -> Dict[str, Any]:
        """Create config with materials bank."""
        config = ConfigFactory.minimal_config()
        config.update(
            {
                "materials": {
                    "default_material": MaterialFactory.material(),
                    "red_material": MaterialFactory.red_material(),
                    "green_material": MaterialFactory.green_material(),
                    "transparent_material": MaterialFactory.transparent_material(),
                },
            }
        )
        config.update(overrides)
        return config

    @staticmethod
    def config_with_templates(**overrides) -> Dict[str, Any]:
        """Create config with materials and templates."""
        config = ConfigFactory.config_with_materials()
        config.update(
            {
                "asset_templates": {
                    "mesh_template": TemplateFactory.mesh_template_with_loader(
                        material_ref="default_material"
                    ),
                    "transparent_mesh": TemplateFactory.asset_template(
                        name="transparent_mesh",
                        material_ref="transparent_material",
                        kind=Kind.MESH.value,
                    ),
                },
                "target_templates": {
                    "explicit_template": TemplateFactory.points_template_explicit(
                        material_ref="green_material"
                    ),
                    "derived_template": TemplateFactory.target_template_derived(
                        material_ref="red_material"
                    ),
                },
            }
        )
        config.update(overrides)
        return config

    @staticmethod
    def config_with_templated_assets(**overrides) -> Dict[str, Any]:
        """Create config with assets that use templates."""
        config = ConfigFactory.config_with_templates()
        config.update(
            {
                "assets": [
                    AssetFactory.base_asset(
                        key="brain_mesh",
                        kind=Kind.MESH.value,
                        templates=["mesh_template"],
                        src=Path("/data/brain.obj"),
                        loader="trimesh_loader",
                    ),
                    AssetFactory.base_asset(
                        key="skull_mesh",
                        kind=Kind.MESH.value,
                        templates=["transparent_mesh"],
                        src=Path("/data/skull.obj"),
                        loader="trimesh_loader",
                    ),
                ],
                "targets": [
                    TargetFactory.base_target(
                        key="target1",
                        templates=["explicit_template"],
                        src=Path("/data/targets.npy"),
                        loader="numpy_points",
                    ),
                    TargetFactory.derived_target(
                        key="target2",
                        templates=["derived_template"],
                        source_key="brain_mesh",
                    ),
                ],
            }
        )
        config.update(overrides)
        return config


class AssetFactory:
    """Factory for creating asset specifications."""

    @staticmethod
    def base_asset(key: str = "test_asset", **overrides) -> Dict[str, Any]:
        """Create base asset specification."""
        asset = {
            "key": key,
            "kind": Kind.MESH.value,
            "role": Role.GEOMETRY.value,
            "src": Path("/path/to/asset.obj"),
            "loader": "trimesh_loader",
            "caps": [Capability.RENDERABLE.value],
            "templates": [],
        }
        asset.update(overrides)
        return asset

    @staticmethod
    def mesh_asset(key: str = "mesh_asset", **overrides) -> Dict[str, Any]:
        """Create mesh asset specification."""
        return AssetFactory.base_asset(
            key=key,
            kind=Kind.MESH.value,
            src=Path("/path/to/mesh.obj"),
            loader="trimesh_loader",
            **overrides,
        )

    @staticmethod
    def points_asset(key: str = "points_asset", **overrides) -> Dict[str, Any]:
        """Create points asset specification."""
        return AssetFactory.base_asset(
            key=key,
            kind=Kind.POINTS.value,
            src=Path("/path/to/points.npy"),
            loader="numpy_points",
            **overrides,
        )

    @staticmethod
    def asset_with_canonicalization(
        key: str = "canon_asset", **overrides
    ) -> Dict[str, Any]:
        """Create asset with canonicalization."""
        asset = AssetFactory.base_asset(key=key)
        asset.update(
            {
                "canonicalization": {
                    "source_space": "RAS",
                    "scale_to_mm": 1.0,
                    "version": "canon-v1",
                }
            }
        )
        asset.update(overrides)
        return asset

    @staticmethod
    def asset_with_material_ref(
        key: str = "material_ref_asset",
        material_ref: str = "default_material",
        **overrides,
    ) -> Dict[str, Any]:
        """Create asset with material reference."""
        return AssetFactory.base_asset(key=key, material_ref=material_ref, **overrides)

    @staticmethod
    def asset_with_templates(
        key: str = "templated_asset", templates: List[str] = None, **overrides
    ) -> Dict[str, Any]:
        """Create asset with templates."""
        if templates is None:
            templates = ["mesh_template"]
        return AssetFactory.base_asset(key=key, templates=templates, **overrides)


class TargetFactory:
    """Factory for creating target specifications."""

    @staticmethod
    def base_target(key: str = "test_target", **overrides) -> Dict[str, Any]:
        """Create base target specification."""
        target = {
            "key": key,
            "kind": Kind.POINTS.value,
            "role": Role.TARGET.value,
            "caps": [Capability.RENDERABLE.value],
            "templates": [],
        }
        target.update(overrides)
        return target

    @staticmethod
    def explicit_target(key: str = "explicit_target", **overrides) -> Dict[str, Any]:
        """Create explicit target with source file."""
        return TargetFactory.base_target(
            key=key,
            src=Path("/path/to/targets.npy"),
            loader="numpy_points",
            **overrides,
        )

    @staticmethod
    def derived_target(
        key: str = "derived_target", source_key: str = "brain_mesh", **overrides
    ) -> Dict[str, Any]:
        """Create target derived from asset."""
        return TargetFactory.base_target(
            key=key, source_key=source_key, reducer="centroid", **overrides
        )

    @staticmethod
    def target_with_material_ref(
        key: str = "material_ref_target",
        material_ref: str = "green_material",
        **overrides,
    ) -> Dict[str, Any]:
        """Create target with material reference."""
        return TargetFactory.explicit_target(
            key=key, material_ref=material_ref, **overrides
        )

    @staticmethod
    def target_with_templates(
        key: str = "templated_target", templates: List[str] = None, **overrides
    ) -> Dict[str, Any]:
        """Create target with templates."""
        if templates is None:
            templates = ["explicit_template"]
        return TargetFactory.explicit_target(key=key, templates=templates, **overrides)


class TransformFactory:
    """Factory for creating transform specifications."""

    @staticmethod
    def translate_op(delta: List[float] = None, **overrides) -> Dict[str, Any]:
        """Create translation transform operation."""
        if delta is None:
            delta = [1.0, 2.0, 3.0]
        op = {
            "kind": "translate_mm",
            "delta": delta,
            "invert": False,
        }
        op.update(overrides)
        return op

    @staticmethod
    def rotate_op(
        angles_deg: List[float] = None, order: str = "ZYX", **overrides
    ) -> Dict[str, Any]:
        """Create rotation transform operation."""
        if angles_deg is None:
            angles_deg = [0.0, 0.0, 90.0]
        op = {
            "kind": "rotate_euler_deg",
            "order": order,
            "angles_deg": angles_deg,
            "invert": False,
        }
        op.update(overrides)
        return op

    @staticmethod
    def sitk_op(path: str, **overrides) -> Dict[str, Any]:
        """Create SITK file transform operation."""
        op = {
            "kind": "sitk_file",
            "path": path,
            "invert": False,
        }
        op.update(overrides)
        return op

    @staticmethod
    def transform_recipe(
        sequence: List[Dict[str, Any]] = None, **overrides
    ) -> Dict[str, Any]:
        """Create transform recipe."""
        if sequence is None:
            sequence = [TransformFactory.translate_op()]
        recipe = {"sequence": sequence}
        recipe.update(overrides)
        return recipe

    @staticmethod
    def transform_ref_key(key: str, **overrides) -> Dict[str, Any]:
        """Create transform reference by key."""
        ref = {"key": key}
        ref.update(overrides)
        return ref

    @staticmethod
    def transform_ref_inline(
        sequence: List[Dict[str, Any]] = None, **overrides
    ) -> Dict[str, Any]:
        """Create inline transform reference."""
        ref = {"inline": TransformFactory.transform_recipe(sequence)}
        ref.update(overrides)
        return ref


class CalibrationFactory:
    """Factory for creating calibration specifications."""

    @staticmethod
    def calibration_source_file(file_path: str, **overrides) -> Dict[str, Any]:
        """Create file-based calibration source."""
        # Note: In real usage, file_path would be validated as existing FilePath
        # For tests, we use string that would need actual file creation
        source = {"file": file_path}
        source.update(overrides)
        return source

    @staticmethod
    def calibration_source_dir(
        dir_path: str, reticle: str = "default_reticle", **overrides
    ) -> Dict[str, Any]:
        """Create directory-based calibration source."""
        source = {"directory": dir_path, "reticle": reticle}
        source.update(overrides)
        return source

    @staticmethod
    def calibration_ref(cal_id: str, probe_code: str, **overrides) -> Dict[str, Any]:
        """Create calibration reference."""
        ref = {"cal_id": cal_id, "probe_code": probe_code}
        ref.update(overrides)
        return ref


class SelectorFactory:
    """Factory for creating selector specifications."""

    @staticmethod
    def name_selector(name: str, **overrides) -> Dict[str, Any]:
        """Create name-based selector."""
        selector = {"kind": "name", "name": name}
        selector.update(overrides)
        return selector

    @staticmethod
    def index_selector(index: int, **overrides) -> Dict[str, Any]:
        """Create index-based selector."""
        selector = {"kind": "index", "index": index}
        selector.update(overrides)
        return selector

    @staticmethod
    def path_selector(path: str, **overrides) -> Dict[str, Any]:
        """Create path-based selector."""
        selector = {"kind": "path", "path": path}
        selector.update(overrides)
        return selector

    @staticmethod
    def label_selector(label, **overrides) -> Dict[str, Any]:
        """Create label-based selector."""
        selector = {"kind": "label", "label": label}
        selector.update(overrides)
        return selector


class MaterialFactory:
    """Factory for creating material specifications."""

    @staticmethod
    def material(
        name: str = "default",
        color: str = "#C8C8C8",
        opacity: float = 1.0,
        wireframe: bool = False,
        visible: bool = True,
        **overrides,
    ) -> Dict[str, Any]:
        """Create material specification."""
        material = {
            "name": name,
            "color": color,
            "opacity": opacity,
            "wireframe": wireframe,
            "visible": visible,
        }
        material.update(overrides)
        return material

    @staticmethod
    def red_material(**overrides) -> Dict[str, Any]:
        """Create red material."""
        return MaterialFactory.material(
            name="red_material", color="#FF0000", **overrides
        )

    @staticmethod
    def green_material(**overrides) -> Dict[str, Any]:
        """Create green material."""
        return MaterialFactory.material(
            name="green_material", color="#00FF00", **overrides
        )

    @staticmethod
    def transparent_material(**overrides) -> Dict[str, Any]:
        """Create transparent material."""
        return MaterialFactory.material(
            name="transparent_material", opacity=0.5, **overrides
        )


class TemplateFactory:
    """Factory for creating template specifications."""

    @staticmethod
    def base_template(
        name: str = "base_template", kind: str = None, role: str = None, **overrides
    ) -> Dict[str, Any]:
        """Create base template specification."""
        template = {"name": name}
        if kind:
            template["kind"] = kind
        if role:
            template["role"] = role
        template.update(overrides)
        return template

    @staticmethod
    def asset_template(
        name: str = "asset_template", material_ref: str = None, **overrides
    ) -> Dict[str, Any]:
        """Create asset template specification."""
        template = TemplateFactory.base_template(
            name=name,
            kind=Kind.MESH.value,
            role=Role.GEOMETRY.value,
        )
        if material_ref:
            template["material_ref"] = material_ref
        template.update(overrides)
        return template

    @staticmethod
    def target_template(
        name: str = "target_template", material_ref: str = None, **overrides
    ) -> Dict[str, Any]:
        """Create target template specification."""
        template = TemplateFactory.base_template(
            name=name,
            kind=Kind.POINTS.value,
            role=Role.TARGET.value,
        )
        if material_ref:
            template["material_ref"] = material_ref
        template.update(overrides)
        return template

    @staticmethod
    def mesh_template_with_loader(
        name: str = "mesh_template",
        src: str = "/template/mesh.obj",
        loader: str = "trimesh_loader",
        **overrides,
    ) -> Dict[str, Any]:
        """Create mesh template with source loader."""
        return TemplateFactory.asset_template(
            name=name, src=Path(src), loader=loader, **overrides
        )

    @staticmethod
    def points_template_explicit(
        name: str = "points_template",
        src: str = "/template/points.npy",
        loader: str = "numpy_points",
        **overrides,
    ) -> Dict[str, Any]:
        """Create points template with explicit source."""
        return TemplateFactory.target_template(
            name=name, src=Path(src), loader=loader, **overrides
        )

    @staticmethod
    def target_template_derived(
        name: str = "derived_template",
        source_key: str = "brain_mesh",
        reducer: str = "centroid",
        **overrides,
    ) -> Dict[str, Any]:
        """Create derived target template."""
        return TemplateFactory.target_template(
            name=name, source_key=source_key, reducer=reducer, **overrides
        )
