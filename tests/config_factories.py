"""Factory classes for generating test configuration data."""

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
        config.update({
            "assets": [
                AssetFactory.mesh_asset(key="brain_mesh"),
                AssetFactory.points_asset(key="landmarks"),
            ],
            "targets": [
                TargetFactory.explicit_target(key="target1"),
            ],
            "scene": {
                "nodes": [
                    {"id": "brain_node", "asset": "brain_mesh"},
                ]
            }
        })
        config.update(overrides)
        return config

    @staticmethod
    def config_with_probes(**overrides) -> Dict[str, Any]:
        """Create config with probes and planning setup."""
        config = ConfigFactory.config_with_basic_assets()
        config.update({
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
        })
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
            "src": "/path/to/asset.obj",
            "loader": "trimesh_loader",
            "caps": [Capability.RENDERABLE.value],
        }
        asset.update(overrides)
        return asset

    @staticmethod
    def mesh_asset(key: str = "mesh_asset", **overrides) -> Dict[str, Any]:
        """Create mesh asset specification."""
        return AssetFactory.base_asset(
            key=key,
            kind=Kind.MESH.value,
            src="/path/to/mesh.obj",
            loader="trimesh_loader",
            **overrides
        )

    @staticmethod
    def points_asset(key: str = "points_asset", **overrides) -> Dict[str, Any]:
        """Create points asset specification."""
        return AssetFactory.base_asset(
            key=key,
            kind=Kind.POINTS.value,
            src="/path/to/points.npy",
            loader="numpy_points",
            **overrides
        )

    @staticmethod
    def asset_with_canonicalization(key: str = "canon_asset", **overrides) -> Dict[str, Any]:
        """Create asset with canonicalization."""
        asset = AssetFactory.base_asset(key=key)
        asset.update({
            "canonicalization": {
                "source_space": "RAS",
                "scale_to_mm": 1.0,
                "version": "canon-v1",
            }
        })
        asset.update(overrides)
        return asset


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
        }
        target.update(overrides)
        return target

    @staticmethod
    def explicit_target(key: str = "explicit_target", **overrides) -> Dict[str, Any]:
        """Create explicit target with source file."""
        return TargetFactory.base_target(
            key=key,
            src="/path/to/targets.npy",
            loader="numpy_points",
            **overrides
        )

    @staticmethod
    def derived_target(key: str = "derived_target", source_key: str = "brain_mesh", **overrides) -> Dict[str, Any]:
        """Create target derived from asset."""
        return TargetFactory.base_target(
            key=key,
            source_key=source_key,
            reducer="centroid",
            **overrides
        )


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
    def rotate_op(angles_deg: List[float] = None, order: str = "ZYX", **overrides) -> Dict[str, Any]:
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
    def transform_recipe(sequence: List[Dict[str, Any]] = None, **overrides) -> Dict[str, Any]:
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
    def transform_ref_inline(sequence: List[Dict[str, Any]] = None, **overrides) -> Dict[str, Any]:
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
    def calibration_source_dir(dir_path: str, reticle: str = "default_reticle", **overrides) -> Dict[str, Any]:
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
