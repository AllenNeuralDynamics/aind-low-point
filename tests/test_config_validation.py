"""Tests for cross-reference validation in ConfigModel._xref method."""

import pytest
from pydantic import ValidationError

from aind_low_point.config import ConfigModel
from tests.config_factories import (
    AssetFactory,
    CalibrationFactory,
    ConfigFactory,
    TargetFactory,
    TransformFactory,
)


class TestConfigCrossReferenceValidation:
    """Test the comprehensive cross-reference validation in ConfigModel._xref."""

    def test_minimal_valid_config(self):
        """Test minimal valid configuration passes validation."""
        config_data = ConfigFactory.minimal_config()
        config = ConfigModel.model_validate(config_data)

        assert config.version == 1
        assert len(config.assets) == 0
        assert len(config.targets) == 0

    def test_scene_asset_reference_validation(self):
        """Test scene nodes must reference existing assets."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [AssetFactory.mesh_asset(key="brain_mesh")],
                "scene": {
                    "nodes": [
                        {"key": "valid_node", "asset": "brain_mesh"},
                        {"key": "invalid_node", "asset": "nonexistent_asset"},
                    ]
                },
            }
        )

        with pytest.raises(
            ValidationError, match="asset 'nonexistent_asset' not found in catalog"
        ):
            ConfigModel.model_validate(config_data)

    def test_scene_transform_reference_validation(self):
        """Test scene node transform references must exist."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [AssetFactory.mesh_asset(key="brain_mesh")],
                "transforms": {"valid_transform": TransformFactory.transform_recipe()},
                "scene": {
                    "nodes": [
                        {
                            "key": "valid_node",
                            "asset": "brain_mesh",
                            "transform": {"key": "nonexistent_transform"},
                        }
                    ]
                },
            }
        )

        with pytest.raises(
            ValidationError, match="transform key 'nonexistent_transform' not found"
        ):
            ConfigModel.model_validate(config_data)

    def test_scene_pose_source_probe_validation(self):
        """Test scene node pose_source_probe must reference existing probe."""
        config_data = ConfigFactory.config_with_probes()
        config_data["scene"]["nodes"].append(
            {
                "key": "probe_node",
                "asset": "brain_mesh",
                "pose_source_probe": "nonexistent_probe",
            }
        )

        with pytest.raises(
            ValidationError,
            match="pose_source_probe 'nonexistent_probe' not in plan.probes",
        ):
            ConfigModel.model_validate(config_data)

    def test_target_source_key_validation(self):
        """Test derived targets must reference existing assets."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [AssetFactory.mesh_asset(key="brain_mesh")],
                "targets": [
                    TargetFactory.derived_target(
                        key="valid_target", source_key="brain_mesh"
                    ),
                    TargetFactory.derived_target(
                        key="invalid_target", source_key="nonexistent_asset"
                    ),
                ],
            }
        )

        with pytest.raises(
            ValidationError, match="source_key 'nonexistent_asset' not found in assets"
        ):
            ConfigModel.model_validate(config_data)

    def test_probe_arc_reference_validation(self):
        """Test probes must reference existing arcs."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "targets": [TargetFactory.explicit_target(key="target1")],
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "valid_probe": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "target1",
                        },
                        "invalid_probe": {
                            "kind": "neuropixels",
                            "arc": "nonexistent_arc",
                            "target": "target1",
                        },
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )

        with pytest.raises(
            ValidationError, match="arc 'nonexistent_arc' not found in plan.arcs"
        ):
            ConfigModel.model_validate(config_data)

    def test_probe_target_reference_validation(self):
        """Test probes must reference existing targets."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "targets": [TargetFactory.explicit_target(key="target1")],
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "valid_probe": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "target1",
                        },
                        "invalid_probe": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "nonexistent_target",
                        },
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )

        with pytest.raises(
            ValidationError, match="catalog target 'nonexistent_target' not found"
        ):
            ConfigModel.model_validate(config_data)

    def test_calibration_file_reticle_validation(self, temp_dir_path):
        """Test calibration files must reference existing reticles."""
        # Create actual directories for the test
        cal_dir1 = temp_dir_path / "cal1"
        cal_dir2 = temp_dir_path / "cal2"
        cal_dir1.mkdir(parents=True)
        cal_dir2.mkdir(parents=True)

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "plan": {
                    "arcs": {},
                    "probes": {},
                    "reticles": {
                        "reticle1": {"offset_RAS": [0.0, 0.0, 0.0], "rotation_z": 0.0}
                    },
                    "calibrations": {
                        "files": {
                            "cal1": CalibrationFactory.calibration_source_dir(
                                str(cal_dir1), reticle="reticle1"
                            ),
                            "cal2": CalibrationFactory.calibration_source_dir(
                                str(cal_dir2), reticle="nonexistent_reticle"
                            ),
                        },
                        "probe_to_ref": {},
                    },
                },
            }
        )

        with pytest.raises(
            ValidationError,
            match="reticle 'nonexistent_reticle' not in plan.reticles",
        ):
            ConfigModel.model_validate(config_data)

    def test_calibration_probe_to_ref_validation(self, temp_dir_path):
        """Test calibration probe_to_ref must reference existing probes and cal
        files."""
        cal_dir = temp_dir_path / "cal"
        cal_dir.mkdir(parents=True)

        config_data = ConfigFactory.config_with_probes()
        config_data["plan"]["reticles"] = {"reticle1": {"offset_RAS": [0.0, 0.0, 0.0]}}
        config_data["plan"]["calibrations"] = {
            "files": {
                "cal1": CalibrationFactory.calibration_source_dir(
                    str(cal_dir), reticle="reticle1"
                )
            },
            "probe_to_ref": {
                "probe1": "cal1:12345",  # Valid: probe exists, cal file exists
                "nonexistent_probe": "cal1:67890",  # Invalid: probe doesn't exist
                # Invalid: cal file doesn't exist
                "probe1_dup": "nonexistent_cal:12345",
            },
        }

        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "probe 'nonexistent_probe' not in plan.probes" in error_msg
        assert "cal_id 'nonexistent_cal' not in files" in error_msg

    def test_canonicalization_transform_reference_validation(self):
        """Test canonicalization definitions must reference valid transforms."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "transforms": {"valid_transform": TransformFactory.transform_recipe()},
                "canonicalizations": {
                    "canon1": {
                        "source_space": "RAS",
                        "scale_to_mm": 1.0,
                        "transform": {"key": "valid_transform"},
                    },
                    "canon2": {
                        "source_space": "LPS",
                        "scale_to_mm": 1.0,
                        "transform": {"key": "nonexistent_transform"},
                    },
                },
            }
        )

        with pytest.raises(
            ValidationError, match="transform key 'nonexistent_transform' not found"
        ):
            ConfigModel.model_validate(config_data)

    def test_canonicalization_ref_validation(self):
        """Test canonicalization_ref fields must reference existing
        canonicalizations."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "canonicalizations": {
                    "canon1": {"source_space": "RAS", "scale_to_mm": 1.0}
                },
                "assets": [
                    AssetFactory.mesh_asset(
                        key="valid_asset", canonicalization_ref="canon1"
                    ),
                    AssetFactory.mesh_asset(
                        key="invalid_asset", canonicalization_ref="nonexistent_canon"
                    ),
                ],
            }
        )

        with pytest.raises(
            ValidationError, match="canonicalization_ref 'nonexistent_canon' not found"
        ):
            ConfigModel.model_validate(config_data)

    def test_file_native_requires_transform(self):
        """Test FILE_NATIVE source_space requires a transform."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "canonicalizations": {
                    "invalid_canon": {
                        "source_space": "FILE_NATIVE",
                        "scale_to_mm": 1.0,
                        # Missing required transform
                    }
                }
            }
        )

        with pytest.raises(ValidationError, match="FILE_NATIVE requires a transform"):
            ConfigModel.model_validate(config_data)

    def test_multiple_validation_errors_reported(self):
        """Test that multiple validation errors are collected and reported."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "scene": {
                    "nodes": [
                        {"key": "node1", "asset": "nonexistent_asset1"},
                        {"key": "node2", "asset": "nonexistent_asset2"},
                    ]
                },
                "plan": {
                    "arcs": {},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "nonexistent_arc",
                            "target": "nonexistent_target",
                        }
                    },
                    "reticles": {},
                    "calibrations": {"files": {}, "probe_to_ref": {}},
                },
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(config_data)

        error_msg = str(exc_info.value)
        # Should contain multiple error messages
        assert "nonexistent_asset1" in error_msg
        assert "nonexistent_asset2" in error_msg
        assert "nonexistent_arc" in error_msg
        assert "nonexistent_target" in error_msg

    def test_valid_complete_config(self, sample_transform_file, temp_dir_path):
        """Test complete valid configuration with all features passes validation."""
        cal_dir = temp_dir_path / "cal"
        cal_dir.mkdir(parents=True)

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "transforms": {
                    "transform1": TransformFactory.transform_recipe(
                        [
                            TransformFactory.translate_op(),
                            TransformFactory.sitk_op(str(sample_transform_file)),
                        ]
                    )
                },
                "canonicalizations": {
                    "canon1": {
                        "source_space": "FILE_NATIVE",
                        "scale_to_mm": 1.0,
                        "transform": {"key": "transform1"},
                    }
                },
                "assets": [
                    AssetFactory.mesh_asset(
                        key="brain_mesh", canonicalization_ref="canon1"
                    )
                ],
                "targets": [
                    TargetFactory.explicit_target(key="target1"),
                    TargetFactory.derived_target(
                        key="target2", source_key="brain_mesh"
                    ),
                ],
                "scene": {
                    "nodes": [
                        {
                            "key": "brain_node",
                            "asset": "brain_mesh",
                            "transform": {"key": "transform1"},
                        }
                    ]
                },
                "plan": {
                    "arcs": {"arc1": 15.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "target1",
                            "auto_scene": False,  # no probe:neuropixels asset
                        }
                    },
                    "reticles": {"reticle1": {"offset_RAS": [0.0, 0.0, 0.0]}},
                    "calibrations": {
                        "files": {
                            "cal1": CalibrationFactory.calibration_source_dir(
                                str(cal_dir), reticle="reticle1"
                            )
                        },
                        "probe_to_ref": {"probe1": "cal1:12345"},
                    },
                },
            }
        )

        # Should not raise any validation errors
        config = ConfigModel.model_validate(config_data)
        assert config.version == 1
        assert len(config.assets) == 1
        assert len(config.targets) == 2
        assert len(config.scene.nodes) == 1
        assert len(config.plan.probes) == 1


class TestMaterialReferenceValidation:
    """Test cross-reference validation for material_ref fields."""

    def test_asset_material_ref_validation(self):
        """Test assets must reference existing materials."""
        from tests.config_factories import ConfigFactory, AssetFactory

        config_data = ConfigFactory.config_with_materials()
        config_data["assets"] = [
            AssetFactory.asset_with_material_ref(
                key="valid_asset", material_ref="default_material"
            ),
            AssetFactory.asset_with_material_ref(
                key="invalid_asset", material_ref="nonexistent_material"
            ),
        ]

        with pytest.raises(
            ValidationError,
            match="material_ref 'nonexistent_material' not found",
        ):
            ConfigModel.model_validate(config_data)

    def test_target_material_ref_validation(self):
        """Test targets must reference existing materials."""
        from tests.config_factories import ConfigFactory, TargetFactory

        config_data = ConfigFactory.config_with_materials()
        config_data["targets"] = [
            TargetFactory.target_with_material_ref(
                key="valid_target", material_ref="green_material"
            ),
            TargetFactory.target_with_material_ref(
                key="invalid_target", material_ref="missing_material"
            ),
        ]

        with pytest.raises(
            ValidationError,
            match="material_ref 'missing_material' not found",
        ):
            ConfigModel.model_validate(config_data)

    def test_template_material_ref_validation(self):
        """Test templates must reference existing materials."""
        from tests.config_factories import ConfigFactory, TemplateFactory

        config_data = ConfigFactory.config_with_materials()
        config_data.update(
            {
                "asset_templates": {
                    "valid_template": TemplateFactory.asset_template(
                        material_ref="default_material"
                    ),
                    "invalid_template": TemplateFactory.asset_template(
                        material_ref="missing_material"
                    ),
                }
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "material_ref 'missing_material' not found" in error_msg


class TestTemplateReferenceValidation:
    """Test cross-reference validation for template fields."""

    def test_asset_template_references(self):
        """Test assets.templates must reference existing asset_templates."""
        from tests.config_factories import ConfigFactory, AssetFactory

        config_data = ConfigFactory.config_with_templates()
        config_data["assets"] = [
            AssetFactory.asset_with_templates(
                key="valid_asset", templates=["mesh_template"]
            ),
            AssetFactory.asset_with_templates(
                key="invalid_asset", templates=["nonexistent_template"]
            ),
        ]

        with pytest.raises(
            ValidationError,
            match="template 'nonexistent_template' not found",
        ):
            ConfigModel.model_validate(config_data)

    def test_target_template_references(self):
        """Test targets.templates must reference existing target_templates."""
        from tests.config_factories import ConfigFactory, TargetFactory

        config_data = ConfigFactory.config_with_templates()
        config_data["targets"] = [
            TargetFactory.target_with_templates(
                key="valid_target", templates=["explicit_template"]
            ),
            TargetFactory.target_with_templates(
                key="invalid_target", templates=["missing_template"]
            ),
        ]

        with pytest.raises(
            ValidationError, match="template 'missing_template' not found"
        ):
            ConfigModel.model_validate(config_data)

    def test_multiple_template_references(self):
        """Test validation of multiple template references."""
        from tests.config_factories import ConfigFactory, AssetFactory

        config_data = ConfigFactory.config_with_templates()
        config_data["assets"] = [
            AssetFactory.asset_with_templates(
                key="multi_template_asset",
                templates=["mesh_template", "nonexistent_template"],
            ),
        ]

        with pytest.raises(
            ValidationError,
            match="template 'nonexistent_template' not found",
        ):
            ConfigModel.model_validate(config_data)


class TestTemplateExpansionValidation:
    """Test template expansion and validation integration."""

    def test_template_expansion_occurs_before_validation(self):
        """Test template expansion happens in _xref_and_expand_templates."""
        from tests.config_factories import ConfigFactory

        # This should pass validation because templates are expanded first
        config_data = ConfigFactory.config_with_templated_assets()

        # Should not raise validation errors
        config = ConfigModel.model_validate(config_data)
        assert len(config.assets) == 2
        assert len(config.targets) == 2

    def test_expanded_assets_have_material_refs_validated(self):
        """Test that after template expansion, material_ref validation occurs."""
        from tests.config_factories import ConfigFactory, AssetFactory, TemplateFactory

        config_data = ConfigFactory.config_with_materials()
        config_data.update(
            {
                "asset_templates": {
                    "bad_template": TemplateFactory.asset_template(
                        material_ref="nonexistent_material"
                    )
                },
                "assets": [
                    AssetFactory.asset_with_templates(
                        key="templated_asset", templates=["bad_template"]
                    )
                ],
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "material_ref 'nonexistent_material' not found" in error_msg

    def test_valid_complete_config_with_templates(self):
        """Test complete valid configuration with templates passes validation."""
        from tests.config_factories import ConfigFactory

        config_data = ConfigFactory.config_with_templated_assets()

        # Should not raise any validation errors
        config = ConfigModel.model_validate(config_data)
        assert config.version == 1
        assert len(config.materials) == 4  # default, red, green, transparent
        assert len(config.asset_templates) == 2
        assert len(config.target_templates) == 2
        assert len(config.assets) == 2
        assert len(config.targets) == 2


class TestBulkAssetSpec:
    """Test BulkAssetSpecModel expansion."""

    def test_bulk_asset_expansion(self):
        """Test bulk assets expand into individual AssetSpecModels."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    {
                        "keys": ["structure:A", "structure:B", "structure:C"],
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/path/to/{name}.obj",
                    }
                ],
            }
        )

        config = ConfigModel.model_validate(config_data)
        assert len(config.assets) == 3
        assert config.assets[0].key == "structure:A"
        assert config.assets[1].key == "structure:B"
        assert config.assets[2].key == "structure:C"
        # Check placeholder substitution
        assert str(config.assets[0].src).endswith("/A.obj")
        assert str(config.assets[1].src).endswith("/B.obj")

    def test_bulk_asset_with_key_placeholder(self):
        """Test bulk assets substitute {key} placeholder correctly."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    {
                        "keys": ["brain", "skull"],
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/models/{key}.stl",
                    }
                ],
            }
        )

        config = ConfigModel.model_validate(config_data)
        assert len(config.assets) == 2
        assert str(config.assets[0].src).endswith("/brain.stl")
        assert str(config.assets[1].src).endswith("/skull.stl")


class TestRangeTargetSpec:
    """Test RangeTargetSpecModel expansion."""

    def test_range_target_expansion(self):
        """Test range targets expand into individual TargetSpecModels."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "targets": [
                    {
                        "key_pattern": "target:hole:{n}",
                        "range": [1, 3],
                        "loader": "numpy_points",
                        "src": "/holes/hole{n}.npy",
                    }
                ],
            }
        )

        config = ConfigModel.model_validate(config_data)
        assert len(config.targets) == 3
        assert config.targets[0].key == "target:hole:1"
        assert config.targets[1].key == "target:hole:2"
        assert config.targets[2].key == "target:hole:3"
        assert str(config.targets[0].src).endswith("/hole1.npy")
        assert str(config.targets[2].src).endswith("/hole3.npy")

    def test_range_target_with_source_key_placeholder(self):
        """Test range targets substitute {n} in source_key."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        # First add assets that the derived targets will reference
        config_data.update(
            {
                "assets": [
                    {
                        "key": "mesh:1",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/m1.obj",
                    },
                    {
                        "key": "mesh:2",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/m2.obj",
                    },
                ],
                "targets": [
                    {
                        "key_pattern": "target:{n}",
                        "range": [1, 2],
                        "source_key": "mesh:{n}",
                    }
                ],
            }
        )

        config = ConfigModel.model_validate(config_data)
        assert len(config.targets) == 2
        assert config.targets[0].source_key == "mesh:1"
        assert config.targets[1].source_key == "mesh:2"


class TestDerivedTargetSpec:
    """Test DerivedTargetSpecModel expansion."""

    def test_derived_target_expansion(self):
        """Test derived targets expand from asset keys."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    {
                        "key": "structure:A",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/a.obj",
                    },
                    {
                        "key": "structure:B",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/b.obj",
                    },
                ],
                "targets": [
                    {
                        "derive_from": ["structure:A", "structure:B"],
                        "key_prefix": "target:",
                    }
                ],
            }
        )

        config = ConfigModel.model_validate(config_data)
        assert len(config.targets) == 2
        assert config.targets[0].key == "target:A"
        assert config.targets[0].source_key == "structure:A"
        assert config.targets[1].key == "target:B"
        assert config.targets[1].source_key == "structure:B"


class TestAutoSceneNodeGeneration:
    """Test automatic scene node generation from assets/targets/probes."""

    def test_auto_scene_from_asset_with_transform(self):
        """Test assets with transform auto-generate scene nodes."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "transforms": {"world": TransformFactory.transform_recipe()},
                "assets": [
                    {
                        "key": "brain",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/brain.obj",
                        "transform": "world",
                        "scene_tags": ["static"],
                    }
                ],
            }
        )

        config = ConfigModel.model_validate(config_data)
        assert len(config.scene.nodes) == 1
        assert config.scene.nodes[0].key == "brain"
        assert config.scene.nodes[0].asset == "brain"
        assert config.scene.nodes[0].tags == ["static"]

    def test_auto_scene_from_asset_with_scene_tags_only(self):
        """Test assets with scene_tags (no transform) auto-generate scene nodes."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    {
                        "key": "brain",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/brain.obj",
                        "scene_tags": ["anatomy"],
                    }
                ],
            }
        )

        config = ConfigModel.model_validate(config_data)
        assert len(config.scene.nodes) == 1
        assert config.scene.nodes[0].key == "brain"

    def test_explicit_scene_node_overrides_auto(self):
        """Test explicit scene nodes override auto-generated ones."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "transforms": {"world": TransformFactory.transform_recipe()},
                "assets": [
                    {
                        "key": "brain",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/brain.obj",
                        "transform": "world",
                        "scene_tags": ["auto_tag"],
                    }
                ],
                "scene": {
                    "nodes": [
                        {
                            "key": "brain",
                            "asset": "brain",
                            "tags": ["explicit_tag"],
                        }
                    ]
                },
            }
        )

        config = ConfigModel.model_validate(config_data)
        # Only the explicit node should exist (no duplicate)
        assert len(config.scene.nodes) == 1
        assert config.scene.nodes[0].tags == ["explicit_tag"]

    def test_auto_scene_disabled(self):
        """Test auto_scene=False suppresses auto-generation."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    {
                        "key": "brain",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/brain.obj",
                        "scene_tags": ["should_not_appear"],
                        "auto_scene": False,
                    }
                ],
            }
        )

        config = ConfigModel.model_validate(config_data)
        assert len(config.scene.nodes) == 0

    def test_auto_scene_from_target(self):
        """Test targets with transform auto-generate scene nodes."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "transforms": {"world": TransformFactory.transform_recipe()},
                "assets": [
                    {
                        "key": "mesh1",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/m.obj",
                    }
                ],
                "targets": [
                    {
                        "key": "target1",
                        "source_key": "mesh1",
                        "transform": "world",
                        "scene_tags": ["target"],
                    }
                ],
            }
        )

        config = ConfigModel.model_validate(config_data)
        # Should have auto-generated node for target
        target_nodes = [n for n in config.scene.nodes if n.key == "target1"]
        assert len(target_nodes) == 1
        assert target_nodes[0].tags == ["target"]

    def test_auto_scene_from_probe(self):
        """Test probes auto-generate scene nodes."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    {
                        "key": "mesh1",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/m.obj",
                    },
                    {
                        "key": "probe:neuropixels",
                        "kind": Kind.MESH.value,
                        "role": Role.GEOMETRY.value,
                        "loader": "trimesh",
                        "src": "/p.obj",
                    },
                ],
                "targets": [{"key": "target1", "source_key": "mesh1"}],
                "plan": {
                    "arcs": {"arc1": 0.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "target1",
                        }
                    },
                },
            }
        )

        config = ConfigModel.model_validate(config_data)
        # Should have auto-generated node for probe
        probe_nodes = [n for n in config.scene.nodes if n.key == "probe:probe1"]
        assert len(probe_nodes) == 1
        assert probe_nodes[0].asset == "probe:neuropixels"
        assert probe_nodes[0].pose_source_probe == "probe1"

    def test_probe_auto_scene_disabled(self):
        """Test probe with auto_scene=False doesn't generate node."""
        from aind_low_point.common import Kind, Role

        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "assets": [
                    {
                        "key": "mesh1",
                        "kind": Kind.MESH.value,
                        "role": Role.ANATOMY.value,
                        "loader": "trimesh",
                        "src": "/m.obj",
                    },
                ],
                "targets": [{"key": "target1", "source_key": "mesh1"}],
                "plan": {
                    "arcs": {"arc1": 0.0},
                    "probes": {
                        "probe1": {
                            "kind": "neuropixels",
                            "arc": "arc1",
                            "target": "target1",
                            "auto_scene": False,
                        }
                    },
                },
            }
        )

        config = ConfigModel.model_validate(config_data)
        probe_nodes = [n for n in config.scene.nodes if "probe" in n.key]
        assert len(probe_nodes) == 0
