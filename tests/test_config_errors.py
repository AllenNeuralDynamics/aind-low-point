"""Tests for error handling and edge cases in config.py models."""

import pytest
from pydantic import ValidationError

from aind_low_point.common import Capability, Kind, Role
from aind_low_point.config import (
    AssetSpecModel,
    ConfigModel,
    MaterialModel,
    TargetSpecModel,
    TransformRecipeModel,
)
from tests.config_factories import AssetFactory, ConfigFactory, TargetFactory


class TestFieldConstraintErrors:
    """Test field constraint validation errors."""

    def test_material_opacity_bounds_error_messages(self):
        """Test opacity validation provides clear error messages."""
        with pytest.raises(ValidationError) as exc_info:
            MaterialModel(opacity=-0.5)
        assert "opacity must be in [0,1]" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            MaterialModel(opacity=1.5)
        assert "opacity must be in [0,1]" in str(exc_info.value)

    def test_bbox_hint_format_error_message(self):
        """Test bbox_hint validation provides clear error message."""
        asset_data = AssetFactory.mesh_asset()
        asset_data["bbox_hint"] = [[1.0, 2.0], [3.0, 4.0]]  # Wrong inner length

        with pytest.raises(ValidationError) as exc_info:
            AssetSpecModel(**asset_data)
        assert "bbox_hint must be [[minx,miny,minz],[maxx,maxy,maxz]]" in str(
            exc_info.value
        )

    def test_required_field_error_messages(self):
        """Test required field validation provides clear error messages."""
        with pytest.raises(ValidationError) as exc_info:
            AssetSpecModel()  # Missing required 'key' field
        assert "Field required" in str(exc_info.value)

    def test_invalid_enum_values(self):
        """Test invalid enum values provide clear error messages."""
        asset_data = AssetFactory.mesh_asset()
        asset_data["kind"] = "invalid_kind"

        with pytest.raises(ValidationError) as exc_info:
            AssetSpecModel(**asset_data)
        # Should mention valid options
        error_msg = str(exc_info.value)
        assert "invalid_kind" in error_msg

    def test_array_length_constraints(self):
        """Test array length constraint error messages."""
        # Test approach_vector length constraint
        target_data = TargetFactory.explicit_target()
        target_data["approach_vector"] = [1.0, 2.0]  # Too short

        with pytest.raises(ValidationError):
            TargetSpecModel(**target_data)

        target_data["approach_vector"] = [1.0, 2.0, 3.0, 4.0]  # Too long
        with pytest.raises(ValidationError):
            TargetSpecModel(**target_data)


class TestModelLogicErrors:
    """Test model-specific validation logic errors."""

    def test_asset_src_loader_mutual_requirement_error(self):
        """Test clear error when only one of src/loader provided."""
        with pytest.raises(
            ValidationError, match="must provide both 'src' and 'loader'"
        ):
            AssetSpecModel(key="test", kind=Kind.MESH.value, src="/path/to/file.obj")

        with pytest.raises(
            ValidationError, match="must provide both 'src' and 'loader'"
        ):
            AssetSpecModel(key="test", kind=Kind.MESH.value, loader="trimesh_loader")

    def test_asset_resource_exclusivity_error(self):
        """Test clear error when both resource and src/loader provided."""
        with pytest.raises(ValidationError, match="Choose either"):
            AssetSpecModel(
                key="test",
                kind=Kind.MESH.value,
                src="/path/to/file.obj",
                loader="trimesh_loader",
                from_resource="resource1",
                selector={"kind": "name", "name": "mesh1"},
            )

    def test_target_source_exclusivity_error(self):
        """Test clear error message for target source method conflicts."""
        # No source methods
        with pytest.raises(ValidationError, match="provide exactly one of"):
            TargetSpecModel(key="test", kind=Kind.POINTS.value, role=Role.TARGET.value)

        # Multiple source methods
        with pytest.raises(ValidationError, match="provide exactly one of"):
            TargetSpecModel(
                key="test",
                kind=Kind.POINTS.value,
                role=Role.TARGET.value,
                src="/path/to/file.npy",
                loader="numpy_points",
                source_key="asset1",
                reducer="centroid",
            )

    def test_target_collidable_restriction_error(self):
        """Test clear error when target is marked as collidable."""
        target_data = TargetFactory.explicit_target()
        target_data["caps"] = [Capability.RENDERABLE.value, Capability.COLLIDABLE.value]

        with pytest.raises(
            ValidationError, match="targets should not be collidable by default"
        ):
            TargetSpecModel(**target_data)

    def test_canonicalization_mutual_exclusion_error(self):
        """Test clear error for canonicalization ref/inline conflict."""
        asset_data = AssetFactory.mesh_asset()
        asset_data.update(
            {
                "canonicalization_ref": "canon1",
                "canonicalization": {"source_space": "RAS", "scale_to_mm": 1.0},
            }
        )

        with pytest.raises(
            ValidationError,
            match="Provide either canonicalization_ref or canonicalization",
        ):
            AssetSpecModel(**asset_data)


class TestTransformOpErrors:
    """Test transform operation validation errors."""

    def test_translate_op_delta_length(self):
        """Test translate operation delta must be length 3."""
        from aind_low_point.config import TranslateTxOpModel

        with pytest.raises(ValidationError):
            TranslateTxOpModel(delta=[1.0, 2.0])  # Too short

        with pytest.raises(ValidationError):
            TranslateTxOpModel(delta=[1.0, 2.0, 3.0, 4.0])  # Too long

    def test_rotate_op_angles_length(self):
        """Test rotation operation angles must be length 3."""
        from aind_low_point.config import RotateEulerTxOpModel

        with pytest.raises(ValidationError):
            RotateEulerTxOpModel(angles_deg=[90.0])  # Too short

    def test_invalid_rotation_order(self):
        """Test invalid rotation order raises error."""
        from aind_low_point.config import RotateEulerTxOpModel

        with pytest.raises(ValidationError):
            RotateEulerTxOpModel(order="INVALID", angles_deg=[0.0, 0.0, 90.0])

    def test_sitk_op_missing_path(self, temp_file_path):
        """Test SITK operation requires path."""
        from aind_low_point.config import LoadSITKTxOpModel

        # This should work with valid path
        temp_file_path.write_text("dummy transform")
        LoadSITKTxOpModel(path=temp_file_path)

        # This should fail with missing/invalid path
        with pytest.raises(ValidationError):
            LoadSITKTxOpModel()  # Missing path entirely

    def test_transform_recipe_with_invalid_op(self):
        """Test transform recipe with invalid operation."""
        with pytest.raises(ValidationError):
            TransformRecipeModel(
                sequence=[{"kind": "unknown_op", "some_param": "value"}]
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_values(self):
        """Test empty string values in various fields."""
        # Empty key should be invalid if we add min_length constraint
        # For now, empty string keys are technically allowed by Pydantic
        # This test demonstrates the validation behavior
        asset = AssetSpecModel(key="", kind=Kind.MESH.value)
        assert asset.key == ""

    def test_none_values_where_not_allowed(self):
        """Test None values in required fields."""
        asset_data = AssetFactory.mesh_asset()
        asset_data["key"] = None

        with pytest.raises(ValidationError):
            AssetSpecModel(**asset_data)

    def test_deeply_nested_transform_recipe_errors(self):
        """Test error handling in deeply nested transform structures."""
        with pytest.raises(ValidationError):
            TransformRecipeModel.model_validate(
                {
                    "sequence": [
                        {"kind": "translate_mm", "delta": [1.0, 2.0, 3.0]},
                        {"kind": "invalid_op", "param": "value"},  # Invalid op
                    ]
                }
            )

    def test_circular_reference_detection(self):
        """Test handling of potential circular references."""
        # This is more of a design consideration - current models don't allow
        # circular references, but test that the structure prevents them
        config_data = ConfigFactory.minimal_config()

        # Derived target referencing itself would be caught by validation
        config_data.update(
            {
                "assets": [AssetFactory.mesh_asset(key="asset1")],
                "targets": [
                    TargetFactory.derived_target(key="target1", source_key="asset1")
                ],
            }
        )

        # Should work fine - no circular reference
        config = ConfigModel.model_validate(config_data)
        assert len(config.targets) == 1

    def test_large_numeric_values(self):
        """Test handling of large numeric values."""
        # Test very large coordinates
        target_data = TargetFactory.explicit_target()
        target_data["approach_vector"] = [1e10, -1e10, 0.0]

        # Should be valid - no explicit bounds on coordinate values
        target = TargetSpecModel(**target_data)
        assert target.approach_vector == [1e10, -1e10, 0.0]

    def test_unicode_string_handling(self):
        """Test Unicode string handling in text fields."""
        # Test Unicode in names and descriptions
        material = MaterialModel(name="测试材料", color="#FF0000")
        assert material.name == "测试材料"

        asset_data = AssetFactory.mesh_asset(key="資產_αβγ")
        asset = AssetSpecModel(**asset_data)
        assert asset.key == "資產_αβγ"


class TestErrorMessageQuality:
    """Test that error messages are helpful and actionable."""

    def test_cross_reference_error_includes_context(self):
        """Test cross-reference errors include helpful context."""
        config_data = ConfigFactory.minimal_config()
        config_data["scene"]["nodes"] = [{"id": "node1", "asset": "missing_asset"}]

        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(config_data)

        error_msg = str(exc_info.value)
        assert "scene.nodes['node1']" in error_msg
        assert "missing_asset" in error_msg
        assert "not found in assets" in error_msg

    def test_validation_error_shows_field_path(self):
        """Test validation errors show the field path."""
        with pytest.raises(ValidationError) as exc_info:
            MaterialModel(opacity=2.0)

        error_msg = str(exc_info.value)
        # Should indicate which field failed
        assert "opacity" in error_msg

    def test_multiple_errors_are_collected(self):
        """Test that multiple validation errors are collected together."""
        config_data = ConfigFactory.minimal_config()
        config_data.update(
            {
                "scene": {
                    "nodes": [
                        {"id": "node1", "asset": "missing1"},
                        {"id": "node2", "asset": "missing2"},
                    ]
                }
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            ConfigModel.model_validate(config_data)

        error_msg = str(exc_info.value)
        # Should contain both errors
        assert "missing1" in error_msg
        assert "missing2" in error_msg

    def test_discriminated_union_error_helpful(self):
        """Test discriminated union errors are helpful."""
        with pytest.raises(ValidationError) as exc_info:
            from aind_low_point.config import TranslateTxOpModel

            TranslateTxOpModel(kind="translate_mm")  # Missing required delta field

        error_msg = str(exc_info.value)
        # Should mention the discriminated field and what went wrong
        assert "delta" in error_msg or "Field required" in error_msg
