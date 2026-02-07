"""Tests for input coercion and normalization in config.py models."""

import pytest
from pydantic import ValidationError

from aind_low_point.config import (
    CalibrationRefModel,
    CalibrationsModel,
    TransformRecipeModel,
    TransformRefModel,
)
from tests.config_factories import CalibrationFactory, TransformFactory


class TestTransformRecipeCoercion:
    """Test TransformRecipeModel input coercion."""

    def test_root_single_op_coercion(self):
        """Test single operation at root is coerced to sequence."""
        # Single op dict should become {sequence: [op]}
        recipe = TransformRecipeModel.model_validate(
            {
                "kind": "translate_mm",
                "delta": [1.0, 2.0, 3.0],
                "invert": False,
            }
        )

        assert len(recipe.sequence) == 1
        assert recipe.sequence[0].kind == "translate_mm"
        assert recipe.sequence[0].delta == [1.0, 2.0, 3.0]

    def test_sequence_single_op_coercion(self):
        """Test single op in sequence field is wrapped in list."""
        recipe = TransformRecipeModel(
            sequence={
                "kind": "rotate_euler_deg",
                "order": "ZYX",
                "angles_deg": [0.0, 0.0, 90.0],
            }
        )

        assert len(recipe.sequence) == 1
        assert recipe.sequence[0].kind == "rotate_euler_deg"
        assert recipe.sequence[0].order == "ZYX"

    def test_sequence_none_becomes_empty_list(self):
        """Test None sequence becomes empty list."""
        recipe = TransformRecipeModel(sequence=None)
        assert recipe.sequence == []

    def test_sequence_list_preserved(self):
        """Test list sequence is preserved as-is."""
        ops = [
            TransformFactory.translate_op(),
            TransformFactory.rotate_op(),
        ]
        recipe = TransformRecipeModel(sequence=ops)

        assert len(recipe.sequence) == 2
        assert recipe.sequence[0].kind == "translate_mm"
        assert recipe.sequence[1].kind == "rotate_euler_deg"

    def test_invalid_sequence_type_error(self):
        """Test invalid sequence type raises TypeError."""
        with pytest.raises(TypeError, match="sequence must be a list"):
            TransformRecipeModel(sequence="invalid")


class TestTransformRefCoercion:
    """Test TransformRefModel input coercion and validation."""

    def test_string_coercion_to_key(self):
        """Test string input is coerced to {key: string}."""
        ref = TransformRefModel.model_validate("my_transform")
        assert ref.key == "my_transform"
        assert ref.inline is None

    def test_list_coercion_to_inline_sequence(self):
        """Test list input is coerced to {inline: {sequence: list}}."""
        ops_list = [
            {"kind": "translate_mm", "delta": [1.0, 2.0, 3.0]},
            {
                "kind": "rotate_euler_deg",
                "order": "ZYX",
                "angles_deg": [0.0, 0.0, 90.0],
            },
        ]

        ref = TransformRefModel.model_validate(ops_list)
        assert ref.key is None
        assert ref.inline is not None
        assert len(ref.inline.sequence) == 2
        assert ref.inline.sequence[0].kind == "translate_mm"

    def test_single_op_dict_coercion_to_inline(self):
        """Test single op dict is coerced to {inline: {sequence: [op]}}."""
        ref = TransformRefModel.model_validate(
            {
                "kind": "translate_mm",
                "delta": [5.0, 0.0, 0.0],
            }
        )

        assert ref.key is None
        assert ref.inline is not None
        assert len(ref.inline.sequence) == 1
        assert ref.inline.sequence[0].kind == "translate_mm"
        assert ref.inline.sequence[0].delta == [5.0, 0.0, 0.0]

    def test_recipe_dict_coercion_to_inline(self):
        """Test recipe dict with 'sequence' is coerced to {inline: recipe}."""
        ref = TransformRefModel.model_validate(
            {
                "sequence": [
                    {"kind": "translate_mm", "delta": [1.0, 2.0, 3.0]},
                ]
            }
        )

        assert ref.key is None
        assert ref.inline is not None
        assert len(ref.inline.sequence) == 1

    def test_inline_single_op_coercion(self):
        """Test inline field single op is coerced to sequence."""
        ref = TransformRefModel(
            inline={
                "kind": "translate_mm",
                "delta": [1.0, 2.0, 3.0],
            }
        )

        assert ref.key is None
        assert ref.inline is not None
        assert len(ref.inline.sequence) == 1
        assert ref.inline.sequence[0].kind == "translate_mm"

    def test_explicit_key_and_inline_structure(self):
        """Test explicit key and inline structure work."""
        # Explicit key reference
        ref_key = TransformRefModel(key="my_transform")
        assert ref_key.key == "my_transform"
        assert ref_key.inline is None

        # Explicit inline structure
        ref_inline = TransformRefModel(
            inline=TransformRecipeModel(sequence=[TransformFactory.translate_op()])
        )
        assert ref_inline.key is None
        assert ref_inline.inline is not None

    def test_xor_validation_both_provided(self):
        """Test that providing both key and inline raises error."""
        with pytest.raises(ValidationError, match="provide exactly one of"):
            TransformRefModel(
                key="my_transform", inline=TransformRecipeModel(sequence=[])
            )

    def test_xor_validation_neither_provided(self):
        """Test that providing neither key nor inline raises error."""
        with pytest.raises(ValidationError, match="provide exactly one of"):
            TransformRefModel()


class TestCalibrationModelNormalization:
    """Test CalibrationsModel string reference normalization."""

    def test_string_ref_normalization(self, temp_file_path):
        """Test string references are converted to CalibrationRefModel."""
        temp_file_path.write_text("dummy calibration file")
        calibrations = CalibrationsModel(
            files={
                "cal1": CalibrationFactory.calibration_source_file(str(temp_file_path))
            },
            probe_to_ref={
                "probe1": "cal1:12345",  # String format
                "probe2": CalibrationRefModel(
                    cal_id="cal1", probe_code="67890"
                ),  # Already object
            },
        )

        # Both should be CalibrationRefModel instances
        assert isinstance(calibrations.probe_to_ref["probe1"], CalibrationRefModel)
        assert isinstance(calibrations.probe_to_ref["probe2"], CalibrationRefModel)

        # Check values
        assert calibrations.probe_to_ref["probe1"].cal_id == "cal1"
        assert calibrations.probe_to_ref["probe1"].probe_code == "12345"
        assert calibrations.probe_to_ref["probe2"].cal_id == "cal1"
        assert calibrations.probe_to_ref["probe2"].probe_code == "67890"

    def test_mixed_ref_types_normalization(self, temp_file_path):
        """Test mixed string and object references are normalized."""
        temp_file_path.write_text("dummy calibration file")
        calibrations = CalibrationsModel(
            files={
                "cal1": CalibrationFactory.calibration_source_file(str(temp_file_path))
            },
            probe_to_ref={
                "probe1": "cal1:probe_A",
                "probe2": {"cal_id": "cal1", "probe_code": "probe_B"},
                "probe3": CalibrationRefModel(cal_id="cal1", probe_code="probe_C"),
            },
        )

        # All should be CalibrationRefModel instances
        for probe_name, ref in calibrations.probe_to_ref.items():
            assert isinstance(ref, CalibrationRefModel)
            assert ref.cal_id == "cal1"

        # Check specific probe codes
        assert calibrations.probe_to_ref["probe1"].probe_code == "probe_A"
        assert calibrations.probe_to_ref["probe2"].probe_code == "probe_B"
        assert calibrations.probe_to_ref["probe3"].probe_code == "probe_C"

    def test_invalid_string_ref_format(self, temp_file_path):
        """Test invalid string reference format raises error during normalization."""
        temp_file_path.write_text("dummy calibration file")
        with pytest.raises(ValidationError, match="Expected '<cal_id>:<probe_code>'"):
            CalibrationsModel(
                files={
                    "cal1": CalibrationFactory.calibration_source_file(
                        str(temp_file_path)
                    )
                },
                probe_to_ref={
                    "probe1": "invalid_format_no_colon",
                },
            )


class TestSelectorCoercion:
    """Test selector discriminated union behavior."""

    def test_name_selector_discrimination(self):
        """Test name selector is correctly discriminated."""
        from aind_low_point.config import NameSelector

        selector_data = {"kind": "name", "name": "mesh1"}
        selector = NameSelector.model_validate(selector_data)

        assert selector.kind == "name"
        assert selector.name == "mesh1"

    def test_index_selector_discrimination(self):
        """Test index selector is correctly discriminated."""
        from aind_low_point.config import IndexSelector

        selector_data = {"kind": "index", "index": 5}
        selector = IndexSelector.model_validate(selector_data)

        assert selector.kind == "index"
        assert selector.index == 5

    def test_path_selector_discrimination(self):
        """Test path selector is correctly discriminated."""
        from aind_low_point.config import PathSelector

        selector_data = {"kind": "path", "path": "/dataset/points"}
        selector = PathSelector.model_validate(selector_data)

        assert selector.kind == "path"
        assert selector.path == "/dataset/points"

    def test_label_selector_discrimination(self):
        """Test label selector with both int and string labels."""
        from aind_low_point.config import LabelSelector

        # Integer label
        selector_int = LabelSelector.model_validate({"kind": "label", "label": 42})
        assert selector_int.kind == "label"
        assert selector_int.label == 42

        # String label
        selector_str = LabelSelector.model_validate(
            {"kind": "label", "label": "cortex"}
        )
        assert selector_str.kind == "label"
        assert selector_str.label == "cortex"

    def test_invalid_selector_kind(self):
        """Test invalid selector kind raises validation error."""
        with pytest.raises(ValidationError):
            from aind_low_point.config import NameSelector

            # This should fail because we're using wrong discriminator
            NameSelector.model_validate({"kind": "invalid_kind", "name": "test"})


class TestTemplateCoercion:
    """Test template field coercion and normalization."""

    def test_templates_list_normalization(self):
        """Test templates field accepts list of strings."""
        from tests.config_factories import AssetFactory

        asset_data = AssetFactory.asset_with_templates(
            templates=["template1", "template2", "template3"]
        )
        from aind_low_point.config import AssetSpecModel

        asset = AssetSpecModel(**asset_data)
        assert asset.templates == ["template1", "template2", "template3"]

    def test_empty_templates_list_default(self):
        """Test templates defaults to empty list."""
        from aind_low_point.config import AssetSpecModel, TargetSpecModel
        from tests.config_factories import AssetFactory, TargetFactory

        asset_data = AssetFactory.mesh_asset()
        asset = AssetSpecModel(**asset_data)
        assert asset.templates == []

        target_data = TargetFactory.explicit_target()
        target = TargetSpecModel(**target_data)
        assert target.templates == []

    def test_single_template_in_list(self):
        """Test single template in list works correctly."""
        from aind_low_point.config import TargetSpecModel
        from tests.config_factories import TargetFactory

        target_data = TargetFactory.target_with_templates(templates=["single_template"])
        target = TargetSpecModel(**target_data)
        assert target.templates == ["single_template"]


class TestMaterialResolution:
    """Test material_ref resolution and precedence."""

    def test_material_ref_field_present(self):
        """Test material_ref field is preserved."""
        from aind_low_point.config import AssetSpecModel
        from tests.config_factories import AssetFactory

        asset_data = AssetFactory.asset_with_material_ref(material_ref="test_material")
        asset = AssetSpecModel(**asset_data)
        assert asset.material_ref == "test_material"

    def test_material_ref_with_inline_material_coexist(self):
        """Test material_ref and inline material can coexist."""
        from aind_low_point.config import AssetSpecModel
        from tests.config_factories import AssetFactory, MaterialFactory

        asset_data = AssetFactory.asset_with_material_ref(
            material_ref="ref_material",
            material=MaterialFactory.material(name="inline_material", color="#FF0000"),
        )
        asset = AssetSpecModel(**asset_data)
        assert asset.material_ref == "ref_material"
        assert asset.material.name == "inline_material"
        assert asset.material.color == "#FF0000"

    def test_none_material_ref_allowed(self):
        """Test material_ref can be None."""
        from aind_low_point.config import AssetSpecModel
        from tests.config_factories import AssetFactory

        asset_data = AssetFactory.mesh_asset(material_ref=None)
        asset = AssetSpecModel(**asset_data)
        assert asset.material_ref is None

    def test_template_material_ref_preserved(self):
        """Test material_ref in templates is preserved."""
        from aind_low_point.config import AssetTemplateModel
        from tests.config_factories import TemplateFactory

        template_data = TemplateFactory.asset_template(material_ref="template_material")
        template = AssetTemplateModel(**template_data)
        assert template.material_ref == "template_material"


class TestTemplateFieldCoercion:
    """Test template-specific field coercion."""

    def test_template_optional_fields_coercion(self):
        """Test optional fields in templates work correctly."""
        from aind_low_point.config import BaseTemplateModel
        from tests.config_factories import TemplateFactory

        # Test with None values
        template_data = TemplateFactory.base_template(
            kind=None, role=None, material_ref=None
        )
        template = BaseTemplateModel(**template_data)
        assert template.kind is None
        assert template.role is None
        assert template.material_ref is None

    def test_asset_template_source_modes_coercion(self):
        """Test asset template source mode fields."""
        from aind_low_point.config import AssetTemplateModel
        from tests.config_factories import SelectorFactory, TemplateFactory

        # Test with resource mode
        template_data = TemplateFactory.asset_template(
            from_resource="test_resource",
            selector=SelectorFactory.name_selector("mesh_data"),
            src=None,
            loader=None,
        )
        template = AssetTemplateModel(**template_data)
        assert template.from_resource == "test_resource"
        assert template.selector.name == "mesh_data"
        assert template.src is None
        assert template.loader is None

    def test_target_template_source_modes_coercion(self):
        """Test target template source mode fields."""
        from aind_low_point.config import TargetTemplateModel
        from tests.config_factories import TemplateFactory

        # Test derived mode
        template_data = TemplateFactory.target_template_derived(
            source_key="source_asset", reducer="mean", src=None, loader=None
        )
        template = TargetTemplateModel(**template_data)
        assert template.source_key == "source_asset"
        assert template.reducer == "mean"
        assert template.src is None
        assert template.loader is None
