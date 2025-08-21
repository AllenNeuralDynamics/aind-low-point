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
