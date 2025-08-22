"""Tests for individual Pydantic model validation in config.py."""

import pytest
from pydantic import ValidationError

from aind_low_point.config import (
    AssetSpecModel,
    CalibrationRefModel,
    CalibrationSourceModel,
    MaterialModel,
    TargetSpecModel,
    TransformRecipeModel,
)
from tests.config_factories import (
    AssetFactory,
    TargetFactory,
    TransformFactory,
)


class TestMaterialModel:
    """Test MaterialModel field validation."""

    def test_valid_material(self):
        """Test valid material creation."""
        material = MaterialModel(
            name="test",
            color="#FF0000",
            opacity=0.5,
            wireframe=True,
            visible=False,
        )
        assert material.name == "test"
        assert material.color == "#FF0000"
        assert material.opacity == 0.5
        assert material.wireframe is True
        assert material.visible is False

    def test_default_values(self):
        """Test default values are applied correctly."""
        material = MaterialModel()
        assert material.name == "default"
        assert material.color == "#C8C8C8"
        assert material.opacity == 1.0
        assert material.wireframe is False
        assert material.visible is True

    def test_opacity_range_validation(self):
        """Test opacity must be in [0,1] range."""
        # Valid opacity values
        MaterialModel(opacity=0.0)
        MaterialModel(opacity=1.0)
        MaterialModel(opacity=0.5)

        # Invalid opacity values
        with pytest.raises(ValidationError, match="opacity must be in \\[0,1\\]"):
            MaterialModel(opacity=-0.1)

        with pytest.raises(ValidationError, match="opacity must be in \\[0,1\\]"):
            MaterialModel(opacity=1.1)


class TestAssetSpecModel:
    """Test AssetSpecModel validation logic."""

    def test_valid_asset_with_src_loader(self):
        """Test asset with src and loader is valid."""
        asset_data = AssetFactory.mesh_asset()
        asset = AssetSpecModel(**asset_data)
        assert asset.key == asset_data["key"]
        assert asset.src is not None
        assert asset.loader is not None

    def test_asset_requires_both_src_and_loader(self):
        """Test that src and loader must both be provided or both be None."""
        # Only src provided
        with pytest.raises(
            ValidationError, match="must provide both 'src' and 'loader'"
        ):
            AssetSpecModel(
                key="test",
                kind="mesh",
                src="/path/to/file.obj",
                # loader missing
            )

        # Only loader provided
        with pytest.raises(
            ValidationError, match="must provide both 'src' and 'loader'"
        ):
            AssetSpecModel(
                key="test",
                kind="mesh",
                loader="trimesh_loader",
                # src missing
            )

    def test_asset_resource_vs_src_loader_exclusive(self):
        """Test that from_resource+selector vs src+loader are mutually exclusive."""
        # Both src+loader and from_resource+selector
        with pytest.raises(
            ValidationError,
            match="Choose either \\(src\\+loader\\) or \\(from_resource\\+selector\\)",
        ):
            AssetSpecModel(
                key="test",
                kind="mesh",
                src="/path/to/file.obj",
                loader="trimesh_loader",
                from_resource="resource1",
                selector={"kind": "name", "name": "mesh1"},
            )

    def test_asset_resource_requires_selector(self):
        """Test that from_resource requires a selector."""
        with pytest.raises(ValidationError, match="you must also provide a selector"):
            AssetSpecModel(
                key="test",
                kind="mesh",
                from_resource="resource1",
                # selector missing
            )

    def test_bbox_hint_validation(self):
        """Test bbox_hint format validation."""
        # Valid bbox
        asset_data = AssetFactory.mesh_asset()
        asset_data["bbox_hint"] = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        asset = AssetSpecModel(**asset_data)
        assert len(asset.bbox_hint) == 2

        # Invalid bbox format
        with pytest.raises(ValidationError, match="bbox_hint must be"):
            asset_data["bbox_hint"] = [[0.0, 1.0], [3.0, 4.0]]  # Wrong inner length
            AssetSpecModel(**asset_data)

    def test_canonicalization_mutual_exclusion(self):
        """Test that canonicalization_ref and canonicalization are mutually
        exclusive."""
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


class TestTargetSpecModel:
    """Test TargetSpecModel validation logic."""

    def test_explicit_target_valid(self):
        """Test explicit target with src and loader."""
        target_data = TargetFactory.explicit_target()
        target = TargetSpecModel(**target_data)
        assert target.src is not None
        assert target.loader is not None

    def test_derived_target_valid(self):
        """Test derived target with source_key and reducer."""
        target_data = TargetFactory.derived_target()
        target = TargetSpecModel(**target_data)
        assert target.source_key is not None
        assert target.reducer is not None

    def test_target_exactly_one_source_method(self):
        """Test that targets must use exactly one source method."""
        # No source method
        with pytest.raises(ValidationError, match="provide exactly one of"):
            TargetSpecModel(key="test")

        # Multiple source methods
        with pytest.raises(ValidationError, match="provide exactly one of"):
            TargetSpecModel(
                key="test",
                src="/path/to/file.npy",
                loader="numpy_points",
                source_key="asset1",
                reducer="centroid",
            )

    def test_target_not_collidable_by_default(self):
        """Test that targets cannot be collidable by default."""
        from aind_low_point.common import Capability

        target_data = TargetFactory.explicit_target()
        target_data["caps"] = [Capability.RENDERABLE, Capability.COLLIDABLE]

        with pytest.raises(ValidationError, match="targets should not be collidable"):
            TargetSpecModel(**target_data)

    def test_approach_vector_length(self):
        """Test approach vector must be length 3."""
        target_data = TargetFactory.explicit_target()

        # Valid length
        target_data["approach_vector"] = [1.0, 0.0, 0.0]
        target = TargetSpecModel(**target_data)
        assert len(target.approach_vector) == 3

        # Invalid length
        target_data["approach_vector"] = [1.0, 0.0]  # Too short
        with pytest.raises(ValidationError):
            TargetSpecModel(**target_data)


class TestTransformRecipeModel:
    """Test TransformRecipeModel coercion and validation."""

    def test_empty_sequence_default(self):
        """Test empty sequence is default."""
        recipe = TransformRecipeModel()
        assert recipe.sequence == []

    def test_single_op_coercion_at_root(self):
        """Test single op at root level is coerced to sequence."""
        recipe = TransformRecipeModel.model_validate(
            {
                "kind": "translate_mm",
                "delta": [1.0, 2.0, 3.0],
            }
        )
        assert len(recipe.sequence) == 1
        assert recipe.sequence[0].kind == "translate_mm"

    def test_sequence_single_op_coercion(self):
        """Test single op in sequence field is coerced to list."""
        recipe = TransformRecipeModel(
            sequence={"kind": "translate_mm", "delta": [1.0, 2.0, 3.0]}
        )
        assert len(recipe.sequence) == 1
        assert recipe.sequence[0].kind == "translate_mm"

    def test_multiple_ops_sequence(self):
        """Test multiple operations in sequence."""
        ops = [
            TransformFactory.translate_op(),
            TransformFactory.rotate_op(),
        ]
        recipe = TransformRecipeModel(sequence=ops)
        assert len(recipe.sequence) == 2
        assert recipe.sequence[0].kind == "translate_mm"
        assert recipe.sequence[1].kind == "rotate_euler_deg"


class TestCalibrationRefModel:
    """Test CalibrationRefModel string parsing."""

    def test_from_string_valid(self):
        """Test valid string parsing."""
        ref = CalibrationRefModel.from_string("cal1:12345")
        assert ref.cal_id == "cal1"
        assert ref.probe_code == "12345"

    def test_from_string_with_spaces(self):
        """Test string parsing with spaces."""
        ref = CalibrationRefModel.from_string(" cal1 : 12345 ")
        assert ref.cal_id == "cal1"
        assert ref.probe_code == "12345"

    def test_from_string_invalid_format(self):
        """Test invalid string format raises error."""
        with pytest.raises(ValueError, match="Expected '<cal_id>:<probe_code>'"):
            CalibrationRefModel.from_string("invalid_format")

    def test_from_string_multiple_colons(self):
        """Test string with multiple colons."""
        ref = CalibrationRefModel.from_string("cal1:probe:12345")
        assert ref.cal_id == "cal1"
        assert ref.probe_code == "probe:12345"  # Everything after first colon


class TestCalibrationSourceModel:
    """Test CalibrationSourceModel validation."""

    def test_file_source_valid(self, temp_file_path):
        """Test file-based calibration source."""
        temp_file_path.touch()  # Create the file
        source = CalibrationSourceModel(file=temp_file_path)
        assert source.file == temp_file_path
        assert source.directory is None
        assert source.reticle is None

    def test_directory_source_valid(self, temp_dir_path):
        """Test directory-based calibration source."""
        temp_dir_path.mkdir()  # Create the directory
        source = CalibrationSourceModel(directory=temp_dir_path, reticle="reticle1")
        assert source.directory == temp_dir_path
        assert source.reticle == "reticle1"
        assert source.file is None

    def test_file_and_directory_exclusive(self, temp_file_path, temp_dir_path):
        """Test that file and directory are mutually exclusive."""
        temp_file_path.touch()
        temp_dir_path.mkdir()

        with pytest.raises(
            ValidationError, match="Specify exactly one of 'file' or 'directory'"
        ):
            CalibrationSourceModel(file=temp_file_path, directory=temp_dir_path)

    def test_neither_file_nor_directory(self):
        """Test that one of file or directory must be provided."""
        with pytest.raises(
            ValidationError, match="Specify exactly one of 'file' or 'directory'"
        ):
            CalibrationSourceModel()

    def test_file_forbids_reticle(self, temp_file_path):
        """Test that reticle cannot be used with file."""
        temp_file_path.touch()
        with pytest.raises(
            ValidationError, match="'reticle' must not be provided when 'file' is used"
        ):
            CalibrationSourceModel(file=temp_file_path, reticle="reticle1")

    def test_directory_requires_reticle(self, temp_dir_path):
        """Test that directory requires reticle."""
        temp_dir_path.mkdir()
        with pytest.raises(
            ValidationError, match="'reticle' is required when 'directory' is used"
        ):
            CalibrationSourceModel(directory=temp_dir_path)


class TestMaterialReference:
    """Test material_ref field validation in BaseSpecModel."""

    def test_material_ref_valid(self):
        """Test material_ref field works correctly."""
        from tests.config_factories import AssetFactory
        
        asset_data = AssetFactory.asset_with_material_ref(
            material_ref="test_material"
        )
        asset = AssetSpecModel(**asset_data)
        assert asset.material_ref == "test_material"

    def test_material_ref_with_inline_material(self):
        """Test material_ref and inline material can coexist."""
        from tests.config_factories import AssetFactory, MaterialFactory
        
        asset_data = AssetFactory.asset_with_material_ref(
            material_ref="test_material",
            material=MaterialFactory.material(color="#FF0000")
        )
        asset = AssetSpecModel(**asset_data)
        assert asset.material_ref == "test_material"
        assert asset.material.color == "#FF0000"

    def test_target_material_ref_valid(self):
        """Test material_ref works in target specs."""
        from tests.config_factories import TargetFactory
        
        target_data = TargetFactory.target_with_material_ref(
            material_ref="target_material"
        )
        target = TargetSpecModel(**target_data)
        assert target.material_ref == "target_material"


class TestBaseTemplateModel:
    """Test BaseTemplateModel validation."""

    def test_base_template_creation(self):
        """Test basic template creation."""
        from aind_low_point.config import BaseTemplateModel
        from tests.config_factories import TemplateFactory
        
        template_data = TemplateFactory.base_template(
            name="test_template",
            kind="mesh",
            role="geometry"
        )
        template = BaseTemplateModel(**template_data)
        assert template.name == "test_template"
        assert template.kind == "mesh"
        assert template.role == "geometry"

    def test_template_material_ref(self):
        """Test template with material reference."""
        from aind_low_point.config import BaseTemplateModel
        from tests.config_factories import TemplateFactory
        
        template_data = TemplateFactory.base_template(
            material_ref="template_material"
        )
        template = BaseTemplateModel(**template_data)
        assert template.material_ref == "template_material"

    def test_template_optional_fields(self):
        """Test template with optional fields."""
        from aind_low_point.config import BaseTemplateModel
        from tests.config_factories import TemplateFactory
        
        template_data = TemplateFactory.base_template(
            tags=["template", "test"],
            metadata={"source": "factory"}
        )
        template = BaseTemplateModel(**template_data)
        assert template.tags == ["template", "test"]
        assert template.metadata == {"source": "factory"}


class TestAssetTemplateModel:
    """Test AssetTemplateModel validation."""

    def test_asset_template_with_loader(self):
        """Test asset template with source loader."""
        from aind_low_point.config import AssetTemplateModel
        from tests.config_factories import TemplateFactory
        
        template_data = TemplateFactory.mesh_template_with_loader()
        template = AssetTemplateModel(**template_data)
        assert str(template.src) == "/template/mesh.obj"
        assert template.loader == "trimesh_loader"
        assert template.kind == "mesh"

    def test_asset_template_with_resource(self):
        """Test asset template with resource selector."""
        from aind_low_point.config import AssetTemplateModel
        from tests.config_factories import TemplateFactory, SelectorFactory
        
        template_data = TemplateFactory.asset_template(
            from_resource="test_resource",
            selector=SelectorFactory.name_selector("mesh1")
        )
        template = AssetTemplateModel(**template_data)
        assert template.from_resource == "test_resource"
        assert template.selector.kind == "name"
        assert template.selector.name == "mesh1"


class TestTargetTemplateModel:
    """Test TargetTemplateModel validation."""

    def test_target_template_explicit(self):
        """Test target template with explicit source."""
        from aind_low_point.config import TargetTemplateModel
        from tests.config_factories import TemplateFactory
        
        template_data = TemplateFactory.points_template_explicit()
        template = TargetTemplateModel(**template_data)
        assert str(template.src) == "/template/points.npy"
        assert template.loader == "numpy_points"
        assert template.kind == "points"

    def test_target_template_derived(self):
        """Test target template with derived source."""
        from aind_low_point.config import TargetTemplateModel
        from tests.config_factories import TemplateFactory
        
        template_data = TemplateFactory.target_template_derived()
        template = TargetTemplateModel(**template_data)
        assert template.source_key == "brain_mesh"
        assert template.reducer == "centroid"

    def test_target_template_specific_fields(self):
        """Test target-specific fields in template."""
        from aind_low_point.config import TargetTemplateModel
        from tests.config_factories import TemplateFactory
        
        template_data = TemplateFactory.target_template(
            approach_vector=[1.0, 0.0, 0.0],
            uncertainty_mm=2.5
        )
        template = TargetTemplateModel(**template_data)
        assert template.approach_vector == [1.0, 0.0, 0.0]
        assert template.uncertainty_mm == 2.5


class TestTemplateApplication:
    """Test templates field in spec models."""

    def test_asset_templates_field(self):
        """Test AssetSpecModel.templates field."""
        from tests.config_factories import AssetFactory
        
        asset_data = AssetFactory.asset_with_templates(
            templates=["template1", "template2"]
        )
        asset = AssetSpecModel(**asset_data)
        assert asset.templates == ["template1", "template2"]

    def test_target_templates_field(self):
        """Test TargetSpecModel.templates field."""
        from tests.config_factories import TargetFactory
        
        target_data = TargetFactory.target_with_templates(
            templates=["target_template1"]
        )
        target = TargetSpecModel(**target_data)
        assert target.templates == ["target_template1"]

    def test_empty_templates_default(self):
        """Test templates defaults to empty list."""
        from tests.config_factories import AssetFactory, TargetFactory
        
        asset_data = AssetFactory.mesh_asset()
        asset = AssetSpecModel(**asset_data)
        assert asset.templates == []

        target_data = TargetFactory.explicit_target()
        target = TargetSpecModel(**target_data)
        assert target.templates == []
