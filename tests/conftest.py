"""Pytest configuration and shared fixtures for config tests."""


import pytest


@pytest.fixture
def temp_file_path(tmp_path):
    """Create a temporary file path for testing."""
    return tmp_path / "test_file.txt"


@pytest.fixture
def temp_dir_path(tmp_path):
    """Create a temporary directory path for testing."""
    return tmp_path / "test_dir"


@pytest.fixture
def sample_transform_file(tmp_path):
    """Create a mock SITK transform file for testing."""
    transform_file = tmp_path / "test_transform.tfm"
    # Create minimal transform file content
    transform_file.write_text(
        "#Insight Transform File V1.0\n"
        "Transform: AffineTransform_double_3_3\n"
        "Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n"
        "FixedParameters: 0 0 0\n"
    )
    return transform_file
