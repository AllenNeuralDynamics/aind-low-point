"""Pytest configuration and shared fixtures for config tests."""

import os

# Speed/repro setup — MUST run before numpy/jax import (i.e. at conftest import,
# which pytest does before collecting test modules). Pin BLAS/OMP threads so the
# xdist workers don't oversubscribe the cores; force JAX to CPU; and share a
# persistent on-disk JAX compile cache so workers and reruns LOAD compiled
# kernels instead of paying the (non-trivial) compile cost on every test.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", ".jax_test_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")

import pytest  # noqa: E402


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
