"""`.npy` sources name a loader, and now one exists.

`EXTENSION_DEFAULTS` has inferred `numpy_points` for `.npy` since the extension
table was written, and two model docstrings cite it, but nothing registered it —
so every `.npy` source failed at build with "unknown loader".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from aind_rutter.config import ConfigModel
from aind_rutter.runtime.build import build_runtime_from_config
from aind_rutter.runtime.loaders import load_geometry, numpy_points
from tests.synthetic_subject import write_subject

POINTS = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])


def _write(path: Path, array: np.ndarray) -> Path:
    np.save(path, array)
    return path.with_suffix(".npy")


def test_the_extension_table_still_names_this_loader() -> None:
    """The rule is vacuous if the inferred name drifts."""
    from aind_rutter.config import EXTENSION_DEFAULTS

    assert EXTENSION_DEFAULTS[".npy"]["loader"] == "numpy_points"


def test_it_is_registered_under_the_inferred_name(tmp_path: Path) -> None:
    source = _write(tmp_path / "points", POINTS)
    loaded = load_geometry(source, loader="numpy_points")
    np.testing.assert_allclose(loaded, POINTS)


def test_integer_points_come_back_as_floats(tmp_path: Path) -> None:
    source = _write(tmp_path / "ints", POINTS.astype(np.int32))
    loaded = numpy_points(str(source))
    assert loaded.dtype == np.float64
    np.testing.assert_allclose(loaded, POINTS)


def test_rows_with_a_non_finite_coordinate_are_dropped(tmp_path: Path) -> None:
    array = np.vstack([POINTS, [np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]])
    source = _write(tmp_path / "ragged", array)
    np.testing.assert_allclose(numpy_points(str(source)), POINTS)


@pytest.mark.parametrize(
    "array",
    [np.zeros(3), np.zeros((4, 2)), np.zeros((2, 3, 3))],
    ids="1d 2col 3d".split(),
)
def test_a_shape_that_is_not_n_by_three_is_refused(
    tmp_path: Path, array: np.ndarray
) -> None:
    source = _write(tmp_path / "wrong_shape", array)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        numpy_points(str(source))


def test_non_numeric_points_are_refused(tmp_path: Path) -> None:
    source = _write(tmp_path / "strings", np.array([["a", "b", "c"]]))
    with pytest.raises(ValueError, match="numeric"):
        numpy_points(str(source))


def test_a_pickled_array_is_refused(tmp_path: Path) -> None:
    """A pickled .npy executes arbitrary code on load; nothing here writes one."""
    source = tmp_path / "pickled.npy"
    np.save(source, np.array([{"not": "points"}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="allow_pickle=False"):
        numpy_points(str(source))


def test_a_config_can_point_an_asset_at_a_npy_file(tmp_path: Path) -> None:
    """End to end: kind and loader are both inferred from the extension."""
    subject = write_subject(tmp_path)
    source = _write(tmp_path / "retro", POINTS)
    document = yaml.safe_load(subject.config.read_text())
    document["assets"].append(
        {
            "key": "npy-points",
            "role": "anatomy",
            "src": str(source),
            "caps": ["renderable"],
            "scene_tags": ["static"],
        }
    )
    path = tmp_path / "with-npy.yml"
    path.write_text(yaml.safe_dump(document))

    runtime = build_runtime_from_config(ConfigModel.from_yaml(path))
    spec = runtime.asset_catalog.assets["npy-points"]
    assert spec.loader == "numpy_points"
    np.testing.assert_allclose(spec.points.raw, POINTS)
