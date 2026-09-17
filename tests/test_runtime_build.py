"""`build_runtime_from_config` on a synthetic subject, end to end.

Every tracked example config points at meshes under ``/mnt``, so until now
nothing exercised the path from a YAML document to a ``RuntimeBundle``: loaders,
template expansion, named transforms, the reducer registry, scene assembly and
the derived probe pivot were each covered only in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from aind_rutter.assets import AssetCatalog
from aind_rutter.config import ConfigModel
from aind_rutter.core import MeshTransformable, PointsTransformable
from aind_rutter.optimization.geometry.recording import RECORDING_GEOMETRY
from aind_rutter.runtime.build import RuntimeBundle, build_runtime_from_config
from tests.synthetic_subject import (
    BRAIN_CENTER_LPS,
    HEAD_PITCH_DEG,
    HOLE_PITCH_MM,
    PROBE_KINDS,
    SHANK_PITCH_MM,
    write_subject,
)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> RuntimeBundle:
    subject = write_subject(tmp_path_factory.mktemp("subject"))
    return build_runtime_from_config(ConfigModel.from_yaml(subject.config))


def test_catalog_holds_every_asset_with_the_right_geometry(
    runtime: RuntimeBundle,
) -> None:
    catalog: AssetCatalog = runtime.asset_catalog
    assert set(catalog.assets) == {
        "brain",
        "well",
        "implant",
        "retro-targets",
        *[f"probe:{kind}" for kind in PROBE_KINDS],
    }
    assert isinstance(catalog.assets["brain"].mesh, MeshTransformable)
    assert isinstance(catalog.assets["retro-targets"].points, PointsTransformable)


def test_derived_target_is_the_reduced_source_asset_under_its_transform(
    runtime: RuntimeBundle,
) -> None:
    """The brain is a sphere, so its centre of mass is its centre."""
    point = np.asarray(runtime.targets_pts["target:brain"]).reshape(3)
    tilt = np.deg2rad(2.0)  # the config's headframe_to_lps rotation about x
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt), -np.sin(tilt)],
            [0.0, np.sin(tilt), np.cos(tilt)],
        ]
    )
    expected = rotation @ np.asarray(BRAIN_CENTER_LPS) + np.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(point, expected, atol=1e-3)


def test_scene_carries_a_node_per_asset_and_per_probe(runtime: RuntimeBundle) -> None:
    nodes = set(runtime.scene.nodes)
    assert {"brain", "well", "implant", "retro-targets"} <= nodes
    assert {"probe:P1", "probe:P2"} <= nodes


def test_collision_labels_get_distinct_bits(runtime: RuntimeBundle) -> None:
    bits = runtime.collision_labels.label_to_bit
    assert set(bits) == {"fixture", "probe", "static"}
    assert len(set(bits.values())) == 3
    assert all(bit and bit & (bit - 1) == 0 for bit in bits.values())


def test_plan_state_reproduces_the_declared_probes(runtime: RuntimeBundle) -> None:
    probes = runtime.plan_state.probes
    assert set(probes) == {"P1", "P2"}
    assert probes["P1"].kind == "2.1"
    assert probes["P1"].arc_id == "a"
    assert probes["P1"].target_key == "target:brain"
    assert probes["P2"].kind == "quadbase-alpha"
    assert probes["P2"].spin == pytest.approx(90.0)
    # The inline target is declared in RAS and stored as declared.
    assert probes["P2"].target_point_RAS == (HOLE_PITCH_MM, 0.0, -5.0)


def test_head_pitch_comes_from_the_plan(runtime: RuntimeBundle) -> None:
    from aind_rutter.optimization.pipeline.runtime_adapter import (
        head_pitch_deg_from_runtime,
    )

    assert head_pitch_deg_from_runtime(runtime) == pytest.approx(HEAD_PITCH_DEG)


@pytest.mark.parametrize("kind", PROBE_KINDS)
def test_probe_pivot_is_derived_from_the_shank_tips(
    runtime: RuntimeBundle, kind: str
) -> None:
    """x and y average the shank tips; z is the kind's recording-array centre."""
    spec = runtime.asset_catalog.assets[f"probe:{kind}"]
    pivot = np.asarray(spec.pivot_LPS, dtype=float)
    n_shanks = len(RECORDING_GEOMETRY[kind].active_ranges_mm)
    expected_y = SHANK_PITCH_MM * (n_shanks - 1) / 2
    np.testing.assert_allclose(pivot[:2], [0.0, expected_y], atol=1e-3)
    assert pivot[2] == pytest.approx(RECORDING_GEOMETRY[kind].active_center_mm)
    assert spec.headstage_hull is not None
