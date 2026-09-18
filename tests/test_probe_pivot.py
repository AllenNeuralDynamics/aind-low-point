"""A configured probe pivot reaches the optimizer, not only the app.

`AssetSpec.pivot_LPS` lets a config override the kinematic pivot. The app honored
it; the optimizer recomputed its own from the shank tips and never read the
field, so setting it moved the drawn probe and left the optimized one where it
was. Every config leaves it null today, which is why nothing had noticed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from aind_rutter.config import ConfigModel
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo
from aind_rutter.optimization.geometry.recording import (
    RECORDING_GEOMETRY,
    pivot_from_shank_tips,
)
from aind_rutter.runtime.build import build_runtime_from_config
from aind_rutter.runtime.probe_context import probe_context_from_runtime
from tests.synthetic_subject import SHANK_PITCH_MM, write_subject

CONFIGURED_PIVOT = [0.25, -0.5, 2.0]
TIPS = np.array([[0.0, 0.0, 0.0], [0.0, SHANK_PITCH_MM, 0.0]])


def test_the_formula_averages_the_tips_and_centres_the_bank() -> None:
    """The same derivation the runtime stores on the spec."""
    kind = "quadbase-alpha"
    derived = pivot_from_shank_tips(kind, TIPS)
    assert derived[2] == pytest.approx(RECORDING_GEOMETRY[kind].active_center_mm)
    np.testing.assert_allclose(derived[:2], [0.0, SHANK_PITCH_MM / 2])


def test_an_unknown_kind_falls_back_to_the_supplied_centre() -> None:
    derived = pivot_from_shank_tips("not-a-kind", TIPS, center_mm=0.7)
    assert derived[2] == pytest.approx(0.7)


def test_a_mesh_with_no_tips_pivots_on_the_axis() -> None:
    derived = pivot_from_shank_tips("2.1", np.zeros((0, 3)))
    np.testing.assert_allclose(derived[:2], [0.0, 0.0])


@pytest.fixture(scope="module")
def subject(tmp_path_factory: pytest.TempPathFactory):
    return write_subject(tmp_path_factory.mktemp("pivot"))


def _runtime_with_pivot(subject, tmp_path: Path, pivot: list[float] | None):
    document = yaml.safe_load(subject.config.read_text())
    if pivot is not None:
        for asset in document["assets"]:
            if asset["key"] == "probe:2.1":
                asset["pivot_LPS"] = pivot
    path = tmp_path / "pivot.yml"
    path.write_text(yaml.safe_dump(document))
    return build_runtime_from_config(ConfigModel.from_yaml(path))


def test_the_optimizer_reads_a_configured_pivot(subject, tmp_path: Path) -> None:
    """The defect: this used to come back as the shank-tip derivation."""
    runtime = _runtime_with_pivot(subject, tmp_path, CONFIGURED_PIVOT)
    context = probe_context_from_runtime(runtime, "P1")
    np.testing.assert_allclose(context.pivot_local, CONFIGURED_PIVOT)


def test_without_one_the_optimizer_derives_it(subject, tmp_path: Path) -> None:
    runtime = _runtime_with_pivot(subject, tmp_path, None)
    context = probe_context_from_runtime(runtime, "P1")
    expected = pivot_from_shank_tips("2.1", context.shank_tips_local)
    np.testing.assert_allclose(context.pivot_local, expected)


def test_the_solver_statics_carry_the_resolved_pivot() -> None:
    """The value reaches the arrays the objective is traced against."""
    from aind_rutter.optimization.geometry.holes import Hole, HoleSection
    from aind_rutter.optimization.objectives.probe_static import _build_probe_static

    probe = ProbeStaticInfo(
        name="p",
        target_LPS=np.array([0.0, 0.0, -3.0]),
        kind="2.1",
        shank_tips_local=TIPS,
        pivot_local=np.asarray(CONFIGURED_PIVOT),
    )
    hole = Hole(
        id=1,
        axis=np.array([0.0, 0.0, 1.0]),
        ref_point=np.zeros(3),
        sections=[
            HoleSection(
                axis=np.array([0.0, 0.0, 1.0]),
                center=np.zeros(3),
                a=0.5,
                b=0.5,
                theta=0.0,
            )
        ],
    )
    (static,) = _build_probe_static(
        [probe],
        [hole],
        SimpleNamespace(probe_to_hole={"p": 1}),
        SimpleNamespace(probe_to_arc_idx={"p": 0}, arc_centroids_deg=(0.0,)),
    )
    np.testing.assert_allclose(static.pivot_local, CONFIGURED_PIVOT)


def test_a_hand_built_probe_still_derives_its_pivot() -> None:
    from aind_rutter.optimization.geometry.holes import Hole, HoleSection
    from aind_rutter.optimization.objectives.probe_static import _build_probe_static

    probe = ProbeStaticInfo(
        name="p",
        target_LPS=np.array([0.0, 0.0, -3.0]),
        kind="2.1",
        shank_tips_local=TIPS,
    )
    hole = Hole(
        id=1,
        axis=np.array([0.0, 0.0, 1.0]),
        ref_point=np.zeros(3),
        sections=[
            HoleSection(
                axis=np.array([0.0, 0.0, 1.0]),
                center=np.zeros(3),
                a=0.5,
                b=0.5,
                theta=0.0,
            )
        ],
    )
    (static,) = _build_probe_static(
        [probe],
        [hole],
        SimpleNamespace(probe_to_hole={"p": 1}),
        SimpleNamespace(probe_to_arc_idx={"p": 0}, arc_centroids_deg=(0.0,)),
    )
    np.testing.assert_allclose(static.pivot_local, pivot_from_shank_tips("2.1", TIPS))
