from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from aind_low_point.core import AffineTransform
from aind_low_point.optimization.geometry import HoleSection
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.pipeline.runtime_adapter import (
    find_well_fixture,
    head_pitch_deg_from_runtime,
    transform_holes_to_lps,
)
from aind_low_point.runtime import RuntimeBundle


def test_head_pitch_deg_from_runtime_reads_subject_from_rig() -> None:
    angle = np.deg2rad(12.5)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    runtime = SimpleNamespace(
        plan_state=SimpleNamespace(
            kinematics=SimpleNamespace(
                subject_from_rig=SimpleNamespace(
                    rotate_translate=(rotation, np.zeros(3))
                )
            )
        )
    )

    assert head_pitch_deg_from_runtime(cast(RuntimeBundle, runtime)) == pytest.approx(
        12.5
    )


def test_transform_holes_to_lps_applies_implant_transform() -> None:
    hole = Hole(
        id=7,
        axis=np.array([0.0, 0.0, 1.0]),
        ref_point=np.array([1.0, 2.0, 3.0]),
        sections=[
            HoleSection(
                axis=np.array([0.0, 0.0, 1.0]),
                center=np.array([4.0, 5.0, 6.0]),
                a=0.4,
                b=0.2,
                theta=0.1,
            )
        ],
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([10.0, 20.0, 30.0])

    transformed = transform_holes_to_lps(
        [hole],
        {"implant_to_lps": AffineTransform(rotation=rotation, translation=translation)},
    )

    assert len(transformed) == 1
    assert np.allclose(transformed[0].axis, [0.0, 0.0, 1.0])
    assert np.allclose(transformed[0].ref_point, [8.0, 21.0, 33.0])
    assert np.allclose(transformed[0].sections[0].center, [5.0, 24.0, 36.0])
    assert transformed[0].sections[0].theta == pytest.approx(0.1)


def test_find_well_fixture_requires_named_well() -> None:
    cone = SimpleNamespace(name="cone")
    well = SimpleNamespace(name="headframe-well")

    assert find_well_fixture(cast(Any, (cone, well))).name == "headframe-well"

    with pytest.raises(ValueError, match="No fixture"):
        find_well_fixture(cast(Any, (cone,)))
