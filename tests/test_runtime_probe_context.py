from __future__ import annotations

import numpy as np
import pytest
import trimesh

from aind_low_point.assets import AssetCatalog, AssetSpec, TargetSpec
from aind_low_point.core import MeshTransformable, PointsTransformable
from aind_low_point.planning import Kinematics, PlanningState, ProbePlan
from aind_low_point.runtime.build import CollisionLabelIndex, RuntimeBundle
from aind_low_point.runtime.probe_context import (
    coverage_weight_for_probe,
    probe_context_from_runtime,
    resolve_plan_target_lps,
)
from aind_low_point.scene import Scene


def _runtime_with_probe() -> RuntimeBundle:
    target_points = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=float)
    catalog = AssetCatalog(
        assets={
            "probe:test": AssetSpec(
                key="probe:test",
                kind="mesh",
                mesh=MeshTransformable(trimesh.creation.box()),
            )
        },
        targets={
            "target:test": TargetSpec(
                key="target:test",
                points=PointsTransformable(target_points),
                metadata={"coverage_weight": 2.5},
            )
        },
    )
    plan_state = PlanningState(
        kinematics=Kinematics(),
        probes={
            "P": ProbePlan(kind="test", arc_id=None, target_key="target:test"),
        },
        target_index={"target:test": target_points},
    )
    return RuntimeBundle(
        asset_catalog=catalog,
        targets_pts={"target:test": target_points},
        scene=Scene(),
        collision_labels=CollisionLabelIndex(label_to_bit={}, bit_to_label={}),
        plan_state=plan_state,
    )


def test_probe_context_resolves_target_mesh_and_coverage_weight() -> None:
    runtime = _runtime_with_probe()

    context = probe_context_from_runtime(runtime, "P", coverage_environ={})

    assert context.name == "P"
    assert context.kind == "test"
    assert np.allclose(context.target_LPS, [2.0, 3.0, 4.0])
    assert context.collision_mesh is not None
    assert context.shank_tips_local.shape[1] == 3
    assert context.coverage_weight == pytest.approx(2.5)


def test_coverage_weight_env_override_wins() -> None:
    runtime = _runtime_with_probe()

    weight = coverage_weight_for_probe(
        runtime, "P", environ={"COVERAGE_WEIGHTS": "other:1,P:4.25"}
    )

    assert weight == pytest.approx(4.25)


def test_resolve_plan_target_lps_can_use_supplied_point_cloud() -> None:
    runtime = _runtime_with_probe()
    plan = runtime.plan_state.probes["P"]
    retro_points = np.array([[10.0, 20.0, 30.0], [12.0, 22.0, 32.0]], dtype=float)

    target = resolve_plan_target_lps(
        plan,
        runtime.plan_state.target_index,
        target_points_LPS=retro_points,
    )

    assert np.allclose(target, [11.0, 21.0, 31.0])
