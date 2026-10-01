"""A probe that records several regions: config, plan state and coverage."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from aind_rutter.build.assemble import build_plan_state_from_config
from aind_rutter.config import ConfigModel
from aind_rutter.domain.plan import RegionRef, resolve_region_points_LPS
from aind_rutter.optimization.clearance.poses import pose_from_optimizer_vars
from aind_rutter.optimization.objectives.coverage import (
    GaussianCoverageData,
    KdeCoverageData,
    _build_kde_grid,
    coverage_per_probe_over_probes,
    probe_coverage,
)
from aind_rutter.optimization.pipeline.probe_setup import region_weighted_cloud
from aind_rutter.plan_io.roundtrip import planning_state_to_plan_model
from tests.config_factories import ConfigFactory, TargetFactory


def _config(also_record) -> dict:
    data = ConfigFactory.config_with_probes()
    data["targets"].append(TargetFactory.explicit_target(key="target2"))
    probe = data["plan"]["probes"]["probe1"]
    probe["also_record"] = also_record
    probe["auto_scene"] = False  # the factory declares no probe mesh
    return data


def test_also_record_takes_the_target_shorthands() -> None:
    cfg = ConfigModel.model_validate(_config(["target2", [1.0, 2.0, 3.0]]))
    refs = cfg.plan.probes["probe1"].also_record
    assert [r.kind for r in refs] == ["catalog", "inline"]


def test_an_unknown_also_record_key_is_refused() -> None:
    with pytest.raises(ValidationError, match="catalog target 'nowhere' not found"):
        ConfigModel.model_validate(_config(["nowhere"]))


def test_region_points_put_the_target_first_and_convert_inline_to_LPS() -> None:
    cfg = ConfigModel.model_validate(_config(["target2", [1.0, 2.0, 3.0]]))
    state = build_plan_state_from_config(cfg)
    plan = state.probes["probe1"]
    assert plan.also_record == (
        RegionRef(key="target2"),
        RegionRef(point_LPS=(-1.0, -2.0, 3.0)),
    )
    index = {"target1": np.array([4.0, 5.0, 6.0]), "target2": np.zeros((2, 3))}
    np.testing.assert_allclose(
        resolve_region_points_LPS(plan, index),
        [[4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [-1.0, -2.0, 3.0]],
    )


def test_a_probe_without_also_record_has_no_region_points() -> None:
    state = build_plan_state_from_config(ConfigModel.model_validate(_config([])))
    assert resolve_region_points_LPS(state.probes["probe1"], {}) is None


def test_regions_name_one_structure_per_target() -> None:
    data = _config(["target2"])
    data["plan"]["probes"]["probe1"]["regions"] = ["MD"]
    with pytest.raises(ValidationError, match="regions names 1 structures for 2"):
        ConfigModel.model_validate(data)
    data["plan"]["probes"]["probe1"]["regions"] = ["MD", "RSP"]
    cfg = ConfigModel.model_validate(data)
    assert build_plan_state_from_config(cfg).probes["probe1"].regions == ("MD", "RSP")
    model = planning_state_to_plan_model(build_plan_state_from_config(cfg), cfg.plan)
    assert model.probes["probe1"].regions == ["MD", "RSP"]


def test_also_record_survives_writing_the_plan_back() -> None:
    cfg = ConfigModel.model_validate(_config(["target2"]))
    model = planning_state_to_plan_model(build_plan_state_from_config(cfg), cfg.plan)
    assert model.probes["probe1"].also_record == cfg.plan.probes["probe1"].also_record


TIPS = jnp.zeros((1, 3), jnp.float32)
MASK = jnp.ones(1, jnp.float32)
ANCHOR = np.array([0.0, 0.0, -4.0])


def _pose(ml: float = 0.0):
    return pose_from_optimizer_vars(
        target_LPS=jnp.asarray(ANCHOR, jnp.float32),
        ap_deg=jnp.float32(0.0),
        ml_deg=jnp.float32(ml),
        spin_deg=jnp.float32(0.0),
        offset_R_mm=jnp.float32(0.0),
        offset_A_mm=jnp.float32(0.0),
        past_target_mm=jnp.float32(0.0),
        recording_center_local=jnp.asarray([0.0, 0.0, 1.6], jnp.float32),
    )


def _data(points=None) -> GaussianCoverageData:
    return GaussianCoverageData(
        target_LPS=jnp.asarray(ANCHOR, jnp.float32),
        sigma_mm=0.5,
        active_start_mm=0.2,
        active_end_mm=3.065,
        region_points_LPS=None if points is None else jnp.asarray(points, jnp.float32),
    )


def test_each_region_scores_what_a_probe_aimed_at_it_alone_would() -> None:
    R, t = _pose()
    other = ANCHOR + [0.0, 0.0, 1.2]
    alone = [
        float(probe_coverage(R, t, TIPS, MASK, _data([p]))) for p in (ANCHOR, other)
    ]
    both = float(probe_coverage(R, t, TIPS, MASK, _data([ANCHOR, other])))
    assert both == pytest.approx(sum(alone), rel=1e-5)
    assert float(probe_coverage(R, t, TIPS, MASK, _data())) == alone[0]


def test_probes_with_different_region_counts_batch_like_one_at_a_time() -> None:
    poses = [_pose(0.0), _pose(10.0)]
    Rs = jnp.stack([p[0] for p in poses])
    ts = jnp.stack([p[1] for p in poses])
    data = (_data(), _data([ANCHOR, ANCHOR + [0.0, 0.5, 1.0], ANCHOR + [0.3, 0, 0]]))
    batched = coverage_per_probe_over_probes(
        Rs, ts, jnp.stack([TIPS, TIPS]), jnp.stack([MASK, MASK]), data
    )
    single = [probe_coverage(Rs[i], ts[i], TIPS, MASK, data[i]) for i in range(2)]
    np.testing.assert_allclose(np.asarray(batched), np.asarray(single), rtol=1e-6)


def _kde(points, weights=None) -> KdeCoverageData:
    grid, origin, spacing = _build_kde_grid(points, weights=weights, sigma_mm=0.3)
    return KdeCoverageData(
        grid=jnp.asarray(grid, jnp.float32),
        origin=jnp.asarray(origin, jnp.float32),
        spacing_mm=spacing,
        active_start_mm=0.2,
        active_end_mm=3.065,
    )


def _cloud(rng, centre, n) -> np.ndarray:
    return centre + rng.normal(0.0, 0.2, (n, 3))


def test_a_single_region_keeps_the_equal_weighting() -> None:
    cloud = _cloud(np.random.default_rng(0), ANCHOR, 40)
    points, weights = region_weighted_cloud([cloud])
    assert weights is None and points is cloud
    np.testing.assert_allclose(
        _build_kde_grid(cloud, weights=np.full(40, 1 / 40), sigma_mm=0.3)[0],
        _build_kde_grid(cloud, sigma_mm=0.3)[0],
    )


def test_each_retro_region_scores_what_its_own_cloud_would() -> None:
    rng = np.random.default_rng(1)
    dense = _cloud(rng, ANCHOR + [0.0, 0.0, 0.8], 400)
    sparse = _cloud(rng, ANCHOR + [0.0, 0.0, 2.2], 12)
    R, t = _pose()
    alone = [float(probe_coverage(R, t, TIPS, MASK, _kde(c))) for c in (dense, sparse)]
    points, weights = region_weighted_cloud([dense, sparse])
    # Each region deposits one kernel's mass, however many points it has.
    mass = _build_kde_grid(points, weights=weights, sigma_mm=0.3)[0].sum()
    one = _build_kde_grid(sparse, sigma_mm=0.3)[0].sum()
    assert mass == pytest.approx(2 * one, rel=1e-4)  # edge kernels lose a sliver
    # The builder snaps each point to a voxel, which can land one voxel apart
    # in grids with different bounds.
    both = float(probe_coverage(R, t, TIPS, MASK, _kde(points, weights)))
    assert both == pytest.approx(sum(alone), rel=5e-3)
