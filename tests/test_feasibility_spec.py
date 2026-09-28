"""One feasibility spec, read the same way by every stage that tests a plan.

Phase 2 constrains ``gain · slack ≥ 0`` and Phase 1 penalises the squared
shortfall of the same rows, so on any pose Phase 1's feasibility penalty is the
weighted sum of squares of Phase 2's negative rows. The problem here is small
enough to compile in seconds: three box-bodied probes, one bore each and one box
fixture, posed once clear of every condition and once violating each.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from aind_rutter.optimization.clearance.voxel_sdf import build_probe_sdf
from aind_rutter.optimization.geometry.holes import Hole
from aind_rutter.optimization.geometry.primitives import HoleSection
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo
from aind_rutter.optimization.objectives.constrained import Phase2Weights, make_phase2
from aind_rutter.optimization.objectives.feasibility import (
    SOFTMIN_BIAS_MEASURED_AT,
    FeasibilitySpec,
)
from aind_rutter.optimization.objectives.sdf_inputs import FixtureSDFData
from aind_rutter.optimization.objectives.soft import (
    PACKED_ARG_ORDER,
    Phase1Weights,
    _build_jit,
    _signature,
    pack_statics,
)
from aind_rutter.optimization.objectives.statics import _build_probe_static
from aind_rutter.optimization.search.spin_restore import RestoreWeights

LAMBDA_THREAD, LAMBDA_PAIR, LAMBDA_FIXTURE, LAMBDA_KINEMATIC = 1.0, 10.0, 30.0, 0.01

# Targets along x; the fixture sits 0.5 mm past the third probe's body.
TARGETS_X = (-2.5, 0.0, 6.0)
ARC_OF = (0, 0, 1)
FIXTURE_X = (7.5, 9.5)

# (arc APs, then (ml, sx, sy, off_R, off_A, depth) per probe).
CLEAR = (
    [-10.0, 10.0]
    + [10.0, 1, 0, 0, 0, 0]
    + [-10.0, 1, 0, 0, 0, 0]
    + [0.0, 1, 0, 0, 0, 0]
)
# Arcs 5° apart, the first two probes 8° apart and tilted into each other, the
# second 0.5 mm off its bore, the third tilted into the fixture.
VIOLATING = (
    [0.0, 5.0] + [-8.0, 1, 0, 0, 0, 0] + [0.0, 1, 0, 0.5, 0, 0] + [-8.0, 1, 0, 0, 0, 0]
)


def _box_sdf(extents, center):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return build_probe_sdf(
        mesh, spacing_mm=0.2, n_surface_points=500, use_cache=False,
        sign_type="pseudonormal",
    )  # fmt: skip


@pytest.fixture(scope="module")
def problem():
    """Statics and fixture for three single-shank probes with 2 × 2 × 4 mm bodies."""
    body = replace(
        _box_sdf((2.0, 2.0, 4.0), (0.0, 0.0, 5.0)),
        shank_centers=np.array([[0.0, 0.0, 1.5]]),
        shank_halves=np.array([[0.035, 0.012, 1.5]]),
    )
    well = _box_sdf(
        (FIXTURE_X[1] - FIXTURE_X[0], 4.0, 4.0), (sum(FIXTURE_X) / 2, 0.0, 0.5)
    )
    fixture = FixtureSDFData(
        name="well",
        grid=jnp.asarray(well.grid, jnp.float32),
        origin=jnp.asarray(well.origin, jnp.float32),
        spacing=jnp.asarray(well.spacing, jnp.float32),
        surface=jnp.asarray(well.surface_points, jnp.float32),
    )
    axis = np.array([0.0, 0.0, 1.0])
    names = [f"p{i}" for i in range(len(TARGETS_X))]
    probes = [
        ProbeStaticInfo(
            name=name,
            target_LPS=np.array([x, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=np.zeros((1, 3)),
        )
        for name, x in zip(names, TARGETS_X)
    ]
    holes = [
        Hole(
            id=i + 1,
            axis=axis,
            ref_point=np.array([x, 0.0, -3.0]),
            sections=[
                HoleSection(
                    axis=axis, center=np.array([x, 0.0, -3.0]), a=0.6, b=0.3, theta=0.0
                )
            ],
        )
        for i, x in enumerate(TARGETS_X)
    ]
    statics = _build_probe_static(
        probes,
        holes,
        SimpleNamespace(probe_to_hole={n: i + 1 for i, n in enumerate(names)}),
        SimpleNamespace(
            probe_to_arc_idx=dict(zip(names, ARC_OF)), arc_centroids_deg=(0.0, 0.0)
        ),
        sdf_by_name={n: body for n in names},
    )
    return statics, (fixture,)


def _phase1_penalty(statics, fixtures, spec, x) -> float:
    """Phase 1's objective with only the feasibility penalties weighted."""
    weights = Phase1Weights(
        spec,
        lambda_thread=LAMBDA_THREAD,
        lambda_clearance=LAMBDA_PAIR,
        lambda_clearance_fixture=LAMBDA_FIXTURE,
        lambda_kinematic=LAMBDA_KINEMATIC,
        lambda_bounds=0.0,
        lambda_unit_circle=0.0,
        lambda_margin_clear=0.0,
        lambda_margin_thread=0.0,
    )
    n_arcs = 2
    objective, _ = _build_jit(
        _signature(statics, n_arcs, weights), weights, fixtures=fixtures
    )
    packed = pack_statics(statics, n_arcs)
    return float(
        objective(
            jnp.asarray(x, jnp.float32), **{k: packed[k] for k in PACKED_ARG_ORDER}
        )
    )


def _phase2_rows(statics, fixtures, spec, x) -> dict[str, np.ndarray]:
    p2 = make_phase2(
        statics, 2, fixtures=fixtures, weights=Phase2Weights(spec),
        grid_dtype=jnp.float32,
    )  # fmt: skip
    return {k: np.asarray(v, np.float64) for k, v in p2["slack_parts"](x).items()}


def _penalty_from_rows(rows: dict[str, np.ndarray]) -> float:
    def sq(group):
        return float(np.sum(np.maximum(0.0, -rows[group]) ** 2))

    return (
        LAMBDA_THREAD * sq("thread")
        + LAMBDA_PAIR * sq("probe_pair")
        + LAMBDA_FIXTURE * sq("probe_fixture")
        + LAMBDA_KINEMATIC * (sq("arc_sep") + sq("ml_sep"))
    )


SPECS = [
    pytest.param(
        FeasibilitySpec(min_clearance_mm=0.1, softmin_correction=False), id="raw"
    ),
    pytest.param(
        FeasibilitySpec(min_clearance_mm=0.1, softmin_correction=True), id="typical"
    ),
]


@pytest.mark.parametrize("spec", SPECS)
def test_phase1_charges_nothing_where_phase2_is_feasible(problem, spec) -> None:
    statics, fixtures = problem
    rows = _phase2_rows(statics, fixtures, spec, CLEAR)
    assert all(np.all(r >= 0) for r in rows.values()), {
        k: float(r.min()) for k, r in rows.items() if r.size
    }
    assert _phase1_penalty(statics, fixtures, spec, CLEAR) == 0.0


@pytest.mark.parametrize("spec", SPECS)
def test_phase1_penalises_exactly_phase2s_violations(problem, spec) -> None:
    statics, fixtures = problem
    rows = _phase2_rows(statics, fixtures, spec, VIOLATING)
    violated = {k for k, r in rows.items() if np.any(r < 0)}
    assert violated == {"thread", "probe_pair", "probe_fixture", "arc_sep", "ml_sep"}
    assert _phase1_penalty(statics, fixtures, spec, VIOLATING) == pytest.approx(
        _penalty_from_rows(rows), rel=1e-4
    )


def test_the_correction_refuses_a_soft_minimum_it_was_not_measured_at() -> None:
    for name, measured in SOFTMIN_BIAS_MEASURED_AT.items():
        with pytest.raises(ValueError, match="re-measure"):
            FeasibilitySpec(
                min_clearance_mm=0.1, softmin_correction=True, **{name: measured * 2}
            )
        FeasibilitySpec(
            min_clearance_mm=0.1, softmin_correction=False, **{name: measured * 2}
        )


@pytest.mark.parametrize("weights_cls", [RestoreWeights, Phase1Weights, Phase2Weights])
def test_every_stage_states_its_spec(weights_cls) -> None:
    """No stage falls back on a margin of its own."""
    with pytest.raises(TypeError):
        weights_cls()
