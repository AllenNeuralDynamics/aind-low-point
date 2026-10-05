"""One feasibility spec and one objective, read the same way by both phases.

Phase 2 minimizes the objective subject to ``gain · slack ≥ 0``, and Phase 1
minimizes the same objective plus the squared shortfall of the same rows, so on
any pose Phase 1's objective is Phase 2's plus the weighted sum of squares of
Phase 2's negative rows. The problem here is small enough to compile in seconds:
three box-bodied probes, one bore each and one box fixture, posed once clear of
every condition and once violating each.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from aind_rutter.optimization.clearance.aggregate import (
    fixture_slack_gains,
    probe_pair_slack_gains,
)
from aind_rutter.optimization.clearance.voxel_sdf import build_probe_sdf
from aind_rutter.optimization.geometry.holes import Hole
from aind_rutter.optimization.geometry.primitives import HoleSection
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo
from aind_rutter.optimization.objectives.constrained import Phase2Weights, make_phase2
from aind_rutter.optimization.objectives.coverage import GaussianCoverageData
from aind_rutter.optimization.objectives.feasibility import (
    CLEARANCE_CAP_MM,
    SOFTMIN_BIAS_MEASURED_AT,
    FeasibilitySpec,
)
from aind_rutter.optimization.objectives.metrics import make_min_clear_one
from aind_rutter.optimization.objectives.objective_spec import ObjectiveSpec
from aind_rutter.optimization.objectives.packing import (
    build_batched_probe_static,
    restore_arrays,
)
from aind_rutter.optimization.objectives.sdf_inputs import FixtureSDFData
from aind_rutter.optimization.objectives.slacks import (
    fixture_clearance_slack,
    pair_clearance_slack,
)
from aind_rutter.optimization.objectives.soft import (
    PACKED_ARG_ORDER,
    Phase1Weights,
    _build_jit,
    _signature,
    pack_statics,
)
from aind_rutter.optimization.objectives.statics import _build_probe_static
from aind_rutter.optimization.search.spin_restore import (
    RestoreWeights,
    make_batched_spin_restore_partial,
)

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
def parts():
    """Three single-shank probes with 2 × 2 × 4 mm bodies, their bores and
    assignments, and the fixture."""
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
    return SimpleNamespace(
        probes=probes,
        holes=holes,
        ha=SimpleNamespace(probe_to_hole={n: i + 1 for i, n in enumerate(names)}),
        aa=SimpleNamespace(
            probe_to_arc_idx=dict(zip(names, ARC_OF)), arc_centroids_deg=(0.0, 0.0)
        ),
        sdf_by_name={n: body for n in names},
        fixtures=(fixture,),
    )


@pytest.fixture(scope="module")
def problem(parts):
    """Statics and fixtures of ``parts``."""
    statics = _build_probe_static(
        parts.probes, parts.holes, parts.ha, parts.aa, sdf_by_name=parts.sdf_by_name
    )
    return statics, parts.fixtures


# The objective alone, so Phase 1's value is its feasibility penalty.
NO_OBJECTIVE = ObjectiveSpec(
    coverage_gain=0.0,
    lambda_bounds=0.0,
    lambda_unit_circle=0.0,
    lambda_margin_clear=0.0,
    lambda_margin_thread=0.0,
)


def _coverage(statics) -> tuple[GaussianCoverageData, ...]:
    return tuple(
        GaussianCoverageData(
            target_LPS=jnp.asarray(s.target_LPS, jnp.float32),
            sigma_mm=1.0,
            active_start_mm=0.0,
            active_end_mm=3.0,
        )
        for s in statics
    )


def _phase1_objective(statics, fixtures, spec, x, objective=NO_OBJECTIVE) -> float:
    weights = Phase1Weights(
        spec,
        objective,
        lambda_thread=LAMBDA_THREAD,
        lambda_clearance=LAMBDA_PAIR,
        lambda_clearance_fixture=LAMBDA_FIXTURE,
        lambda_kinematic=LAMBDA_KINEMATIC,
    )
    n_arcs = 2
    fn, _ = _build_jit(
        _signature(statics, n_arcs, weights),
        weights,
        coverage_data=_coverage(statics),
        fixtures=fixtures,
    )
    packed = pack_statics(statics, n_arcs)
    return float(
        fn(jnp.asarray(x, jnp.float32), **{k: packed[k] for k in PACKED_ARG_ORDER})
    )


def _phase2(statics, fixtures, spec, objective=NO_OBJECTIVE):
    return make_phase2(
        statics, 2, coverage_data=_coverage(statics), fixtures=fixtures,
        weights=Phase2Weights(spec, objective), grid_dtype=jnp.float32,
    )  # fmt: skip


def _phase2_rows(statics, fixtures, spec, x) -> dict[str, np.ndarray]:
    p2 = _phase2(statics, fixtures, spec)
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
    assert _phase1_objective(statics, fixtures, spec, CLEAR) == 0.0


@pytest.mark.parametrize("spec", SPECS)
def test_phase1_penalises_exactly_phase2s_violations(problem, spec) -> None:
    statics, fixtures = problem
    rows = _phase2_rows(statics, fixtures, spec, VIOLATING)
    violated = {k for k, r in rows.items() if np.any(r < 0)}
    assert violated == {"thread", "probe_pair", "probe_fixture", "arc_sep", "ml_sep"}
    assert _phase1_objective(statics, fixtures, spec, VIOLATING) == pytest.approx(
        _penalty_from_rows(rows), rel=1e-4
    )


# Every term live: the comfort caps sit inside the CLEAR pose's angles.
LIVE = ObjectiveSpec(
    coverage_gain=2.0, lambda_margin_clear=1.0, tau_clear_mm=0.5,
    comfortable_ap_deg=5.0, comfortable_ml_deg=5.0,
)  # fmt: skip
OBJECTIVES = [
    pytest.param(LIVE, id="hard"),
    pytest.param(
        replace(LIVE, smooth_clearance_reward=True, smooth_thread_reward=True),
        id="smooth",
    ),
]


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("spec", SPECS)
def test_phase1_minimizes_phase2s_objective_plus_its_violations(
    problem, spec, objective
) -> None:
    statics, fixtures = problem
    p2 = _phase2(statics, fixtures, spec, objective)
    for x in (CLEAR, VIOLATING):
        rows = {k: np.asarray(v, np.float64) for k, v in p2["slack_parts"](x).items()}
        assert _phase1_objective(
            statics, fixtures, spec, x, objective
        ) == pytest.approx(p2["fun"](x) + _penalty_from_rows(rows), rel=1e-5)
    # Each term moves the value, so equality is not two zeros agreeing.
    assert abs(p2["fun"](CLEAR) - _phase2(statics, fixtures, spec)["fun"](CLEAR)) > 0.1


@pytest.mark.parametrize("probe", range(len(TARGETS_X)))
def test_the_restore_charges_phase1s_penalty(parts, problem, probe) -> None:
    """Across a probe's spin sweep, the restore's loss and Phase 1's penalty
    differ by a constant: the terms that spin does not move."""
    statics, fixtures = problem
    spec = FeasibilitySpec(min_clearance_mm=0.1, softmin_correction=True)
    n_arcs, n_spins = 2, 8
    bs = build_batched_probe_static(
        [(parts.ha, parts.aa)], parts.probes, parts.holes, n_arcs=n_arcs,
        sdf_by_name=parts.sdf_by_name,
    )  # fmt: skip
    restore = make_batched_spin_restore_partial(
        bs,
        RestoreWeights(
            spec,
            lambda_thread=LAMBDA_THREAD,
            lambda_clearance=LAMBDA_PAIR,
            lambda_clearance_fixture=LAMBDA_FIXTURE,
        ),
        n_spins=n_spins,
        tables=pack_statics(statics, n_arcs)["sdf_table"],
        fixtures=fixtures,
    )
    # VIOLATING without its bore offset, which the restore pins at zero.
    arcs, mls = VIOLATING[:n_arcs], VIOLATING[n_arcs::6]
    y = np.array(arcs + [v for ml in mls for v in (ml, 1.0, 0.0)], np.float32)
    losses = np.asarray(
        restore.spin_losses(jnp.asarray(y), probe, *(a[0] for a in restore_arrays(bs)))
    )
    angles = np.asarray(
        jnp.linspace(0.0, 2.0 * jnp.pi, n_spins, endpoint=False).astype(jnp.float32)
    )
    phase1 = []
    for angle in angles:
        x = arcs + [v for ml in mls for v in (ml, 1.0, 0.0, 0.0, 0.0, 0.0)]
        x[n_arcs + 6 * probe + 1 : n_arcs + 6 * probe + 3] = (
            np.cos(angle),
            np.sin(angle),
        )
        phase1.append(_phase1_objective(statics, fixtures, spec, x))
    assert np.ptp(losses) > 1.0, "spin must move the penalty for the test to bite"
    np.testing.assert_allclose(
        np.asarray(phase1) - losses, phase1[0] - losses[0], atol=1e-4 * np.ptp(losses)
    )


@pytest.mark.parametrize("x", [CLEAR, VIOLATING], ids=["clear", "violating"])
def test_min_clear_is_the_worst_of_phase2s_clearance_rows(problem, x) -> None:
    statics, fixtures = problem
    spec = FeasibilitySpec(min_clearance_mm=0.1, softmin_correction=True)
    packed = pack_statics(statics, 2)
    min_clear = float(
        make_min_clear_one(2, len(statics), fixtures, spec)(
            jnp.asarray(x, jnp.float32), *(packed[k] for k in PACKED_ARG_ORDER)
        )
    )
    rows = _phase2_rows(statics, fixtures, spec, x)
    worst = min(
        np.min(rows["probe_pair"] / np.asarray(probe_pair_slack_gains())),
        np.min(rows["probe_fixture"] / np.asarray(fixture_slack_gains())),
    )
    assert min_clear == pytest.approx(worst, rel=1e-5, abs=1e-6)


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


def test_the_body_margin_holds_bodies_and_leaves_shanks() -> None:
    soft = jnp.full((1, 4), 0.3, jnp.float32)
    both = FeasibilitySpec(min_clearance_mm=0.1, softmin_correction=False)
    apart = replace(both, min_body_clearance_mm=0.5)
    np.testing.assert_allclose(pair_clearance_slack(soft, both), [[0.2] * 4], atol=1e-6)
    np.testing.assert_allclose(
        pair_clearance_slack(soft, apart), [[-0.2, 0.2, 0.2, 0.2]], atol=1e-6
    )
    np.testing.assert_allclose(
        fixture_clearance_slack(soft[:, :2], apart), [[0.2, 0.2]], atol=1e-6
    )


def test_the_body_fixture_margin_holds_bodies_against_fixtures_alone() -> None:
    soft = jnp.full((1, 4), 0.3, jnp.float32)
    spec = FeasibilitySpec(
        min_clearance_mm=0.1,
        softmin_correction=False,
        min_body_fixture_clearance_mm=0.4,
    )
    np.testing.assert_allclose(pair_clearance_slack(soft, spec), [[0.2] * 4], atol=1e-6)
    np.testing.assert_allclose(
        fixture_clearance_slack(soft[:, :2], spec), [[-0.1, 0.2]], atol=1e-6
    )


@pytest.mark.parametrize(
    "name",
    ["min_clearance_mm", "min_body_clearance_mm", "min_body_fixture_clearance_mm"],
)
def test_a_margin_past_the_clearance_cap_is_refused(name) -> None:
    kwargs = {"min_clearance_mm": 0.1, name: CLEARANCE_CAP_MM}
    with pytest.raises(ValueError, match="clearance rows read to"):
        FeasibilitySpec(softmin_correction=True, **kwargs)
