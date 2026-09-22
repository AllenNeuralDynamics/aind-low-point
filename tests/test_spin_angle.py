"""Solving over spin angles through the full-layout model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aind_rutter.optimization.objectives.layout import (
    PHASE1_PER_PROBE_VARS,
    SPIN_COS,
    SPIN_SIN,
)
from aind_rutter.optimization.objectives.spin_angle import (
    SpinAngle,
    from_angle_layout,
    to_angle_layout,
)
from aind_rutter.optimization.pipeline.fixtures import phase1_bounds
from aind_rutter.optimization.pipeline.phase1 import angle_bounds
from aind_rutter.optimization.pipeline.settings import Phase1Settings, Phase2Settings
from aind_rutter.optimization.search.minimizers import make_staged_rprop

N_ARCS, N_PROBES = 2, 3
N_VARS = N_ARCS + PHASE1_PER_PROBE_VARS * N_PROBES
COS = N_ARCS + PHASE1_PER_PROBE_VARS * np.arange(N_PROBES) + SPIN_COS
SIN = N_ARCS + PHASE1_PER_PROBE_VARS * np.arange(N_PROBES) + SPIN_SIN


def _model_point(rng) -> np.ndarray:
    x = rng.normal(size=N_VARS)
    radii = rng.uniform(0.3, 1.1, N_PROBES)
    theta = rng.uniform(-np.pi, np.pi, N_PROBES)
    x[COS], x[SIN] = radii * np.cos(theta), radii * np.sin(theta)
    return x


def _model_constraints(x):
    """A smooth vector function reading every variable, spins through arctan2."""
    spin = np.arctan2(x[SIN], x[COS])
    return np.concatenate([np.sin(x) * x[0], np.cos(2 * spin) + x[1] * spin])


def _model_constraints_jac(x):
    m = _model_constraints(x).size
    J = np.zeros((m, N_VARS))
    J[:N_VARS] = np.diag(np.cos(x) * x[0])
    J[:N_VARS, 0] += np.sin(x)
    spin = np.arctan2(x[SIN], x[COS])
    r2 = x[SIN] ** 2 + x[COS] ** 2
    d_spin = -2 * np.sin(2 * spin) + x[1]
    for k in range(N_PROBES):
        row = N_VARS + k
        J[row, COS[k]] = d_spin[k] * -x[SIN[k]] / r2[k]
        J[row, SIN[k]] = d_spin[k] * x[COS[k]] / r2[k]
        J[row, 1] = spin[k]
    return J


def test_round_trip_keeps_every_direction_and_puts_spins_on_the_circle() -> None:
    spin = SpinAngle(N_ARCS, N_PROBES, N_VARS)
    x = _model_point(np.random.default_rng(0))
    z = spin.to_solver(x)
    assert z.size == spin.n_solver_vars == N_VARS - N_PROBES
    back = spin.to_model(z)
    others = np.setdiff1d(np.arange(N_VARS), np.concatenate([COS, SIN]))
    np.testing.assert_array_equal(back[others], x[others])
    np.testing.assert_allclose(np.hypot(back[COS], back[SIN]), 1.0)
    np.testing.assert_allclose(
        np.arctan2(back[SIN], back[COS]), np.arctan2(x[SIN], x[COS])
    )
    np.testing.assert_allclose(spin.to_solver(back), z)


def test_pulled_back_derivatives_match_finite_differences_in_the_angle() -> None:
    spin = SpinAngle(N_ARCS, N_PROBES, N_VARS)
    x = _model_point(np.random.default_rng(1))
    objective = lambda x: float(np.sum(_model_constraints(x) ** 2))  # noqa: E731
    gradient = lambda x: 2 * _model_constraints(x) @ _model_constraints_jac(x)  # noqa: E731
    constraint = {
        "type": "ineq",
        "fun": _model_constraints,
        "jac": _model_constraints_jac,
    }
    fun, jac, (con,) = spin.wrap(objective, gradient, [constraint])
    assert con["type"] == "ineq"
    z = spin.to_solver(x)
    h = 1e-6
    steps = np.eye(z.size) * h
    fd_grad = np.array([(fun(z + e) - fun(z - e)) / (2 * h) for e in steps])
    fd_jac = np.stack(
        [(con["fun"](z + e) - con["fun"](z - e)) / (2 * h) for e in steps], 1
    )
    np.testing.assert_allclose(jac(z), fd_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(con["jac"](z), fd_jac, rtol=1e-5, atol=1e-6)


def test_angles_are_unbounded_and_other_bounds_carry_over() -> None:
    spin = SpinAngle(N_ARCS, N_PROBES, N_VARS)
    model_bounds = [(-float(i) - 1, float(i) + 1) for i in range(N_VARS)]
    solver_bounds = spin.bounds(model_bounds)
    kept = [i for i in range(N_VARS) if i not in set(SIN)]
    assert len(solver_bounds) == spin.n_solver_vars
    for slot, i in enumerate(kept):
        expected = (-np.inf, np.inf) if i in set(COS) else model_bounds[i]
        assert solver_bounds[slot] == expected


def test_angle_parameterization_reaches_both_stages_and_phase2_needs_ipopt(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUTTER_SPIN_PARAM", "angle")
    assert Phase1Settings().spin_param == Phase2Settings().spin_param == "angle"
    with pytest.raises(ValueError, match="ipopt"):
        Phase2Settings(solver="trust-constr")


def test_angle_layout_keeps_the_shape_and_round_trips() -> None:
    rng = np.random.default_rng(2)
    x = np.stack([_model_point(rng) for _ in range(3)]).astype(np.float32)
    z = to_angle_layout(x, N_ARCS)
    assert z.shape == x.shape
    np.testing.assert_array_equal(z[:, SIN], 0.0)
    back = np.asarray(from_angle_layout(z, N_ARCS))
    others = np.setdiff1d(np.arange(N_VARS), np.concatenate([COS, SIN]))
    np.testing.assert_array_equal(back[:, others], x[:, others])
    np.testing.assert_allclose(np.hypot(back[:, COS], back[:, SIN]), 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        np.arctan2(back[:, SIN], back[:, COS]),
        np.arctan2(x[:, SIN], x[:, COS]),
        rtol=1e-6,
    )


def test_angle_layout_gradient_is_the_ipopt_pull_back() -> None:
    """Autodiff through the in-place layout and the hand-written pull-back agree,
    and the sine slots, which the expansion overwrites, get no gradient."""
    weights = jnp.linspace(0.5, 2.0, N_VARS)

    def model(x):
        return jnp.sum(weights * jnp.sin(x)) + x[COS[0]] * x[SIN[1]]

    x = _model_point(np.random.default_rng(3)).astype(np.float32)
    z = to_angle_layout(x, N_ARCS)
    grad_z = np.asarray(jax.grad(lambda z: model(from_angle_layout(z, N_ARCS)))(z))
    np.testing.assert_array_equal(grad_z[SIN], 0.0)

    spin = SpinAngle(N_ARCS, N_PROBES, N_VARS)
    z_ipopt = spin.to_solver(x)
    grad_x = np.asarray(jax.grad(model)(jnp.asarray(spin.to_model(z_ipopt))))
    expected = spin.pull_back(z_ipopt, grad_x)
    kept = np.setdiff1d(np.arange(N_VARS), SIN)
    np.testing.assert_allclose(grad_z[kept], expected, rtol=1e-5, atol=1e-6)


def _cosine_well_grad(target):
    """RProp gradient of ``1 - cos(x - target)``, per coordinate and candidate."""
    target = jnp.asarray(target, jnp.float32)
    return lambda x, cov_weight: jnp.sin(x - target)


def test_rprop_wraps_a_periodic_coordinate_that_a_clip_would_stop() -> None:
    # The first minimum lies 0.28 rad away across the +-pi seam; the second
    # coordinate is an ordinary bounded one.
    target, lo, hi = [3.0, 0.5], [-np.pi, -1.0], [np.pi, 1.0]
    x0 = jnp.asarray([[-3.0, -0.5]], jnp.float32)
    grad = _cosine_well_grad(target)
    wrapped = make_staged_rprop(grad, periodic=np.array([True, False]))
    clipped = make_staged_rprop(grad)
    x_wrapped = np.asarray(wrapped(x0, [], lo, hi, 0.0, 200))[0]
    x_clipped = np.asarray(clipped(x0, [], lo, hi, 0.0, 200))[0]
    assert abs(np.angle(np.exp(1j * (x_wrapped[0] - target[0])))) < 1e-4
    assert -np.pi <= x_wrapped[0] < np.pi
    np.testing.assert_allclose(x_wrapped[1], target[1], atol=1e-4)
    np.testing.assert_allclose(x_clipped[0], -np.pi, rtol=1e-6)


def test_rprop_with_nothing_periodic_matches_the_clipped_run() -> None:
    target, lo, hi = [0.9, -0.4], [-1.0, -1.0], [0.5, 1.0]
    x0 = jnp.asarray([[0.0, 0.8], [0.3, -0.9]], jnp.float32)
    grad = _cosine_well_grad(target)
    none_wrap = make_staged_rprop(grad, periodic=np.zeros(2, bool))
    np.testing.assert_array_equal(
        np.asarray(none_wrap(x0, [], lo, hi, 0.0, 50)),
        np.asarray(make_staged_rprop(grad)(x0, [], lo, hi, 0.0, 50)),
    )


def test_phase1_angle_steps_as_far_as_its_cosine_component() -> None:
    bounds = np.array(phase1_bounds(N_ARCS, N_PROBES), np.float32)
    lo, hi = bounds[:, 0], bounds[:, 1]
    lo_a, hi_a, periodic, step_scale = angle_bounds(lo, hi, N_ARCS)
    np.testing.assert_array_equal(np.flatnonzero(periodic), COS)
    np.testing.assert_allclose(hi_a[COS] - lo_a[COS], 2 * np.pi, rtol=1e-6)
    np.testing.assert_array_equal(lo_a[SIN], hi_a[SIN])
    np.testing.assert_allclose(
        (step_scale * (hi_a - lo_a))[COS], (hi - lo)[COS], rtol=1e-6
    )
    others = np.setdiff1d(np.arange(N_VARS), np.concatenate([COS, SIN]))
    np.testing.assert_array_equal(lo_a[others], lo[others])
    np.testing.assert_array_equal(step_scale[others], 1.0)
