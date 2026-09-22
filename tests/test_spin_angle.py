"""Solving over spin angles through the full-layout model."""

from __future__ import annotations

import numpy as np
import pytest

from aind_rutter.optimization.objectives.layout import (
    PHASE1_PER_PROBE_VARS,
    SPIN_COS,
    SPIN_SIN,
)
from aind_rutter.optimization.objectives.spin_angle import SpinAngle
from aind_rutter.optimization.pipeline.settings import Phase2Settings

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


def test_angle_parameterization_requires_ipopt() -> None:
    assert Phase2Settings(spin_param="angle").spin_param == "angle"
    with pytest.raises(ValueError, match="ipopt"):
        Phase2Settings(spin_param="angle", solver="trust-constr")
