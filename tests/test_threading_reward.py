"""The worst-shank threading reward, hard and smoothed."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from aind_rutter.optimization.objectives.rewards import saturating_reward_worst

TAU = 0.5
VALID = jnp.ones((1, 3))


def _grad(slack, beta):
    return np.asarray(
        jax.grad(lambda s: saturating_reward_worst(s, TAU, VALID, beta=beta))(slack)
    )


def test_soft_reward_approaches_the_hard_one_as_beta_grows() -> None:
    slack = jnp.array([[0.30, 0.45, 0.80]])
    hard = float(saturating_reward_worst(slack, TAU, VALID))
    gaps = [
        abs(float(saturating_reward_worst(slack, TAU, VALID, beta=b)) - hard)
        for b in (5.0, 20.0, 200.0)
    ]
    assert gaps[0] > gaps[1] >= gaps[2]
    assert gaps[2] < 1e-5


def test_soft_reward_gradient_is_continuous_where_shanks_tie() -> None:
    """Across a tie the hard gradient moves wholesale from one shank to the
    other; the soft one barely changes."""
    eps = 1e-4
    before = jnp.array([[0.30 - eps, 0.30 + eps, 0.80]])
    after = jnp.array([[0.30 + eps, 0.30 - eps, 0.80]])
    hard_jump = np.abs(_grad(before, None) - _grad(after, None)).max()
    soft_jump = np.abs(_grad(before, 20.0) - _grad(after, 20.0)).max()
    assert hard_jump > 0.5
    assert soft_jump < 1e-2 * hard_jump


def test_soft_reward_gradient_is_continuous_at_zero_slack() -> None:
    """The hard gate has zero gradient below zero slack and 1/tau above it."""
    below = jnp.array([[-1e-4, 0.45, 0.80]])
    above = jnp.array([[1e-4, 0.45, 0.80]])
    assert abs(_grad(above, None)[0, 0] - _grad(below, None)[0, 0]) > 0.5
    assert abs(_grad(above, 20.0)[0, 0] - _grad(below, 20.0)[0, 0]) < 1e-2


def test_probe_without_valid_entries_adds_nothing_and_keeps_gradients_finite() -> None:
    slack = jnp.array([[0.30, 0.45, 0.80], [0.1, 0.2, 0.3]])
    valid = jnp.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    for beta in (None, 20.0):
        reward = saturating_reward_worst(slack, TAU, valid, beta=beta)
        alone = saturating_reward_worst(slack[:1], TAU, valid[:1], beta=beta)
        np.testing.assert_allclose(float(reward), float(alone), rtol=1e-6)
        grad = jax.grad(
            lambda s, b=beta: saturating_reward_worst(s, TAU, valid, beta=b)
        )(slack)
        assert np.isfinite(np.asarray(grad)).all()
        assert not np.asarray(grad)[1].any()
