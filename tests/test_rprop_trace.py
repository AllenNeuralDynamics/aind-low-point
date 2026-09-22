"""The RProp trace records how the loop behaves without changing what it does.

The channels exist to answer whether kinks freeze coordinates at ``eta_min`` and
whether coordinates pressed against a bound inflate ``η`` to its ceiling while
making no progress. Both are read off a run, so the recording must leave the
iterate untouched.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from aind_rutter.optimization.search.minimizers import (
    TRACE_CHANNELS,
    make_staged_rprop,
)

CH = {name: i for i, name in enumerate(TRACE_CHANNELS)}


def _cosine_well_grad(target):
    """RProp gradient of ``1 - cos(x - target)``, per coordinate and candidate."""
    target = jnp.asarray(target, jnp.float32)
    return lambda x, cov_weight: jnp.sin(x - target)


def test_tracing_does_not_move_the_iterate() -> None:
    target, lo, hi = [0.9, -0.4], [-1.0, -1.0], [0.5, 1.0]
    x0 = jnp.asarray([[0.0, 0.8], [0.3, -0.9]], jnp.float32)
    grad = _cosine_well_grad(target)
    plain = make_staged_rprop(grad)(x0, [], lo, hi, 0.0, 50)
    x, _eta, _stats = make_staged_rprop(grad, trace=50)(x0, [], lo, hi, 0.0, 50)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(x))


def test_trace_shape_and_final_eta_follow_the_batch() -> None:
    target, lo, hi = [0.9, -0.4], [-1.0, -1.0], [0.5, 1.0]
    x0 = jnp.asarray([[0.0, 0.8], [0.3, -0.9]], jnp.float32)
    run = make_staged_rprop(_cosine_well_grad(target), trace=30)
    x, eta, stats = run(x0, [], lo, hi, 0.0, 30)
    assert stats.shape == (30, 2, len(TRACE_CHANNELS))
    assert eta.shape == x0.shape
    rates = np.asarray(stats)[..., : CH["step"]]
    assert ((rates >= 0.0) & (rates <= 1.0)).all()
    assert (np.asarray(stats)[..., CH["step"]] >= 0.0).all()


def test_a_coordinate_held_against_its_bound_reports_clipping() -> None:
    # The well is 2-pi periodic, so target 5.0 is the minimum at -1.28: outside
    # the box, and the iterate ends pressed against ``lo`` with every step clipped.
    target, lo, hi = [5.0, -0.4], [-1.0, -1.0], [0.5, 1.0]
    x0 = jnp.asarray([[0.0, 0.8]], jnp.float32)
    run = make_staged_rprop(_cosine_well_grad(target), trace=40)
    x, _eta, stats = run(x0, [], lo, hi, 0.0, 40)
    np.testing.assert_allclose(np.asarray(x)[0, 0], -1.0, rtol=1e-6)
    assert np.asarray(stats)[-1, 0, CH["clipped"]] > 0.0


def test_rows_past_the_steps_taken_stay_zero() -> None:
    target, lo, hi = [0.9, -0.4], [-1.0, -1.0], [0.5, 1.0]
    x0 = jnp.asarray([[0.0, 0.8]], jnp.float32)
    run = make_staged_rprop(_cosine_well_grad(target), trace=20)
    _x, _eta, stats = run(x0, [], lo, hi, 0.0, 5)
    s = np.asarray(stats)
    assert s[:5].any()
    np.testing.assert_array_equal(s[5:], 0.0)


def test_steps_past_the_buffer_are_dropped_not_wrapped() -> None:
    target, lo, hi = [0.9, -0.4], [-1.0, -1.0], [0.5, 1.0]
    x0 = jnp.asarray([[0.0, 0.8]], jnp.float32)
    grad = _cosine_well_grad(target)
    short, long = 10, 60
    x_short, _e, stats = make_staged_rprop(grad, trace=short)(x0, [], lo, hi, 0.0, long)
    x_full = make_staged_rprop(grad)(x0, [], lo, hi, 0.0, long)
    # The iterate still takes all `long` steps; only the recording stops early.
    np.testing.assert_array_equal(np.asarray(x_short), np.asarray(x_full))
    assert np.asarray(stats).shape[0] == short


def test_pinned_coordinates_are_left_out_of_the_rates() -> None:
    # Second coordinate pinned by lo == hi: were it counted, no rate could reach 1.
    target = [0.9, -0.4]
    lo, hi = [-1.0, 0.25], [0.5, 0.25]
    x0 = jnp.asarray([[0.0, 0.25]], jnp.float32)
    run = make_staged_rprop(_cosine_well_grad(target), trace=30)
    _x, _eta, stats = run(x0, [], lo, hi, 0.0, 30)
    # The target sits above hi, so the one movable coordinate presses on its bound
    # without ever flipping sign: eta climbs 0.02 to 0.5 of the range, ~18 steps at
    # x1.2. A rate of 1.0 counts one of one, so the pinned coordinate is excluded.
    assert np.asarray(stats)[-1, 0, CH["eta_ceil"]] == 1.0
