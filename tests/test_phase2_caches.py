"""What the Phase-2 compiled-kernel cache may and may not be reused for.

The key carries the shapes of the probe, fixture and brain grids but not their
values, and it never carried the coverage data at all, so an entry is only valid
for the inputs that built it. Two guarantees keep that safe: every weight is in
the key, and the worker setup clears the cache, so nothing survives from a
previous subject or well mode.
"""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from aind_rutter.optimization.objectives.constrained import (
    Phase2Weights,
    _signature,
    _weights_key,
    cache_stats,
    clear_jit_cache,
)


def _statics(n: int = 2, grid: tuple[int, int, int] = (4, 4, 4)) -> list:
    return [
        SimpleNamespace(sdf_data={"grid": np.zeros(grid), "surface": np.zeros((8, 3))})
        for _ in range(n)
    ]


def _fixture(grid: tuple[int, int, int] = (5, 5, 5)) -> SimpleNamespace:
    return SimpleNamespace(name="well", grid=np.zeros(grid))


def _bump(weights: Phase2Weights, name: str):
    """A copy of `weights` with one field changed to a different value."""
    current = getattr(weights, name)
    if isinstance(current, bool):
        return replace(weights, **{name: not current})
    if isinstance(current, int):
        return replace(weights, **{name: current + 1})
    return replace(weights, **{name: current + 1.0})


@pytest.mark.parametrize("field", [f.name for f in fields(Phase2Weights)])
def test_every_weight_reaches_the_cache_key(field: str) -> None:
    """A weight missing from the key is baked in and then reused at a new value."""
    base = Phase2Weights()
    assert _weights_key(base) != _weights_key(_bump(base, field))


def test_the_key_names_its_weights() -> None:
    """Named pairs, so reordering the dataclass cannot silently shift the key."""
    key = dict(_weights_key(Phase2Weights()))
    assert set(key) == {f.name for f in fields(Phase2Weights)}
    assert key["lambda_unit_circle"] == pytest.approx(10.0)


def test_float_noise_below_six_decimals_does_not_change_the_key() -> None:
    base = Phase2Weights()
    nudged = replace(base, lambda_cov=base.lambda_cov + 1e-9)
    assert _weights_key(base) == _weights_key(nudged)


def test_the_signature_separates_shapes_that_change_the_trace() -> None:
    weights = Phase2Weights()
    base = _signature(_statics(), 1, weights, (_fixture(),))
    assert base != _signature(_statics(n=3), 1, weights, (_fixture(),))
    assert base != _signature(_statics(), 2, weights, (_fixture(),))
    assert base != _signature(_statics(grid=(8, 4, 4)), 1, weights, (_fixture(),))
    assert base != _signature(_statics(), 1, weights, (_fixture(grid=(6, 5, 5)),))
    assert base != _signature(
        _statics(), 1, weights, (_fixture(),), SimpleNamespace(grid=np.zeros((3, 3, 3)))
    )


def test_the_signature_cannot_tell_two_subjects_of_one_shape_apart() -> None:
    """Why the cache has to be cleared rather than keyed harder.

    Hashing every grid on each of hundreds of candidate solves would cost more
    than it saves, so the key stays shape-based and `_init` drops the cache.
    """
    weights = Phase2Weights()
    one = _signature(
        _statics(), 1, weights, (SimpleNamespace(grid=np.zeros((5, 5, 5))),)
    )
    other = _signature(
        _statics(), 1, weights, (SimpleNamespace(grid=np.ones((5, 5, 5))),)
    )
    assert one == other


def test_clearing_drops_every_entry_and_resets_the_counters() -> None:
    from aind_rutter.optimization.objectives import constrained as phase2

    phase2._JIT_CACHE[("stale",)] = {"obj": None}
    phase2._CACHE_STATS["hits"] = 7
    clear_jit_cache()
    assert cache_stats() == {"hits": 0, "misses": 0, "entries": 0}
