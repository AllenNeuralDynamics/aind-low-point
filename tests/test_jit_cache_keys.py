"""Every weight a traced kernel reads has to be in the key that caches it.

A weight left out is baked into the compiled kernel at whatever value it had on
the first build, and every later build with a different value silently gets the
old one back. D2 fixed this for Phase 2 by deriving the key from the dataclass
rather than a hand-written list of names; Phase 1 and the reduced objective kept
their lists, and both had drifted.

The tests are structural — they compare the key against the dataclass — because
an omission is invisible in any single run's output.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

from aind_rutter.optimization.objectives.phase1 import (
    Phase1Weights,
)
from aind_rutter.optimization.objectives.phase1 import (
    _weights_key as phase1_key,
)
from aind_rutter.optimization.objectives.phase2 import (
    Phase2Weights,
)
from aind_rutter.optimization.objectives.phase2 import (
    _weights_key as phase2_key,
)
from aind_rutter.optimization.objectives.probe_static import JointWeights
from aind_rutter.optimization.objectives.reduced_jax import (
    _weights_key as reduced_key,
)

CASES = [
    pytest.param(Phase1Weights, phase1_key, id="phase1"),
    pytest.param(JointWeights, reduced_key, id="reduced"),
    pytest.param(Phase2Weights, phase2_key, id="phase2"),
]


@pytest.mark.parametrize(("weights_cls", "key"), CASES)
def test_changing_any_weight_changes_the_key(weights_cls, key) -> None:
    """The defect: `lambda_unit_circle` moved the objective and not the key, so
    a kernel traced at one value was reused at another."""
    base = weights_cls()
    collisions = []
    for field in dc.fields(base):
        value = getattr(base, field.name)
        if isinstance(value, bool):
            changed = not value
        elif isinstance(value, (int, float)):
            changed = type(value)(value + 7)
        else:
            continue
        if key(dc.replace(base, **{field.name: changed})) == key(base):
            collisions.append(field.name)
    assert not collisions, f"weights absent from the cache key: {collisions}"


@pytest.mark.parametrize(("weights_cls", "key"), CASES)
def test_the_key_is_hashable(weights_cls, key) -> None:
    """It is a dict key in the module-level compile cache."""
    hash(key(weights_cls()))


@pytest.mark.parametrize(("weights_cls", "key"), CASES)
def test_equal_weights_give_equal_keys(weights_cls, key) -> None:
    """Otherwise every build recompiles."""
    assert key(weights_cls()) == key(weights_cls())
