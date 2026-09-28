"""Every weight a traced kernel reads has to be in the key that caches it.

A weight left out is baked into the compiled kernel at whatever value it had on
the first build, and every later build with a different value silently gets the
old one back. The key is derived from the dataclass, the spec's fields
included, rather than from a hand-written list of names.

The tests are structural — they compare the key against the dataclass — because
an omission is invisible in any single run's output.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

from aind_rutter.optimization.objectives.constrained import (
    Phase2Weights,
)
from aind_rutter.optimization.objectives.constrained import (
    _weights_key as phase2_key,
)
from aind_rutter.optimization.objectives.feasibility import FeasibilitySpec
from aind_rutter.optimization.objectives.objective_spec import ObjectiveSpec
from aind_rutter.optimization.objectives.soft import (
    Phase1Weights,
)
from aind_rutter.optimization.objectives.soft import (
    _weights_key as phase1_key,
)

SPEC = FeasibilitySpec(min_clearance_mm=0.1, softmin_correction=False)
OBJECTIVE = ObjectiveSpec()

CASES = [
    pytest.param(Phase1Weights, phase1_key, id="phase1"),
    pytest.param(Phase2Weights, phase2_key, id="phase2"),
]


def _variants(value, name: str):
    """(name, changed value) for every field of ``value``, into nested dataclasses."""
    if isinstance(value, bool):
        yield name, not value
    elif isinstance(value, (int, float)):
        yield name, type(value)(value + 7)
    elif dc.is_dataclass(value):
        for field in dc.fields(value):
            for sub, changed in _variants(getattr(value, field.name), field.name):
                yield f"{name}.{sub}", dc.replace(value, **{field.name: changed})


@pytest.mark.parametrize(("weights_cls", "key"), CASES)
def test_changing_any_weight_changes_the_key(weights_cls, key) -> None:
    """The defect: `lambda_unit_circle` moved the objective and not the key, so
    a kernel traced at one value was reused at another."""
    base = weights_cls(SPEC, OBJECTIVE)
    collisions = []
    for field in dc.fields(base):
        for name, changed in _variants(getattr(base, field.name), field.name):
            if key(dc.replace(base, **{field.name: changed})) == key(base):
                collisions.append(name)
    assert not collisions, f"weights absent from the cache key: {collisions}"


@pytest.mark.parametrize(("weights_cls", "key"), CASES)
def test_the_key_is_hashable(weights_cls, key) -> None:
    """It is a dict key in the module-level compile cache."""
    hash(key(weights_cls(SPEC, OBJECTIVE)))


@pytest.mark.parametrize(("weights_cls", "key"), CASES)
def test_equal_weights_give_equal_keys(weights_cls, key) -> None:
    """Otherwise every build recompiles."""
    assert key(weights_cls(SPEC, OBJECTIVE)) == key(weights_cls(SPEC, OBJECTIVE))
