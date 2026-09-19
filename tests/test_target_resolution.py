"""The app and the optimizer resolve a probe's target the same way.

They used to be separate implementations with opposite priority: the app read the
inline RAS point first, the optimizer the catalog key. Nothing could set both at
once, so the disagreement never fired — but it sat one careless assignment away
from sending the two halves of the program to different places.
"""

from __future__ import annotations

import numpy as np
import pytest

from aind_rutter.domain.plan import ProbePlan, resolve_target_LPS
from aind_rutter.runtime.probe_context import resolve_plan_target_lps

# RAS (1, 2, 3) is LPS (-1, -2, 3): L = -R, P = -A, S = S.
INLINE_RAS = (1.0, 2.0, 3.0)
INLINE_LPS = (-1.0, -2.0, 3.0)
KEYED_LPS = np.array([4.0, 5.0, 6.0])
INDEX = {"target:brain": KEYED_LPS}


def _plan(**fields) -> ProbePlan:
    return ProbePlan(kind="neuropixels", arc_id="a", **fields)


def test_a_keyed_target_comes_from_the_index() -> None:
    plan = _plan(target_key="target:brain")
    np.testing.assert_allclose(resolve_target_LPS(plan, INDEX), KEYED_LPS)


def test_an_inline_target_converts_from_ras() -> None:
    plan = _plan(target_point_RAS=INLINE_RAS)
    np.testing.assert_allclose(resolve_target_LPS(plan, INDEX), INLINE_LPS)


def test_a_point_cloud_target_is_averaged() -> None:
    cloud = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    plan = _plan(target_key="cloud")
    np.testing.assert_allclose(
        resolve_target_LPS(plan, {"cloud": cloud}), [1.0, 2.0, 3.0]
    )


def test_supplied_points_override_the_plan() -> None:
    plan = _plan(target_key="target:brain")
    cloud = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    resolved = resolve_target_LPS(plan, INDEX, points_LPS=cloud)
    np.testing.assert_allclose(resolved, [2.0, 2.0, 2.0])


def test_naming_a_target_both_ways_is_refused() -> None:
    """A state holding two targets is corrupt, not a case with a winner."""
    plan = _plan(target_key="target:brain", target_point_RAS=INLINE_RAS)
    with pytest.raises(ValueError, match="exactly one"):
        resolve_target_LPS(plan, INDEX)


@pytest.mark.parametrize(
    "plan",
    [
        ProbePlan(kind="np", arc_id="a"),
        ProbePlan(kind="np", arc_id="a", target_key="gone"),
    ],
    ids=["no target", "key not in index"],
)
def test_an_unresolvable_target_stops_the_optimizer(plan: ProbePlan) -> None:
    """The optimizer must not quietly aim at the origin."""
    with pytest.raises(RuntimeError):
        resolve_target_LPS(plan, INDEX, strict=True)


@pytest.mark.parametrize(
    "plan",
    [
        ProbePlan(kind="np", arc_id="a"),
        ProbePlan(kind="np", arc_id="a", target_key="gone"),
    ],
    ids=["no target", "key not in index"],
)
def test_an_unresolvable_target_lets_the_app_keep_drawing(plan: ProbePlan) -> None:
    with pytest.warns(UserWarning):
        np.testing.assert_allclose(resolve_target_LPS(plan, INDEX), np.zeros(3))


@pytest.mark.parametrize(
    "plan",
    [
        ProbePlan(kind="np", arc_id="a", target_key="target:brain"),
        ProbePlan(kind="np", arc_id="a", target_point_RAS=INLINE_RAS),
    ],
    ids=["keyed", "inline"],
)
def test_both_entry_points_agree(plan: ProbePlan) -> None:
    """The defect: two implementations, two answers for one plan."""
    np.testing.assert_allclose(
        resolve_plan_target_lps(plan, INDEX), resolve_target_LPS(plan, INDEX)
    )
