"""Phase-2 diagnostics: logged IPOPT parity, start-pose jitter, rank lists, and the
slack-group split of the Phase-2 constraint vector."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

cyipopt = pytest.importorskip("cyipopt")

from aind_rutter.optimization.geometry.holes import Hole  # noqa: E402
from aind_rutter.optimization.geometry.primitives import HoleSection  # noqa: E402
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo  # noqa: E402
from aind_rutter.optimization.objectives.constrained import make_phase2  # noqa: E402
from aind_rutter.optimization.objectives.slack_layout import (  # noqa: E402
    PADDED_SLACK,
    SLACK_GROUPS,
)
from aind_rutter.optimization.objectives.soft import (  # noqa: E402
    PHASE1_PER_PROBE_VARS,
)
from aind_rutter.optimization.objectives.statics import (  # noqa: E402
    _build_probe_static,
)
from aind_rutter.optimization.pipeline.phase2_diagnostics import (  # noqa: E402
    PER_PROBE_VARS,
    minimize_ipopt_logged,
    perturb_pose,
    read_ranks,
)


def _toy_problem():
    def fun(x):
        return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2 + 0.1 * x[2] ** 2

    def jac(x):
        return np.array([2 * (x[0] - 1.0), 2 * (x[1] - 2.0), 0.2 * x[2]])

    def con(x):
        # Last entry mimics a padded constraint.
        return np.array([1.0 - x[0] ** 2 - x[1] ** 2, x[0] + x[2], PADDED_SLACK])

    def con_jac(x):
        return np.array([[-2 * x[0], -2 * x[1], 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    constraints = [{"type": "ineq", "fun": con, "jac": con_jac}]
    bounds = [(-2.0, 2.0)] * 3
    options = {
        "hessian_approximation": "limited-memory",
        "max_iter": 200,
        "tol": 1e-8,
        "print_level": 0,
        "sb": "yes",
    }
    return fun, jac, constraints, bounds, options


def test_logged_solve_is_identical_to_minimize_ipopt() -> None:
    fun, jac, constraints, bounds, options = _toy_problem()
    x0 = np.array([0.5, 0.5, 0.0])
    ref = cyipopt.minimize_ipopt(
        fun, x0, jac=jac, bounds=bounds, constraints=constraints, options=dict(options)
    )
    res, hist = minimize_ipopt_logged(
        fun,
        x0,
        jac=jac,
        bounds=bounds,
        constraints=constraints,
        options=dict(options),
        group_sizes=(2, 1),
        padded_slack=PADDED_SLACK,
    )
    np.testing.assert_array_equal(res.x, ref.x)
    assert res.nit == ref.nit and res.status == ref.status

    assert hist["iter"][0] == 0 and hist["iter"][-1] == res.nit
    assert hist["group_min"].shape == (len(hist["iter"]), 2)
    g_final = constraints[0]["fun"](res.x)
    assert hist["group_min"][-1, 0] == pytest.approx(g_final[:2].min(), abs=1e-6)
    # The padded group has no real constraints.
    assert np.isnan(hist["group_min"][:, 1]).all()
    assert (hist["group_nviol"][:, 1] == 0).all()


def test_logged_solve_releases_the_objective_without_gc() -> None:
    fun, jac, constraints, bounds, options = _toy_problem()

    class Objective:
        def __call__(self, x):
            return fun(x)

    objective = Objective()
    alive = weakref.ref(objective)
    gc.disable()
    try:
        minimize_ipopt_logged(
            objective,
            np.array([0.5, 0.5, 0.0]),
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options=dict(options),
            group_sizes=(2, 1),
        )
        del objective
        assert alive() is None
    finally:
        gc.enable()


def _pose_and_bounds(n_arcs: int = 2, n_probes: int = 3):
    pose = np.zeros(n_arcs + PER_PROBE_VARS * n_probes)
    pose[:n_arcs] = [10.0, -20.0]
    for i in range(n_probes):
        b = n_arcs + PER_PROBE_VARS * i
        pose[b], pose[b + 1], pose[b + 2] = 5.0 * i, 0.6, 0.8
    bounds = [(-60.0, 60.0)] * n_arcs + [
        (-45.0, 45.0),
        (-1.1, 1.1),
        (-1.1, 1.1),
        (-3.0, 3.0),
        (-3.0, 3.0),
        (-2.0, 2.0),
    ] * n_probes
    return pose, bounds


def test_perturb_pose_is_reproducible_bounded_and_keeps_spin_radius() -> None:
    assert PER_PROBE_VARS == PHASE1_PER_PROBE_VARS
    pose, bounds = _pose_and_bounds()
    a = perturb_pose(pose, 2, 3, bounds, scale=1.0, seed=(0, 7))
    np.testing.assert_array_equal(a, perturb_pose(pose, 2, 3, bounds, 1.0, (0, 7)))
    assert not np.array_equal(a, perturb_pose(pose, 2, 3, bounds, 1.0, (0, 8)))
    np.testing.assert_array_equal(perturb_pose(pose, 2, 3, bounds, 0.0, (0, 7)), pose)
    np.testing.assert_allclose(np.hypot(a[3::6], a[4::6]), 1.0)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    wild = perturb_pose(pose, 2, 3, bounds, scale=1000.0, seed=(0, 7))
    assert ((wild >= lo) & (wild <= hi)).all()


def test_read_ranks_accepts_commas_lines_and_comments(tmp_path) -> None:
    path = tmp_path / "ranks.txt"
    path.write_text("3, 5\n# band 2\n7\n\n11 13  # tail\n")
    assert read_ranks(path) == [3, 5, 7, 11, 13]


def test_slack_parts_concatenate_to_the_constraint_vector() -> None:
    axis = np.array([0.0, 0.0, 1.0])
    sec = HoleSection(axis=axis, center=np.zeros(3), a=0.6, b=0.3, theta=0.0)
    hole = Hole(id=1, axis=axis, ref_point=np.zeros(3), sections=[sec])
    probes = [
        ProbeStaticInfo(
            name=name,
            target_LPS=np.array([dx, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=np.array([[0.0, 0.0, 0.0]]),
        )
        for name, dx in (("p", 0.0), ("q", 1.0))
    ]
    ha = SimpleNamespace(probe_to_hole={"p": 1, "q": 1})
    aa = SimpleNamespace(probe_to_arc_idx={"p": 0, "q": 0}, arc_centroids_deg=(0.0,))
    statics = _build_probe_static(probes, [hole], ha, aa)
    p2 = make_phase2(statics, 1)

    x = np.array(
        [0.0] + [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [20.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    )
    parts = p2["slack_parts"](x)
    assert tuple(parts) == SLACK_GROUPS
    flat = np.concatenate([part.reshape(-1) for part in parts.values()])
    np.testing.assert_array_equal(flat, p2["constraints"][0]["fun"](x))
    assert parts["thread"].ndim == 3 and parts["thread"].shape[0] == 2
    assert parts["ml_sep"].shape == (1,)
    assert set(p2["slack_labels"]) >= {"probe_pairs", "fixtures", "pair_categories"}
