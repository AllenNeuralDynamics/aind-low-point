"""Tests for arc AP placement helpers."""

from __future__ import annotations

import numpy as np

from aind_low_point.optimization.arc_placement import bounded_isotonic_arc_aps

_SEP = 16.0
_LO, _HI = -60.0, 60.0


def _max_violation(ap, lo, hi, c):
    """Max of separation shortfall and box overflow; zero means feasible."""
    order = np.argsort(c)
    gaps = np.diff(ap[order])
    sep_v = float((_SEP - gaps).max()) if gaps.size else 0.0
    box_v = max(float((lo - ap).max()), float((ap - hi).max()))
    return max(0.0, sep_v, box_v)


def _qp_oracle(c, lo, hi):
    """Independent SLSQP solve of the same projection."""
    from scipy.optimize import minimize

    order = np.argsort(c)
    cons = [
        {"type": "ineq", "fun": (lambda a, i=i: a[order[i + 1]] - a[order[i]] - _SEP)}
        for i in range(len(c) - 1)
    ]
    res = minimize(
        lambda a: float(np.sum((a - c) ** 2)),
        np.clip(c, lo, hi).astype(float),
        method="SLSQP",
        bounds=list(zip(lo, hi)),
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 800},
    )
    return res.x


def test_bounded_isotonic_slack_unchanged():
    c = np.array([-40.0, -10.0, 20.0])
    out = bounded_isotonic_arc_aps(c, np.full(3, _LO), np.full(3, _HI), _SEP)
    np.testing.assert_allclose(out, c, atol=1e-9)


def test_bounded_isotonic_binding_splits_deficit():
    c = np.array([0.0, 5.0, 10.0])
    out = bounded_isotonic_arc_aps(c, np.full(3, _LO), np.full(3, _HI), _SEP)
    np.testing.assert_allclose(out, [-11.0, 5.0, 21.0], atol=1e-7)


def test_bounded_isotonic_box_clamp_stays_feasible():
    c = np.array([0.0, 5.0, 10.0])
    lo = np.array([_LO, _LO, _LO])
    hi = np.array([_HI, _HI, 18.0])
    out = bounded_isotonic_arc_aps(c, lo, hi, _SEP)
    assert _max_violation(out, lo, hi, c) < 1e-7
    np.testing.assert_allclose(out, _qp_oracle(c, lo, hi), atol=1e-5)


def _feasible_instance(rng):
    n = int(rng.integers(2, 8))
    slack = (_HI - _LO) - (n - 1) * _SEP
    extra = rng.uniform(0.0, 1.0, n - 1)
    extra *= rng.uniform(0.0, slack) / max(extra.sum(), 1e-9)
    true = np.concatenate([[0.0], np.cumsum(_SEP + extra)])
    true = true + rng.uniform(_LO - true.min(), _HI - true.max())
    c = np.clip(true + rng.uniform(-8.0, 8.0, n), _LO, _HI)
    lo = np.maximum(np.minimum(true, c) - rng.uniform(1.0, 12.0, n), _LO)
    hi = np.minimum(np.maximum(true, c) + rng.uniform(1.0, 12.0, n), _HI)
    return c, lo, hi


def test_bounded_isotonic_feasible_and_l2_optimal_random():
    rng = np.random.default_rng(7)
    worst_v = worst_obj = 0.0
    for _ in range(200):
        c, lo, hi = _feasible_instance(rng)
        out = bounded_isotonic_arc_aps(c, lo, hi, _SEP)
        worst_v = max(worst_v, _max_violation(out, lo, hi, c))
        oracle = _qp_oracle(c, lo, hi)
        if _max_violation(oracle, lo, hi, c) < 1e-6:
            worst_obj = max(
                worst_obj,
                abs(float(np.sum((out - c) ** 2)) - float(np.sum((oracle - c) ** 2))),
            )
    assert worst_v < 1e-6, f"infeasible output: {worst_v}"
    assert worst_obj < 1e-4, f"not L2-optimal: {worst_obj}"
