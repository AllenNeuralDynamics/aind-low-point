"""Tests for ``arc_first_principled.ml_anchors_mrv`` — the MRV/CSP ML+spin
anchor selector for the seed emitter."""

from __future__ import annotations

import numpy as np

from aind_low_point.optimization.arc_first_principled import ml_anchors_mrv

_SEP = 16.0


def _arr(*x):
    return np.array(x, dtype=float)


def _separated(res):
    mls = [r[0] for r in res]
    return all(
        abs(mls[i] - mls[j]) >= _SEP - 1e-9
        for i in range(len(mls))
        for j in range(i + 1, len(mls))
    )


def test_ml_mrv_nearest_ap_and_separated():
    # (ml, spin, ap) per anchor. p0 nearest-AP anchor is ml=0; p1 has one.
    anchors = [
        (_arr(0, 30, 60), _arr(10, 20, 30), _arr(0, 1, 2)),
        (_arr(20.0), _arr(40.0), _arr(0.5)),
    ]
    res = ml_anchors_mrv(anchors, target_ap=0.0, min_ml_sep_deg=_SEP)
    assert _separated(res)
    assert res[0][0] == 0.0           # p0 got its nearest-AP anchor
    assert res[1] == (20.0, 40.0, 0.5)


def test_ml_mrv_resolves_conflict_via_most_constrained_first():
    # p1 has a single anchor (ml=10); p0's nearest (ml=0) conflicts, so p0
    # must take its far anchor (ml=100). MRV places p1 first and p0 adapts.
    anchors = [
        (_arr(0, 100), _arr(1, 2), _arr(0, 5)),
        (_arr(10.0), _arr(9.0), _arr(0.5)),
    ]
    res = ml_anchors_mrv(anchors, target_ap=0.0, min_ml_sep_deg=_SEP)
    assert _separated(res)
    assert res[1][0] == 10.0
    assert res[0][0] == 100.0


def test_ml_mrv_infeasible_returns_none():
    anchors = [(_arr(0.0), _arr(1.0), _arr(0.0)), (_arr(5.0), _arr(1.0), _arr(0.0))]
    assert ml_anchors_mrv(anchors, target_ap=0.0, min_ml_sep_deg=_SEP) is None


def test_ml_mrv_emits_spin_from_chosen_anchor():
    anchors = [(_arr(0, 30), _arr(111, 222), _arr(0, 1))]
    res = ml_anchors_mrv(anchors, target_ap=0.0, min_ml_sep_deg=_SEP)
    assert res[0] == (0.0, 111.0, 0.0)   # spin tracks the ml=0 anchor


def test_ml_mrv_empty():
    assert ml_anchors_mrv([], target_ap=0.0, min_ml_sep_deg=_SEP) == []


def test_ml_mrv_finds_all_feasible_random():
    rng = np.random.default_rng(3)
    for _ in range(400):
        n = int(rng.integers(2, 7))
        tgt = np.cumsum(_SEP + rng.uniform(0, 20, n))
        tgt -= tgt.mean()                          # a guaranteed-separable set
        anchors = []
        for k in range(n):
            mls = np.concatenate([[tgt[k]], rng.uniform(-90, 90, rng.integers(0, 6))])
            spins = rng.uniform(-180, 180, len(mls))
            aps = rng.uniform(-50, 50, len(mls))
            anchors.append((mls, spins, aps))
        res = ml_anchors_mrv(anchors, float(rng.uniform(-40, 40)), _SEP)
        assert res is not None, "missed a feasible instance"
        assert _separated(res)
