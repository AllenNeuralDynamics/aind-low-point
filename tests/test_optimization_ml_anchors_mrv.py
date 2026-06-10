"""Tests for ``enumeration.seed_emission.ml_anchors_mrv``.

The helper is the MRV/CSP ML+spin anchor selector for the seed emitter.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from aind_low_point.optimization.enumeration.seed_emission import (
    emit_seed,
    ml_anchors_mrv,
)

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
    combo, gap = ml_anchors_mrv(anchors, target_ap=0.0, min_ml_sep_deg=_SEP)
    assert gap >= _SEP - 1e-9 and _separated(combo)
    assert combo[0][0] == 0.0  # p0 got its nearest-AP anchor
    assert combo[1] == (20.0, 40.0, 0.5)


def test_ml_mrv_resolves_conflict_via_most_constrained_first():
    # p1 has a single anchor (ml=10); p0's nearest (ml=0) conflicts, so p0
    # must take its far anchor (ml=100). MRV places p1 first and p0 adapts.
    anchors = [
        (_arr(0, 100), _arr(1, 2), _arr(0, 5)),
        (_arr(10.0), _arr(9.0), _arr(0.5)),
    ]
    combo, gap = ml_anchors_mrv(anchors, target_ap=0.0, min_ml_sep_deg=_SEP)
    assert gap >= _SEP - 1e-9 and _separated(combo)
    assert combo[1][0] == 10.0
    assert combo[0][0] == 100.0


def test_ml_mrv_best_effort_when_unseparable():
    # 0 and 5 can't be 16 apart -> best-effort (NOT None), min_gap = 5 < 16.
    anchors = [(_arr(0.0), _arr(1.0), _arr(0.0)), (_arr(5.0), _arr(1.0), _arr(0.0))]
    combo, gap = ml_anchors_mrv(anchors, target_ap=0.0, min_ml_sep_deg=_SEP)
    assert combo == [(0.0, 1.0, 0.0), (5.0, 1.0, 0.0)]
    assert abs(gap - 5.0) < 1e-9 and gap < _SEP


def test_ml_mrv_none_only_when_a_probe_has_no_anchors():
    anchors = [(_arr(), _arr(), _arr()), (_arr(5.0), _arr(1.0), _arr(0.0))]
    assert ml_anchors_mrv(anchors, target_ap=0.0, min_ml_sep_deg=_SEP) is None


def test_ml_mrv_emits_spin_from_chosen_anchor():
    anchors = [(_arr(0, 30), _arr(111, 222), _arr(0, 1))]
    combo, _ = ml_anchors_mrv(anchors, target_ap=0.0, min_ml_sep_deg=_SEP)
    assert combo[0] == (0.0, 111.0, 0.0)  # spin tracks the ml=0 anchor


def test_ml_mrv_empty():
    assert ml_anchors_mrv([], target_ap=0.0, min_ml_sep_deg=_SEP) == ([], float("inf"))


def test_ml_mrv_finds_all_feasible_random():
    rng = np.random.default_rng(3)
    for _ in range(400):
        n = int(rng.integers(2, 7))
        tgt = np.cumsum(_SEP + rng.uniform(0, 20, n))
        tgt -= tgt.mean()  # a guaranteed-separable set
        anchors = []
        for k in range(n):
            mls = np.concatenate([[tgt[k]], rng.uniform(-90, 90, rng.integers(0, 6))])
            spins = rng.uniform(-180, 180, len(mls))
            aps = rng.uniform(-50, 50, len(mls))
            anchors.append((mls, spins, aps))
        combo, gap = ml_anchors_mrv(anchors, float(rng.uniform(-40, 40)), _SEP)
        assert gap >= _SEP - 1e-9, "feasible instance came back sub-16°"
        assert _separated(combo)


# ---------------------------------------------------------------------------
# emit_seed — joint AP + ML/spin seed wrapper
# ---------------------------------------------------------------------------


def _atlas(ap, ml, spin):
    return SimpleNamespace(ap_sorted=ap, ml_sorted=ml, spin_sorted=spin)


def test_emit_seed_separated_aps_and_mls_with_spin():
    aa = _atlas(
        {
            (0, 10): _arr(-40, -38, -36),
            (1, 11): _arr(-39, -37, -35),
            (2, 12): _arr(35, 37, 39),
        },
        {(0, 10): _arr(0, 5, 10), (1, 11): _arr(30, 35, 40), (2, 12): _arr(0, 5, 10)},
        {
            (0, 10): _arr(100, 101, 102),
            (1, 11): _arr(200, 201, 202),
            (2, 12): _arr(50, 51, 52),
        },
    )
    arcs = [
        {
            "members": [(0, 10, "A"), (1, 11, "B")],
            "ap_lo": -40,
            "ap_hi": -35,
            "ap_desired": -37.5,
        },
        {"members": [(2, 12, "C")], "ap_lo": 35, "ap_hi": 39, "ap_desired": 37.0},
    ]
    arc_aps, ml, spin, gap = emit_seed(arcs, aa)
    assert gap >= _SEP - 1e-9  # not a soft seed
    assert abs(arc_aps[0] - arc_aps[1]) >= _SEP - 1e-9  # arcs separated
    assert -40 <= arc_aps[0] <= -35 and 35 <= arc_aps[1] <= 39  # within windows
    assert abs(ml["A"] - ml["B"]) >= _SEP - 1e-9  # within-arc ML sep
    assert set(spin) == {"A", "B", "C"}  # spin for every probe
    assert spin["A"] == 101.0  # spin tracks the ml=5 anchor


def test_emit_seed_best_effort_when_atlas_cannot_separate():
    # Atlas has A,B anchors only 3° apart -> emit_seed still emits (does NOT
    # drop the candidate), flagged via min_ml_gap < 16; optimizer enforces 16.
    aa = _atlas(
        {(0, 10): _arr(-37), (1, 11): _arr(-37)},
        {(0, 10): _arr(5), (1, 11): _arr(8)},
        {(0, 10): _arr(1), (1, 11): _arr(2)},
    )
    arcs = [
        {
            "members": [(0, 10, "A"), (1, 11, "B")],
            "ap_lo": -40,
            "ap_hi": -35,
            "ap_desired": -37.0,
        }
    ]
    arc_aps, ml, spin, gap = emit_seed(arcs, aa)
    assert abs(gap - 3.0) < 1e-9 and gap < _SEP  # best-effort, flagged
    assert ml == {"A": 5.0, "B": 8.0}


def test_emit_seed_empty():
    assert emit_seed([], _atlas({}, {}, {})) == ([], {}, {}, float("inf"))
