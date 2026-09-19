"""Padded constraint rows are identified by geometry, never by slack values."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from aind_rutter.optimization.geometry.holes import Hole
from aind_rutter.optimization.geometry.primitives import HoleSection
from aind_rutter.optimization.geometry.probes import ProbeStaticInfo
from aind_rutter.optimization.objectives.constrained import (
    PADDED_SLACK,
    _padding_mask,
    make_phase2,
)
from aind_rutter.optimization.objectives.statics import _build_probe_static


def _problem(**kwargs):
    axis = np.array([0.0, 0.0, 1.0])
    section = HoleSection(axis=axis, center=np.zeros(3), a=0.6, b=0.3, theta=0.0)
    hole = Hole(id=1, axis=axis, ref_point=np.zeros(3), sections=[section])
    probes = [
        ProbeStaticInfo(
            name=name,
            target_LPS=np.array([dx, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=np.array([[0.0, 0.0, 0.0]]),
        )
        for name, dx in (("p", 0.0), ("q", 1.0))
    ]
    statics = _build_probe_static(
        probes,
        [hole],
        SimpleNamespace(probe_to_hole={"p": 1, "q": 1}),
        SimpleNamespace(probe_to_arc_idx={"p": 0, "q": 0}, arc_centroids_deg=(0.0,)),
    )
    return make_phase2(statics, 1, **kwargs)


POSE = np.array(
    [0.0] + [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [20.0, 1.0, 0.0, 0.0, 0.0, 0.0]
)


def test_padded_rows_are_dropped_without_changing_live_values() -> None:
    full = _problem()
    live = _problem(drop_padded_rows=True)

    all_values = full["constraints"][0]["fun"](POSE)
    mask = np.asarray(live["live_rows"])
    assert mask.size == all_values.size, "the mask must index the whole slack vector"
    assert mask.sum() < mask.size, "this geometry should have padded rows"

    kept = live["constraints"][0]["fun"](POSE)
    assert kept.size == int(mask.sum())
    np.testing.assert_array_equal(kept, all_values[mask])
    assert (all_values[~mask] >= PADDED_SLACK - 1.0).all()

    jac = live["constraints"][0]["jac"](POSE)
    assert jac.shape == (int(mask.sum()), POSE.size)
    np.testing.assert_array_equal(jac, full["constraints"][0]["jac"](POSE)[mask])
    # A dropped row is a constant the solver would only regularize past.
    assert not np.abs(full["constraints"][0]["jac"](POSE)[~mask]).any()


def test_masking_is_off_by_default_and_for_exact_hessians() -> None:
    default = _problem()["constraints"][0]["fun"](POSE)
    exact_hessian = _problem(drop_padded_rows=True, hessian="dense")
    assert exact_hessian["constraints"][0]["fun"](POSE).size == default.size


def _packed(section, shank, same_arc) -> dict:
    return {
        "section_mask": np.asarray(section, dtype=float),
        "shank_mask": np.asarray(shank, dtype=float),
        "same_arc_mask": np.asarray(same_arc, dtype=float),
    }


LABELS = {
    "probe_pairs": [(0, 1)],
    "pair_categories": ["body_body", "corners", "obb", "shank_shank"],
    "fixtures": ["well", "cone"],
    "fixture_probes": [0, 1],
    "fixture_categories": ["body", "obb"],
}


def test_clearance_rows_stay_live_whatever_they_evaluate_to() -> None:
    """A lookup that falls off a fixture's grid returns the padding sentinel, so
    reading the mask off slack values drops rows that carry gradient once a pose
    brings the probe close."""
    mask = _padding_mask(
        _packed([[1, 0]], [[1, 1, 0]], [[0]]), LABELS, n_arcs=1, has_brain=False
    )
    thread, clearance = mask[:6], mask[6:]
    np.testing.assert_array_equal(thread, [True, True, False, False, False, False])
    assert clearance.size == 1 * 4 + 2 * 2 * 2
    assert clearance.all(), "clearance rows are never inferred from their values"


def test_brain_and_separation_rows_follow_their_masks() -> None:
    mask = _padding_mask(
        _packed([[1, 1], [1, 0]], [[1, 1, 0], [1, 0, 0]], [[0, 1], [0, 0]]),
        {**LABELS, "probe_pairs": [], "fixtures": []},
        n_arcs=3,
        has_brain=True,
    )
    thread, rest = mask[:12], mask[12:]
    # probe p: 2 sections x 2 shanks; probe q: 1 x 1.
    assert thread.sum() == 2 * 2 + 1 * 1
    brain, arc_sep, ml_sep = rest[:6], rest[6:9], rest[9:]
    np.testing.assert_array_equal(brain, [True, True, False, True, False, False])
    assert arc_sep.all() and arc_sep.size == 3
    np.testing.assert_array_equal(ml_sep, [True])
