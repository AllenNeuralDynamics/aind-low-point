"""Tests for ``aind_low_point.optimization.objectives.scalar``."""

from __future__ import annotations

import numpy as np
import pytest

from aind_low_point.optimization.geometry import Capsule, HoleSection
from aind_low_point.optimization.geometry.holes import Hole
from aind_low_point.optimization.geometry.recording import RecordingGeometry
from aind_low_point.optimization.objectives.density import gaussian_density
from aind_low_point.optimization.objectives.scalar import (
    ObjectiveWeights,
    OptimizerContext,
    ProbeContext,
    VariableLayout,
    evaluate_objective,
    evaluate_probe,
    headstage_capsule,
    kinematic_separations,
    make_objective,
    pairwise_headstage_clearances,
)

# -- VariableLayout --------------------------------------------------------


def test_variable_layout_n_vars():
    layout = VariableLayout(arc_ids=("a", "b"), probe_names=("p1", "p2", "p3"))
    # 2 arcs + 3 probes × 5 vars = 17
    assert layout.n_vars == 17


def test_variable_layout_arc_aps_and_probe_vars():
    layout = VariableLayout(arc_ids=("arc0", "arc1"), probe_names=("p", "q"))
    x = np.arange(layout.n_vars, dtype=np.float64)
    arc_aps = layout.arc_aps(x)
    assert np.allclose(arc_aps, [0.0, 1.0])
    p_vars = layout.probe_vars(x, 0)
    assert np.allclose(p_vars, [2.0, 3.0, 4.0, 5.0, 6.0])
    q_vars = layout.probe_vars(x, 1)
    assert np.allclose(q_vars, [7.0, 8.0, 9.0, 10.0, 11.0])


def test_variable_layout_arc_ap_lookup():
    layout = VariableLayout(arc_ids=("a", "b", "c"), probe_names=("p",))
    x = np.array([10.0, 20.0, 30.0, 0, 0, 0, 0, 0])
    assert layout.arc_ap(x, "b") == 20.0


# -- evaluation primitives -------------------------------------------------


def _make_axis_aligned_hole() -> Hole:
    """A bore at origin along +z with three sections.

    Slot major axis is aligned with world ``-x`` (theta = π/2). With
    ``cap_basis([0,0,1])`` returning ``(e1=+y, e2=-x)``, theta = π/2
    rotates the major direction from ``e1`` to ``e2`` — i.e., to
    world ``-x``. This is what we need for tests that place 4 shanks
    along local ``+x`` (so post-rotation the row spans world ``±x``,
    i.e., along the slot major).
    """
    axis = np.array([0.0, 0.0, 1.0])
    theta = np.pi / 2
    sections = [
        HoleSection(
            axis=axis, center=np.array([0, 0, 0.5]), a=0.65, b=0.42, theta=theta
        ),
        HoleSection(
            axis=axis, center=np.array([0, 0, 0.0]), a=0.60, b=0.35, theta=theta
        ),
        HoleSection(
            axis=axis, center=np.array([0, 0, -0.5]), a=0.60, b=0.35, theta=theta
        ),
    ]
    return Hole(id=0, axis=axis, ref_point=np.zeros(3), sections=sections)


def _make_single_probe_ctx(
    target=(0, 0, -2.0), arc_id="arc0", kind="2.4"
) -> ProbeContext:
    """4-shank context with target below the hole entry."""
    tips_local = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.75, 0.0, 0.0],
        ]
    )
    geom = RecordingGeometry(active_ranges_mm=tuple([(0.200, 0.905)] * 4))
    return ProbeContext(
        name="p1",
        target_LPS=np.asarray(target, dtype=float),
        kind=kind,
        arc_id=arc_id,
        shank_tips_local=tips_local,
        assigned_hole=_make_axis_aligned_hole(),
        density_fn=gaussian_density(target, sigma_mm=0.4),
        recording_geom=geom,
    )


def test_evaluate_probe_axis_aligned_pose():
    """At zero rotations and zero offsets/depth, the kinematic chain
    auto-centers the recording array on the target (pivot redesign).
    The probe should thread the hole AND have positive coverage."""
    probe = _make_single_probe_ctx(target=(0, 0, -2.0))
    layout = VariableLayout(arc_ids=("arc0",), probe_names=("p1",))
    ctx = OptimizerContext(layout=layout, probes=(probe,))
    ev = evaluate_probe(
        probe,
        ap_deg=0.0,
        ml_deg=0.0,
        spin_deg=0.0,
        off_R_mm=0.0,
        off_A_mm=0.0,
        past_target_mm=0.0,
        ctx=ctx,
    )
    assert len(ev.shanks) == 4
    # All threading g <= 0 (probe inside hole)
    assert ev.threading_gs.max() <= 0.0
    # Coverage > 0
    assert ev.coverage > 0.0


def test_headstage_capsule_above_pose_tip():
    R = np.eye(3)
    pose_tip = np.zeros(3)
    layout = VariableLayout(arc_ids=("a",), probe_names=("p",))
    ctx = OptimizerContext(layout=layout, probes=())
    cap = headstage_capsule(R, pose_tip, ctx)
    # Default: 10 mm above origin along +z, length 5 mm, radius 2 mm
    assert np.allclose(cap.p0, [0, 0, 10.0])
    assert np.allclose(cap.p1, [0, 0, 15.0])
    assert cap.radius == pytest.approx(2.0)


# -- pairwise --------------------------------------------------------------


def test_pairwise_headstage_clearance_two_capsules():
    """Two parallel headstage capsules at xy-offset 4 mm, radii 2 each
    → clearance = 4 - 4 = 0 (touching)."""
    from aind_low_point.optimization.objectives.scalar import ProbeEvaluation

    cap_a = Capsule(np.array([0, 0, 10]), np.array([0, 0, 15]), 2.0)
    cap_b = Capsule(np.array([4, 0, 10]), np.array([4, 0, 15]), 2.0)
    evals = [
        ProbeEvaluation(
            R=np.eye(3),
            pose_tip=np.zeros(3),
            shanks=[],
            headstage=cap_a,
            coverage=0.0,
            threading_gs=np.zeros(0),
        ),
        ProbeEvaluation(
            R=np.eye(3),
            pose_tip=np.zeros(3),
            shanks=[],
            headstage=cap_b,
            coverage=0.0,
            threading_gs=np.zeros(0),
        ),
    ]
    out = pairwise_headstage_clearances(evals)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(0.0, abs=1e-9)


def test_pairwise_headstage_clearance_single_probe_empty():
    from aind_low_point.optimization.objectives.scalar import ProbeEvaluation

    evals = [
        ProbeEvaluation(
            R=np.eye(3),
            pose_tip=np.zeros(3),
            shanks=[],
            headstage=Capsule(np.zeros(3), np.array([0, 0, 5]), 2.0),
            coverage=0.0,
            threading_gs=np.zeros(0),
        )
    ]
    out = pairwise_headstage_clearances(evals)
    assert out.shape == (0,)


# -- kinematic separations -------------------------------------------------


def test_kinematic_separations_arc_pairs():
    arc_aps = np.array([0.0, 20.0, 40.0])
    probe_mls = np.zeros(0)
    probe_arc_idxs = np.zeros(0, dtype=np.int64)
    ap_seps, ml_seps = kinematic_separations(arc_aps, probe_mls, probe_arc_idxs)
    # C(3, 2) = 3 pairs: (0,1)=20, (0,2)=40, (1,2)=20
    assert sorted(ap_seps.tolist()) == [20.0, 20.0, 40.0]
    assert ml_seps.shape == (0,)


def test_kinematic_separations_within_arc_only():
    arc_aps = np.array([0.0])
    probe_mls = np.array([0.0, 17.0, 5.0])
    probe_arc_idxs = np.zeros(3, dtype=np.int64)  # all on arc 0
    _, ml_seps = kinematic_separations(arc_aps, probe_mls, probe_arc_idxs)
    # Pairs: (0,1)=17, (0,2)=5, (1,2)=12
    assert sorted(ml_seps.tolist()) == [5.0, 12.0, 17.0]


def test_kinematic_separations_cross_arc_no_ml_constraint():
    arc_aps = np.array([0.0, 20.0])
    probe_mls = np.array([0.0, 5.0])
    probe_arc_idxs = np.array([0, 1])  # different arcs
    _, ml_seps = kinematic_separations(arc_aps, probe_mls, probe_arc_idxs)
    # No within-arc pairs.
    assert ml_seps.shape == (0,)


# -- end-to-end objective --------------------------------------------------


def test_objective_breakdown_at_good_pose_negative_total():
    """At zero rotations / zero offsets / zero depth, the kinematic
    chain auto-centers the recording array on the target. Positive
    coverage, zero threading penalty, zero kinematic penalty →
    negative total (we minimise)."""
    probe = _make_single_probe_ctx(target=(0, 0, -2.0))
    layout = VariableLayout(arc_ids=("arc0",), probe_names=("p1",))
    ctx = OptimizerContext(layout=layout, probes=(probe,))

    # x = [ap_arc0, ml, spin, off_R, off_A, depth]
    x = np.zeros(6)
    breakdown = evaluate_objective(x, ctx)
    assert breakdown.coverage_total > 0.0
    assert breakdown.threading_penalty == pytest.approx(0.0, abs=1e-9)
    assert breakdown.kinematic_penalty == pytest.approx(0.0)
    assert breakdown.total < 0.0


def test_objective_threading_violation_blows_up_penalty():
    """Move the probe far from the hole: threading constraints blow
    up and the penalty term dominates."""
    probe = _make_single_probe_ctx(target=(0, 0, -2.0))
    layout = VariableLayout(arc_ids=("arc0",), probe_names=("p1",))
    ctx = OptimizerContext(layout=layout, probes=(probe,))

    # Big offset → outer shanks fall outside the slot ovals.
    x = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.5525])
    breakdown = evaluate_objective(x, ctx)
    assert breakdown.threading_penalty > 0.0
    assert breakdown.total > breakdown.coverage_total  # penalty wins


def test_objective_kinematic_violation_penalty():
    """Two probes on the same arc with ML separation < 16° trigger
    the kinematic penalty term."""
    probe_a = _make_single_probe_ctx(target=(0, 0, -2.0), arc_id="arc0")
    # Make a second probe on the same arc with a different name
    probe_b = ProbeContext(
        name="p2",
        target_LPS=probe_a.target_LPS,
        kind=probe_a.kind,
        arc_id="arc0",
        shank_tips_local=probe_a.shank_tips_local,
        assigned_hole=probe_a.assigned_hole,
        density_fn=probe_a.density_fn,
        recording_geom=probe_a.recording_geom,
    )
    layout = VariableLayout(arc_ids=("arc0",), probe_names=("p1", "p2"))
    ctx = OptimizerContext(layout=layout, probes=(probe_a, probe_b))

    # ML difference of 5° (< 16° threshold)
    x = np.array(
        [
            0.0,  # ap_arc0
            0.0,
            0.0,
            0.375,
            0.0,
            0.5525,  # p1: ml=0
            5.0,
            0.0,
            0.375,
            0.0,
            0.5525,  # p2: ml=5
        ]
    )
    breakdown = evaluate_objective(x, ctx)
    assert breakdown.kinematic_penalty > 0.0


def test_objective_kinematic_penalty_zero_when_well_separated():
    probe_a = _make_single_probe_ctx(target=(0, 0, -2.0), arc_id="arc0")
    probe_b = ProbeContext(
        name="p2",
        target_LPS=probe_a.target_LPS,
        kind=probe_a.kind,
        arc_id="arc0",
        shank_tips_local=probe_a.shank_tips_local,
        assigned_hole=probe_a.assigned_hole,
        density_fn=probe_a.density_fn,
        recording_geom=probe_a.recording_geom,
    )
    layout = VariableLayout(arc_ids=("arc0",), probe_names=("p1", "p2"))
    ctx = OptimizerContext(layout=layout, probes=(probe_a, probe_b))

    # ML difference of 20° (> 16° threshold)
    x = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.375,
            0.0,
            0.5525,
            20.0,
            0.0,
            0.375,
            0.0,
            0.5525,
        ]
    )
    breakdown = evaluate_objective(x, ctx)
    assert breakdown.kinematic_penalty == pytest.approx(0.0)


def test_make_objective_returns_callable():
    probe = _make_single_probe_ctx(target=(0, 0, -2.0))
    layout = VariableLayout(arc_ids=("arc0",), probe_names=("p1",))
    ctx = OptimizerContext(layout=layout, probes=(probe,))
    J = make_objective(ctx)
    x = np.array([0.0, 0.0, 0.0, 0.375, 0.0, 0.5525])
    assert isinstance(J(x), float)
    # Same value as scalar_objective
    from aind_low_point.optimization.objectives.scalar import scalar_objective

    assert J(x) == scalar_objective(x, ctx)


def test_objective_x_shape_validation():
    probe = _make_single_probe_ctx(target=(0, 0, -2.0))
    layout = VariableLayout(arc_ids=("arc0",), probe_names=("p1",))
    ctx = OptimizerContext(layout=layout, probes=(probe,))
    x = np.zeros(7)  # wrong size (expected 6)
    with pytest.raises(ValueError, match="expected"):
        evaluate_objective(x, ctx)


# -- weights tweak ---------------------------------------------------------


def test_objective_weights_zero_disables_terms():
    """Setting all penalty/margin weights to 0 should give -coverage exactly."""
    probe = _make_single_probe_ctx(target=(0, 0, -2.0))
    layout = VariableLayout(arc_ids=("arc0",), probe_names=("p1",))
    weights = ObjectiveWeights(
        lambda_threading=0.0,
        lambda_clearance=0.0,
        lambda_kinematic=0.0,
        lambda_margin=0.0,
    )
    ctx = OptimizerContext(layout=layout, probes=(probe,), weights=weights)
    x = np.array([0.0, 0.0, 0.0, 0.375, 0.0, 0.5525])
    breakdown = evaluate_objective(x, ctx)
    assert breakdown.total == pytest.approx(-breakdown.coverage_total)
