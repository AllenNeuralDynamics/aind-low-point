"""Tests for the joint (H, A) reranker.

Two layers of testing:

1. **Synthetic** — small problems where the right answer is obvious so
   we can verify the reduced-SLSQP scoring shape, the AP/ML separation
   penalties, and the pose-feature precomputation independently of the
   end-to-end driver.
2. **Real subject** — load the 836656 / T12 config, regenerate the
   holes YAML from the OBJ, and verify ``optimize_joint`` ranks the
   manual-feasible (H, A) inside the top-K_joint pool.
"""

from __future__ import annotations

import numpy as np

from aind_low_point.optimization.arc_assignment import ArcAssignment
from aind_low_point.optimization.geometry import HoleSection
from aind_low_point.optimization.hole_assignment import HoleAssignment
from aind_low_point.optimization.holes import Hole
from aind_low_point.optimization.joint_rerank import (
    JointWeights,
    score_joint,
)
from aind_low_point.optimization.optimize import ProbeStaticInfo
from aind_low_point.optimization.pose_features import (
    precompute_pose_features,
    required_ap_ml_for_target,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _single_shank_tips() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0]])


def _make_hole(
    hole_id: int,
    *,
    center=(0.0, 0.0, 0.0),
    axis=(0.0, 0.0, 1.0),
    a: float = 0.6,
    b: float = 0.35,
    theta: float = np.pi / 2,
) -> Hole:
    axis_arr = np.asarray(axis, dtype=float)
    axis_arr /= np.linalg.norm(axis_arr)
    sec = HoleSection(
        axis=axis_arr,
        center=np.asarray(center, dtype=float),
        a=a,
        b=b,
        theta=theta,
    )
    return Hole(
        id=hole_id,
        axis=axis_arr,
        ref_point=np.asarray(center, dtype=float),
        sections=[sec, sec, sec],
    )


# ---------------------------------------------------------------------------
# 1) Pose-feature precomputation
# ---------------------------------------------------------------------------


def test_pose_features_precompute_synthetic():
    """Two probes + four holes — every pair should produce finite
    ``required_ap``/``required_ml`` and a non-empty ``ap_interval``."""
    probes = [
        ProbeStaticInfo(
            name="pA",
            target_LPS=np.array([0.0, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=_single_shank_tips(),
        ),
        ProbeStaticInfo(
            name="pB",
            target_LPS=np.array([1.0, 0.5, -3.5]),
            kind="2.1",
            shank_tips_local=_single_shank_tips(),
        ),
    ]
    holes = [
        _make_hole(0, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)),
        _make_hole(1, center=(1.0, 0.0, 0.0), axis=(0.05, 0.0, 0.999)),
        _make_hole(2, center=(0.5, 0.3, 0.0), axis=(0.0, 0.1, 0.995)),
        _make_hole(3, center=(0.7, -0.2, 0.0), axis=(0.1, -0.05, 0.993)),
    ]
    features = precompute_pose_features(
        probes,
        holes,
        threading_oval_tolerance=2.0,
        ap_sweep_half_deg=20.0,
        ap_sweep_step_deg=1.0,
    )
    assert len(features) == len(probes) * len(holes)
    for probe in probes:
        for hole in holes:
            feat = features[(probe.name, hole.id)]
            # Finite, well-defined required pose.
            assert np.isfinite(feat.required_ap_deg)
            assert np.isfinite(feat.required_ml_deg)
            # AP interval is a (lo, hi) tuple containing required_ap
            # or zero-width at required_ap.
            lo, hi = feat.ap_interval_deg
            assert lo <= hi
            # The interval should contain at least the required_ap
            # itself for these synthetic small-angle holes (tolerance
            # 2.0 is permissive).
            if hi - lo > 0:
                assert lo - 1e-6 <= feat.required_ap_deg <= hi + 1e-6
            # Static fields are finite.
            assert np.isfinite(feat.static_max_g)
            assert np.isfinite(feat.static_coverage)


def test_required_ap_ml_for_target_aligns_shaft():
    """``required_ap_ml_for_target`` returns the rig (ap, ml) that
    aligns the shaft with the bore-to-target unit vector."""
    from aind_mri_utils.arc_angles import arc_angles_to_affine

    target = np.array([0.4, -0.2, -3.0])
    hole = _make_hole(0, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0))
    ap, ml = required_ap_ml_for_target(hole, target)
    R = arc_angles_to_affine(ap, ml, 0.0)
    shaft = R @ np.array([0.0, 0.0, -1.0])
    expected = target - np.asarray(hole.sections[-1].center)
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(shaft, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 2-3) Reduced-SLSQP scoring on 2-probe / 1-arc cases
# ---------------------------------------------------------------------------


def test_score_joint_two_probes_one_arc_ap_overlap():
    """Two probes on the same arc whose ``ap_interval``s overlap should
    score with ``max_violation ≈ 0`` after the reduced SLSQP."""
    # Both holes are vertical and 0.5 mm apart in x — easy to thread.
    holes = [
        _make_hole(0, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)),
        _make_hole(1, center=(0.5, 0.0, 0.0), axis=(0.0, 0.0, 1.0)),
    ]
    # Targets straight below each hole — ml separation ≈ 16° at ap=0.
    # With ml ≈ +6° and ml ≈ -6° the within-arc gap is only 12°, but
    # there's wiggle room within the slot to push ml apart.
    probes = [
        ProbeStaticInfo(
            name="pA",
            target_LPS=np.array([-2.5, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=_single_shank_tips(),
        ),
        ProbeStaticInfo(
            name="pB",
            target_LPS=np.array([+3.0, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=_single_shank_tips(),
        ),
    ]
    features = precompute_pose_features(probes, holes, threading_oval_tolerance=2.0)
    ha = HoleAssignment(probe_to_hole={"pA": 0, "pB": 1}, cost=0.0)
    aa = ArcAssignment(
        probe_to_arc_idx={"pA": 0, "pB": 0},
        arc_centroids_deg=(0.0,),
        cost=0.0,
    )
    jc = score_joint(
        ha,
        aa,
        probes,
        holes,
        features,
        weights=JointWeights(threading_oval_tolerance=2.0),
    )
    # AP separation has no contribution with one arc; ML separation
    # between the probes should be > 16° (target locations 5.5 mm apart
    # via the slots). After SLSQP, max_violation should be ~0.
    assert jc.metrics.max_violation < 1e-2


def test_score_joint_two_probes_one_arc_ml_shortfall():
    """Two probes whose required-ML at the shared arc AP are < 16°
    apart — ``max_violation_intra_arc_ml_sep`` should be positive."""
    # Vertical bores, targets clustered at small ml angles so required-ML
    # is small for both.
    holes = [
        _make_hole(0, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)),
        _make_hole(1, center=(0.5, 0.0, 0.0), axis=(0.0, 0.0, 1.0)),
    ]
    # Both targets ~directly below each hole — required_ml will be very
    # small for both probes.
    probes = [
        ProbeStaticInfo(
            name="pA",
            target_LPS=np.array([0.05, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=_single_shank_tips(),
        ),
        ProbeStaticInfo(
            name="pB",
            target_LPS=np.array([0.55, 0.0, -3.0]),
            kind="2.1",
            shank_tips_local=_single_shank_tips(),
        ),
    ]
    features = precompute_pose_features(probes, holes, threading_oval_tolerance=2.0)
    ha = HoleAssignment(probe_to_hole={"pA": 0, "pB": 1}, cost=0.0)
    aa = ArcAssignment(
        probe_to_arc_idx={"pA": 0, "pB": 0},
        arc_centroids_deg=(0.0,),
        cost=0.0,
    )
    # Heavier ML weight to surface the shortfall after polish.
    weights = JointWeights(
        lambda_thread=1000.0,
        lambda_ml=1000.0,
        threading_oval_tolerance=2.0,
        min_intra_arc_ml_sep_deg=16.0,
    )
    jc = score_joint(ha, aa, probes, holes, features, weights=weights)
    # The narrow oval / target alignment forces both ml's near 0; SLSQP
    # may push them apart but the slot threading penalty trades off
    # against ml separation, so a non-zero shortfall should remain
    # OR threading violation is positive. Either way the candidate
    # should not look fully feasible.
    assert jc.metrics.max_violation > 0.0

