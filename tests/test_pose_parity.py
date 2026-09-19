"""Three implementations of where a probe's tip lands, held to one answer.

The optimizer traces a JAX pose (`sdf.kernels.pose_from_optimizer_vars`), the
FCL gate uses a numpy one (`geometry.kinematics.pose_from_optimizer_vars`), and
the app builds its own in `ProbePose.from_planning_state`. Parity covered the
rotation only — `tests/test_export_kinematics_parity.py` — so the *position*
half, which is what actually decides where a probe is driven, went unchecked.

All three compute

    tip = (target + off_LPS) + R @ [0, 0, -past_target] - R @ pivot_local

and differ in three places worth pinning: the app clamps the result to the rig's
translational limits, the numpy version makes the pivot optional, and the JAX
version hardcodes the RAS→LPS offset flip the other two compute.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aind_rutter.domain.plan import PlanningState, ProbePlan
from aind_rutter.domain.pose import ProbePose
from aind_rutter.domain.probe_kinds import recording_center_local_for_kind
from aind_rutter.domain.rig import Kinematics
from aind_rutter.optimization.clearance.poses import (
    pose_from_optimizer_vars as pose_jax,
)
from aind_rutter.optimization.geometry.kinematics import (
    pose_from_optimizer_vars as pose_numpy,
)

TARGET = np.array([1.5, -3.2, -4.0])
KIND = "2.1"

# (ap, ml, spin, off_R, off_A, past_target), all within the rig's angular
# limits — AP ±75°, ML ±42° — because the app clamps to them and the two
# optimizer builders do not. `test_the_app_differs_only_by_the_clamp` covers
# what happens outside.
CASES = [
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 2.5),
    (30.0, -10.0, 45.0, 0.0, 0.0, 0.0),
    (30.0, -10.0, 45.0, 1.25, -0.75, 3.0),
    (-20.0, 8.0, -110.0, -2.0, 1.0, -1.5),
    (16.0, -41.0, 3.0, 0.5, 0.5, 6.0),
    (61.0, -42.0, 179.0, -1.0, -1.0, 0.25),
    (-75.0, 42.0, -179.0, 3.0, -3.0, 9.0),
]
IDS = [f"ap{ap:g}_ml{ml:g}_spin{spin:g}" for ap, ml, spin, *_ in CASES]

# Outside them, so the app's clamp bites.
OUT_OF_LIMITS = [
    (61.0, -45.0, 179.0, -1.0, -1.0, 0.25),
    (16.0, -49.0, 3.0, 0.5, 0.5, 6.0),
    (-89.0, 45.0, -179.0, 3.0, -3.0, 9.0),
]


def _numpy_pose(ap, ml, spin, off_r, off_a, past, pivot):
    """The FCL gate's builder. Keyword-only, unlike the traced one."""
    return pose_numpy(
        target_LPS=TARGET,
        ap_deg=ap,
        ml_deg=ml,
        spin_deg=spin,
        offset_R_mm=off_r,
        offset_A_mm=off_a,
        past_target_mm=past,
        recording_center_local=pivot,
    )


def _jax_pose(ap, ml, spin, off_r, off_a, past, pivot):
    """The traced builder, at float64 so the comparison is about the formula."""
    return pose_jax(
        jnp.asarray(TARGET, jnp.float64),
        jnp.float64(ap),
        jnp.float64(ml),
        jnp.float64(spin),
        jnp.float64(off_r),
        jnp.float64(off_a),
        jnp.float64(past),
        jnp.asarray(pivot, jnp.float64),
    )


def _app_tip(ap, ml, spin, off_r, off_a, past) -> np.ndarray:
    """What the app draws and the rig export emits.

    No catalog, so the pivot is the kind-keyed recording centre — the same
    vector handed to the other two.
    """
    plan = ProbePlan(
        kind=KIND,
        arc_id=None,
        bind_ap_to_arc=False,
        ap_local=ap,
        ml_local=ml,
        spin=spin,
        offsets_LP=(-off_r, -off_a),
        past_target_mm=past,
        target_key="target:x",
    )
    state = PlanningState(
        kinematics=Kinematics(),
        probes={"P1": plan},
        target_index={"target:x": TARGET},
    )
    return np.asarray(ProbePose.from_planning_state(state, "P1").tip, dtype=np.float64)


@pytest.mark.parametrize(("ap", "ml", "spin", "off_r", "off_a", "past"), CASES, ids=IDS)
def test_the_numpy_and_jax_tips_agree(ap, ml, spin, off_r, off_a, past) -> None:
    pivot = recording_center_local_for_kind(KIND)
    _, tip_np = _numpy_pose(ap, ml, spin, off_r, off_a, past, pivot)
    _, tip_jx = _jax_pose(ap, ml, spin, off_r, off_a, past, pivot)
    np.testing.assert_allclose(np.asarray(tip_jx, dtype=np.float64), tip_np, atol=1e-6)


@pytest.mark.parametrize(("ap", "ml", "spin", "off_r", "off_a", "past"), CASES, ids=IDS)
def test_the_app_tip_agrees_with_the_optimizer(
    ap, ml, spin, off_r, off_a, past
) -> None:
    """The one the survey named: nothing tested this."""
    pivot = recording_center_local_for_kind(KIND)
    _, tip_np = _numpy_pose(ap, ml, spin, off_r, off_a, past, pivot)
    np.testing.assert_allclose(
        _app_tip(ap, ml, spin, off_r, off_a, past), tip_np, atol=1e-6
    )


@pytest.mark.parametrize(("ap", "ml", "spin", "off_r", "off_a", "past"), CASES, ids=IDS)
def test_all_three_rotations_agree(ap, ml, spin, off_r, off_a, past) -> None:
    pivot = recording_center_local_for_kind(KIND)
    r_np, _ = _numpy_pose(ap, ml, spin, off_r, off_a, past, pivot)
    r_jx, _ = _jax_pose(ap, ml, spin, off_r, off_a, past, pivot)
    assert np.abs(np.asarray(r_jx, dtype=np.float64) - r_np).max() < 1e-6


def test_the_offset_is_the_ras_to_lps_flip() -> None:
    """JAX hardcodes `(-off_R, -off_A, 0)`; the others call the converter."""
    pivot = np.zeros(3)
    _, with_offset = _numpy_pose(0, 0, 0, 2.0, 3.0, 0.0, pivot)
    np.testing.assert_allclose(with_offset, TARGET + np.array([-2.0, -3.0, 0.0]))


def test_past_target_drives_along_the_shaft() -> None:
    """At zero rotation the shaft is -z, so depth moves the tip that way."""
    pivot = np.zeros(3)
    _, at_zero = _numpy_pose(0, 0, 0, 0, 0, 0.0, pivot)
    _, deeper = _numpy_pose(0, 0, 0, 0, 0, 4.0, pivot)
    np.testing.assert_allclose(deeper - at_zero, [0.0, 0.0, -4.0])


def test_the_pivot_puts_the_recording_centre_on_target() -> None:
    """`past_target_mm = 0` lands the bank on the target, not the tip."""
    pivot = recording_center_local_for_kind(KIND)
    r, tip = _numpy_pose(0, 0, 0, 0, 0, 0.0, pivot)
    np.testing.assert_allclose(tip + r @ pivot, TARGET, atol=1e-9)


@pytest.mark.parametrize(
    ("ap", "ml", "spin", "off_r", "off_a", "past"),
    OUT_OF_LIMITS,
    ids=[f"ap{ap:g}_ml{ml:g}" for ap, ml, *_ in OUT_OF_LIMITS],
)
def test_the_app_differs_only_by_the_clamp(ap, ml, spin, off_r, off_a, past) -> None:
    """Outside AP ±75° or ML ±42° the app moves the probe and the optimizer does
    not — the app resolves angles through `Kinematics.clamp_angles`, the two
    optimizer builders take whatever they are handed.

    Feeding the clamped angles to the optimizer reproduces the app exactly, so
    the clamp is the whole of the difference rather than a second formula.
    """
    pivot = recording_center_local_for_kind(KIND)
    raw = _numpy_pose(ap, ml, spin, off_r, off_a, past, pivot)[1]
    app = _app_tip(ap, ml, spin, off_r, off_a, past)
    assert not np.allclose(app, raw, atol=1e-6), "case is inside the limits"

    clamped_ap, clamped_ml, clamped_spin = Kinematics().clamp_angles(ap, ml, spin)
    reclamped = _numpy_pose(
        clamped_ap, clamped_ml, clamped_spin, off_r, off_a, past, pivot
    )[1]
    np.testing.assert_allclose(app, reclamped, atol=1e-6)
