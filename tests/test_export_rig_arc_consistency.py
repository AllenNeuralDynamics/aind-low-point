"""Guard the rig-AP sign in ``export_plan_geometry``.

The top-level ``arc_angles_rig_deg`` block and each probe's ``angles_rig_deg``
must use the same convention: ``rig_ap = subject_ap + head_pitch`` (mouse mounted
nose-down; see dev memory rig_ap_sign_convention). A regression once flipped the
top-level block to ``subject_ap - head_pitch``, so the dashboard arc readout
disagreed with the per-probe rig AP by ``2 * head_pitch``. This is jax-free so it
runs without the optimizer extras.
"""

from __future__ import annotations

import numpy as np

from aind_low_point.assets import AssetCatalog
from aind_low_point.build_runtime import export_plan_geometry
from aind_low_point.core import AffineTransform
from aind_low_point.planning import Kinematics, PlanningState, ProbePlan

HEAD_PITCH_DEG = 14.0


def _rx(deg: float) -> np.ndarray:
    """Rotation about the LPS L axis (x); head_pitch extractor reads this back."""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _arc_probe(arc_id: str) -> ProbePlan:
    return ProbePlan(
        kind="neuropixels",
        arc_id=arc_id,
        bind_ap_to_arc=True,
        ml_local=3.0,
        spin=1.0,
        past_target_mm=2.0,
        offsets_RA=(0.0, 0.0),
        target_key=None,
        calibrated=False,
    )


def test_top_level_arc_rig_matches_per_probe_rig_ap() -> None:
    arc_angles = {"a": 6.0, "b": -17.0, "c": -47.0}
    state = PlanningState(
        kinematics=Kinematics(
            arc_angles=dict(arc_angles),
            subject_from_rig=AffineTransform(rotation=_rx(HEAD_PITCH_DEG)),
        ),
        probes={
            "P_a": _arc_probe("a"),
            "P_b": _arc_probe("b"),
            "P_c": _arc_probe("c"),
        },
    )

    payload = export_plan_geometry(state, AssetCatalog(assets={}))

    assert payload["head_pitch_about_L_deg"] == np.float64(HEAD_PITCH_DEG)

    # Top-level rig arc = subject arc + head pitch.
    for arc_id, subj in payload["arc_angles_subject_deg"].items():
        assert payload["arc_angles_rig_deg"][arc_id] == np.float64(
            subj + HEAD_PITCH_DEG
        )

    # The actual bug: each arc-bound probe's rig AP must equal the top-level rig
    # arc angle for its arc (they disagreed by 2*head_pitch before the fix).
    for probe in payload["probes"].values():
        arc_id = (probe["arc"] or {}).get("id")
        assert arc_id in payload["arc_angles_rig_deg"]
        np.testing.assert_allclose(
            probe["angles_rig_deg"]["ap"],
            payload["arc_angles_rig_deg"][arc_id],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            probe["angles_rig_deg"]["ap"],
            probe["angles_subject_deg"]["ap"] + HEAD_PITCH_DEG,
            atol=1e-9,
        )
