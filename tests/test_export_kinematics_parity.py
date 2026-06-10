"""Guard the two-codebase kinematics drift behind the rig export.

The optimizer builds probe rotations with ``sdf_jax.arc_angles_to_rotation``; the
app / rig export (``ProbePose``, ``export_plan_geometry``) builds them with
``aind_mri_utils.arc_angles.arc_angles_to_affine``. The emitted rig poses are only
faithful to the optimized poses if those two agree. This caught a real class of bug
(the retro-target frame + rig-AP sign issues); keep it in CI so the two builders
can't silently diverge.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aind_mri_utils.arc_angles import arc_angles_to_affine

from aind_low_point.optimization.sdf_jax import arc_angles_to_rotation


@pytest.mark.parametrize(
    "ap,ml,spin",
    [
        (0.0, 0.0, 0.0),
        (30.0, -10.0, 45.0),
        (-20.0, 8.0, -110.0),
        (16.0, -49.0, 3.0),
        (-34.2, 7.9, -28.0),
        (61.0, -45.0, 179.0),  # rig-reachable subject-AP window edges (head pitch 14)
        (-89.0, 45.0, -179.0),
    ],
)
def test_arc_angle_rotation_builders_agree(ap: float, ml: float, spin: float) -> None:
    """app/export ``arc_angles_to_affine`` == optimizer ``arc_angles_to_rotation``."""
    r_app = np.asarray(arc_angles_to_affine(ap, ml, spin), dtype=float)[:3, :3]
    r_opt = np.asarray(
        arc_angles_to_rotation(jnp.float32(ap), jnp.float32(ml), jnp.float32(spin)),
        dtype=float,
    )
    assert np.abs(r_app - r_opt).max() < 1e-4
