"""Spin-only feasibility restoration — legacy no-op pass-through.

This module's ``spin_restore_jax`` is currently a no-op under Patch B's
``(sx, sy)`` unit-circle spin layout. The legacy 2D scalar-spin sweep
(coarse + fine grids over a single spin angle) is incompatible with the
new layout — replacing it would require a 4D sweep
``(sx_a, sy_a, sx_b, sy_b)`` which has not been implemented.

All production spin restoration now goes through
:func:`aind_low_point.optimization.batched_spin_restore.make_batched_spin_restore_chunked`.
This module is kept so that the existing call site in ``joint_rerank``
(which dispatches on ``use_jax_spin``) doesn't crash; it returns
``y`` unchanged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def spin_restore_jax(
    y: NDArray,
    statics,
    n_arcs: int,
    **_unused,
) -> NDArray:
    """No-op pass-through (see module docstring)."""
    return np.asarray(y, dtype=np.float64)
