"""Probe spin/shank kinematics helpers (closed-form, numpy-only).

Detecting 4-shank probes and computing the spin that aligns a probe's local +y
axis with a world-frame direction, used by the seed/restore stages.
"""

from __future__ import annotations

import numpy as np


def is_four_shank(probe_static) -> bool:
    """Detect 4-shank probes dynamically from the static info.

    Covers quadbase-alpha, quadbase-dovetail, NP 2.4, and any future
    4-shank kinds. The threading constraint (H1) limits these to
    {slot, slot + 180°} regardless of name.
    """
    return len(probe_static.shank_tips_local) >= 4


def _orbit_basis(ap_deg: float, ml_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (a, b) such that the world-frame image of probe local +y
    under ``arc_angles_to_affine(ap, ml, spin)`` is::

        u(spin) = sin(spin) * a + cos(spin) * b

    Derivation: ``R_LPS = R_x(ap) R_y(-ml) R_z(-spin)`` and
    ``R_z(-spin) [0,1,0] = (sin spin, cos spin, 0)``. So::

        u(spin) = R_x(ap) R_y(-ml) (sin spin, cos spin, 0)
                = sin spin · R_x(ap) R_y(-ml) [1,0,0]
                + cos spin · R_x(ap) R_y(-ml) [0,1,0]

    Avoids calling ``arc_angles_to_affine`` twice per heuristic eval.
    """
    ap = np.deg2rad(ap_deg)
    ml = np.deg2rad(ml_deg)
    cap, sap = np.cos(ap), np.sin(ap)
    cml, sml = np.cos(ml), np.sin(ml)
    a = np.array([cml, -sap * sml, cap * sml])
    b = np.array([0.0, cap, sap])
    return a, b


def spin_to_align_y_with(
    target_dir_world: np.ndarray,
    ap_deg: float,
    ml_deg: float,
) -> float:
    """Closed-form spin (deg) that best aligns probe local +y with
    ``target_dir_world`` under ``arc_angles_to_affine(ap, ml, spin)``.

    ``u(spin) · sm`` is maximised at ``spin = atan2(a · sm, b · sm)``
    (see :func:`_orbit_basis`). When ``sm`` doesn't lie in the orbit
    plane (off bore-aligned (ap, ml)), the residual is the projection
    error — non-zero but typically <30° for our manual plans.
    """
    a, b = _orbit_basis(ap_deg, ml_deg)
    sm = np.asarray(target_dir_world, dtype=float)
    sm = sm / max(float(np.linalg.norm(sm)), 1e-12)
    return float(np.degrees(np.arctan2(float(a @ sm), float(b @ sm))))
