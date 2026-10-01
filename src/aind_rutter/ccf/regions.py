"""Part of a CCF region: the halves of its bounding box a definition keeps.

The atlas has no label for posterior dorsomedial striatum or ventral CA1, so a
recording target smaller than its structure is the structure cut at the middle
of its own extent, once per named half. Points and bounds are physical LPS
millimetres: +x left, +y posterior, +z dorsal.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class Half(StrEnum):
    DORSAL = "dorsal"
    VENTRAL = "ventral"
    ANTERIOR = "anterior"
    POSTERIOR = "posterior"
    MEDIAL = "medial"
    LATERAL = "lateral"


_OPPOSITE = {
    Half.DORSAL: Half.VENTRAL,
    Half.ANTERIOR: Half.POSTERIOR,
    Half.MEDIAL: Half.LATERAL,
}
_OPPOSITE |= {b: a for a, b in _OPPOSITE.items()}


def parse_halves(keep: Iterable[str]) -> tuple[Half, ...]:
    """Validate a definition's ``keep`` list."""
    try:
        halves = tuple(Half(k) for k in keep)
    except ValueError:
        raise ValueError(
            f"keep must name halves from {[h.value for h in Half]}, got {list(keep)}"
        ) from None
    for half in halves:
        if _OPPOSITE[half] in halves:
            raise ValueError(f"keep names both {half} and {_OPPOSITE[half]}")
    return halves


def half_mask(
    points_LPS: NDArray[np.float64],
    keep: Iterable[str],
    *,
    hemisphere: str,
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Which points lie in every kept half of the box ``lo``–``hi``.

    The box is the whole region's, so points tested against it need not be the
    region's own voxels. Medial and lateral need one hemisphere to mean a side.
    """
    points = np.asarray(points_LPS, dtype=np.float64)
    mid = (np.asarray(lo, np.float64) + np.asarray(hi, np.float64)) / 2
    mask = np.ones(len(points), bool)
    for half in parse_halves(keep):
        if half in (Half.MEDIAL, Half.LATERAL):
            side = hemisphere.lower()[:1]
            if side not in ("l", "r"):
                raise ValueError(
                    f"keep={half.value!r} needs hemisphere 'left' or 'right', "
                    f"got {hemisphere!r}"
                )
            # +x is left, so the midline is at lower x for the left hemisphere.
            toward_low_x = (half is Half.MEDIAL) == (side == "l")
            mask &= points[:, 0] < mid[0] if toward_low_x else points[:, 0] >= mid[0]
        elif half is Half.POSTERIOR:
            mask &= points[:, 1] > mid[1]
        elif half is Half.ANTERIOR:
            mask &= points[:, 1] <= mid[1]
        elif half is Half.DORSAL:
            mask &= points[:, 2] > mid[2]
        else:
            mask &= points[:, 2] <= mid[2]
    return mask
