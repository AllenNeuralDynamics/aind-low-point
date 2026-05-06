"""Hole spec loader for the placement optimizer.

Reads the per-implant YAML produced by ``scripts/extract_implant_holes.py``
and returns a list of :class:`Hole` objects, each carrying its
per-section :class:`HoleSection` caps in LPS-mm.

Schema (one implant's bores):

.. code-block:: yaml

    holes:
      - id: 0
        axis_LPS:      [-0.191, -0.142, 0.971]
        ref_point_LPS: [-1.938, -1.531, -0.445]
        sections:
          - {s_mm: 0.167, center_LPS: [...], a_mm: 0.649,
             b_mm: 0.414, theta_rad: 2.574}
          - {s_mm: 0.000, center_LPS: [...], a_mm: 0.602,
             b_mm: 0.348, theta_rad: 2.697}
          - {s_mm: -0.167, center_LPS: [...], a_mm: 0.599,
             b_mm: 0.350, theta_rad: 2.688}

Sections are ordered top-to-bottom along ``axis`` (i.e. by
descending ``s_mm``). The bottom section is the straight bore for
typical chamfered implants, and its ``theta_rad`` defines the slot's
major-axis orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from aind_low_point.optimization.geometry import HoleSection, cap_basis


@dataclass(frozen=True, slots=True)
class Hole:
    """A bore through an implant, defined by an axis + per-section ovals.

    Sections are ordered from top (e.g. chamfer entry) to bottom
    (deepest into implant material) by ``s_mm`` along ``axis``. The
    bottom section is typically the straight bore; its ``theta`` is
    the canonical slot major-axis angle, used to pre-align probe spin.
    """

    id: int
    axis: NDArray[np.floating]
    ref_point: NDArray[np.floating]
    sections: list[HoleSection]

    @property
    def slot_theta_rad(self) -> float:
        """Canonical slot major-axis angle (radians) from the bottom section."""
        return float(self.sections[-1].theta)

    def slot_major_dir(self) -> NDArray[np.floating]:
        """Unit vector along the slot's major axis in LPS-mm world frame."""
        e1, e2 = cap_basis(self.axis)
        c = np.cos(self.slot_theta_rad)
        s = np.sin(self.slot_theta_rad)
        return c * e1 + s * e2


def load_holes(yaml_path: Path | str) -> list[Hole]:
    """Read the per-implant hole spec YAML and return a list of :class:`Hole`."""
    yaml_path = Path(yaml_path)
    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict) or "holes" not in data:
        raise ValueError(f"{yaml_path}: missing 'holes' key")

    holes: list[Hole] = []
    for entry in data["holes"]:
        axis = np.asarray(entry["axis_LPS"], dtype=float)
        ref = np.asarray(entry["ref_point_LPS"], dtype=float)
        sections = [
            HoleSection(
                axis=axis,
                center=np.asarray(s["center_LPS"], dtype=float),
                a=float(s["a_mm"]),
                b=float(s["b_mm"]),
                theta=float(s["theta_rad"]),
            )
            for s in entry["sections"]
        ]
        holes.append(
            Hole(
                id=int(entry["id"]),
                axis=axis,
                ref_point=ref,
                sections=sections,
            )
        )
    return holes


def find_hole_by_id(holes: list[Hole], hole_id: int) -> Hole:
    """Return the :class:`Hole` with matching ``id`` or raise ``KeyError``."""
    for h in holes:
        if h.id == hole_id:
            return h
    raise KeyError(
        f"hole id={hole_id} not found; have "
        f"{sorted(h.id for h in holes)}"
    )
