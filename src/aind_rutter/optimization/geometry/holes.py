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

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from aind_rutter.optimization.geometry.primitives import HoleSection, cap_basis

# Threading margin: the threading check models each shank as a thin CENTERLINE,
# so ``g <= 0`` only means the centerline is inside the bore oval — a centerline
# grazing the wall puts the finite-width shank EDGE through it (which FCL then
# reports as a collision). Inset each bore oval's semi-axes by this effective
# shank radius (mm) so ``g <= 0`` instead means the real shank clears the real
# wall, mapping threading feasibility onto FCL feasibility. Empirically
# 0.06–0.08 mm perfectly separates the FCL-clearing probes from the FCL-colliders
# on the 837229 density plans (0.07 is the robust midpoint). Applied at both
# probe-static builders (``_build_probe_static`` and ``build_batched_probe_static``)
# so every optimizer stage — spin restore, Phase-1, Phase-2 — sees the inset.
# NOT applied to the hole-assignment gate (``static_threading_max_g``), which
# stays pure centerline geometry. ``RUTTER_THREADING_MARGIN_MM=0`` reproduces the
# legacy centerline check.
DEFAULT_THREADING_MARGIN_MM: float = 0.07


def threading_margin_mm() -> float:
    """Effective shank-radius inset (mm) applied to bore ovals (env-overridable)."""
    return float(
        os.environ.get("RUTTER_THREADING_MARGIN_MM", DEFAULT_THREADING_MARGIN_MM)
    )


@dataclass(frozen=True, slots=True)
class HoleWall:
    """A planar wall cutting into a bore. Solid lies on the ``+normal`` side."""

    point: NDArray[np.floating]
    normal: NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class Hole:
    """A bore through an implant, defined by an axis + per-section ovals.

    Sections are ordered from top (e.g. chamfer entry) to bottom
    (deepest into implant material) by ``s_mm`` along ``axis``. The
    bottom section is typically the straight bore; its ``theta`` is
    the canonical slot major-axis angle, used to pre-align probe spin.
    ``walls`` are planes that cut into the bore, such as an implant edge running
    through the channel; a shank must clear every section oval and every wall.
    """

    id: int
    axis: NDArray[np.floating]
    ref_point: NDArray[np.floating]
    sections: list[HoleSection]
    walls: tuple[HoleWall, ...] = ()

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


MAX_WALLS_PAD: int = 2
# Offset for padded wall rows: with a zero normal every point sits 1 m on the open side.
NO_WALL_OFFSET_MM: float = 1e3


def pack_walls(
    walls: Sequence[HoleWall], margin_mm: float, n_pad: int = MAX_WALLS_PAD
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Kernel arrays ``(normals (n_pad, 3), offsets (n_pad,))`` for a hole's walls.

    A point ``p`` clears wall ``w`` when ``normals[w] @ p <= offsets[w]``. Offsets move
    toward the bore by ``margin_mm``, matching the oval inset. Padded rows never bind.
    """
    if len(walls) > n_pad:
        raise ValueError(f"hole has {len(walls)} walls; kernels pad to {n_pad}")
    normals = np.zeros((n_pad, 3), dtype=np.float32)
    offsets = np.full(n_pad, NO_WALL_OFFSET_MM, dtype=np.float32)
    for k, w in enumerate(walls):
        normals[k] = w.normal
        offsets[k] = float(np.dot(w.normal, w.point)) - margin_mm
    return normals, offsets


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
        walls = tuple(
            HoleWall(
                point=np.asarray(w["point_LPS"], dtype=float),
                normal=np.asarray(w["normal_LPS"], dtype=float)
                / np.linalg.norm(w["normal_LPS"]),
            )
            for w in entry.get("walls", ())
        )
        holes.append(
            Hole(
                id=int(entry["id"]),
                axis=axis,
                ref_point=ref,
                sections=sections,
                walls=walls,
            )
        )
    return holes


def find_hole_by_id(holes: list[Hole], hole_id: int) -> Hole:
    """Return the :class:`Hole` with matching ``id`` or raise ``KeyError``."""
    for h in holes:
        if h.id == hole_id:
            return h
    raise KeyError(f"hole id={hole_id} not found; have {sorted(h.id for h in holes)}")
