"""Ground-truth FCL validator.

This module performs pure validation:

  - Build per-probe-pair and per-probe-fixture FCL BVH queries
  - Expose ``slacks(x) -> np.ndarray`` returning signed clearance per
    pair (positive = clear, negative = sentinel for "colliding")
  - Expose ``is_feasible(x, margin=0.0) -> bool`` for binary
    accept/reject decisions
  - Expose ``pair_names`` for diagnostic labelling

Phase 2 with the dual-rep clearance (body voxel SDF + analytic OBBs
for shanks + transition zones + fixture-OBB-vs-fixture-surface, all
post-2026-05-24) is the only stage that *optimizes* against geometry;
the FCL validator is the ground-truth check that Phase 2's JAX
representation matches the raw mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import fcl
import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.geometry.kinematics import pose_from_optimizer_vars
from aind_low_point.optimization.objectives.phase1 import (
    PHASE1_PER_PROBE_VARS,
    FixtureSDFData,
)

# ---------------------------------------------------------------------------
# Module-level FCL request objects (cheap to share)
# ---------------------------------------------------------------------------

_FCL_DISTANCE_REQUEST = fcl.DistanceRequest(enable_signed_distance=True)
_FCL_COLLISION_REQUEST = fcl.CollisionRequest(num_max_contacts=1)

# Sentinel returned when FCL ``distance == 0`` and ``collide`` reports a
# contact set: we don't trust the ``penetration_depth`` for thin BVH so
# we just flag the pair as "in collision" with a fixed negative value.
# Pure-validator role — never used as a gradient source.
FCL_COLLISION_SENTINEL_MM = -1.0


# ---------------------------------------------------------------------------
# Per-pair FCL queries
# ---------------------------------------------------------------------------


def _signed_clearance_fcl(
    bvh_a: fcl.CollisionObject,
    bvh_b: fcl.CollisionObject,
    R_a: NDArray,
    t_a: NDArray,
    R_b: NDArray,
    t_b: NDArray,
    sentinel: float,
) -> float:
    """Signed clearance between two FCL BVH meshes at given world poses.

    ``> 0``: true clearance in mm. ``sentinel`` (negative): boolean
    collision detected (``fcl.collide`` non-empty); we don't trust the
    penetration depth on thin BVH meshes, so we flag the pair.
    """
    bvh_a.setTransform(
        fcl.Transform(
            np.ascontiguousarray(R_a, dtype=np.float64),
            np.ascontiguousarray(t_a, dtype=np.float64),
        )
    )
    bvh_b.setTransform(
        fcl.Transform(
            np.ascontiguousarray(R_b, dtype=np.float64),
            np.ascontiguousarray(t_b, dtype=np.float64),
        )
    )
    dr = fcl.DistanceResult()
    fcl.distance(bvh_a, bvh_b, _FCL_DISTANCE_REQUEST, dr)
    d = float(dr.min_distance)
    if d > 0:
        return d
    cr = fcl.CollisionResult()
    fcl.collide(bvh_a, bvh_b, _FCL_COLLISION_REQUEST, cr)
    return sentinel if cr.contacts else 0.0


def _signed_clearance_fcl_fixed_b(
    bvh_a: fcl.CollisionObject,
    bvh_b_world: fcl.CollisionObject,
    R_a: NDArray,
    t_a: NDArray,
    sentinel: float,
) -> float:
    """Same as :func:`_signed_clearance_fcl` but ``bvh_b_world`` is
    pre-transformed (e.g. a static fixture). Avoids a redundant
    ``setTransform`` on the static side.
    """
    bvh_a.setTransform(
        fcl.Transform(
            np.ascontiguousarray(R_a, dtype=np.float64),
            np.ascontiguousarray(t_a, dtype=np.float64),
        )
    )
    dr = fcl.DistanceResult()
    fcl.distance(bvh_a, bvh_b_world, _FCL_DISTANCE_REQUEST, dr)
    d = float(dr.min_distance)
    if d > 0:
        return d
    cr = fcl.CollisionResult()
    fcl.collide(bvh_a, bvh_b_world, _FCL_COLLISION_REQUEST, cr)
    return sentinel if cr.contacts else 0.0


# ---------------------------------------------------------------------------
# Pose helper
# ---------------------------------------------------------------------------


def _poses_from_x(
    x: NDArray,
    statics: list,
    n_arcs: int,
) -> tuple[list[NDArray], list[NDArray]]:
    """World poses ``(Rs, ts)`` per probe from the Phase 1/2 ``x``
    layout ``(arc_aps, (ml, sx, sy, off_R, off_A, depth) × P)``.
    """
    arc_aps = x[:n_arcs]
    Rs: list[NDArray] = []
    ts: list[NDArray] = []
    for i, st in enumerate(statics):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        ml = float(x[off + 0])
        sx = float(x[off + 1])
        sy = float(x[off + 2])
        off_R = float(x[off + 3])
        off_A = float(x[off + 4])
        depth = float(x[off + 5])
        spin_deg = float(np.degrees(np.arctan2(sy, sx)))
        ap = float(arc_aps[st.arc_idx])
        R, t = pose_from_optimizer_vars(
            target_LPS=st.target_LPS,
            ap_deg=ap,
            ml_deg=ml,
            spin_deg=spin_deg,
            offset_R_mm=off_R,
            offset_A_mm=off_A,
            past_target_mm=depth,
            recording_center_local=st.pivot_local,
        )
        Rs.append(np.asarray(R, dtype=np.float64))
        ts.append(np.asarray(t, dtype=np.float64))
    return Rs, ts


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FCLValidator:
    """Pure-FCL ground-truth validator for one ``(statics, fixtures)``
    set-up.

    Use it to check whether an optimised pose ``x`` (Phase 1 / Phase 2
    exit) is feasible under raw collision geometry.
    """

    statics: list
    n_arcs: int
    pair_names: tuple[str, ...]
    bvhs_by_idx: list[fcl.CollisionObject | None]
    fcl_pair_list: list[tuple[int, int]]
    fixture_bvh_list: list[tuple[int, fcl.CollisionObject]]
    fixture_names: tuple[str, ...]
    sentinel: float = FCL_COLLISION_SENTINEL_MM

    def slacks(self, x: NDArray) -> NDArray:
        """Signed clearance per pair: positive ⇒ clear in mm, negative
        ⇒ in collision (sentinel value). Order matches ``pair_names``.
        """
        Rs, ts = _poses_from_x(x, self.statics, self.n_arcs)
        out: list[float] = []
        for ia, ib in self.fcl_pair_list:
            ba = self.bvhs_by_idx[ia]
            bb = self.bvhs_by_idx[ib]
            if ba is None or bb is None:
                out.append(float("inf"))
                continue
            out.append(
                _signed_clearance_fcl(
                    ba,
                    bb,
                    Rs[ia],
                    ts[ia],
                    Rs[ib],
                    ts[ib],
                    self.sentinel,
                )
            )
        for fx_idx, bvh in self.fixture_bvh_list:
            for i, st in enumerate(self.statics):
                ba = self.bvhs_by_idx[i]
                if ba is None:
                    continue
                out.append(
                    _signed_clearance_fcl_fixed_b(
                        ba,
                        bvh,
                        Rs[i],
                        ts[i],
                        self.sentinel,
                    )
                )
        return np.asarray(out, dtype=np.float64)

    def is_feasible(self, x: NDArray, *, margin: float = 0.0) -> bool:
        """Boolean accept/reject. ``margin > 0`` requires every pair to
        be clear by at least that amount (mm)."""
        s = self.slacks(x)
        if s.size == 0:
            return True
        return bool(np.min(s) >= margin - 1e-9)

    def violating_pairs(
        self, x: NDArray, *, margin: float = 0.0
    ) -> list[tuple[str, float]]:
        """Per-pair diagnostic: returns ``(name, slack_mm)`` for every
        pair whose slack is below ``margin``."""
        s = self.slacks(x)
        out: list[tuple[str, float]] = []
        for name, slack in zip(self.pair_names, s):
            if slack < margin - 1e-9:
                out.append((name, float(slack)))
        return out


def make_fcl_validator(
    statics: list,
    n_arcs: int,
    *,
    fixtures: tuple[FixtureSDFData, ...] = (),
    fixture_bvhs: dict[str, fcl.CollisionObject] | None = None,
) -> FCLValidator:
    """Build an :class:`FCLValidator` for the given probe/fixture set.

    Parameters
    ----------
    statics
        Per-probe static info (same objects reduced / Phase 1 / Phase 2
        use). Each must have ``bvh_obj`` populated for the probe-probe
        and probe-fixture checks to query.
    n_arcs
        Number of arcs in the ``x`` layout.
    fixtures
        Optional tuple of :class:`FixtureSDFData`. The validator only
        uses ``fx.name`` to look up the matching BVH in ``fixture_bvhs``.
    fixture_bvhs
        Optional ``{name: fcl.CollisionObject}`` mapping for
        probe-vs-fixture FCL checks. Fixtures present in ``fixtures``
        but missing from this dict are silently skipped.
    """
    bvhs_by_idx: list[fcl.CollisionObject | None] = [
        getattr(st, "bvh_obj", None) for st in statics
    ]

    fcl_pair_list: list[tuple[int, int]] = []
    pair_names: list[str] = []
    for i in range(len(statics)):
        if bvhs_by_idx[i] is None:
            continue
        for j in range(i + 1, len(statics)):
            if bvhs_by_idx[j] is None:
                continue
            fcl_pair_list.append((i, j))
            pair_names.append(f"{statics[i].name}↔{statics[j].name}")

    fixture_bvh_list: list[tuple[int, fcl.CollisionObject]] = []
    fixture_names: list[str] = []
    if fixture_bvhs:
        for fx_idx, fx in enumerate(fixtures):
            bvh = fixture_bvhs.get(fx.name)
            if bvh is None:
                continue
            fixture_bvh_list.append((fx_idx, bvh))
            for i in range(len(statics)):
                if bvhs_by_idx[i] is None:
                    continue
                pair_names.append(f"{statics[i].name}↔{fx.name}")
                fixture_names.append(fx.name)

    return FCLValidator(
        statics=statics,
        n_arcs=n_arcs,
        pair_names=tuple(pair_names),
        bvhs_by_idx=bvhs_by_idx,
        fcl_pair_list=fcl_pair_list,
        fixture_bvh_list=fixture_bvh_list,
        fixture_names=tuple(fixture_names),
    )
