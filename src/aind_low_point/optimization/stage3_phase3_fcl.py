"""Stage 3 Phase 3: hard-constrained polish against full FCL mesh.

Phase 3 is nearly identical to Phase 2 in structure:

  - x layout: same as Phase 1 / Phase 2
    ``(arc_aps, (ml, sx, sy, off_R, off_A, depth) × P)``
  - Objective: same as Phase 2 (coverage + soft bounds + saturating
    margin bonuses on JAX SDF clearances + threading)
  - Constraints: same set (threading + clearance + fixture clearance +
    AP-sep + ML-sep)

The one change: **probe-probe and probe-fixture clearance constraints
use raw-mesh FCL queries** instead of JAX dual-rep SDF. This catches
the residual JAX-vs-FCL magnitude gap left after Phase 2 (the JAX
α-wrap envelope undersells thin-shell penetrations by ~3× — Phase 2
can polish to JAX-feasible while still in FCL penetration).

FCL has no analytic gradient, so SLSQP uses **finite differences** on
the FCL slacks. The other constraints (threading / AP-sep / ML-sep)
keep their analytic Jacobian via a separate scipy ineq dict, so SLSQP
mixes analytic + FD constraint Jacobians.

Per the python-fcl notes in ``pitfalls.md`` for BVH-vs-BVH:
  - ``fcl.distance`` > 0 ⇒ true clearance in mm (analytic gradient OK
    in principle but FD anyway for simplicity)
  - ``fcl.distance`` == 0 + ``fcl.collide`` non-empty ⇒ COLLIDING; the
    depth value from ``fcl.collide.penetration_depth`` is garbage on
    thin meshes, so we return a fixed negative sentinel (``-1.0`` mm)
    rather than a noisy depth.

The sentinel makes the FCL constraint piecewise-constant near the
feasibility boundary. SLSQP's FD Jacobian still picks up the
discontinuity as a steep gradient, which is enough to nudge x toward
the feasible side once a perturbation crosses back into ``distance
> 0``. Acceptable because Phase 3 polishes a near-feasible warm start
from Phase 2 — it's not solving from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

import fcl
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.coverage_jax import (
    CoverageData,
    probe_coverage,
)
from aind_low_point.optimization.joint_rerank_jax import (
    MAX_SECTIONS_PAD,
    MAX_SHANKS_PAD,
    _softplus_squared,
    threading_g_matrix,
)
from aind_low_point.optimization.kinematics import (
    pose_from_optimizer_vars as np_pose_from_optimizer_vars,
)
from aind_low_point.optimization.sdf_jax import (
    pairwise_signed_clearance_dual,
    pairwise_signed_clearance_probe_fixture_body,
    pose_from_optimizer_vars,
    smooth_abs,
    spin_deg_from_sxy,
)
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    FixtureSDFData,
    _pack_statics,
    _saturating_reward_mean,
)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase3Weights:
    """Weights for Stage 3 Phase 3 (FCL hard-constrained form).

    Same dataclass as ``Phase2Weights`` (kept distinct for clarity and
    to allow per-phase tuning). All defaults match Phase 2.
    """

    lambda_bounds: float = 1.0
    lambda_margin_clear: float = 1.0
    lambda_margin_thread: float = 1.0
    tau_clear_mm: float = 0.2
    tau_thread_gunits: float = 0.5

    min_clearance_mm: float = 0.0
    threading_oval_tolerance: float = 0.0
    min_arc_ap_sep_deg: float = 16.0
    min_intra_arc_ml_sep_deg: float = 16.0
    comfortable_ap_deg: float = 50.0
    comfortable_ml_deg: float = 50.0

    softmin_beta: float = 20.0
    top_k_body_body: int = 16
    top_k_body_shank: int = 8
    top_k_shank_shank: int = 8

    shaft_length_mm: float = 10.0

    # FCL sentinel for "colliding, depth unknown" — see module docstring.
    fcl_collision_sentinel_mm: float = -1.0


_FCL_DISTANCE_REQUEST = fcl.DistanceRequest(enable_signed_distance=True)


def _signed_clearance_fcl(
    bvh_a, bvh_b, R_a, t_a, R_b, t_b, sentinel: float
) -> float:
    """Signed clearance between two FCL BVH meshes at the given poses.

    Returns ``fcl.distance(...)`` when ``> 0`` (truly separated). When
    ``fcl.distance`` reports 0 (BVH-vs-BVH overlap), we use the
    boolean ``fcl.collide`` check; non-empty contact set ⇒ return
    ``sentinel`` (typically -1.0 mm). See module docstring for why we
    avoid ``fcl.collide.penetration_depth``.
    """
    bvh_a.setTransform(fcl.Transform(
        np.ascontiguousarray(R_a, dtype=np.float64),
        np.ascontiguousarray(t_a, dtype=np.float64),
    ))
    bvh_b.setTransform(fcl.Transform(
        np.ascontiguousarray(R_b, dtype=np.float64),
        np.ascontiguousarray(t_b, dtype=np.float64),
    ))
    dr = fcl.DistanceResult()
    fcl.distance(bvh_a, bvh_b, _FCL_DISTANCE_REQUEST, dr)
    d = float(dr.min_distance)
    if d > 0:
        return d
    # Overlap territory. Boolean check.
    cr = fcl.CollisionResult()
    fcl.collide(bvh_a, bvh_b, fcl.CollisionRequest(num_max_contacts=1), cr)
    return sentinel if cr.contacts else 0.0


def _signed_clearance_fcl_fixed_b(
    bvh_a, bvh_b_world, R_a, t_a, sentinel: float
) -> float:
    """FCL clearance where ``bvh_b_world`` is already at world identity
    (e.g., a fixture). Avoids re-transforming the static side.
    """
    bvh_a.setTransform(fcl.Transform(
        np.ascontiguousarray(R_a, dtype=np.float64),
        np.ascontiguousarray(t_a, dtype=np.float64),
    ))
    dr = fcl.DistanceResult()
    fcl.distance(bvh_a, bvh_b_world, _FCL_DISTANCE_REQUEST, dr)
    d = float(dr.min_distance)
    if d > 0:
        return d
    cr = fcl.CollisionResult()
    fcl.collide(bvh_a, bvh_b_world, fcl.CollisionRequest(num_max_contacts=1), cr)
    return sentinel if cr.contacts else 0.0


# ---------------------------------------------------------------------------
# JIT cache (shared with Phase 2 idiom)
# ---------------------------------------------------------------------------


_JIT_CACHE: dict[Hashable, dict] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def _weights_key(w: Phase3Weights) -> tuple:
    return tuple(
        float(getattr(w, f))
        for f in (
            "lambda_bounds", "lambda_margin_clear", "lambda_margin_thread",
            "tau_clear_mm", "tau_thread_gunits", "min_clearance_mm",
            "threading_oval_tolerance", "min_arc_ap_sep_deg",
            "min_intra_arc_ml_sep_deg", "comfortable_ap_deg",
            "comfortable_ml_deg", "softmin_beta", "shaft_length_mm",
            "fcl_collision_sentinel_mm",
        )
    ) + (
        int(w.top_k_body_body), int(w.top_k_body_shank),
        int(w.top_k_shank_shank),
    )


def _signature(statics, n_arcs, weights, fixtures):
    has_sdf = any(s.sdf_data is not None for s in statics)
    per_probe_sdf_shapes: tuple = ()
    if has_sdf:
        per_probe_sdf_shapes = tuple(
            None if s.sdf_data is None
            else tuple(int(x) for x in np.asarray(s.sdf_data["grid"]).shape)
            for s in statics
        )
    fix_shapes = tuple(
        tuple(int(d) for d in np.asarray(fx.grid).shape) for fx in fixtures
    )
    return (
        len(statics), int(n_arcs), MAX_SHANKS_PAD, MAX_SECTIONS_PAD,
        has_sdf, per_probe_sdf_shapes, fix_shapes, _weights_key(weights),
    )


def _poses_from_x(x, n_arcs, n_probes, target_LPS, pivot_local, arc_idx):
    """JAX-traceable pose computation (same as Phase 2)."""
    arc_aps = x[:n_arcs]
    Rs, ts = [], []
    for i in range(n_probes):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        ml = x[off + 0]
        sx = x[off + 1]
        sy = x[off + 2]
        off_R = x[off + 3]
        off_A = x[off + 4]
        depth = x[off + 5]
        spin_deg = spin_deg_from_sxy(sx, sy)
        ap = arc_aps[arc_idx[i]]
        R, t = pose_from_optimizer_vars(
            target_LPS=target_LPS[i],
            ap_deg=ap, ml_deg=ml, spin_deg=spin_deg,
            offset_R_mm=off_R, offset_A_mm=off_A,
            past_target_mm=depth,
            recording_center_local=pivot_local[i],
        )
        Rs.append(R)
        ts.append(t)
    return Rs, ts


def _np_poses_from_x(x, statics, n_arcs):
    """Numpy pose computation for FCL queries (no JAX trace needed)."""
    arc_aps = x[:n_arcs]
    Rs, ts = [], []
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
        R, t = np_pose_from_optimizer_vars(
            target_LPS=st.target_LPS, ap_deg=ap, ml_deg=ml, spin_deg=spin_deg,
            offset_R_mm=off_R, offset_A_mm=off_A, past_target_mm=depth,
            recording_center_local=st.pivot_local,
        )
        Rs.append(R)
        ts.append(t)
    return Rs, ts


_LARGE_SLACK = 1e3


def _build_jit(
    signature: tuple,
    weights: Phase3Weights,
    coverage_data: tuple[CoverageData, ...] | None,
    fixtures: tuple[FixtureSDFData, ...],
    coverage_n_samples: int = 41,
) -> dict:
    """Build JIT'd objective + analytic slacks (threading/AP/ML).

    FCL slacks are NOT JIT'd — they're built in ``make_phase3`` since
    they query python-fcl outside JAX.
    """
    n_probes, n_arcs, _ms, _msec, has_sdf, sdf_shapes, _fix_shapes, _w_key = signature

    sdf_pair_list: list[tuple[int, int]] = []
    if has_sdf:
        for i in range(n_probes):
            if sdf_shapes[i] is None:
                continue
            for j in range(i + 1, n_probes):
                if sdf_shapes[j] is None:
                    continue
                sdf_pair_list.append((i, j))

    arc_pairs = np.asarray(
        [(a, b) for a in range(n_arcs) for b in range(a + 1, n_arcs)],
        dtype=np.int32,
    ).reshape(-1, 2)
    arc_pairs_j = jnp.asarray(arc_pairs)

    lb = float(weights.lambda_bounds)
    lmc = float(weights.lambda_margin_clear)
    lmt = float(weights.lambda_margin_thread)
    tau_c = float(weights.tau_clear_mm)
    tau_t = float(weights.tau_thread_gunits)
    thread_tol = float(weights.threading_oval_tolerance)
    min_arc_ap = float(weights.min_arc_ap_sep_deg)
    min_intra_ml = float(weights.min_intra_arc_ml_sep_deg)
    cap = float(weights.comfortable_ap_deg)
    cml = float(weights.comfortable_ml_deg)
    beta = float(weights.softmin_beta)
    tk_bb = int(weights.top_k_body_body)
    tk_bs = int(weights.top_k_body_shank)
    tk_ss = int(weights.top_k_shank_shank)
    shaft_len = float(weights.shaft_length_mm)

    def _objective(
        x,
        target_LPS, pivot_local, arc_idx,
        tips_local, shank_mask,
        s_axes, s_centers, s_e1, s_e2,
        s_cos, s_sin, s_a, s_b, section_mask,
        same_arc_mask,
        sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces,
        shank_obb_centers, shank_obb_halves,
    ):
        Rs, ts = _poses_from_x(
            x, n_arcs, n_probes, target_LPS, pivot_local, arc_idx,
        )
        arc_aps = x[:n_arcs]
        ml_vals = jnp.stack(
            [x[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)]
        )

        # Coverage
        coverage_total = jnp.float32(0.0)
        if coverage_data is not None:
            for i in range(n_probes):
                coverage_total = coverage_total + probe_coverage(
                    Rs[i], ts[i], tips_local[i], shank_mask[i],
                    coverage_data[i], n_samples=coverage_n_samples,
                )

        # Soft bounds (smooth_abs)
        j_bounds = _softplus_squared(smooth_abs(arc_aps) - cap)
        j_bounds = j_bounds + _softplus_squared(smooth_abs(ml_vals) - cml)

        # Saturating margin bonuses: same JAX SDF path as Phase 2 (the
        # bonus reads the underlying soft geometry; FCL replaces only the
        # HARD CONSTRAINT signal where binary feasibility matters).
        thread_slacks_flat, thread_masks_flat = [], []
        for i in range(n_probes):
            g = threading_g_matrix(
                Rs[i], ts[i], tips_local[i],
                s_axes[i], s_centers[i], s_e1[i], s_e2[i],
                s_cos[i], s_sin[i], s_a[i], s_b[i],
                shaft_length_mm=shaft_len,
            )
            valid = section_mask[i][:, None] * shank_mask[i][None, :]
            slack = thread_tol - g
            thread_slacks_flat.append(slack.reshape(-1))
            thread_masks_flat.append(valid.reshape(-1))

        pair_hard_clearances: list[jnp.ndarray] = []
        for ia, ib in sdf_pair_list:
            (hbb, _), (hbs, _), (hss, _) = pairwise_signed_clearance_dual(
                Rs[ia], ts[ia], Rs[ib], ts[ib],
                sdf_grids[ia], sdf_origins[ia], sdf_spacings[ia],
                sdf_grids[ib], sdf_origins[ib], sdf_spacings[ib],
                sdf_surfaces[ia], sdf_surfaces[ib],
                shank_obb_centers[ia], shank_obb_halves[ia],
                shank_obb_centers[ib], shank_obb_halves[ib],
                beta=beta, top_k_body_body=tk_bb,
                top_k_body_shank=tk_bs, top_k_shank_shank=tk_ss,
            )
            pair_hard_clearances.append(jnp.minimum(jnp.minimum(hbb, hbs), hss))

        fixture_hard_clearances: list[jnp.ndarray] = []
        for fx in fixtures:
            for i in range(n_probes):
                if has_sdf and sdf_shapes[i] is None:
                    continue
                h, _ = pairwise_signed_clearance_probe_fixture_body(
                    Rs[i], ts[i],
                    sdf_grids[i], sdf_origins[i], sdf_spacings[i],
                    fx.grid, fx.origin, fx.spacing,
                    sdf_surfaces[i], fx.surface,
                    beta=beta, top_k=tk_bb,
                )
                fixture_hard_clearances.append(h)

        all_clears = pair_hard_clearances + fixture_hard_clearances
        reward_clear = (
            _saturating_reward_mean(jnp.stack(all_clears), tau_c)
            if all_clears else jnp.float32(0.0)
        )
        slacks = (
            jnp.concatenate(thread_slacks_flat) if thread_slacks_flat
            else jnp.zeros(1)
        )
        masks = (
            jnp.concatenate(thread_masks_flat) if thread_masks_flat
            else jnp.zeros(1)
        )
        reward_thread = _saturating_reward_mean(slacks, tau_t, valid=masks)

        return (
            -coverage_total
            + lb * j_bounds
            - lmc * reward_clear
            - lmt * reward_thread
        )

    def _analytic_slacks(
        x,
        target_LPS, pivot_local, arc_idx,
        tips_local, shank_mask,
        s_axes, s_centers, s_e1, s_e2,
        s_cos, s_sin, s_a, s_b, section_mask,
        same_arc_mask,
        sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces,
        shank_obb_centers, shank_obb_halves,
    ):
        """Threading + AP-sep + intra-arc ML-sep slacks (analytic Jac)."""
        Rs, ts = _poses_from_x(
            x, n_arcs, n_probes, target_LPS, pivot_local, arc_idx,
        )
        arc_aps = x[:n_arcs]

        thread_slacks: list[jnp.ndarray] = []
        for i in range(n_probes):
            g = threading_g_matrix(
                Rs[i], ts[i], tips_local[i],
                s_axes[i], s_centers[i], s_e1[i], s_e2[i],
                s_cos[i], s_sin[i], s_a[i], s_b[i],
                shaft_length_mm=shaft_len,
            )
            valid = section_mask[i][:, None] * shank_mask[i][None, :]
            slack = thread_tol - g
            slack_masked = jnp.where(valid > 0, slack, _LARGE_SLACK)
            thread_slacks.append(slack_masked.reshape(-1))
        thread_vec = (
            jnp.concatenate(thread_slacks) if thread_slacks else jnp.zeros(0)
        )

        if arc_pairs.shape[0] > 0:
            ap_diffs = smooth_abs(
                arc_aps[arc_pairs_j[:, 0]] - arc_aps[arc_pairs_j[:, 1]]
            )
            ap_sep_vec = ap_diffs - min_arc_ap
        else:
            ap_sep_vec = jnp.zeros(0)

        ml_vals = jnp.stack(
            [x[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)]
        )
        ml_diff = smooth_abs(ml_vals[:, None] - ml_vals[None, :])
        ml_slack = ml_diff - min_intra_ml
        iu, ju = np.triu_indices(n_probes, k=1)
        if iu.size > 0:
            ml_slack_flat = ml_slack[iu, ju]
            mask_flat = same_arc_mask[iu, ju]
            ml_sep_vec = jnp.where(mask_flat > 0, ml_slack_flat, _LARGE_SLACK)
        else:
            ml_sep_vec = jnp.zeros(0)

        return jnp.concatenate([thread_vec, ap_sep_vec, ml_sep_vec])

    return dict(
        obj=jax.jit(_objective),
        obj_grad=jax.jit(jax.grad(_objective)),
        analytic_slacks=jax.jit(_analytic_slacks),
        analytic_slacks_jac=jax.jit(jax.jacfwd(_analytic_slacks)),
        sdf_pair_list=sdf_pair_list,
    )


def cache_stats() -> dict:
    return {**_CACHE_STATS, "entries": len(_JIT_CACHE)}


def make_phase3(
    statics,
    n_arcs: int,
    *,
    coverage_data: tuple[CoverageData, ...] | None = None,
    fixtures: tuple[FixtureSDFData, ...] = (),
    fixture_bvhs: dict[str, fcl.BVHModel] | None = None,
    weights: Phase3Weights = Phase3Weights(),
    coverage_n_samples: int = 41,
    min_clearance_mm: float | None = None,
) -> dict:
    """Build Phase 3 scipy callables.

    Parameters
    ----------
    statics
        Probe statics (same as Phase 2). Each must have ``bvh_obj``
        populated for the FCL constraints to query.
    n_arcs
        Number of arcs.
    coverage_data, fixtures, weights, coverage_n_samples
        Same as Phase 2.
    fixture_bvhs
        Optional ``{name: fcl.BVHModel}`` mapping for probe-vs-fixture
        FCL checks. Names should match ``fixtures[i].name``. If a
        fixture is in ``fixtures`` but not in ``fixture_bvhs``, its
        clearance constraint is silently skipped.
    min_clearance_mm
        Override for ``weights.min_clearance_mm`` — Phase 3 often wants
        a positive buffer (e.g., 0.05 mm) so the optimizer doesn't sit
        right on the FCL boundary where the sentinel jump lives.

    Returns
    -------
    dict with keys ``fun``, ``jac``, ``constraints`` — same shape as
    Phase 2 but ``constraints`` is a list of TWO ineq dicts: one with
    analytic Jacobian (threading + AP + ML) and one with no jac (FCL
    clearances; scipy uses FD).
    """
    sig = _signature(statics, n_arcs, weights, fixtures)
    if sig not in _JIT_CACHE:
        _JIT_CACHE[sig] = _build_jit(
            sig, weights, coverage_data, fixtures, coverage_n_samples,
        )
        _CACHE_STATS["misses"] += 1
    else:
        _CACHE_STATS["hits"] += 1
    jit = _JIT_CACHE[sig]
    packed = _pack_statics(statics, n_arcs)

    bvhs_by_idx = [st.bvh_obj for st in statics]
    has_bvh = [bvh is not None for bvh in bvhs_by_idx]
    fcl_pair_list = [
        (ia, ib) for (ia, ib) in jit["sdf_pair_list"]
        if has_bvh[ia] and has_bvh[ib]
    ]
    fixture_bvh_list: list[tuple[int, fcl.BVHModel]] = []
    if fixtures and fixture_bvhs:
        for fx_idx, fx in enumerate(fixtures):
            bvh = fixture_bvhs.get(fx.name)
            if bvh is None:
                continue
            # Set fixture transform to identity once.
            bvh.setTransform(fcl.Transform(
                np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64),
            ))
            fixture_bvh_list.append((fx_idx, bvh))

    sentinel = float(weights.fcl_collision_sentinel_mm)
    min_clear = (
        float(min_clearance_mm) if min_clearance_mm is not None
        else float(weights.min_clearance_mm)
    )

    def fun(x: NDArray) -> float:
        return float(jit["obj"](jnp.asarray(x, dtype=jnp.float32), **packed))

    def jac(x: NDArray) -> NDArray:
        g = jit["obj_grad"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(g, dtype=np.float64)

    def analytic_slacks_fn(x: NDArray) -> NDArray:
        s = jit["analytic_slacks"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(s, dtype=np.float64)

    def analytic_slacks_jac_fn(x: NDArray) -> NDArray:
        J = jit["analytic_slacks_jac"](
            jnp.asarray(x, dtype=jnp.float32), **packed
        )
        return np.asarray(J, dtype=np.float64)

    def fcl_slacks_fn(x: NDArray) -> NDArray:
        """FCL probe-probe + probe-fixture clearance slacks.

        For each pair, returns ``d − min_clearance_mm``. ``d`` is the
        FCL signed clearance with sentinel for "colliding".
        """
        Rs, ts = _np_poses_from_x(x, statics, n_arcs)
        out: list[float] = []
        # Probe-probe
        for ia, ib in fcl_pair_list:
            d = _signed_clearance_fcl(
                bvhs_by_idx[ia], bvhs_by_idx[ib],
                Rs[ia], ts[ia], Rs[ib], ts[ib], sentinel,
            )
            out.append(d - min_clear)
        # Probe-fixture
        for _fx_idx, bvh in fixture_bvh_list:
            for i, st in enumerate(statics):
                if bvhs_by_idx[i] is None:
                    continue
                d = _signed_clearance_fcl_fixed_b(
                    bvhs_by_idx[i], bvh, Rs[i], ts[i], sentinel,
                )
                out.append(d - min_clear)
        return np.asarray(out, dtype=np.float64) if out else np.zeros(0)

    constraints = [
        {
            "type": "ineq",
            "fun": analytic_slacks_fn,
            "jac": analytic_slacks_jac_fn,
        },
    ]
    if fcl_pair_list or fixture_bvh_list:
        constraints.append({
            "type": "ineq",
            "fun": fcl_slacks_fn,
            # no jac -> scipy uses FD
        })

    return dict(
        fun=fun,
        jac=jac,
        constraints=constraints,
        n_fcl_pair=len(fcl_pair_list),
        n_fcl_fixture_pair=len(fixture_bvh_list) * sum(
            1 for b in bvhs_by_idx if b is not None
        ),
    )
