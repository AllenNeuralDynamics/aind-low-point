"""Stage 3 Phase 2: hard-constrained polish with coverage maximisation.

Phase 2 follows Phase 1's soft-penalty warm-up. It uses the *same* x
layout (``(arc_aps, (ml, sx, sy, off_R, off_A, depth) × P)``) but moves
all feasibility terms from the objective into SLSQP inequality
constraints:

  - Threading: ``g_thread ≤ tol`` per (probe, shank, section)
  - Clearance probe-probe (dual-rep): ``d_soft ≥ min_clear`` per
    (pair, category)
  - Clearance probe-fixture (body): same, per (probe, fixture)
  - Arc-AP separation: ``smooth_abs(ap_diff) ≥ min_sep`` per arc pair
  - Intra-arc ML separation: same per intra-arc pair

The objective shrinks to coverage + soft bounds + saturating margin
bonuses (clearance + threading) — exactly the bonuses from Phase 1.
SLSQP enforces strict feasibility via the constraints; the margin
bonuses keep the gradient meaningful inside the feasible region (where
coverage may be locally flat).

Shares all geometry/density kernels with Phase 1 — no duplicate code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

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
class Phase2Weights:
    """Weights for Stage 3 Phase 2 (hard-constrained form).

    Phase 2 drops the feasibility-penalty λ's (threading, clearance,
    kinematic) — those terms are now constraints. Keeps coverage, soft
    bounds, and the two saturating margin bonuses.
    """

    lambda_bounds: float = 1.0

    lambda_margin_clear: float = 1.0
    lambda_margin_thread: float = 1.0
    tau_clear_mm: float = 0.2
    tau_thread_gunits: float = 0.5

    # Constraint thresholds (passed into the slack functions).
    min_clearance_mm: float = 0.0
    threading_oval_tolerance: float = 0.0
    min_arc_ap_sep_deg: float = 16.0
    min_intra_arc_ml_sep_deg: float = 16.0
    comfortable_ap_deg: float = 50.0
    comfortable_ml_deg: float = 50.0

    # Soft-min knobs for dual-rep clearance.
    softmin_beta: float = 20.0
    top_k_body_body: int = 16
    top_k_body_shank: int = 8
    top_k_shank_shank: int = 8

    shaft_length_mm: float = 10.0


# ---------------------------------------------------------------------------
# Pose helper used by both objective and constraints
# ---------------------------------------------------------------------------


def _poses_from_x(x, n_arcs, n_probes, target_LPS, pivot_local, arc_idx):
    """Compute (Rs, ts) per probe from a Phase 2 x vector."""
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


# ---------------------------------------------------------------------------
# JIT cache
# ---------------------------------------------------------------------------


_JIT_CACHE: dict[Hashable, dict] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def _weights_key(w: Phase2Weights) -> tuple:
    return tuple(
        float(getattr(w, f))
        for f in (
            "lambda_bounds", "lambda_margin_clear", "lambda_margin_thread",
            "tau_clear_mm", "tau_thread_gunits", "min_clearance_mm",
            "threading_oval_tolerance", "min_arc_ap_sep_deg",
            "min_intra_arc_ml_sep_deg", "comfortable_ap_deg",
            "comfortable_ml_deg", "softmin_beta", "shaft_length_mm",
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


# ---------------------------------------------------------------------------
# Build JIT'd objective + slacks
# ---------------------------------------------------------------------------


_LARGE_SLACK = 1e3  # sentinel for masked-out (padded) constraints


def _build_jit(
    signature: tuple,
    weights: Phase2Weights,
    coverage_data: tuple[CoverageData, ...] | None,
    fixtures: tuple[FixtureSDFData, ...],
    coverage_n_samples: int = 41,
) -> dict:
    """Build JIT'd (obj, obj_grad, all_slacks, all_slacks_jac) for one signature."""
    n_probes, n_arcs, _ms, _msec, has_sdf, sdf_shapes, _fix_shapes, _w_key = signature

    # Pre-build the probe-probe SDF pair list (skipping probes without
    # SDF). Same logic as Phase 1.
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
    min_clear = float(weights.min_clearance_mm)
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

    # ---- Objective: scalar minimised by SLSQP ----
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
        arc_aps = x[:n_arcs]
        Rs, ts = _poses_from_x(
            x, n_arcs, n_probes, target_LPS, pivot_local, arc_idx,
        )

        # Coverage
        coverage_total = jnp.float32(0.0)
        if coverage_data is not None:
            for i in range(n_probes):
                coverage_total = coverage_total + probe_coverage(
                    Rs[i], ts[i], tips_local[i], shank_mask[i],
                    coverage_data[i], n_samples=coverage_n_samples,
                )

        # Soft bounds: pull-back from comfort range (smooth_abs).
        ml_vals = jnp.stack(
            [x[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)]
        )
        j_bounds = _softplus_squared(smooth_abs(arc_aps) - cap)
        j_bounds = j_bounds + _softplus_squared(smooth_abs(ml_vals) - cml)

        # Margin bonuses: saturating per-pair (clear) and per-tuple (thread).
        # Mirror Phase 1's computation but skip the soft penalty terms.
        thread_slacks_flat: list[jnp.ndarray] = []
        thread_masks_flat: list[jnp.ndarray] = []
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
                beta=beta,
                top_k_body_body=tk_bb,
                top_k_body_shank=tk_bs,
                top_k_shank_shank=tk_ss,
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
        slacks = jnp.concatenate(thread_slacks_flat) if thread_slacks_flat else jnp.zeros(1)
        masks = jnp.concatenate(thread_masks_flat) if thread_masks_flat else jnp.zeros(1)
        reward_thread = _saturating_reward_mean(slacks, tau_t, valid=masks)

        return (
            - coverage_total
            + lb * j_bounds
            - lmc * reward_clear
            - lmt * reward_thread
        )

    # ---- All slacks: scipy sees ineq[g(x) ≥ 0] over the concat'd vector ----
    def _all_slacks(
        x,
        target_LPS, pivot_local, arc_idx,
        tips_local, shank_mask,
        s_axes, s_centers, s_e1, s_e2,
        s_cos, s_sin, s_a, s_b, section_mask,
        same_arc_mask,
        sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces,
        shank_obb_centers, shank_obb_halves,
    ):
        arc_aps = x[:n_arcs]
        Rs, ts = _poses_from_x(
            x, n_arcs, n_probes, target_LPS, pivot_local, arc_idx,
        )

        # Threading slacks: tol - g, masked. Padded entries → +LARGE.
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

        # Clearance probe-probe: d_soft − min_clear per (pair, category).
        clear_pp_slacks: list[jnp.ndarray] = []
        for ia, ib in sdf_pair_list:
            (_, sbb), (_, sbs), (_, sss) = pairwise_signed_clearance_dual(
                Rs[ia], ts[ia], Rs[ib], ts[ib],
                sdf_grids[ia], sdf_origins[ia], sdf_spacings[ia],
                sdf_grids[ib], sdf_origins[ib], sdf_spacings[ib],
                sdf_surfaces[ia], sdf_surfaces[ib],
                shank_obb_centers[ia], shank_obb_halves[ia],
                shank_obb_centers[ib], shank_obb_halves[ib],
                beta=beta,
                top_k_body_body=tk_bb,
                top_k_body_shank=tk_bs,
                top_k_shank_shank=tk_ss,
            )
            for d_soft in (sbb, sbs, sss):
                clear_pp_slacks.append(d_soft - min_clear)
        clear_pp_vec = (
            jnp.stack(clear_pp_slacks) if clear_pp_slacks else jnp.zeros(0)
        )

        # Clearance probe-fixture (body): d_soft − min_clear.
        clear_pf_slacks: list[jnp.ndarray] = []
        for fx in fixtures:
            for i in range(n_probes):
                if has_sdf and sdf_shapes[i] is None:
                    continue
                _, s = pairwise_signed_clearance_probe_fixture_body(
                    Rs[i], ts[i],
                    sdf_grids[i], sdf_origins[i], sdf_spacings[i],
                    fx.grid, fx.origin, fx.spacing,
                    sdf_surfaces[i], fx.surface,
                    beta=beta, top_k=tk_bb,
                )
                clear_pf_slacks.append(s - min_clear)
        clear_pf_vec = (
            jnp.stack(clear_pf_slacks) if clear_pf_slacks else jnp.zeros(0)
        )

        # Arc-AP separation: smooth_abs(diff) − min_arc_ap_sep.
        if arc_pairs.shape[0] > 0:
            ap_diffs = smooth_abs(
                arc_aps[arc_pairs_j[:, 0]] - arc_aps[arc_pairs_j[:, 1]]
            )
            ap_sep_vec = ap_diffs - min_arc_ap
        else:
            ap_sep_vec = jnp.zeros(0)

        # Intra-arc ML separation: smooth_abs(ml_diff) − min_ml_sep,
        # but only over same-arc pairs (others get +LARGE so SLSQP ignores).
        ml_vals = jnp.stack(
            [x[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)]
        )
        ml_diff = smooth_abs(ml_vals[:, None] - ml_vals[None, :])
        ml_slack = ml_diff - min_intra_ml
        # Take upper triangle to avoid duplicates; mask off non-same-arc.
        iu, ju = np.triu_indices(n_probes, k=1)
        if iu.size > 0:
            ml_slack_flat = ml_slack[iu, ju]
            mask_flat = same_arc_mask[iu, ju]
            ml_sep_vec = jnp.where(mask_flat > 0, ml_slack_flat, _LARGE_SLACK)
        else:
            ml_sep_vec = jnp.zeros(0)

        return jnp.concatenate([
            thread_vec, clear_pp_vec, clear_pf_vec, ap_sep_vec, ml_sep_vec,
        ])

    return dict(
        obj=jax.jit(_objective),
        obj_grad=jax.jit(jax.grad(_objective)),
        slacks=jax.jit(_all_slacks),
        slacks_jac=jax.jit(jax.jacfwd(_all_slacks)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cache_stats() -> dict:
    return {**_CACHE_STATS, "entries": len(_JIT_CACHE)}


def make_phase2(
    statics,
    n_arcs: int,
    coverage_data: tuple[CoverageData, ...] | None = None,
    fixtures: tuple[FixtureSDFData, ...] = (),
    weights: Phase2Weights = Phase2Weights(),
    *,
    coverage_n_samples: int = 41,
) -> dict:
    """Build Phase 2 scipy callables.

    Returns a dict with:

      - ``fun(x) → scalar``: objective (-coverage + bounds + margin bonus)
      - ``jac(x) → (n_vars,)``: objective gradient
      - ``constraints``: list of one scipy ``ineq`` dict over the
        concatenated slack vector
      - ``n_constraints``: total slack count (for diagnostics)
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

    def fun(x: NDArray) -> float:
        return float(jit["obj"](jnp.asarray(x, dtype=jnp.float32), **packed))

    def jac(x: NDArray) -> NDArray:
        g = jit["obj_grad"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(g, dtype=np.float64)

    def slacks_fn(x: NDArray) -> NDArray:
        s = jit["slacks"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(s, dtype=np.float64)

    def slacks_jac(x: NDArray) -> NDArray:
        J = jit["slacks_jac"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(J, dtype=np.float64)

    # Probe constraint count via a single call at the lifted x0
    # (caller passes x0 in below for the count; here we just expose
    # the callable).
    return dict(
        fun=fun,
        jac=jac,
        constraints=[{
            "type": "ineq",
            "fun": slacks_fn,
            "jac": slacks_jac,
        }],
    )
