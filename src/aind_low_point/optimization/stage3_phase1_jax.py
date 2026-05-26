"""Stage 3 Phase 1: soft-penalty JAX objective with coverage, offsets, depth.

This is the *soft* phase of the new Stage 3 design (2026-05-22). Phase 2
(the existing :mod:`stage3_jax` constraints + ``coverage_objective``)
runs after Phase 1 to do the hard-constrained final polish.

What Phase 1 adds beyond Stage 2's reduced objective:

  - Coverage maximisation (the actual Stage 3 reason for being).
  - Three new per-probe DOFs: ``off_R, off_A, past_target_mm``
    (offsets along the rig-R and rig-A axes, and insertion depth past
    the target).
  - Saturating per-pair clearance margin reward (rewards every pair
    having clearance, not just the worst).
  - Saturating per-(probe, shank, section) threading margin reward
    (rewards every shank-section being deep inside its oval).

What Phase 1 inherits from Stage 2 (Patches A + B):

  - α-wrap envelope SDF body + analytic shank OBBs (dual-rep clearance,
    three categories: body-body, body-shank, shank-shank).
  - Soft-min top-k aggregation per category.
  - ``smooth_abs`` in AP/ML separations and bounds.
  - ``(sx, sy)`` unit-circle spin reparameterization (no ±180° wrap).

x layout: ``(arc_aps, (ml, sx, sy, off_R, off_A, depth) × P)`` — 6 DOFs
per probe. Convert to/from Stage 2's reduced y (n_arcs + 3P) and from
Stage 3's old full x (n_arcs + 5P scalar spin) at the boundaries.

Coverage is supplied as a Python callable that returns a scalar given
the world poses ``(Rs, ts)``. The JAX kernel computes everything else;
the coverage term is added back at the final scipy interface (i.e.,
finite-diff'd through coverage but analytic-grad'd through the JAX
kernel). A future patch will port coverage to JAX for full analytic
gradients.
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
    GaussianCoverageData,
    KdeCoverageData,
    probe_coverage,
)
from aind_low_point.optimization.joint_rerank_jax import (
    MAX_SECTIONS_PAD,
    MAX_SHANKS_PAD,
    _softplus_squared,
    threading_g_matrix,
)
from aind_low_point.optimization.sdf_jax import (
    FIXTURE_PAIR_SLACK_GAINS,
    PROBE_PAIR_SLACK_GAINS,
    dual_rep_fixture_clearance,
    dual_rep_pair_clearance,
    pose_from_optimizer_vars,
    smooth_abs,
    spin_deg_from_sxy,
    unit_circle_penalty,
)


@dataclass(frozen=True)
class FixtureSDFData:
    """Static-in-world fixture body SDF (α-wrap envelope).

    Used for probe-vs-fixture body clearance in Phase 1. Built once
    from the fixture mesh (already canonicalized to world LPS) and
    closure-captured by the JIT'd objective.
    """
    name: str
    grid: jnp.ndarray
    origin: jnp.ndarray
    spacing: jnp.ndarray
    surface: jnp.ndarray


PHASE1_PER_PROBE_VARS = 6  # (ml, sx, sy, off_R, off_A, depth)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def phase1_n_vars(n_arcs: int, n_probes: int) -> int:
    """Phase 1 x-vector length: ``n_arcs + 6P``."""
    return n_arcs + PHASE1_PER_PROBE_VARS * n_probes


def phase1_unpack(x: NDArray, n_arcs: int, probe_idx: int) -> tuple[float, ...]:
    """Return ``(ml, sx, sy, off_R, off_A, depth)`` for probe ``probe_idx``."""
    off = n_arcs + PHASE1_PER_PROBE_VARS * probe_idx
    return tuple(float(x[off + k]) for k in range(PHASE1_PER_PROBE_VARS))


def reduced_to_phase1(
    reduced_y: NDArray, n_arcs: int, n_probes: int
) -> NDArray:
    """Lift a Stage 2 reduced y ``(arc_aps, (ml, sx, sy) × P)`` to a
    Phase 1 x ``(arc_aps, (ml, sx, sy, 0, 0, 0) × P)``.
    """
    out = np.zeros(phase1_n_vars(n_arcs, n_probes), dtype=np.float64)
    out[:n_arcs] = np.asarray(reduced_y[:n_arcs], dtype=np.float64)
    for i in range(n_probes):
        out_off = n_arcs + PHASE1_PER_PROBE_VARS * i
        red_off = n_arcs + 3 * i
        out[out_off + 0] = float(reduced_y[red_off + 0])  # ml
        out[out_off + 1] = float(reduced_y[red_off + 1])  # sx
        out[out_off + 2] = float(reduced_y[red_off + 2])  # sy
        # off_R, off_A, depth default to 0
    return out


def phase1_to_full_x(
    phase1_x: NDArray, n_arcs: int, n_probes: int
) -> NDArray:
    """Convert Phase 1 x ``(ml, sx, sy, off_R, off_A, depth) × P`` to the
    legacy Stage 3 full x ``(ml, spin_deg, off_R, off_A, depth) × P``.

    Spin in degrees recovered via ``atan2(sy, sx)``. Used at the
    handoff into the hard-constrained Phase 2 (which still uses scalar
    spin until Patch B propagates fully there).
    """
    out = np.zeros(n_arcs + 5 * n_probes, dtype=np.float64)
    out[:n_arcs] = np.asarray(phase1_x[:n_arcs], dtype=np.float64)
    for i in range(n_probes):
        in_off = n_arcs + PHASE1_PER_PROBE_VARS * i
        out_off = n_arcs + 5 * i
        ml = float(phase1_x[in_off + 0])
        sx = float(phase1_x[in_off + 1])
        sy = float(phase1_x[in_off + 2])
        spin = float(np.degrees(np.arctan2(sy, sx)))
        out[out_off + 0] = ml
        out[out_off + 1] = spin
        out[out_off + 2] = float(phase1_x[in_off + 3])  # off_R
        out[out_off + 3] = float(phase1_x[in_off + 4])  # off_A
        out[out_off + 4] = float(phase1_x[in_off + 5])  # depth
    return out


# ---------------------------------------------------------------------------
# Saturating reward helper
# ---------------------------------------------------------------------------


def _saturating_reward_mean(
    slack: jnp.ndarray,
    tau: float,
    valid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Mean of ``1 − exp(−max(slack, 0)/τ)`` over valid entries.

    Saturating per-element reward, gated to zero when ``slack ≤ 0``
    (infeasibility is handled by the penalty terms — the reward only
    fires for actual margin). Mean form for problem-size invariance.
    """
    safe = jnp.maximum(0.0, slack)
    h = 1.0 - jnp.exp(-safe / tau)
    if valid is None:
        return jnp.mean(h)
    h_masked = jnp.where(valid > 0, h, 0.0)
    n_valid = jnp.maximum(jnp.sum(valid), 1.0)
    return jnp.sum(h_masked) / n_valid


# ---------------------------------------------------------------------------
# Phase 1 weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase1Weights:
    """Weights for Stage 3 Phase 1 (soft-penalty form).

    Defaults sized so that:
      - Penalty terms (λ_thread, λ_clearance, λ_kinematic) dominate when
        infeasible — order ~100s vs coverage ~17.
      - Saturating margin rewards stay ≤ 12% of coverage in the
        saturated limit (mean form ⇒ max contribution = λ each).
      - ``smooth_abs`` ε = 1e-3 deg matches Stage 2.
    """

    lambda_thread: float = 100.0
    # Per-category weight (body-body, body-shank, shank-shank). The
    # categories surface different geometric failures (deep body
    # overlap vs grazing shank contact) and weighting each at 100 —
    # same as Stage 2's single-min — keeps each signal effective.
    lambda_clearance: float = 100.0
    lambda_kinematic: float = 100.0
    lambda_bounds: float = 1.0
    # See sdf_jax.unit_circle_penalty: keeps (sx, sy) magnitude ≈ 1
    # so poses are consistent across stages.
    lambda_unit_circle: float = 100.0

    # Saturating margin rewards (mean form ⇒ max contribution = λ each).
    lambda_margin_clear: float = 1.0
    lambda_margin_thread: float = 1.0
    tau_clear_mm: float = 0.2          # saturation scale for pair clearance (mm)
    tau_thread_gunits: float = 0.5     # saturation scale for threading slack (g-units)
    # Probe-vs-fixture body clearance: reuses tau_clear_mm; same penalty
    # form as probe-probe. Set to 0 to disable fixture clearance term.
    lambda_clearance_fixture: float = 100.0
    lambda_margin_clear_fixture: float = 1.0

    # Pass-throughs to existing modules. ``min_clearance_mm`` includes
    # a 0.1 mm safety buffer over the α-wrap envelope's own ~50 µm
    # offset — covers the ~4% soft-FN rate where the envelope misses a
    # sharp feature and the raw mesh sticks out past it.
    min_clearance_mm: float = 0.1      # threshold for the hard clearance penalty
    threading_oval_tolerance: float = 0.0
    min_arc_ap_sep_deg: float = 16.0
    min_intra_arc_ml_sep_deg: float = 16.0
    comfortable_ap_deg: float = 50.0
    comfortable_ml_deg: float = 50.0

    # Soft-min knobs for dual-rep clearance (matches Stage 2 defaults).
    softmin_beta: float = 20.0
    top_k_body_body: int = 16
    top_k_body_shank: int = 8
    top_k_shank_shank: int = 8

    # Shaft length used by threading_g_matrix.
    shaft_length_mm: float = 10.0


# ---------------------------------------------------------------------------
# JIT-built objective
# ---------------------------------------------------------------------------


_JIT_CACHE: dict[Hashable, tuple[Callable, Callable]] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def _weights_key(w: Phase1Weights) -> tuple:
    return tuple(
        float(getattr(w, f))
        for f in (
            "lambda_thread", "lambda_clearance", "lambda_kinematic",
            "lambda_bounds", "lambda_margin_clear", "lambda_margin_thread",
            "lambda_clearance_fixture", "lambda_margin_clear_fixture",
            "tau_clear_mm", "tau_thread_gunits", "min_clearance_mm",
            "threading_oval_tolerance", "min_arc_ap_sep_deg",
            "min_intra_arc_ml_sep_deg", "comfortable_ap_deg",
            "comfortable_ml_deg", "softmin_beta", "shaft_length_mm",
        )
    ) + (int(w.top_k_body_body), int(w.top_k_body_shank), int(w.top_k_shank_shank))


def _signature(statics, n_arcs: int, weights: Phase1Weights) -> tuple:
    """Cache key — same per-probe SDF/shank-OBB shape info as Stage 2."""
    has_sdf = any(s.sdf_data is not None for s in statics)
    per_probe_sdf_shapes: tuple = ()
    per_probe_shank_counts: tuple = ()
    n_surf = 0
    if has_sdf:
        shapes = []
        counts = []
        for s in statics:
            if s.sdf_data is None:
                shapes.append(None)
                counts.append(0)
            else:
                shapes.append(
                    tuple(int(x) for x in np.asarray(s.sdf_data["grid"]).shape)
                )
                if n_surf == 0:
                    n_surf = int(np.asarray(s.sdf_data["surface"]).shape[0])
                centers = s.sdf_data.get("shank_centers")
                counts.append(
                    int(np.asarray(centers).shape[0]) if centers is not None else 0
                )
        per_probe_sdf_shapes = tuple(shapes)
        per_probe_shank_counts = tuple(counts)
    return (
        len(statics),
        int(n_arcs),
        MAX_SHANKS_PAD,
        MAX_SECTIONS_PAD,
        has_sdf,
        per_probe_sdf_shapes,
        per_probe_shank_counts,
        n_surf,
        _weights_key(weights),
    )


def _build_jit(
    signature: tuple,
    weights: Phase1Weights,
    coverage_data: tuple[CoverageData, ...] | None = None,
    fixtures: tuple[FixtureSDFData, ...] = (),
    coverage_n_samples: int = 41,
) -> tuple[Callable, Callable]:
    """Build the (fn, grad) pair for one signature.

    All terms — coverage AND probe-fixture clearance — run inside JAX
    when their respective data is provided. ``coverage_data`` has one
    entry per probe (mixed Gaussian / KDE modes supported). ``fixtures``
    is a tuple of static-in-world fixture SDFs; each contributes a
    probe-vs-fixture body clearance penalty and a saturating margin
    reward across the P × n_fixtures pair list.
    """
    (
        n_probes, n_arcs, _max_shanks, _max_sections,
        has_sdf, per_probe_sdf_shapes, _per_probe_shank_counts,
        _n_surf, _w_key,
    ) = signature

    sdf_pair_list: list[tuple[int, int]] = []
    if has_sdf:
        for i in range(n_probes):
            if per_probe_sdf_shapes[i] is None:
                continue
            for j in range(i + 1, n_probes):
                if per_probe_sdf_shapes[j] is None:
                    continue
                sdf_pair_list.append((i, j))

    lt = float(weights.lambda_thread)
    lc = float(weights.lambda_clearance)
    lk = float(weights.lambda_kinematic)
    lb = float(weights.lambda_bounds)
    luc = float(getattr(weights, "lambda_unit_circle", 100.0))
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

    arc_pairs = jnp.asarray(
        [(a, b) for a in range(n_arcs) for b in range(a + 1, n_arcs)],
        dtype=jnp.int32,
    ).reshape(-1, 2)

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
        Rs = []
        ts = []
        thread_g_list = []
        thread_mask_list = []
        j_thread = jnp.float32(0.0)
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
            g = threading_g_matrix(
                R, t, tips_local[i],
                s_axes[i], s_centers[i], s_e1[i], s_e2[i],
                s_cos[i], s_sin[i], s_a[i], s_b[i],
                shaft_length_mm=shaft_len,
            )  # (S, SH)
            valid_g = section_mask[i][:, None] * shank_mask[i][None, :]
            # Penalty (Patch A: clamped, finite — no inf): max(0, g - tol)²
            excess = jnp.maximum(0.0, g - thread_tol)
            j_thread = j_thread + jnp.sum(valid_g * excess * excess)
            # Slack for the margin reward: tol - g.
            slack = thread_tol - g
            thread_g_list.append(slack.reshape(-1))
            thread_mask_list.append(valid_g.reshape(-1))

        # AP separation (smooth_abs over arc-pair differences)
        if arc_pairs.shape[0] > 0:
            ap_diffs = smooth_abs(
                arc_aps[arc_pairs[:, 0]] - arc_aps[arc_pairs[:, 1]]
            )
            short_ap = jnp.maximum(0.0, min_arc_ap - ap_diffs)
            j_arc_ap = jnp.sum(short_ap * short_ap)
        else:
            j_arc_ap = jnp.float32(0.0)

        # Intra-arc ML separation
        ml_vals = jnp.stack(
            [x[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)]
        )
        ml_diff = smooth_abs(ml_vals[:, None] - ml_vals[None, :])
        short_ml = jnp.maximum(0.0, min_intra_ml - ml_diff)
        j_ml = jnp.sum(same_arc_mask * short_ml * short_ml)

        # Soft bounds (smooth_abs ⇒ comfortable-range pull-back).
        j_bounds = _softplus_squared(smooth_abs(arc_aps) - cap)
        j_bounds = j_bounds + _softplus_squared(smooth_abs(ml_vals) - cml)

        # Dual-rep clearance via 3-helper split (matches Stage 2
        # joint_rerank_jax for shared XLA cache + per-call perf).
        # Pre-compute world-frame body surface samples once per probe
        # so the per-pair calls don't redo ``surface @ R.T + t``.
        world_surfaces = [
            sdf_surfaces[i] @ Rs[i].T + ts[i] for i in range(n_probes)
        ]
        j_clear = jnp.float32(0.0)
        pair_hard_clearances = []
        for ia, ib in sdf_pair_list:
            pc = dual_rep_pair_clearance(
                Rs[ia], ts[ia], Rs[ib], ts[ib],
                sdf_grids[ia], sdf_origins[ia], sdf_spacings[ia],
                sdf_grids[ib], sdf_origins[ib], sdf_spacings[ib],
                world_surfaces[ia], world_surfaces[ib],
                shank_obb_centers[ia], shank_obb_halves[ia],
                shank_obb_centers[ib], shank_obb_halves[ib],
                beta=beta,
                top_k_body_body=tk_bb,
                top_k_body_shank=tk_bs,
                top_k_shank_shank=tk_ss,
            )
            softs = (
                pc.body_body[1], pc.body_shank_corners[1],
                pc.body_shank_obb[1], pc.shank_shank[1],
            )
            for d_soft, gain in zip(softs, PROBE_PAIR_SLACK_GAINS):
                short = jnp.maximum(0.0, min_clear - d_soft) * gain
                j_clear = j_clear + short * short
            # Margin uses the worst hard category clearance per pair.
            hards = (
                pc.body_body[0], pc.body_shank_corners[0],
                pc.body_shank_obb[0], pc.shank_shank[0],
            )
            pair_hard_clearances.append(jnp.min(jnp.stack(hards)))

        # Probe-vs-fixture clearance: dual-rep (body + OBB).
        j_clear_fixture = jnp.float32(0.0)
        fixture_hard_clearances: list[jnp.ndarray] = []
        if fixtures:
            for fx in fixtures:
                for i in range(n_probes):
                    if has_sdf and per_probe_sdf_shapes[i] is None:
                        continue
                    fc = dual_rep_fixture_clearance(
                        Rs[i], ts[i],
                        sdf_grids[i], sdf_origins[i], sdf_spacings[i],
                        fx.grid, fx.origin, fx.spacing,
                        world_surfaces[i], fx.surface,
                        shank_obb_centers[i], shank_obb_halves[i],
                        beta=beta, top_k_body=tk_bb, top_k_obb=tk_bs,
                    )
                    softs = (fc.body[1], fc.obb[1])
                    for d_soft, gain in zip(softs, FIXTURE_PAIR_SLACK_GAINS):
                        short = jnp.maximum(0.0, min_clear - d_soft) * gain
                        j_clear_fixture = j_clear_fixture + short * short
                    fixture_hard_clearances.append(jnp.minimum(fc.body[0], fc.obb[0]))

        # Saturating per-pair clearance margin reward (mean form). Combines
        # probe-probe and probe-fixture clearances under one mean so the
        # reward is consistent regardless of how many fixtures are loaded.
        all_hard_clearances = pair_hard_clearances + fixture_hard_clearances
        if all_hard_clearances:
            all_clears = jnp.stack(all_hard_clearances)
            reward_clear = _saturating_reward_mean(all_clears, tau_c)
        else:
            reward_clear = jnp.float32(0.0)

        # Saturating per-(probe, shank, section) threading margin reward.
        if thread_g_list:
            slacks = jnp.concatenate(thread_g_list)
            masks = jnp.concatenate(thread_mask_list)
            reward_thread = _saturating_reward_mean(slacks, tau_t, valid=masks)
        else:
            reward_thread = jnp.float32(0.0)

        # Coverage: sum across probes. ``coverage_data`` is a Python
        # tuple closed over the trace; per-probe mode (Gaussian vs KDE)
        # is fixed at JIT-build time.
        if coverage_data is not None:
            coverage_total = jnp.float32(0.0)
            for i in range(n_probes):
                coverage_total = coverage_total + probe_coverage(
                    Rs[i], ts[i], tips_local[i], shank_mask[i],
                    coverage_data[i], n_samples=coverage_n_samples,
                )
        else:
            coverage_total = jnp.float32(0.0)

        # Unit-circle pull on (sx, sy). x layout is
        # (arc_aps, (ml, sx, sy, off_R, off_A, depth) × P) — stride 6.
        sx_arr = x[n_arcs + 1::PHASE1_PER_PROBE_VARS][:n_probes]
        sy_arr = x[n_arcs + 2::PHASE1_PER_PROBE_VARS][:n_probes]
        j_unit_circle = unit_circle_penalty(sx_arr, sy_arr)

        return (
            - coverage_total
            + lt * j_thread
            + lc * j_clear
            + float(weights.lambda_clearance_fixture) * j_clear_fixture
            + lk * (j_arc_ap + j_ml)
            + lb * j_bounds
            + luc * j_unit_circle
            - lmc * reward_clear
            - lmt * reward_thread
        )

    jit_obj = jax.jit(_objective)
    jit_grad = jax.jit(jax.grad(_objective))
    return jit_obj, jit_grad


def _pack_statics(statics, n_arcs: int) -> dict:
    """Pack per-candidate static data into padded jnp tensors. Mirrors
    Stage 2's ``_pack_statics`` exactly so per-probe data lines up.
    """
    P = len(statics)
    max_shanks, max_sections = MAX_SHANKS_PAD, MAX_SECTIONS_PAD
    target_LPS = np.zeros((P, 3), dtype=np.float32)
    pivot_local = np.zeros((P, 3), dtype=np.float32)
    arc_idx = np.zeros(P, dtype=np.int32)
    tips_local = np.zeros((P, max_shanks, 3), dtype=np.float32)
    shank_mask = np.zeros((P, max_shanks), dtype=np.float32)
    s_axes = np.zeros((P, max_sections, 3), dtype=np.float32)
    s_axes[:, :, 2] = 1.0
    s_centers = np.zeros((P, max_sections, 3), dtype=np.float32)
    s_e1 = np.zeros((P, max_sections, 3), dtype=np.float32)
    s_e1[:, :, 0] = 1.0
    s_e2 = np.zeros((P, max_sections, 3), dtype=np.float32)
    s_e2[:, :, 1] = 1.0
    s_cos = np.ones((P, max_sections), dtype=np.float32)
    s_sin = np.zeros((P, max_sections), dtype=np.float32)
    s_a = np.ones((P, max_sections), dtype=np.float32)
    s_b = np.ones((P, max_sections), dtype=np.float32)
    section_mask = np.zeros((P, max_sections), dtype=np.float32)
    for i, s in enumerate(statics):
        target_LPS[i] = s.target_LPS
        pivot_local[i] = s.pivot_local
        arc_idx[i] = s.arc_idx
        ns = min(int(s.shank_tips_local.shape[0]), max_shanks)
        if ns > 0:
            tips_local[i, :ns] = s.shank_tips_local[:ns]
            shank_mask[i, :ns] = 1.0
        nsec = min(int(s.section_axes.shape[0]), max_sections)
        if nsec > 0:
            s_axes[i, :nsec] = s.section_axes[:nsec]
            s_centers[i, :nsec] = s.section_centers[:nsec]
            s_e1[i, :nsec] = s.section_e1[:nsec]
            s_e2[i, :nsec] = s.section_e2[:nsec]
            s_cos[i, :nsec] = s.section_cos_theta[:nsec]
            s_sin[i, :nsec] = s.section_sin_theta[:nsec]
            s_a[i, :nsec] = s.section_a[:nsec]
            s_b[i, :nsec] = s.section_b[:nsec]
            section_mask[i, :nsec] = 1.0

    same_arc_mask = np.zeros((P, P), dtype=np.float32)
    for i in range(P):
        for j in range(i + 1, P):
            if statics[i].arc_idx == statics[j].arc_idx:
                same_arc_mask[i, j] = 1.0

    out = dict(
        target_LPS=jnp.asarray(target_LPS),
        pivot_local=jnp.asarray(pivot_local),
        arc_idx=jnp.asarray(arc_idx),
        tips_local=jnp.asarray(tips_local),
        shank_mask=jnp.asarray(shank_mask),
        s_axes=jnp.asarray(s_axes),
        s_centers=jnp.asarray(s_centers),
        s_e1=jnp.asarray(s_e1),
        s_e2=jnp.asarray(s_e2),
        s_cos=jnp.asarray(s_cos),
        s_sin=jnp.asarray(s_sin),
        s_a=jnp.asarray(s_a),
        s_b=jnp.asarray(s_b),
        section_mask=jnp.asarray(section_mask),
        same_arc_mask=jnp.asarray(same_arc_mask),
    )
    # SDF + shank-OBB tuples, per-probe shapes (heterogeneous across kinds).
    sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces = [], [], [], []
    shank_obb_centers, shank_obb_halves = [], []
    for s in statics:
        if s.sdf_data is not None:
            sdf_grids.append(jnp.asarray(s.sdf_data["grid"], dtype=jnp.float32))
            sdf_origins.append(jnp.asarray(s.sdf_data["origin"], dtype=jnp.float32))
            sdf_spacings.append(jnp.asarray(s.sdf_data["spacing"], dtype=jnp.float32))
            sdf_surfaces.append(jnp.asarray(s.sdf_data["surface"], dtype=jnp.float32))
            shank_obb_centers.append(
                jnp.asarray(s.sdf_data.get("shank_centers",
                                            np.zeros((0, 3), dtype=np.float32)),
                            dtype=jnp.float32)
            )
            shank_obb_halves.append(
                jnp.asarray(s.sdf_data.get("shank_halves",
                                            np.zeros((0, 3), dtype=np.float32)),
                            dtype=jnp.float32)
            )
        else:
            sdf_grids.append(jnp.zeros((2, 2, 2), dtype=jnp.float32))
            sdf_origins.append(jnp.zeros(3, dtype=jnp.float32))
            sdf_spacings.append(jnp.float32(1.0))
            sdf_surfaces.append(jnp.zeros((1, 3), dtype=jnp.float32))
            shank_obb_centers.append(jnp.zeros((0, 3), dtype=jnp.float32))
            shank_obb_halves.append(jnp.zeros((0, 3), dtype=jnp.float32))
    out["sdf_grids"] = tuple(sdf_grids)
    out["sdf_origins"] = tuple(sdf_origins)
    out["sdf_spacings"] = tuple(sdf_spacings)
    out["sdf_surfaces"] = tuple(sdf_surfaces)
    out["shank_obb_centers"] = tuple(shank_obb_centers)
    out["shank_obb_halves"] = tuple(shank_obb_halves)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cache_stats() -> dict:
    return {**_CACHE_STATS, "entries": len(_JIT_CACHE)}


def make_phase1_objective(
    statics,
    n_arcs: int,
    coverage_data: tuple[CoverageData, ...] | None = None,
    fixtures: tuple[FixtureSDFData, ...] = (),
    weights: Phase1Weights = Phase1Weights(),
    *,
    coverage_n_samples: int = 41,
) -> tuple[Callable[[NDArray], float], Callable[[NDArray], NDArray]]:
    """Build ``(fun, jac)`` scipy callables for Phase 1's soft objective.

    All terms — feasibility penalties, margin rewards, AND coverage —
    are computed inside one JIT'd JAX kernel with analytic gradient.
    Per-probe coverage data is supplied via ``coverage_data``; each
    entry is either a :class:`~coverage_jax.GaussianCoverageData`
    (single target with σ) or :class:`~coverage_jax.KdeCoverageData`
    (pre-baked density grid from retro points). Mixed modes across
    probes are fine — the per-probe mode is Python-static at trace
    time.

    When ``coverage_data is None``, coverage contributes nothing —
    useful for testing the Phase 1 machinery without committing to a
    density representation.

    Parameters
    ----------
    statics
        List of ``_ProbeStatic`` (from joint_rerank). Must have
        ``sdf_data`` populated (α-wrap envelope + shank OBBs).
    n_arcs
        Number of arcs.
    coverage_data
        Tuple of length ``len(statics)`` — one CoverageData per probe.
        Use :func:`coverage_jax.build_coverage_data_from_probe_context`
        to construct each entry from a Stage 3 ProbeContext.
    weights
        :class:`Phase1Weights` — see defaults; tuned for coverage ~17
        to dominate the soft-penalty plus margin-reward signals.
    coverage_n_samples
        Simpson's-rule sample count per shank (default 41, matching
        the legacy Stage 3 coverage).
    """
    # Extend the cache signature with fixture grid shapes so distinct
    # fixture sets don't collide.
    fix_shapes = tuple(
        tuple(int(d) for d in np.asarray(fx.grid).shape) for fx in fixtures
    )
    sig = _signature(statics, n_arcs, weights) + (fix_shapes,)
    if sig not in _JIT_CACHE:
        _JIT_CACHE[sig] = _build_jit(
            sig[:-1], weights,
            coverage_data=coverage_data,
            fixtures=fixtures,
            coverage_n_samples=coverage_n_samples,
        )
        _CACHE_STATS["misses"] += 1
    else:
        _CACHE_STATS["hits"] += 1
    jit_obj, jit_grad = _JIT_CACHE[sig]
    packed = _pack_statics(statics, n_arcs)

    def fun(x: NDArray) -> float:
        return float(jit_obj(jnp.asarray(x, dtype=jnp.float32), **packed))

    def jac(x: NDArray) -> NDArray:
        g = jit_grad(jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(g, dtype=np.float64)

    return fun, jac
