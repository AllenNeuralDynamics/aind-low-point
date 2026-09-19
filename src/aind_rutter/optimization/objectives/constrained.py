"""Phase 2 hard-constrained JAX polish with coverage maximisation.

Phase 2 follows Phase 1's soft-penalty warm-up. It uses the *same* x
layout (``(arc_aps, (ml, sx, sy, off_R, off_A, depth) × P)``) but moves
all feasibility terms from the objective into solver inequality
constraints:

  - Threading: ``g_thread ≤ tol`` per (probe, shank, section)
  - Clearance probe-probe (dual-rep): ``d_soft ≥ min_clear`` per
    (pair, category)
  - Clearance probe-fixture (body): same, per (probe, fixture)
  - Arc-AP separation: ``smooth_abs(ap_diff) ≥ min_sep`` per arc pair
  - Intra-arc ML separation: same per intra-arc pair

The objective shrinks to coverage + soft bounds + saturating margin
bonuses (clearance + threading) — exactly the bonuses from Phase 1.
The constrained solver enforces strict feasibility via the constraints; the
margin bonuses keep the gradient meaningful inside the feasible region (where
coverage may be locally flat).

Shares all geometry/density kernels with Phase 1 — no duplicate code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aind_rutter.optimization.clearance.kernels import (
    SLACK_GAIN_BODY_BODY,
    SLACK_GAIN_BODY_SHANK_CORNERS,
    SLACK_GAIN_FIXTURE_BODY,
    FixtureClearance,
    PairClearance,
    pose_from_optimizer_vars,
    smooth_abs,
    spin_deg_from_sxy,
    trilinear_sdf,
    unit_circle_penalty,
)
from aind_rutter.optimization.clearance.sweep import (
    build_padded_fixture_table,
    cast_fixture_grids,
    cast_packed_grids,
    swept_fixture_clearances,
    swept_pair_clearances,
)
from aind_rutter.optimization.objectives.cache_keys import weights_cache_key
from aind_rutter.optimization.objectives.coverage import (
    CoverageData,
    coverage_per_probe_over_probes,
    normalized_coverage_objective,
    probe_coverage,
)
from aind_rutter.optimization.objectives.soft import (
    PHASE1_PER_PROBE_VARS,
    BrainSDFData,
    FixtureSDFData,
    _pack_statics,
    _saturating_reward_mean,
    _saturating_reward_worst,
)
from aind_rutter.optimization.objectives.threading import (
    MAX_SECTIONS_PAD,
    MAX_SHANKS_PAD,
    _softplus_squared,
    threading_g_matrix,
)
from aind_rutter.optimization.pipeline.records import Phase2Problem

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase2Weights:
    """Weights for the Phase 2 hard-constrained objective.

    Phase 2 drops the feasibility-penalty λ's (threading, clearance,
    kinematic) — those terms are now constraints. Keeps coverage, soft
    bounds, and the two saturating margin bonuses.
    """

    lambda_bounds: float = 1.0
    # See sdf.kernels.unit_circle_penalty. Reduced 100 -> 10 (2026-05-26).
    lambda_unit_circle: float = 10.0

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
    # Gain on the OBB-based slack categories (body-shank box, shank-shank, probe
    # OBB vs fixture). The default keeps them two orders above the mm-native
    # voxel-SDF categories; lowering it evens out the KKT scaling.
    obb_slack_gain: float = 100.0
    # Shape the clearance reward with the soft minimum instead of the hard one:
    # the hard min's gradient is one sample's and jumps as the closest sample
    # changes, which stalls the line search.
    smooth_clearance_reward: bool = False
    top_k_body_body: int = 16
    top_k_body_shank: int = 8
    top_k_shank_shank: int = 8

    shaft_length_mm: float = 10.0

    # Brain containment: each shank tip must stay this far inside the brain
    # surface (hard constraint, only active when a brain SDF is passed).
    brain_margin_mm: float = 0.2

    # Coverage. ``lambda_cov`` is the overall coverage-vs-clearance gain
    # (multiplies the coverage scalar in the objective). When
    # ``coverage_ceilings`` are passed to make_phase2, coverage becomes the
    # normalized [0,1] blend ``(1 - cov_alpha)·mean(norm) + cov_alpha·worst(norm)``;
    # ``cov_alpha`` = 0 ⇒ pure average coverage, 1 ⇒ pure minimax on the laggard.
    lambda_cov: float = 1.0
    cov_alpha: float = 0.0
    softmin_beta_cov: float = 20.0


# ---------------------------------------------------------------------------
# Pose helper used by both objective and constraints
# ---------------------------------------------------------------------------


def _poses_from_x(x, n_arcs, n_probes, target_LPS, pivot_local, arc_idx):
    """Stacked (Rs, ts) ``(P,3,3)``/``(P,3)`` per probe from a Phase 2 x vector,
    vmapped over probes (one pose subgraph instead of P unrolled copies). x after
    the arcs is (ml, sx, sy, off_R, off_A, depth) × P."""
    arc_aps = x[:n_arcs]
    xp = x[n_arcs : n_arcs + PHASE1_PER_PROBE_VARS * n_probes].reshape(
        n_probes, PHASE1_PER_PROBE_VARS
    )
    aps = arc_aps[arc_idx]

    def _one(xp6, ap, target, pivot):
        return pose_from_optimizer_vars(
            target_LPS=target,
            ap_deg=ap,
            ml_deg=xp6[0],
            spin_deg=spin_deg_from_sxy(xp6[1], xp6[2]),
            offset_R_mm=xp6[3],
            offset_A_mm=xp6[4],
            past_target_mm=xp6[5],
            recording_center_local=pivot,
        )

    return jax.vmap(_one)(xp, aps, target_LPS, pivot_local)


def _threading_g_per_probe(
    Rs,
    ts,
    tips_local,
    s_axes,
    s_centers,
    s_e1,
    s_e2,
    s_cos,
    s_sin,
    s_a,
    s_b,
    section_mask,
    shank_mask,
    w_normals=None,
    w_offsets=None,
    *,
    shaft_len,
):
    """vmap ``threading_g_matrix`` over probes → ``(g, valid)``, each
    ``(P, S, SH)`` — one threading subgraph instead of P unrolled copies. Both
    the objective (reward) and the constraint vector consume this; flatten with
    ``.reshape(-1)`` for the probe-major order the old per-probe loop produced."""

    def _one(R, t, tips, sax, scen, se1, se2, scos, ssin, sa, sb, sec_m, sh_m, wn, wo):
        g = threading_g_matrix(
            R,
            t,
            tips,
            sax,
            scen,
            se1,
            se2,
            scos,
            ssin,
            sa,
            sb,
            shaft_length_mm=shaft_len,
            w_normals=wn,
            w_offsets=wo,
        )
        return g, sec_m[:, None] * sh_m[None, :]

    return jax.vmap(_one)(
        Rs,
        ts,
        tips_local,
        s_axes,
        s_centers,
        s_e1,
        s_e2,
        s_cos,
        s_sin,
        s_a,
        s_b,
        section_mask,
        shank_mask,
        w_normals,
        w_offsets,
    )


# ---------------------------------------------------------------------------
# JIT cache
# ---------------------------------------------------------------------------


_JIT_CACHE: dict[Hashable, dict] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


_weights_key = weights_cache_key


def _signature(statics, n_arcs, weights, fixtures, brain_sdf=None):
    has_sdf = any(s.sdf_data is not None for s in statics)
    per_probe_sdf_shapes: tuple = ()
    if has_sdf:
        per_probe_sdf_shapes = tuple(
            None
            if s.sdf_data is None
            else tuple(int(x) for x in np.asarray(s.sdf_data["grid"]).shape)
            for s in statics
        )
    fix_shapes = tuple(
        tuple(int(d) for d in np.asarray(fx.grid).shape) for fx in fixtures
    )
    brain_shape = (
        tuple(int(d) for d in np.asarray(brain_sdf.grid).shape)
        if brain_sdf is not None
        else None
    )
    return (
        len(statics),
        int(n_arcs),
        MAX_SHANKS_PAD,
        MAX_SECTIONS_PAD,
        has_sdf,
        per_probe_sdf_shapes,
        fix_shapes,
        _weights_key(weights),
        brain_shape,
    )


# ---------------------------------------------------------------------------
# Build JIT'd objective + slacks
# ---------------------------------------------------------------------------


_LARGE_SLACK = 1e3  # sentinel for masked-out (padded) constraints
PADDED_SLACK = _LARGE_SLACK
# Constraint groups, in the order they are concatenated into the slack vector.
SLACK_GROUPS = ("thread", "probe_pair", "probe_fixture", "brain", "arc_sep", "ml_sep")


def _build_jit(  # noqa: C901
    signature: tuple,
    weights: Phase2Weights,
    coverage_data: tuple[CoverageData, ...] | None,
    fixtures: tuple[FixtureSDFData, ...],
    coverage_n_samples: int = 41,
    brain_sdf: "BrainSDFData | None" = None,
    coverage_ceilings: "tuple[float, ...] | None" = None,
    coverage_weights: "tuple[float, ...] | None" = None,
) -> dict:
    """Build JIT'd (obj, obj_grad, all_slacks, all_slacks_jac) for one signature."""
    (
        n_probes,
        n_arcs,
        _ms,
        _msec,
        has_sdf,
        sdf_shapes,
        _fix_shapes,
        _w_key,
        _brain_shape,
    ) = signature

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
    luc = float(getattr(weights, "lambda_unit_circle", 100.0))
    lmc = float(weights.lambda_margin_clear)
    lmt = float(weights.lambda_margin_thread)
    tau_c = float(weights.tau_clear_mm)
    tau_t = float(weights.tau_thread_gunits)
    min_clear = float(weights.min_clearance_mm)
    obb_gain = float(weights.obb_slack_gain)
    smooth_reward = bool(weights.smooth_clearance_reward)
    pair_gains = (
        SLACK_GAIN_BODY_BODY,
        SLACK_GAIN_BODY_SHANK_CORNERS,
        obb_gain,
        obb_gain,
    )
    fixture_gains = (SLACK_GAIN_FIXTURE_BODY, obb_gain)
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
    brain_margin = float(getattr(weights, "brain_margin_mm", 0.2))
    if brain_sdf is not None:
        brain_grid = jnp.asarray(brain_sdf.grid)
        brain_origin = jnp.asarray(brain_sdf.origin)
        brain_spacing = jnp.asarray(brain_sdf.spacing)

    # Coverage normalization constants (baked into the trace; keyed in make_phase2).
    cov_ceilings = (
        jnp.asarray(coverage_ceilings, dtype=jnp.float32)
        if coverage_ceilings is not None
        else None
    )
    cov_weights = (
        jnp.asarray(coverage_weights, dtype=jnp.float32)
        if coverage_weights is not None
        else None
    )
    lambda_cov = float(getattr(weights, "lambda_cov", 1.0))
    cov_alpha = float(getattr(weights, "cov_alpha", 0.0))
    beta_cov = float(getattr(weights, "softmin_beta_cov", 20.0))

    # Padded fixture table (stacked edge-padded grids + n_real_f), built ONCE
    # from the closure-captured fixtures so the fixture × probe clearance is a
    # single fused vmap in both _objective and _all_slacks.
    fix_table = build_padded_fixture_table(fixtures)

    # ---- Objective: scalar minimised by the constrained solver ----
    def _objective(
        x,
        target_LPS,
        pivot_local,
        arc_idx,
        tips_local,
        shank_mask,
        s_axes,
        s_centers,
        s_e1,
        s_e2,
        s_cos,
        s_sin,
        s_a,
        s_b,
        section_mask,
        same_arc_mask,
        sdf_grids,
        sdf_origins,
        sdf_spacings,
        sdf_surfaces,
        shank_obb_centers,
        shank_obb_halves,
        sdf_table=None,
        w_normals=None,
        w_offsets=None,
    ):
        arc_aps = x[:n_arcs]
        Rs, ts = _poses_from_x(
            x,
            n_arcs,
            n_probes,
            target_LPS,
            pivot_local,
            arc_idx,
        )

        # Coverage. With ceilings present, switch to the weighted normalized
        # objective (+ optional soft-min fairness floor); else legacy raw sum.
        coverage_total = jnp.float32(0.0)
        if coverage_data is not None:
            if cov_ceilings is not None:
                cov_pp = coverage_per_probe_over_probes(
                    Rs,
                    ts,
                    tips_local,
                    shank_mask,
                    coverage_data,
                    n_samples=coverage_n_samples,
                )
                coverage_total = normalized_coverage_objective(
                    cov_pp,
                    cov_ceilings,
                    alpha=cov_alpha,
                    softmin_beta=beta_cov,
                    weights=cov_weights,
                )
            else:
                for i in range(n_probes):
                    coverage_total = coverage_total + probe_coverage(
                        Rs[i],
                        ts[i],
                        tips_local[i],
                        shank_mask[i],
                        coverage_data[i],
                        n_samples=coverage_n_samples,
                    )

        # Soft bounds: pull-back from comfort range (smooth_abs).
        ml_vals = jnp.stack(
            [x[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)]
        )
        j_bounds = _softplus_squared(smooth_abs(arc_aps) - cap)
        j_bounds = j_bounds + _softplus_squared(smooth_abs(ml_vals) - cml)

        # Margin bonuses: saturating per-pair (clear) and per-tuple (thread).
        # Mirror Phase 1's computation but skip the soft penalty terms.
        _tg, _tvalid = _threading_g_per_probe(
            Rs,
            ts,
            tips_local,
            s_axes,
            s_centers,
            s_e1,
            s_e2,
            s_cos,
            s_sin,
            s_a,
            s_b,
            section_mask,
            shank_mask,
            w_normals,
            w_offsets,
            shaft_len=shaft_len,
        )
        # (P, -1): per-probe view for worst-shank reward. Hard constraint
        # computes its own flat view from _tg/_tvalid below.
        _n_probes_t = _tg.shape[0]
        _thread_slacks_pp = (thread_tol - _tg).reshape(_n_probes_t, -1)
        _thread_masks_pp = _tvalid.reshape(_n_probes_t, -1)

        # Probe-probe clearance, vmapped over the static pair list (one dual-rep
        # subgraph vs C(P,2) unrolled — see clearance_sweep). Objective only needs
        # the per-pair worst-category hard clearance for the saturating reward.
        if sdf_pair_list and sdf_table is not None:
            _pa = jnp.asarray([a for a, _ in sdf_pair_list], jnp.int32)
            _pb = jnp.asarray([b for _, b in sdf_pair_list], jnp.int32)
            _phard, _psoft = swept_pair_clearances(
                Rs,
                ts,
                sdf_table,
                _pa,
                _pb,
                beta=beta,
                top_k_body_body=tk_bb,
                top_k_body_shank=tk_bs,
                top_k_shank_shank=tk_ss,
            )
            _pvals = _psoft if smooth_reward else _phard
            pair_hard_clearances = jnp.min(_pvals, axis=1)  # (n_pairs,)
        else:
            pair_hard_clearances = None

        # Probe-vs-fixture clearance, vmapped over probes per fixture (objective
        # needs only the per-(fixture,probe) worst-category hard clearance).
        if fixtures and sdf_table is not None:
            _fidx = [
                i for i in range(n_probes) if (not has_sdf) or sdf_shapes[i] is not None
            ]
            _fh, _fs = swept_fixture_clearances(
                Rs,
                ts,
                sdf_table,
                fix_table,
                _fidx,
                beta=beta,
                top_k_body=tk_bb,
                top_k_obb=tk_bs,
            )
            _fvals = _fs if smooth_reward else _fh
            fixture_hard_clearances = jnp.min(_fvals, axis=2).reshape(-1)
        else:
            fixture_hard_clearances = None

        # pair_hard_clearances / fixture_hard_clearances are (N,) arrays or None.
        _hard_parts = []
        if pair_hard_clearances is not None:
            _hard_parts.append(pair_hard_clearances)
        if fixture_hard_clearances is not None:
            _hard_parts.append(fixture_hard_clearances)
        reward_clear = (
            _saturating_reward_mean(jnp.concatenate(_hard_parts), tau_c)
            if _hard_parts
            else jnp.float32(0.0)
        )
        reward_thread = _saturating_reward_worst(
            _thread_slacks_pp, tau_t, _thread_masks_pp
        )

        # Unit-circle pull on (sx, sy). x stride = PHASE1_PER_PROBE_VARS = 6.
        sx_arr = x[n_arcs + 1 :: PHASE1_PER_PROBE_VARS][:n_probes]
        sy_arr = x[n_arcs + 2 :: PHASE1_PER_PROBE_VARS][:n_probes]
        j_unit_circle = unit_circle_penalty(sx_arr, sy_arr)

        return (
            -lambda_cov * coverage_total
            + lb * j_bounds
            + luc * j_unit_circle
            - lmc * reward_clear
            - lmt * reward_thread
        )

    # ---- Slack groups in natural shapes, SLACK_GROUPS order; scipy/IPOPT see ----
    # ---- the flat concatenation ineq[g(x) ≥ 0] built by _all_slacks.          ----
    def _slack_parts(
        x,
        target_LPS,
        pivot_local,
        arc_idx,
        tips_local,
        shank_mask,
        s_axes,
        s_centers,
        s_e1,
        s_e2,
        s_cos,
        s_sin,
        s_a,
        s_b,
        section_mask,
        same_arc_mask,
        sdf_grids,
        sdf_origins,
        sdf_spacings,
        sdf_surfaces,
        shank_obb_centers,
        shank_obb_halves,
        sdf_table=None,
        w_normals=None,
        w_offsets=None,
    ):
        arc_aps = x[:n_arcs]
        Rs, ts = _poses_from_x(
            x,
            n_arcs,
            n_probes,
            target_LPS,
            pivot_local,
            arc_idx,
        )

        # Threading slacks: tol - g, masked. Padded entries → +LARGE. vmapped
        # over probes; reshape(-1) is probe-major (== old per-probe concatenate).
        _tg, _tvalid = _threading_g_per_probe(
            Rs,
            ts,
            tips_local,
            s_axes,
            s_centers,
            s_e1,
            s_e2,
            s_cos,
            s_sin,
            s_a,
            s_b,
            section_mask,
            shank_mask,
            w_normals,
            w_offsets,
            shaft_len=shaft_len,
        )
        thread_vec = jnp.where(_tvalid > 0, thread_tol - _tg, _LARGE_SLACK)

        # Clearance probe-probe: d_soft − min_clear per (pair, category), vmapped
        # over the static pair list (one dual-rep subgraph vs C(P,2) unrolled).
        # ``soft`` is (n_pairs, 4) in PROBE_PAIR_SLACK_GAINS category order, so
        # ``.reshape(-1)`` reproduces the pair-major/category-minor constraint
        # order EXACTLY (each slack is independent ⇒ bit-exact, no reduction).
        if sdf_pair_list and sdf_table is not None:
            _pa = jnp.asarray([a for a, _ in sdf_pair_list], jnp.int32)
            _pb = jnp.asarray([b for _, b in sdf_pair_list], jnp.int32)
            _, _soft = swept_pair_clearances(
                Rs,
                ts,
                sdf_table,
                _pa,
                _pb,
                beta=beta,
                top_k_body_body=tk_bb,
                top_k_body_shank=tk_bs,
                top_k_shank_shank=tk_ss,
            )
            _gains = jnp.asarray(pair_gains, jnp.float32)
            clear_pp_vec = (_soft - min_clear) * _gains
        else:
            clear_pp_vec = jnp.zeros(0)

        # Probe-fixture clearance: dual-rep (body voxel-SDF + probe-OBB
        # vs fixture surface samples). See PairClearance / FixtureClearance
        # in sdf.kernels for category definitions.
        # Probe-fixture clearance, vmapped over probes per fixture. ``soft`` is
        # (n_fix, n_sdf, 2) in FIXTURE_PAIR_SLACK_GAINS category order, so
        # ``.reshape(-1)`` reproduces the fixture-major/probe-minor/category-minor
        # constraint order EXACTLY (each slack independent ⇒ bit-exact).
        if fixtures and sdf_table is not None:
            _fidx = [
                i for i in range(n_probes) if (not has_sdf) or sdf_shapes[i] is not None
            ]
            _, _fsoft = swept_fixture_clearances(
                Rs,
                ts,
                sdf_table,
                fix_table,
                _fidx,
                beta=beta,
                top_k_body=tk_bb,
                top_k_obb=tk_bs,
            )
            _fgains = jnp.asarray(fixture_gains, jnp.float32)
            clear_pf_vec = (_fsoft - min_clear) * _fgains
        else:
            clear_pf_vec = jnp.zeros(0)

        # Brain containment: each shank tip must be inside the brain by at
        # least ``brain_margin``. SDF is negative inside, so the slack is
        # ``-(d + margin) ≥ 0``. Padded shanks → +LARGE (don't constrain).
        brain_vec = jnp.zeros(0)
        if brain_sdf is not None:
            # Batched over probes: (P, max_shanks, 3) world tips → one gather.
            # reshape(-1) is probe-major (== the old per-probe concatenate).
            world_tips = (
                jnp.matmul(tips_local, jnp.transpose(Rs, (0, 2, 1))) + ts[:, None, :]
            )
            d = trilinear_sdf(brain_grid, brain_origin, brain_spacing, world_tips)
            s = -(d + brain_margin)
            brain_vec = jnp.where(shank_mask > 0, s, _LARGE_SLACK)

        # Arc-AP separation: smooth_abs(diff) − min_arc_ap_sep.
        if arc_pairs.shape[0] > 0:
            ap_diffs = smooth_abs(
                arc_aps[arc_pairs_j[:, 0]] - arc_aps[arc_pairs_j[:, 1]]
            )
            ap_sep_vec = ap_diffs - min_arc_ap
        else:
            ap_sep_vec = jnp.zeros(0)

        # Intra-arc ML separation: smooth_abs(ml_diff) − min_ml_sep,
        # but only over same-arc pairs (others get +LARGE so the solver ignores).
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

        return (
            thread_vec,
            clear_pp_vec,
            clear_pf_vec,
            brain_vec,
            ap_sep_vec,
            ml_sep_vec,
        )

    def _all_slacks(*args, **kwargs):
        # Flattening each group reproduces the constraint order scipy/IPOPT see.
        return jnp.concatenate(
            [part.reshape(-1) for part in _slack_parts(*args, **kwargs)]
        )

    # Exact second-order terms, two flavours. DENSE = full n×n matrix (~n
    # grad-evals/iter, ~44x slower per cand). HVP = Hessian-VECTOR product
    # (forward-over-reverse, ~2x a gradient) — trust-constr/IPOPT use HVP-based
    # CG internally, so they never need the dense matrix.
    _grad_obj = jax.grad(_objective)

    def _obj_hessp(x, p, **packed):
        # ∇²f · p
        return jax.jvp(lambda xx: _grad_obj(xx, **packed), (x,), (p,))[1]

    def _slacks_hess(x, v, **packed):
        # dense Σ_i v_i ∇²g_i = ∇²(v·slacks)  (the constraint Lagrangian term)
        return jax.hessian(lambda xx: jnp.vdot(_all_slacks(xx, **packed), v))(x)

    def _slacks_hessp(x, v, p, **packed):
        # (Σ_i v_i ∇²g_i) · p  via HVP of the scalar (v·slacks)
        gv = jax.grad(lambda xx: jnp.vdot(_all_slacks(xx, **packed), v))
        return jax.jvp(gv, (x,), (p,))[1]

    return dict(
        obj=jax.jit(_objective),
        obj_grad=jax.jit(jax.grad(_objective)),
        obj_hess=jax.jit(jax.hessian(_objective)),
        obj_hessp=jax.jit(_obj_hessp),
        slacks=jax.jit(_all_slacks),
        slack_parts=jax.jit(_slack_parts),
        slack_labels={
            "probe_pairs": [tuple(pair) for pair in sdf_pair_list],
            "pair_categories": list(PairClearance._fields),
            "pair_gains": list(pair_gains),
            "fixtures": [fx.name for fx in fixtures],
            "fixture_probes": [
                i for i in range(n_probes) if (not has_sdf) or sdf_shapes[i] is not None
            ],
            "fixture_categories": list(FixtureClearance._fields),
            "fixture_gains": list(fixture_gains),
        },
        slacks_jac=jax.jit(jax.jacfwd(_all_slacks)),
        slacks_hess=jax.jit(_slacks_hess),
        slacks_hessp=jax.jit(_slacks_hessp),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cache_stats() -> dict:
    return {**_CACHE_STATS, "entries": len(_JIT_CACHE)}


def clear_jit_cache() -> None:
    """Drop every compiled kernel and the hit counters.

    The cache key carries the shapes of the fixture, brain and probe grids but
    not their values, and coverage data is not in it at all, so an entry is only
    valid for the inputs that built it. A caller that changes subject or well
    mode must clear the cache, or it gets the previous subject's kernel with the
    previous subject's geometry compiled in.
    """
    _JIT_CACHE.clear()
    _CACHE_STATS.update(hits=0, misses=0)


def _padding_mask(
    packed: dict, labels: dict, *, n_arcs: int, has_brain: bool
) -> NDArray:
    """Which rows of the slack vector constrain something, in ``SLACK_GROUPS``
    order.

    Uniform compiled shapes across candidates need per-probe padding, and those
    rows reach the solver as constants with no gradient. They do not make the KKT
    matrix singular — each inequality carries its own slack, so such a row reads
    ``[0 … 0 | −1]`` — but each still costs a Jacobian row, a slack and a barrier
    term in every factorization. The same masks that create the padding say which
    rows they are, so the mask is a property of the probe geometry alone. Reading
    it off slack *values* instead also drops live rows that merely sit at the
    out-of-grid sentinel, which share the padding value but do carry gradient
    once a pose brings them near a fixture.
    """
    section = np.asarray(packed["section_mask"]) > 0  # (probes, sections)
    shank = np.asarray(packed["shank_mask"]) > 0  # (probes, shanks)
    same_arc = np.asarray(packed["same_arc_mask"]) > 0  # (probes, probes)
    iu, ju = np.triu_indices(shank.shape[0], k=1)
    by_group = {
        "thread": (section[:, :, None] & shank[:, None, :]).reshape(-1),
        "probe_pair": np.ones(
            len(labels["probe_pairs"]) * len(labels["pair_categories"]), bool
        ),
        "probe_fixture": np.ones(
            len(labels["fixtures"])
            * len(labels["fixture_probes"])
            * len(labels["fixture_categories"]),
            bool,
        ),
        "brain": shank.reshape(-1) if has_brain else np.zeros(0, bool),
        "arc_sep": np.ones(n_arcs * (n_arcs - 1) // 2, bool),
        "ml_sep": same_arc[iu, ju],
    }
    return np.concatenate([by_group[name] for name in SLACK_GROUPS])


def _live_row_constraints(slacks_fn, slacks_jac, live_rows: NDArray, drop_padded: bool):
    """The constraint callables the solver sees.

    Masking happens outside the traced function, so the compiled slack vector is
    unchanged and only IPOPT's view of it shrinks.
    """
    if not drop_padded or live_rows.all():
        return slacks_fn, slacks_jac

    def live_fn(x: NDArray) -> NDArray:
        return slacks_fn(x)[live_rows]

    def live_jac(x: NDArray) -> NDArray:
        return slacks_jac(x)[live_rows]

    return live_fn, live_jac


def make_phase2(
    statics,
    n_arcs: int,
    coverage_data: tuple[CoverageData, ...] | None = None,
    fixtures: tuple[FixtureSDFData, ...] = (),
    weights: Phase2Weights = Phase2Weights(),
    *,
    coverage_n_samples: int = 41,
    brain_sdf: "BrainSDFData | None" = None,
    coverage_ceilings: "tuple[float, ...] | None" = None,
    coverage_weights: "tuple[float, ...] | None" = None,
    grid_dtype=jnp.bfloat16,
    hessian: str = "none",
    drop_padded_rows: bool = False,
) -> Phase2Problem:
    """Build Phase 2 scipy callables.

    Returns a dict with:

      - ``fun(x) → scalar``: objective (-coverage + bounds + margin bonus)
      - ``jac(x) → (n_vars,)``: objective gradient
      - ``constraints``: list of one scipy ``ineq`` dict over the
        concatenated slack vector
      - ``n_constraints``: total slack count (for diagnostics)
      - ``slack_parts(x)``: the slack vector split into ``SLACK_GROUPS``, each in
        its natural shape (padded entries at ``PADDED_SLACK``)
      - ``slack_labels``: probe pairs, fixtures, categories and gains that index
        the ``probe_pair`` / ``probe_fixture`` groups
    """
    # Ceilings/weights are baked into the trace as constants, so they must be in
    # the cache key (like the Phase-1 builder).
    ceil_key = (
        tuple(round(float(c), 6) for c in coverage_ceilings)
        if coverage_ceilings is not None
        else None
    )
    wcov_key = (
        tuple(round(float(w), 6) for w in coverage_weights)
        if coverage_weights is not None
        else None
    )
    # bf16 collision-grid storage (probe + every fixture); see clearance_sweep
    # for the policy. Cast fixtures BEFORE _build_jit closure-captures them; the
    # dtype is in the cache key for the same reason.
    fixtures = cast_fixture_grids(fixtures, grid_dtype)
    dtype_key = jnp.dtype(grid_dtype).name
    # The shape signature is what _build_jit unpacks; the rest of the key covers
    # the other values it bakes into the trace. Kept as two names rather than a
    # slice, so adding a component to the key cannot silently shift the unpack.
    shape_sig = _signature(statics, n_arcs, weights, fixtures, brain_sdf)
    sig = shape_sig + (ceil_key, wcov_key, dtype_key, int(coverage_n_samples))
    if sig not in _JIT_CACHE:
        _JIT_CACHE[sig] = _build_jit(
            shape_sig,
            weights,
            coverage_data,
            fixtures,
            coverage_n_samples,
            brain_sdf=brain_sdf,
            coverage_ceilings=coverage_ceilings,
            coverage_weights=coverage_weights,
        )
        _CACHE_STATS["misses"] += 1
    else:
        _CACHE_STATS["hits"] += 1
    jit = _JIT_CACHE[sig]
    packed = cast_packed_grids(_pack_statics(statics, n_arcs), grid_dtype)

    def fun(x: NDArray) -> float:
        return float(jit["obj"](jnp.asarray(x, dtype=jnp.float32), **packed))

    def jac(x: NDArray) -> NDArray:
        g = jit["obj_grad"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(g, dtype=np.float64)

    def slacks_fn(x: NDArray) -> NDArray:
        s = jit["slacks"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(s, dtype=np.float64)

    def slack_parts(x: NDArray) -> dict[str, NDArray]:
        parts = jit["slack_parts"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return {
            name: np.asarray(part, dtype=np.float64)
            for name, part in zip(SLACK_GROUPS, parts)
        }

    def slacks_jac(x: NDArray) -> NDArray:
        J = jit["slacks_jac"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(J, dtype=np.float64)

    def obj_hess(x: NDArray) -> NDArray:  # dense ∇²f
        H = jit["obj_hess"](jnp.asarray(x, dtype=jnp.float32), **packed)
        return np.asarray(H, dtype=np.float64)

    def obj_hessp(x: NDArray, p: NDArray) -> NDArray:  # ∇²f · p (HVP)
        hp = jit["obj_hessp"](
            jnp.asarray(x, dtype=jnp.float32),
            jnp.asarray(p, dtype=jnp.float32),
            **packed,
        )
        return np.asarray(hp, dtype=np.float64)

    def con_hess_dense(x: NDArray, v: NDArray) -> NDArray:
        H = jit["slacks_hess"](
            jnp.asarray(x, dtype=jnp.float32),
            jnp.asarray(v, dtype=jnp.float32),
            **packed,
        )
        return np.asarray(H, dtype=np.float64)

    def con_hess_linop(x: NDArray, v: NDArray):
        # NonlinearConstraint.hess may return a LinearOperator whose matvec is
        # the HVP, so trust-constr's CG never materializes the dense Σ vᵢ∇²gᵢ.
        from scipy.sparse.linalg import LinearOperator

        n = int(np.asarray(x).shape[0])
        xj = jnp.asarray(x, dtype=jnp.float32)
        vj = jnp.asarray(v, dtype=jnp.float32)

        def matvec(p):
            hp = jit["slacks_hessp"](
                xj, vj, jnp.asarray(p, dtype=jnp.float32), **packed
            )
            return np.asarray(hp, dtype=np.float64)

        return LinearOperator((n, n), matvec=matvec)

    from scipy.optimize import NonlinearConstraint

    # ``hessian`` mode: "none" (BFGS approx), "dense" (exact n×n — ~44x slower),
    # or "hessp" (exact Hessian-VECTOR products, ~2x a gradient — the affordable
    # exact second-order). Objective via hess/hessp; constraint via the matching
    # dense matrix / LinearOperator.
    # The exact-Hessian modes return per-row structures the mask wouldn't line up
    # with, so they keep every row.
    live_rows = _padding_mask(
        packed,
        jit["slack_labels"],
        n_arcs=n_arcs,
        has_brain=brain_sdf is not None,
    )
    con_fn, con_jac = _live_row_constraints(
        slacks_fn,
        slacks_jac,
        live_rows,
        drop_padded_rows and hessian == "none",
    )

    nlc_kw = dict(fun=con_fn, lb=0.0, ub=np.inf, jac=con_jac)
    hess_out = hessp_out = None
    if hessian == "dense":
        nlc_kw["hess"] = con_hess_dense
        hess_out = obj_hess
    elif hessian == "hessp":
        nlc_kw["hess"] = con_hess_linop
        hessp_out = obj_hessp
    return dict(
        fun=fun,
        jac=jac,
        hess=hess_out,
        hessp=hessp_out,
        constraints=[
            {
                "type": "ineq",
                "fun": con_fn,
                "jac": con_jac,
            }
        ],
        live_rows=live_rows,
        constraints_nlc=[NonlinearConstraint(**nlc_kw)],
        slack_parts=slack_parts,
        slack_labels=jit["slack_labels"],
    )
