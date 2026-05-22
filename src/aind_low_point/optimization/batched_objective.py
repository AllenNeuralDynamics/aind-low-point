"""Batched reduced objective for Stage 2 polish.

Vmaps the existing per-candidate ``_reduced_objective`` over a batch of
``B`` candidates. Per-probe static fields (target, pivot, shank tips,
SDF grids) are closure-captured because they don't vary across the
batch — only ``y``, ``arc_idx``, and the assigned-hole section data do.

Designed alongside ``batched_static.py``. See
``dev/target_valid_atlas_design.md`` Phase 2.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.batched_static import BatchedProbeStatic
from aind_low_point.optimization.joint_rerank import JointWeights
from aind_low_point.optimization.joint_rerank_jax import threading_g_matrix
from aind_low_point.optimization.sdf_jax import (
    pairwise_signed_clearance,
    pose_from_optimizer_vars,
    spin_deg_from_sxy,
)


def _softplus_squared(values: jnp.ndarray) -> jnp.ndarray:
    """smooth(relu(x))^2 — penalty for positive ``values``."""
    sp = jnp.log1p(jnp.exp(-jnp.abs(values))) + jnp.maximum(values, 0.0)
    return jnp.sum(sp * sp)


def _threading_g_for_probe(
    R: jnp.ndarray,
    pose_tip: jnp.ndarray,
    tips_local: jnp.ndarray,        # (SH, 3)
    shank_mask: jnp.ndarray,        # (SH,)
    s_axes: jnp.ndarray,            # (S, 3)
    s_centers: jnp.ndarray,         # (S, 3)
    s_e1: jnp.ndarray,              # (S, 3)
    s_e2: jnp.ndarray,              # (S, 3)
    s_cos: jnp.ndarray,             # (S,)
    s_sin: jnp.ndarray,             # (S,)
    s_a: jnp.ndarray,               # (S,)
    s_b: jnp.ndarray,               # (S,)
    section_mask: jnp.ndarray,      # (S,)
    threading_tol: float,
) -> jnp.ndarray:
    """Scalar threading penalty for one probe.

    Uses the existing ``threading_g_matrix`` which returns ``g - 1`` so
    the threshold is at zero (boundary of the ellipse). Padded entries
    contribute zero via masks.
    """
    g = threading_g_matrix(
        R, pose_tip, tips_local,
        s_axes, s_centers, s_e1, s_e2,
        s_cos, s_sin, s_a, s_b,
    )  # shape (S, SH); g <= 0 ⇒ inside oval
    valid = section_mask[:, None] * shank_mask[None, :]  # (S, SH)
    excess = jnp.maximum(0.0, g - threading_tol)
    excess = jnp.where(jnp.isinf(g), 0.0, excess)
    return jnp.sum(valid * excess * excess)


def make_batched_reduced_objective(
    static: BatchedProbeStatic,
    weights: JointWeights,
) -> tuple[Callable, Callable]:
    """Build the batched (vmapped over candidates) reduced objective and
    gradient.

    Captures per-probe constants from ``static`` (target, pivot, shank
    tips, SDF table) into the closure. Returns:

    - ``obj_fn(y, varying) -> (B,)`` where ``y: (B, n_vars)`` and
      ``varying`` is a tuple of the per-candidate batched arrays
      (arc_idx, section fields).
    - ``grad_fn(y, varying) -> (B, n_vars)`` — vmapped grad w.r.t. y.

    Calling convention chosen so that ``static`` (a Python dataclass)
    can be closure-captured but per-candidate arrays are passed in by
    the optimizer step so it can update them between steps.

    The function signature also accepts ``static`` as a dummy to keep
    the closure light — only the constant arrays are baked.
    """
    K = static.K
    n_arcs = static.n_arcs
    SH = static.SH
    S = static.S

    # Closure-captured per-probe constants (same across batch)
    target_LPS = static.probe_target_lps[0]        # (K, 3)
    pivot_local = static.probe_pivot_local[0]      # (K, 3)
    shank_tips = static.probe_shank_tips[0]        # (K, SH, 3)
    shank_mask = static.probe_shank_mask[0].astype(jnp.float32)  # (K, SH)

    # SDF kind table (constant). sdf_kind_id is per (B, K), but per-probe
    # kind doesn't change across candidates in this problem — capture
    # per-probe kind ids as constants.
    sdf_kind_id = np.asarray(static.sdf_kind_id[0])  # (K,) int
    has_sdf_per_probe = sdf_kind_id >= 0
    sdf_pair_list: list[tuple[int, int]] = []
    for i in range(K):
        if not has_sdf_per_probe[i]:
            continue
        for j in range(i + 1, K):
            if not has_sdf_per_probe[j]:
                continue
            sdf_pair_list.append((i, j))

    sdf_grids = static.sdf_grids                    # (N_kinds, GX, GY, GZ)
    sdf_origins = static.sdf_origins                # (N_kinds, 3)
    sdf_spacings = static.sdf_spacings              # (N_kinds,)
    sdf_surface_points = static.sdf_surface_points  # (N_kinds, N_surf, 3)

    # Pre-compute arc-pair index list (for AP separation)
    arc_pairs = jnp.asarray(
        [(a, b) for a in range(n_arcs) for b in range(a + 1, n_arcs)],
        dtype=jnp.int32,
    ).reshape(-1, 2)

    # Weight scalars (Python floats — baked into trace)
    lambda_thread = float(weights.lambda_thread)
    lambda_arc_ap = float(weights.lambda_arc_ap)
    lambda_ml = float(weights.lambda_ml)
    lambda_bounds = float(weights.lambda_bounds)
    lambda_clearance = float(weights.lambda_clearance)
    min_arc_ap_sep = float(weights.min_arc_ap_sep_deg)
    min_intra_ml_sep = float(weights.min_intra_arc_ml_sep_deg)
    comfortable_ap = float(weights.comfortable_ap_deg)
    comfortable_ml = float(weights.comfortable_ml_deg)
    threading_tol = float(weights.threading_oval_tolerance)
    min_clearance = float(weights.min_clearance_mm)

    def _obj_one(
        y: jnp.ndarray,                # (n_arcs + 2*K,)
        arc_idx: jnp.ndarray,          # (K,) int
        s_axes: jnp.ndarray,           # (K, S, 3)
        s_centers: jnp.ndarray,        # (K, S, 3)
        s_e1: jnp.ndarray,             # (K, S, 3)
        s_e2: jnp.ndarray,             # (K, S, 3)
        s_cos: jnp.ndarray,            # (K, S)
        s_sin: jnp.ndarray,            # (K, S)
        s_a: jnp.ndarray,              # (K, S)
        s_b: jnp.ndarray,              # (K, S)
        section_mask: jnp.ndarray,     # (K, S)
    ) -> jnp.ndarray:
        arc_aps = y[:n_arcs]

        # Per-probe pose + threading penalty (unrolled Python loop —
        # K is small (7); unrolling keeps the JIT trace flat). y layout
        # per probe is (ml, sx, sy) under Patch B's (sx, sy)
        # reparameterization.
        j_thread = jnp.float32(0.0)
        Rs: list[jnp.ndarray] = []
        ts: list[jnp.ndarray] = []
        for i in range(K):
            off = n_arcs + 3 * i
            ml = y[off]
            sx = y[off + 1]
            sy = y[off + 2]
            spin = spin_deg_from_sxy(sx, sy)
            ap = arc_aps[arc_idx[i]]
            R, t = pose_from_optimizer_vars(
                target_LPS=target_LPS[i],
                ap_deg=ap, ml_deg=ml, spin_deg=spin,
                offset_R_mm=jnp.float32(0.0),
                offset_A_mm=jnp.float32(0.0),
                past_target_mm=jnp.float32(0.0),
                recording_center_local=pivot_local[i],
            )
            Rs.append(R)
            ts.append(t)
            j_thread = j_thread + _threading_g_for_probe(
                R, t,
                shank_tips[i], shank_mask[i],
                s_axes[i], s_centers[i], s_e1[i], s_e2[i],
                s_cos[i], s_sin[i], s_a[i], s_b[i], section_mask[i],
                threading_tol,
            )

        # AP separation
        if arc_pairs.shape[0] > 0:
            ap_diffs = jnp.abs(arc_aps[arc_pairs[:, 0]] - arc_aps[arc_pairs[:, 1]])
            short_ap = jnp.maximum(0.0, min_arc_ap_sep - ap_diffs)
            j_arc_ap = jnp.sum(short_ap * short_ap)
        else:
            j_arc_ap = jnp.float32(0.0)

        # Intra-arc ML separation. same_arc_mask is (K, K) upper triangle
        # where probes i, j share an arc. Compute inline since arc_idx
        # varies per candidate. y per probe is (ml, sx, sy) → ml at
        # stride 3.
        ml_vals = y[n_arcs::3][:K]
        ml_diff = jnp.abs(ml_vals[:, None] - ml_vals[None, :])
        same = (arc_idx[:, None] == arc_idx[None, :])
        upper = jnp.triu(jnp.ones((K, K), dtype=jnp.float32), k=1)
        same_arc_mask = same.astype(jnp.float32) * upper
        short_ml = jnp.maximum(0.0, min_intra_ml_sep - ml_diff)
        j_ml = jnp.sum(same_arc_mask * short_ml * short_ml)

        # Soft bounds
        j_bounds = _softplus_squared(jnp.abs(arc_aps) - comfortable_ap)
        j_bounds = j_bounds + _softplus_squared(jnp.abs(ml_vals) - comfortable_ml)

        # SDF clearance over fixed pair list
        j_clear = jnp.float32(0.0)
        for ia, ib in sdf_pair_list:
            ka = int(sdf_kind_id[ia])
            kb = int(sdf_kind_id[ib])
            d = pairwise_signed_clearance(
                Rs[ia], ts[ia], Rs[ib], ts[ib],
                sdf_grids[ka], sdf_origins[ka], sdf_spacings[ka],
                sdf_grids[kb], sdf_origins[kb], sdf_spacings[kb],
                sdf_surface_points[ka], sdf_surface_points[kb],
            )
            short = jnp.maximum(0.0, min_clearance - d)
            j_clear = j_clear + short * short

        return (
            lambda_thread * j_thread
            + lambda_arc_ap * j_arc_ap
            + lambda_ml * j_ml
            + lambda_bounds * j_bounds
            + lambda_clearance * j_clear
        )

    # Vmap over batch axis 0; jit the array-only call site so BatchedProbeStatic
    # (a plain Python dataclass) doesn't need to flow through jit.
    _obj_batched_jit = jax.jit(
        jax.vmap(_obj_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    )

    _grad_one = jax.grad(_obj_one)
    _grad_batched_jit = jax.jit(
        jax.vmap(_grad_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    )

    def _arrays(bs: BatchedProbeStatic):
        return (
            bs.probe_arc_idx,
            bs.section_axes, bs.section_centers,
            bs.section_e1, bs.section_e2,
            bs.section_cos_theta, bs.section_sin_theta,
            bs.section_a, bs.section_b,
            bs.section_mask.astype(jnp.float32),
        )

    def obj_batched(y: jnp.ndarray, bs: BatchedProbeStatic) -> jnp.ndarray:
        return _obj_batched_jit(y, *_arrays(bs))

    def grad_batched(y: jnp.ndarray, bs: BatchedProbeStatic) -> jnp.ndarray:
        return _grad_batched_jit(y, *_arrays(bs))

    # Expose the underlying array-arg jit'd functions + the array
    # extractor so chunked callers (e.g. polish_all_with_batched_spin
    # _restore) can pass per-chunk bs arrays as JIT runtime args. With
    # bs flowing as runtime args, one JIT compile serves all same-
    # shape chunk calls — vs. the closure-capture path which bakes bs
    # into the trace and forces a full recompile per chunk.
    obj_batched.from_arrays = _obj_batched_jit  # type: ignore[attr-defined]
    grad_batched.from_arrays = _grad_batched_jit  # type: ignore[attr-defined]
    obj_batched.extract_arrays = _arrays  # type: ignore[attr-defined]

    return obj_batched, grad_batched
