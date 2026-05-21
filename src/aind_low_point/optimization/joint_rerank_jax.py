"""JAX-traceable rewrite of ``_reduced_objective`` with module-level
JIT cache so the compile cost is paid **once** per (probe-set, weight,
shape) signature — not per (H, A) candidate.

Each call to :func:`make_jax_reduced_objective` packs the per-candidate
varying static data (target_LPS, arc_idx, hole sections) into padded
``jnp`` arrays and dispatches into the cached JIT. The closure-captured
data (tips_local, pivot_local, per-probe SDF grids/surfaces) does NOT
appear in the trace at all — it's passed as runtime args, so the trace
identity is stable across all candidates with the same shape signature.

Padding: shanks pad to ``MAX_SHANKS = 8``, sections pad to
``MAX_SECTIONS = 32`` per probe. A boolean ``shank_mask`` /
``section_mask`` zeros out padding contributions in the loss. Probes
without SDF data drop out of the clearance pair list at pack time.

The cache key encodes ``(n_probes, n_arcs, has_sdf, sdf_grid_shape,
n_surface_points, weights)``; two candidates with the same probe set
and weights but different hole assignments share a cached XLA program.
"""

from __future__ import annotations

import os
from typing import Callable, Hashable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

# Persistent JAX compile cache. Each spawn-mode worker would otherwise
# repay the ~20s XLA compile cost (the in-memory ``_JIT_CACHE`` below is
# per-process). With the disk cache, the first worker to compile a given
# signature writes it; subsequent workers (and re-runs) load in <1s.
# Override path via the ``AIND_JAX_CACHE_DIR`` env var.
_JAX_CACHE_DIR = os.environ.get("AIND_JAX_CACHE_DIR",
                                 "/tmp/aind_low_point_jax_cache")
try:
    os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
except Exception:
    # Disk cache is a perf optimization, not a correctness requirement.
    # If JAX rejects the config (older version, etc.) we silently fall
    # back to per-process in-memory caching.
    pass

from aind_low_point.optimization.sdf_jax import (
    pairwise_signed_clearance,
    pose_from_optimizer_vars,
)

MAX_SHANKS_PAD = 4
MAX_SECTIONS_PAD = 8

_JIT_CACHE: dict[Hashable, tuple[Callable, Callable]] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def cache_stats() -> dict:
    """Return ``{hits, misses, entries}`` for diagnostic logging."""
    return {**_CACHE_STATS, "entries": len(_JIT_CACHE)}


def _weights_key(weights) -> tuple:
    """Hashable representation of the joint-reranker weights."""
    return (
        float(weights.lambda_thread),
        float(weights.lambda_arc_ap),
        float(weights.lambda_ml),
        float(weights.lambda_bounds),
        float(weights.lambda_clearance),
        float(weights.min_arc_ap_sep_deg),
        float(weights.min_intra_arc_ml_sep_deg),
        float(weights.comfortable_ap_deg),
        float(weights.comfortable_ml_deg),
        float(weights.threading_oval_tolerance),
        float(weights.min_clearance_mm),
    )


def _signature(statics, n_arcs: int, weights) -> tuple:
    """Cache key encoding everything the trace depends on, including the
    per-probe SDF grid shapes (which vary across probe kinds since each
    grid is sized to its probe bbox)."""
    has_sdf = any(s.sdf_data is not None for s in statics)
    per_probe_sdf_shapes: tuple = ()
    n_surf = 0
    if has_sdf:
        shapes = []
        for s in statics:
            if s.sdf_data is None:
                shapes.append(None)
            else:
                shapes.append(
                    tuple(int(x) for x in np.asarray(s.sdf_data["grid"]).shape)
                )
                if n_surf == 0:
                    n_surf = int(np.asarray(s.sdf_data["surface"]).shape[0])
        per_probe_sdf_shapes = tuple(shapes)
    return (
        len(statics),
        int(n_arcs),
        MAX_SHANKS_PAD,
        MAX_SECTIONS_PAD,
        has_sdf,
        per_probe_sdf_shapes,
        n_surf,
        _weights_key(weights),
    )


def _softplus_squared(values: jnp.ndarray) -> jnp.ndarray:
    sp = jnp.maximum(0.0, values) + jnp.log1p(jnp.exp(-jnp.abs(values)))
    return jnp.sum(sp * sp)


def threading_g_matrix(
    R: jnp.ndarray,
    pose_tip: jnp.ndarray,
    tips_local: jnp.ndarray,
    s_axes: jnp.ndarray,
    s_centers: jnp.ndarray,
    s_e1: jnp.ndarray,
    s_e2: jnp.ndarray,
    s_cos: jnp.ndarray,
    s_sin: jnp.ndarray,
    s_a: jnp.ndarray,
    s_b: jnp.ndarray,
    shaft_length_mm: float = 10.0,
) -> jnp.ndarray:
    """Raw (n_sections, n_shanks) oval g values for one probe pose.

    ``g <= tol`` is feasible. No masking, no thresholding — the caller
    applies whichever (mask-aware sum-of-squares for Stage 2, or
    tol-minus-g slack for Stage 3 SLSQP)."""
    tip_world = tips_local @ R.T + pose_tip
    shaft_dir = R @ jnp.array([0.0, 0.0, 1.0])
    line_d = shaft_length_mm * shaft_dir
    denom = s_axes @ line_d
    rel_centers = tip_world[None, :, :] - s_centers[:, None, :]
    num = jnp.einsum("skd,sd->sk", rel_centers, s_axes)
    safe_denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    t = -num / safe_denom[:, None]
    pts = tip_world[None, :, :] + t[..., None] * line_d
    rel_to_center = pts - s_centers[:, None, :]
    u_w = jnp.einsum("skd,sd->sk", rel_to_center, s_e1)
    v_w = jnp.einsum("skd,sd->sk", rel_to_center, s_e2)
    u = s_cos[:, None] * u_w + s_sin[:, None] * v_w
    v = -s_sin[:, None] * u_w + s_cos[:, None] * v_w
    g = (u / s_a[:, None]) ** 2 + (v / s_b[:, None]) ** 2 - 1.0
    # Parallel-to-section ⇒ +inf so the caller can treat it as
    # "always outside the oval" (a no-op for slack: -inf, harmless
    # for the penalty: huge positive squared, but the mask cancels).
    return jnp.where(jnp.abs(denom)[:, None] < 1e-12, jnp.inf, g)


def _build_jit(signature: tuple, weights) -> tuple[Callable, Callable]:
    """Construct the JIT'd objective + grad for one signature."""
    (
        n_probes, n_arcs, max_shanks, max_sections,
        has_sdf, per_probe_sdf_shapes, n_surf, _w_key,
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

    # Pre-compute arc-pair and ML-pair index lists (size depends only on
    # n_arcs and n_probes — same for all candidates with this signature).
    arc_pairs = jnp.asarray(
        [(a, b) for a in range(n_arcs) for b in range(a + 1, n_arcs)],
        dtype=jnp.int32,
    ).reshape(-1, 2)

    def _threading_g_one(
        R, pose_tip,
        tips_local, shank_mask,
        s_axes, s_centers, s_e1, s_e2,
        s_cos, s_sin, s_a, s_b, section_mask,
    ):
        """Mask-weighted threading penalty for one probe (scalar)."""
        g = threading_g_matrix(
            R, pose_tip, tips_local,
            s_axes, s_centers, s_e1, s_e2,
            s_cos, s_sin, s_a, s_b,
        )
        # Zero out padded entries (s_a/s_b=1, masks=0) — also handles
        # the ``+inf`` returned for shaft-parallel-to-section.
        valid = section_mask[:, None] * shank_mask[None, :]
        excess = jnp.maximum(0.0, g - threading_tol)
        # Replace inf with 0 where mask=0 so multiplying by 0 doesn't
        # produce NaN.
        excess = jnp.where(jnp.isinf(g), 0.0, excess)
        return jnp.sum(valid * excess * excess)

    def _objective(
        y,
        target_LPS, pivot_local, arc_idx,
        tips_local, shank_mask,
        s_axes, s_centers, s_e1, s_e2,
        s_cos, s_sin, s_a, s_b, section_mask,
        same_arc_mask,
        # SDF: tuples-of-arrays so each probe can have its own grid shape.
        # When has_sdf=False these are empty tuples; the pair loop is also
        # empty so they aren't referenced.
        sdf_grids, sdf_origins, sdf_spacings, sdf_surfaces,
    ):
        arc_aps = y[:n_arcs]

        # Per-probe pose + threading penalty
        j_thread = jnp.float32(0.0)
        Rs = []
        ts = []
        for i in range(n_probes):
            off = n_arcs + 2 * i
            ml = y[off]
            spin = y[off + 1]
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
            j_thread = j_thread + _threading_g_one(
                R, t,
                tips_local[i], shank_mask[i],
                s_axes[i], s_centers[i], s_e1[i], s_e2[i],
                s_cos[i], s_sin[i], s_a[i], s_b[i], section_mask[i],
            )

        # AP separation across arc pairs
        if arc_pairs.shape[0] > 0:
            ap_diffs = jnp.abs(arc_aps[arc_pairs[:, 0]] - arc_aps[arc_pairs[:, 1]])
            short_ap = jnp.maximum(0.0, min_arc_ap_sep - ap_diffs)
            j_arc_ap = jnp.sum(short_ap * short_ap)
        else:
            j_arc_ap = jnp.float32(0.0)

        # Intra-arc ML separation: pre-built mask of same-arc pairs (P, P)
        ml_vals = y[n_arcs::2][:n_probes]  # broadcast-safe slice
        ml_diff = jnp.abs(ml_vals[:, None] - ml_vals[None, :])
        short_ml = jnp.maximum(0.0, min_intra_ml_sep - ml_diff)
        # Use upper triangle only (each pair once)
        j_ml = jnp.sum(same_arc_mask * short_ml * short_ml)

        # Soft bounds
        j_bounds = _softplus_squared(jnp.abs(arc_aps) - comfortable_ap)
        j_bounds = j_bounds + _softplus_squared(jnp.abs(ml_vals) - comfortable_ml)

        # SDF clearance: unroll at trace time over the static pair list.
        # Per-probe grids have heterogeneous shapes (sized to each probe's
        # bbox), so vmap-over-stacked-grids isn't possible; the unrolled
        # Python loop bakes each grid's shape into the trace.
        j_clear = jnp.float32(0.0)
        for ia, ib in sdf_pair_list:
            d = pairwise_signed_clearance(
                Rs[ia], ts[ia], Rs[ib], ts[ib],
                sdf_grids[ia], sdf_origins[ia], sdf_spacings[ia],
                sdf_grids[ib], sdf_origins[ib], sdf_spacings[ib],
                sdf_surfaces[ia], sdf_surfaces[ib],
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

    jit_obj = jax.jit(_objective)
    jit_grad = jax.jit(jax.grad(_objective))
    return jit_obj, jit_grad


def _pack_statics(
    statics, n_arcs: int, max_shanks: int, max_sections: int,
    has_sdf: bool, sdf_grid_shape, n_surf: int,
) -> dict:
    """Pack per-candidate static data into padded jnp tensors."""
    P = len(statics)
    target_LPS = np.zeros((P, 3), dtype=np.float32)
    pivot_local = np.zeros((P, 3), dtype=np.float32)
    arc_idx = np.zeros(P, dtype=np.int32)
    tips_local = np.zeros((P, max_shanks, 3), dtype=np.float32)
    shank_mask = np.zeros((P, max_shanks), dtype=np.float32)
    s_axes = np.zeros((P, max_sections, 3), dtype=np.float32)
    # Initialise axis to a valid unit vector to keep numerics sane in padding.
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

    # Upper-triangular same-arc mask (excludes self-pairs)
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
    # Per-probe SDF data: keep as tuples-of-arrays since each probe's
    # grid is sized to its own bbox. The trace bakes the shapes in.
    sdf_grids = []
    sdf_origins = []
    sdf_spacings = []
    sdf_surfaces = []
    for s in statics:
        if has_sdf and s.sdf_data is not None:
            sdf_grids.append(jnp.asarray(s.sdf_data["grid"], dtype=jnp.float32))
            sdf_origins.append(jnp.asarray(s.sdf_data["origin"], dtype=jnp.float32))
            sdf_spacings.append(jnp.asarray(s.sdf_data["spacing"], dtype=jnp.float32))
            sdf_surfaces.append(jnp.asarray(s.sdf_data["surface"], dtype=jnp.float32))
        else:
            # Placeholder so positional indexing stays valid; never read.
            sdf_grids.append(jnp.zeros((2, 2, 2), dtype=jnp.float32))
            sdf_origins.append(jnp.zeros(3, dtype=jnp.float32))
            sdf_spacings.append(jnp.float32(1.0))
            sdf_surfaces.append(jnp.zeros((1, 3), dtype=jnp.float32))
    out["sdf_grids"] = tuple(sdf_grids)
    out["sdf_origins"] = tuple(sdf_origins)
    out["sdf_spacings"] = tuple(sdf_spacings)
    out["sdf_surfaces"] = tuple(sdf_surfaces)
    return out


def make_jax_reduced_objective(
    statics, n_arcs: int, weights,
) -> tuple[Callable[[NDArray], float], Callable[[NDArray], NDArray]]:
    """Build ``(fun, jac)`` scipy callables backed by the module-level
    JIT cache. Compile once per (probe-set, weights, shape) signature;
    reuse across all (H, A) candidates with the same signature."""
    sig = _signature(statics, n_arcs, weights)
    if sig not in _JIT_CACHE:
        _JIT_CACHE[sig] = _build_jit(sig, weights)
        _CACHE_STATS["misses"] += 1
    else:
        _CACHE_STATS["hits"] += 1
    jit_obj, jit_grad = _JIT_CACHE[sig]

    packed = _pack_statics(
        statics, n_arcs,
        MAX_SHANKS_PAD, MAX_SECTIONS_PAD,
        has_sdf=sig[4], sdf_grid_shape=sig[5], n_surf=sig[6],
    )

    def fun(y: NDArray) -> float:
        return float(jit_obj(jnp.asarray(y, dtype=jnp.float32), **packed))

    def jac(y: NDArray) -> NDArray:
        g = jit_grad(jnp.asarray(y, dtype=jnp.float32), **packed)
        return np.asarray(g, dtype=np.float64)

    return fun, jac
