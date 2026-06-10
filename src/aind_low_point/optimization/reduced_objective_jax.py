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
_JAX_CACHE_DIR = os.environ.get("AIND_JAX_CACHE_DIR", "/tmp/aind_low_point_jax_cache")
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

from aind_low_point.optimization.sdf_jax import (  # noqa: E402
    PROBE_PAIR_SLACK_GAINS,
    dual_rep_pair_clearance,
    pose_from_optimizer_vars,
    smooth_abs,
    spin_deg_from_sxy,
    unit_circle_penalty,
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
    grid is sized to its probe bbox) and per-probe shank-OBB counts.
    """
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
    applies whichever mask-aware sum-of-squares or hard-constraint slack the
    caller needs."""
    tip_world = tips_local @ R.T + pose_tip
    shaft_dir = R @ jnp.array([0.0, 0.0, 1.0])
    line_d = shaft_length_mm * shaft_dir
    denom = s_axes @ line_d
    rel_centers = tip_world[None, :, :] - s_centers[:, None, :]
    num = jnp.einsum("skd,sd->sk", rel_centers, s_axes)
    # ``denom`` is the dot product of shaft direction with section axis.
    # For typical bounds (ap, ml ≤ 60°) it's in [0.25, 1.0]; the
    # ``shaft ⊥ section`` case (denom → 0) is geometrically unreachable
    # in our problem. The where here is a defensive guard for exact-zero
    # division; the gradient flows through the ``denom`` branch
    # everywhere ``|denom| ≥ 1e-12`` (i.e., always in practice).
    safe_denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    t = -num / safe_denom[:, None]
    pts = tip_world[None, :, :] + t[..., None] * line_d
    rel_to_center = pts - s_centers[:, None, :]
    u_w = jnp.einsum("skd,sd->sk", rel_to_center, s_e1)
    v_w = jnp.einsum("skd,sd->sk", rel_to_center, s_e2)
    u = s_cos[:, None] * u_w + s_sin[:, None] * v_w
    v = -s_sin[:, None] * u_w + s_cos[:, None] * v_w
    # No more ``where(parallel, inf, g)`` override (removed Patch B):
    # the ``+inf`` branch's gradient was NaN, corrupting optimizer gradients.
    # Guard the ellipse-axis divisions too: padded sections arrive with
    # ``s_a = s_b = 0`` from BatchedProbeStatic's zero-fill (the phase1
    # packer fills 1.0), so ``u / s_a`` is ``0/0 = NaN`` there — and the
    # caller's ``valid_g * excess²`` mask does NOT rescue it (``0 * NaN =
    # NaN``, value AND gradient). ``safe_{a,b}`` keep ``g`` finite for
    # padded/degenerate sections (masked out downstream); valid sections
    # have ``|s| ≫ 1e-12`` so the guard is an exact no-op for them.
    safe_a = jnp.where(jnp.abs(s_a) < 1e-12, 1.0, s_a)
    safe_b = jnp.where(jnp.abs(s_b) < 1e-12, 1.0, s_b)
    return (u / safe_a[:, None]) ** 2 + (v / safe_b[:, None]) ** 2 - 1.0


def _build_jit(signature: tuple, weights) -> tuple[Callable, Callable]:
    """Construct the JIT'd objective + grad for one signature."""
    (
        n_probes,
        n_arcs,
        max_shanks,
        max_sections,
        has_sdf,
        per_probe_sdf_shapes,
        per_probe_shank_counts,
        n_surf,
        _w_key,
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
    lambda_unit_circle = float(getattr(weights, "lambda_unit_circle", 100.0))
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
        R,
        pose_tip,
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
    ):
        """Mask-weighted threading penalty for one probe (scalar)."""
        g = threading_g_matrix(
            R,
            pose_tip,
            tips_local,
            s_axes,
            s_centers,
            s_e1,
            s_e2,
            s_cos,
            s_sin,
            s_a,
            s_b,
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
        # SDF: tuples-of-arrays so each probe can have its own grid shape.
        # When has_sdf=False these are empty tuples; the pair loop is also
        # empty so they aren't referenced.
        sdf_grids,
        sdf_origins,
        sdf_spacings,
        sdf_surfaces,
        shank_obb_centers,
        shank_obb_halves,
    ):
        arc_aps = y[:n_arcs]

        # Per-probe pose + threading penalty. y layout is
        # ``(arc_aps, (ml, sx, sy) × P)`` — spin parameterized as a 2D
        # unit-circle vector to avoid the ±180° wraparound discontinuity
        # from the scalar-angle layout.
        j_thread = jnp.float32(0.0)
        Rs = []
        ts = []
        for i in range(n_probes):
            off = n_arcs + 3 * i
            ml = y[off]
            sx = y[off + 1]
            sy = y[off + 2]
            spin = spin_deg_from_sxy(sx, sy)
            ap = arc_aps[arc_idx[i]]
            R, t = pose_from_optimizer_vars(
                target_LPS=target_LPS[i],
                ap_deg=ap,
                ml_deg=ml,
                spin_deg=spin,
                offset_R_mm=jnp.float32(0.0),
                offset_A_mm=jnp.float32(0.0),
                past_target_mm=jnp.float32(0.0),
                recording_center_local=pivot_local[i],
            )
            Rs.append(R)
            ts.append(t)
            j_thread = j_thread + _threading_g_one(
                R,
                t,
                tips_local[i],
                shank_mask[i],
                s_axes[i],
                s_centers[i],
                s_e1[i],
                s_e2[i],
                s_cos[i],
                s_sin[i],
                s_a[i],
                s_b[i],
                section_mask[i],
            )

        # AP separation across arc pairs. ``smooth_abs`` keeps the
        # gradient continuous as ap_i, ap_j pass through equality (vs
        # jnp.abs which flips sign at zero).
        if arc_pairs.shape[0] > 0:
            ap_diffs = smooth_abs(arc_aps[arc_pairs[:, 0]] - arc_aps[arc_pairs[:, 1]])
            short_ap = jnp.maximum(0.0, min_arc_ap_sep - ap_diffs)
            j_arc_ap = jnp.sum(short_ap * short_ap)
        else:
            j_arc_ap = jnp.float32(0.0)

        # Intra-arc ML separation: pre-built mask of same-arc pairs (P, P).
        # y layout per probe is (ml, sx, sy) → ml at stride 3.
        ml_vals = y[n_arcs::3][:n_probes]  # broadcast-safe slice
        ml_diff = smooth_abs(ml_vals[:, None] - ml_vals[None, :])
        short_ml = jnp.maximum(0.0, min_intra_ml_sep - ml_diff)
        # Use upper triangle only (each pair once)
        j_ml = jnp.sum(same_arc_mask * short_ml * short_ml)

        # Soft bounds
        j_bounds = _softplus_squared(smooth_abs(arc_aps) - comfortable_ap)
        j_bounds = j_bounds + _softplus_squared(smooth_abs(ml_vals) - comfortable_ml)

        # Dual-rep clearance, three categories. Body-body is computed
        # ONCE across all P pairs via ``jax.vmap`` (single XLA kernel
        # launch) — the per-pair body-body lookups dominate the kernel-
        # launch overhead per the 2026-05-23 jax.profiler trace (~95
        # launches per obj call from the unrolled Python pair loop).
        # Body-shank + shank-shank keep the per-pair Python loop for
        # now: they need shank-count masking to vmap and are a smaller
        # share of cost.
        #
        # Pre-compute world-frame body surface samples once per probe so
        # neither path re-does ``surface @ R.T + t`` per pair (XLA CSE
        # leaves it duplicated across iterations — HLO dump 2026-05-23).
        world_surfaces = [
            sdf_surfaces[i] @ Rs[i].T + ts[i]
            if per_probe_sdf_shapes[i] is not None
            else None
            for i in range(n_probes)
        ]

        # All three dual-rep categories computed per-pair. CPU
        # vmap-across-pairs of any path that scatters gradient back to
        # per-probe state (R, t, world_surface) regresses on the
        # gradient pass — no HW atomics, no efficient parallel scatter
        # accumulation. See [[vmap-cpu-gpu-polish-arch]].
        # The per-pair helpers split body-body, body-shank-corners
        # (trilinear), and shank-only (analytic) so a future GPU mode
        # can vmap each independently.
        j_clear = jnp.float32(0.0)
        for ia, ib in sdf_pair_list:
            pc = dual_rep_pair_clearance(
                Rs[ia],
                ts[ia],
                Rs[ib],
                ts[ib],
                sdf_grids[ia],
                sdf_origins[ia],
                sdf_spacings[ia],
                sdf_grids[ib],
                sdf_origins[ib],
                sdf_spacings[ib],
                world_surfaces[ia],
                world_surfaces[ib],
                shank_obb_centers[ia],
                shank_obb_halves[ia],
                shank_obb_centers[ib],
                shank_obb_halves[ib],
            )
            softs = (
                pc.body_body[1],
                pc.body_shank_corners[1],
                pc.body_shank_obb[1],
                pc.shank_shank[1],
            )
            for d_soft, gain in zip(softs, PROBE_PAIR_SLACK_GAINS):
                short = jnp.maximum(0.0, min_clearance - d_soft) * gain
                j_clear = j_clear + short * short

        # Unit-circle pull on (sx, sy): magnitude is geometrically
        # free under ``spin_deg = atan2(sy, sx)`` but the optimiser
        # benefits from a consistent unit magnitude across stages.
        # Stride-3 slice of y after arc_aps gives the (ml, sx, sy)
        # per-probe block; sx at offset+1, sy at offset+2.
        sx_arr = y[n_arcs + 1 :: 3][:n_probes]
        sy_arr = y[n_arcs + 2 :: 3][:n_probes]
        j_unit_circle = unit_circle_penalty(sx_arr, sy_arr)

        return (
            lambda_thread * j_thread
            + lambda_arc_ap * j_arc_ap
            + lambda_ml * j_ml
            + lambda_bounds * j_bounds
            + lambda_clearance * j_clear
            + lambda_unit_circle * j_unit_circle
        )

    jit_obj = jax.jit(_objective)
    jit_grad = jax.jit(jax.grad(_objective))
    return jit_obj, jit_grad


def _pack_statics(
    statics,
    n_arcs: int,
    max_shanks: int,
    max_sections: int,
    has_sdf: bool,
    sdf_grid_shape,
    n_surf: int,
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
    # Shank OBBs follow the same per-probe-static-shape pattern.
    sdf_grids = []
    sdf_origins = []
    sdf_spacings = []
    sdf_surfaces = []
    shank_centers_tuple = []
    shank_halves_tuple = []
    for s in statics:
        if has_sdf and s.sdf_data is not None:
            sdf_grids.append(jnp.asarray(s.sdf_data["grid"], dtype=jnp.float32))
            sdf_origins.append(jnp.asarray(s.sdf_data["origin"], dtype=jnp.float32))
            sdf_spacings.append(jnp.asarray(s.sdf_data["spacing"], dtype=jnp.float32))
            sdf_surfaces.append(jnp.asarray(s.sdf_data["surface"], dtype=jnp.float32))
            shank_centers_tuple.append(
                jnp.asarray(
                    s.sdf_data.get("shank_centers", np.zeros((0, 3), dtype=np.float32)),
                    dtype=jnp.float32,
                )
            )
            shank_halves_tuple.append(
                jnp.asarray(
                    s.sdf_data.get("shank_halves", np.zeros((0, 3), dtype=np.float32)),
                    dtype=jnp.float32,
                )
            )
        else:
            sdf_grids.append(jnp.zeros((2, 2, 2), dtype=jnp.float32))
            sdf_origins.append(jnp.zeros(3, dtype=jnp.float32))
            sdf_spacings.append(jnp.float32(1.0))
            sdf_surfaces.append(jnp.zeros((1, 3), dtype=jnp.float32))
            shank_centers_tuple.append(jnp.zeros((0, 3), dtype=jnp.float32))
            shank_halves_tuple.append(jnp.zeros((0, 3), dtype=jnp.float32))
    out["sdf_grids"] = tuple(sdf_grids)
    out["sdf_origins"] = tuple(sdf_origins)
    out["sdf_spacings"] = tuple(sdf_spacings)
    out["sdf_surfaces"] = tuple(sdf_surfaces)
    out["shank_obb_centers"] = tuple(shank_centers_tuple)
    out["shank_obb_halves"] = tuple(shank_halves_tuple)
    return out


def make_jax_reduced_objective(
    statics,
    n_arcs: int,
    weights,
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
        statics,
        n_arcs,
        MAX_SHANKS_PAD,
        MAX_SECTIONS_PAD,
        has_sdf=sig[4],
        sdf_grid_shape=sig[5],
        n_surf=sig[7],
    )

    def fun(y: NDArray) -> float:
        return float(jit_obj(jnp.asarray(y, dtype=jnp.float32), **packed))

    def jac(y: NDArray) -> NDArray:
        g = jit_grad(jnp.asarray(y, dtype=jnp.float32), **packed)
        return np.asarray(g, dtype=np.float64)

    return fun, jac
