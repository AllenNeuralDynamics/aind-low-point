"""JAX constraint vectors + analytic Jacobians for Stage 3's full-x
SLSQP.

Stage 3 minimises ``-coverage_total`` subject to four inequality groups:

  - ``threading``     : per (probe, shank, section) oval-inside slack
  - ``clearance``     : per pair signed clearance (already in SDF JAX,
                        wired via :mod:`sdf_clearance` — not handled here)
  - ``arc_ap_separation``       : ``|ap_i - ap_j| - min_arc_ap_sep``
  - ``intra_arc_ml_separation`` : ``|ml_i - ml_j| - min_intra_ml_sep``
                                  for probes sharing an arc

Before this port the three non-clearance groups were evaluated with
NumPy and SciPy SLSQP used finite-differences for their Jacobians.
With 17 vars × 4 groups × 2 evals/var that's ~136 calls into
``evaluate_constraints`` per SLSQP iteration — and each call also runs
coverage MC sampling. Replacing each constraint's FD with an analytic
``jax.jacrev`` collapses those calls into one traced forward + reverse
sweep per call, and the trace is JIT-cached per (probe-set, hole
assignments, weights) signature.

Full-x layout: ``x = [arc_ap_0, …, arc_ap_{A-1},
(ml, spin, off_R, off_A, depth)_0, …, (ml, spin, off_R, off_A, depth)_{K-1}]``.

This is the inner-solve full-vector form used by ``_inner_solve_one``
and ``polish_seed``; it's distinct from Stage 2's reduced ``y``
(no off_R/off_A/depth there), so we don't reuse Stage 2's pack/JIT.
"""

from __future__ import annotations

from typing import Callable, Hashable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.joint_rerank_jax import (
    MAX_SECTIONS_PAD,
    MAX_SHANKS_PAD,
    threading_g_matrix,
)
from aind_low_point.optimization.sdf_jax import pose_from_optimizer_vars

_JIT_CACHE: dict[Hashable, dict[str, tuple[Callable, Callable]]] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def cache_stats() -> dict:
    return {**_CACHE_STATS, "entries": len(_JIT_CACHE)}


def _signature(ctx) -> tuple:
    """Cache key — shapes + key thresholds. Per-probe section/shank
    counts are folded into masks at pack time so the JIT signature
    only depends on pad sizes."""
    arc_ids = tuple(ctx.layout.arc_ids)
    probe_arcs = tuple(
        ctx.layout.arc_ids.index(p.arc_id) for p in ctx.probes
    )
    return (
        len(ctx.probes),
        len(arc_ids),
        MAX_SHANKS_PAD,
        MAX_SECTIONS_PAD,
        probe_arcs,
        float(ctx.threading_oval_tolerance),
        float(ctx.min_arc_ap_sep_deg),
        float(ctx.min_within_arc_ml_sep_deg),
        float(ctx.shaft_length_mm),
    )


def _build_jit(sig: tuple) -> dict[str, tuple[Callable, Callable]]:
    """Build the per-group JIT'd ``(fun, jac)`` callables for one
    signature.

    ``x`` layout (length ``n_arcs + 5 * n_probes``):
        ``x[:n_arcs]``  arc AP centroids (deg)
        ``x[n_arcs + 5i + 0]`` ml_i
        ``x[n_arcs + 5i + 1]`` spin_i
        ``x[n_arcs + 5i + 2]`` off_R_i
        ``x[n_arcs + 5i + 3]`` off_A_i
        ``x[n_arcs + 5i + 4]`` depth_i (past_target_mm)
    """
    (
        n_probes, n_arcs, max_shanks, max_sections,
        probe_arcs, threading_tol, min_arc_ap_sep, min_intra_ml_sep,
        shaft_length_mm,
    ) = sig
    probe_arc_idx_static = np.asarray(probe_arcs, dtype=np.int32)

    arc_pairs = jnp.asarray(
        [(a, b) for a in range(n_arcs) for b in range(a + 1, n_arcs)],
        dtype=jnp.int32,
    ).reshape(-1, 2)

    # Same-arc probe pair list (Python-static, baked into trace)
    ml_pair_idx: list[tuple[int, int]] = []
    for i in range(n_probes):
        for j in range(i + 1, n_probes):
            if probe_arc_idx_static[i] == probe_arc_idx_static[j]:
                ml_pair_idx.append((i, j))
    ml_pair_arr = jnp.asarray(ml_pair_idx, dtype=jnp.int32).reshape(-1, 2)

    def _poses_per_probe(x, target_LPS, pivot_local):
        """Compute (R, pose_tip) for every probe at full-x ``x``."""
        arc_aps = x[:n_arcs]
        Rs = []
        ts = []
        for i in range(n_probes):
            off = n_arcs + 5 * i
            ml = x[off + 0]
            spin = x[off + 1]
            off_R = x[off + 2]
            off_A = x[off + 3]
            depth = x[off + 4]
            ap = arc_aps[probe_arc_idx_static[i]]
            R, t = pose_from_optimizer_vars(
                target_LPS=target_LPS[i],
                ap_deg=ap, ml_deg=ml, spin_deg=spin,
                offset_R_mm=off_R, offset_A_mm=off_A,
                past_target_mm=depth,
                recording_center_local=pivot_local[i],
            )
            Rs.append(R)
            ts.append(t)
        return Rs, ts

    def _threading_slack(
        x,
        target_LPS, pivot_local,
        tips_local, shank_mask,
        s_axes, s_centers, s_e1, s_e2,
        s_cos, s_sin, s_a, s_b, section_mask,
    ):
        """Per-(probe, section, shank) slack: ``tol - g``. ``>= 0`` ⇒
        inside the oval (feasible). Padded entries get a fixed
        large-positive slack so SLSQP ignores them."""
        Rs, ts = _poses_per_probe(x, target_LPS, pivot_local)
        out = []
        for i in range(n_probes):
            g = threading_g_matrix(
                Rs[i], ts[i],
                tips_local[i],
                s_axes[i], s_centers[i], s_e1[i], s_e2[i],
                s_cos[i], s_sin[i], s_a[i], s_b[i],
                shaft_length_mm=shaft_length_mm,
            )
            # Convert g to slack (tol - g >= 0 is feasible). Padded
            # entries get +1e6 (always feasible from SLSQP's view) so
            # they don't constrain. ``threading_g_matrix`` returns inf
            # for shaft-parallel-to-section — treat that as +1e6 too.
            slack = threading_tol - g
            mask = section_mask[i][:, None] * shank_mask[i][None, :]
            slack = jnp.where(jnp.isinf(g), 1e6, slack)
            slack = jnp.where(mask > 0.5, slack, 1e6)
            out.append(slack.reshape(-1))
        return jnp.concatenate(out)

    def _arc_ap_sep_slack(x):
        arc_aps = x[:n_arcs]
        if arc_pairs.shape[0] == 0:
            return jnp.zeros(0, dtype=x.dtype)
        diffs = jnp.abs(arc_aps[arc_pairs[:, 0]] - arc_aps[arc_pairs[:, 1]])
        return diffs - min_arc_ap_sep

    def _intra_ml_sep_slack(x):
        if ml_pair_arr.shape[0] == 0:
            return jnp.zeros(0, dtype=x.dtype)
        # Vectorised gather of ml values via strided indexing.
        ml_idx = n_arcs + 5 * jnp.arange(n_probes)
        mls = x[ml_idx]
        diffs = jnp.abs(mls[ml_pair_arr[:, 0]] - mls[ml_pair_arr[:, 1]])
        return diffs - min_intra_ml_sep

    return {
        "threading": (jax.jit(_threading_slack), jax.jit(jax.jacrev(_threading_slack))),
        "arc_ap_separation": (jax.jit(_arc_ap_sep_slack), jax.jit(jax.jacrev(_arc_ap_sep_slack))),
        "intra_arc_ml_separation": (
            jax.jit(_intra_ml_sep_slack),
            jax.jit(jax.jacrev(_intra_ml_sep_slack)),
        ),
    }


def _pack_static(ctx) -> dict:
    """Pack per-probe static arrays into padded jnp tensors. Same
    layout/padding convention as Stage 2's :func:`_pack_statics`."""
    P = len(ctx.probes)
    target_LPS = np.zeros((P, 3), dtype=np.float32)
    pivot_local = np.zeros((P, 3), dtype=np.float32)
    tips_local = np.zeros((P, MAX_SHANKS_PAD, 3), dtype=np.float32)
    shank_mask = np.zeros((P, MAX_SHANKS_PAD), dtype=np.float32)
    s_axes = np.zeros((P, MAX_SECTIONS_PAD, 3), dtype=np.float32)
    s_axes[:, :, 2] = 1.0
    s_centers = np.zeros((P, MAX_SECTIONS_PAD, 3), dtype=np.float32)
    s_e1 = np.zeros((P, MAX_SECTIONS_PAD, 3), dtype=np.float32)
    s_e1[:, :, 0] = 1.0
    s_e2 = np.zeros((P, MAX_SECTIONS_PAD, 3), dtype=np.float32)
    s_e2[:, :, 1] = 1.0
    s_cos = np.ones((P, MAX_SECTIONS_PAD), dtype=np.float32)
    s_sin = np.zeros((P, MAX_SECTIONS_PAD), dtype=np.float32)
    s_a = np.ones((P, MAX_SECTIONS_PAD), dtype=np.float32)
    s_b = np.ones((P, MAX_SECTIONS_PAD), dtype=np.float32)
    section_mask = np.zeros((P, MAX_SECTIONS_PAD), dtype=np.float32)

    from aind_low_point.optimization.geometry import cap_basis

    for i, probe in enumerate(ctx.probes):
        target_LPS[i] = np.asarray(probe.target_LPS, dtype=np.float32)
        tips = np.asarray(probe.shank_tips_local, dtype=np.float32)
        if tips.shape[0] > 0:
            pivot_local[i] = np.array(
                [
                    float(tips[:, 0].mean()),
                    float(tips[:, 1].mean()),
                    float(probe.recording_geom.active_center_mm),
                ],
                dtype=np.float32,
            )
            ns = min(int(tips.shape[0]), MAX_SHANKS_PAD)
            tips_local[i, :ns] = tips[:ns]
            shank_mask[i, :ns] = 1.0
        sections = probe.assigned_hole.sections
        nsec = min(len(sections), MAX_SECTIONS_PAD)
        for k in range(nsec):
            sec = sections[k]
            ax = np.asarray(sec.axis, dtype=np.float32)
            s_axes[i, k] = ax
            e1, e2 = cap_basis(ax.astype(np.float64))
            s_e1[i, k] = e1.astype(np.float32)
            s_e2[i, k] = e2.astype(np.float32)
            s_centers[i, k] = np.asarray(sec.center, dtype=np.float32)
            s_cos[i, k] = float(np.cos(sec.theta))
            s_sin[i, k] = float(np.sin(sec.theta))
            s_a[i, k] = float(sec.a)
            s_b[i, k] = float(sec.b)
            section_mask[i, k] = 1.0

    return dict(
        target_LPS=jnp.asarray(target_LPS),
        pivot_local=jnp.asarray(pivot_local),
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
    )


def make_stage3_constraints(ctx) -> dict[str, dict]:
    """Return ``{group_name: {'fun': callable, 'jac': callable}}``
    where each ``fun(x) -> ndarray`` and ``jac(x) -> ndarray`` are
    scipy-SLSQP-compatible.

    Compile is cached per (probe-set + hole assignment + threshold)
    signature, so the second (and later) Stage 3 polish on the same
    context pays only the runtime args copy."""
    sig = _signature(ctx)
    if sig not in _JIT_CACHE:
        _JIT_CACHE[sig] = _build_jit(sig)
        _CACHE_STATS["misses"] += 1
    else:
        _CACHE_STATS["hits"] += 1
    jits = _JIT_CACHE[sig]
    packed = _pack_static(ctx)
    threading_args = (
        packed["target_LPS"], packed["pivot_local"],
        packed["tips_local"], packed["shank_mask"],
        packed["s_axes"], packed["s_centers"],
        packed["s_e1"], packed["s_e2"],
        packed["s_cos"], packed["s_sin"],
        packed["s_a"], packed["s_b"], packed["section_mask"],
    )

    th_fn, th_jac = jits["threading"]
    ap_fn, ap_jac = jits["arc_ap_separation"]
    ml_fn, ml_jac = jits["intra_arc_ml_separation"]

    def threading_fun(x: NDArray) -> NDArray:
        arr = th_fn(jnp.asarray(x, dtype=jnp.float32), *threading_args)
        out = np.asarray(arr, dtype=np.float64)
        # SciPy SLSQP can't handle empty arrays; substitute a single
        # always-feasible scalar.
        return out if out.size > 0 else np.array([1.0])

    def threading_jac_fn(x: NDArray) -> NDArray:
        jacm = th_jac(jnp.asarray(x, dtype=jnp.float32), *threading_args)
        out = np.asarray(jacm, dtype=np.float64)
        if out.size == 0:
            return np.zeros((1, x.shape[0]), dtype=np.float64)
        return out

    def arc_ap_fun(x: NDArray) -> NDArray:
        arr = ap_fn(jnp.asarray(x, dtype=jnp.float32))
        out = np.asarray(arr, dtype=np.float64)
        return out if out.size > 0 else np.array([1.0])

    def arc_ap_jac_fn(x: NDArray) -> NDArray:
        jacm = ap_jac(jnp.asarray(x, dtype=jnp.float32))
        out = np.asarray(jacm, dtype=np.float64)
        if out.size == 0:
            return np.zeros((1, x.shape[0]), dtype=np.float64)
        return out

    def intra_ml_fun(x: NDArray) -> NDArray:
        arr = ml_fn(jnp.asarray(x, dtype=jnp.float32))
        out = np.asarray(arr, dtype=np.float64)
        return out if out.size > 0 else np.array([1.0])

    def intra_ml_jac_fn(x: NDArray) -> NDArray:
        jacm = ml_jac(jnp.asarray(x, dtype=jnp.float32))
        out = np.asarray(jacm, dtype=np.float64)
        if out.size == 0:
            return np.zeros((1, x.shape[0]), dtype=np.float64)
        return out

    return {
        "threading": {"fun": threading_fun, "jac": threading_jac_fn},
        "arc_ap_separation": {"fun": arc_ap_fun, "jac": arc_ap_jac_fn},
        "intra_arc_ml_separation": {"fun": intra_ml_fun, "jac": intra_ml_jac_fn},
    }
