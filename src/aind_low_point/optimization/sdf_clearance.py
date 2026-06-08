"""Bridge between the optimizer's ``OptimizerContext`` and the
JAX-based SDF pairwise-clearance kernels.

The inner solve's scipy SLSQP needs:

* a constraint function ``g(x) -> (n_pairs,)`` returning signed
  clearances per probe pair (positive ⇒ clear);
* its Jacobian ``J(x) -> (n_pairs, n_x)``.

We compute both via JAX:

* :func:`pair_clearance_at_x` extracts the per-probe variables from
  ``x``, runs :func:`pose_from_optimizer_vars` on each, and calls
  :func:`pairwise_signed_clearance` on the two posed probes.
* :func:`build_sdf_clearance_callbacks` JIT-compiles per pair and
  returns the (fun, jac) pair scipy expects.

Smooth SDF gradients survive overlap (unlike FCL's witness-point
gradient, which becomes noise once two meshes intersect), so SLSQP
can escape an overlapping seed without needing the spin/push
restoration step.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from aind_low_point.optimization.sdf import ProbeSDF
from aind_low_point.optimization.sdf_jax import (
    pairwise_signed_clearance,
    pose_from_optimizer_vars,
)


@dataclass(frozen=True)
class _ProbeJaxData:
    """Per-probe static data needed by the JAX clearance function.

    Captures everything from ``ProbeContext`` and ``ProbeSDF`` that
    has to be passed to the JIT-compiled kernel as a static argument
    (constant across SLSQP iterations).
    """

    target_LPS: jnp.ndarray  # (3,)
    pivot_local: jnp.ndarray  # (3,)
    arc_idx: int  # which entry of ``x[:n_arcs]`` to read for this probe
    var_offset: int  # offset into x for this probe's 5 vars
    sdf_grid: jnp.ndarray  # (Nx, Ny, Nz)
    sdf_origin: jnp.ndarray  # (3,)
    sdf_spacing: jnp.ndarray  # scalar
    surface: jnp.ndarray  # (N, 3)


def _pose_for_probe(
    x: jnp.ndarray,
    n_arcs: int,
    probe: _ProbeJaxData,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract this probe's optimizer vars from ``x`` and compute its
    world ``(R, t)`` pose. Pure JAX so ``jacrev`` works through it.
    """
    arc_ap = x[probe.arc_idx]
    off = probe.var_offset
    ml = x[off + 0]
    spin = x[off + 1]
    off_R = x[off + 2]
    off_A = x[off + 3]
    depth = x[off + 4]
    R, t = pose_from_optimizer_vars(
        target_LPS=probe.target_LPS,
        ap_deg=arc_ap,
        ml_deg=ml,
        spin_deg=spin,
        offset_R_mm=off_R,
        offset_A_mm=off_A,
        past_target_mm=depth,
        recording_center_local=probe.pivot_local,
    )
    return R, t


def pair_clearance_at_x(
    x: jnp.ndarray,
    n_arcs: int,
    probe_a: _ProbeJaxData,
    probe_b: _ProbeJaxData,
) -> jnp.ndarray:
    """Signed clearance for one probe pair at optimizer state ``x``.

    Returns a scalar — negative means overlap, positive means clear.
    Differentiable w.r.t. every entry of ``x`` (only the entries
    corresponding to the two probes' vars + the arc APs they're on
    contribute non-zero gradient).
    """
    R_a, t_a = _pose_for_probe(x, n_arcs, probe_a)
    R_b, t_b = _pose_for_probe(x, n_arcs, probe_b)
    return pairwise_signed_clearance(
        R_a,
        t_a,
        R_b,
        t_b,
        probe_a.sdf_grid,
        probe_a.sdf_origin,
        probe_a.sdf_spacing,
        probe_b.sdf_grid,
        probe_b.sdf_origin,
        probe_b.sdf_spacing,
        probe_a.surface,
        probe_b.surface,
    )


def build_sdf_clearance_callbacks(
    n_arcs: int,
    probe_data: list[_ProbeJaxData],
    safety_clearance_mm: float = 0.0,
    clearance_overlap_allowance_mm: float = 0.0,
):
    """Return ``(fun, jac)`` for scipy.optimize.minimize's constraint
    dict ``{'type': 'ineq', 'fun': fun, 'jac': jac}``.

    ``fun(x) -> (n_pairs,)`` returns signed slacks
    ``clearance - (safety - allowance)`` so SLSQP's feasibility
    condition ``fun(x) >= 0`` matches the existing FCL-based
    constraint formulation. ``jac(x) -> (n_pairs, n_x)`` is the
    analytic Jacobian via ``jax.grad``.

    Both functions return numpy arrays — scipy doesn't grok JAX
    tracers. Per-pair JIT compilation happens lazily on first call;
    each shape combination is cached.
    """
    pairs: list[tuple[int, int]] = []
    n_probes = len(probe_data)
    for i in range(n_probes):
        for j in range(i + 1, n_probes):
            pairs.append((i, j))

    # JIT each pair's scalar clearance + its grad. Shapes only vary
    # by probe kind, so the cache is small.
    def _make_pair_fn(i, j):
        def fn(x):
            return pair_clearance_at_x(x, n_arcs, probe_data[i], probe_data[j])

        return fn

    pair_funcs = [jax.jit(_make_pair_fn(i, j)) for (i, j) in pairs]
    pair_grads = [jax.jit(jax.grad(_make_pair_fn(i, j))) for (i, j) in pairs]

    threshold = safety_clearance_mm - clearance_overlap_allowance_mm

    def fun(x: NDArray) -> NDArray:
        xj = jnp.asarray(x, dtype=jnp.float32)
        out = np.empty(len(pairs), dtype=np.float64)
        for k, fn in enumerate(pair_funcs):
            out[k] = float(fn(xj)) - threshold
        return out

    def jac(x: NDArray) -> NDArray:
        xj = jnp.asarray(x, dtype=jnp.float32)
        out = np.empty((len(pairs), x.shape[0]), dtype=np.float64)
        for k, jf in enumerate(pair_grads):
            out[k] = np.asarray(jf(xj), dtype=np.float64)
        return out

    return fun, jac


def build_probe_jax_data_for_context(
    ctx,
    sdf_by_probe_name: dict[str, ProbeSDF],
) -> list[_ProbeJaxData]:
    """Materialise a list of :class:`_ProbeJaxData` from an existing
    ``OptimizerContext`` (one per probe, in layout order) plus per-probe
    SDFs.

    The pivot_local is recomputed via the same formula the optimizer
    uses (``(centroid_x, centroid_y, active_center_mm)`` from shank
    tips + recording geom) so the JAX pose matches the optimizer's.
    """
    out: list[_ProbeJaxData] = []
    arc_id_to_idx = {a: i for i, a in enumerate(ctx.layout.arc_ids)}
    for i, probe in enumerate(ctx.probes):
        sdf = sdf_by_probe_name[probe.name]
        tips = np.asarray(probe.shank_tips_local, dtype=np.float64)
        geom = probe.recording_geom
        if tips.shape[0] > 0:
            pivot = np.array(
                [tips[:, 0].mean(), tips[:, 1].mean(), geom.active_center_mm],
                dtype=np.float64,
            )
        else:
            pivot = np.array([0.0, 0.0, geom.active_center_mm], dtype=np.float64)
        out.append(
            _ProbeJaxData(
                target_LPS=jnp.asarray(probe.target_LPS, dtype=jnp.float32),
                pivot_local=jnp.asarray(pivot, dtype=jnp.float32),
                arc_idx=arc_id_to_idx[probe.arc_id],
                var_offset=ctx.layout.num_arcs + 5 * i,
                sdf_grid=jnp.asarray(sdf.grid, dtype=jnp.float32),
                sdf_origin=jnp.asarray(sdf.origin, dtype=jnp.float32),
                sdf_spacing=jnp.asarray(sdf.spacing, dtype=jnp.float32),
                surface=jnp.asarray(sdf.surface_points, dtype=jnp.float32),
            )
        )
    return out
