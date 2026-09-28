"""Batched Phase-1 objective: vmap the EXISTING per-candidate _objective.

Step 1 of the batched-ADAM build. Rather than re-implement the Phase-1
objective in BatchedProbeStatic form, we vmap the existing per-candidate
``_objective`` (built by ``_build_jit``) over a batch of candidates:
  - per-candidate axes (mapped): x, arc_idx, hole sections, same_arc_mask
  - shared axes (None): probe target/pivot/tips, SDFs, shank OBBs
  - fixtures=[well] + coverage_data are closure-captured (shared)

Correctness is free: it's literally the same function, so the batched
value must equal the per-candidate ``_objective`` looped over the batch.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import Any, Callable

import jax
import jax.numpy as jnp

from aind_rutter.optimization.clearance.sweep import (
    cast_fixture_grids,
    cast_packed_grids,
)
from aind_rutter.optimization.objectives.soft import (
    PACKED_ARG_ORDER,
    PACKED_PER_CAND_KEYS,
    Phase1Weights,
    _build_jit,
    _signature,
    pack_statics,
)
from aind_rutter.optimization.objectives.spin_angle import from_angle_layout
from aind_rutter.optimization.pipeline.records import (
    BatchedGradientFn,
    BatchedObjectiveFn,
    Phase1ChunkedFns,
)

ARG_ORDER = list(PACKED_ARG_ORDER)
PER_CAND = set(PACKED_PER_CAND_KEYS)


def make_batched_phase1_chunked(
    template_statics,
    n_arcs,
    weights,
    fixtures,
    coverage_data=None,
    grid_dtype=jnp.float32,
    brain_sdf=None,
    coverage_ceilings=None,
    coverage_weights=None,
) -> Phase1ChunkedFns:
    """Build the reusable pieces for VRAM-chunked evaluation:
    (vobj, vgrad, build_arglist).

    vobj/vgrad are compiled vmap functions taking (x, *arglist) — reuse
    them across same-sized chunks (compile once). build_arglist(statics)
    packs a chunk's per-row arrays into the positional arglist.

    ``grid_dtype`` stores the probe SDF voxel grids at that dtype on device.
    Pass ``jnp.bfloat16`` for the real mixed-precision kernel: halves the
    dominant array (Nx*Ny*Nz) → roughly doubles the chunk ceiling, and
    ``trilinear_sdf`` gathers the bf16 corners and blends in fp32. The grid is
    ~200x larger than the surface points, so bf16 on the grid alone captures
    the storage win.
    """
    # bf16 the fixture grids too (all collision SDFs share the dtype), before
    # _build_jit closure-captures them. See clearance_sweep for the policy.
    fixtures = cast_fixture_grids(fixtures, grid_dtype)
    base_sig = _signature(template_statics, n_arcs, weights)
    jit_obj, _ = _build_jit(
        base_sig,
        weights,
        coverage_data=coverage_data,
        fixtures=fixtures,
        brain_sdf=brain_sdf,
        coverage_ceilings=coverage_ceilings,
        coverage_weights=coverage_weights,
    )

    def obj_pos(x, *args):
        return jit_obj(x, **dict(zip(ARG_ORDER, args)))

    in_axes = (0,) + tuple(0 if k in PER_CAND else None for k in ARG_ORDER)
    vmapped_grad = jax.vmap(jax.grad(obj_pos), in_axes=in_axes)  # raw (un-jit)
    vobj = jax.jit(jax.vmap(obj_pos, in_axes=in_axes))
    vgrad = jax.jit(vmapped_grad)

    # Shared per-probe constants are identical across all candidates;
    # build them once from the template.
    tpack = pack_statics(template_statics, n_arcs)
    # bf16 grid storage for both the per-probe tuple (fixture loop) and the
    # padded swept-pair table (pair loop). See clearance_sweep for the policy.
    shared = cast_packed_grids(
        {k: tpack[k] for k in ARG_ORDER if k not in PER_CAND}, grid_dtype
    )

    def build_arglist(statics_list):
        # Only PER_CAND keys are used per chunk; the SDF tuples + sdf_table come
        # from the shared template — skip the per-chunk grid→device conversion.
        packs = [pack_statics(s, n_arcs, build_sdf=False) for s in statics_list]
        stacked = {k: jnp.stack([jnp.asarray(p[k]) for p in packs]) for k in PER_CAND}
        return [stacked[k] if k in PER_CAND else shared[k] for k in ARG_ORDER]

    return vobj, vgrad, build_arglist


#: Per-iteration channels recorded when ``make_staged_rprop`` traces, in order.
#: All but ``step`` are fractions of the coordinates a stage actually moves;
#: ``step`` is their mean absolute displacement.
TRACE_CHANNELS = ("flip", "eta_floor", "eta_ceil", "clipped", "wrapped", "step")


def make_staged_rprop(
    vmapped_grad_cw,
    *,
    eta0_frac=0.02,
    etamax_frac=0.5,
    eta_min=1e-6,
    grow=1.2,
    shrink=0.5,
    periodic=None,
    trace=0,
) -> Callable[..., Any]:
    """Projected iRprop− (sign-based resilient backprop).

    ``run(x0, arglist, lo, hi, cov_weight, n_steps)`` takes the bounds, the
    coverage weight and the step count as RUNTIME args, so the reduced
    (offsets/depth pinned via ``lo==hi``, ``cov_weight=0``) and full stages hit
    the same XLA executable.

    Magnitude-INVARIANT: each coordinate steps ``sign(g)·η_i`` with a per-coord
    step ``η_i`` that grows (×``grow``) on a consistent gradient sign and shrinks
    (×``shrink``) on a sign flip; on a flip the step is skipped and the stored
    gradient zeroed (iRprop−), so no double-counting. Immune to ADAM's stale-``v``
    freeze (ignores gradient magnitude) and the right regime for a DETERMINISTIC
    full-batch ill-conditioned objective.

    ``η`` is initialised/capped RELATIVE TO THE BOUND RANGE per coordinate
    (``η0 = eta0_frac·(hi−lo)``), which preconditions the scale disparity (deg vs
    mm vs cos/sin) and naturally pins ``lo==hi`` coords (reduced offsets/depth) at
    ``η=0``. Pass the cov_weight-aware grad (``vmapped_grad_cw``).

    ``periodic`` marks coordinates that wrap into ``[lo, hi)`` instead of being
    clipped, for variables the objective reads with that period. The wrap leaves
    the state intact: a periodic gradient has the same sign one period over.

    ``trace`` records ``TRACE_CHANNELS`` per iteration for that many iterations,
    and makes ``run`` return ``(x, eta, stats)`` with ``stats`` of shape
    ``(trace, batch, len(TRACE_CHANNELS))`` and the final per-coordinate ``η``.
    It is a static Python flag, so the untraced path emits the computation it
    emitted before the channels existed. Its buffer is sized statically because
    ``n_steps`` is a runtime argument; rows past the steps actually taken stay
    zero, and steps past ``trace`` are dropped."""

    def run(x0, arglist, lo, hi, cov_weight, n_steps):
        lo_j = jnp.asarray(lo, jnp.float32)
        hi_j = jnp.asarray(hi, jnp.float32)
        cw = jnp.asarray(cov_weight, jnp.float32)
        rng = hi_j - lo_j  # per-coord range; 0 for pinned coords
        eta0 = jnp.broadcast_to(eta0_frac * rng, x0.shape)
        eta_max = etamax_frac * rng
        if periodic is not None:
            wraps = jnp.asarray(periodic, bool)
            period = jnp.where(wraps, rng, 1.0)
        if trace:
            # Pinned coordinates never move, so counting them would dilute every
            # rate by however many a stage happens to pin.
            movable = rng > 0
            n_movable = jnp.maximum(jnp.count_nonzero(movable), 1).astype(jnp.float32)

        def body(i, st):
            x, eta, g_prev, stats = st
            g = vmapped_grad_cw(x, cw, *arglist)
            prod = g * g_prev
            eta = jnp.where(
                prod > 0,
                jnp.minimum(eta * grow, eta_max),
                jnp.where(prod < 0, jnp.maximum(eta * shrink, eta_min), eta),
            )
            g_eff = jnp.where(prod < 0, 0.0, g)  # iRprop−: skip step on a flip
            x_new = x - jnp.sign(g_eff) * eta
            if periodic is None:
                x_next = jnp.clip(x_new, lo_j, hi_j)
            else:
                wrapped = lo_j + jnp.mod(x_new - lo_j, period)
                x_next = jnp.where(wraps, wrapped, jnp.clip(x_new, lo_j, hi_j))
            if trace:
                moved = x_new != x_next  # the projection or the wrap intervened
                is_wrap = (
                    moved & wraps if periodic is not None else jnp.zeros_like(moved)
                )

                def rate(m):
                    return jnp.count_nonzero(m & movable, axis=-1) / n_movable

                stats = stats.at[i].set(
                    jnp.stack(
                        [
                            rate(prod < 0),
                            rate(eta <= eta_min),
                            rate(eta >= eta_max),
                            rate(moved & ~is_wrap),
                            rate(is_wrap),
                            jnp.sum(jnp.abs(x_next - x) * movable, axis=-1) / n_movable,
                        ],
                        axis=-1,
                    )
                )
            return (x_next, eta, g_eff, stats)

        buf = (
            jnp.zeros((trace, x0.shape[0], len(TRACE_CHANNELS)), jnp.float32)
            if trace
            else None
        )
        init = (x0, eta0, jnp.zeros_like(x0), buf)
        x, eta, _, stats = jax.lax.fori_loop(0, n_steps, body, init)
        return (x, eta, stats) if trace else x

    return jax.jit(run)


def build_cw_fns(
    st,
    n_arcs,
    cov,
    thick,
    brain,
    *,
    weights: Phase1Weights,
    coverage_ceilings=None,
    coverage_weights=None,
    spin_angle=False,
) -> tuple[BatchedObjectiveFn, BatchedGradientFn, list[Any]]:
    """Replicate make_batched_phase1_chunked's cov_weight grad/obj (bf16 grids).

    With ``spin_angle`` both take the angle layout of
    :func:`~aind_rutter.optimization.objectives.spin_angle.to_angle_layout`.
    """
    w = weights
    sig = _signature(st, n_arcs, w)
    # bf16 all collision grids (fixture + probe + table), like the chunked
    # builder. See clearance_sweep for the policy.
    (thick,) = cast_fixture_grids((thick,), jnp.bfloat16)
    jit_obj, _ = _build_jit(
        sig,
        w,
        coverage_data=cov,
        fixtures=(thick,),
        brain_sdf=brain,
        coverage_ceilings=coverage_ceilings,
        coverage_weights=coverage_weights,
    )

    def obj_cw(x, cov_weight, *args):
        if spin_angle:
            x = from_angle_layout(x, n_arcs)
        return jit_obj(x, cov_weight=cov_weight, **dict(zip(ARG_ORDER, args)))

    in_axes = (0, None) + tuple(0 if k in PER_CAND else None for k in ARG_ORDER)
    vobj = jax.jit(jax.vmap(obj_cw, in_axes=in_axes))
    vgrad = jax.jit(jax.vmap(jax.grad(obj_cw, argnums=0), in_axes=in_axes))
    pack = pack_statics(st, n_arcs)
    shared = cast_packed_grids(
        {k: pack[k] for k in ARG_ORDER if k not in PER_CAND}, jnp.bfloat16
    )
    stacked = {k: jnp.stack([jnp.asarray(pack[k])]) for k in PER_CAND}
    arglist = [stacked[k] if k in PER_CAND else shared[k] for k in ARG_ORDER]
    return vobj, vgrad, arglist
