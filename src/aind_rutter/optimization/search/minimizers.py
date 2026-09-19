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
    _pack_statics,
    _signature,
)
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
    ``trilinear_sdf`` then does the gather/blend in bf16 with fp32 output
    (the cross-point reduction stays fp32). The grid is ~200x larger than
    the surface points, so bf16 on the grid alone captures the storage win.
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
    tpack = _pack_statics(template_statics, n_arcs)
    # bf16 grid storage for both the per-probe tuple (fixture loop) and the
    # padded swept-pair table (pair loop). See clearance_sweep for the policy.
    shared = cast_packed_grids(
        {k: tpack[k] for k in ARG_ORDER if k not in PER_CAND}, grid_dtype
    )

    def build_arglist(statics_list):
        # Only PER_CAND keys are used per chunk; the SDF tuples + sdf_table come
        # from the shared template — skip the per-chunk grid→device conversion.
        packs = [_pack_statics(s, n_arcs, build_sdf=False) for s in statics_list]
        stacked = {k: jnp.stack([jnp.asarray(p[k]) for p in packs]) for k in PER_CAND}
        return [stacked[k] if k in PER_CAND else shared[k] for k in ARG_ORDER]

    return vobj, vgrad, build_arglist


def make_staged_rprop(
    vmapped_grad_cw,
    *,
    eta0_frac=0.02,
    etamax_frac=0.5,
    eta_min=1e-6,
    grow=1.2,
    shrink=0.5,
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
    ``η=0``. Pass the cov_weight-aware grad (``vmapped_grad_cw``)."""

    def run(x0, arglist, lo, hi, cov_weight, n_steps):
        lo_j = jnp.asarray(lo, jnp.float32)
        hi_j = jnp.asarray(hi, jnp.float32)
        cw = jnp.asarray(cov_weight, jnp.float32)
        rng = hi_j - lo_j  # per-coord range; 0 for pinned coords
        eta0 = jnp.broadcast_to(eta0_frac * rng, x0.shape)
        eta_max = etamax_frac * rng

        def body(i, st):
            x, eta, g_prev = st
            g = vmapped_grad_cw(x, cw, *arglist)
            prod = g * g_prev
            eta = jnp.where(
                prod > 0,
                jnp.minimum(eta * grow, eta_max),
                jnp.where(prod < 0, jnp.maximum(eta * shrink, eta_min), eta),
            )
            g_eff = jnp.where(prod < 0, 0.0, g)  # iRprop−: skip step on a flip
            x = jnp.clip(x - jnp.sign(g_eff) * eta, lo_j, hi_j)
            return (x, eta, g_eff)

        init = (x0, eta0, jnp.zeros_like(x0))
        return jax.lax.fori_loop(0, n_steps, body, init)[0]

    return jax.jit(run)


def build_cw_fns(
    st,
    n_arcs,
    cov,
    thick,
    brain,
    weights=None,
    coverage_ceilings=None,
    coverage_weights=None,
) -> tuple[BatchedObjectiveFn, BatchedGradientFn, list[Any]]:
    """Replicate make_batched_phase1_chunked's cov_weight grad/obj (bf16 grids)."""
    w = weights if weights is not None else Phase1Weights()
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
        return jit_obj(x, cov_weight=cov_weight, **dict(zip(ARG_ORDER, args)))

    in_axes = (0, None) + tuple(0 if k in PER_CAND else None for k in ARG_ORDER)
    vobj = jax.jit(jax.vmap(obj_cw, in_axes=in_axes))
    vgrad = jax.jit(jax.vmap(jax.grad(obj_cw, argnums=0), in_axes=in_axes))
    pack = _pack_statics(st, n_arcs)
    shared = cast_packed_grids(
        {k: pack[k] for k in ARG_ORDER if k not in PER_CAND}, jnp.bfloat16
    )
    stacked = {k: jnp.stack([jnp.asarray(pack[k])]) for k in PER_CAND}
    arglist = [stacked[k] if k in PER_CAND else shared[k] for k in ARG_ORDER]
    return vobj, vgrad, arglist
