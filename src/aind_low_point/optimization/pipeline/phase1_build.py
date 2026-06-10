"""Batched Phase-1 objective: vmap the EXISTING per-candidate _objective.

Step 1 of the batched-ADAM build. Rather than re-implement the Phase-1
objective in BatchedProbeStatic form, we vmap the existing per-candidate
``_objective`` (built by ``_build_jit``) over a batch of candidates:
  - per-candidate axes (mapped): x, arc_idx, hole sections, same_arc_mask
  - shared axes (None): probe target/pivot/tips, SDFs, shank OBBs
  - fixtures=[well] + coverage_data are closure-captured (shared)

Correctness is free: it's literally the same function, so the batched
value must equal the per-candidate ``make_phase1_objective`` looped over
the batch. This script builds it and validates that exact match.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.clearance_sweep import (
    cast_fixture_grids,
    cast_packed_grids,
)
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.pipeline.contracts import (
    BatchedGradientFn,
    BatchedObjectiveFn,
    Phase1ChunkedFns,
    Phase1ObjectiveFns,
)
from aind_low_point.optimization.pipeline.runtime_adapter import (
    OptimizationRuntime,
)
from aind_low_point.optimization.stage3_phase1_jax import (
    PACKED_ARG_ORDER,
    PACKED_PER_CAND_KEYS,
    Phase1Weights,
    _build_jit,
    _pack_statics,
    _signature,
    make_phase1_objective,
)

ARG_ORDER = list(PACKED_ARG_ORDER)
PER_CAND = set(PACKED_PER_CAND_KEYS)


def make_batched_phase1_objective(
    statics_list,
    n_arcs,
    weights,
    fixtures,
    coverage_data=None,
    brain_sdf=None,
    coverage_ceilings=None,
    coverage_weights=None,
) -> Phase1ObjectiveFns:
    """vmap the per-candidate _objective over `statics_list`. Returns
    (batched_obj(x_B)->(B,), batched_grad(x_B)->(B,nvars))."""
    base_sig = _signature(statics_list[0], n_arcs, weights)
    jit_obj, _ = _build_jit(
        base_sig,
        weights,
        coverage_data=coverage_data,
        fixtures=fixtures,
        brain_sdf=brain_sdf,
        coverage_ceilings=coverage_ceilings,
        coverage_weights=coverage_weights,
    )

    packs = [_pack_statics(s, n_arcs) for s in statics_list]
    # Shared per-probe constants. Some keys (target/pivot/tips/shank_mask)
    # are uniform arrays; the SDF keys are LISTS of per-probe arrays with
    # heterogeneous shapes (different probe kinds) — pass those through
    # as-is (broadcast via in_axes=None), do NOT stack.
    shared = {k: packs[0][k] for k in ARG_ORDER if k not in PER_CAND}
    stacked = {k: jnp.stack([jnp.asarray(p[k]) for p in packs]) for k in PER_CAND}

    def obj_pos(x, *args):
        return jit_obj(x, **dict(zip(ARG_ORDER, args)))

    in_axes = (0,) + tuple(0 if k in PER_CAND else None for k in ARG_ORDER)
    vobj = jax.jit(jax.vmap(obj_pos, in_axes=in_axes))
    vgrad = jax.jit(jax.vmap(jax.grad(obj_pos), in_axes=in_axes))

    arglist = [stacked[k] if k in PER_CAND else shared[k] for k in ARG_ORDER]

    def batched_obj(x_B):
        return vobj(jnp.asarray(x_B, jnp.float32), *arglist)

    def batched_grad(x_B):
        return vgrad(jnp.asarray(x_B, jnp.float32), *arglist)

    return batched_obj, batched_grad


def make_batched_phase1_chunked(  # noqa: C901
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
    """Like make_batched_phase1_objective, but returns the reusable pieces
    for VRAM-chunked evaluation: (vobj, vgrad, build_arglist).

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

    # cov_weight-aware variant: cov_weight rides as a shared (in_axes=None)
    # runtime scalar so ONE compiled kernel serves the reduced (cov_weight=0)
    # and full (cov_weight=1) stages. grad is still w.r.t. x only (argnums=0).
    def obj_pos_cw(x, cov_weight, *args):
        return jit_obj(x, cov_weight=cov_weight, **dict(zip(ARG_ORDER, args)))

    in_axes_cw = (0, None) + tuple(0 if k in PER_CAND else None for k in ARG_ORDER)
    vmapped_grad_cw = jax.vmap(jax.grad(obj_pos_cw, argnums=0), in_axes=in_axes_cw)

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

    def make_adam(lo, hi, *, steps, lr, b1=0.9, b2=0.999, eps=1e-8):
        """Compiled projected ADAM: the entire `steps` loop is one kernel
        (lax.fori_loop) — no Python loop, no per-step host sync."""
        lo_j = jnp.asarray(lo, jnp.float32)
        hi_j = jnp.asarray(hi, jnp.float32)

        def run(x0, arglist):
            z = jnp.zeros_like(x0)

            def body(i, st):
                x, m, v = st
                g = vmapped_grad(x, *arglist)
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                tt = i.astype(jnp.float32) + 1.0
                mh = m / (1 - jnp.power(b1, tt))
                vh = v / (1 - jnp.power(b2, tt))
                x = jnp.clip(x - lr * mh / (jnp.sqrt(vh) + eps), lo_j, hi_j)
                return (x, m, v)

            return jax.lax.fori_loop(0, steps, body, (x0, z, z))[0]

        return jax.jit(run)

    def make_staged_adam(
        *,
        lr,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        schedule="const",
        min_lr_frac=0.0,
        period=50,
        grad_clip=0.0,
    ):
        """One compiled projected-ADAM kernel shared across stages.

        ``run(x0, arglist, lo, hi, cov_weight, n_steps)`` takes the bounds,
        the coverage weight, AND the step count as RUNTIME args, so the
        reduced (offsets/depth pinned via ``lo==hi``, ``cov_weight=0``) and
        full (real bounds, ``cov_weight=1``) stages hit the SAME XLA
        executable — no second compile. ``n_steps`` is the dynamic upper
        bound of the ``fori_loop`` (so the two stages can differ in length
        without recompiling).

        ``schedule`` sets the per-step learning rate ``lr(i)`` (the lever for
        ADAM's effective-step decay — ``v`` accumulates and stalls long runs):
          - ``"const"`` — flat ``lr`` (byte-identical to the un-scheduled kernel)
          - ``"cosine"`` — single cosine anneal ``lr → lr·min_lr_frac`` over
            ``n_steps`` (settle)
          - ``"cosine_restart"`` — cosine warm restarts (SGDR) with ``period``
            steps per cycle: periodic LR spikes to escape shallow minima.
          - ``"moment_restart"`` — reset the ADAM moments ``m,v`` to 0 every
            ``period`` steps (with bias correction restarted): the segmented
            momentum-reset hack baked into ONE continuous kernel. The reset
            yields a full ``lr·sign(g)`` step that re-energizes a stalled run —
            this is the mechanism (not the LR) that the segmented schedule
            exploits.
        """
        base = float(lr)
        mn = base * float(min_lr_frac)
        per = max(float(period), 1.0)

        def lr_at(i, n_steps):
            t = i.astype(jnp.float32)
            if schedule == "cosine":
                n = jnp.maximum(n_steps.astype(jnp.float32), 1.0)
                return mn + 0.5 * (base - mn) * (1.0 + jnp.cos(jnp.pi * t / n))
            if schedule == "cosine_restart":
                frac = jnp.mod(t, per) / per
                return mn + 0.5 * (base - mn) * (1.0 + jnp.cos(jnp.pi * frac))
            return jnp.float32(base)

        def run(x0, arglist, lo, hi, cov_weight, n_steps):
            lo_j = jnp.asarray(lo, jnp.float32)
            hi_j = jnp.asarray(hi, jnp.float32)
            cw = jnp.asarray(cov_weight, jnp.float32)
            z = jnp.zeros_like(x0)

            def body(i, st):
                x, m, v = st
                g = vmapped_grad_cw(x, cw, *arglist)
                if grad_clip > 0.0:
                    # per-candidate global-norm clip: caps the magnitude the 2nd
                    # moment v sees (stops the seed/coverage spike poisoning √v̂)
                    # while preserving direction.
                    gn = jnp.sqrt(jnp.sum(g * g, axis=-1, keepdims=True))
                    g = g * jnp.minimum(1.0, grad_clip / (gn + 1e-12))
                if schedule == "moment_restart":
                    # reset m,v at each period boundary; bias-correct with the
                    # WITHIN-cycle step count so the post-reset step is full-size.
                    seg_i = jnp.mod(i.astype(jnp.float32), per)
                    keep = (seg_i != 0.0).astype(jnp.float32)
                    m = m * keep
                    v = v * keep
                    tt = seg_i + 1.0
                else:
                    tt = i.astype(jnp.float32) + 1.0
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                mh = m / (1 - jnp.power(b1, tt))
                vh = v / (1 - jnp.power(b2, tt))
                lr_i = lr_at(i, n_steps)
                x = jnp.clip(x - lr_i * mh / (jnp.sqrt(vh) + eps), lo_j, hi_j)
                return (x, m, v)

            return jax.lax.fori_loop(0, n_steps, body, (x0, z, z))[0]

        return jax.jit(run, static_argnums=())

    return vobj, vgrad, build_arglist, make_adam, make_staged_adam


def make_staged_rprop(
    vmapped_grad_cw,
    *,
    eta0_frac=0.02,
    etamax_frac=0.5,
    eta_min=1e-6,
    grow=1.2,
    shrink=0.5,
) -> Callable[..., Any]:
    """Projected iRprop− (sign-based resilient backprop) on the same interface as
    ``make_staged_adam``'s ``run(x0, arglist, lo, hi, cov_weight, n_steps)``.

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


def main() -> int:
    opt = OptimizationRuntime.from_config_path(
        "examples/836656-config-T12.yml", "scratch/0283-300-04.holes.yml"
    )
    _cfg, _rt, probes, holes, sdf_by_name, bvh_cache, fixtures, well, _fixture_bvhs = (
        opt.as_legacy_setup()
    )

    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    cand_idxs = [4195, 1035, 230, 2291]
    statics_list = []
    x_list = []
    for idx in cand_idxs:
        cand = data["candidates"][idx]
        st = _build_probe_static(
            probes,
            holes,
            cand.ha,
            cand.aa,
            bvh_cache=bvh_cache,
            sdf_by_name=sdf_by_name,
        )
        statics_list.append(st)
        x_list.append(np.asarray(data["augmented_phase1_x"][idx], np.float32))
    n_arcs = data["results"][cand_idxs[0]].n_arcs
    weights = Phase1Weights()

    print("Building batched objective (vmap of per-cand _objective)...")
    bobj, bgrad = make_batched_phase1_objective(
        statics_list, n_arcs, weights, (well,), coverage_data=None
    )
    x_B = np.stack(x_list)
    bvals = np.asarray(bobj(x_B))

    print(f"\n{'cand':>6} {'batched':>14} {'per-cand':>14} {'abs diff':>12}")
    print("-" * 50)
    maxdiff = 0.0
    for i, idx in enumerate(cand_idxs):
        fun, _ = make_phase1_objective(
            statics_list[i],
            n_arcs,
            coverage_data=None,
            fixtures=(well,),
            weights=weights,
        )
        pv = float(fun(x_list[i]))
        d = abs(float(bvals[i]) - pv)
        maxdiff = max(maxdiff, d)
        print(f"{idx:>6} {float(bvals[i]):>14.5f} {pv:>14.5f} {d:>12.2e}")
    print(
        f"\nmax abs diff (batched vs per-cand): {maxdiff:.3e}  "
        f"{'PASS' if maxdiff < 1e-2 else 'FAIL'}"
    )

    # gradient sanity: finite + matches per-cand grad on cand 0
    g = np.asarray(bgrad(x_B))
    fun0, jac0 = make_phase1_objective(
        statics_list[0], n_arcs, coverage_data=None, fixtures=(well,), weights=weights
    )
    g0 = np.asarray(jac0(x_list[0]))
    gdiff = float(np.max(np.abs(g[0] - g0)))
    print(
        f"grad[0] max abs diff vs per-cand jac: {gdiff:.3e}  "
        f"(finite={np.isfinite(g).all()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


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
