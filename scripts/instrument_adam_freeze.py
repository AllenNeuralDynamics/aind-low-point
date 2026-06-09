"""Instrument the ADAM 'freeze': per-step gradient norm, effective denominator
√v̂, step size, and objective — for const vs moment_restart.

Tests the hypothesis that the continuous run stalls because the 2nd moment v
inflates from early (deep-collision) gradients and stays high (b2=0.999 long
memory), shrinking the effective step lr·m̂/(√v̂+ε) to ~0 before the basin floor.
moment_restart zeroes m,v every `period` steps, which should sawtooth √v̂ back
down and revive the step.

Runs the REAL schedule (500 reduced @cov0 + 500 full @cov1, m,v reset at the
stage boundary) with manual ADAM matching make_staged_adam exactly, logging a
single candidate's scalars each step. Saves a 2x2 plot.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.instrument_adam_freeze
Env:  IDX=4195  PERIOD=50  LR=0.02
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle

import jax
import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.clearance_sweep import (
    cast_fixture_grids,
    cast_packed_grids,
)
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.optimizer_vars import build_y
from aind_low_point.optimization.stage3_phase1_jax import (
    Phase1Weights,
    _build_jit,
    _pack_statics,
    _signature,
)
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.arc_first_mrv import Enumerator, build_or_load_atlas
from scripts.batched_phase1_build import ARG_ORDER, PER_CAND
from scripts.log_candidate_trajectories import reduced_lohi, restore_spins_mrv
from scripts.restore_well_adam_manual import setup
from scripts.run_phase1_sample import (
    build_coverage_data,
    maybe_build_brain_sdf,
    phase1_bounds,
)
from scripts.thick_well_sdf import fit_well_cone, make_thick_well_sdf

IDX = int(_os.environ.get("IDX", "4195"))
PERIOD = int(_os.environ.get("PERIOD", "50"))
LR = float(_os.environ.get("LR", "0.02"))
B1, B2, EPS = 0.9, 0.999, 1e-8
OUT_PNG = _os.environ.get("OUT_PNG", "scratch/adam_freeze.png")


def build_cw_fns(
    st,
    n_arcs,
    cov,
    thick,
    brain,
    weights=None,
    coverage_ceilings=None,
    coverage_weights=None,
):
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


def run_logged(x0, vobj, vgrad, arglist, lo_r, hi_r, lo, hi, *, reset_period):
    """Manual ADAM, 500 reduced (cw0) + 500 full (cw1), logging per-step scalars
    for candidate 0. reset_period<=0 ⇒ const (reset only at the stage boundary)."""
    x = jnp.asarray(x0[None], jnp.float32)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    rows = []
    for stage, (cw, loj, hij) in enumerate(((0.0, lo_r, hi_r), (1.0, lo, hi))):
        loj = jnp.asarray(loj, jnp.float32)
        hij = jnp.asarray(hij, jnp.float32)
        m = jnp.zeros_like(x)  # stage-boundary reset (new run() call in prod)
        v = jnp.zeros_like(x)
        for j in range(500):
            g = vgrad(x, cw, *arglist)
            if reset_period > 0 and j % reset_period == 0 and j > 0:
                m = jnp.zeros_like(m)
                v = jnp.zeros_like(v)
                local = 0
            else:
                local = j % reset_period if reset_period > 0 else j
            m = B1 * m + (1 - B1) * g
            v = B2 * v + (1 - B2) * g * g
            tt = float(local) + 1.0
            mh = m / (1 - B1**tt)
            vh = v / (1 - B2**tt)
            denom = jnp.sqrt(vh) + EPS
            upd = LR * mh / denom
            x = jnp.clip(x - upd, loj, hij)
            gn = float(jnp.linalg.norm(g[0]))
            dn = float(jnp.mean(denom[0]))
            sn = float(jnp.linalg.norm(upd[0]))
            ob = float(vobj(x, cw, *arglist)[0])
            rows.append((stage * 500 + j, gn, dn, sn, ob))
    return np.array(rows)


def main() -> int:
    cfg, rt, probes, holes, sdf, bvh, fixtures, well, fixture_bvhs = setup()
    brain = maybe_build_brain_sdf(rt, compile_all_transforms(cfg.transforms))
    mesh = rt.asset_catalog.get_geometry("well").raw
    thick = make_thick_well_sdf(mesh, well, cone=fit_well_cone(mesh))
    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    cand = data["candidates"][IDX]
    n_arcs = int(data["results"][IDX].n_arcs)
    K = len(probes)
    enum = Enumerator(*build_or_load_atlas(), ml_margin_deg=0.0, ml_mode="greedy")

    arc_l, ml_l, sp_l = restore_spins_mrv(
        n_arcs,
        [IDX],
        probes=probes,
        holes=holes,
        data=data,
        sdf_by_name=sdf,
        well=thick,
        enum=enum,
    )
    st = _build_probe_static(
        probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf
    )
    z = np.zeros(K)
    x0 = build_y(
        np.asarray(arc_l[0]), n_arcs, np.asarray(ml_l[0]), np.asarray(sp_l[0]), z, z, z
    ).astype(np.float32)
    cov = build_coverage_data(probes, st)
    vobj, vgrad, arglist = build_cw_fns(st, n_arcs, cov, thick, brain)

    bounds = phase1_bounds(n_arcs, K)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    lo_r, hi_r = reduced_lohi(lo, hi, n_arcs, K)

    print(f"#{IDX}: logging const vs moment_restart(period {PERIOD})...", flush=True)
    A = run_logged(x0, vobj, vgrad, arglist, lo_r, hi_r, lo, hi, reset_period=0)
    B = run_logged(x0, vobj, vgrad, arglist, lo_r, hi_r, lo, hi, reset_period=PERIOD)

    # checkpoints
    print(
        f"\n{'step':>5} | {'const: |g|':>10} {'√v̂':>8} {'|Δx|':>9} {'obj':>10} | "
        f"{'mrst: |g|':>10} {'√v̂':>8} {'|Δx|':>9} {'obj':>10}"
    )
    for s in (1, 100, 250, 499, 500, 600, 750, 999):
        a, b = A[s], B[s]
        print(
            f"{s:>5} | {a[1]:>10.3g} {a[2]:>8.3g} {a[3]:>9.3g} {a[4]:>10.4g} | "
            f"{b[1]:>10.3g} {b[2]:>8.3g} {b[3]:>9.3g} {b[4]:>10.4g}"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        titles = [
            "gradient norm |g|",
            "effective denom √v̂ (mean)",
            "step size |Δx|",
            "objective",
        ]
        for k, t in enumerate(titles):
            axk = ax[k // 2][k % 2]
            axk.plot(A[:, 0], A[:, k + 1], label="const", lw=1)
            axk.plot(B[:, 0], B[:, k + 1], label="moment_restart", lw=1, alpha=0.8)
            axk.axvline(500, color="k", ls=":", lw=0.7)  # reduced→full boundary
            axk.set_title(t)
            axk.set_yscale("log" if k < 3 else "linear")
            axk.legend(fontsize=8)
        fig.suptitle(f"#{IDX}: ADAM freeze — const vs moment_restart (period {PERIOD})")
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=110)
        print(f"\nsaved plot → {OUT_PNG}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
