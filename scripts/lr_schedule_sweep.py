"""Sweep ADAM LR schedules to replace the momentum-reset RESTART hack.

The segmented schedule (10x50, momentum reset each segment) finds ~2x the
feasibles of one continuous 500-step run — because ADAM's second moment ``v``
accumulates and the effective step decays, so a long continuous run stalls
before the basin floor; the reset re-energizes it. That's a tuning symptom, not
a feature. This sweeps principled fixes against both baselines, on the
calibration 545 with the THICK well (the better well model), 5000 surf:

  - const-b2.999  : flat-lr ADAM (= the pool's continuous schedule)  [baseline]
  - moment-rst50  : ADAM with m,v reset every 50 (the validated hack)
  - const-b2.99   : ADAM, lower b2 → v forgets faster, less stall
  - clip1000/100  : ADAM + per-cand grad-norm clip (stop the spike poisoning v)
  - rprop         : iRprop− — sign-based, magnitude-invariant, freeze-free

Restore + statics + grad-kernel + arglist are built ONCE per n_arcs group and
reused across configs (only the staged-ADAM loop + FCL differ). FCL (true mesh)
gates all. Reports feasible / winners / wall per config.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.lr_schedule_sweep
Env:  CHUNK=64  LIMIT=0  LR=0.02
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.stage3_phase1_jax import Phase1Weights
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.arc_first_mrv import Enumerator, build_or_load_atlas
from scripts.batched_phase1_build import (
    make_batched_phase1_chunked,
    make_staged_rprop,
)
from scripts.instrument_adam_freeze import build_cw_fns
from scripts.log_candidate_trajectories import (
    reduced_lohi,
    restore_spins_mrv,
    select_sample,
)
from scripts.restore_well_adam_manual import setup
from scripts.run_phase1_sample import (
    build_coverage_data,
    maybe_build_brain_sdf,
    phase1_bounds,
)
from scripts.test_h1_chain_cand4195 import build_y
from scripts.thick_well_sdf import fit_well_cone, make_thick_well_sdf

CHUNK = int(_os.environ.get("CHUNK", "64"))
LIMIT = int(_os.environ.get("LIMIT", "0"))
LR = float(_os.environ.get("LR", "0.02"))
OUT = _os.environ.get("OUT", "scratch/lr_schedule_sweep.pkl")

# (name, optimizer, kwargs, segmented?)
CONFIGS = [
    ("const-b2.999", "adam", dict(lr=LR, b2=0.999, schedule="const"), False),
    (
        "moment-rst50",
        "adam",
        dict(lr=LR, b2=0.999, schedule="moment_restart", period=50),
        False,
    ),
    ("const-b2.99", "adam", dict(lr=LR, b2=0.99, schedule="const"), False),
    (
        "clip1000",
        "adam",
        dict(lr=LR, b2=0.999, schedule="const", grad_clip=1000.0),
        False,
    ),
    (
        "clip100",
        "adam",
        dict(lr=LR, b2=0.999, schedule="const", grad_clip=100.0),
        False,
    ),
    ("rprop", "rprop", dict(eta0_frac=0.02, etamax_frac=0.5), False),
]


def run_group(n_arcs, idxs, *, common):
    probes, holes, data, bvh = (common[k] for k in ("probes", "holes", "data", "bvh"))
    well, thick, brain, fixtures, fbvh = (
        common[k] for k in ("well", "thick", "brain", "fixtures", "fixture_bvhs")
    )
    K = len(probes)
    bounds = phase1_bounds(n_arcs, K)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    lo_r, hi_r = reduced_lohi(lo, hi, n_arcs, K)

    arc_l, ml_l, sp_l = restore_spins_mrv(
        n_arcs,
        idxs,
        probes=probes,
        holes=holes,
        data=data,
        sdf_by_name=common["sdf"],
        well=thick,
        enum=common["enum"],
    )
    st, x0_rows = [], []
    for i, idx in enumerate(idxs):
        cand = data["candidates"][idx]
        st.append(
            _build_probe_static(
                probes,
                holes,
                cand.ha,
                cand.aa,
                bvh_cache=bvh,
                sdf_by_name=common["sdf"],
            )
        )
        z = np.zeros(K)
        x0_rows.append(
            build_y(
                np.asarray(arc_l[i]),
                n_arcs,
                np.asarray(ml_l[i]),
                np.asarray(sp_l[i]),
                z,
                z,
                z,
            )
        )
    x0 = np.stack(x0_rows).astype(np.float32)
    n = x0.shape[0]
    npad = (-n) % CHUNK
    if npad:
        st += [st[-1]] * npad
        x0 = np.concatenate([x0, np.repeat(x0[-1:], npad, 0)], 0)
    ntot = x0.shape[0]
    x0d = jnp.asarray(x0, jnp.float32)

    cov = build_coverage_data(probes, st[0])
    _a, _b, barg, _c, mk = make_batched_phase1_chunked(
        st[0],
        n_arcs,
        Phase1Weights(),
        (thick,),
        coverage_data=cov,
        grid_dtype=jnp.bfloat16,
        brain_sdf=brain,
    )
    args_by_chunk = [barg(st[s : s + CHUNK]) for s in range(0, ntot, CHUNK)]

    # RProp needs the cov-aware grad directly (same computation as mk's internal
    # one); build it once if any config uses it.
    vgrad_cw = None
    if any(opt == "rprop" for _, opt, _, _ in common["configs"]):
        _vo, vgrad_cw, _ar = build_cw_fns(st[0], n_arcs, cov, thick, brain)

    def fcl_all(xfull):
        out = np.full(n, np.nan)
        for i in range(n):
            v = make_fcl_validator(
                st[i], n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fbvh
            )
            s = np.asarray(v.slacks(xfull[i].astype(np.float64)))
            out[i] = float(s.min()) if s.size else 0.0
        return out

    out = {}
    for name, opt, kw, segmented in common["configs"]:
        run = make_staged_rprop(vgrad_cw, **kw) if opt == "rprop" else mk(**kw)
        t0 = time.time()
        xf = np.zeros_like(x0)
        for ci, s in enumerate(range(0, ntot, CHUNK)):
            a = args_by_chunk[ci]
            xc = x0d[s : s + CHUNK]
            if segmented:
                for _ in range(10):
                    xc = run(xc, a, lo_r, hi_r, 0.0, 50)
                for _ in range(10):
                    xc = run(xc, a, lo, hi, 1.0, 50)
            else:
                xc = run(xc, a, lo_r, hi_r, 0.0, 500)
                xc = run(xc, a, lo, hi, 1.0, 500)
            xf[s : s + CHUNK] = np.asarray(xc)
        jax.block_until_ready(jnp.asarray(xf))
        dt = time.time() - t0
        out[name] = dict(fcl=fcl_all(xf[:n]), t=dt)
        print(f"    {name:>16}: {dt:6.1f}s", flush=True)
    return list(idxs), out


def main() -> int:
    cfg, rt, probes, holes, sdf, bvh, fixtures, well, fixture_bvhs = setup()
    brain = maybe_build_brain_sdf(rt, compile_all_transforms(cfg.transforms))
    mesh = rt.asset_catalog.get_geometry("well").raw
    thick = make_thick_well_sdf(mesh, well, cone=fit_well_cone(mesh))
    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    idxs, good, top, log_s = select_sample(data)
    if LIMIT:
        idxs = idxs[:LIMIT]
    good_set = set(good)
    print(
        f"LR-schedule sweep (thick well, 5000 surf): {len(idxs)} cands, "
        f"{len(CONFIGS)} configs, lr={LR}"
    )

    enum = Enumerator(*build_or_load_atlas(), ml_margin_deg=0.0, ml_mode="greedy")
    common = dict(
        probes=probes,
        holes=holes,
        data=data,
        bvh=bvh,
        well=well,
        thick=thick,
        brain=brain,
        fixtures=fixtures,
        fixture_bvhs=fixture_bvhs,
        sdf=sdf,
        enum=enum,
        configs=CONFIGS,
    )
    by_arcs: dict[int, list[int]] = {}
    for idx in idxs:
        by_arcs.setdefault(int(data["results"][idx].n_arcs), []).append(idx)

    all_ids = []
    agg = {name: {"fcl": [], "t": 0.0} for name, _o, _, _ in CONFIGS}
    for n_arcs in sorted(by_arcs):
        g = by_arcs[n_arcs]
        print(f"  [n_arcs={n_arcs}] {len(g)} cands...", flush=True)
        ids, out = run_group(n_arcs, g, common=common)
        all_ids += ids
        for name in agg:
            agg[name]["fcl"].append(out[name]["fcl"])
            agg[name]["t"] += out[name]["t"]
    ids = np.array(all_ids)
    isgood = np.array([i in good_set for i in ids])

    results = {}
    for name in agg:
        fcl = np.concatenate(agg[name]["fcl"])
        feas = fcl >= -1e-4
        results[name] = dict(
            fcl=fcl,
            feasible=int(feas.sum()),
            winners=int((feas & isgood).sum()),
            t=agg[name]["t"],
        )
    pickle.dump(dict(ids=ids, is_good=isgood, results=results, lr=LR), open(OUT, "wb"))
    print(f"\nsaved → {OUT}\n")
    print(f"{'schedule':>16} {'feasible':>9} {'winners':>8} {'wall(s)':>9}")
    print("-" * 46)
    for name, _o, _, _ in CONFIGS:
        r = results[name]
        print(f"{name:>16} {r['feasible']:>9} {r['winners']:>8} {r['t']:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
