"""Sweep the coarse→fine surf SPLIT to find the feasibility-vs-walltime knee.

Surface points are the dual-rep gather count: 1000 surf is ~2.5x faster/step than
5000 but FN-risky. The coarse→fine schedule runs the basin-finding bulk @1000 and
only a FINISH @5000. This sweeps the split on BOTH stages: each ``(name, rf, ff)``
config runs the reduced stage as ``(500-rf)`` coarse + ``rf`` fine steps, and the
full stage as ``(500-ff)`` coarse + ``ff`` fine steps, against an all-5000
reference. Reports feasible count + wall. The goal: the FEWEST fine steps (least
wall time) that don't lose feasibles vs all-fine. FCL (true mesh) gates all, so a
too-coarse finish just shows up as lost feasibles.

``COARSE_N`` sets the low-surf count (the appropriate value is itself worth a
sweep — lower = faster but more FN). Uses the chosen MINIMIZER (whatever the
lr_schedule_sweep crowned) on both fidelities — the split interacts with the
optimizer, so it must be the winner.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.coarse_fine_split
Env:  MINIMIZER=rprop|mrst|const  COARSE_N=1000  LIMIT=0  CHUNK=64
      SPLITS="r0_f100:0:100,r0_f50:0:50,..."  (name:reduced_fine:full_fine)
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
from scripts.batched_phase1_build import make_batched_phase1_chunked, make_staged_rprop
from scripts.coarse_fine_surf import build_sdf_by_name
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

MINIMIZER = _os.environ.get("MINIMIZER", "rprop").lower()
COARSE_N = int(
    _os.environ.get("COARSE_N", "1000")
)  # low-surf count for the coarse pass
CHUNK = int(_os.environ.get("CHUNK", "64"))
LIMIT = int(_os.environ.get("LIMIT", "0"))
LR = float(_os.environ.get("LR", "0.02"))
OUT = _os.environ.get("OUT", "scratch/coarse_fine_split.pkl")

# Per-stage fine-step counts (reduced_fine, full_fine): how many of each 500-step
# stage run at FINE (5000) surf at the END; the rest run at coarse (COARSE_N).
# Tests the split on BOTH stages, not just full. Override via SPLITS env as
# "name:rf:ff,name:rf:ff,...".
_DEFAULT_SPLITS = [
    ("r0_f500", 0, 500),  # reduced all-coarse, full all-fine
    ("r0_f200", 0, 200),
    ("r0_f100", 0, 100),
    ("r0_f50", 0, 50),
    ("r0_f0", 0, 0),  # all coarse (both stages)
    ("r100_f100", 100, 100),  # fine finish on BOTH stages
    ("r100_f50", 100, 50),
    ("r50_f50", 50, 50),
]
if _os.environ.get("SPLITS"):
    SPLITS = []
    for tok in _os.environ["SPLITS"].split(","):
        nm, rf, ff = tok.split(":")
        SPLITS.append((nm, int(rf), int(ff)))
else:
    SPLITS = _DEFAULT_SPLITS


def make_runner(mkad, vgrad_cw):
    """Return run(x0, arglist, lo, hi, cov_weight, n_steps) for the chosen
    minimizer on a given fidelity's kernel pieces."""
    if MINIMIZER == "rprop":
        return make_staged_rprop(vgrad_cw, eta0_frac=0.02, etamax_frac=0.5)
    if MINIMIZER == "mrst":
        return mkad(lr=LR, b2=0.999, schedule="moment_restart", period=50)
    return mkad(lr=LR, b2=0.999, schedule="const")


def run_group(n_arcs, idxs, *, common):
    probes, holes, data, bvh = (common[k] for k in ("probes", "holes", "data", "bvh"))
    thick, brain, fixtures, fbvh = (
        common[k] for k in ("thick", "brain", "fixtures", "fixture_bvhs")
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
        sdf_by_name=common["sdf5"],
        well=thick,
        enum=common["enum"],
    )
    st5, st1, x0_rows = [], [], []
    for i, idx in enumerate(idxs):
        cand = data["candidates"][idx]
        st5.append(
            _build_probe_static(
                probes,
                holes,
                cand.ha,
                cand.aa,
                bvh_cache=bvh,
                sdf_by_name=common["sdf5"],
            )
        )
        st1.append(
            _build_probe_static(
                probes,
                holes,
                cand.ha,
                cand.aa,
                bvh_cache=bvh,
                sdf_by_name=common["sdf1"],
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
        st5 += [st5[-1]] * npad
        st1 += [st1[-1]] * npad
        x0 = np.concatenate([x0, np.repeat(x0[-1:], npad, 0)], 0)
    ntot = x0.shape[0]
    x0d = jnp.asarray(x0, jnp.float32)

    cov = build_coverage_data(probes, st5[0])
    mk = lambda s: make_batched_phase1_chunked(  # noqa: E731
        s,
        n_arcs,
        Phase1Weights(),
        (thick,),
        coverage_data=cov,
        grid_dtype=jnp.bfloat16,
        brain_sdf=brain,
    )
    _a, _b, barg5, _c, mkad5 = mk(st5[0])
    _a, _b, barg1, _c, mkad1 = mk(st1[0])
    vg5 = vg1 = None
    if MINIMIZER == "rprop":
        vg5 = build_cw_fns(st5[0], n_arcs, cov, thick, brain)[1]
        vg1 = build_cw_fns(st1[0], n_arcs, cov, thick, brain)[1]
    run5 = make_runner(mkad5, vg5)
    run1 = make_runner(mkad1, vg1)
    a5 = [barg5(st5[s : s + CHUNK]) for s in range(0, ntot, CHUNK)]
    a1 = [barg1(st1[s : s + CHUNK]) for s in range(0, ntot, CHUNK)]

    def fcl_all(xf):
        out = np.full(n, np.nan)
        for i in range(n):
            v = make_fcl_validator(
                st5[i], n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fbvh
            )
            s = np.asarray(v.slacks(xf[i].astype(np.float64)))
            out[i] = float(s.min()) if s.size else 0.0
        return out

    def execute(schedule):
        """schedule(chunk_idx, x_chunk) -> final x_chunk; timed over all chunks."""
        t0 = time.time()
        xf = np.zeros_like(x0)
        for ci, s in enumerate(range(0, ntot, CHUNK)):
            xf[s : s + CHUNK] = np.asarray(schedule(ci, x0d[s : s + CHUNK]))
        jax.block_until_ready(jnp.asarray(xf))
        return xf, time.time() - t0

    out = {}
    # all-5000 reference (both stages all fine)
    xf, dt = execute(
        lambda ci, x: run5(
            run5(x, a5[ci], lo_r, hi_r, 0.0, 500), a5[ci], lo, hi, 1.0, 500
        )
    )
    out["allfine"] = dict(fcl=fcl_all(xf[:n]), t=dt, poses=xf[:n].copy())
    print(f"    allfine: {dt:6.1f}s", flush=True)
    # coarse→fine at each (reduced_fine, full_fine) split
    for name, rf, ff in common["splits"]:
        rc, fc = 500 - rf, 500 - ff

        def sched(ci, x, rc=rc, rf=rf, fc=fc, ff=ff):
            if rc > 0:
                x = run1(x, a1[ci], lo_r, hi_r, 0.0, rc)  # reduced coarse
            if rf > 0:
                x = run5(x, a5[ci], lo_r, hi_r, 0.0, rf)  # reduced fine finish
            if fc > 0:
                x = run1(x, a1[ci], lo, hi, 1.0, fc)  # full coarse
            if ff > 0:
                x = run5(x, a5[ci], lo, hi, 1.0, ff)  # full fine finish
            return x

        xf, dt = execute(sched)
        out[name] = dict(fcl=fcl_all(xf[:n]), t=dt, poses=xf[:n].copy())
        print(f"    {name}: {dt:6.1f}s", flush=True)
    return list(idxs), out


def main() -> int:
    cfg, rt, probes, holes, sdf5, bvh, fixtures, well, fixture_bvhs = setup()
    brain = maybe_build_brain_sdf(rt, compile_all_transforms(cfg.transforms))
    mesh = rt.asset_catalog.get_geometry("well").raw
    thick = make_thick_well_sdf(mesh, well, cone=fit_well_cone(mesh))
    sdf1 = build_sdf_by_name(probes, rt, COARSE_N)
    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    idxs, good, _t, _l = select_sample(data)
    if LIMIT:
        idxs = idxs[:LIMIT]
    good_set = set(good)
    print(
        f"coarse/fine split sweep: minimizer={MINIMIZER}, coarse_N={COARSE_N}, "
        f"{len(idxs)} cands, splits={[s[0] for s in SPLITS]}"
    )

    enum = Enumerator(*build_or_load_atlas(), ml_margin_deg=0.0, ml_mode="greedy")
    common = dict(
        probes=probes,
        holes=holes,
        data=data,
        bvh=bvh,
        thick=thick,
        brain=brain,
        fixtures=fixtures,
        fixture_bvhs=fixture_bvhs,
        sdf5=sdf5,
        sdf1=sdf1,
        enum=enum,
        splits=SPLITS,
    )
    by_arcs: dict[int, list[int]] = {}
    for idx in idxs:
        by_arcs.setdefault(int(data["results"][idx].n_arcs), []).append(idx)

    names = ["allfine"] + [s[0] for s in SPLITS]
    agg = {nm: {"fcl": [], "t": 0.0} for nm in names}
    all_ids = []
    for n_arcs in sorted(by_arcs):
        g = by_arcs[n_arcs]
        print(f"  [n_arcs={n_arcs}] {len(g)} cands...", flush=True)
        ids, out = run_group(n_arcs, g, common=common)
        all_ids += ids
        for nm in names:
            agg[nm]["fcl"].append(out[nm]["fcl"])
            agg[nm]["t"] += out[nm]["t"]
    ids = np.array(all_ids)
    isgood = np.array([i in good_set for i in ids])

    results = {}
    for nm in names:
        fcl = np.concatenate(agg[nm]["fcl"])
        feas = fcl >= -1e-4
        results[nm] = dict(
            feasible=int(feas.sum()), winners=int((feas & isgood).sum()), t=agg[nm]["t"]
        )
    pickle.dump(
        dict(ids=ids, results=results, minimizer=MINIMIZER, coarse_n=COARSE_N),
        open(OUT, "wb"),
    )
    base = results["allfine"]["t"]
    print(
        f"\nsaved → {OUT}\n\n{'split':>10} {'feasible':>9} {'winners':>8} "
        f"{'wall(s)':>9} {'speedup':>8}"
    )
    print("-" * 48)
    for nm in names:
        r = results[nm]
        print(
            f"{nm:>10} {r['feasible']:>9} {r['winners']:>8} {r['t']:>9.1f} "
            f"{base / max(r['t'], 1e-9):>7.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
