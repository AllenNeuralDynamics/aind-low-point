"""Coarse→fine surface-point schedule: 1000 surf early, 5000 surf to finish.

Surface points are the dual-rep gather count (the DRAM-bound term): 1000 is
2.5x faster per step than 5000 but a false-NEGATIVE risk (misses thin contacts).
The hypothesis: run the basin-finding bulk at 1000 (fast, FN-tolerant) and only
the FINISH at 5000 (accurate, gate-quality), recovering most of the speedup
while preserving feasibility — IF the coarse basin survives the fine finish
(the "impossible basin" risk: a coarse run that settles into a fine-infeasible
spot the fine finish can't escape).

Surf count is baked into the kernel shape, so this uses TWO compiled kernels
(1000 / 5000) with the pose handed across (x is fidelity-independent). FCL
(true mesh) gates both. Compares, on the calibration sample, all-5000 vs
coarse→fine: feasible count (did we lose winners?) and wall time.

Schedule: reduced 500 @1k → full FULL_COARSE @1k → full FULL_FINE @5k
   (FULL_COARSE + FULL_FINE = 500, matching the all-5000 baseline's full 500).

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.coarse_fine_surf
Env:  FULL_FINE=100  CHUNK=64  LIMIT=0   (LIMIT caps the sample for a quick look)
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

from aind_low_point.optimization.objectives.fcl_validator import make_fcl_validator
from aind_low_point.optimization.objectives.phase1 import Phase1Weights
from aind_low_point.optimization.objectives.probe_static import _build_probe_static
from aind_low_point.optimization.objectives.variables import build_y
from aind_low_point.optimization.pipeline.enumeration import (
    Enumerator,
    build_or_load_atlas,
)
from aind_low_point.optimization.pipeline.phase1_build import (
    make_batched_phase1_chunked,
)
from aind_low_point.optimization.pipeline.phase1_geometry import (
    build_coverage_data,
    maybe_build_brain_sdf,
    phase1_bounds,
)
from aind_low_point.optimization.pipeline.restore import setup
from aind_low_point.optimization.sdf import build_sdf_by_name
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.log_candidate_trajectories import (
    reduced_lohi,
    restore_spins_mrv,
    select_sample,
)

FULL_FINE = int(_os.environ.get("FULL_FINE", "100"))
FULL_COARSE = 500 - FULL_FINE
CHUNK = int(_os.environ.get("CHUNK", "64"))
LIMIT = int(_os.environ.get("LIMIT", "0"))
OUT = _os.environ.get("OUT", "scratch/coarse_fine_surf.pkl")


def run_group(n_arcs, idxs, *, common):
    probes, holes, data, bvh = (common[k] for k in ("probes", "holes", "data", "bvh"))
    well, brain, fixtures, fbvh = (
        common[k] for k in ("well", "brain", "fixtures", "fixture_bvhs")
    )
    K = len(probes)
    names = [p.name for p in probes]
    bounds = phase1_bounds(n_arcs, K)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    lo_r, hi_r = reduced_lohi(lo, hi, n_arcs, K)

    # one shared restore (uses the well + threading rep, not the 5000 surf pts)
    arc_l, ml_l, sp_l = restore_spins_mrv(
        n_arcs,
        idxs,
        probes=probes,
        holes=holes,
        data=data,
        sdf_by_name=common["sdf5"],
        well=well,
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

    cov = build_coverage_data(probes, st5[0])
    mk = lambda s: make_batched_phase1_chunked(  # noqa: E731
        s,
        n_arcs,
        Phase1Weights(),
        (well,),
        coverage_data=cov,
        grid_dtype=jnp.bfloat16,
        brain_sdf=brain,
    )
    _a, _b, barg5, _c, mkad5 = mk(st5[0])
    _a, _b, barg1, _c, mkad1 = mk(st1[0])
    run5, run1 = mkad5(lr=0.02), mkad1(lr=0.02)
    x0d = jnp.asarray(x0, jnp.float32)

    def fcl_all(xfull, stl):
        out = np.full(n, np.nan)
        for i in range(n):
            v = make_fcl_validator(
                stl[i], n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fbvh
            )
            s = np.asarray(v.slacks(xfull[i].astype(np.float64)))
            out[i] = float(s.min()) if s.size else 0.0
        return out

    # ---- baseline: all-5000 continuous 500 reduced + 500 full ----
    t0 = time.time()
    xb = np.zeros_like(x0)
    for s in range(0, ntot, CHUNK):
        a5 = barg5(st5[s : s + CHUNK])
        x1 = run5(x0d[s : s + CHUNK], a5, lo_r, hi_r, 0.0, 500)
        x2 = run5(x1, a5, lo, hi, 1.0, 500)
        xb[s : s + CHUNK] = np.asarray(x2)
    jax.block_until_ready(jnp.asarray(xb))
    t_base = time.time() - t0

    # ---- coarse→fine: 500 red @1k + FULL_COARSE full @1k + FULL_FINE full @5k ----
    t0 = time.time()
    xc = np.zeros_like(x0)
    for s in range(0, ntot, CHUNK):
        a1 = barg1(st1[s : s + CHUNK])
        a5 = barg5(st5[s : s + CHUNK])
        x1 = run1(x0d[s : s + CHUNK], a1, lo_r, hi_r, 0.0, 500)
        x1 = run1(x1, a1, lo, hi, 1.0, FULL_COARSE)
        x2 = run5(x1, a5, lo, hi, 1.0, FULL_FINE)
        xc[s : s + CHUNK] = np.asarray(x2)
    jax.block_until_ready(jnp.asarray(xc))
    t_cf = time.time() - t0

    fb = fcl_all(xb[:n], st5[:n])
    fc = fcl_all(xc[:n], st5[:n])
    return dict(
        ids=list(idxs), fcl_base=fb, fcl_cf=fc, t_base=t_base, t_cf=t_cf, names=names
    )


def main() -> int:
    cfg, rt, probes, holes, sdf5, bvh, fixtures, well, fixture_bvhs = setup()
    brain = maybe_build_brain_sdf(rt, compile_all_transforms(cfg.transforms))
    sdf1 = build_sdf_by_name(probes, rt, 1000)
    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    idxs, good, top, log_s = select_sample(data)
    if LIMIT:
        idxs = idxs[:LIMIT]
    good_set = set(good)
    print(
        f"coarse→fine surf: {len(idxs)} cands; schedule red500@1k + "
        f"full{FULL_COARSE}@1k + full{FULL_FINE}@5k  vs  all-5000 red500+full500"
    )

    atlas_payload = build_or_load_atlas()
    enum = Enumerator(
        atlas_payload.atlas,
        atlas_payload.probe_names,
        ml_margin_deg=0.0,
        ml_mode="greedy",
    )
    common = dict(
        probes=probes,
        holes=holes,
        data=data,
        bvh=bvh,
        well=well,
        brain=brain,
        fixtures=fixtures,
        fixture_bvhs=fixture_bvhs,
        sdf5=sdf5,
        sdf1=sdf1,
        enum=enum,
    )
    by_arcs: dict[int, list[int]] = {}
    for idx in idxs:
        by_arcs.setdefault(int(data["results"][idx].n_arcs), []).append(idx)

    ids, fb, fc, tb, tc = [], [], [], 0.0, 0.0
    for n_arcs in sorted(by_arcs):
        g = by_arcs[n_arcs]
        print(f"  [n_arcs={n_arcs}] {len(g)} cands...", flush=True)
        r = run_group(n_arcs, g, common=common)
        ids += r["ids"]
        fb.append(r["fcl_base"])
        fc.append(r["fcl_cf"])
        tb += r["t_base"]
        tc += r["t_cf"]
    fb = np.concatenate(fb)
    fc = np.concatenate(fc)
    ids = np.array(ids)
    feas_b, feas_c = fb >= -1e-4, fc >= -1e-4
    isgood = np.array([i in good_set for i in ids])

    pickle.dump(
        dict(
            ids=ids,
            fcl_base=fb,
            fcl_cf=fc,
            t_base=tb,
            t_cf=tc,
            is_good=isgood,
            full_fine=FULL_FINE,
        ),
        open(OUT, "wb"),
    )
    print(f"\nsaved → {OUT}")
    print(f"\n{'':>12} {'feasible':>10} {'winners':>9} {'wall (s)':>10}")
    print(
        f"{'all-5000':>12} {int(feas_b.sum()):>10} "
        f"{int((feas_b & isgood).sum()):>9} {tb:>10.1f}"
    )
    print(
        f"{'coarse→fine':>12} {int(feas_c.sum()):>10} "
        f"{int((feas_c & isgood).sum()):>9} {tc:>10.1f}"
    )
    gained = int((feas_c & ~feas_b).sum())
    lost = int((feas_b & ~feas_c).sum())
    print(
        f"\nhead-to-head: coarse→fine vs all-5000: GAINED {gained}, LOST {lost} "
        f"(net {gained - lost:+d});  speedup {tb / max(tc, 1e-9):.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
