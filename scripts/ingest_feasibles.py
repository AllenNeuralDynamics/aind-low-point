"""Exhaustive feasible-set ingestion: FCL ALL 8908, coverage-rank the feasibles.

FCL is ~9 ms/cand, so FCL-ing the whole pool against all fixtures is ~1.3 min —
cheaper than guessing a cut-point. This gets the COMPLETE FCL-feasible set, then
computes coverage(pose) only for the feasibles and ranks them by coverage to
yield the definitive "top-N diverse feasibles". Saves the ranked feasible set.

Run:  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.ingest_feasibles
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
import time
from pathlib import Path

import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.coverage_jax import coverage_total_over_probes
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.ingest_analysis import _assign_key, _poses
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import build_coverage_data, build_fixture_sdf_data

MANUAL = 4195


def main() -> int:
    cfg = ConfigModel.from_yaml("examples/836656-config-T12.yml")
    rt = build_runtime_from_config(cfg)
    probes = [_probe_static_info(rt.plan_state, rt, n)
              for n in rt.plan_state.probes]
    holes = load_holes(Path("scratch/0283-300-04.holes.yml"))
    comp = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in comp:
        R, t = comp["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    bvh = {p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
           for p in probes}
    fx = build_fixture_sdf_data(rt)
    fbvh = {f.name: make_fcl_bvh(rt.asset_catalog.get_geometry(f.name).raw)
            for f in fx}

    rer = pickle.load(open("scratch/full_rerank_0283.pkl", "rb"))
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    recs = rer["records"]
    viol = np.array([r["viol"] for r in recs])
    # PRINCIPLE: the soft SDF / soft `viol` is for POSE OPTIMIZATION ONLY.
    # Hard feasibility and ranking must use the EXACT metric (FCL on the real
    # meshes), never the soft penalty. So FCL EVERY candidate — the soft viol
    # gates nothing. FCL is ~9 ms/cand → the full pool is ~1.3 min. (We sort by
    # viol only to order the printout / report the feasibility-vs-soft band; the
    # feasible SET and its coverage ranking do not depend on it.)
    order = np.argsort(viol)
    fcl_set = [recs[i] for i in order]
    print(f"FCL on ALL {len(fcl_set)} cands (hard feasibility; soft viol unused "
          f"as a gate)...")

    t0 = time.time()
    feas = []
    for r in fcl_set:
        c = pool["candidates"][r["idx"]]
        st = _build_probe_static(probes, holes, c.ha, c.aa, bvh_cache=bvh,
                                 sdf_by_name=None)
        v = make_fcl_validator(st, r["n_arcs"], fixtures=tuple(fx),
                               fixture_bvhs=fbvh)
        fcl = float(np.asarray(v.slacks(r["pose"])).min())
        if fcl >= -1e-4:
            feas.append((r, c, fcl))
    print(f"  {time.time()-t0:.1f}s; {len(feas)} FCL-feasible")

    # Coverage only for the feasibles.
    cov_data = None
    rows = []
    for r, c, fcl in feas:
        st = _build_probe_static(probes, holes, c.ha, c.aa, bvh_cache=bvh,
                                 sdf_by_name=None)
        if cov_data is None:
            cov_data = build_coverage_data(probes, st)
        Rs, ts, tips, mask = _poses(st, np.asarray(r["pose"], float),
                                    r["n_arcs"])
        coverage = float(coverage_total_over_probes(
            Rs, ts, tips, mask, cov_data, n_samples=41))
        rows.append(dict(idx=r["idx"], n_arcs=r["n_arcs"], viol=r["viol"],
                         coverage=coverage, fcl=fcl, key=_assign_key(c)))

    rows.sort(key=lambda x: -x["coverage"])
    print(f"\n=== {len(rows)} FCL-feasible plans, ranked by COVERAGE ===")
    print(f"{'rank':>4} {'cand':>6} {'coverage':>9} {'fcl':>7} {'viol':>8}")
    for j, x in enumerate(rows):
        tag = " <-- MANUAL" if x["idx"] == MANUAL else ""
        print(f"{j+1:>4} {x['idx']:>6} {x['coverage']:>9.3f} {x['fcl']:>+7.3f} "
              f"{x['viol']:>+8.3f}{tag}")

    keys = {tuple(x["key"]) for x in rows}
    print(f"\ndistinct (hole,arc) families: {len(keys)}/{len(rows)} "
          f"(all unique = maximally diverse)" if len(keys) == len(rows)
          else f"\ndistinct (hole,arc) families: {len(keys)}/{len(rows)}")
    mrow = next((x for x in rows if x["idx"] == MANUAL), None)
    if mrow:
        mr = [x["idx"] for x in rows].index(MANUAL) + 1
        print(f"manual #{MANUAL}: coverage {mrow['coverage']:.3f}, "
              f"coverage-rank {mr}/{len(rows)}")

    out = Path("scratch/feasibles_by_coverage.pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(rows=rows), f)
    print(f"\nsaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
