"""Ingestion analysis over the coverage-aware rerank artifact.

For the top-K of the rerank ranking, recompute the three signals needed to
pick "top-N diverse feasibles" for Phase 2:
  - coverage(pose)  — isolates coverage from the combined objective
                      (penalties = viol + coverage)
  - FCL vs ALL fixtures (headframe+cone+well) — honest feasibility
  - (hole-assignment, arc-partition) key — for diversity

Reports: feasibles ranked by COVERAGE (the real top-N), assignment diversity
among them, and the feasibility-vs-rank curve. Saves the enriched records.

Run:  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.ingest_analysis
Env:  TOPK (default 300)
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.coverage_jax import coverage_total_over_probes
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf_jax import pose_from_optimizer_vars
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import build_coverage_data, build_fixture_sdf_data

PPV = 6
TOPK = int(_os.environ.get("TOPK", "300"))
MANUAL = 4195


def _poses(st, x, n_arcs):
    """Reconstruct (Rs (P,3,3), ts (P,3), tips (P,maxsh,3), mask (P,maxsh))
    from a Phase 1 x at this candidate's statics."""
    arc_aps = x[:n_arcs]
    Rs, ts, tips = [], [], []
    for i, s in enumerate(st):
        off = n_arcs + PPV * i
        ml, sx, sy, oR, oA, dep = x[off:off + 6]
        spin = float(np.degrees(np.arctan2(sy, sx)))
        R, tt = pose_from_optimizer_vars(
            target_LPS=jnp.asarray(s.target_LPS, jnp.float32),
            ap_deg=jnp.float32(arc_aps[s.arc_idx]), ml_deg=jnp.float32(ml),
            spin_deg=jnp.float32(spin), offset_R_mm=jnp.float32(oR),
            offset_A_mm=jnp.float32(oA), past_target_mm=jnp.float32(dep),
            recording_center_local=jnp.asarray(s.pivot_local, jnp.float32))
        Rs.append(R)
        ts.append(tt)
        tips.append(np.asarray(s.shank_tips_local, np.float32))
    P = len(st)
    maxsh = max(len(t) for t in tips)
    tips_p = np.zeros((P, maxsh, 3), np.float32)
    mask_p = np.zeros((P, maxsh), np.float32)
    for i in range(P):
        tips_p[i, :len(tips[i])] = tips[i]
        mask_p[i, :len(tips[i])] = 1.0
    return jnp.stack(Rs), jnp.stack(ts), jnp.asarray(tips_p), jnp.asarray(mask_p)


def _assign_key(cand):
    ha = tuple(sorted(cand.ha.probe_to_hole.items()))
    aa = tuple(sorted(cand.aa.probe_to_arc_idx.items()))
    return (ha, aa)


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
    fixtures = build_fixture_sdf_data(rt)
    fbvh = {f.name: make_fcl_bvh(rt.asset_catalog.get_geometry(f.name).raw)
            for f in fixtures}

    rer = pickle.load(open("scratch/full_rerank_0283.pkl", "rb"))
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    records = rer["records"]
    cov_data = None

    K = min(TOPK, len(records))
    print(f"enriching top-{K} (coverage + all-fixture FCL + assignment)...")
    rows = []
    for k in range(K):
        r = records[k]
        cand = pool["candidates"][r["idx"]]
        st = _build_probe_static(probes, holes, cand.ha, cand.aa,
                                 bvh_cache=bvh, sdf_by_name=None)
        if cov_data is None:
            cov_data = build_coverage_data(probes, st)
        Rs, ts, tips, mask = _poses(st, np.asarray(r["pose"], float), r["n_arcs"])
        coverage = float(coverage_total_over_probes(
            Rs, ts, tips, mask, cov_data, n_samples=41))
        v = make_fcl_validator(st, r["n_arcs"], fixtures=tuple(fixtures),
                               fixture_bvhs=fbvh)
        fcl = float(np.asarray(v.slacks(r["pose"])).min())
        rows.append(dict(
            idx=r["idx"], rank=k, n_arcs=r["n_arcs"], viol=r["viol"],
            coverage=coverage, penalties=r["viol"] + coverage, fcl=fcl,
            feasible=fcl >= -1e-4, key=_assign_key(cand)))

    feas = [x for x in rows if x["feasible"]]
    print(f"\nFCL-feasible in top-{K}: {len(feas)}/{K}")

    # (1) feasibles ranked by COVERAGE (high = better)
    feas_by_cov = sorted(feas, key=lambda x: -x["coverage"])
    print("\n=== feasibles ranked by COVERAGE ===")
    print(f"{'covrank':>7} {'cand':>6} {'sortrank':>8} {'coverage':>9} "
          f"{'penalty':>8} {'fcl':>7}")
    for j, x in enumerate(feas_by_cov[:25]):
        tag = " <-- MANUAL" if x["idx"] == MANUAL else ""
        print(f"{j+1:>7} {x['idx']:>6} {x['rank']+1:>8} {x['coverage']:>9.3f} "
              f"{x['penalties']:>8.3f} {x['fcl']:>+7.3f}{tag}")
    man = next((x for x in feas if x["idx"] == MANUAL), None)
    if man:
        mrank = [x["idx"] for x in feas_by_cov].index(MANUAL) + 1
        print(f"manual #{MANUAL}: coverage {man['coverage']:.3f}, "
              f"coverage-rank {mrank}/{len(feas)} among feasibles")

    # (2) diversity among top-coverage feasibles
    print("\n=== diversity (distinct hole+arc assignment) ===")
    for N in (10, 20, len(feas_by_cov)):
        keys = {tuple(x["key"]) for x in feas_by_cov[:N]}
        holes_sets = {x["key"][0] for x in feas_by_cov[:N]}
        print(f"  top-{N:>3} feasibles by coverage: {len(keys)} distinct "
              f"(ha,arc) families, {len(holes_sets)} distinct hole-tuples")

    # (3) feasibility vs rank (deciles)
    print("\n=== feasibility vs sort-rank ===")
    band = max(1, K // 10)
    for b in range(0, K, band):
        chunk = rows[b:b + band]
        nf = sum(1 for x in chunk if x["feasible"])
        print(f"  rank {b+1:>4}-{b+len(chunk):>4}: {nf}/{len(chunk)} feasible")

    out = Path("scratch/ingest_top_enriched.pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(rows=rows, topk=K), f)
    print(f"\nsaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
