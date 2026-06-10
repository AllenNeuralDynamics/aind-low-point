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

import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.coverage_jax import (
    coverage_per_probe_over_probes,
)
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.optimizer_vars import _poses
from aind_low_point.optimization.pipeline.phase1_geometry import (
    build_coverage_data,
    build_fixture_sdf_data,
)
from aind_low_point.optimization.pipeline.probe_setup import (
    _probe_static_info,
    _transform_holes,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms

TOPK = int(_os.environ.get("TOPK", "300"))
MANUAL = 4195


def _assign_key(cand):
    ha = tuple(sorted(cand.ha.probe_to_hole.items()))
    aa = tuple(sorted(cand.aa.probe_to_arc_idx.items()))
    return (ha, aa)


def main() -> int:
    cfg = ConfigModel.from_yaml("examples/836656-config-T12.yml")
    rt = build_runtime_from_config(cfg)
    probes = [_probe_static_info(rt.plan_state, rt, n) for n in rt.plan_state.probes]
    holes = load_holes(Path("scratch/0283-300-04.holes.yml"))
    comp = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in comp:
        R, t = comp["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    bvh = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fixtures = build_fixture_sdf_data(rt)
    fbvh = {
        f.name: make_fcl_bvh(rt.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }

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
        st = _build_probe_static(
            probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=None
        )
        if cov_data is None:
            cov_data = build_coverage_data(probes, st)
        Rs, ts, tips, mask = _poses(st, np.asarray(r["pose"], float), r["n_arcs"])
        cov_pp = np.asarray(
            coverage_per_probe_over_probes(Rs, ts, tips, mask, cov_data, n_samples=41),
            float,
        )
        coverage = float(cov_pp.sum())
        names = [s.name for s in st]
        v = make_fcl_validator(
            st, r["n_arcs"], fixtures=tuple(fixtures), fixture_bvhs=fbvh
        )
        fcl = float(np.asarray(v.slacks(r["pose"])).min())
        rows.append(
            dict(
                idx=r["idx"],
                rank=k,
                n_arcs=r["n_arcs"],
                viol=r["viol"],
                coverage=coverage,
                penalties=r["viol"] + coverage,
                fcl=fcl,
                feasible=fcl >= -1e-4,
                key=_assign_key(cand),
                cov_pp=dict(zip(names, cov_pp)),
            )
        )

    feas = [x for x in rows if x["feasible"]]
    print(f"\nFCL-feasible in top-{K}: {len(feas)}/{K}")

    # (1) feasibles ranked by COVERAGE (high = better)
    feas_by_cov = sorted(feas, key=lambda x: -x["coverage"])
    print("\n=== feasibles ranked by COVERAGE ===")
    print(
        f"{'covrank':>7} {'cand':>6} {'sortrank':>8} {'coverage':>9} "
        f"{'penalty':>8} {'fcl':>7}"
    )
    for j, x in enumerate(feas_by_cov[:25]):
        tag = " <-- MANUAL" if x["idx"] == MANUAL else ""
        print(
            f"{j + 1:>7} {x['idx']:>6} {x['rank'] + 1:>8} {x['coverage']:>9.3f} "
            f"{x['penalties']:>8.3f} {x['fcl']:>+7.3f}{tag}"
        )
    man = next((x for x in feas if x["idx"] == MANUAL), None)
    if man:
        mrank = [x["idx"] for x in feas_by_cov].index(MANUAL) + 1
        print(
            f"manual #{MANUAL}: coverage {man['coverage']:.3f}, "
            f"coverage-rank {mrank}/{len(feas)} among feasibles"
        )

    # (1b) PER-PROBE coverage breakout — exposes the "sacrifice one probe for
    # total" pattern. Columns = probes (probe order); ``min`` flags the worst
    # probe; ``min/mean`` is a fairness ratio (1.0 = even, ->0 = one starved).
    if feas:
        pnames = list(feas[0]["cov_pp"].keys())
        print("\n=== per-probe coverage (top feasibles by total) ===")
        hdr = " ".join(f"{nm:>7}" for nm in pnames)
        print(f"{'cand':>6} {'total':>7} {hdr} {'min':>7} {'min/mean':>8}")

        def _row(x, tag=""):
            vals = [x["cov_pp"][nm] for nm in pnames]
            mn, mean = min(vals), sum(vals) / len(vals)
            cells = " ".join(f"{v:>7.3f}" for v in vals)
            worst = pnames[int(np.argmin(vals))]
            print(
                f"{x['idx']:>6} {x['coverage']:>7.3f} {cells} "
                f"{mn:>7.3f} {mn / mean if mean else 0:>8.2f}  worst={worst}{tag}"
            )

        for x in feas_by_cov[:15]:
            _row(x, tag=" <-- MANUAL" if x["idx"] == MANUAL else "")
        if man and man["idx"] not in {x["idx"] for x in feas_by_cov[:15]}:
            _row(man, tag=" <-- MANUAL")
        # Which probe is starved most often across the feasibles?
        from collections import Counter

        worst_ct = Counter(min(x["cov_pp"], key=x["cov_pp"].get) for x in feas)
        print(
            f"worst-probe frequency across {len(feas)} feasibles: "
            f"{dict(worst_ct.most_common())}"
        )

    # (2) diversity among top-coverage feasibles
    print("\n=== diversity (distinct hole+arc assignment) ===")
    for N in (10, 20, len(feas_by_cov)):
        keys = {tuple(x["key"]) for x in feas_by_cov[:N]}
        holes_sets = {x["key"][0] for x in feas_by_cov[:N]}
        print(
            f"  top-{N:>3} feasibles by coverage: {len(keys)} distinct "
            f"(ha,arc) families, {len(holes_sets)} distinct hole-tuples"
        )

    # (3) feasibility vs rank (deciles)
    print("\n=== feasibility vs sort-rank ===")
    band = max(1, K // 10)
    for b in range(0, K, band):
        chunk = rows[b : b + band]
        nf = sum(1 for x in chunk if x["feasible"])
        print(f"  rank {b + 1:>4}-{b + len(chunk):>4}: {nf}/{len(chunk)} feasible")

    out = Path("scratch/ingest_top_enriched.pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(rows=rows, topk=K), f)
    print(f"\nsaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
