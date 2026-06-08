"""(C, generalized) Does "restore-basin beats beam-diversity" hold across
candidates, or is it specific to the manual one?

For each candidate: generate the beam's spin proposals (+ heuristic facing),
seed ADAM DIRECTLY from each (NO restore) and cluster the final basins, vs the
production restore→ADAM path (one spin-restored pose). Reports, per candidate,
whether the single restored basin beats the best feasible beam→ADAM basin.

Candidates default to high-coverage stage-3 feasibles from the durable run.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.beam_adam_cluster
Env:  IDXS=4195,1035,4747,6423,697
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle

import numpy as np

from aind_low_point.optimization.coverage_jax import coverage_total_over_probes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.ingest_analysis import _poses
from scripts.restore_seed_compare import facing_seeds
from scripts.restore_well_adam_manual import (
    build_adam_kernel,
    run_restore,
    setup,
    spins_deg_from_reduced,
)
from scripts.run_phase1_sample import build_coverage_data, maybe_build_brain_sdf
from scripts.spin_heuristic_search import (
    beam_search_assignments,
    build_coupling_graph,
    per_probe_spin_candidates,
)

IDXS = [int(x) for x in _os.environ.get("IDXS", "4195,1035,4747,6423,697").split(",")]
BEAM_B = 64
CLUSTER_DEG = 25.0


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _bin(sp):
    return tuple(int(round(_wrap(s) / CLUSTER_DEG)) for s in sp)


def with_spins(base_x, spins_deg, n_arcs, K):
    """Override only the (sx, sy) of a Phase-1 x; keep ml/offsets/depth/arc."""
    x = np.asarray(base_x, np.float64).copy()
    for k in range(K):
        off = n_arcs + 6 * k
        s = np.deg2rad(spins_deg[k])
        x[off + 1] = np.cos(s)
        x[off + 2] = np.sin(s)
    return x.astype(np.float32)


def run_cand(
    idx,
    *,
    rt,
    probes,
    holes,
    sdf_by_name,
    bvh,
    fixtures,
    well,
    fixture_bvhs,
    pool,
    brain_sdf,
    probe_kind,
    mesh_by_kind,
    names,
    K,
):
    cand = pool["candidates"][idx]
    n_arcs = int(pool["results"][idx].n_arcs)
    st = _build_probe_static(
        probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
    )
    cov_data = build_coverage_data(probes, st)
    adam_eval = build_adam_kernel(st, n_arcs, K, well, cov_data, brain_sdf=brain_sdf)
    v = make_fcl_validator(
        st, n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fixture_bvhs
    )

    def cov(x):
        Rs, ts, tips, mask = _poses(st, x, n_arcs)
        return float(coverage_total_over_probes(Rs, ts, tips, mask, cov_data, 41))

    # Seed from the durable L-BFGS base (refined ml/offsets/depth); vary spin.
    base = np.asarray(pool["augmented_phase1_x"][idx], np.float64)
    arc_aps = base[:n_arcs]
    mls = np.array([base[n_arcs + 6 * k] for k in range(K)])
    target_LPS = np.array([s.target_LPS for s in st])
    seed = {i: float(cand.spin_seed.get(p.name, 0.0)) for i, p in enumerate(probes)}

    coupling, _t, _c = build_coupling_graph(
        st, arc_aps, mls, target_LPS, mesh_by_kind, probe_kind
    )
    cands = per_probe_spin_candidates(
        st, coupling, target_LPS, arc_aps, mls, probe_kind, seed_spins=seed
    )
    beam = beam_search_assignments(
        st, cands, coupling, target_LPS, arc_aps, mls, probe_kind, beam_B=BEAM_B
    )
    facing = facing_seeds(rt, st, arc_aps, mls, target_LPS, probe_kind, well, names)
    spin_seeds = [np.array([_wrap(dict(a.spins)[i]) for i in range(K)]) for a in beam]
    spin_seeds.append(np.array([_wrap(facing[n]) for n in names]))

    x0 = [with_spins(base, sp, n_arcs, K) for sp in spin_seeds]
    _viol, xa = adam_eval(x0)
    n = len(spin_seeds)
    basins = {
        _bin(
            [
                float(
                    np.degrees(
                        np.arctan2(xa[b][n_arcs + 6 * k + 2], xa[b][n_arcs + 6 * k + 1])
                    )
                )
                for k in range(K)
            ]
        ): b
        for b in range(n)
    }
    feas = []
    for b in range(n):
        fcl = float(np.asarray(v.slacks(xa[b])).min())
        if fcl >= -1e-4:
            feas.append((cov(xa[b]), fcl))
    best_cov, best_fcl = max(feas) if feas else (0.0, -1.0)

    # restore (2 rounds) spins on the durable base → ADAM (one spin-restored pose)
    y2 = run_restore(
        cand, probes, holes, sdf_by_name, n_arcs, well, with_well=True, n_rounds=2
    )
    rsp = spins_deg_from_reduced(y2, n_arcs, K)
    xr = adam_eval([with_spins(base, rsp, n_arcs, K)])[1][0]
    rfcl = float(np.asarray(v.slacks(xr)).min())
    # durable base spins → ADAM (≈ what the durable rerank did)
    xd = adam_eval([base.astype(np.float32)])[1][0]
    dfcl = float(np.asarray(v.slacks(xd)).min())
    return dict(
        idx=idx,
        n_basins=len(basins),
        n_feas=len(feas),
        beam_best_cov=best_cov,
        beam_best_fcl=best_fcl,
        restore_cov=cov(xr),
        restore_fcl=rfcl,
        restore_feas=rfcl >= -1e-4,
        durable_cov=cov(xd),
        durable_fcl=dfcl,
        durable_feas=dfcl >= -1e-4,
    )


def main() -> int:
    cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    K = len(probes)
    names = [p.name for p in probes]
    probe_kind = {p.name: p.kind for p in probes}
    mesh_by_kind = {
        k: np.asarray(rt.asset_catalog.get_geometry(f"probe:{k}").raw.vertices)
        for k in set(probe_kind.values())
    }
    comp = compile_all_transforms(cfg.transforms)
    brain_sdf = maybe_build_brain_sdf(rt, comp)
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))

    print(f"cand   | restore→ADAM (1 pose)  | beam→ADAM-direct ({BEAM_B}+1 seeds)")
    print(
        f"{'idx':>5}  | {'cov':>6} {'fcl':>7} {'':>4} | "
        f"{'basins':>6} {'feas':>4} {'best_cov':>8} {'best_fcl':>8} | verdict"
    )
    for idx in IDXS:
        r = run_cand(
            idx,
            rt=rt,
            probes=probes,
            holes=holes,
            sdf_by_name=sdf_by_name,
            bvh=bvh,
            fixtures=fixtures,
            well=well,
            fixture_bvhs=fixture_bvhs,
            pool=pool,
            brain_sdf=brain_sdf,
            probe_kind=probe_kind,
            mesh_by_kind=mesh_by_kind,
            names=names,
            K=K,
        )
        dwin = r["restore_cov"] - r["beam_best_cov"]
        verdict = (
            f"restore +{dwin:.2f} cov"
            if dwin > 0.05
            else f"beam +{-dwin:.2f} cov"
            if dwin < -0.05
            else "tie"
        )
        rfeas = "FEAS" if r["restore_feas"] else "infes"
        print(
            f"{idx:>5}  | {r['restore_cov']:>6.2f} {r['restore_fcl']:>+7.3f} "
            f"{rfeas:>4} | {r['n_basins']:>6} {r['n_feas']:>4} "
            f"{r['beam_best_cov']:>8.2f} {r['beam_best_fcl']:>+8.3f} | {verdict}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
