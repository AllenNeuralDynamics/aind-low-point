"""Beam-search spin assignments for cand 4195, polish top-K, report.

Uses the H1 (corrected closed-form) + H2 (perpendicular-to-gap) candidates
and seeded current spin per probe; beam-searches over the joint assignment
ordered by coupling degree; polishes the top-K assignments through
Phase 1 + Phase 2 + FCL validator.

Compares to the H1-only baseline (1 residual viol on VM↔CA1) from
``test_h1_chain_cand4195.py``.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
import time
from pathlib import Path

import numpy as np
import yaml

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import PHASE1_PER_PROBE_VARS
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config, save_plan_to_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.save_chain_plans import _apply_x_to_plan_state
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import build_coverage_data, build_fixture_sdf_data
from scripts.spin_heuristic_search import (
    beam_search_assignments,
    build_coupling_graph,
    is_four_shank,
    per_probe_spin_candidates,
)
from scripts.test_h1_chain_cand4195 import (
    build_y, extract_spins, run_chain,
)


def main() -> int:
    print("Loading config / probes / SDFs / fixtures...", flush=True)
    cfg = ConfigModel.from_yaml("examples/836656-config-T12.yml")
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(Path("scratch/0283-300-04.holes.yml"))
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        R, t = compiled["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    fixtures = build_fixture_sdf_data(runtime)
    bvh_cache = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fixture_bvhs = {
        f.name: make_fcl_bvh(runtime.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }

    with open("scratch/full_polish_0283.pkl", "rb") as f:
        data = pickle.load(f)
    cand_idx = 4195
    cand = data["candidates"][cand_idx]
    jc = data["results"][cand_idx]
    statics = _build_probe_static(
        probes, holes, cand.ha, cand.aa,
        bvh_cache=bvh_cache, sdf_by_name=sdf_by_name,
    )
    n_arcs = jc.n_arcs
    n_probes = len(statics)
    coverage_data = build_coverage_data(probes, statics)
    validator = make_fcl_validator(
        statics, n_arcs, fixtures=fixtures, fixture_bvhs=fixture_bvhs,
    )
    probe_kind_by_name = {p.name: p.kind for p in probes}

    # Pull arc APs, ml, offsets, depth from the augmented warm-start
    x_aug = np.asarray(
        data["augmented_phase1_x"][cand_idx], dtype=np.float64,
    )
    arc_aps = x_aug[:n_arcs]
    mls_aug = np.array([
        x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)
    ])
    offR_aug = np.array([
        x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i + 3]
        for i in range(n_probes)
    ])
    offA_aug = np.array([
        x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i + 4]
        for i in range(n_probes)
    ])
    dep_aug = np.array([
        x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i + 5]
        for i in range(n_probes)
    ])
    spin_aug = extract_spins(x_aug, n_arcs, n_probes)
    target_LPS = np.array([st.target_LPS for st in statics])

    # Build per-probe candidates (H1 + H2 + seed safety net)
    coupling = build_coupling_graph(target_LPS)
    seed_spins = {i: float(spin_aug[i]) for i in range(n_probes)}
    spin_cands = per_probe_spin_candidates(
        statics, coupling, target_LPS, arc_aps, mls_aug,
        probe_kind_by_name, seed_spins=seed_spins,
    )
    print("\nPer-probe candidate count (H1 + H2 + seed):")
    for i, st in enumerate(statics):
        four = is_four_shank(st)
        print(f"  {st.name:<5} (4S={str(four):<5})  n_cands={len(spin_cands[i])}  "
              f"spins={[f'{c:+5.1f}' for c in spin_cands[i]]}")

    # Beam search
    print("\nBeam search (B=64)...", flush=True)
    t_bs0 = time.time()
    beam = beam_search_assignments(
        statics, spin_cands, coupling, target_LPS,
        arc_aps, mls_aug, probe_kind_by_name, beam_B=64,
    )
    print(f"  beam: {len(beam)} complete assignments in {time.time()-t_bs0:.1f}s")
    print(f"  top-10 scores (lower=better, H2 alignment): "
          f"{[f'{a.score:.2f}' for a in beam[:10]]}")

    # Polish top-K via the chain
    K = 16
    to_polish = beam[:K]
    print(f"\nPolishing top-{K} via P1+P2+FCL...", flush=True)
    results = []
    for k, asg in enumerate(to_polish):
        overrides = dict(asg.spins)
        spins_seed = np.array([overrides[i] for i in range(n_probes)])
        y0 = build_y(arc_aps, n_arcs, mls_aug, spins_seed,
                     offR_aug, offA_aug, dep_aug)
        t0 = time.time()
        x2, s_fcl, feas, cov = run_chain(
            y0, statics, n_arcs, coverage_data, fixtures, validator,
        )
        wall = time.time() - t0
        n_viol = int((s_fcl < -1e-4).sum()) if s_fcl.size else 0
        viol_pairs = (
            [validator.pair_names[i] for i in range(s_fcl.size)
             if s_fcl[i] < -1e-4]
            if hasattr(validator, "pair_names") else []
        )
        tag = "FEAS" if feas else "FAIL"
        final = extract_spins(x2, n_arcs, n_probes)
        results.append((k, asg.score, feas, n_viol, float(s_fcl.min()),
                         cov, viol_pairs, final, wall))
        seed_str = " ".join(f"{s:+5.0f}" for s in spins_seed)
        final_str = " ".join(f"{f:+5.0f}" for f in final)
        print(f"  k={k:2d} score={asg.score:5.2f}  seed=[{seed_str}]  "
              f"→  [{final_str}]  {tag} viol={n_viol}  "
              f"fcl_min={s_fcl.min():+.3f}  cov={cov:5.2f}  "
              f"({wall:.1f}s)  viols={viol_pairs[:3]}")

    # Best by (feas, viol, fcl_min, cov)
    def key(r):
        _, _, feas, n_viol, fcl_min, cov, *_ = r
        # Sort: feas first, then by n_viol asc, then fcl_min desc, cov desc
        return (not feas, n_viol, -fcl_min, -cov)
    results_sorted = sorted(results, key=key)
    print("\nBest 3 by feasibility:")
    for r in results_sorted[:3]:
        k, score, feas, n_viol, fcl_min, cov, viol_pairs, final, wall = r
        tag = "FEAS" if feas else "FAIL"
        print(f"  k={k}: {tag}  viol={n_viol}  fcl_min={fcl_min:+.3f}  "
              f"cov={cov:5.2f}  viols={viol_pairs[:3]}")
        spin_str = ", ".join(f"{statics[i].name}={final[i]:+6.1f}"
                              for i in range(n_probes))
        print(f"           final spins: {spin_str}")

    # Save the FEAS results as trame-compatible config.yml files.
    out_dir = Path("examples/836656-config-T12_4195_beam")
    out_dir.mkdir(parents=True, exist_ok=True)
    feas_results = [r for r in results_sorted if r[2]]
    print(f"\nSaving {len(feas_results)} FEAS plans to {out_dir}/")
    # Reload the chain output for each FEAS k — we need x2 not just final spins.
    # Map k → x2 from `results` order matches `to_polish` order.
    # `results` was appended in to_polish order; rebuild a dict keyed by k.
    x2_by_k = {}
    for k_idx, asg in enumerate(to_polish):
        # results was appended once per k in order, so x2 lives at index k_idx
        pass
    # Actually we need x2 per k — but I didn't save x2. Re-polish the FEAS ones
    # for save. (Cheaper than threading x2 through the result tuple.)
    for save_idx, r in enumerate(feas_results, start=1):
        k, _score, feas, n_viol, fcl_min, cov, viol_pairs, final_spins, _ = r
        asg = to_polish[k]
        overrides = dict(asg.spins)
        spins_seed = np.array([overrides[i] for i in range(n_probes)])
        y0 = build_y(arc_aps, n_arcs, mls_aug, spins_seed,
                     offR_aug, offA_aug, dep_aug)
        x2, _, _, _ = run_chain(
            y0, statics, n_arcs, coverage_data, fixtures, validator,
        )
        # Fresh runtime → mutate plan_state with x2 → dump.
        cfg_local = ConfigModel.from_yaml("examples/836656-config-T12.yml")
        rt_local = build_runtime_from_config(cfg_local)
        statics_local = _build_probe_static(
            probes, holes, cand.ha, cand.aa,
            bvh_cache=bvh_cache, sdf_by_name=sdf_by_name,
        )
        _apply_x_to_plan_state(rt_local.plan_state, x2, statics_local, n_arcs)
        candidate_cfg = save_plan_to_config(rt_local.plan_state, cfg_local)
        fname = (f"plan-{save_idx:02d}-feas-k{k:02d}-"
                 f"slack{fcl_min:+.3f}-cov{cov:05.2f}.yml")
        with open(out_dir / fname, "w") as f:
            yaml.safe_dump(candidate_cfg.model_dump(mode="json"), f,
                            sort_keys=False, default_flow_style=False)
        print(f"  {fname}")

    # Compare to manual
    with open("examples/836656-config-T12.plan.yml") as f:
        plan_data = yaml.safe_load(f)
    manual_spins = {n: float(p["spin"]) for n, p in plan_data["probes"].items()}
    print("\nManual reference spins:")
    print("  " + ", ".join(f"{st.name}={manual_spins[st.name]:+6.1f}"
                            for st in statics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
