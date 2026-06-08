"""Manual candidate through the full chain from the MRV (emit_seed) seed.

Takes ONE candidate (default the manual, #4195 in ``full_polish_0283.pkl``) and
runs it through the staged stack, seeded entirely from the MRV enumerator's
joint seed (``emit_seed``: convex isotonic arc-AP + MRV/CSP ML+spin anchor pick)
rather than the production ``cand.ml_seed`` / atlas-centroid warm start:

  0. MRV seed              : arc_aps + ml + spin from ``emit_seed`` on the
                             candidate's (probe→hole, probe→arc) decision.
  1. spin restore          : round-robin basin search (8 spins), seeded from the
                             MRV ml + MRV spin, well fixture in the objective.
  2. ADAM restricted       : reduced DOF — coverage OFF, offsets/depth PINNED to
                             0 (arc/ml/spin free), clearance-first.  STAGE1 steps.
  3. ADAM full             : full DOF — coverage ON, all DOFs free.  STAGE2 steps.
  4. trust-constr Phase 2  : hard feasibility constraints + coverage (the
                             production handoff polish), seeded from ADAM-full.

Brain-containment is ON by default wherever the config has a brain asset (tips
must stay inside the brain); BRAIN=0 disables it. Reports FCL min-slack,
coverage, and final spins-vs-manual at each stage, so we can see whether the MRV
seed reaches a plausible (collision-free, in-brain) geometry without the
production L-BFGS reduced polish.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.manual_mrv_chain
Env:  IDX=4195  STAGE1=500  STAGE2=500  S1_WELL=1  BRAIN=1  P2_ITER=80
      MIN_CLEAR=0.3  RESTORE_ROUNDS=4
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml
from scipy.optimize import minimize

from aind_low_point.optimization.arc_first_principled import emit_seed
from aind_low_point.optimization.batched_objective import (
    make_batched_reduced_objective,
)
from aind_low_point.optimization.batched_spin_restore import (
    make_batched_spin_restore_partial,
)
from aind_low_point.optimization.batched_static import build_batched_probe_static
from aind_low_point.optimization.joint_rerank import JointWeights, _build_probe_static
from aind_low_point.optimization.stage3_phase1_jax import (
    Phase1Weights,
    make_phase1_objective,
)
from aind_low_point.optimization.stage3_phase2_jax import Phase2Weights, make_phase2
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.arc_first_mrv import (
    MIN_ARC_AP_SEP_DEG,
    MIN_ML_SEP_DEG,
    Enumerator,
    build_or_load_atlas,
)
from scripts.restore_well_adam_manual import (
    PPV,
    build_adam_kernel,
    setup,
    spins_deg_from_reduced,
)
from scripts.run_phase1_sample import (
    build_coverage_data,
    maybe_build_brain_sdf,
    phase1_bounds,
)
from scripts.test_h1_chain_cand4195 import build_y, extract_spins

IDX = int(_os.environ.get("IDX", "4195"))
# OPT: unconstrained-stage minimizer. "lbfgs" (default) = scipy L-BFGS-B on the
# same JAX (fun, jac); "adam" = the batched projected-ADAM kernel. While we're
# diagnosing the basin pathology, L-BFGS (quasi-Newton) is the cleaner tool.
OPT = _os.environ.get("OPT", "lbfgs").lower()
# L-BFGS maxiter per stage (quasi-Newton needs far fewer than ADAM's step count);
# STAGE1/STAGE2 are the ADAM step counts when OPT=adam.
ITER1 = int(_os.environ.get("ITER1", "80"))
ITER2 = int(_os.environ.get("ITER2", "80"))
STAGE1 = int(_os.environ.get("STAGE1", "500"))
STAGE2 = int(_os.environ.get("STAGE2", "500"))
S1_WELL = _os.environ.get("S1_WELL", "1") == "1"
BRAIN = _os.environ.get("BRAIN", "1") == "1"
P2_ITER = int(_os.environ.get("P2_ITER", "80"))
MIN_CLEAR = float(_os.environ.get("MIN_CLEAR", "0.3"))
RESTORE_ROUNDS = int(_os.environ.get("RESTORE_ROUNDS", "4"))
# N_SPINS: spin-basin resolution for the round-robin restore (8 = 45° spacing;
# raise to widen / refine the basins explored).
N_SPINS = int(_os.environ.get("N_SPINS", "16"))
# SEED: spin seed source. "manual" = take spins straight from the manual plan
# (no restore — the basin-reachability test); "mrv" = MRV emit_seed spin +
# round-robin spin restore.
SEED = _os.environ.get("SEED", "manual").lower()
FCL_TOL = -1e-4


def reduced_bounds(n_arcs, K):
    """Phase-1 bounds with offsets (off_R, off_A) and depth pinned to (0, 0)."""
    b = list(phase1_bounds(n_arcs, K))
    for k in range(K):
        for off in (3, 4, 5):
            b[n_arcs + PPV * k + off] = (0.0, 0.0)
    return b


def mrv_seed(cand, enum, n_arcs):
    """The joint MRV seed for the candidate's (probe→hole, probe→arc) decision.

    Builds one ``emit_seed`` arc per arc index — members ``(probe_idx, hole_id,
    name)``, AP window = member AP-envelope intersection, desired AP = the arc
    centroid — and returns ``(arc_aps[n_arcs], ml_seed, spin_seed, min_ml_gap)``
    with ``arc_aps`` indexed by arc, and the ml/spin dicts keyed by probe name.
    """
    p2h = cand.ha.probe_to_hole
    p2a = cand.aa.probe_to_arc_idx
    centroids = cand.aa.arc_centroids_deg
    arcs: dict[int, list[tuple[int, int, str]]] = {}
    for name, ai in p2a.items():
        p = enum.names.index(name)
        arcs.setdefault(ai, []).append((p, p2h[name], name))
    arc_order = sorted(arcs)
    seed_arcs = []
    for ai in arc_order:
        members = arcs[ai]
        lo = max(enum.arr.ap_min_max[(p, h)][0] for p, h, _ in members)
        hi = min(enum.arr.ap_min_max[(p, h)][1] for p, h, _ in members)
        desired = float(centroids[ai]) if ai < len(centroids) else 0.5 * (lo + hi)
        seed_arcs.append(
            {"members": members, "ap_lo": lo, "ap_hi": hi, "ap_desired": desired}
        )
    res = emit_seed(
        seed_arcs,
        enum.arr,
        min_arc_ap_sep_deg=MIN_ARC_AP_SEP_DEG,
        min_ml_sep_deg=MIN_ML_SEP_DEG,
    )
    if res is None:
        raise RuntimeError(f"emit_seed returned None for cand #{IDX} (no anchors)")
    aps_list, ml_seed, spin_seed, min_gap = res
    arc_aps = np.zeros(n_arcs)
    for ai, ap in zip(arc_order, aps_list):
        arc_aps[ai] = float(ap)
    return arc_aps, ml_seed, spin_seed, float(min_gap)


def restore_from_mrv(
    cand, probes, holes, sdf_by_name, n_arcs, well, arc_aps, ml_seed, spin_seed
):
    """Round-robin spin restore for one candidate seeded from the MRV ml + spin
    (not the production cand.ml_seed). Returns per-probe restored spin degrees."""
    K = len(probes)
    bs = build_batched_probe_static(
        [(cand.ha, cand.aa)],
        probes,
        holes,
        n_arcs=n_arcs,
        sdf_by_name=sdf_by_name,
        head_pitch_deg=0.0,
    )
    weights = JointWeights()
    fixtures = (well,) if S1_WELL else ()
    restore = make_batched_spin_restore_partial(
        bs, weights, n_spins=N_SPINS, n_rounds=RESTORE_ROUNDS, fixtures=fixtures
    )
    obj_batched, _ = make_batched_reduced_objective(bs, weights, fixtures)
    varying = obj_batched.extract_arrays(bs)

    # Reduced layout: [arc_aps (n_arcs), then per probe (ml, cos spin, sin spin)].
    y0 = np.zeros(n_arcs + 3 * K, np.float32)
    y0[:n_arcs] = arc_aps
    for k, p in enumerate(probes):
        sp = np.deg2rad(float(spin_seed.get(p.name, 0.0)))
        y0[n_arcs + 3 * k] = float(ml_seed.get(p.name, 0.0))
        y0[n_arcs + 3 * k + 1] = float(np.cos(sp))
        y0[n_arcs + 3 * k + 2] = float(np.sin(sp))
    y_r = restore(jnp.asarray(y0[None, :]), *varying)
    y_r.block_until_ready()
    return spins_deg_from_reduced(np.asarray(y_r[0], np.float64), n_arcs, K)


def _wrap_deg(x: float) -> float:
    return float(((x + 180.0) % 360.0) - 180.0)


def unconstrained(
    st,
    n_arcs,
    K,
    x0,
    *,
    coverage_data,
    well_obj,
    brain_sdf,
    bounds,
    lbfgs_iter,
    adam_steps,
):
    """One unconstrained stage. OPT=lbfgs → scipy L-BFGS-B on the JAX (fun, jac);
    OPT=adam → the batched projected-ADAM kernel. Same objective either way."""
    fixtures = () if well_obj is None else (well_obj,)
    if OPT == "adam":
        ev = build_adam_kernel(
            st,
            n_arcs,
            K,
            well_obj,
            coverage_data=coverage_data,
            brain_sdf=brain_sdf,
            bounds=bounds,
            steps=adam_steps,
        )
        return ev([x0])[1][0].astype(np.float64)
    fun, jac = make_phase1_objective(
        st,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=Phase1Weights(),
        brain_sdf=brain_sdf,
    )
    r = minimize(
        fun,
        x0,
        jac=jac,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=lbfgs_iter, ftol=1e-5, gtol=1e-5),
    )
    return np.asarray(r.x, np.float64)


def main() -> int:
    cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    K = len(probes)
    names = [p.name for p in probes]

    compiled = compile_all_transforms(cfg.transforms)
    brain_sdf = maybe_build_brain_sdf(rt, compiled) if BRAIN else None
    iters = f"{STAGE1}/{STAGE2}" if OPT == "adam" else f"{ITER1}/{ITER2}"
    print(
        f"seed={SEED}  unconstrained={OPT} ({iters})  final=trust-constr; "
        f"brain={'ON' if brain_sdf is not None else 'OFF'}; "
        f"well-in-reduced={S1_WELL}; p2_iter={P2_ITER}; min_clear={MIN_CLEAR}"
    )

    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    cand = pool["candidates"][IDX]
    n_arcs = int(pool["results"][IDX].n_arcs)
    with open("examples/836656-config-T12.plan.yml") as f:
        _plan = yaml.safe_load(f)
    manual_spins = {n: float(p["spin"]) for n, p in _plan["probes"].items()}

    # ---- 0. MRV seed ----------------------------------------------------
    enum = Enumerator(*build_or_load_atlas(), ml_margin_deg=0.0, ml_mode="greedy")
    arc_aps, ml_seed, spin_seed, min_gap = mrv_seed(cand, enum, n_arcs)
    flag = " (best-effort, <16°)" if min_gap < MIN_ML_SEP_DEG else ""
    print(f"\nMRV seed (#{IDX}, n_arcs={n_arcs}): min_ml_gap={min_gap:.2f}°{flag}")
    print(f"  arc_aps = {np.round(arc_aps, 2).tolist()}")
    print(f"  {'probe':<6} {'ml_seed':>9} {'spin_seed':>10} {'manual':>9}")
    for n in names:
        print(
            f"  {n:<6} {ml_seed.get(n, 0.0):>9.2f} {spin_seed.get(n, 0.0):>10.2f} "
            f"{manual_spins.get(n, float('nan')):>9.2f}"
        )

    st = _build_probe_static(
        probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
    )
    cov_data = build_coverage_data(probes, st)
    validator = make_fcl_validator(
        st, n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fixture_bvhs
    )

    # coverage-only objective (all feasibility lambdas zeroed) for reporting
    cov_w = Phase1Weights(
        lambda_thread=0.0,
        lambda_clearance=0.0,
        lambda_kinematic=0.0,
        lambda_bounds=0.0,
        lambda_clearance_fixture=0.0,
        lambda_margin_clear=0.0,
        lambda_margin_thread=0.0,
        lambda_margin_clear_fixture=0.0,
        lambda_unit_circle=0.0,
    )
    cov_fn, _ = make_phase1_objective(
        st, n_arcs, coverage_data=cov_data, fixtures=tuple(fixtures), weights=cov_w
    )

    def decode(x):
        """Full decoded pose: arc APs + per-probe (ml, spin°, off_R, off_A, depth)."""
        x = np.asarray(x, np.float64)
        rows = []
        for i, n in enumerate(names):
            o = n_arcs + PPV * i
            rows.append(
                dict(
                    name=n,
                    ml=float(x[o]),
                    spin=float(np.degrees(np.arctan2(x[o + 2], x[o + 1]))),
                    off_R=float(x[o + 3]),
                    off_A=float(x[o + 4]),
                    depth=float(x[o + 5]),
                )
            )
        return x[:n_arcs].tolist(), rows

    stage_recs: list = []

    def report(label, x):
        x = np.asarray(x, np.float64)
        s = np.asarray(validator.slacks(x))
        fcl_min = float(s.min()) if s.size else 0.0
        feas = bool(s.size == 0 or fcl_min >= FCL_TOL)
        cov = -float(cov_fn(x))
        n_v = int((s < FCL_TOL).sum()) if s.size else 0
        vp = (
            validator.violating_pairs(x, margin=FCL_TOL)
            if hasattr(validator, "violating_pairs")
            else []
        )
        aps, poses = decode(x)
        print(
            f"  {label:<16} fcl_min={fcl_min:>+7.3f}  n_viol={n_v}/{s.size}  "
            f"cov={cov:>6.2f}  {'FEAS' if feas else 'infes'}"
        )
        if vp:
            print("      viol: " + ", ".join(f"{nm}={sl:+.3f}" for nm, sl in vp))
        stage_recs.append(
            dict(
                label=label,
                x=x.copy(),
                arc_aps=aps,
                poses=poses,
                slacks=s.copy(),
                fcl_min=fcl_min,
                feas=feas,
                cov=cov,
                viol=[(nm, float(sl)) for nm, sl in vp],
            )
        )
        return feas, fcl_min, cov

    # ---- 1. spin seed ---------------------------------------------------
    zero = np.zeros(K)
    mls = np.array([ml_seed.get(n, 0.0) for n in names])
    if SEED == "manual":
        spins0 = np.array([manual_spins[n] for n in names])
        print(f"\nspin seed = MANUAL plan: {[round(float(s), 1) for s in spins0]}")
    else:
        t0 = time.time()
        spins0 = np.asarray(
            restore_from_mrv(
                cand,
                probes,
                holes,
                sdf_by_name,
                n_arcs,
                well,
                arc_aps,
                ml_seed,
                spin_seed,
            )
        )
        print(
            f"\nspin restore ({time.time() - t0:.1f}s): "
            f"{[round(float(s), 1) for s in spins0]}"
        )
    x_seed = build_y(arc_aps, n_arcs, mls, spins0, zero, zero, zero)

    print("\nstage results:")
    report("seed", x_seed)

    # ---- 2. restricted: clearance-only, offsets/depth pinned ------------
    t0 = time.time()
    x1 = unconstrained(
        st,
        n_arcs,
        K,
        x_seed,
        coverage_data=None,
        well_obj=well if S1_WELL else None,
        brain_sdf=brain_sdf,
        bounds=reduced_bounds(n_arcs, K),
        lbfgs_iter=ITER1,
        adam_steps=STAGE1,
    )
    print(f"  [{OPT}-restricted {time.time() - t0:.1f}s]")
    report(f"{OPT}-restricted", x1)

    # ---- 3. full: coverage on, all DOF free -----------------------------
    t0 = time.time()
    x2 = unconstrained(
        st,
        n_arcs,
        K,
        x1,
        coverage_data=cov_data,
        well_obj=well,
        brain_sdf=brain_sdf,
        bounds=phase1_bounds(n_arcs, K),
        lbfgs_iter=ITER2,
        adam_steps=STAGE2,
    )
    print(f"  [{OPT}-full {time.time() - t0:.1f}s]")
    report(f"{OPT}-full", x2)

    # ---- 4. trust-constr Phase 2 ----------------------------------------
    t0 = time.time()
    p2 = make_phase2(
        st,
        n_arcs,
        coverage_data=cov_data,
        fixtures=tuple(fixtures),
        weights=Phase2Weights(min_clearance_mm=MIN_CLEAR),
    )
    r3 = minimize(
        p2["fun"],
        x2,
        jac=p2["jac"],
        method="trust-constr",
        bounds=phase1_bounds(n_arcs, K),
        constraints=p2["constraints_nlc"],
        options=dict(
            maxiter=P2_ITER, xtol=1e-6, gtol=1e-5, initial_tr_radius=1.0, verbose=0
        ),
    )
    x3 = np.asarray(r3.x, np.float64)
    print(f"  [trust-constr {time.time() - t0:.1f}s]")
    feas3, fcl3, cov3 = report("trust-constr", x3)

    # ---- final spins vs manual -----------------------------------------
    final = extract_spins(x3, n_arcs, K)
    print("\nfinal spins vs manual (Δ wrapped to ±180°):")
    print(f"  {'probe':<6} {'manual':>8} {'final':>8} {'Δ':>7}")
    for i, n in enumerate(names):
        d = _wrap_deg(final[i] - manual_spins.get(n, 0.0))
        print(f"  {n:<6} {manual_spins.get(n, 0.0):>8.2f} {final[i]:>8.2f} {d:>+7.1f}")

    print(
        f"\n=== #{IDX}: final {'FEASIBLE' if feas3 else 'INFEASIBLE'}  "
        f"fcl_min={fcl3:+.3f}  cov={cov3:.2f} ==="
    )
    out = Path(_os.environ.get("OUT", "scratch/manual_mrv_chain.pkl"))
    with open(out, "wb") as f:
        pickle.dump(
            dict(
                idx=IDX,
                n_arcs=n_arcs,
                names=names,
                seed_mode=SEED,
                opt=OPT,
                min_ml_gap=min_gap,
                arc_aps=arc_aps,
                ml_seed=ml_seed,
                spin_seed=spin_seed,
                seed_spins=spins0,
                manual_spins=manual_spins,
                # full decoded pose + slacks + violating pairs per stage
                stages=stage_recs,
                x_seed=x_seed,
                x1=x1,
                x2=x2,
                x3=x3,
                feas=feas3,
                fcl_min=fcl3,
                cov=cov3,
                brain=brain_sdf is not None,
            ),
            f,
        )
    print(f"saved → {out}  ({len(stage_recs)} stage records w/ full pose)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
