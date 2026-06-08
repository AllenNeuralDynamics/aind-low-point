"""Run the chain on cand 4195 with H1-seeded spins per probe.

For each probe, pick the H1 candidate closest to the manual spin (within
slot 180° symmetry for 4-shank, raw distance for 1-shank). Build a
phase1_x with those spin overrides on top of cand 4195's polished
ml/depth/offsets. Run Phase 1 + Phase 2 + FCL validator.

Compare:
  - Per-probe seed spin (H1 choice)  vs  manual  vs  final after chain
  - FCL feasibility of the final pose
  - Coverage at the final pose

Also runs three reference seeds for comparison:
  1. Current pkl polish (the seed-spin from augmented_phase1_x)
  2. Manual spins exactly (cheat — confirms the manual basin is reachable)
  3. H1 closest-to-manual (the actual H1 heuristic test)
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
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    Phase1Weights,
    make_phase1_objective,
)
from aind_low_point.optimization.stage3_phase2_jax import (
    Phase2Weights,
    make_phase2,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    phase1_bounds,
)
from scripts.spin_heuristic_search import (
    is_four_shank,
    spin_to_align_y_with,
)


def _wrap_deg(x: float) -> float:
    return float(((x + 180.0) % 360.0) - 180.0)


def h1_candidates_for(st, ap_deg: float, ml_deg: float) -> list[float]:
    sm_world = st.assigned_hole.slot_major_dir()
    align = spin_to_align_y_with(sm_world, ap_deg, ml_deg)
    if is_four_shank(st):
        return [_wrap_deg(align), _wrap_deg(align + 180.0)]
    return [_wrap_deg(align + k * 90.0) for k in range(4)]


def h1_closest_to(cands: list[float], target_deg: float, four_shank: bool) -> float:
    """Return the H1 candidate closest to target_deg (modulo 180° for
    4-shank's slot symmetry)."""
    if four_shank:
        # 4-shank: snap-to-nearest considering 180° equivalence
        errs = [
            min(abs(_wrap_deg(target_deg - c)), abs(_wrap_deg(target_deg - c - 180.0)))
            for c in cands
        ]
    else:
        errs = [abs(_wrap_deg(target_deg - c)) for c in cands]
    i = int(np.argmin(errs))
    return cands[i]


def build_y(
    arc_aps: np.ndarray,
    n_arcs: int,
    mls: np.ndarray,
    spins_deg: np.ndarray,
    offsets_R: np.ndarray,
    offsets_A: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    n_probes = len(mls)
    y = np.zeros(n_arcs + PHASE1_PER_PROBE_VARS * n_probes, dtype=np.float64)
    y[:n_arcs] = arc_aps
    for i in range(n_probes):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        rad = np.deg2rad(spins_deg[i])
        y[off + 0] = mls[i]
        y[off + 1] = np.cos(rad)
        y[off + 2] = np.sin(rad)
        y[off + 3] = offsets_R[i]
        y[off + 4] = offsets_A[i]
        y[off + 5] = depths[i]
    return y


def extract_spins(y: np.ndarray, n_arcs: int, n_probes: int) -> np.ndarray:
    out = np.zeros(n_probes, dtype=np.float64)
    for i in range(n_probes):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        out[i] = np.degrees(np.arctan2(y[off + 2], y[off + 1]))
    return out


def run_chain(
    y0: np.ndarray,
    statics: list,
    n_arcs: int,
    coverage_data,
    fixtures,
    validator,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    n_probes = len(statics)
    bounds = phase1_bounds(n_arcs, n_probes)
    p1_fun, p1_jac = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=Phase1Weights(),
    )
    r1 = minimize(
        p1_fun,
        y0,
        jac=p1_jac,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=80, ftol=1e-5, gtol=1e-5),
    )
    x1 = np.asarray(r1.x, dtype=np.float64)
    p2 = make_phase2(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=Phase2Weights(min_clearance_mm=0.3),
    )
    r2 = minimize(
        p2["fun"],
        x1,
        jac=p2["jac"],
        method="trust-constr",
        bounds=bounds,
        constraints=p2["constraints_nlc"],
        options=dict(
            maxiter=80, xtol=1e-6, gtol=1e-5, initial_tr_radius=1.0, verbose=0
        ),
    )
    x2 = np.asarray(r2.x, dtype=np.float64)
    s_fcl = validator.slacks(x2)
    feas = bool(s_fcl.size == 0 or s_fcl.min() >= -1e-4)
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
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=cov_w,
    )
    cov = -float(cov_fn(x2))
    return x2, s_fcl, feas, cov


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
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
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

    with open("examples/836656-config-T12.plan.yml") as f:
        plan_data = yaml.safe_load(f)
    manual_spins = {n: float(p["spin"]) for n, p in plan_data["probes"].items()}

    with open("/tmp/full_polish_unitcircle.pkl", "rb") as f:
        data = pickle.load(f)
    cand_idx = 4195
    cand = data["candidates"][cand_idx]
    jc = data["results"][cand_idx]
    statics = _build_probe_static(
        probes,
        holes,
        cand.ha,
        cand.aa,
        bvh_cache=bvh_cache,
        sdf_by_name=sdf_by_name,
    )
    n_arcs = jc.n_arcs
    n_probes = len(statics)

    coverage_data = build_coverage_data(probes, statics)
    validator = make_fcl_validator(
        statics,
        n_arcs,
        fixtures=fixtures,
        fixture_bvhs=fixture_bvhs,
    )
    probe_kind_by_name = {p.name: p.kind for p in probes}

    # Augmented warm-start (the pkl's polished seed for cand 4195)
    x_aug = np.asarray(
        data["augmented_phase1_x"][cand_idx],
        dtype=np.float64,
    )
    arc_aps = x_aug[:n_arcs]
    mls_aug = np.array(
        [x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i] for i in range(n_probes)]
    )
    offR_aug = np.array(
        [x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i + 3] for i in range(n_probes)]
    )
    offA_aug = np.array(
        [x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i + 4] for i in range(n_probes)]
    )
    dep_aug = np.array(
        [x_aug[n_arcs + PHASE1_PER_PROBE_VARS * i + 5] for i in range(n_probes)]
    )
    spin_aug = extract_spins(x_aug, n_arcs, n_probes)

    # Print H1 candidates per probe, and the closest-to-manual pick
    print(f"\nCand {cand_idx} — H1 candidate analysis (arc APs = {arc_aps})\n")
    h1_pick = np.zeros(n_probes)
    for i, st in enumerate(statics):
        kind = probe_kind_by_name[st.name]
        four_shank = is_four_shank(st)
        ap_i = float(arc_aps[st.arc_idx])
        ml_i = float(mls_aug[i])
        cands = h1_candidates_for(st, ap_i, ml_i)
        manual = manual_spins[st.name]
        pick = h1_closest_to(cands, manual, four_shank)
        h1_pick[i] = pick
        cs_str = ", ".join(f"{c:+6.1f}" for c in cands)
        print(
            f"  {st.name:<5} ({kind:<18}, 4S={str(four_shank):<5})  "
            f"manual={manual:+7.2f}  H1=[{cs_str}]  pick={pick:+7.2f}  "
            f"Δ(pick-manual)={_wrap_deg(pick - manual):+6.2f}°"
        )

    # Three seeds to compare
    seeds = {
        "pkl_seed": spin_aug.copy(),
        "manual": np.array([manual_spins[st.name] for st in statics]),
        "h1": h1_pick.copy(),
    }

    print("\nSeed spins side by side:")
    print(f"  {'probe':<6} {'pkl':>8} {'manual':>8} {'h1':>8}")
    for i, st in enumerate(statics):
        print(
            f"  {st.name:<6} {seeds['pkl_seed'][i]:+8.2f} "
            f"{seeds['manual'][i]:+8.2f} {seeds['h1'][i]:+8.2f}"
        )

    # Run chain from each seed
    results = {}
    for label, spins_seed in seeds.items():
        # Keep ml/offsets/depth from augmented seed; only override spin
        y0 = build_y(arc_aps, n_arcs, mls_aug, spins_seed, offR_aug, offA_aug, dep_aug)
        print(f"\n--- Running chain from {label} seed ---", flush=True)
        t0 = time.time()
        x2, s_fcl, feas, cov = run_chain(
            y0,
            statics,
            n_arcs,
            coverage_data,
            fixtures,
            validator,
        )
        wall = time.time() - t0
        final_spins = extract_spins(x2, n_arcs, n_probes)
        results[label] = (x2, s_fcl, feas, cov, final_spins, wall)
        n_viol = int((s_fcl < -1e-4).sum()) if s_fcl.size else 0
        tag = "FEAS" if feas else "FAIL"
        print(
            f"  {tag}  fcl_min={s_fcl.min():+.4f}  "
            f"n_viol={n_viol}/{s_fcl.size}  cov={cov:.2f}  "
            f"wall={wall:.1f}s"
        )

    # Compare per-probe final spins
    print("\nFinal spins per seed (Δ vs manual in parens):")
    header = (
        f"  {'probe':<6} {'manual':>8}  "
        f"{'pkl_final':>12}  {'manual_final':>14}  {'h1_final':>10}"
    )
    print(header)
    for i, st in enumerate(statics):
        manual = manual_spins[st.name]
        line = f"  {st.name:<6} {manual:+8.2f}  "
        for label in ("pkl_seed", "manual", "h1"):
            final = results[label][4][i]
            d = _wrap_deg(final - manual)
            # For 4-shank, also report 180°-equivalent distance
            if is_four_shank(st):
                d_eq = min(abs(d), abs(_wrap_deg(final - manual - 180.0)))
                line += f"  {final:+8.2f}(Δ{d_eq:+5.1f})"
            else:
                line += f"  {final:+8.2f}(Δ{d:+5.1f})"
        print(line)

    print("\nFCL summary:")
    for label, (_, s_fcl, feas, cov, _, wall) in results.items():
        n_viol = int((s_fcl < -1e-4).sum()) if s_fcl.size else 0
        tag = "FEAS" if feas else "FAIL"
        print(
            f"  {label:<10}: {tag}  fcl_min={s_fcl.min():+.4f}  "
            f"n_viol={n_viol}/{s_fcl.size}  cov={cov:.2f}  wall={wall:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
