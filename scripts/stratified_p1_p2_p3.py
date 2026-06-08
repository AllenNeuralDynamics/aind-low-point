"""Stratified Phase 1 → Phase 2 → Phase 3 validation.

Samples Stage 2 polish output across ``max_violation`` bins and runs
the full Phase 1 → Phase 2 → Phase 3 chain on each. Reports per-bin
success rate to identify where Phase 1/2/3 stops being able to recover
from Stage 2 violations.

Bins:
  strict       : max_viol ≤ 0.001
  mild         : 0.001 < max_viol ≤ 0.1
  moderate     : 0.1 < max_viol ≤ 1.0
  hard         : 1.0 < max_viol ≤ 10.0
  catastrophic : max_viol > 10.0

For each cand:
  Phase 1: soft-penalty SLSQP (warm-up with offsets+depth+coverage)
  Phase 2: hard-constraint JAX SDF SLSQP
  Phase 3: hard-constraint FCL raw-mesh SLSQP (ground truth)

Final verdict: FCL-feasible iff Phase 3 ends with all FCL slacks ≥ −1e-4.

Run::
    uv run --python 3.13 python -m scripts.stratified_p1_p2_p3 \\
        examples/836656-config-T12.yml scratch/0283-300-04.holes.yml \\
        --polish-pkl /tmp/full_polish_post_sat.pkl --n-per-bin 3
"""

from __future__ import annotations

import argparse
import os as _os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    Phase1Weights,
    make_phase1_objective,
    phase1_n_vars,
    reduced_to_phase1,
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

BINS = [
    ("strict", 0.0, 0.001),
    ("mild", 0.001, 0.1),
    ("moderate", 0.1, 1.0),
    ("hard", 1.0, 10.0),
    ("catastrophic", 10.0, float("inf")),
]


# Bins for ranking by offset-polish fn (the fixture-aware fixability signal
# from augment_polish_with_offsets.py).
OFFSET_FN_BINS = [
    ("very_fixable", -float("inf"), 0.0),
    ("fixable", 0.0, 100.0),
    ("borderline", 100.0, 1_000.0),
    ("hard", 1_000.0, 10_000.0),
    ("doomed", 10_000.0, float("inf")),
]


# Bins for the violation-only signal from eval_violation_at_augmented.py
# (cleaner than the full Phase 1 fn — no coverage / margin reward bias).
VIOLATION_FN_BINS = [
    ("clean", 0.0, 10.0),  # essentially no violation
    ("light", 10.0, 100.0),  # small violations
    ("moderate", 100.0, 1_000.0),  # noticeable
    ("hard", 1_000.0, 10_000.0),  # large
    ("doomed", 10_000.0, float("inf")),  # catastrophic
]


def offset_fn_sample(offset_fn_arr, rng, n_per_bin):
    out = {}
    for name, lo, hi in OFFSET_FN_BINS:
        bucket = [
            i
            for i, v in enumerate(offset_fn_arr)
            if not np.isnan(v)
            and lo < float(v) <= hi
            or (lo == -float("inf") and float(v) <= hi)
        ]
        if not bucket:
            out[name] = []
            continue
        k = min(n_per_bin, len(bucket))
        out[name] = rng.choice(bucket, size=k, replace=False).tolist()
    return out


def violation_fn_sample(violation_fn_arr, rng, n_per_bin):
    out = {}
    for name, lo, hi in VIOLATION_FN_BINS:
        bucket = [
            i
            for i, v in enumerate(violation_fn_arr)
            if not np.isnan(v) and lo <= float(v) < hi
        ]
        if not bucket:
            out[name] = []
            continue
        k = min(n_per_bin, len(bucket))
        out[name] = rng.choice(bucket, size=k, replace=False).tolist()
    return out


@dataclass
class CandResult:
    cand_idx: int
    bin_name: str
    mv_stage2: float
    p1_fn_end: float
    p1_nit: int
    p1_wall: float
    p2_min_slack: float
    p2_n_violating: int
    p2_nit: int
    p2_wall: float
    p3_min_slack_analytic: float
    p3_min_slack_fcl: float
    p3_n_violating_fcl: int
    p3_nit: int
    p3_wall: float
    fcl_feasible: bool


def stratified_sample(results, rng, n_per_bin):
    out = {}
    for name, lo, hi in BINS:
        bucket = [
            i
            for i, r in enumerate(results)
            if lo < r.metrics.max_violation <= hi
            or (lo == 0.0 and r.metrics.max_violation <= hi)
        ]
        if not bucket:
            out[name] = []
            continue
        k = min(n_per_bin, len(bucket))
        out[name] = rng.choice(bucket, size=k, replace=False).tolist()
    return out


def main() -> int:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument(
        "--polish-pkl", type=Path, default=Path("/tmp/full_polish_post_sat.pkl")
    )
    p.add_argument("--n-per-bin", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p1-iter", type=int, default=80)
    p.add_argument("--p2-iter", type=int, default=80)
    p.add_argument("--p3-iter", type=int, default=40)
    p.add_argument(
        "--rank-by-offset-fn",
        action="store_true",
        help="Sample top-N by augmented offset_polish_fn "
        "(if available in pkl) instead of mv bins.",
    )
    p.add_argument(
        "--rank-by-violation-fn",
        action="store_true",
        help="Sample top-N by violation_fn (cleaner signal — "
        "no coverage bias). Requires "
        "eval_violation_at_augmented.py output.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="With --rank-by-offset-fn, number of cands to pick.",
    )
    p.add_argument(
        "--stratify-by-offset-fn",
        action="store_true",
        help="Stratified sample across offset_polish_fn bins (requires augmented pkl).",
    )
    p.add_argument(
        "--stratify-by-violation-fn",
        action="store_true",
        help="Stratified sample across violation_fn bins "
        "(requires eval_violation_at_augmented.py output).",
    )
    args = p.parse_args()

    print("Loading config + building probes / SDFs / fixtures...", flush=True)
    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(args.holes)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes = _transform_holes(holes, R, t)

    t_setup = time.time()
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
    print(f"  setup: {time.time() - t_setup:.1f}s", flush=True)

    with open(args.polish_pkl, "rb") as f:
        data = pickle.load(f)

    augmented_x = data.get("augmented_phase1_x")
    offset_fn = data.get("offset_polish_fn")
    have_augmented = augmented_x is not None and offset_fn is not None
    if have_augmented:
        print(
            f"Augmented pkl detected: offset_polish_fn available "
            f"({int(np.sum(np.array(offset_fn) < 200))} cands < 200, "
            f"{int(np.sum(np.array(offset_fn) < 1000))} < 1000)"
        )

    if args.rank_by_violation_fn:
        viol = data.get("violation_fn")
        if viol is None:
            raise SystemExit(
                "--rank-by-violation-fn requires eval_violation_at_augmented.py output"
            )
        order = np.argsort(np.asarray(viol))
        idxs_flat = order[: args.top_n].tolist()
        picks = {"top_by_violation_fn": idxs_flat}
        print(f"Top {args.top_n} cands by violation_fn:")
        for i in idxs_flat[:15]:
            print(f"  cand#{int(i):<5} viol_fn={float(viol[int(i)]):.2f}")
        if args.top_n > 15:
            print(f"  ... (showing first 15 of {args.top_n})")
    elif args.rank_by_offset_fn:
        if not have_augmented:
            raise SystemExit("--rank-by-offset-fn requires an augmented pkl")
        order = np.argsort(np.asarray(offset_fn))
        idxs_flat = order[: args.top_n].tolist()
        picks = {"top_by_offset_fn": idxs_flat}
        print(f"Top {args.top_n} cands by offset_polish_fn:")
        for i in idxs_flat:
            print(f"  cand#{int(i):<5} fn={float(offset_fn[int(i)]):+.2f}")
    elif args.stratify_by_violation_fn:
        viol = data.get("violation_fn")
        if viol is None:
            raise SystemExit(
                "--stratify-by-violation-fn requires "
                "eval_violation_at_augmented.py output"
            )
        rng = np.random.default_rng(args.seed)
        picks = violation_fn_sample(np.asarray(viol), rng, args.n_per_bin)
        total = sum(len(idxs) for idxs in picks.values())
        print(
            f"Sampled {total} cands across {len(VIOLATION_FN_BINS)} violation_fn bins:"
        )
        for name, idxs in picks.items():
            vs = [f"{float(viol[int(i)]):.1f}" for i in idxs]
            print(
                f"  {name:<10}: {len(idxs):>2}  "
                f"cands={[int(i) for i in idxs]}  viol={vs}"
            )
    elif args.stratify_by_offset_fn:
        if not have_augmented:
            raise SystemExit("--stratify-by-offset-fn requires augmented pkl")
        rng = np.random.default_rng(args.seed)
        picks = offset_fn_sample(np.asarray(offset_fn), rng, args.n_per_bin)
        total = sum(len(idxs) for idxs in picks.values())
        print(
            f"Sampled {total} cands across {len(OFFSET_FN_BINS)} offset_polish_fn bins:"
        )
        for name, idxs in picks.items():
            fns = [f"{float(offset_fn[int(i)]):+.1f}" for i in idxs]
            print(
                f"  {name:<13}: {len(idxs):>2}  "
                f"cands={[int(i) for i in idxs]}  fn={fns}"
            )
    else:
        rng = np.random.default_rng(args.seed)
        picks = stratified_sample(data["results"], rng, args.n_per_bin)
        total = sum(len(idxs) for idxs in picks.values())
        print(f"Sampled {total} cands across {len(BINS)} bins:")
        for name, idxs in picks.items():
            print(f"  {name:<12}: {len(idxs):>2}  {[int(i) for i in idxs]}")
    print()

    results: list[CandResult] = []

    for bin_name, idxs in picks.items():
        for cand_idx in idxs:
            cand_idx = int(cand_idx)
            cand = data["candidates"][cand_idx]
            jc = data["results"][cand_idx]
            mv = float(jc.metrics.max_violation)

            statics = _build_probe_static(
                probes,
                holes,
                cand.ha,
                cand.aa,
                bvh_cache=bvh_cache,
                sdf_by_name=sdf_by_name,
            )
            n_arcs = jc.n_arcs
            _n_vars = phase1_n_vars(n_arcs, len(statics))
            coverage_data = build_coverage_data(probes, statics)
            bounds = phase1_bounds(n_arcs, len(statics))
            if have_augmented and len(augmented_x[cand_idx]) > 0:
                x0 = np.asarray(augmented_x[cand_idx], dtype=np.float64)
            else:
                x0 = reduced_to_phase1(jc.reduced_y, n_arcs, len(statics))

            # ---- Phase 1 ----
            p1_fun, p1_jac = make_phase1_objective(
                statics,
                n_arcs,
                coverage_data=coverage_data,
                fixtures=fixtures,
                weights=Phase1Weights(),
            )
            t0 = time.time()
            r1 = minimize(
                p1_fun,
                x0,
                jac=p1_jac,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=args.p1_iter, ftol=1e-5, gtol=1e-5),
            )
            p1_wall = time.time() - t0
            x1 = np.asarray(r1.x, dtype=np.float64)

            # ---- Phase 2 ----
            p2 = make_phase2(
                statics,
                n_arcs,
                coverage_data=coverage_data,
                fixtures=fixtures,
                weights=Phase2Weights(min_clearance_mm=0.3),
            )
            t0 = time.time()
            r2 = minimize(
                p2["fun"],
                x1,
                jac=p2["jac"],
                method="trust-constr",
                bounds=bounds,
                constraints=p2["constraints_nlc"],
                options=dict(
                    maxiter=args.p2_iter,
                    xtol=1e-6,
                    gtol=1e-5,
                    initial_tr_radius=1.0,
                    verbose=0,
                ),
            )
            p2_wall = time.time() - t0
            x2 = np.asarray(r2.x, dtype=np.float64)
            s_p2 = p2["constraints"][0]["fun"](x2)
            p2_min = float(np.min(s_p2))
            p2_violating = int(np.sum(s_p2 < -1e-4))

            # ---- FCL validator (Phase 3 retired 2026-05-24) ----
            # Phase 2 is the only stage that optimises against geometry.
            # We just validate x2 against raw FCL meshes — no polish.
            validator = make_fcl_validator(
                statics,
                n_arcs,
                fixtures=fixtures,
                fixture_bvhs=fixture_bvhs,
            )
            t0 = time.time()
            s_fcl = validator.slacks(x2)
            p3_wall = time.time() - t0
            _x3 = x2  # validator-only; no pose change
            r3_nit = 0
            p3_fcl = float(np.min(s_fcl)) if s_fcl.size else 0.0
            p3_fcl_viol = int(np.sum(s_fcl < -1e-4)) if s_fcl.size else 0
            # No analytic re-check — Phase 2 already enforced its analytic
            # constraints. Report p2_min as a proxy.
            p3_an = float(p2_min)
            fcl_feas = p3_fcl >= -1e-4

            cr = CandResult(
                cand_idx=cand_idx,
                bin_name=bin_name,
                mv_stage2=mv,
                p1_fn_end=float(r1.fun),
                p1_nit=int(r1.nit),
                p1_wall=p1_wall,
                p2_min_slack=p2_min,
                p2_n_violating=p2_violating,
                p2_nit=int(r2.nit),
                p2_wall=p2_wall,
                p3_min_slack_analytic=p3_an,
                p3_min_slack_fcl=p3_fcl,
                p3_n_violating_fcl=p3_fcl_viol,
                p3_nit=r3_nit,
                p3_wall=p3_wall,
                fcl_feasible=fcl_feas,
            )
            results.append(cr)
            tag = "FEAS" if fcl_feas else "FAIL"
            print(
                f"  {bin_name:<12} cand#{cand_idx:<5} mv={mv:>9.4f} "
                f"P1[fn={r1.fun:+8.2f} nit={r1.nit:>3}] "
                f"P2[min_s={p2_min:+.4f} nv={p2_violating}] "
                f"P3[fcl={p3_fcl:+.4f} an={p3_an:+.4f}] "
                f"wall={p1_wall + p2_wall + p3_wall:.0f}s {tag}",
                flush=True,
            )

    # ---- Per-bin summary ----
    print("\n" + "=" * 78)
    print("Per-bin success rate (Phase 3 FCL-feasible / n_sampled)")
    print("=" * 78)
    print(
        f"{'bin':<12} {'n':>3} {'FEAS':>5} {'pct':>6}  "
        f"{'P1 wall':>8} {'P2 wall':>8} {'P3 wall':>8}"
    )
    for name, _lo, _hi in BINS:
        bin_results = [r for r in results if r.bin_name == name]
        if not bin_results:
            continue
        n = len(bin_results)
        feas = sum(r.fcl_feasible for r in bin_results)
        wall_p1 = np.mean([r.p1_wall for r in bin_results])
        wall_p2 = np.mean([r.p2_wall for r in bin_results])
        wall_p3 = np.mean([r.p3_wall for r in bin_results])
        print(
            f"{name:<12} {n:>3} {feas:>5} {feas / n * 100:>5.0f}%  "
            f"{wall_p1:>7.1f}s {wall_p2:>7.1f}s {wall_p3:>7.1f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
