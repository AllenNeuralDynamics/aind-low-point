"""Second pass over the augmented pkl: at each cand's offset-polished
pose, evaluate the violation-only portion of the Phase 1 objective.

Decouples the fixability signal from coverage / margin rewards. The
augmentation polish kept the full Phase 1 objective on purpose (so the
20-iter preview lands in a Phase-1-meaningful basin), but the resulting
``fn`` is not a clean fixability metric — coverage and margin rewards
bias it.

This pass evaluates the same objective with:

  - ``coverage_data=None``                  (drops ``-coverage_total``)
  - ``lambda_margin_clear=0``               (drops clear-margin reward)
  - ``lambda_margin_thread=0``              (drops thread-margin reward)
  - ``lambda_margin_clear_fixture=0``       (drops fixture-margin reward)

What remains is ``λ_thread * j_thread + λ_clear * j_clear + λ_fix * j_clear_fixture +
λ_kin * (j_arc_ap + j_ml) + λ_bounds * j_bounds`` — non-negative,
violation-only. ``violation_fn ≈ 0`` ⇒ cand is genuinely free of
soft-constraint violation at the offset-polished pose.

Run::
    uv run --python 3.13 python -m scripts.eval_violation_at_augmented \\
        examples/836656-config-T12.yml /tmp/836656-holes.yml \\
        --in-pkl /tmp/full_polish_lbfgsb_augmented.pkl
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os as _os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


_W: dict = {}


def _worker_init(config_path: str, holes_path: str):
    _os.environ["JAX_PLATFORMS"] = "cpu"

    from aind_low_point.config import ConfigModel
    from aind_low_point.optimization.headstages import make_fcl_bvh
    from aind_low_point.optimization.holes import load_holes
    from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
    from aind_low_point.runtime import build_runtime_from_config
    from aind_low_point.runtime.transforms import compile_all_transforms
    from scripts.run_optimizer import _probe_static_info, _transform_holes
    from scripts.run_phase1_sample import build_fixture_sdf_data

    cfg = ConfigModel.from_yaml(Path(config_path))
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(Path(holes_path))
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
    _W.update(dict(
        probes=probes, holes=holes, sdf_by_name=sdf_by_name,
        fixtures=fixtures, bvh_cache=bvh_cache,
    ))


def _eval_cand(payload):
    """Returns ``(idx, violation_fn, coverage_value)`` at the augmented
    pose for one cand.
    """
    from aind_low_point.optimization.joint_rerank import _build_probe_static
    from aind_low_point.optimization.stage3_phase1_jax import (
        Phase1Weights,
        make_phase1_objective,
    )
    from scripts.run_phase1_sample import build_coverage_data

    idx, ha, aa, n_arcs, x_aug = payload
    statics = _build_probe_static(
        _W["probes"], _W["holes"], ha, aa,
        bvh_cache=_W["bvh_cache"], sdf_by_name=_W["sdf_by_name"],
    )
    # Violation-only: drop coverage + margin rewards.
    weights_viol = Phase1Weights(
        lambda_margin_clear=0.0,
        lambda_margin_thread=0.0,
        lambda_margin_clear_fixture=0.0,
    )
    try:
        fn_viol, _ = make_phase1_objective(
            statics, n_arcs, coverage_data=None,
            fixtures=_W["fixtures"], weights=weights_viol,
        )
        viol = float(fn_viol(np.asarray(x_aug, dtype=np.float64)))
    except Exception:
        viol = float("inf")

    # Coverage at the same pose (for downstream lex_key tiebreaker).
    try:
        cov_data = build_coverage_data(_W["probes"], statics)
        fn_full, _ = make_phase1_objective(
            statics, n_arcs, coverage_data=cov_data,
            fixtures=_W["fixtures"],
            weights=Phase1Weights(
                # Zero everything but coverage so we can recover -coverage
                # from the fn (we'll negate it).
                lambda_thread=0.0,
                lambda_clearance=0.0,
                lambda_kinematic=0.0,
                lambda_bounds=0.0,
                lambda_clearance_fixture=0.0,
                lambda_margin_clear=0.0,
                lambda_margin_thread=0.0,
                lambda_margin_clear_fixture=0.0,
            ),
        )
        # With everything zeroed except coverage, fn = -coverage_total.
        # Negate to get the (positive) coverage value.
        cov_val = -float(fn_full(np.asarray(x_aug, dtype=np.float64)))
    except Exception:
        cov_val = float("nan")

    return idx, viol, cov_val


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--in-pkl", type=Path,
                   default=Path("/tmp/full_polish_lbfgsb_augmented.pkl"))
    p.add_argument("--out-pkl", type=Path, default=None,
                   help="Output pkl path (default: overwrite in-pkl)")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_path = args.out_pkl or args.in_pkl

    with open(args.in_pkl, "rb") as f:
        data = pickle.load(f)
    cands = data["candidates"]
    results = data["results"]
    aug_x = data.get("augmented_phase1_x")
    if aug_x is None:
        raise SystemExit("Input pkl lacks ``augmented_phase1_x`` — run "
                         "augment_polish_with_offsets.py first.")
    n_total = len(cands)
    n_process = args.limit if args.limit else n_total
    print(f"Loaded {n_total} cands; will eval {n_process}", flush=True)

    viol = np.full(n_total, np.nan, dtype=np.float64)
    cov = np.full(n_total, np.nan, dtype=np.float64)

    prev_jax = _os.environ.get("JAX_PLATFORMS")
    _os.environ["JAX_PLATFORMS"] = "cpu"
    ctx = mp.get_context("fork")

    t0 = time.time()
    try:
        with ProcessPoolExecutor(
            max_workers=args.n_workers, mp_context=ctx,
            initializer=_worker_init,
            initargs=(str(args.config), str(args.holes)),
        ) as pool:
            payloads = [
                (i, cands[i].ha, cands[i].aa, results[i].n_arcs, aug_x[i])
                for i in range(n_process)
                if len(aug_x[i]) > 0
            ]
            n_done = 0
            for idx, v, c in pool.map(_eval_cand, payloads, chunksize=20):
                viol[idx] = v
                cov[idx] = c
                n_done += 1
                if n_done % 500 == 0 or n_done == len(payloads):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (len(payloads) - n_done) / rate if rate > 0 else 0
                    print(f"  {n_done}/{len(payloads)}  ({rate:.1f} cands/s, "
                          f"ETA {eta:.0f}s)", flush=True)
    finally:
        if prev_jax is None:
            _os.environ.pop("JAX_PLATFORMS", None)
        else:
            _os.environ["JAX_PLATFORMS"] = prev_jax

    wall = time.time() - t0
    print(f"\nEval wall: {wall:.0f}s", flush=True)

    finite = viol[~np.isnan(viol)]
    if finite.size:
        print("\nviolation-only fn stats:")
        print(f"  min={finite.min():.2f}  max={finite.max():.2e}  "
              f"median={np.median(finite):.2f}")
        for thresh in (1, 10, 100, 1000):
            n_below = int(np.sum(finite < thresh))
            print(f"  fn < {thresh:>5}: {n_below} cands "
                  f"({n_below / finite.size * 100:.1f}%)")
    finite_cov = cov[~np.isnan(cov)]
    if finite_cov.size:
        print("\ncoverage at augmented pose:")
        print(f"  min={finite_cov.min():.2f}  max={finite_cov.max():.2f}  "
              f"median={np.median(finite_cov):.2f}")

    data["violation_fn"] = viol
    data["coverage_at_aug"] = cov
    print(f"\nSaving to {out_path}...", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved ({out_path.stat().st_size / (1024*1024):.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
