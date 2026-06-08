"""Augment a Stage 2 polish pkl with a per-cand offset-only polish.

For each candidate in the input pkl:

  1. Lift ``reduced_y`` to Phase 1 layout (off=depth=0)
  2. Run a tiny L-BFGS-B with only ``(off_R, off_A, depth)`` per probe
     free (21 vars) — Phase 1 objective, other DOF pinned at Stage 2's
     polished values. ~20 iter, ~1 s per cand.
  3. Record the offset-polished ``phase1_x`` (45-dim) and the residual
     fn (the "fixability" signal — fixable cands drop to fn < 200;
     doomed cands stay > 20000)

Output an augmented pkl with two new arrays:

  - ``augmented_phase1_x``: ``(n_cands, 45)`` — warm-start for Phase 1
  - ``offset_polish_fn``: ``(n_cands,)`` — fixability signal

Downstream chain (Phase 1 / Phase 2 / FCL validator) reads the
augmented warm-start instead of lifting reduced_y with zero offsets.

Run::
    uv run --python 3.13 python -m scripts.augment_polish_with_offsets \\
        examples/836656-config-T12.yml scratch/0283-300-04.holes.yml \\
        --in-pkl /tmp/full_polish_lbfgsb.pkl \\
        --out-pkl /tmp/full_polish_lbfgsb_augmented.pkl
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
from scipy.optimize import minimize

# Worker-local globals (set in _worker_init)
_W: dict = {}


def _worker_init(config_path: str, holes_path: str):
    """Build per-worker probes / SDFs / fixtures / coverage data once."""
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
    _W.update(
        dict(
            probes=probes,
            holes=holes,
            sdf_by_name=sdf_by_name,
            fixtures=fixtures,
            bvh_cache=bvh_cache,
        )
    )


def _offset_only_bounds(
    x_full, n_arcs: int, n_probes: int, off_max: float = 2.0, depth_max: float = 3.0
):
    from aind_low_point.optimization.stage3_phase1_jax import (
        PHASE1_PER_PROBE_VARS,
    )

    bounds: list[tuple[float, float]] = []
    for j in range(n_arcs):
        v = float(x_full[j])
        bounds.append((v, v))
    for i in range(n_probes):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        for k in range(3):
            v = float(x_full[off + k])
            bounds.append((v, v))
        bounds.append((-off_max, off_max))
        bounds.append((-off_max, off_max))
        bounds.append((-depth_max, depth_max))
    return bounds


def _augment_cand(payload):
    """Per-cand worker. Returns ``(idx, phase1_x, fn_opt)``."""
    from aind_low_point.optimization.joint_rerank import _build_probe_static
    from aind_low_point.optimization.stage3_phase1_jax import (
        Phase1Weights,
        make_phase1_objective,
        reduced_to_phase1,
    )
    from scripts.run_phase1_sample import build_coverage_data

    idx, ha, aa, reduced_y, n_arcs, offset_iter = payload
    statics = _build_probe_static(
        _W["probes"],
        _W["holes"],
        ha,
        aa,
        bvh_cache=_W["bvh_cache"],
        sdf_by_name=_W["sdf_by_name"],
    )
    n_probes = len(statics)
    coverage_data = build_coverage_data(_W["probes"], statics)
    fn_p1, jac_p1 = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=_W["fixtures"],
        weights=Phase1Weights(),
    )
    x0 = reduced_to_phase1(reduced_y, n_arcs, n_probes)
    bounds = _offset_only_bounds(x0, n_arcs, n_probes)
    try:
        r = minimize(
            fn_p1,
            x0,
            jac=jac_p1,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=offset_iter, ftol=1e-5, gtol=1e-5),
        )
        return idx, np.asarray(r.x, dtype=np.float64), float(r.fun)
    except Exception:
        return idx, x0, float("inf")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--in-pkl", type=Path, default=Path("/tmp/full_polish_lbfgsb.pkl"))
    p.add_argument(
        "--out-pkl", type=Path, default=Path("/tmp/full_polish_lbfgsb_augmented.pkl")
    )
    p.add_argument("--offset-iter", type=int, default=20)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Augment only the first N cands (for testing)",
    )
    args = p.parse_args()

    with open(args.in_pkl, "rb") as f:
        data = pickle.load(f)
    cands = data["candidates"]
    results = data["results"]
    n_total = len(cands)
    n_process = args.limit if args.limit else n_total
    print(f"Loaded {n_total} cands; will augment {n_process}", flush=True)

    n_workers = args.n_workers or max(1, (mp.cpu_count() or 2) - 2)
    print(f"Spawning {n_workers} workers (init ~5-10 s each)...", flush=True)

    # Pin parent JAX to CPU before spawn so workers inherit
    prev_jax = _os.environ.get("JAX_PLATFORMS")
    _os.environ["JAX_PLATFORMS"] = "cpu"

    # n_arcs varies per cand (2 or 3) → phase1_x length varies → store as
    # an object array of per-cand arrays.
    augmented_x: list[np.ndarray] = [np.zeros(0) for _ in range(n_total)]
    offset_fn = np.full(n_total, np.nan, dtype=np.float64)

    # Fork preserves parent FS access (spawn-mode workers were hitting
    # PermissionError on /mnt/vast/scratch MRI paths). JAX-on-CPU + fork
    # is supported.
    ctx = mp.get_context("fork")
    t0 = time.time()
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(str(args.config), str(args.holes)),
        ) as pool:
            payloads = []
            for i in range(n_process):
                cand = cands[i]
                jc = results[i]
                payloads.append(
                    (
                        i,
                        cand.ha,
                        cand.aa,
                        jc.reduced_y,
                        jc.n_arcs,
                        args.offset_iter,
                    )
                )

            n_done = 0
            for idx, x_aug, fn_opt in pool.map(_augment_cand, payloads, chunksize=10):
                augmented_x[idx] = x_aug
                offset_fn[idx] = fn_opt
                n_done += 1
                if n_done % 200 == 0 or n_done == n_process:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (n_process - n_done) / rate if rate > 0 else 0
                    print(
                        f"  {n_done}/{n_process}  ({rate:.1f} cands/s, ETA {eta:.0f}s)",
                        flush=True,
                    )
    finally:
        if prev_jax is None:
            _os.environ.pop("JAX_PLATFORMS", None)
        else:
            _os.environ["JAX_PLATFORMS"] = prev_jax

    wall = time.time() - t0
    print(
        f"\nAugmentation wall: {wall:.0f}s ({n_process / wall:.1f} cands/s)", flush=True
    )

    # Stats on the fixability signal
    finite = offset_fn[~np.isnan(offset_fn)]
    if finite.size:
        print("\nOffset-polish residual fn:")
        print(
            f"  min={finite.min():.2f}  max={finite.max():.2e}  "
            f"median={np.median(finite):.2f}"
        )
        for thresh in (200, 1000, 5000):
            n_below = int(np.sum(finite < thresh))
            print(
                f"  fn < {thresh:>5}: {n_below} cands "
                f"({n_below / finite.size * 100:.1f}%)"
            )

    # Save augmented pkl
    out = dict(data)  # shallow copy
    out["augmented_phase1_x"] = augmented_x  # list of (44- or 45-,) arrays
    out["offset_polish_fn"] = offset_fn
    print(f"\nSaving augmented pkl to {args.out_pkl}...", flush=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = args.out_pkl.stat().st_size / (1024 * 1024)
    print(f"  saved ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
