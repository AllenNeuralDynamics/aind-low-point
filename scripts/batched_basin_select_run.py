"""Multi-basin batched basin-select on a limited sample (GPU).

Assembles the production form end-to-end on ~N_CAND candidates:
  per candidate -> 5 basins (incumbent grid-sweep + 4 cheap H1/flip)
  -> flatten (cand x basin) -> ONE batched ADAM pass -> per-cand argmin.

Reports: manual rank by basin-selected violation, FCL on the top-K
(dev-only check), and ms/candidate timing for sizing the full pool run.

Run on GPU:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.batched_basin_select_run
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import Phase1Weights
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.batched_phase1_build import make_batched_phase1_chunked
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import build_fixture_sdf_data, phase1_bounds
from scripts.spin_heuristic_search import is_four_shank, spin_to_align_y_with
from scripts.test_h1_chain_cand4195 import build_y, extract_spins

N_CAND = 500
PPV = 6
FCL_TOPK = 25
N_SURF = int(_os.environ.get("N_SURF", "5000"))
STEPS = int(_os.environ.get("STEPS", "200"))
FLIP_DEGS = [int(x) for x in _os.environ.get("FLIP_DEGS", "0,90,180,270").split(",")]
# BF16_STORE: mixed-precision kernel — probe + well SDF grids stored bf16 on
# device, native bf16 gather/blend in trilinear_sdf, fp32 reduction. Validated
# FN-safe (manual #1, 4/4 feasible) with ~24% faster steady-state + lower VRAM.
BF16_STORE = _os.environ.get("BF16_STORE", "0") == "1"
# PROFILE_CHUNK: build the kernel, run ONLY the first chunk (which triggers
# XLA compile + the autotuner transient → the true process-peak VRAM), print a
# parseable PROFILE line, and exit. Driven by scripts/profile_chunk_vram.py,
# which sweeps CHUNK across subprocesses and linear-fits peak(chunk).
PROFILE_CHUNK = _os.environ.get("PROFILE_CHUNK", "0") == "1"


def main() -> int:
    print(f"JAX devices: {jax.devices()}")
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
    print(f"n_surface_points = {N_SURF}")
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw,
            n_surface_points=N_SURF,
        )
        for p in probes
    }
    bvh_cache = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fixtures = build_fixture_sdf_data(runtime)
    well = next(f for f in fixtures if "well" in f.name.lower())
    fixture_bvhs = {
        well.name: make_fcl_bvh(runtime.asset_catalog.get_geometry(well.name).raw)
    }

    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    vf = np.asarray(data["violation_fn"], float)
    results = data["results"]
    n_arcs = 3
    n_probes = len(probes)
    eligible = [i for i in np.argsort(vf) if results[i].n_arcs == n_arcs]
    idxs = list(
        np.asarray(eligible)[np.linspace(0, len(eligible) - 1, N_CAND).astype(int)]
    )
    if 4195 not in idxs:
        idxs[-1] = 4195
    man_pos = idxs.index(4195)

    bounds = phase1_bounds(n_arcs, n_probes)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)

    print(f"Building statics + basins for {len(idxs)} candidates...")
    t0 = time.time()
    statics_flat, x0_rows, cand_of_row = [], [], []
    statics_by_cand = {}
    for ci, idx in enumerate(idxs):
        cand = data["candidates"][idx]
        st = _build_probe_static(
            probes,
            holes,
            cand.ha,
            cand.aa,
            bvh_cache=bvh_cache,
            sdf_by_name=sdf_by_name,
        )
        statics_by_cand[ci] = st
        x_aug = np.asarray(data["augmented_phase1_x"][idx], float)
        arc_aps = x_aug[:n_arcs]
        mls = np.array([x_aug[n_arcs + PPV * i] for i in range(n_probes)])
        inc = extract_spins(x_aug, n_arcs, n_probes)
        h1 = np.array(
            [
                spin_to_align_y_with(
                    s.assigned_hole.slot_major_dir(),
                    float(arc_aps[s.arc_idx]),
                    float(mls[i]),
                )
                for i, s in enumerate(st)
            ]
        )
        one = np.array([not is_four_shank(s) for s in st])
        basins = [inc] + [np.where(one, h1 + d, h1) for d in FLIP_DEGS]
        zero = np.zeros(n_probes)
        for sp in basins:
            statics_flat.append(st)
            x0_rows.append(build_y(arc_aps, n_arcs, mls, sp, zero, zero, zero))
            cand_of_row.append(ci)
    x0 = np.stack(x0_rows).astype(np.float32)
    n_basin = len(basins)
    cand_of_row = np.array(cand_of_row)
    print(
        f"  {time.time() - t0:.1f}s; {x0.shape[0]} rows "
        f"({len(idxs)} cands x {n_basin} basins)"
    )

    CHUNK = int(_os.environ.get("CHUNK", "96"))
    print(
        f"Chunked COMPILED ADAM (chunk={CHUNK}, {STEPS} steps, lr 0.02, "
        f"{n_basin} basins)..."
    )
    grid_dtype = jnp.bfloat16 if BF16_STORE else jnp.float32
    well_obj = well
    if BF16_STORE:
        print("BF16_STORE: probe + well SDF grids stored bf16 (real mixed kernel)")
        # bf16 the well grid for the objective only; FCL uses the BVH.
        well_obj = replace(well, grid=jnp.asarray(well.grid, jnp.bfloat16))
    vobj, vgrad, build_arglist, make_adam, _mks = make_batched_phase1_chunked(
        statics_flat[0],
        n_arcs,
        Phase1Weights(),
        (well_obj,),
        coverage_data=None,
        grid_dtype=grid_dtype,
    )
    run_adam = make_adam(lo, hi, steps=STEPS, lr=0.02)
    dev = jax.devices()[0]
    n_rows = x0.shape[0]
    n_chunks = (n_rows + CHUNK - 1) // CHUNK
    viol = np.zeros(n_rows)
    x_adam = np.zeros_like(x0)
    baseline = None
    t_first = 0.0
    t_second = 0.0
    t0 = time.time()
    for ci, start in enumerate(range(0, n_rows, CHUNK)):
        tc = time.time()
        rows = list(range(start, min(start + CHUNK, n_rows)))
        pad = CHUNK - len(rows)
        rows_p = rows + [rows[-1]] * pad
        arglist = build_arglist([statics_flat[r] for r in rows_p])
        if baseline is None:
            baseline = dev.memory_stats().get("bytes_in_use", 0)
        xa = np.asarray(run_adam(jnp.asarray(x0[rows_p], jnp.float32), arglist))
        vc = np.asarray(vobj(jnp.asarray(xa, jnp.float32), *arglist))
        viol[rows] = vc[: len(rows)]
        x_adam[rows] = xa[: len(rows)]
        if ci == 0:
            t_first = time.time() - tc
        elif ci == 1:
            t_second = time.time() - tc  # post-compile steady (1 chunk)
            if PROFILE_CHUNK:
                break  # ci==0 captured compile+autotuner peak; ci==1 = steady
    t_adam = time.time() - t0
    steady = (t_adam - t_first) / max(1, n_chunks - 1)  # per-chunk, post-compile
    ms = dev.memory_stats()
    peak = ms.get("peak_bytes_in_use", 0)
    limit = ms.get("bytes_limit", 0)
    per_row = max(1.0, peak - baseline) / CHUNK
    sugg = int(0.8 * (limit - baseline) / per_row) if per_row > 0 else 0
    full_chunks = (8908 * n_basin + CHUNK - 1) // CHUNK
    full_est = t_first + full_chunks * steady
    print(
        f"  {t_adam:.1f}s total; first-chunk(compile) {t_first:.1f}s; "
        f"steady {steady:.2f}s/chunk = {steady / CHUNK * n_basin * 1000:.1f} ms/cand"
    )
    print(
        f"  VRAM: peak {peak / 1e9:.2f} GB / limit {limit / 1e9:.2f} GB; "
        f"~{per_row / 1e6:.1f} MB/row; chunk @80% ≈ {sugg} rows"
    )
    print(
        f"  full-pool est (8908 x {n_basin} basins, this chunk): "
        f"{full_est / 60:.0f} min"
    )

    if PROFILE_CHUNK:
        # Machine-parseable line for profile_chunk_vram.py. steady is the
        # SECOND chunk's wall (post-compile, one chunk) — the single-chunk
        # loop can't measure it, so we report t_second directly.
        print(
            f"PROFILE chunk={CHUNK} peak={peak} baseline={baseline} "
            f"limit={limit} steady={t_second:.4f} t_first={t_first:.4f}"
        )
        return 0

    # Per-candidate argmin over its basins.
    best_viol = np.full(len(idxs), np.inf)
    best_row = np.full(len(idxs), -1, int)
    for r in range(x0.shape[0]):
        c = cand_of_row[r]
        if viol[r] < best_viol[c]:
            best_viol[c] = viol[r]
            best_row[c] = r

    rank = np.argsort(best_viol)
    order = {int(c): r for r, c in enumerate(rank)}
    print(f"\n=== basin-selected ranking ({len(idxs)} cands) ===")
    print(
        f"Manual (#4195): basin-sel viol {best_viol[man_pos]:+.3f}, "
        f"rank {order[man_pos] + 1}/{len(idxs)} "
        f"(frozen violation_fn was 676 / pool-rank #4641)"
    )
    print(
        f"candidates with best_viol < 0 (soft-feasible-ish): "
        f"{int((best_viol < 0).sum())}/{len(idxs)}"
    )

    # Dev-only FCL on the top-K of the basin-selected ranking.
    print(f"\nFCL on top-{FCL_TOPK} of basin-selected ranking:")
    n_feas = 0
    for k in range(min(FCL_TOPK, len(idxs))):
        c = int(rank[k])
        st = statics_by_cand[c]
        v = make_fcl_validator(
            st,
            n_arcs,
            fixtures=(well,),
            fixture_bvhs={well.name: fixture_bvhs[well.name]},
        )
        fcl = float(np.asarray(v.slacks(x_adam[best_row[c]])).min())
        feas = fcl >= -1e-4
        n_feas += feas
        if k < 12 or idxs[c] == 4195:
            tag = " <-- MANUAL" if idxs[c] == 4195 else ""
            print(
                f"  rank {k + 1:>2}  cand {idxs[c]:>5}  viol {best_viol[c]:>+8.3f}  "
                f"fcl {fcl:>+7.3f}  {'FEAS' if feas else 'infeas'}{tag}"
            )
    print(f"\n  FCL-feasible in top-{FCL_TOPK}: {n_feas}/{FCL_TOPK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
