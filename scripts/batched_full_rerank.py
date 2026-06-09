"""Durable full-pool batched-ADAM basin-select rerank (all 8908 candidates).

The production rerank over the full pool (vs an earlier 500-sample,
single-group diagnostic prototype). Key properties:

  * ALL candidates, not a linspace sample.
  * GROUPED by n_arcs (3-arc and 2-arc have different x-layouts) — one kernel
    build + ADAM pass per group, NOT padded to a common arc count.
  * Statics are built PER GROUP and freed on return; the top-K FCL pass
    REBUILDS statics on demand, so peak host RAM never holds all candidates'
    statics at once.
  * Saves a durable artifact: per-candidate best basin-select violation, the
    winning pose, and FCL verdicts on the top-K, to ``scratch/full_rerank_0283.pkl``.

Kernel/basin/config are the validated ones from the diagnostic: bf16 grid
storage (BF16_STORE default on), 150 steps, 3 basins (incumbent + h1 ± 180),
5000 surface points, chunk 64 (profiled throughput plateau).

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.batched_full_rerank
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
from aind_low_point.optimization.optimizer_vars import build_y, extract_spins
from aind_low_point.optimization.probe_kinematics import (
    is_four_shank,
    spin_to_align_y_with,
)
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import Phase1Weights
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.batched_phase1_build import (
    ARG_ORDER,
    PER_CAND,
    make_batched_phase1_chunked,
)
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    maybe_build_brain_sdf,
    phase1_bounds,
)

PPV = 6
N_SURF = int(_os.environ.get("N_SURF", "5000"))
STEPS = int(_os.environ.get("STEPS", "150"))
# TWO_STAGE=1 runs a clearance-first REDUCED stage (coverage off, offsets/depth
# pinned, STAGE1 steps) before the full coverage-aware stage (STAGE2 steps) —
# both on ONE shared kernel via runtime cov_weight + bounds (no 2nd compile).
# Default 0 = the legacy single full-stage pass (STEPS), byte-identical.
TWO_STAGE = _os.environ.get("TWO_STAGE", "0") == "1"
STAGE1 = int(_os.environ.get("STAGE1", "150"))
STAGE2 = int(_os.environ.get("STAGE2", str(STEPS)))
FLIP_DEGS = [int(x) for x in _os.environ.get("FLIP_DEGS", "0,180").split(",")]
CHUNK = int(_os.environ.get("CHUNK", "64"))
# Async pipeline depth: chunks kept in flight before the host syncs the
# oldest. >=2 overlaps host slice-prep with GPU compute. 0 = synchronous.
PIPELINE_DEPTH = int(_os.environ.get("PIPELINE_DEPTH", "2"))
# Print an in-loop progress line every PROGRESS_EVERY drained chunks.
PROGRESS_EVERY = int(_os.environ.get("PROGRESS_EVERY", "25"))
BF16_STORE = _os.environ.get("BF16_STORE", "1") == "1"
# COVERAGE: include the Phase 1 coverage term (-coverage_total). On = faithful
# Phase 1 (anchors recording center to target, opposes depth retraction).
# Off reproduces the feasibility-only sort. Per-probe Gaussian/KDE auto-picked
# by build_coverage_data (discrete target -> Gaussian, point cloud -> KDE).
COVERAGE = _os.environ.get("COVERAGE", "1") == "1"
# COV_NORM: normalize per-probe coverage by its achievable ceiling (so regions
# weigh equally regardless of shank count / active area / σ / density) and blend
# average vs worst region by COV_ALPHA in [0,1]. Off = legacy plain-sum coverage.
COV_NORM = _os.environ.get("COV_NORM", "0") == "1"
COV_ALPHA = float(_os.environ.get("COV_ALPHA", "0.2"))
FCL_TOPK = int(_os.environ.get("FCL_TOPK", "100"))
# LIMIT caps candidates per n_arcs group (0 = all) — for a fast smoke test.
LIMIT = int(_os.environ.get("LIMIT", "0"))
OUT_PATH = Path(_os.environ.get("OUT", "scratch/full_rerank_0283.pkl"))
MANUAL = 4195  # n_arcs==3 ground-truth, for the summary line


def run_group(  # noqa: C901
    n_arcs,
    idxs,
    *,
    probes,
    holes,
    data,
    sdf_by_name,
    bvh_cache,
    well_obj,
    brain_sdf=None,
):
    """Basin-select one n_arcs group. Builds statics locally (freed on
    return), runs ONE chunked ADAM pass, returns per-cand records
    ``[{idx, n_arcs, viol, pose}]`` (NO statics retained)."""
    n_probes = len(probes)
    bounds = phase1_bounds(n_arcs, n_probes)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    # Reduced-stage bounds: pin (off_R, off_A, depth) to 0 so the clearance-first
    # stage moves only arc/ml/spin (same kernel; the bounds are runtime args).
    lo_r, hi_r = lo.copy(), hi.copy()
    for k in range(n_probes):
        for off in (3, 4, 5):
            lo_r[n_arcs + PPV * k + off] = 0.0
            hi_r[n_arcs + PPV * k + off] = 0.0

    print(f"[n_arcs={n_arcs}] building statics+basins for {len(idxs)} cands...")
    t0 = time.time()
    statics_flat, x0_rows, cand_of_row = [], [], []
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

    # Coverage targets are per-probe-fixed (target_LPS/sigma/kind, not the hole
    # assignment), so build once from any candidate's 7-probe statics and pass
    # as the shared closure — no per-candidate plumbing.
    coverage_data = build_coverage_data(probes, statics_flat[0]) if COVERAGE else None
    # Optional per-region normalization + soft-min fairness floor. Ceilings are
    # per-probe-fixed (target/σ/shank geometry), so compute once per group.
    ceilings, weights = None, Phase1Weights()
    if COVERAGE and COV_NORM:
        from aind_low_point.optimization.coverage_jax import (
            coverage_ceiling_per_probe,
        )

        ceilings = tuple(coverage_ceiling_per_probe(statics_flat[0], coverage_data))
        weights = Phase1Weights(cov_alpha=COV_ALPHA)
        print(
            f"  coverage NORMALIZED; ceilings={[round(c, 3) for c in ceilings]}, "
            f"α={COV_ALPHA}"
        )
    grid_dtype = jnp.bfloat16 if BF16_STORE else jnp.float32
    vobj, _vgrad, build_arglist, make_adam, make_staged_adam = (
        make_batched_phase1_chunked(
            statics_flat[0],
            n_arcs,
            weights,
            (well_obj,),
            coverage_data=coverage_data,
            grid_dtype=grid_dtype,
            brain_sdf=brain_sdf,
            coverage_ceilings=ceilings,
        )
    )
    if TWO_STAGE:
        # ONE compiled kernel; reduced (pinned bounds, cov_weight=0, STAGE1)
        # then full (full bounds, cov_weight=1, STAGE2). Bounds / cov_weight /
        # step-count are runtime args ⇒ no second compile.
        run_staged = make_staged_adam(lr=0.02)

        def run_adam(x0c, cargs):
            x1 = run_staged(x0c, cargs, lo_r, hi_r, 0.0, STAGE1)
            return run_staged(x1, cargs, lo, hi, 1.0, STAGE2)
    else:
        run_adam = make_adam(lo, hi, steps=STEPS, lr=0.02)
    n_rows = x0.shape[0]
    # Pad rows up to a CHUNK multiple so every chunk is a clean fixed-size
    # device slice (the vmap is compiled for exactly CHUNK rows).
    n_pad = (-n_rows) % CHUNK
    if n_pad:
        statics_flat = statics_flat + [statics_flat[-1]] * n_pad
        x0 = np.concatenate([x0, np.repeat(x0[-1:], n_pad, axis=0)], axis=0)
    n_tot = x0.shape[0]

    # Pre-stage ALL per-candidate inputs on device ONCE (stacked over rows).
    # Per-chunk work is then a device-side slice — no per-chunk host packing,
    # no host->device transfer inside the loop.
    full_arglist = build_arglist(statics_flat)
    per_cand_pos = [k in PER_CAND for k in ARG_ORDER]
    x0_dev = jnp.asarray(x0, jnp.float32)

    n_chunks = n_tot // CHUNK
    steps_str = f"reduced {STAGE1}→full {STAGE2}" if TWO_STAGE else f"{STEPS} steps"
    print(
        f"  pipelined ADAM (chunk={CHUNK}, depth={PIPELINE_DEPTH}, "
        f"{steps_str}, {n_basin} basins, {n_chunks} chunks)..."
    )
    t0 = time.time()
    viol = np.full(n_tot, np.inf)
    x_adam = np.zeros((n_tot, x0.shape[1]), np.float32)
    drained = [0]

    def _drain(item):
        # Syncs on GPU completion of this chunk, so the drain count tracks
        # real progress (the async dispatch loop races ahead of the GPU).
        s0, xa_f, vc_f = item
        x_adam[s0 : s0 + CHUNK] = np.asarray(xa_f)
        viol[s0 : s0 + CHUNK] = np.asarray(vc_f)
        drained[0] += 1
        d = drained[0]
        if d % PROGRESS_EVERY == 0 or d == n_chunks:
            el = time.time() - t0
            eta = el / d * (n_chunks - d)
            print(
                f"    chunk {d}/{n_chunks} ({100 * d / n_chunks:.0f}%)  "
                f"{el:.0f}s elapsed  ETA {eta:.0f}s",
                flush=True,
            )

    # Async double-buffer: dispatch run_adam/vobj (which return un-synced
    # device arrays) and keep up to PIPELINE_DEPTH chunks in flight before
    # syncing the oldest — overlaps host slice-prep with GPU compute.
    inflight: list = []
    for s in range(0, n_tot, CHUNK):
        cargs = [
            a[s : s + CHUNK] if p else a for a, p in zip(full_arglist, per_cand_pos)
        ]
        xa = run_adam(x0_dev[s : s + CHUNK], cargs)
        vc = vobj(xa, *cargs)
        inflight.append((s, xa, vc))
        if len(inflight) > PIPELINE_DEPTH:
            _drain(inflight.pop(0))
    for item in inflight:
        _drain(item)
    viol = viol[:n_rows]
    x_adam = x_adam[:n_rows]
    print(f"  {time.time() - t0:.1f}s ADAM")

    # Per-candidate argmin over its basins → one record per candidate.
    records = []
    for ci, idx in enumerate(idxs):
        mask = cand_of_row == ci
        rrows = np.nonzero(mask)[0]
        br = rrows[int(np.argmin(viol[rrows]))]
        records.append(
            dict(
                idx=int(idx),
                n_arcs=int(n_arcs),
                viol=float(viol[br]),
                pose=x_adam[br].copy(),
            )
        )
    return records  # statics_flat/x_adam go out of scope → freed


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
    print(f"n_surface_points = {N_SURF}; bf16_store = {BF16_STORE}")
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
    # FCL validation uses ALL fixtures (headframe+cone+well) for an honest
    # feasibility verdict; the soft sort uses well only (well dominates).
    fixture_bvhs = {
        f.name: make_fcl_bvh(runtime.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }
    well_obj = (
        replace(well, grid=jnp.asarray(well.grid, jnp.bfloat16)) if BF16_STORE else well
    )
    # Brain containment: ON whenever the config has a brain asset (don't let a
    # depth-greedy ADAM puncture the brain bottom for coverage).
    brain_sdf = maybe_build_brain_sdf(runtime, compiled)
    print(
        "brain-containment: "
        f"{'ON' if brain_sdf is not None else 'OFF (no brain asset)'}"
    )

    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    results = data["results"]
    n_total = len(results)

    # Group ALL candidates by n_arcs (no padding) and run each group.
    by_arcs: dict[int, list[int]] = {}
    for i in range(n_total):
        by_arcs.setdefault(int(results[i].n_arcs), []).append(i)
    if LIMIT > 0:
        # Keep the manual cand plus a head sample per group for smoke tests.
        for k in by_arcs:
            head = by_arcs[k][:LIMIT]
            if k == 3 and MANUAL not in head:
                head[-1] = MANUAL
            by_arcs[k] = head
    print(
        f"pool {n_total} cands; groups "
        f"{ {k: len(v) for k, v in sorted(by_arcs.items())} }"
    )

    t_all = time.time()
    records = []
    for n_arcs in sorted(by_arcs):
        records += run_group(
            n_arcs,
            by_arcs[n_arcs],
            probes=probes,
            holes=holes,
            data=data,
            sdf_by_name=sdf_by_name,
            bvh_cache=bvh_cache,
            well_obj=well_obj,
            brain_sdf=brain_sdf,
        )
    print(
        f"\nall groups done in {(time.time() - t_all) / 60:.1f} min "
        f"({len(records)} candidates)"
    )

    # Global ranking by basin-selected violation.
    records.sort(key=lambda r: r["viol"])
    for rank, r in enumerate(records):
        r["rank"] = rank
    idx_to_rec = {r["idx"]: r for r in records}

    n_neg = sum(1 for r in records if r["viol"] < 0)
    man = idx_to_rec.get(MANUAL)
    obj_label = "objective (-coverage+penalties)" if COVERAGE else "violation"
    feasibility_label = (
        "NOT a feasibility count with coverage on" if COVERAGE else "soft-feasible"
    )
    print(f"\n=== global basin-selected ranking ({len(records)} cands) ===")
    print(f"sort key = Phase 1 {obj_label}")
    if man is not None:
        print(
            f"Manual (#{MANUAL}): val {man['viol']:+.3f}, "
            f"rank {man['rank'] + 1}/{len(records)}"
        )
    print(f"sort-value < 0: {n_neg}/{len(records)} ({feasibility_label})")

    # FCL on the top-K — REBUILD statics on demand (no global retention).
    print(f"\nFCL on top-{FCL_TOPK} (rebuilding statics per cand):")
    n_feas = 0
    for k in range(min(FCL_TOPK, len(records))):
        r = records[k]
        cand = data["candidates"][r["idx"]]
        st = _build_probe_static(
            probes,
            holes,
            cand.ha,
            cand.aa,
            bvh_cache=bvh_cache,
            sdf_by_name=sdf_by_name,
        )
        v = make_fcl_validator(
            st, r["n_arcs"], fixtures=tuple(fixtures), fixture_bvhs=fixture_bvhs
        )
        fcl = float(np.asarray(v.slacks(r["pose"])).min())
        r["fcl"] = fcl
        feas = fcl >= -1e-4
        r["fcl_feasible"] = bool(feas)
        n_feas += feas
        if k < 15 or r["idx"] == MANUAL:
            tag = " <-- MANUAL" if r["idx"] == MANUAL else ""
            print(
                f"  rank {k + 1:>3}  cand {r['idx']:>5}  n_arcs {r['n_arcs']}  "
                f"viol {r['viol']:>+8.3f}  fcl {fcl:>+7.3f}  "
                f"{'FEAS' if feas else 'infeas'}{tag}"
            )
    print(f"\n  FCL-feasible in top-{FCL_TOPK}: {n_feas}/{FCL_TOPK}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(
            dict(
                records=records,
                config=dict(
                    steps=STEPS,
                    flip_degs=FLIP_DEGS,
                    n_surf=N_SURF,
                    bf16_store=BF16_STORE,
                    chunk=CHUNK,
                    fcl_topk=FCL_TOPK,
                    coverage=COVERAGE,
                    soft_fixtures=["well"],
                    fcl_fixtures=[f.name for f in fixtures],
                ),
                holes_path="scratch/0283-300-04.holes.yml",
                source_pool="scratch/full_polish_0283.pkl",
            ),
            f,
        )
    print(f"\nsaved durable rerank → {OUT_PATH} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
