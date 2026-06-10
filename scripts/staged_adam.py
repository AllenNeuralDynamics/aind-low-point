"""Two-stage ADAM (NO L-BFGS) on the 45 stage-3-feasible candidates, GROUPED.

Pipeline per candidate, seeded from the NEW MRV greedy-stab ml + restore-with-well
spins (atlas arc AP centroids):

  Stage 1  reduced DOF, clearance-first   : coverage OFF, offsets/depth PINNED to 0
                                            (arc/ml/spin free), STAGE1 ADAM steps.
  Stage 2  full DOF, coverage-aware       : coverage ON, all DOFs free, STAGE2 steps.

GROUPED for compile/throughput: candidates are bucketed by ``n_arcs`` (the x
layout differs), and per group we build each kernel ONCE and run all candidates
as a single batched ADAM pass (the spin restore is likewise batched over the
group). This is the ``batched_full_rerank.run_group`` pattern — vmapped vobj /
make_adam over a stacked, device-resident arglist — so the GPU saturates instead
of paying a per-candidate re-trace/re-compile at batch-1.

Replaces BOTH L-BFGS calls (reduced + offset-augment) with a reduced-ADAM stage,
then hands to full ADAM. Reports FCL (all-fixture min slack), coverage, and the
durable stored pose (restore→L-BFGS→augment→ADAM) as a baseline.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.staged_adam
Env:  STAGE1=500  STAGE2=500  S1_WELL=1  IDXS=<override the 45 feasibles>
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle

import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.arc_first_principled import emit_seed
from aind_low_point.optimization.batched_objective import (
    make_batched_reduced_objective,
)
from aind_low_point.optimization.batched_spin_restore import (
    make_batched_spin_restore_partial,
)
from aind_low_point.optimization.batched_static import build_batched_probe_static
from aind_low_point.optimization.coverage_jax import coverage_total_over_probes
from aind_low_point.optimization.joint_rerank import JointWeights, _build_probe_static
from aind_low_point.optimization.optimizer_vars import _poses
from aind_low_point.optimization.pipeline.enumeration import (
    MIN_ARC_AP_SEP_DEG,
    MIN_ML_SEP_DEG,
    Enumerator,
    build_or_load_atlas,
)
from aind_low_point.optimization.pipeline.phase1_build import (
    ARG_ORDER,
    PER_CAND,
    make_batched_phase1_chunked,
)
from aind_low_point.optimization.pipeline.phase1_geometry import (
    build_coverage_data,
    phase1_bounds,
)
from aind_low_point.optimization.pipeline.restore import (
    build_y,
    enum_seed_y0,
    setup,
    spins_deg_from_reduced,
)
from aind_low_point.optimization.stage3_phase1_jax import Phase1Weights
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator

PPV = 6
STAGE1 = int(_os.environ.get("STAGE1", "500"))
STAGE2 = int(_os.environ.get("STAGE2", "500"))
# S1_WELL=1 keeps the well fixture in the reduced Stage-1 objective (default);
# S1_WELL=0 drops it, to test whether well-in-reduced changes the outcome.
S1_WELL = _os.environ.get("S1_WELL", "1") == "1"
CHUNK = int(_os.environ.get("CHUNK", "64"))
RESTORE_CHUNK = int(_os.environ.get("RESTORE_CHUNK", "64"))
RESTORE_ROUNDS = int(_os.environ.get("RESTORE_ROUNDS", "4"))
PIPELINE_DEPTH = int(_os.environ.get("PIPELINE_DEPTH", "2"))
FCL_TOL = -1e-4


def emit_ml_seed(enum, probe_to_hole, probe_to_arc_idx, arc_centroids):
    """ml seed for a fixed (hole, arc) decision via the shared ``emit_seed``
    source of truth (convex isotonic arc-AP + MRV/CSP ML-anchor pick).

    Builds one ``emit_seed`` arc per arc index — members ``(p, h, name)``, AP
    window = member AP-envelope intersection, desired AP = the arc centroid —
    and returns just the per-probe-name ml component (spins come from the spin
    restore, arc APs from the centroids, so only ml is consumed here).
    """
    arcs: dict[int, list[tuple[int, int, str]]] = {}
    for name, ai in probe_to_arc_idx.items():
        p = enum.names.index(name)
        arcs.setdefault(ai, []).append((p, probe_to_hole[name], name))
    seed_arcs = []
    for ai in sorted(arcs):
        members = arcs[ai]
        lo = max(enum.arr.ap_min_max[(p, h)][0] for p, h, _ in members)
        hi = min(enum.arr.ap_min_max[(p, h)][1] for p, h, _ in members)
        desired = (
            float(arc_centroids[ai]) if ai < len(arc_centroids) else 0.5 * (lo + hi)
        )
        seed_arcs.append(
            {"members": members, "ap_lo": lo, "ap_hi": hi, "ap_desired": desired}
        )
    _aps, ml_seed, _spin, _gap = emit_seed(
        seed_arcs,
        enum.arr,
        min_arc_ap_sep_deg=MIN_ARC_AP_SEP_DEG,
        min_ml_sep_deg=MIN_ML_SEP_DEG,
    )
    return ml_seed


def reduced_bounds(n_arcs, K):
    """Phase-1 bounds with offsets (off_R, off_A) and depth pinned to (0, 0)."""
    b = list(phase1_bounds(n_arcs, K))
    for k in range(K):
        for off in (3, 4, 5):
            b[n_arcs + PPV * k + off] = (0.0, 0.0)
    return b


def restore_spins_group(
    n_arcs, idxs, *, probes, holes, pool, sdf_by_name, well, with_well
):
    """Batched round-robin spin restore over a whole n_arcs group, CHUNKED over
    candidates (mirrors parallel_stage2's spin-restore loop). The restore JIT is
    built once from the first chunk's probe-set constants; per-chunk bs flows as
    runtime args so one compile serves every same-shape chunk. Chunking caps the
    ``(n_arcs, n_spins, n_cand, n_surf)`` intermediate that OOMs on big groups.
    Returns a list (parallel to idxs) of per-probe spin-degree vectors."""
    K = len(probes)
    weights = JointWeights()
    fixtures = (well,) if with_well else ()
    cands = [pool["candidates"][idx] for idx in idxs]
    seeds = np.stack([enum_seed_y0(c, probes, n_arcs) for c in cands])
    B = len(idxs)

    # Build the restore JIT + objective once from the first chunk's bs.
    hi0 = min(RESTORE_CHUNK, B)
    probe_set_bs = build_batched_probe_static(
        [(c.ha, c.aa) for c in cands[:hi0]],
        probes,
        holes,
        n_arcs=n_arcs,
        sdf_by_name=sdf_by_name,
        head_pitch_deg=0.0,
    )
    restore = make_batched_spin_restore_partial(
        probe_set_bs, weights, n_spins=8, n_rounds=RESTORE_ROUNDS, fixtures=fixtures
    )
    obj_batched, _ = make_batched_reduced_objective(probe_set_bs, weights, fixtures)
    out: list = []
    for lo in range(0, B, RESTORE_CHUNK):
        hi = min(lo + RESTORE_CHUNK, B)
        bs_chunk = (
            probe_set_bs
            if lo == 0
            else build_batched_probe_static(
                [(c.ha, c.aa) for c in cands[lo:hi]],
                probes,
                holes,
                n_arcs=n_arcs,
                sdf_by_name=sdf_by_name,
                head_pitch_deg=0.0,
            )
        )
        varying = obj_batched.extract_arrays(bs_chunk)
        y_r = restore(jnp.asarray(seeds[lo:hi]), *varying)
        y_r.block_until_ready()
        out.extend(
            spins_deg_from_reduced(np.asarray(y_r[b], np.float64), n_arcs, K)
            for b in range(hi - lo)
        )
    return out


def adam_pass(statics_flat, x0, n_arcs, *, coverage_data, well_obj, bounds, steps):
    """One batched, chunked, pipelined ADAM pass over a group's rows. Builds the
    kernel once and runs every row as a stacked device-resident batch."""
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    fixtures = () if well_obj is None else (well_obj,)
    _vobj, _vg, build_arglist, make_adam, _mks = make_batched_phase1_chunked(
        statics_flat[0],
        n_arcs,
        Phase1Weights(),
        fixtures,
        coverage_data=coverage_data,
        grid_dtype=jnp.float32,
    )
    run_adam = make_adam(lo, hi, steps=steps, lr=0.02)

    n_rows = x0.shape[0]
    n_pad = (-n_rows) % CHUNK
    sf = statics_flat + [statics_flat[-1]] * n_pad if n_pad else statics_flat
    x0p = (
        np.concatenate([x0, np.repeat(x0[-1:], n_pad, 0)], 0) if n_pad else x0
    ).astype(np.float32)
    full_arglist = build_arglist(sf)
    per_cand_pos = [k in PER_CAND for k in ARG_ORDER]
    x0_dev = jnp.asarray(x0p, jnp.float32)
    n_tot = x0p.shape[0]

    xout = np.zeros((n_tot, x0p.shape[1]), np.float32)
    inflight: list = []

    def drain(item):
        s0, xa_f = item
        xout[s0 : s0 + CHUNK] = np.asarray(xa_f)

    for s in range(0, n_tot, CHUNK):
        cargs = [
            a[s : s + CHUNK] if p else a for a, p in zip(full_arglist, per_cand_pos)
        ]
        xa = run_adam(x0_dev[s : s + CHUNK], cargs)
        inflight.append((s, xa))
        if len(inflight) > PIPELINE_DEPTH:
            drain(inflight.pop(0))
    for item in inflight:
        drain(item)
    return xout[:n_rows]


def main() -> int:
    _cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    K = len(probes)
    names = [p.name for p in probes]
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    h2 = pickle.load(open("scratch/phase2_handoff.pkl", "rb"))
    rer = pickle.load(open("scratch/full_rerank_0283.pkl", "rb"))
    stored = {r["idx"]: r for r in rer["records"]}

    env_idxs = _os.environ.get("IDXS")
    if env_idxs:
        idxs = [int(x) for x in env_idxs.split(",")]
    else:
        idxs = [r["idx"] for r in h2["all"] if r["fcl"] >= -0.2]

    atlas_payload = build_or_load_atlas()
    enum = Enumerator(
        atlas_payload.atlas,
        atlas_payload.probe_names,
        ml_margin_deg=0.0,
        ml_mode="greedy",
    )

    by_arcs: dict[int, list[int]] = {}
    for idx in idxs:
        by_arcs.setdefault(int(pool["results"][idx].n_arcs), []).append(idx)

    groups_str = ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_arcs.items()))
    print(
        f"staged ADAM (reduced {STAGE1} → full {STAGE2}); NO L-BFGS; "
        f"grouped by n_arcs {{ {groups_str} }}"
    )
    print(
        f"seed = MRV greedy ml + restore-with-well spins; well-in-reduced={S1_WELL}\n"
    )

    rows = []
    for n_arcs in sorted(by_arcs):
        g = by_arcs[n_arcs]
        print(
            f"[n_arcs={n_arcs}] {len(g)} cands: restore → reduced ADAM → full ADAM ...",
            flush=True,
        )
        spins = restore_spins_group(
            n_arcs,
            g,
            probes=probes,
            holes=holes,
            pool=pool,
            sdf_by_name=sdf_by_name,
            well=well,
            with_well=True,
        )
        statics_flat, x0_rows = [], []
        for idx, sp in zip(g, spins):
            cand = pool["candidates"][idx]
            st = _build_probe_static(
                probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
            )
            ml_map = emit_ml_seed(
                enum,
                cand.ha.probe_to_hole,
                cand.aa.probe_to_arc_idx,
                cand.aa.arc_centroids_deg,
            )
            mls = np.array([ml_map[n] for n in names])
            arc_aps = np.zeros(n_arcs)
            for a in range(min(n_arcs, len(cand.aa.arc_centroids_deg))):
                arc_aps[a] = float(cand.aa.arc_centroids_deg[a])
            zero = np.zeros(K)
            statics_flat.append(st)
            x0_rows.append(build_y(arc_aps, n_arcs, mls, sp, zero, zero, zero))
        x0 = np.stack(x0_rows).astype(np.float32)
        cov_data = build_coverage_data(probes, statics_flat[0])

        x1 = adam_pass(
            statics_flat,
            x0,
            n_arcs,
            coverage_data=None,
            well_obj=well if S1_WELL else None,
            bounds=reduced_bounds(n_arcs, K),
            steps=STAGE1,
        )
        x2 = adam_pass(
            statics_flat,
            x1,
            n_arcs,
            coverage_data=cov_data,
            well_obj=well,
            bounds=phase1_bounds(n_arcs, K),
            steps=STAGE2,
        )

        for ci, idx in enumerate(g):
            st = statics_flat[ci]
            v = make_fcl_validator(
                st, n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fixture_bvhs
            )

            def fcl(x, v=v):
                return float(np.asarray(v.slacks(np.asarray(x, np.float64))).min())

            def cov(x, st=st, n_arcs=n_arcs):
                Rs, ts, tips, mask = _poses(st, x, n_arcs)
                return float(
                    coverage_total_over_probes(Rs, ts, tips, mask, cov_data, 41)
                )

            f1, c1 = fcl(x1[ci]), cov(x1[ci])
            f2, c2 = fcl(x2[ci]), cov(x2[ci])
            sp_pose = np.asarray(stored[idx]["pose"], np.float64)
            rows.append(
                dict(
                    idx=idx,
                    n_arcs=n_arcs,
                    s1_fcl=f1,
                    s1_cov=c1,
                    s2_fcl=f2,
                    s2_cov=c2,
                    feas=bool(f2 >= FCL_TOL),
                    dur_fcl=fcl(sp_pose),
                    dur_cov=cov(sp_pose),
                )
            )

    rows.sort(key=lambda r: idxs.index(r["idx"]))
    print(
        f"\n{'idx':>5} | {'S1: fcl   cov':>14} | {'S2: fcl   cov  feas':>20} | "
        f"{'durable: fcl  cov':>18}"
    )
    for r in rows:
        print(
            f"{r['idx']:>5} | {r['s1_fcl']:>+7.3f} {r['s1_cov']:>6.2f} | "
            f"{r['s2_fcl']:>+7.3f} {r['s2_cov']:>6.2f} "
            f"{'FEAS' if r['feas'] else 'infes':>5} | "
            f"{r['dur_fcl']:>+7.3f} {r['dur_cov']:>6.2f}"
        )

    n = len(rows)
    n_feas = sum(r["feas"] for r in rows)
    dur_feas = sum(r["dur_fcl"] >= FCL_TOL for r in rows)
    win = sum(r["feas"] and r["s2_cov"] > r["dur_cov"] + 0.5 for r in rows)
    print(f"\n=== staged ADAM feasible: {n_feas}/{n} (well-in-reduced={S1_WELL}) ===")
    print(f"    durable stored feasible:   {dur_feas}/{n}")
    print(f"    staged feas & higher cov:  {win}/{n}")
    tag = "well" if S1_WELL else "nowell"
    with open(f"scratch/staged_adam_{tag}.pkl", "wb") as f:
        pickle.dump(
            {"rows": rows, "stage1": STAGE1, "stage2": STAGE2, "s1_well": S1_WELL}, f
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
