"""Log per-candidate min-clearance trajectories through reduced→full ADAM.

Tooling to calibrate the early-cull thresholds (mechanism 2): run a sample of
candidates through the two-stage ADAM in SEGMENTS, recording each candidate's
min soft-clearance (mm; the cull metric) after every segment, plus its FINAL
FCL outcome. Offline we then ask: "of candidates below clearance T at segment
S, did ANY become FCL-feasible?" — the largest (T, S) with zero good-candidate
losses is a safe cull.

The sample deliberately includes the known-good candidates (phase2 handoff,
fcl >= -0.2) plus a random junk sample, so we can verify a candidate cull
wouldn't drop an eventual winner.

Seeds mirror ``batched_full_rerank`` (arc/ml from the warm-start, incumbent
spins, offsets/depth 0). Reduced = cov_weight 0 + offsets/depth pinned; full =
cov_weight 1 + all free. Output → scratch/cand_trajectories.pkl.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.log_candidate_trajectories
Env:  SEG=50  N_SEG_R=10  N_SEG_F=10  N_JUNK=200  N_SURF=5000  CHUNK=64  SEED=0
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle

import jax
import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.batched_objective import (
    make_batched_reduced_objective,
)
from aind_low_point.optimization.batched_spin_restore import (
    make_batched_spin_restore_partial,
)
from aind_low_point.optimization.batched_static import build_batched_probe_static
from aind_low_point.optimization.clearance_metrics import make_min_clear_one
from aind_low_point.optimization.joint_rerank import JointWeights, _build_probe_static
from aind_low_point.optimization.optimizer_vars import build_y, extract_spins
from aind_low_point.optimization.pipeline.enumeration import (
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
    maybe_build_brain_sdf,
    phase1_bounds,
)
from aind_low_point.optimization.pipeline.restore import setup, spins_deg_from_reduced
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    Phase1Weights,
)
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.manual_mrv_chain import mrv_seed
from scripts.staged_adam import restore_spins_group

PPV = PHASE1_PER_PROBE_VARS
SEG = int(_os.environ.get("SEG", "50"))
N_SEG_R = int(_os.environ.get("N_SEG_R", "10"))  # reduced segments (×SEG steps)
N_SEG_F = int(_os.environ.get("N_SEG_F", "10"))  # full segments
N_TOP = int(_os.environ.get("N_TOP", "100"))  # ranks after winners, taken whole
N_LOG = int(_os.environ.get("N_LOG", "400"))  # log-sampled from the rank tail
CHUNK = int(_os.environ.get("CHUNK", "64"))
RNG_SEED = int(_os.environ.get("SEED", "0"))
BF16 = _os.environ.get("BF16_STORE", "1") == "1"
# SEED_MODE: "mrv" = MRV emit_seed (arc/ml/spin) → spin restore from it (the new
# seed source); "restore" = restore from the candidate's production ml seed;
# "augmented" = the warm-start's incumbent spins (no restore).
SEED_MODE = _os.environ.get("SEED_MODE", "mrv").lower()
N_SPINS = int(_os.environ.get("N_SPINS", "16"))  # spin-restore basin resolution
RESTORE_CHUNK = int(_os.environ.get("RESTORE_CHUNK", "64"))
OUT = _os.environ.get("OUT", "scratch/cand_trajectories.pkl")


def reduced_lohi(lo, hi, n_arcs, K):
    lo_r, hi_r = lo.copy(), hi.copy()
    for k in range(K):
        for off in (3, 4, 5):
            lo_r[n_arcs + PPV * k + off] = 0.0
            hi_r[n_arcs + PPV * k + off] = 0.0
    return lo_r, hi_r


def restore_spins_mrv(n_arcs, idxs, *, probes, holes, data, sdf_by_name, well, enum):
    """MRV seed → batched spin restore. For each config, ``emit_seed`` gives
    (arc_aps, ml, spin) from the atlas anchors; the round-robin restore (N_SPINS
    basins) refines spin from there. Returns per-cand (arc_aps, ml-array,
    restored-spin-deg) — the MRV-seeded pose to start reduced ADAM from. Configs
    whose ``emit_seed`` returns None fall back to the candidate's production seed
    (counted in ``n_fail``)."""
    K = len(probes)
    names = [p.name for p in probes]
    cands = [data["candidates"][idx] for idx in idxs]
    arc_aps_l, ml_l, y0_l, n_fail = [], [], [], 0
    for cand in cands:
        res = None
        try:
            res = mrv_seed(cand, enum, n_arcs)
        except Exception:
            res = None
        if res is None:
            n_fail += 1
            arc_aps = np.zeros(n_arcs)
            for a in range(min(n_arcs, len(cand.aa.arc_centroids_deg))):
                arc_aps[a] = float(cand.aa.arc_centroids_deg[a])
            ml_seed, spin_seed = dict(cand.ml_seed), dict(cand.spin_seed)
        else:
            arc_aps, ml_seed, spin_seed, _gap = res
        y0 = np.zeros(n_arcs + 3 * K, np.float32)
        y0[:n_arcs] = arc_aps
        for k, p in enumerate(probes):
            sp = np.deg2rad(float(spin_seed.get(p.name, 0.0)))
            y0[n_arcs + 3 * k] = float(ml_seed.get(p.name, 0.0))
            y0[n_arcs + 3 * k + 1] = float(np.cos(sp))
            y0[n_arcs + 3 * k + 2] = float(np.sin(sp))
        arc_aps_l.append(np.asarray(arc_aps, np.float64))
        ml_l.append(np.array([float(ml_seed.get(n, 0.0)) for n in names]))
        y0_l.append(y0)
    seeds = np.stack(y0_l)

    # Batched restore, chunked (build the JIT once from the first chunk).
    weights = JointWeights()
    fixtures = (well,)
    B = len(idxs)
    bs0 = build_batched_probe_static(
        [(c.ha, c.aa) for c in cands[: min(RESTORE_CHUNK, B)]],
        probes,
        holes,
        n_arcs=n_arcs,
        sdf_by_name=sdf_by_name,
        head_pitch_deg=0.0,
    )
    restore = make_batched_spin_restore_partial(
        bs0, weights, n_spins=N_SPINS, n_rounds=4, fixtures=fixtures
    )
    obj_b, _ = make_batched_reduced_objective(bs0, weights, fixtures)
    spins = []
    for lo in range(0, B, RESTORE_CHUNK):
        hi = min(lo + RESTORE_CHUNK, B)
        bs = (
            bs0
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
        y_r = restore(jnp.asarray(seeds[lo:hi]), *obj_b.extract_arrays(bs))
        y_r.block_until_ready()
        spins.extend(
            spins_deg_from_reduced(np.asarray(y_r[b], np.float64), n_arcs, K)
            for b in range(hi - lo)
        )
    if n_fail:
        print(f"    ({n_fail}/{B} configs fell back to production seed: emit None)")
    return arc_aps_l, ml_l, spins


def run_group(
    n_arcs,
    idxs,
    *,
    probes,
    holes,
    data,
    sdf_by_name,
    bvh,
    well,
    brain_sdf,
    fixtures,
    fixture_bvhs,
    enum=None,
):
    K = len(probes)
    bounds = phase1_bounds(n_arcs, K)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    lo_r, hi_r = reduced_lohi(lo, hi, n_arcs, K)

    names = [p.name for p in probes]
    # Seed source (point 0 = the restore-cull metric):
    #   mrv     — MRV emit_seed (arc/ml/spin) → restore from it
    #   restore — restore from the candidate's production ml seed
    #   augmented — warm-start incumbent spins, no restore
    restored_spins, mrv_arc, mrv_ml, mrv_sp = None, None, None, None
    if SEED_MODE == "mrv":
        mrv_arc, mrv_ml, mrv_sp = restore_spins_mrv(
            n_arcs,
            idxs,
            probes=probes,
            holes=holes,
            data=data,
            sdf_by_name=sdf_by_name,
            well=well,
            enum=enum,
        )
    elif SEED_MODE == "restore":
        restored_spins = restore_spins_group(
            n_arcs,
            idxs,
            probes=probes,
            holes=holes,
            pool=data,
            sdf_by_name=sdf_by_name,
            well=well,
            with_well=True,
        )
    statics_flat, x0_rows = [], []
    for i, idx in enumerate(idxs):
        cand = data["candidates"][idx]
        st = _build_probe_static(
            probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
        )
        if SEED_MODE == "mrv":
            arc_aps, mls, sp = mrv_arc[i], mrv_ml[i], np.asarray(mrv_sp[i])
        elif SEED_MODE == "restore":
            arc_aps = np.zeros(n_arcs)
            for a in range(min(n_arcs, len(cand.aa.arc_centroids_deg))):
                arc_aps[a] = float(cand.aa.arc_centroids_deg[a])
            mls = np.array([float(cand.ml_seed.get(n, 0.0)) for n in names])
            sp = np.asarray(restored_spins[i])
        else:
            xa = np.asarray(data["augmented_phase1_x"][idx], float)
            arc_aps = xa[:n_arcs]
            mls = np.array([xa[n_arcs + PPV * j] for j in range(K)])
            sp = extract_spins(xa, n_arcs, K)
        z = np.zeros(K)
        statics_flat.append(st)
        x0_rows.append(build_y(arc_aps, n_arcs, mls, sp, z, z, z))
    x0 = np.stack(x0_rows).astype(np.float32)

    w = Phase1Weights()
    cov = build_coverage_data(probes, statics_flat[0])
    grid_dtype = jnp.bfloat16 if BF16 else jnp.float32
    _vo, _vg, build_arglist, _ma, make_staged_adam = make_batched_phase1_chunked(
        statics_flat[0],
        n_arcs,
        w,
        (well,),
        coverage_data=cov,
        grid_dtype=grid_dtype,
        brain_sdf=brain_sdf,
    )
    run_staged = make_staged_adam(lr=0.02)

    min_clear_one = make_min_clear_one(n_arcs, K, (well,), w)
    in_axes = (0,) + tuple(0 if k in PER_CAND else None for k in ARG_ORDER)
    min_clear_b = jax.jit(jax.vmap(min_clear_one, in_axes=in_axes))

    # Pad to CHUNK, then segmented reduced→full, logging clearance per segment.
    n = x0.shape[0]
    npad = (-n) % CHUNK
    if npad:
        statics_flat = statics_flat + [statics_flat[-1]] * npad
        x0 = np.concatenate([x0, np.repeat(x0[-1:], npad, 0)], 0)
    full_arglist = build_arglist(statics_flat)
    per_cand = [k in PER_CAND for k in ARG_ORDER]
    ntot = x0.shape[0]

    seg_labels, traj = [], []  # traj: list of (ntot,) min clearance per segment
    x_dev = jnp.asarray(x0, jnp.float32)

    def log(x_all):
        out = np.empty(ntot, np.float32)
        for s in range(0, ntot, CHUNK):
            cargs = [
                a[s : s + CHUNK] if p else a for a, p in zip(full_arglist, per_cand)
            ]
            out[s : s + CHUNK] = np.asarray(min_clear_b(x_all[s : s + CHUNK], *cargs))
        return out

    traj.append(log(x_dev))
    seg_labels.append((SEED_MODE, 0))
    for phase, lo_p, hi_p, cw, nseg in (
        ("reduced", lo_r, hi_r, 0.0, N_SEG_R),
        ("full", lo, hi, 1.0, N_SEG_F),
    ):
        for sidx in range(nseg):
            new = np.empty_like(x0)
            for s in range(0, ntot, CHUNK):
                cargs = [
                    a[s : s + CHUNK] if p else a for a, p in zip(full_arglist, per_cand)
                ]
                xc = run_staged(x_dev[s : s + CHUNK], cargs, lo_p, hi_p, cw, SEG)
                new[s : s + CHUNK] = np.asarray(xc)
            x_dev = jnp.asarray(new, jnp.float32)
            traj.append(log(x_dev))
            seg_labels.append((phase, (sidx + 1) * SEG))

    # Final FCL per candidate (ground-truth feasibility).
    x_final = np.asarray(x_dev)[:n]
    fcl_min = np.empty(n, np.float64)
    for i in range(n):
        v = make_fcl_validator(
            statics_flat[i], n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fixture_bvhs
        )
        s = np.asarray(v.slacks(x_final[i].astype(np.float64)))
        fcl_min[i] = float(s.min()) if s.size else 0.0
    traj = np.stack(traj)[:, :n]  # (n_seg, n_cand) min clearance, ALL candidates
    return seg_labels, traj, fcl_min


def select_sample(data):
    """The deterministic calibration sample: all known-good winners (phase2
    handoff fcl >= -0.2) + N_TOP hardest near-misses + N_LOG log-sampled tail.
    Returns (idxs, good, top, log_s) — reused by the thick-well A/B so both
    runs score the identical candidate set."""
    h2 = pickle.load(open("scratch/phase2_handoff.pkl", "rb"))
    good = [r["idx"] for r in h2["all"] if r["fcl"] >= -0.2]
    good_set = set(good)
    # Distractors selected by the OLD run's rank (viol objective, rank 0 = best),
    # excluding the winners: the next N_TOP taken whole (the hardest near-misses),
    # then N_LOG LOG-sampled from the tail (dense near the top, sparse at depth).
    rr = pickle.load(open("scratch/full_rerank_0283.pkl", "rb"))
    ranked = [
        r["idx"]
        for r in sorted(rr["records"], key=lambda r: r["rank"])
        if r["idx"] not in good_set
    ]
    top = ranked[:N_TOP]
    rest = ranked[N_TOP:]
    if len(rest) <= N_LOG:
        log_s = list(rest)
    else:
        pos = np.unique(np.round(np.geomspace(1, len(rest), N_LOG)).astype(int)) - 1
        pos = pos[(pos >= 0) & (pos < len(rest))]
        if len(pos) < N_LOG:  # dedup shrank it → fill from the unused ranks
            extra = np.setdiff1d(np.arange(len(rest)), pos)
            pad = np.random.default_rng(RNG_SEED).choice(
                extra, N_LOG - len(pos), replace=False
            )
            pos = np.sort(np.concatenate([pos, pad]))
        log_s = [rest[i] for i in pos[:N_LOG]]
    idxs = sorted(good_set | set(top + log_s))
    return idxs, good, top, log_s


def main() -> int:
    cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    data = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    brain = maybe_build_brain_sdf(rt, compile_all_transforms(cfg.transforms))

    idxs, good, top, log_s = select_sample(data)
    print(
        f"sample: {len(good)} winners + {len(top)} top(rank≤{N_TOP + len(good)}) + "
        f"{len(log_s)} log-sampled = {len(idxs)} cands; seed={SEED_MODE}; "
        f"reduced {N_SEG_R}×{SEG} → full {N_SEG_F}×{SEG}, bf16={BF16}"
    )

    enum = None
    if SEED_MODE == "mrv":
        enum = Enumerator(*build_or_load_atlas(), ml_margin_deg=0.0, ml_mode="greedy")

    by_arcs: dict[int, list[int]] = {}
    for idx in idxs:
        by_arcs.setdefault(int(data["results"][idx].n_arcs), []).append(idx)

    seg_labels = None
    all_ids, all_traj, all_fcl, all_good = [], [], [], []
    for n_arcs in sorted(by_arcs):
        g = by_arcs[n_arcs]
        print(f"[n_arcs={n_arcs}] {len(g)} cands...", flush=True)
        sl, traj, fcl_min = run_group(
            n_arcs,
            g,
            probes=probes,
            holes=holes,
            data=data,
            sdf_by_name=sdf_by_name,
            bvh=bvh,
            well=well,
            brain_sdf=brain,
            fixtures=fixtures,
            fixture_bvhs=fixture_bvhs,
            enum=enum,
        )
        seg_labels = sl
        all_ids += g
        all_traj.append(traj)
        all_fcl.append(fcl_min)
        all_good += [idx in set(good) for idx in g]

    traj = np.concatenate(all_traj, axis=1)  # (n_seg, n_cand)
    fcl = np.concatenate(all_fcl)
    ids = np.array(all_ids)
    is_good = np.array(all_good)
    feasible = fcl >= -1e-4

    with open(OUT, "wb") as f:
        pickle.dump(
            dict(
                seg_labels=seg_labels,
                ids=ids,
                traj_min_clear=traj,
                final_fcl=fcl,
                feasible=feasible,
                was_known_good=is_good,
                seg=SEG,
            ),
            f,
        )
    print(f"\nsaved → {OUT}  traj shape {traj.shape} (segments × candidates)")
    print(
        f"final FCL-feasible: {int(feasible.sum())}/{len(ids)} "
        f"(known-good in sample: {int(is_good.sum())})"
    )

    # Quick safe-cull preview: at each segment, the most negative clearance among
    # the EVENTUAL-feasible candidates = the deepest a winner ever sits ⇒ any
    # cull below that at that segment is provably safe on this sample.
    if feasible.any():
        print(
            f"\n{'segment':>16} {'min clear of winners (mm)':>26}  "
            f"{'junk below it (cullable)':>24}"
        )
        for si, (ph, stp) in enumerate(seg_labels):
            win = traj[si, feasible]
            floor = float(win.min())
            cullable = int(np.sum((traj[si] < floor) & ~feasible))
            print(
                f"{ph}@{stp:<4} ({si:>2})".rjust(16)
                + f" {floor:>26.3f}  {cullable:>20}/{int((~feasible).sum())}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
