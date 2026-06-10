"""Run the FULL MRV pool through restore → reduced 500 → full 500 (TUNED).

Enumerates the MRV pool (3 arcs, <=4 probes/arc; ~19k candidates), seeds each
from the joint ``emit_seed`` (arc/ml/spin) via ``Enumerator.seed``, spin-restores
(N_SPINS=16), then runs each candidate through a two-stage (reduced→full)
optimization and a per-candidate FCL gate. Output → OUT (then cull/select/
trust-constr from there).

The optimizer is TUNED (see dev memory: well_sdf_thin_skin_thickening,
adam_moment_restart_schedule, coarse_fine_surf_tuning):
  - WELL=thick   — solidified well SDF (fixes the thin-skin false-negative);
                   FCL still uses the true thin mesh (honest gate).
  - MINIMIZER=rprop — sign-based iRprop− (no ADAM v-freeze; the winning minimizer).
  - coarse→fine surf — reduced/full each run (STAGE-REDUCED_FINE/FULL_FINE) steps
                   @COARSE_N surf then the FINE_* finish @5000 (the homotopy win).

Two documented presets (copy-paste commands in dev/POOL_RUN_CONFIGS.md):
  THROUGHPUT (default): COARSE_N=1000, REDUCED_FINE=FULL_FINE=50   (~2.16x; 545:105/20)
  YIELD:                COARSE_N=3000, REDUCED_FINE=FULL_FINE=100  (~1.31x; 545:123/21)
  BASELINE:             MINIMIZER=adam_const WELL=thin COARSE_N=5000  (old 165-feas run)

Per candidate saves the final + reduced-checkpoint pose, min dual-rep clearance
(the cull metric), coverage, the discrete decision, and the MRV seed gap.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 alp-phase1
Env:  MINIMIZER=rprop WELL=thick COARSE_N=1000 REDUCED_FINE=50 FULL_FINE=50
      STAGE1=500 STAGE2=500 N_SPINS=16 CHUNK=256 RESTORE_CHUNK=128 FCL_TOPK=300
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import gc
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.objectives.batched_reduced import (
    make_batched_reduced_objective,
)
from aind_low_point.optimization.objectives.batched_static import (
    build_batched_probe_static,
)
from aind_low_point.optimization.objectives.clearance_metrics import make_min_clear_one
from aind_low_point.optimization.objectives.fcl_validator import make_fcl_validator
from aind_low_point.optimization.objectives.phase1 import Phase1Weights
from aind_low_point.optimization.objectives.probe_static import (
    JointWeights,
    _build_probe_static,
)
from aind_low_point.optimization.objectives.spin_restore import (
    make_batched_spin_restore_partial,
)
from aind_low_point.optimization.objectives.variables import build_y
from aind_low_point.optimization.pipeline.contracts import (
    ArglistBuilder,
    EnumeratorCandidate,
    MRVArcAssignment,
    MRVHoleAssignment,
    Partition,
    Phase1PoolPayload,
    Phase1PoolRecord,
    ProbeToHole,
    SeedMap,
)
from aind_low_point.optimization.pipeline.enumeration import (
    Enumerator,
    build_or_load_atlas,
)
from aind_low_point.optimization.pipeline.phase1_build import (
    ARG_ORDER,
    PER_CAND,
    build_cw_fns,
    make_batched_phase1_chunked,
    make_staged_rprop,
)
from aind_low_point.optimization.pipeline.phase1_geometry import (
    build_coverage_data,
    phase1_bounds,
)
from aind_low_point.optimization.pipeline.restore import (
    PPV,
    setup,
    setup_runtime,
    spins_deg_from_reduced,
)
from aind_low_point.planning import AP_LIMIT_DEG

STAGE1 = int(_os.environ.get("STAGE1", "500"))
STAGE2 = int(_os.environ.get("STAGE2", "500"))
N_SPINS = int(_os.environ.get("N_SPINS", "16"))
RESTORE_ROUNDS = int(_os.environ.get("RESTORE_ROUNDS", "4"))
CHUNK = int(_os.environ.get("CHUNK", "256"))
RESTORE_CHUNK = int(_os.environ.get("RESTORE_CHUNK", "128"))
PIPELINE_DEPTH = int(_os.environ.get("PIPELINE_DEPTH", "2"))
FCL_TOPK = int(_os.environ.get("FCL_TOPK", "300"))
LIMIT = int(_os.environ.get("LIMIT", "0"))
MAX_ARCS = int(_os.environ.get("MAX_ARCS", "3"))
MAX_PPA = int(_os.environ.get("MAX_PROBES_PER_ARC", "4"))
# ONLY_NARCS=N restricts this invocation to a single n_arcs group. Lets a bash
# loop run one group per PROCESS (fresh GPU each time) so the BFC allocator can't
# fragment across groups — the cause of the cross-group restore OOM. The resume
# logic (records keyed by n_arcs) accumulates groups across invocations.
ONLY_NARCS = int(_os.environ.get("ONLY_NARCS", "0"))
BF16 = _os.environ.get("BF16_STORE", "1") == "1"
PROGRESS_EVERY = int(_os.environ.get("PROGRESS_EVERY", "25"))
OUT = _os.environ.get("OUT", "scratch/mrv_pool_results.pkl")
# Seed cache is SUBJECT-SPECIFIC (enumerated candidates depend on the subject's
# config); default keys off the config stem so subjects never share seeds.
_CFG_STEM = _os.path.splitext(
    _os.path.basename(_os.environ.get("CONFIG", "examples/836656-config-T12.yml"))
)[0]
SEED_CACHE = _os.environ.get("SEED_CACHE", f"scratch/mrv_seeds_{_CFG_STEM}.pkl")

# Tuned-optimizer knobs (defaults = the THROUGHPUT preset).
MINIMIZER = _os.environ.get(
    "MINIMIZER", "rprop"
).lower()  # rprop|moment_restart|adam_const
WELL_MODE = _os.environ.get("WELL", "thick").lower()  # thick|thin
# Coverage normalization: divide each probe's coverage by its achievable ceiling
# (so shank-count / area / σ / density weigh equally), blend average vs worst
# region by COV_ALPHA in [0,1] (0 = pure average, 1 = pure minimax laggard), and
# apply per-target priority weights (from the target spec's ``coverage_weight``,
# overridable via COVERAGE_WEIGHTS env). COV_WEIGHT is the overall coverage gain
# vs clearance in the full stage (coverage is now a [0,1] scalar, count-free).
COV_NORM = _os.environ.get("COV_NORM", "0") == "1"
COV_ALPHA = float(_os.environ.get("COV_ALPHA", "0.2"))
COV_WEIGHT = float(_os.environ.get("COV_WEIGHT", "1.0"))
# Print the normalization summary once (not once per arc-group).
_group_log_once = [True]
COARSE_N = int(
    _os.environ.get("COARSE_N", "1000")
)  # coarse surf count (5000 = single-fidelity)
REDUCED_FINE = int(
    _os.environ.get("REDUCED_FINE", "50")
)  # fine @5000 steps ending the reduced stage
FULL_FINE = int(
    _os.environ.get("FULL_FINE", "50")
)  # fine @5000 steps ending the full stage
TWO_FIDELITY = COARSE_N < 5000


@dataclass
class MRVCand:
    ha: MRVHoleAssignment
    aa: MRVArcAssignment
    ml_seed: SeedMap
    spin_seed: SeedMap
    min_ml_gap: float
    n_arcs: int
    probe_to_hole: ProbeToHole
    partition: Partition


def wrap(cand_dict: EnumeratorCandidate, enum: Enumerator) -> MRVCand | None:
    """MRV dict → candidate with .ha/.aa stand-ins (interface-compatible with
    _build_probe_static + the optimizer). Returns None if emit_seed fails."""
    res = enum.seed(cand_dict)
    if res is None:
        return None
    arc_aps, ml_seed, spin_seed, gap = res
    groups = list(cand_dict["partition"])  # same iteration order seed() used
    p2a = {name: ai for ai, grp in enumerate(groups) for name in grp}
    ha = MRVHoleAssignment(probe_to_hole=cand_dict["probe_to_hole"])
    aa = MRVArcAssignment(probe_to_arc_idx=p2a, arc_centroids_deg=list(arc_aps))
    return MRVCand(
        ha,
        aa,
        dict(ml_seed),
        dict(spin_seed),
        float(gap),
        len(groups),
        cand_dict["probe_to_hole"],
        cand_dict["partition"],
    )


def reduced_lohi(
    lo: np.ndarray, hi: np.ndarray, n_arcs: int, K: int
) -> tuple[np.ndarray, np.ndarray]:
    lo_r, hi_r = lo.copy(), hi.copy()
    for k in range(K):
        for off in (3, 4, 5):
            lo_r[n_arcs + PPV * k + off] = 0.0
            hi_r[n_arcs + PPV * k + off] = 0.0
    return lo_r, hi_r


def restore_group(
    n_arcs,
    cands: list[MRVCand],
    *,
    probes,
    holes,
    sdf_by_name,
    well,
    head_pitch_deg=0.0,
):
    """Batched spin restore seeded from each cand's MRV ml/spin. Returns per-cand
    restored spin-degrees (parallel to cands)."""
    K = len(probes)
    names = [p.name for p in probes]
    weights = JointWeights()
    fixtures = (well,)
    seed_rows: list[np.ndarray] = []
    for c in cands:
        y0 = np.zeros(n_arcs + 3 * K, np.float32)
        for a in range(min(n_arcs, len(c.aa.arc_centroids_deg))):
            y0[a] = float(c.aa.arc_centroids_deg[a])
        for k, p in enumerate(probes):
            sp = np.deg2rad(float(c.spin_seed.get(p.name, 0.0)))
            y0[n_arcs + 3 * k] = float(c.ml_seed.get(p.name, 0.0))
            y0[n_arcs + 3 * k + 1] = float(np.cos(sp))
            y0[n_arcs + 3 * k + 2] = float(np.sin(sp))
        seed_rows.append(y0)
    seeds = np.stack(seed_rows)
    B = len(cands)
    initial_pairs = cast(Any, [(c.ha, c.aa) for c in cands[: min(RESTORE_CHUNK, B)]])
    bs0 = build_batched_probe_static(
        initial_pairs,
        probes,
        holes,
        n_arcs=n_arcs,
        sdf_by_name=sdf_by_name,
        head_pitch_deg=head_pitch_deg,
    )
    restore = make_batched_spin_restore_partial(
        bs0, weights, n_spins=N_SPINS, n_rounds=RESTORE_ROUNDS, fixtures=fixtures
    )
    obj_b, _ = make_batched_reduced_objective(bs0, weights, fixtures)
    obj_b = cast(Any, obj_b)
    out: list[np.ndarray] = []
    for lo in range(0, B, RESTORE_CHUNK):
        hi = min(lo + RESTORE_CHUNK, B)
        bs = (
            bs0
            if lo == 0
            else build_batched_probe_static(
                cast(Any, [(c.ha, c.aa) for c in cands[lo:hi]]),
                probes,
                holes,
                n_arcs=n_arcs,
                sdf_by_name=sdf_by_name,
                head_pitch_deg=head_pitch_deg,
            )
        )
        y_r = restore(jnp.asarray(seeds[lo:hi]), *obj_b.extract_arrays(bs))
        y_r.block_until_ready()
        out.extend(
            spins_deg_from_reduced(np.asarray(y_r[b], np.float64), n_arcs, K)
            for b in range(hi - lo)
        )
    _ = names
    return out


def make_runner(mkad: Callable[..., Callable], vgrad_cw) -> Callable:
    """run(x0, arglist, lo, hi, cov_weight, n_steps) for the chosen MINIMIZER."""
    if MINIMIZER == "rprop":
        return make_staged_rprop(vgrad_cw, eta0_frac=0.02, etamax_frac=0.5)
    if MINIMIZER == "moment_restart":
        return mkad(lr=0.02, b2=0.999, schedule="moment_restart", period=50)
    return mkad(lr=0.02, b2=0.999, schedule="const")  # adam_const (old baseline)


def _kernel(
    st0, n_arcs, cov, well_soft, brain_sdf, grid_dtype, ceilings=None, cov_weights=None
) -> tuple[Callable, ArglistBuilder, Callable]:
    """Build (vobj, build_arglist, run) for one fidelity's template statics."""
    weights = Phase1Weights(cov_alpha=COV_ALPHA) if COV_NORM else Phase1Weights()
    vobj, _vg, barg, _ma, mkad = make_batched_phase1_chunked(
        st0,
        n_arcs,
        weights,
        (well_soft,),
        coverage_data=cov,
        grid_dtype=grid_dtype,
        brain_sdf=brain_sdf,
        coverage_ceilings=ceilings,
        coverage_weights=cov_weights,
    )
    vg = (
        build_cw_fns(
            st0,
            n_arcs,
            cov,
            well_soft,
            brain_sdf,
            weights=weights,
            coverage_ceilings=ceilings,
            coverage_weights=cov_weights,
        )[1]
        if MINIMIZER == "rprop"
        else None
    )
    return vobj, barg, make_runner(mkad, vg)


def run_group(  # noqa: C901
    n_arcs: int,
    cands: list[MRVCand],
    *,
    probes,
    holes,
    sdf_fine,
    sdf_coarse,
    bvh,
    well_soft,
    brain_sdf,
    head_pitch_deg=0.0,
):
    K = len(probes)
    names = [p.name for p in probes]
    bounds = phase1_bounds(n_arcs, K, head_pitch_deg)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    lo_r, hi_r = reduced_lohi(lo, hi, n_arcs, K)

    t0 = time.time()
    spins = restore_group(
        n_arcs,
        cands,
        probes=probes,
        holes=holes,
        sdf_by_name=sdf_fine,
        well=well_soft,
        head_pitch_deg=head_pitch_deg,
    )
    print(f"  restore {time.time() - t0:.1f}s", flush=True)

    st_f, st_c, x0_rows = [], [], []
    for c, sp in zip(cands, spins):
        st_f.append(
            _build_probe_static(
                probes,
                holes,
                cast(Any, c.ha),
                cast(Any, c.aa),
                bvh_cache=bvh,
                sdf_by_name=sdf_fine,
            )
        )
        if TWO_FIDELITY:
            st_c.append(
                _build_probe_static(
                    probes,
                    holes,
                    cast(Any, c.ha),
                    cast(Any, c.aa),
                    bvh_cache=bvh,
                    sdf_by_name=sdf_coarse,
                )
            )
        arc_aps = np.zeros(n_arcs)
        for a in range(min(n_arcs, len(c.aa.arc_centroids_deg))):
            arc_aps[a] = float(c.aa.arc_centroids_deg[a])
        mls = np.array([float(c.ml_seed.get(n, 0.0)) for n in names])
        z = np.zeros(K)
        x0_rows.append(build_y(arc_aps, n_arcs, mls, np.asarray(sp), z, z, z))
    x0 = np.stack(x0_rows).astype(np.float32)

    cov = build_coverage_data(probes, st_f[0])
    # Per-region normalization: ceilings (achievable per-probe coverage) +
    # per-target priority weights are per-probe-fixed (target/σ/density/geometry),
    # so compute once per group from any candidate's statics. Only active when
    # COV_NORM is set; otherwise pass None ⇒ legacy plain-sum coverage.
    ceilings, cov_weights = None, None
    if COV_NORM:
        from aind_low_point.optimization.objectives.coverage import (
            coverage_ceiling_per_probe,
        )

        ceilings = tuple(float(c) for c in coverage_ceiling_per_probe(st_f[0], cov))
        cov_weights = tuple(float(p.coverage_weight) for p in probes)
        if _group_log_once[0]:
            print(
                f"  coverage NORMALIZED; ceilings={[round(c, 3) for c in ceilings]}, "
                f"weights={[round(w, 3) for w in cov_weights]}, α={COV_ALPHA}, "
                f"gain λ_cov={COV_WEIGHT}",
                flush=True,
            )
            _group_log_once[0] = False
    grid_dtype = jnp.bfloat16 if BF16 else jnp.float32
    vobj, barg_f, run_f = _kernel(
        st_f[0], n_arcs, cov, well_soft, brain_sdf, grid_dtype, ceilings, cov_weights
    )
    if TWO_FIDELITY:
        _vo, barg_c, run_c = _kernel(
            st_c[0],
            n_arcs,
            cov,
            well_soft,
            brain_sdf,
            grid_dtype,
            ceilings,
            cov_weights,
        )
    else:
        barg_c, run_c, st_c = barg_f, run_f, st_f

    n_rows = x0.shape[0]
    n_pad = (-n_rows) % CHUNK
    if n_pad:
        st_f = st_f + [st_f[-1]] * n_pad
        st_c = st_c + [st_c[-1]] * n_pad
        x0 = np.concatenate([x0, np.repeat(x0[-1:], n_pad, 0)], 0)
    n_tot = x0.shape[0]
    x0_dev = jnp.asarray(x0, jnp.float32)

    rc, rf = STAGE1 - REDUCED_FINE, REDUCED_FINE
    fc, ff = STAGE2 - FULL_FINE, FULL_FINE
    n_chunks = n_tot // CHUNK
    fid = f"coarse{COARSE_N}→fine" if TWO_FIDELITY else "fine"
    print(
        f"  {MINIMIZER} {fid} (chunk={CHUNK}, reduced {rc}c+{rf}f → "
        f"full {fc}c+{ff}f, {n_chunks} chunks)...",
        flush=True,
    )
    t0 = time.time()
    x_out = np.zeros_like(x0)  # full@end pose
    x_red = np.zeros_like(x0)  # reduced@end pose (cull checkpoint)
    for ci, s in enumerate(range(0, n_tot, CHUNK)):
        cargs_f = barg_f(st_f[s : s + CHUNK])
        cargs_c = barg_c(st_c[s : s + CHUNK]) if TWO_FIDELITY else cargs_f
        x = x0_dev[s : s + CHUNK]
        if rc > 0:
            x = run_c(x, cargs_c, lo_r, hi_r, 0.0, rc)  # reduced coarse
        if rf > 0:
            x = run_f(x, cargs_f, lo_r, hi_r, 0.0, rf)  # reduced fine finish
        x_red[s : s + CHUNK] = np.asarray(x)
        if fc > 0:
            x = run_c(x, cargs_c, lo, hi, COV_WEIGHT, fc)  # full coarse
        if ff > 0:
            x = run_f(x, cargs_f, lo, hi, COV_WEIGHT, ff)  # full fine finish
        x_out[s : s + CHUNK] = np.asarray(x)
        if (ci + 1) % PROGRESS_EVERY == 0 or ci + 1 == n_chunks:
            el = time.time() - t0
            print(
                f"    chunk {ci + 1}/{n_chunks}  {el:.0f}s  "
                f"ETA {el / (ci + 1) * (n_chunks - ci - 1):.0f}s",
                flush=True,
            )
    print(f"  {time.time() - t0:.1f}s optimize", flush=True)

    # Final scores at fine fidelity (soft clearance uses the thick well too).
    in_axes = (0,) + tuple(0 if k in PER_CAND else None for k in ARG_ORDER)
    clear_b = jax.jit(
        jax.vmap(make_min_clear_one(n_arcs, K, (well_soft,), Phase1Weights()), in_axes)
    )
    obj = np.empty(n_tot, np.float32)
    clr = np.empty(n_tot, np.float32)  # full@end clearance (cull metric)
    clr_red = np.empty(n_tot, np.float32)  # reduced@end clearance (checkpoint)
    x_dev = jnp.asarray(x_out, jnp.float32)
    xr_dev = jnp.asarray(x_red, jnp.float32)
    for s in range(0, n_tot, CHUNK):
        cargs = barg_f(st_f[s : s + CHUNK])
        obj[s : s + CHUNK] = np.asarray(vobj(x_dev[s : s + CHUNK], *cargs))
        clr[s : s + CHUNK] = np.asarray(clear_b(x_dev[s : s + CHUNK], *cargs))
        clr_red[s : s + CHUNK] = np.asarray(clear_b(xr_dev[s : s + CHUNK], *cargs))
    return (
        st_f[:n_rows],
        x_out[:n_rows],
        x_red[:n_rows],
        obj[:n_rows],
        clr[:n_rows],
        clr_red[:n_rows],
    )


def make_phase1_pool_record(
    c: MRVCand,
    n_arcs: int,
    x: np.ndarray,
    x_reduced: np.ndarray,
    objective: float,
    min_clear: float,
    min_clear_reduced: float,
    fcl: float,
) -> Phase1PoolRecord:
    return dict(
        n_arcs=n_arcs,
        probe_to_hole=c.probe_to_hole,
        partition=c.partition,
        # Arc assignment saved EXPLICITLY (not reconstructed from the
        # frozenset partition, whose iteration order is hash-random
        # across processes) so Phase 2 rebuilds the identical `aa`
        # the pose `x` was optimized against. See phase2_ipopt.
        probe_to_arc_idx=dict(c.aa.probe_to_arc_idx),
        arc_centroids_deg=list(c.aa.arc_centroids_deg),
        min_ml_gap=c.min_ml_gap,
        x=x.astype(np.float32),
        x_reduced=x_reduced.astype(np.float32),
        objective=float(objective),
        min_clear=float(min_clear),
        min_clear_reduced=float(min_clear_reduced),
        fcl=float(fcl),
    )


def save_results(records: list[Phase1PoolRecord]) -> None:
    payload: Phase1PoolPayload = dict(
        records=records,
        stage1=STAGE1,
        stage2=STAGE2,
        n_spins=N_SPINS,
        max_arcs=MAX_ARCS,
        max_ppa=MAX_PPA,
        minimizer=MINIMIZER,
        well=WELL_MODE,
        coarse_n=COARSE_N,
        reduced_fine=REDUCED_FINE,
        full_fine=FULL_FINE,
    )
    with open(OUT, "wb") as f:
        pickle.dump(payload, f)


def load_or_seed_groups(
    enum_factory: Callable[[], Enumerator],
) -> dict[int, list[MRVCand]]:
    """Enumerate+seed the MRV pool once, cache grouped-by-n_arcs to SEED_CACHE.
    On restart (cache present, no LIMIT) just reload — the seed CSP is ~14 min."""
    if SEED_CACHE and not LIMIT and _os.path.exists(SEED_CACHE):
        cached_by_arcs = cast(
            dict[int, list[MRVCand]], pickle.load(open(SEED_CACHE, "rb"))
        )
        print(
            f"loaded seeds from {SEED_CACHE}: groups "
            + ", ".join(f"{k}:{len(v)}" for k, v in sorted(cached_by_arcs.items())),
            flush=True,
        )
        return cached_by_arcs

    print(
        f"enumerating MRV pool (arcs<={MAX_ARCS}, probes/arc<={MAX_PPA})...", flush=True
    )
    enum = enum_factory()
    t0 = time.time()
    raw = enum.enumerate()
    print(f"  {len(raw)} discrete candidates in {time.time() - t0:.1f}s", flush=True)
    if LIMIT:
        raw = raw[:LIMIT]
    t0 = time.time()
    cands, n_fail = [], 0
    for d in raw:
        c = wrap(d, enum)
        if c is None:
            n_fail += 1
        else:
            cands.append(c)
    print(
        f"  seeded {len(cands)} cands ({n_fail} dropped: no anchors) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    by_arcs: dict[int, list[MRVCand]] = {}
    for c in cands:
        by_arcs.setdefault(c.n_arcs, []).append(c)
    print("  groups " + ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_arcs.items())))
    if SEED_CACHE and not LIMIT:
        with open(SEED_CACHE, "wb") as f:
            pickle.dump(by_arcs, f)
        print(f"  cached seeds → {SEED_CACHE}", flush=True)
    return by_arcs


def main() -> int:
    opt = setup_runtime()
    _cfg, _rt, probes, holes, sdf_fine, bvh, fixtures, well_thin, fixture_bvhs = setup(
        opt
    )
    brain = opt.brain_sdf()

    # Tuned optimizer: thick well (soft side only; FCL uses true mesh) + coarse SDF.
    well_soft = opt.thick_well_fixture(well_thin) if WELL_MODE == "thick" else well_thin
    sdf_coarse = opt.probe_sdfs(COARSE_N) if TWO_FIDELITY else sdf_fine
    print(
        f"config: minimizer={MINIMIZER} well={WELL_MODE} "
        f"{'coarse' + str(COARSE_N) + '→fine' if TWO_FIDELITY else 'fine-only'} "
        f"reduced {STAGE1 - REDUCED_FINE}c+{REDUCED_FINE}f → "
        f"full {STAGE2 - FULL_FINE}c+{FULL_FINE}f → {OUT}",
        flush=True,
    )

    def _enum_factory() -> Enumerator:
        atlas_payload = build_or_load_atlas()
        # rig AP = subject AP + head_pitch (head nose-down) → rig-reachable subject
        # window = rig[±AP_LIMIT] − head_pitch (mirrors phase1_bounds /
        # _ap_bounds_deg). See dev memory rig_ap_sign_convention. ML is invariant.
        ap_range = (
            -AP_LIMIT_DEG - atlas_payload.head_pitch_deg,
            AP_LIMIT_DEG - atlas_payload.head_pitch_deg,
        )
        return Enumerator(
            atlas_payload.atlas,
            atlas_payload.probe_names,
            ml_margin_deg=0.0,
            ml_mode="greedy",
            max_arcs=MAX_ARCS,
            max_probes_per_arc=MAX_PPA,
            ap_range=ap_range,
        )

    by_arcs = load_or_seed_groups(_enum_factory)

    if ONLY_NARCS:
        by_arcs = {k: v for k, v in by_arcs.items() if k == ONLY_NARCS}
        print(
            f"  ONLY_NARCS={ONLY_NARCS}: "
            + (f"{len(next(iter(by_arcs.values())))} cands" if by_arcs else "no cands")
        )

    # Resume: reload any previously-saved records and skip those n_arcs groups.
    records: list[Phase1PoolRecord] = []
    done_narcs: set = set()
    if _os.path.exists(OUT):
        try:
            prev = cast(Phase1PoolPayload, pickle.load(open(OUT, "rb")))
            records = list(prev.get("records", []))
            done_narcs = {r["n_arcs"] for r in records}
            if done_narcs:
                print(
                    f"resuming from {OUT}: {len(records)} records, "
                    f"done groups {sorted(done_narcs)}",
                    flush=True,
                )
        except Exception as e:
            print(f"  (could not resume from {OUT}: {e})", flush=True)

    # Largest group first: the hungriest spin-restore runs on the cleanest GPU.
    for n_arcs in sorted(by_arcs, key=lambda k: -len(by_arcs[k])):
        if n_arcs in done_narcs:
            print(f"\n[n_arcs={n_arcs}] already in {OUT}, skipping", flush=True)
            continue
        g = by_arcs[n_arcs]
        print(f"\n[n_arcs={n_arcs}] {len(g)} cands", flush=True)
        statics_flat, x_out, x_red, obj, clr, clr_red = run_group(
            n_arcs,
            g,
            probes=probes,
            holes=holes,
            sdf_fine=sdf_fine,
            sdf_coarse=sdf_coarse,
            bvh=bvh,
            well_soft=well_soft,
            brain_sdf=brain,
            head_pitch_deg=opt.head_pitch_deg,
        )
        # FCL on the top-K by clearance (slow per-cand CPU check).
        order = np.argsort(-clr)
        topk = set(order[:FCL_TOPK].tolist())
        fcl = np.full(len(g), np.nan)
        for i in topk:
            v = make_fcl_validator(
                statics_flat[i],
                n_arcs,
                fixtures=tuple(fixtures),
                fixture_bvhs=fixture_bvhs,
            )
            s = np.asarray(v.slacks(x_out[i].astype(np.float64)))
            fcl[i] = float(s.min()) if s.size else 0.0
        for i, c in enumerate(g):
            records.append(
                make_phase1_pool_record(
                    c,
                    n_arcs,
                    x_out[i],
                    x_red[i],
                    float(obj[i]),
                    float(clr[i]),
                    float(clr_red[i]),
                    float(fcl[i]),
                )
            )
        nf = int(np.nansum(fcl >= -1e-4))
        print(f"  FCL-feasible in top-{FCL_TOPK}: {nf}/{min(FCL_TOPK, len(g))}")
        # Incremental save + free this group's GPU buffers before the next group
        # compiles its own kernels (cross-group accumulation caused the OOM).
        save_results(records)
        print(f"  saved {len(records)} records → {OUT}", flush=True)
        del statics_flat, x_out, x_red, obj, clr, clr_red
        gc.collect()
        jax.clear_caches()

    save_results(records)
    print(f"\nsaved {len(records)} records → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
