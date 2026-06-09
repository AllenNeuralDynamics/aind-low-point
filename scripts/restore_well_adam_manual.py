"""Focused test: does round-robin spin restore WITH the well fixture, followed
by batched ADAM (NO L-BFGS), recover the manual candidate's FCL-feasible spin
basin — and is a SINGLE restore basin enough, or do we need a wide spin search?

Motivation. Spin loss is multi-modal (≈180° basins). The restore is a *coarse*
basin selector (coordinate descent over an 8-pt circle, 2 rounds) and emits ONE
spin per probe. Earlier work found the manual basin only with a large spin
search; the open worry is that one restore basin is insufficient and that the
reduced objective without the well ranks basins uncorrelated with FCL.

This script, for the manual candidate (idx 4195 by default):
  1. builds the enum seed (atlas spin_seed / ml_seed / arc centroids),
  2. runs the round-robin restore TWICE — with the well fixture and without —
     to isolate the well's effect on basin selection,
  3. seeds the production ADAM kernel (coverage on, no L-BFGS) from each restore
     output with three basin sets:
        A  restore only                (1 basin  — "is one basin enough?")
        B  restore + h1 + 1-shank flip (3 basins — production shape)
        C  restore × all 2^K flips     (wide joint spin search)
  4. FCL-validates the basin-selected pose for each (restore×set) combo.

Reference: the durable rerank pose for the same candidate (chain A =
restore→L-BFGS→ADAM) and its FCL verdict, so we can see whether restore-only
lands in the same basin L-BFGS reached.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.restore_well_adam_manual
Env:  IDXS=4195[,..]  STEPS=150  N_SURF=5000
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle
import time
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.batched_objective import (
    make_batched_reduced_objective,
)
from aind_low_point.optimization.batched_spin_restore import (
    make_batched_spin_restore_partial,
)
from aind_low_point.optimization.batched_static import build_batched_probe_static
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import JointWeights, _build_probe_static
from aind_low_point.optimization.optimizer_vars import build_y
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
    make_batched_phase1_chunked,
)
from scripts.run_optimizer import (
    _probe_static_info,
    _transform_holes,
    retro_opts_from_env,
)
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    phase1_bounds,
)

PPV = 6
N_SURF = int(_os.environ.get("N_SURF", "5000"))
STEPS = int(_os.environ.get("STEPS", "150"))
# ADAM batch chunk — caps GPU memory (128 basins × 5000 surf OOMs in one shot).
CHUNK = int(_os.environ.get("CHUNK", "32"))
# WIDE=1 adds the joint 2^K flip set (the large spin search); off by default.
WIDE = _os.environ.get("WIDE", "0") == "1"
IDXS = [int(x) for x in _os.environ.get("IDXS", "4195").split(",")]
# Subject is config-driven: CONFIG selects the YAML, HOLES the implant-bore file
# (placed into the scene by the config's own implant_to_lps in setup()). Defaults
# reproduce the 836656 test subject; override for any other subject.
CONFIG = _os.environ.get("CONFIG", "examples/836656-config-T12.yml")
HOLES = _os.environ.get("HOLES", "scratch/0283-300-04.holes.yml")
POOL_PKL = "scratch/full_polish_0283.pkl"
RERANK_PKL = "scratch/full_rerank_0283.pkl"


def setup():
    cfg = ConfigModel.from_yaml(CONFIG)
    rt = build_runtime_from_config(cfg)
    _ro = retro_opts_from_env(rt)
    probes = [
        _probe_static_info(rt.plan_state, rt, n, _ro) for n in rt.plan_state.probes
    ]
    holes = load_holes(Path(HOLES))
    comp = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in comp:
        R, t = comp["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            rt.asset_catalog.get_geometry(f"probe:{p.kind}").raw,
            n_surface_points=N_SURF,
        )
        for p in probes
    }
    bvh = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    fixtures = build_fixture_sdf_data(rt)
    well = next(f for f in fixtures if "well" in f.name.lower())
    fixture_bvhs = {
        f.name: make_fcl_bvh(rt.asset_catalog.get_geometry(f.name).raw)
        for f in fixtures
    }
    return cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs


def enum_seed_y0(cand, probes, n_arcs, seed_spins_deg=None):
    """Reduced-layout y0 from the candidate's atlas warm starts (mirrors
    polish_all_with_batched_spin_restore). ``seed_spins_deg`` (per-probe-name
    dict OR length-K array) overrides the atlas spin seed if given."""
    K = len(probes)
    y0 = np.zeros(n_arcs + 3 * K, np.float32)
    for a in range(min(n_arcs, len(cand.aa.arc_centroids_deg))):
        y0[a] = float(cand.aa.arc_centroids_deg[a])
    for k, p in enumerate(probes):
        if seed_spins_deg is None:
            sp = float(cand.spin_seed.get(p.name, 0.0))
        elif isinstance(seed_spins_deg, dict):
            sp = float(seed_spins_deg[p.name])
        else:
            sp = float(seed_spins_deg[k])
        spin = np.deg2rad(sp)
        y0[n_arcs + 3 * k] = float(cand.ml_seed.get(p.name, 0.0))
        y0[n_arcs + 3 * k + 1] = float(np.cos(spin))
        y0[n_arcs + 3 * k + 2] = float(np.sin(spin))
    return y0


def run_restore(
    cand,
    probes,
    holes,
    sdf_by_name,
    n_arcs,
    well,
    *,
    with_well,
    seed_spins_deg=None,
    n_rounds=2,
):
    """Round-robin spin restore for one candidate; returns reduced-layout
    y0_restored (n_arcs+3K,). ``seed_spins_deg`` overrides the atlas spin seed
    (to test whether a heuristic seed reaches a different basin); ``n_rounds``
    sets the coordinate-descent sweep count (production uses 2)."""
    pairs = [(cand.ha, cand.aa)]
    bs = build_batched_probe_static(
        pairs, probes, holes, n_arcs=n_arcs, sdf_by_name=sdf_by_name, head_pitch_deg=0.0
    )
    weights = JointWeights()
    fixtures = (well,) if with_well else ()
    restore = make_batched_spin_restore_partial(
        bs, weights, n_spins=8, n_rounds=n_rounds, fixtures=fixtures
    )
    obj_batched, _ = make_batched_reduced_objective(bs, weights, fixtures)
    varying = obj_batched.extract_arrays(bs)
    y0 = jnp.asarray(enum_seed_y0(cand, probes, n_arcs, seed_spins_deg)[None, :])
    y_r = restore(y0, *varying)
    y_r.block_until_ready()
    return np.asarray(y_r[0], np.float64)


def spins_deg_from_reduced(y_red, n_arcs, K):
    out = []
    for k in range(K):
        sx = y_red[n_arcs + 3 * k + 1]
        sy = y_red[n_arcs + 3 * k + 2]
        out.append(float(np.degrees(np.arctan2(sy, sx))))
    return np.array(out)


def spins_deg_from_phase1(x, n_arcs, K):
    return np.array(
        [
            float(
                np.degrees(np.arctan2(x[n_arcs + PPV * k + 2], x[n_arcs + PPV * k + 1]))
            )
            for k in range(K)
        ]
    )


def build_adam_kernel(
    st,
    n_arcs,
    n_probes,
    well_obj,
    coverage_data,
    brain_sdf=None,
    bounds=None,
    steps=None,
):
    """Build the production Phase-1 ADAM kernel once for ONE candidate (st and
    coverage are constant across basin sets). Returns an ``eval(x0_rows)``
    closure mapping basin rows → (viol[R], x_adam[R, n_vars]). ``brain_sdf``
    (optional) turns on the brain-containment term. ``coverage_data=None``
    drops the coverage term (clearance-first reduced stage). ``bounds`` (a list
    of (lo, hi) pairs) overrides the default Phase-1 bounds — pin offsets/depth
    to (0, 0) to freeze them in the reduced stage. ``steps`` overrides STEPS.
    ``well_obj=None`` drops the well fixture clearance term entirely."""
    if bounds is None:
        bounds = phase1_bounds(n_arcs, n_probes)
    lo = np.array([b[0] for b in bounds], np.float32)
    hi = np.array([b[1] for b in bounds], np.float32)
    fixtures = () if well_obj is None else (well_obj,)
    vobj, _g, build_arglist, make_adam, _mks = make_batched_phase1_chunked(
        st,
        n_arcs,
        Phase1Weights(),
        fixtures,
        coverage_data=coverage_data,
        grid_dtype=jnp.float32,
        brain_sdf=brain_sdf,
    )
    run_adam = make_adam(lo, hi, steps=STEPS if steps is None else steps, lr=0.02)

    def _eval(x0_rows):
        # Fixed-size CHUNK so the kernel compiles for ONE batch shape (a 128-row
        # set at 5000 surf pts OOMs in a single shot). Pad the last chunk to the
        # chunk size and slice the padding back out.
        x0_all = np.stack(x0_rows).astype(np.float32)
        n = x0_all.shape[0]
        cs = min(CHUNK, n)
        npad = (-n) % cs
        if npad:
            x0_all = np.concatenate([x0_all, np.repeat(x0_all[-1:], npad, 0)], 0)
        viol = np.empty(x0_all.shape[0], np.float32)
        xout = np.empty_like(x0_all)
        for s in range(0, x0_all.shape[0], cs):
            x0 = jnp.asarray(x0_all[s : s + cs])
            cargs = build_arglist([st] * cs)
            xa = run_adam(x0, cargs)
            vc = vobj(xa, *cargs)
            xa.block_until_ready()
            xout[s : s + cs] = np.asarray(xa)
            viol[s : s + cs] = np.asarray(vc)
        return viol[:n], xout[:n]

    return _eval


def make_basin_sets(y_red, st, n_arcs, n_probes):
    """Three basin sets (each a list of spin-degree vectors of length K) from a
    restore output: A=restore-only, B=restore+h1+1-shank-flip, C=restore×2^K flips.
    """
    arc_aps = y_red[:n_arcs]
    mls = np.array([y_red[n_arcs + 3 * i] for i in range(n_probes)])
    restore_sp = spins_deg_from_reduced(y_red, n_arcs, n_probes)
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
    sets = {
        "A_restore1": [restore_sp],
        "B_prod3": [restore_sp, h1, np.where(one, h1 + 180.0, h1)],
    }
    if WIDE:
        sets["C_flip2^K"] = [
            restore_sp + 180.0 * np.array(bits)
            for bits in product([0, 1], repeat=n_probes)
        ]
    return arc_aps, mls, sets


def fcl_verdict(
    idx, pose, n_arcs, probes, holes, bvh, sdf_by_name, fixtures, fixture_bvhs, pool
):
    cand = pool["candidates"][idx]
    st = _build_probe_static(
        probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
    )
    v = make_fcl_validator(
        st, n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fixture_bvhs
    )
    return float(np.asarray(v.slacks(pose)).min())


def main() -> int:
    print(f"JAX devices: {jax.devices()}; STEPS={STEPS} N_SURF={N_SURF} IDXS={IDXS}")
    cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    K = len(probes)
    pool = pickle.load(open(POOL_PKL, "rb"))
    rerank = pickle.load(open(RERANK_PKL, "rb"))
    rec_by_idx = {r["idx"]: r for r in rerank["records"]}

    for idx in IDXS:
        cand = pool["candidates"][idx]
        n_arcs = int(pool["results"][idx].n_arcs)
        st = _build_probe_static(
            probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
        )
        coverage_data = build_coverage_data(probes, st)
        adam_eval = build_adam_kernel(st, n_arcs, K, well, coverage_data)
        seed_sp = spins_deg_from_reduced(enum_seed_y0(cand, probes, n_arcs), n_arcs, K)

        # Reference: durable chain-A pose (restore→L-BFGS→ADAM) for this cand.
        ref = rec_by_idx.get(idx, {})
        ref_pose = ref.get("pose")
        ref_sp = (
            spins_deg_from_phase1(np.asarray(ref_pose), n_arcs, K)
            if ref_pose is not None
            else None
        )
        ref_fcl = ref.get("fcl")

        print(
            f"\n{'=' * 78}\ncand {idx}  n_arcs={n_arcs}  "
            f"holes={dict(cand.ha.probe_to_hole)}"
        )
        names = [p.name for p in probes]
        print(f"probes        : {names}")
        print(f"enum spin_seed: {np.round(seed_sp, 1)}")
        if ref_sp is not None:
            print(
                f"chain-A spins : {np.round(ref_sp, 1)}  "
                f"(durable rerank, fcl={ref_fcl})"
            )

        for with_well in (True, False):
            tag = "WELL" if with_well else "no-well"
            t0 = time.time()
            y_red = run_restore(
                cand, probes, holes, sdf_by_name, n_arcs, well, with_well=with_well
            )
            rest_sp = spins_deg_from_reduced(y_red, n_arcs, K)
            d_ref = (
                np.round(np.abs(((rest_sp - ref_sp + 180) % 360) - 180), 1)
                if ref_sp is not None
                else None
            )
            print(
                f"\n[restore {tag}] {time.time() - t0:.1f}s  "
                f"spins={np.round(rest_sp, 1)}"
            )
            if d_ref is not None:
                print(
                    f"   |Δ to chain-A| per probe: {d_ref}  "
                    f"(max {float(np.max(d_ref)):.1f}°)"
                )

            arc_aps, mls, sets = make_basin_sets(y_red, st, n_arcs, K)
            for sname, basins in sets.items():
                zero = np.zeros(K)
                x0_rows = [
                    build_y(arc_aps, n_arcs, mls, sp, zero, zero, zero) for sp in basins
                ]
                t1 = time.time()
                viol, xa = adam_eval(x0_rows)
                br = int(np.argmin(viol))
                fcl = fcl_verdict(
                    idx,
                    xa[br],
                    n_arcs,
                    probes,
                    holes,
                    bvh,
                    sdf_by_name,
                    fixtures,
                    fixture_bvhs,
                    pool,
                )
                feas = fcl >= -1e-4
                win_sp = spins_deg_from_phase1(xa[br], n_arcs, K)
                print(
                    f"   {sname:>11} ({len(basins):>3} basin): "
                    f"viol {viol[br]:>+8.3f}  fcl {fcl:>+7.3f}  "
                    f"{'FEAS' if feas else 'infeas'}  {time.time() - t1:.0f}s  "
                    f"win_spins={np.round(win_sp, 1)}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
