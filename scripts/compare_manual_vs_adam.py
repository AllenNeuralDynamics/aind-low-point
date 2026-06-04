"""Compare the manual T12 plan against the restore-with-well -> ADAM plan for the
manual candidate (4195), on the metrics that matter:

  * FCL clearance (per-pair signed slack + the binding minimum),
  * coverage (the Phase-1 coverage objective),
  * brain containment — are all shank tips inside the brain mesh? (don't puncture
    through the bottom; a depth-greedy ADAM could win coverage but exit the brain),
  * BLA spin 0-vs-180 — settle whether the manual(0)/optimizer(-180) disagreement
    is a REAL placement difference or a convention/symmetry artifact, resolved the
    same way the trame app resolves a pose (ProbePose.from_planning_state), with a
    tip round-trip proving the optimizer kinematics == the app's.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.compare_manual_vs_adam
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle

import numpy as np
import yaml
from scipy.spatial import cKDTree

from aind_low_point.config import PlanningModel
from aind_low_point.optimization.coverage_jax import coverage_total_over_probes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.kinematics import pose_from_optimizer_vars
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.planning import ProbePose, _resolved_angles
from aind_low_point.runtime.export import apply_plan_model_to_state
from aind_low_point.runtime.transforms import compile_all_transforms
from aind_low_point.state_change import PlanStore
from scripts.ingest_analysis import _poses
from scripts.restore_well_adam_manual import (
    build_adam_kernel,
    make_basin_sets,
    run_restore,
    setup,
    spins_deg_from_phase1,
)
from scripts.run_phase1_sample import build_brain_sdf, build_coverage_data

PPV = 6
IDX = 4195
PLAN = "examples/836656-config-T12.plan.yml"
POOL_PKL = "scratch/full_polish_0283.pkl"
# Brain-containment term in the ADAM objective (on by default; BRAIN=0 to compare
# against the unconstrained ADAM plan).
BRAIN = _os.environ.get("BRAIN", "1") == "1"


def manual_x_from_plan(plan: PlanningModel, st, n_arcs):
    """Build the optimizer x (45-vec) for the manual plan at this candidate's
    statics: arc_aps from the bound arc values, per-probe (ml,sx,sy,oR,oA,depth)
    straight from the manual probe declarations."""
    x = np.zeros(n_arcs + PPV * len(st), np.float64)
    for i, s in enumerate(st):
        p = plan.probes[s.name]
        ap = float(plan.arcs[p.arc]) if (p.arc and p.bind_ap_to_arc) else p.ap_local
        x[s.arc_idx] = ap
        off = n_arcs + PPV * i
        spin = np.deg2rad(float(p.spin))
        x[off + 0] = float(p.slider_ml)
        x[off + 1] = float(np.cos(spin))
        x[off + 2] = float(np.sin(spin))
        x[off + 3] = float(p.offsets_RA[0])
        x[off + 4] = float(p.offsets_RA[1])
        x[off + 5] = float(p.past_target_mm)
    return x


def adam_x_for_cand(cand, probes, holes, sdf_by_name, bvh, well, st, n_arcs,
                    brain_sdf=None):
    """restore WITH well -> ADAM (set A, single basin) -> basin-selected pose.
    ``brain_sdf`` turns on the brain-containment term in the ADAM objective."""
    y_red = run_restore(cand, probes, holes, sdf_by_name, n_arcs, well,
                        with_well=True)
    cov = build_coverage_data(probes, st)
    adam_eval = build_adam_kernel(st, n_arcs, len(probes), well, cov,
                                  brain_sdf=brain_sdf)
    arc_aps, mls, sets = make_basin_sets(y_red, st, n_arcs, len(probes))
    basins = sets["A_restore1"]
    from scripts.test_h1_chain_cand4195 import build_y
    zero = np.zeros(len(probes))
    x0 = [build_y(arc_aps, n_arcs, mls, sp, zero, zero, zero) for sp in basins]
    viol, xa = adam_eval(x0)
    return np.asarray(xa[int(np.argmin(viol))], np.float64)


def world_shank_tips(st, x, n_arcs):
    """(P, maxsh, 3) world shank tips + (P, maxsh) mask for an x."""
    Rs, ts, tips, mask = _poses(st, x, n_arcs)
    out = []
    for i in range(len(st)):
        R = np.asarray(Rs[i])
        out.append(np.asarray(tips[i]) @ R.T + np.asarray(ts[i]))
    return out, mask


def fcl_table(label, v, x):
    s = np.asarray(v.slacks(x))
    order = np.argsort(s)
    print(f"\n  [{label}] FCL min = {s.min():+.3f} mm  (feasible={s.min() >= -1e-4})")
    print("    tightest pairs:")
    for j in order[:6]:
        print(f"      {v.pair_names[j]:<28} {s[j]:+.3f}")
    return float(s.min())


def main() -> int:
    cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    names = [p.name for p in probes]
    pool = pickle.load(open(POOL_PKL, "rb"))
    cand = pool["candidates"][IDX]
    n_arcs = int(pool["results"][IDX].n_arcs)
    st = _build_probe_static(probes, holes, cand.ha, cand.aa,
                             bvh_cache=bvh, sdf_by_name=sdf_by_name)

    comp = compile_all_transforms(cfg.transforms)
    brain_sdf = build_brain_sdf(rt, comp) if BRAIN else None
    print(f"brain-containment term: {'ON' if BRAIN else 'OFF'}")

    manual_pm = PlanningModel.model_validate(yaml.safe_load(open(PLAN)))
    manual_x = manual_x_from_plan(manual_pm, st, n_arcs)
    adam_x = adam_x_for_cand(cand, probes, holes, sdf_by_name, bvh, well, st,
                             n_arcs, brain_sdf=brain_sdf)

    print(f"\n{'='*72}\ncand {IDX}  probes={names}")
    print(f"manual spins: {np.round(spins_deg_from_phase1(manual_x, n_arcs, 7),1)}")
    print(f"adam   spins: {np.round(spins_deg_from_phase1(adam_x, n_arcs, 7),1)}")

    # --- FCL clearance --------------------------------------------------------
    v = make_fcl_validator(st, n_arcs, fixtures=tuple(fixtures),
                           fixture_bvhs=fixture_bvhs)
    fcl_m = fcl_table("manual", v, manual_x)
    fcl_a = fcl_table("adam", v, adam_x)

    # --- coverage -------------------------------------------------------------
    cov_data = build_coverage_data(probes, st)
    def cov(x):
        Rs, ts, tips, mask = _poses(st, x, n_arcs)
        return float(coverage_total_over_probes(Rs, ts, tips, mask, cov_data,
                                                n_samples=41))
    cov_m, cov_a = cov(manual_x), cov(adam_x)

    # --- brain containment (world-frame brain mesh) + depth -------------------
    brain_in = {}
    try:
        import trimesh
        bg = rt.asset_catalog.get_geometry("brain").raw
        R, t = comp["headframe_to_lps"].rotate_translate
        bw = trimesh.Trimesh(np.asarray(bg.vertices) @ np.asarray(R).T
                             + np.asarray(t), np.asarray(bg.faces), process=False)
        for label, x in (("manual", manual_x), ("adam", adam_x)):
            tips_w, mask = world_shank_tips(st, x, n_arcs)
            allpts = np.concatenate([tw[m > 0] for tw, m in zip(tips_w, mask)])
            inside = bw.contains(allpts)
            brain_in[label] = (int(inside.sum()), len(inside))
    except Exception as e:
        print(f"\n  [brain] containment check skipped: {e}")

    print(f"\n{'='*72}\nSUMMARY  (manual vs ADAM)")
    print(f"  FCL min clearance : manual {fcl_m:+.3f}   adam {fcl_a:+.3f}   "
          f"{'ADAM' if fcl_a > fcl_m else 'manual'} more clearance")
    print(f"  coverage          : manual {cov_m:.3f}   adam {cov_a:.3f}   "
          f"{'ADAM' if cov_a > cov_m else 'manual'} higher")
    if brain_in:
        for label in ("manual", "adam"):
            ins, tot = brain_in[label]
            print(f"  brain tips inside : {label:<6} {ins}/{tot}"
                  f"{'  <-- PUNCTURE' if ins < tot else ''}")
    print("  depth past_target (mm), per probe:")
    for i, s in enumerate(st):
        print(f"      {s.name:<5} manual {manual_x[n_arcs+PPV*i+5]:+.3f}   "
              f"adam {adam_x[n_arcs+PPV*i+5]:+.3f}")

    # --- BLA spin 0-vs-180: real or convention? (app-faithful) ---------------
    bla = "BLA"
    store = PlanStore(rt.plan_state)
    apply_plan_model_to_state(manual_pm, store)
    ap_m, ml_m, spin_m = _resolved_angles(bla, store.state)
    sidx = names.index(bla)
    s_bla = st[sidx]
    pm = manual_pm.probes[bla]

    def bla_world(spin_deg):
        R, t = pose_from_optimizer_vars(
            target_LPS=np.asarray(s_bla.target_LPS, float),
            ap_deg=ap_m, ml_deg=ml_m, spin_deg=spin_deg,
            offset_R_mm=pm.offsets_RA[0], offset_A_mm=pm.offsets_RA[1],
            past_target_mm=pm.past_target_mm,
            recording_center_local=np.asarray(s_bla.pivot_local, float))
        verts = np.asarray(
            rt.asset_catalog.get_geometry(f"probe:{pm.kind}").raw.vertices, float)
        return np.asarray(verts) @ np.asarray(R).T + np.asarray(t), np.asarray(t)

    # app agreement: optimizer tip vs ProbePose tip at the manual pose
    pose_app = ProbePose.from_planning_state(store.state, bla,
                                             catalog=rt.asset_catalog)
    _, tip_opt = bla_world(spin_m)
    tip_err = float(np.linalg.norm(np.asarray(pose_app.tip) - tip_opt))

    W0, _ = bla_world(0.0)
    W180, _ = bla_world(180.0)
    haus = max(cKDTree(W180).query(W0)[0].max(), cKDTree(W0).query(W180)[0].max())
    same = cKDTree(W180).query(W0)[0]  # nearest-neighbour set distance
    print(f"\n{'='*72}\nBLA spin 0 vs 180 ({pm.kind}):")
    print(f"  optimizer-vs-app tip agreement: {tip_err:.2e} mm "
          f"({'AGREE' if tip_err < 1e-3 else 'MISMATCH'})")
    print(f"  body-mesh Hausdorff(0,180)      : {haus:.3f} mm")
    print(f"  nn set-distance(0->180) max/mean: {same.max():.3f} / {same.mean():.3f}")
    verdict = ("SAME occupied volume -> 180 is a symmetry/convention artifact"
               if same.max() < 0.5 else
               "DIFFERENT occupied volume -> 180 is a REAL alternate placement")
    print(f"  verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
