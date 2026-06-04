"""Apply the H1/H2 spin heuristics + beam search to the MANUAL T12 configuration
(cand 4195) and print the plausible spin assignments it proposes — the spin-basin
*generation* step, before any ADAM/FCL polish.

Uses the manual plan's per-probe (arc-ap, ml, target) as the geometry the
heuristics reason over, and the manual spins as the warm-start seed. Prints:
  * the coupling graph (which probe pairs are close enough to coordinate spin),
  * per-probe H1 slot-aligned spin + the full candidate spin set,
  * the top beam assignments (the plausible spin vectors), vs the manual.

Run:  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.spin_heuristics_manual
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
from pathlib import Path

import numpy as np
import yaml
from aind_mri_utils.arc_angles import arc_angles_to_affine

from aind_low_point.config import ConfigModel, PlanningModel
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import build_fixture_sdf_data
from scripts.spin_heuristic_search import (
    beam_search_assignments,
    body_long_axis_local,
    is_four_shank,
    optimal_spin_for_gap,
    per_probe_spin_candidates,
    spin_to_align_y_with,
    swept_overlap,
    swept_profile,
    swept_surface_world,
)

ALPHA_FACING = 2.0   # tightest-partner-dominates suppression exponent

CONFIG = "examples/836656-config-T12.yml"
PLAN = "examples/836656-config-T12.plan.yml"
HOLES = "scratch/0283-300-04.holes.yml"
POOL_PKL = "scratch/full_polish_0283.pkl"
IDX = 4195
ARC_VAL = {"a": 13.0, "b": -10.0, "c": -43.0}  # manual plan arcs


def _wrap(a):
    return float(((a + 180.0) % 360.0) - 180.0)


def main() -> int:  # noqa: C901
    cfg = ConfigModel.from_yaml(CONFIG)
    rt = build_runtime_from_config(cfg)
    probes = [_probe_static_info(rt.plan_state, rt, n) for n in rt.plan_state.probes]
    holes = load_holes(Path(HOLES))
    comp = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in comp:
        R, t = comp["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    pm = PlanningModel.model_validate(yaml.safe_load(open(PLAN)))
    pool = pickle.load(open(POOL_PKL, "rb"))
    cand = pool["candidates"][IDX]
    n_arcs = int(pool["results"][IDX].n_arcs)
    statics = _build_probe_static(probes, holes, cand.ha, cand.aa,
                                  bvh_cache=None, sdf_by_name=None)
    names = [st.name for st in statics]
    probe_kind = {p.name: p.kind for p in probes}

    # Manual geometry the heuristics reason over.
    arc_aps = np.zeros(n_arcs)
    ml = np.zeros(len(statics))
    seed = {}
    manual = {}
    for i, st in enumerate(statics):
        p = pm.probes[st.name]
        arc_aps[st.arc_idx] = ARC_VAL[p.arc]
        ml[i] = float(p.slider_ml)
        seed[i] = float(p.spin)
        manual[i] = float(p.spin)
    target_LPS = np.array([st.target_LPS for st in statics])

    # Per-probe swept volume (revolution of the probe about its insertion axis).
    prof_by_kind = {}
    profs, R0s, surfs = [], [], []
    for i, st in enumerate(statics):
        kind = probe_kind[st.name]
        if kind not in prof_by_kind:
            mesh = rt.asset_catalog.get_geometry(f"probe:{kind}").raw
            prof_by_kind[kind] = swept_profile(mesh.vertices, st.pivot_local)
        prof = prof_by_kind[kind]
        profs.append(prof)
        R0 = arc_angles_to_affine(float(arc_aps[st.arc_idx]), float(ml[i]), 0.0)
        R0s.append(R0)
        surfs.append(swept_surface_world(prof, R0, target_LPS[i]))

    print(f"cand {IDX}  probes={names}\n")
    print("per-type swept volume (revolution about insertion axis):")
    for kind, (zc, rmax, z_lo, z_hi) in prof_by_kind.items():
        rmed = float(np.median(rmax[rmax > 0])) if (rmax > 0).any() else 0.0
        print(f"  {kind:<18} r_max={rmax.max():.2f} mm  r_median={rmed:.2f} mm  "
              f"z=[{z_lo:.1f}, {z_hi:.1f}] mm")

    # Coupling: actual swept-solid intersection.
    coupling = {i: [] for i in range(len(statics))}
    gaps = {}
    contact = {}
    for i in range(len(statics)):
        for j in range(i + 1, len(statics)):
            ov, gap, ctr = swept_overlap(
                profs[i], R0s[i], target_LPS[i], surfs[i],
                profs[j], R0s[j], target_LPS[j], surfs[j])
            gaps[(i, j)] = gap
            if ov:
                coupling[i].append(j)
                coupling[j].append(i)
                contact[(i, j)] = ctr
    print("\ncoupling graph (swept volumes intersect → spins can interact):")
    for i in range(len(statics)):
        nb = [names[j] for j in coupling.get(i, [])]
        kind = "4-shank" if is_four_shank(statics[i]) else "1-shank"
        print(f"  {names[i]:<5} ({kind})  ↔ {nb if nb else '(decoupled)'}")
    print("\n  pairwise swept-volume gaps (mm; negative = overlap):")
    for (i, j), g in sorted(gaps.items(), key=lambda kv: kv[1]):
        print(f"    {names[i]:>4}↔{names[j]:<4} {g:+6.2f}"
              f"{'  COUPLED' if g < 0 else ''}")

    # --- Unified H2 facing: present narrow profile toward the TIGHTEST coupled
    # partner (probe body or well wall); looser partners suppressed by
    # (tightness/t_max)^ALPHA. Gated on swept-volume intersection — decoupled
    # probes' spins are free. Tightness = penetration depth.
    import jax.numpy as jnp

    from aind_low_point.optimization.sdf_jax import trilinear_sdf
    well = next(f for f in build_fixture_sdf_data(rt) if "well" in f.name.lower())
    wg, wo, wsp = (jnp.asarray(well.grid), jnp.asarray(well.origin),
                   jnp.asarray(well.spacing))
    print("\nunified facing (narrow profile → tightest coupled partner; "
          "probes + well):")
    for i, st in enumerate(statics):
        ap_i, ml_i = float(arc_aps[st.arc_idx]), float(ml[i])
        # Facing direction = from probe i's spin axis toward the swept-overlap
        # CONTACT centroid (where the bodies actually conflict), not the deep
        # targets. optimal_spin_for_gap extracts the transverse component.
        partners = [(names[j], -gaps[(min(i, j), max(i, j))],
                     contact[(min(i, j), max(i, j))] - target_LPS[i])
                    for j in coupling[i]]
        d = np.asarray(trilinear_sdf(wg, wo, wsp, jnp.asarray(surfs[i])))
        if float(d.min()) < 0:   # swept body hits the well wall for some spin
            km = int(d.argmin())
            partners.append(("WELL", float(-d.min()), surfs[i][km] - target_LPS[i]))
        if not partners:
            print(f"  {names[i]:<5} decoupled — spin free")
            continue
        partners.sort(key=lambda p: -p[1])
        lab0, t_max, gdir0 = partners[0]
        opt_a, opt_b = optimal_spin_for_gap(
            body_long_axis_local(probe_kind[st.name]), ap_i, ml_i, gdir0)
        wstr = " ".join(f"{lab}={(t / t_max) ** ALPHA_FACING:.2f}"
                        for lab, t, _ in partners)
        print(f"  {names[i]:<5} tightest={lab0:<4}(pen={t_max:.2f})  "
              f"face→{_wrap(opt_a):+6.0f}|{_wrap(opt_b):+6.0f}  "
              f"manual={manual[i]:+6.0f}  weights[{wstr}]")

    spin_cands = per_probe_spin_candidates(
        statics, coupling, target_LPS, arc_aps, ml, probe_kind, seed_spins=seed)
    print("\nper-probe candidate spins (H1 threading + H2 geometric optimum):")
    for i, st in enumerate(statics):
        h1 = spin_to_align_y_with(st.assigned_hole.slot_major_dir(),
                                  float(arc_aps[st.arc_idx]), float(ml[i]))
        print(f"  {names[i]:<5}  slot-aligned(H1)={_wrap(h1):+7.1f}  "
              f"manual={manual[i]:+7.1f}  cands={[round(c) for c in spin_cands[i]]}")

    beam = beam_search_assignments(
        statics, spin_cands, coupling, target_LPS, arc_aps, ml, probe_kind,
        beam_B=64)
    print(f"\nbeam search → {len(beam)} assignments; top 10 plausible spin vectors "
          f"(probe order {names}):")
    man_vec = np.array([manual[i] for i in range(len(statics))])
    for r, asg in enumerate(beam[:10], 1):
        d = dict(asg.spins)
        vec = np.array([_wrap(d[i]) for i in range(len(statics))])
        dman = np.array([abs(_wrap(vec[i] - man_vec[i])) for i in range(len(vec))])
        nmatch = int((dman <= 20).sum())
        sv = np.round(vec, 0).astype(int).tolist()
        print(f"  #{r:<2} score={asg.score:6.2f}  spins={sv}"
              f"  | match-manual {nmatch}/{len(vec)} (Δmax {dman.max():.0f}°)")
    print(f"\n  manual spins (reference):           "
          f"{np.round(man_vec,0).astype(int).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
