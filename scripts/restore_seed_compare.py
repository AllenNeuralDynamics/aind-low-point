"""Does the round-robin restore's basin depend on its spin SEED?

Seeds the restore three ways for the manual candidate (4195) — the production
atlas anchor, the manual spins, and the computed contact-facing heuristic — runs
restore (with well) then ADAM (brain on) from each, and compares the basins. If
restore washes the seed out (full-circle sweep), all three converge; if the
joint coordinate descent is seed-sensitive, the contested probes diverge.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.restore_seed_compare
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle

import jax.numpy as jnp
import numpy as np
import yaml
from aind_mri_utils.arc_angles import arc_angles_to_affine

from aind_low_point.config import PlanningModel
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.optimizer_vars import build_y
from aind_low_point.optimization.probe_kinematics import (
    is_four_shank,
    spin_to_align_y_with,
)
from aind_low_point.optimization.sdf_jax import trilinear_sdf
from aind_low_point.optimization.stage3_phase3_fcl import make_fcl_validator
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.restore_well_adam_manual import (
    build_adam_kernel,
    make_basin_sets,
    run_restore,
    setup,
    spins_deg_from_phase1,
    spins_deg_from_reduced,
)
from scripts.run_phase1_sample import build_coverage_data, maybe_build_brain_sdf
from scripts.spin_heuristic_search import (
    body_long_axis_local,
    build_coupling_graph,
    optimal_spin_for_gap,
    swept_profile,
    swept_surface_world,
)

IDX = 4195
PLAN = "examples/836656-config-T12.plan.yml"
ARC_VAL = {"a": 13.0, "b": -10.0, "c": -43.0}


def _wrap(a):
    return float(((a + 180.0) % 360.0) - 180.0)


def facing_seeds(rt, statics, arc_aps, ml, target_LPS, probe_kind, well, names):
    """Per-probe contact-facing seed (deg): narrow profile → tightest coupled
    partner's swept-overlap contact (probe or well); 4-shank snapped to slot."""
    mesh_by_kind = {
        k: np.asarray(rt.asset_catalog.get_geometry(f"probe:{k}").raw.vertices)
        for k in set(probe_kind.values())
    }
    coupling, tight, contact = build_coupling_graph(
        statics, arc_aps, ml, target_LPS, mesh_by_kind, probe_kind
    )
    profs = {
        k: swept_profile(mesh_by_kind[k], statics[0].pivot_local) for k in mesh_by_kind
    }  # pivot per-kind below
    wg, wo, wsp = (
        jnp.asarray(well.grid),
        jnp.asarray(well.origin),
        jnp.asarray(well.spacing),
    )
    seeds = {}
    for i, st in enumerate(statics):
        ap_i, ml_i = float(arc_aps[st.arc_idx]), float(ml[i])
        prof = swept_profile(mesh_by_kind[probe_kind[st.name]], st.pivot_local)
        R0 = arc_angles_to_affine(ap_i, ml_i, 0.0)
        surf = swept_surface_world(prof, R0, target_LPS[i])
        partners = [
            (
                tight[(min(i, j), max(i, j))],
                contact[(min(i, j), max(i, j))] - target_LPS[i],
            )
            for j in coupling[i]
        ]
        d = np.asarray(trilinear_sdf(wg, wo, wsp, jnp.asarray(surf)))
        if float(d.min()) < 0:
            partners.append((float(-d.min()), surf[int(d.argmin())] - target_LPS[i]))
        if not partners:  # decoupled → keep atlas/threading
            seeds[st.name] = spin_to_align_y_with(
                st.assigned_hole.slot_major_dir(), ap_i, ml_i
            )
            continue
        partners.sort(key=lambda p: -p[0])
        opt_a, opt_b = optimal_spin_for_gap(
            body_long_axis_local(probe_kind[st.name]), ap_i, ml_i, partners[0][1]
        )
        if is_four_shank(st):  # snap to threadable slot
            slot = spin_to_align_y_with(st.assigned_hole.slot_major_dir(), ap_i, ml_i)
            cands = [slot, slot + 180.0]
            seeds[st.name] = min(
                [opt_a, opt_b], key=lambda s: min(abs(_wrap(s - c)) for c in cands)
            )
            seeds[st.name] = min(cands, key=lambda c: abs(_wrap(seeds[st.name] - c)))
        else:
            seeds[st.name] = opt_a
    del profs
    return seeds


def main() -> int:
    cfg, rt, probes, holes, sdf_by_name, bvh, fixtures, well, fixture_bvhs = setup()
    names = [p.name for p in probes]
    K = len(probes)
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    cand = pool["candidates"][IDX]
    n_arcs = int(pool["results"][IDX].n_arcs)
    st = _build_probe_static(
        probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
    )
    comp = compile_all_transforms(cfg.transforms)
    brain_sdf = maybe_build_brain_sdf(rt, comp)
    pm = PlanningModel.model_validate(yaml.safe_load(open(PLAN)))

    arc_aps = np.zeros(n_arcs)
    ml = np.zeros(K)
    manual_seed = {}
    for i, s in enumerate(st):
        p = pm.probes[s.name]
        arc_aps[s.arc_idx] = ARC_VAL[p.arc]
        ml[i] = float(p.slider_ml)
        manual_seed[s.name] = float(p.spin)
    target_LPS = np.array([s.target_LPS for s in st])
    probe_kind = {p.name: p.kind for p in probes}
    facing = facing_seeds(rt, st, arc_aps, ml, target_LPS, probe_kind, well, names)

    print(f"cand {IDX}  probes={names}")
    print(f"atlas  seed : {np.round([cand.spin_seed.get(n, 0.0) for n in names], 0)}")
    print(f"manual seed : {np.round([manual_seed[n] for n in names], 0)}")
    print(f"facing seed : {np.round([facing[n] for n in names], 0)}")

    cov = build_coverage_data(probes, st)
    adam_eval = build_adam_kernel(st, n_arcs, K, well, cov, brain_sdf=brain_sdf)

    seeds = {"atlas": None, "manual": manual_seed, "facing": facing}
    for label, seed in seeds.items():
        y_red = run_restore(
            cand,
            probes,
            holes,
            sdf_by_name,
            n_arcs,
            well,
            with_well=True,
            seed_spins_deg=seed,
        )
        rest_sp = spins_deg_from_reduced(y_red, n_arcs, K)
        arc_aps_r, mls, sets = make_basin_sets(y_red, st, n_arcs, K)
        zero = np.zeros(K)
        x0 = [
            build_y(arc_aps_r, n_arcs, mls, sp, zero, zero, zero)
            for sp in sets["A_restore1"]
        ]
        viol, xa = adam_eval(x0)
        br = int(np.argmin(viol))
        v = make_fcl_validator(
            st, n_arcs, fixtures=tuple(fixtures), fixture_bvhs=fixture_bvhs
        )
        fcl = float(np.asarray(v.slacks(xa[br])).min())
        asp = np.round(spins_deg_from_phase1(xa[br], n_arcs, K), 0)
        print(f"\n[{label:>6}] restore spins = {np.round(rest_sp, 0)}")
        print(f"         ADAM spins    = {asp}")
        print(
            f"         viol {viol[br]:+.3f}  fcl {fcl:+.3f} "
            f"{'FEAS' if fcl >= -1e-4 else 'infeas'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
