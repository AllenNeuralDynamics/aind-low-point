"""Pilot: does ADAM from CHEAP basins (no beam) reach feasibility?

Tests the claim that the joint beam search is unnecessary — that cheap
diverse basins + ADAM's own joint optimization match beam-basins + ADAM.

For cand 4195 (manual tuple) and 1035 (feasible alt), compare basin
proposal strategies, each run through batched ADAM (lr=0.02, 200 steps),
reporting per-basin final FCL:
  * incumbent : the production frozen-basin spin
  * H1        : per-probe slot-major (single basin)
  * cheap4    : H1 with the 1-shank probes' spins shifted {0,90,180,270}
                (4 basins, microseconds, NO beam)
  * beam4     : the joint beam top-4 (the expensive reference)

Claim holds if cheap4's best FCL matches beam4's / incumbent feasibility.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
from pathlib import Path

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
from scripts.batched_adam_test import adam
from scripts.batched_phase1_build import make_batched_phase1_objective
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import build_fixture_sdf_data, phase1_bounds
from scripts.spin_heuristic_search import (
    beam_search_assignments,
    build_coupling_graph,
    is_four_shank,
    per_probe_spin_candidates,
    spin_to_align_y_with,
)
from scripts.test_h1_chain_cand4195 import build_y, extract_spins

PPV = 6


def main() -> int:
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
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
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

    for cand_idx in (4195, 1035):
        cand = data["candidates"][cand_idx]
        jc = data["results"][cand_idx]
        statics = _build_probe_static(
            probes,
            holes,
            cand.ha,
            cand.aa,
            bvh_cache=bvh_cache,
            sdf_by_name=sdf_by_name,
        )
        n_arcs = jc.n_arcs
        n_probes = len(statics)
        validator = make_fcl_validator(
            statics,
            n_arcs,
            fixtures=(well,),
            fixture_bvhs={well.name: fixture_bvhs[well.name]},
        )
        x_aug = np.asarray(data["augmented_phase1_x"][cand_idx], float)
        arc_aps = x_aug[:n_arcs]
        mls = np.array([x_aug[n_arcs + PPV * i] for i in range(n_probes)])
        spin_inc = extract_spins(x_aug, n_arcs, n_probes)

        # H1 per probe (slot-major).
        h1 = np.array(
            [
                spin_to_align_y_with(
                    st.assigned_hole.slot_major_dir(),
                    float(arc_aps[st.arc_idx]),
                    float(mls[i]),
                )
                for i, st in enumerate(statics)
            ]
        )
        one_shank = np.array([not is_four_shank(st) for st in statics])

        basin_sets = {"incumbent": [spin_inc], "H1": [h1]}
        # cheap4: H1, shift 1-shank spins by {0,90,180,270}
        basin_sets["cheap4"] = [
            np.where(one_shank, h1 + d, h1) for d in (0, 90, 180, 270)
        ]
        # beam4: joint beam top-4
        coupling = build_coupling_graph(np.array([st.target_LPS for st in statics]))
        sc = per_probe_spin_candidates(
            statics,
            coupling,
            np.array([st.target_LPS for st in statics]),
            arc_aps,
            mls,
            {p.name: p.kind for p in probes},
            seed_spins={i: float(spin_inc[i]) for i in range(n_probes)},
        )
        beam = beam_search_assignments(
            statics,
            sc,
            coupling,
            np.array([st.target_LPS for st in statics]),
            arc_aps,
            mls,
            {p.name: p.kind for p in probes},
            beam_B=16,
        )
        basin_sets["beam4"] = [
            np.array([dict(a.spins)[i] for i in range(n_probes)]) for a in beam[:4]
        ]

        zero = np.zeros(n_probes)
        labels, rows = [], []
        for name, spins_list in basin_sets.items():
            for sp in spins_list:
                labels.append(name)
                rows.append(build_y(arc_aps, n_arcs, mls, sp, zero, zero, zero))
        x0 = np.stack(rows).astype(np.float32)
        B = x0.shape[0]
        bounds = phase1_bounds(n_arcs, n_probes)
        lo = np.array([b[0] for b in bounds], np.float32)
        hi = np.array([b[1] for b in bounds], np.float32)
        bobj, bgrad = make_batched_phase1_objective(
            [statics] * B, n_arcs, Phase1Weights(), (well,), coverage_data=None
        )
        x_adam = adam(x0, bgrad, lo, hi, steps=200, lr=0.02)
        _viol = np.asarray(bobj(x_adam))
        fcls = np.array(
            [float(np.asarray(validator.slacks(x_adam[i])).min()) for i in range(B)]
        )

        print(
            f"\n=== cand {cand_idx} "
            f"(manual tuple={'yes' if cand_idx == 4195 else 'no'}) ==="
        )
        best = {}
        for name in basin_sets:
            mask = np.array([label == name for label in labels])
            best_fcl = fcls[mask].max()
            best[name] = best_fcl
            feas = (fcls[mask] >= -1e-4).sum()
            print(
                f"  {name:<10} best FCL {best_fcl:>+7.3f}  "
                f"({feas}/{mask.sum()} basins feasible)"
            )
        print(
            f"  -> cheap4 best {best['cheap4']:+.3f} vs beam4 best "
            f"{best['beam4']:+.3f}  "
            f"{'CHEAP HOLDS' if best['cheap4'] >= -1e-4 else 'cheap FAILS'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
