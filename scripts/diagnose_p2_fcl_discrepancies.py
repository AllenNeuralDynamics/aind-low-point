"""For each Stage 2 "feasible" cand that fails probe-probe FCL,
break down which dual-rep category JAX under-reports.

For each (a, b) probe pair in the worst-violator list:
  * FCL signed clearance (truth)
  * JAX dual-rep: hbb (body-body), hbs (body-shank), hss (shank-shank)
  * Per-cat soft-min (sbb, sbs, sss) — what SLSQP actually sees

Reports the gap and which category(ies) under-report.
"""

from __future__ import annotations

import argparse
import os as _os
import pickle
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import fcl
import jax.numpy as jnp
import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import (
    _build_probe_static,
    _signed_pair_clearance,
)
from aind_low_point.optimization.kinematics import pose_from_optimizer_vars
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.sdf_jax import (
    pairwise_signed_clearance_dual,
)
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument(
        "--polish-pkl", type=Path, default=Path("/tmp/full_polish_post_sat.pkl")
    )
    p.add_argument(
        "--cands",
        type=str,
        default="288,708,814,1079,1328,1533,1775,2088,2714,2772",
        help="Comma-separated cand indices to inspect",
    )
    args = p.parse_args()

    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes_list = load_holes(args.holes)
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes_list = _transform_holes(holes_list, R, t)

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

    with open(args.polish_pkl, "rb") as f:
        data = pickle.load(f)
    cand_idxs = [int(c) for c in args.cands.split(",")]

    print(
        f"{'cand':>6}  {'pair':<14}  {'FCL':>9}  "
        f"{'hbb':>9} {'hbs':>9} {'hss':>9}  "
        f"{'sbb':>9} {'sbs':>9} {'sss':>9}"
    )
    print("-" * 110)

    for cand_idx in cand_idxs:
        cand = data["candidates"][cand_idx]
        jc = data["results"][cand_idx]
        statics = _build_probe_static(
            probes,
            holes_list,
            cand.ha,
            cand.aa,
            bvh_cache=bvh_cache,
            sdf_by_name=sdf_by_name,
        )
        y = np.asarray(jc.reduced_y, dtype=np.float64)
        n_arcs = jc.n_arcs

        # Compute poses for all probes
        poses = {}
        for i, st in enumerate(statics):
            off = n_arcs + 3 * i
            ml = float(y[off])
            sx = float(y[off + 1])
            sy = float(y[off + 2])
            spin = float(np.degrees(np.arctan2(sy, sx)))
            ap = float(y[st.arc_idx])
            R_w, t_w = pose_from_optimizer_vars(
                target_LPS=st.target_LPS,
                ap_deg=ap,
                ml_deg=ml,
                spin_deg=spin,
                offset_R_mm=0.0,
                offset_A_mm=0.0,
                past_target_mm=0.0,
                recording_center_local=st.pivot_local,
            )
            poses[st.name] = (R_w, t_w, st)
            st.bvh_obj.setTransform(
                fcl.Transform(
                    np.ascontiguousarray(R_w, dtype=np.float64),
                    np.ascontiguousarray(t_w, dtype=np.float64),
                )
            )

        # Find worst pair via FCL
        K = len(statics)
        worst_pair = None
        worst_d = float("inf")
        for a in range(K):
            for b in range(a + 1, K):
                d = _signed_pair_clearance(
                    statics[a].bvh_obj,
                    statics[b].bvh_obj,
                )
                if d < worst_d:
                    worst_d = d
                    worst_pair = (statics[a].name, statics[b].name)

        if worst_pair is None:
            continue
        a_name, b_name = worst_pair
        R_a, t_a, sa_st = poses[a_name]
        R_b, t_b, sb_st = poses[b_name]
        sa = sa_st.sdf_data
        sb = sb_st.sdf_data

        (hbb, sbb), (hbs, sbs), (hss, sss) = pairwise_signed_clearance_dual(
            jnp.asarray(R_a, dtype=jnp.float32),
            jnp.asarray(t_a, dtype=jnp.float32),
            jnp.asarray(R_b, dtype=jnp.float32),
            jnp.asarray(t_b, dtype=jnp.float32),
            sa["grid"],
            sa["origin"],
            sa["spacing"],
            sb["grid"],
            sb["origin"],
            sb["spacing"],
            sa["surface"],
            sb["surface"],
            sa["shank_centers"],
            sa["shank_halves"],
            sb["shank_centers"],
            sb["shank_halves"],
        )
        pair_label = f"{a_name}/{b_name}"
        print(
            f"{cand_idx:>6}  {pair_label:<14}  {worst_d:+9.4f}  "
            f"{float(hbb):+9.4f} {float(hbs):+9.4f} {float(hss):+9.4f}  "
            f"{float(sbb):+9.4f} {float(sbs):+9.4f} {float(sss):+9.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
