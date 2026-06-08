"""Investigate probe-vs-well clearance discrepancy.

Phase 1 polish on cand 5113 ends with BLA penetrating the well by
~1 mm per FCL, but the JAX fixture clearance penalty doesn't fire
(bumping λ_clearance_fixture 100→2000 gives identical fn). So JAX
must be reporting BLA-well as CLEAR. This script confirms the
discrepancy at the polished pose and pinpoints where each side's
SDF sees the geometry.
"""

from __future__ import annotations

import argparse
import os as _os
import pickle
import sys as _sys
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")
_sys.path.insert(0, str(Path(__file__).resolve().parent))

import fcl
import jax.numpy as jnp
import numpy as np
from run_optimizer import _probe_static_info, _transform_holes
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.kinematics import pose_from_optimizer_vars
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.sdf_jax import (
    pairwise_signed_clearance_probe_fixture_body,
    trilinear_sdf,
)
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    Phase1Weights,
    make_phase1_objective,
    reduced_to_phase1,
)
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_phase1_sample import (
    build_coverage_data,
    build_fixture_sdf_data,
    phase1_bounds,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument(
        "--polish-pkl", type=Path, default=Path("/tmp/full_polish_patchAB.pkl")
    )
    p.add_argument("--cand", type=int, default=5113)
    p.add_argument("--probe", type=str, default="BLA")
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
    fixtures = build_fixture_sdf_data(runtime)
    well = next(f for f in fixtures if f.name == "well")
    print(
        f"well SDF grid: {well.grid.shape}, "
        f"origin {np.asarray(well.origin).round(2)}, "
        f"spacing {float(well.spacing)} mm, "
        f"n_surf samples {len(np.asarray(well.surface))}"
    )

    well_mesh = runtime.asset_catalog.get_geometry("well").raw
    bvh_cache = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }
    well_bvh = make_fcl_bvh(well_mesh)

    with open(args.polish_pkl, "rb") as f:
        data = pickle.load(f)
    cand = data["candidates"][args.cand]
    jc = data["results"][args.cand]
    statics = _build_probe_static(
        probes,
        holes_list,
        cand.ha,
        cand.aa,
        bvh_cache=bvh_cache,
        sdf_by_name=sdf_by_name,
    )
    n_arcs = jc.n_arcs

    # Run Phase 1 to polish (so we hit the same end state as the orchestrator)
    coverage_data = build_coverage_data(probes, statics)
    fun, jac = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=Phase1Weights(),
    )
    bounds = phase1_bounds(n_arcs, len(statics))
    x0 = reduced_to_phase1(jc.reduced_y, n_arcs, len(statics))
    res = minimize(
        fun,
        x0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        options=dict(maxiter=200, ftol=1e-5),
    )
    x_polished = np.asarray(res.x, dtype=np.float64)
    print(f"\nPhase 1 polish: fn0={float(fun(x0)):.2f} → {res.fun:.2f}, iter={res.nit}")

    # Extract BLA's pose
    idx = next(i for i, st in enumerate(statics) if st.name == args.probe)
    sb = statics[idx]
    arc_aps = x_polished[:n_arcs]
    off = n_arcs + PHASE1_PER_PROBE_VARS * idx
    ml = float(x_polished[off + 0])
    sx = float(x_polished[off + 1])
    sy = float(x_polished[off + 2])
    off_R = float(x_polished[off + 3])
    off_A = float(x_polished[off + 4])
    depth = float(x_polished[off + 5])
    spin_deg = float(np.degrees(np.arctan2(sy, sx)))
    ap = float(arc_aps[sb.arc_idx])
    R_p, t_p = pose_from_optimizer_vars(
        target_LPS=sb.target_LPS,
        ap_deg=ap,
        ml_deg=ml,
        spin_deg=spin_deg,
        offset_R_mm=off_R,
        offset_A_mm=off_A,
        past_target_mm=depth,
        recording_center_local=sb.pivot_local,
    )
    print(f"\n{args.probe} pose:")
    print(f"  R: {R_p.round(3).tolist()}")
    print(f"  t: {t_p.round(3).tolist()}")
    print(
        f"  vars: ml={ml:.3f}, spin={spin_deg:.3f}°, "
        f"off_R={off_R:.3f}, off_A={off_A:.3f}, depth={depth:.3f}"
    )

    # FCL distance probe-vs-well
    sb.bvh_obj.setTransform(
        fcl.Transform(
            np.ascontiguousarray(R_p),
            np.ascontiguousarray(t_p),
        )
    )
    dr = fcl.DistanceResult()
    fcl.distance(
        sb.bvh_obj, well_bvh, fcl.DistanceRequest(enable_signed_distance=True), dr
    )
    fcl_d = float(dr.min_distance)
    if fcl_d <= 0:
        cr = fcl.CollisionResult()
        fcl.collide(
            sb.bvh_obj,
            well_bvh,
            fcl.CollisionRequest(num_max_contacts=8, enable_contact=True),
            cr,
        )
        if cr.contacts:
            fcl_d = -float(max(c.penetration_depth for c in cr.contacts))
    print(f"\nFCL ({args.probe}-vs-well, full mesh): {fcl_d:+.4f} mm")

    # === Gradients at the polished pose ===
    print(f"\n{'=' * 70}")
    print("Gradients at polished pose")
    print(f"{'=' * 70}")
    g_full = np.asarray(jac(x_polished), dtype=np.float64)
    bla_slots = slice(
        n_arcs + PHASE1_PER_PROBE_VARS * idx, n_arcs + PHASE1_PER_PROBE_VARS * (idx + 1)
    )
    print(
        f"|g_full| max={np.abs(g_full).max():.4g}, "
        f"||g_full||={np.linalg.norm(g_full):.4g}"
    )
    print(
        f"g_full[BLA] = {g_full[bla_slots].round(5).tolist()}  "
        f"(ml, sx, sy, off_R, off_A, depth)"
    )
    print(f"g_full[arcs] = {g_full[:n_arcs].round(5).tolist()}")

    # Fixture-only gradient (zero out all other weights)
    from dataclasses import replace as _replace

    w0 = _replace(
        Phase1Weights(),
        lambda_thread=0.0,
        lambda_clearance=0.0,
        lambda_kinematic=0.0,
        lambda_bounds=0.0,
        lambda_margin_clear=0.0,
        lambda_margin_thread=0.0,
        lambda_clearance_fixture=100.0,
    )
    fun_f, jac_f = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=None,
        fixtures=fixtures,
        weights=w0,
    )
    g_fix = np.asarray(jac_f(x_polished), dtype=np.float64)
    print("\n[fixture-only, λ=100, no coverage/other terms]")
    print(f"  obj={float(fun_f(x_polished)):.6g}")
    print(
        f"  |g_fix| max={np.abs(g_fix).max():.4g}, "
        f"||g_fix||={np.linalg.norm(g_fix):.4g}"
    )
    print(f"  g_fix[BLA] = {g_fix[bla_slots].round(5).tolist()}")

    # Bump λ ×10 and re-polish from x_polished — does BLA move out of well?
    w_hi = _replace(Phase1Weights(), lambda_clearance_fixture=1000.0)
    fun_hi, jac_hi = make_phase1_objective(
        statics,
        n_arcs,
        coverage_data=coverage_data,
        fixtures=fixtures,
        weights=w_hi,
    )
    res_hi = minimize(
        fun_hi,
        x_polished,
        jac=jac_hi,
        method="SLSQP",
        bounds=bounds,
        options=dict(maxiter=200, ftol=1e-5),
    )
    x_hi = np.asarray(res_hi.x, dtype=np.float64)
    print("\nRe-polish with λ_clearance_fixture=1000 from x_polished:")
    print(
        f"  fn: {float(fun_hi(x_polished)):.2f} → {float(res_hi.fun):.2f}, "
        f"iter={res_hi.nit}"
    )
    dx_bla = x_hi[bla_slots] - x_polished[bla_slots]
    print(f"  Δ(BLA vars) = {dx_bla.round(5).tolist()}")
    print(f"  Δ(arcs) = {(x_hi[:n_arcs] - x_polished[:n_arcs]).round(5).tolist()}")

    # Recompute FCL at high-λ end pose
    ap_hi = float(x_hi[sb.arc_idx])
    ml_hi = float(x_hi[bla_slots.start + 0])
    sx_hi = float(x_hi[bla_slots.start + 1])
    sy_hi = float(x_hi[bla_slots.start + 2])
    off_R_hi = float(x_hi[bla_slots.start + 3])
    off_A_hi = float(x_hi[bla_slots.start + 4])
    depth_hi = float(x_hi[bla_slots.start + 5])
    R_hi, t_hi = pose_from_optimizer_vars(
        target_LPS=sb.target_LPS,
        ap_deg=ap_hi,
        ml_deg=ml_hi,
        spin_deg=float(np.degrees(np.arctan2(sy_hi, sx_hi))),
        offset_R_mm=off_R_hi,
        offset_A_mm=off_A_hi,
        past_target_mm=depth_hi,
        recording_center_local=sb.pivot_local,
    )
    sb.bvh_obj.setTransform(
        fcl.Transform(
            np.ascontiguousarray(R_hi),
            np.ascontiguousarray(t_hi),
        )
    )
    dr_hi = fcl.DistanceResult()
    fcl.distance(
        sb.bvh_obj, well_bvh, fcl.DistanceRequest(enable_signed_distance=True), dr_hi
    )
    fcl_d_hi = float(dr_hi.min_distance)
    if fcl_d_hi <= 0:
        cr = fcl.CollisionResult()
        fcl.collide(sb.bvh_obj, well_bvh, fcl.CollisionRequest(num_max_contacts=1), cr)
        fcl_d_hi = -1.0 if cr.contacts else fcl_d_hi
    print(f"  FCL after high-λ polish: {fcl_d_hi:+.4f} mm")

    # JAX clearance probe-vs-well-body
    probe_sdf = sdf_by_name[args.probe]
    hard, soft = pairwise_signed_clearance_probe_fixture_body(
        jnp.asarray(R_p, dtype=jnp.float32),
        jnp.asarray(t_p, dtype=jnp.float32),
        jnp.asarray(probe_sdf.grid, dtype=jnp.float32),
        jnp.asarray(probe_sdf.origin, dtype=jnp.float32),
        jnp.asarray(probe_sdf.spacing, dtype=jnp.float32),
        well.grid,
        well.origin,
        well.spacing,
        jnp.asarray(probe_sdf.surface_points, dtype=jnp.float32),
        well.surface,
    )
    print(
        f"JAX clearance probe_fixture_body: hard={float(hard):+.4f}, soft={float(soft):+.4f}"
    )

    # === Look at FCL contact points and check both SDFs ===
    print(f"\n{'=' * 70}")
    print("FCL contact points → both SDFs")
    print(f"{'=' * 70}")
    cr = fcl.CollisionResult()
    fcl.collide(
        sb.bvh_obj,
        well_bvh,
        fcl.CollisionRequest(num_max_contacts=8, enable_contact=True),
        cr,
    )
    print(f"  {len(cr.contacts) if cr.contacts else 0} contacts:")
    for k, c in enumerate(cr.contacts or []):
        pos_world = np.asarray(c.pos, dtype=np.float64)
        depth = float(c.penetration_depth)
        # Probe-local: (world - t_p) @ R_p
        local_p = (pos_world - t_p) @ R_p
        d_in_probe = float(
            np.asarray(
                trilinear_sdf(
                    jnp.asarray(probe_sdf.grid, dtype=jnp.float32),
                    jnp.asarray(probe_sdf.origin, dtype=jnp.float32),
                    jnp.asarray(probe_sdf.spacing, dtype=jnp.float32),
                    jnp.asarray(local_p.reshape(1, 3), dtype=jnp.float32),
                )
            )[0]
        )
        # Well is at identity (static), so world coord == local coord
        d_in_well = float(
            np.asarray(
                trilinear_sdf(
                    well.grid,
                    well.origin,
                    well.spacing,
                    jnp.asarray(pos_world.reshape(1, 3), dtype=jnp.float32),
                )
            )[0]
        )
        print(f"  contact[{k}] world={pos_world.round(3).tolist()} depth={depth:+.4f}")
        print(
            f"    {args.probe}-SDF at contact: {d_in_probe:+.4f} (local_p={local_p.round(3).tolist()})"
        )
        print(f"    well-SDF  at contact: {d_in_well:+.4f}")

    # === Look at where probe surface samples are wrt the well ===
    print(f"\n{'=' * 70}")
    print(
        f"How many of {args.probe}'s {len(np.asarray(probe_sdf.surface_points))} surface "
        f"samples are inside the well envelope?"
    )
    print(f"{'=' * 70}")
    world_surface_p = np.asarray(probe_sdf.surface_points) @ R_p.T + t_p
    d_p_in_well = np.asarray(
        trilinear_sdf(
            well.grid,
            well.origin,
            well.spacing,
            jnp.asarray(world_surface_p, dtype=jnp.float32),
        )
    )
    in_well_grid = d_p_in_well < 1e2
    di = d_p_in_well[in_well_grid]
    if len(di) > 0:
        n_inside = int(np.sum(di < 0))
        print(
            f"  {n_inside}/{len(di)} probe samples have well-SDF < 0 (inside well envelope)"
        )
        print(f"  min well-SDF over probe samples: {di.min():+.4f} mm")
        print(f"  10 deepest: {sorted(di)[:10]}")
    else:
        print("  no probe surface samples fell inside well's SDF grid bbox")

    # And the reverse: how many of well's surface samples are inside the probe
    print(
        f"\nHow many of well's {len(np.asarray(well.surface))} surface "
        f"samples are inside {args.probe}'s body envelope?"
    )
    well_world_samples = np.asarray(well.surface)
    well_in_p_local = (well_world_samples - t_p) @ R_p
    d_w_in_p = np.asarray(
        trilinear_sdf(
            jnp.asarray(probe_sdf.grid, dtype=jnp.float32),
            jnp.asarray(probe_sdf.origin, dtype=jnp.float32),
            jnp.asarray(probe_sdf.spacing, dtype=jnp.float32),
            jnp.asarray(well_in_p_local, dtype=jnp.float32),
        )
    )
    in_probe = d_w_in_p < 1e2
    di = d_w_in_p[in_probe]
    if len(di) > 0:
        n_inside = int(np.sum(di < 0))
        print(f"  {n_inside}/{len(di)} well samples inside probe body envelope")
        print(f"  min probe-SDF over well samples: {di.min():+.4f} mm")
    else:
        print("  no well surface samples fell inside probe's SDF grid bbox")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
