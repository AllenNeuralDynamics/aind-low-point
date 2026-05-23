"""Pin down WHERE the α-wrap envelope is non-conservative.

For one Phase-1-polished cand whose worst probe-probe pair shows
JAX hbb > 0 but FCL collides:

  1. Get the FCL contact points (these are on the raw mesh surfaces).
  2. Evaluate JAX envelope SDF at each contact in BOTH probes' local
     frames. Should be ~-offset_mm (envelope is offset_mm beyond true
     surface). If POSITIVE, envelope is shrunken.
  3. Evaluate true-mesh signed distance at each contact (via
     ``trimesh.proximity.signed_distance``). Should be 0 (contact is
     on raw surface).
  4. Report the gap: ``envelope_SDF - true_dist`` per contact.
     Expected ~-offset_mm everywhere; actual values tell us where
     α-wrap shrunk the envelope.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pickle
from pathlib import Path

import fcl
import jax.numpy as jnp
import numpy as np
import trimesh
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.kinematics import pose_from_optimizer_vars
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.sdf_jax import trilinear_sdf
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS, Phase1Weights, make_phase1_objective,
    reduced_to_phase1,
)
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes
from scripts.run_phase1_sample import (
    build_coverage_data, build_fixture_sdf_data, phase1_bounds,
)


def main(cand_idx=1042, probe_a="BLA", probe_b="RSP", offset_mm=0.2):
    cfg = ConfigModel.from_yaml(Path("examples/836656-config-T12.yml"))
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(Path("/tmp/836656-holes.yml"))
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        T = compiled["implant_to_lps"]
        R, t = T.rotate_translate
        holes = _transform_holes(holes, R, t)

    raw_meshes = {
        p.name: runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        for p in probes
    }
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            raw_meshes[p.name], offset_mm=offset_mm,
        )
        for p in probes
    }
    print(f"(using offset_mm={offset_mm})")
    fixtures = build_fixture_sdf_data(runtime)
    bvh_cache = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }

    with open("/tmp/full_polish_post_sat.pkl", "rb") as f:
        data = pickle.load(f)
    cand = data["candidates"][cand_idx]
    jc = data["results"][cand_idx]
    statics = _build_probe_static(
        probes, holes, cand.ha, cand.aa,
        bvh_cache=bvh_cache, sdf_by_name=sdf_by_name,
    )
    n_arcs = jc.n_arcs
    coverage_data = build_coverage_data(probes, statics)

    # Phase 1 polish to get the failing pose
    fun, jac = make_phase1_objective(
        statics, n_arcs, coverage_data=coverage_data, fixtures=fixtures,
        weights=Phase1Weights(),
    )
    bounds = phase1_bounds(n_arcs, len(statics))
    x0 = reduced_to_phase1(jc.reduced_y, n_arcs, len(statics))
    res = minimize(fun, x0, jac=jac, method="SLSQP", bounds=bounds,
                   options=dict(maxiter=80, ftol=1e-5))
    x = res.x

    # Extract poses for the two probes of interest
    poses = {}
    for i, st in enumerate(statics):
        if st.name not in (probe_a, probe_b):
            continue
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        ml = float(x[off + 0])
        sx = float(x[off + 1])
        sy = float(x[off + 2])
        off_R = float(x[off + 3])
        off_A = float(x[off + 4])
        depth = float(x[off + 5])
        spin = float(np.degrees(np.arctan2(sy, sx)))
        ap = float(x[st.arc_idx])
        R_w, t_w = pose_from_optimizer_vars(
            target_LPS=st.target_LPS, ap_deg=ap, ml_deg=ml, spin_deg=spin,
            offset_R_mm=off_R, offset_A_mm=off_A, past_target_mm=depth,
            recording_center_local=st.pivot_local,
        )
        poses[st.name] = (R_w, t_w, st)

    Ra, ta, sa_st = poses[probe_a]
    Rb, tb, sb_st = poses[probe_b]

    # Set FCL transforms + get contact points
    sa_st.bvh_obj.setTransform(fcl.Transform(
        np.ascontiguousarray(Ra), np.ascontiguousarray(ta),
    ))
    sb_st.bvh_obj.setTransform(fcl.Transform(
        np.ascontiguousarray(Rb), np.ascontiguousarray(tb),
    ))
    cr = fcl.CollisionResult()
    fcl.collide(sa_st.bvh_obj, sb_st.bvh_obj,
                fcl.CollisionRequest(num_max_contacts=20, enable_contact=True), cr)
    print(f"\nFCL contact check {probe_a} vs {probe_b} on cand {cand_idx}:")
    print(f"  n_contacts: {len(cr.contacts) if cr.contacts else 0}")
    if not cr.contacts:
        print("  no contacts — FCL says clear. Done.")
        return

    sa = sa_st.sdf_data
    sb = sb_st.sdf_data
    grid_a = jnp.asarray(sa["grid"], dtype=jnp.float32)
    origin_a = jnp.asarray(sa["origin"], dtype=jnp.float32)
    spacing_a = jnp.asarray(sa["spacing"], dtype=jnp.float32)
    grid_b = jnp.asarray(sb["grid"], dtype=jnp.float32)
    origin_b = jnp.asarray(sb["origin"], dtype=jnp.float32)
    spacing_b = jnp.asarray(sb["spacing"], dtype=jnp.float32)

    # Per-contact analysis
    print(f"\n{'idx':>3}  {'pos_world':<35}  "
          f"{'env_a':>8}  {'env_b':>8}  "
          f"{'mesh_a':>9}  {'mesh_b':>9}  "
          f"{'env_a−mesh_a':>13}  {'env_b−mesh_b':>13}")
    print("-" * 130)

    mesh_a = raw_meshes[probe_a]
    mesh_b = raw_meshes[probe_b]

    # Transform raw meshes to world to compute true signed distance
    mesh_a_world = mesh_a.copy()
    mesh_a_world.apply_transform(
        np.block([[Ra, ta.reshape(-1, 1)], [np.zeros((1, 3)), np.ones((1, 1))]])
    )
    mesh_b_world = mesh_b.copy()
    mesh_b_world.apply_transform(
        np.block([[Rb, tb.reshape(-1, 1)], [np.zeros((1, 3)), np.ones((1, 1))]])
    )

    contact_pts = np.stack([np.asarray(c.pos, dtype=np.float64) for c in cr.contacts])

    # True mesh signed distance via trimesh proximity
    true_dist_a = trimesh.proximity.signed_distance(mesh_a_world, contact_pts)
    true_dist_b = trimesh.proximity.signed_distance(mesh_b_world, contact_pts)

    # JAX envelope SDF at contact points (in each probe's local frame)
    local_a = (contact_pts - ta) @ Ra
    local_b = (contact_pts - tb) @ Rb
    env_a_vals = np.asarray(trilinear_sdf(
        grid_a, origin_a, spacing_a, jnp.asarray(local_a, dtype=jnp.float32),
    ))
    env_b_vals = np.asarray(trilinear_sdf(
        grid_b, origin_b, spacing_b, jnp.asarray(local_b, dtype=jnp.float32),
    ))

    for i, (p, ea, eb, ma, mb) in enumerate(zip(
        contact_pts, env_a_vals, env_b_vals, true_dist_a, true_dist_b,
    )):
        gap_a = ea - ma
        gap_b = eb - mb
        print(f"{i:>3}  {str(p.round(2).tolist()):<35}  "
              f"{ea:>+8.3f}  {eb:>+8.3f}  "
              f"{ma:>+9.4f}  {mb:>+9.4f}  "
              f"{gap_a:>+13.4f}  {gap_b:>+13.4f}")

    print("\nLegend:")
    print("  env_*    = JAX trilinear SDF of α-wrap envelope at the contact "
          "(in that probe's local frame)")
    print(f"             EXPECTED ~−offset_mm (=−0.2) if envelope is "
          "conservative outside true mesh")
    print("  mesh_*   = trimesh.proximity.signed_distance from raw mesh "
          "(negative = inside)")
    print("             Contacts are ON raw surface, so |mesh| should be "
          "≈ 0")
    print("  env−mesh = how much envelope SDF differs from true mesh "
          "distance")
    print("             EXPECTED ≈ −offset_mm (−0.2) — envelope is "
          "INFLATED outward")
    print("             If POSITIVE: envelope is SHRUNKEN (inside true "
          "mesh) — the bug.")


if __name__ == "__main__":
    import sys
    cand = int(sys.argv[1]) if len(sys.argv) > 1 else 1042
    a = sys.argv[2] if len(sys.argv) > 2 else "BLA"
    b = sys.argv[3] if len(sys.argv) > 3 else "RSP"
    off = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    main(cand, a, b, off)
