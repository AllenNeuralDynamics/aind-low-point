"""Isolated parity check: JAX α-wrap SDF clearance vs FCL on full mesh.

Walks one specific cand from the polished pkl through the same flow
the orchestrator uses, then compares per-pair clearance at the *same*
pose between:

  - JAX dual-rep (body-body, body-shank, shank-shank, with alpha-wrap)
  - FCL on raw collision_mesh (penetration depth on overlap)

Reports the discrepancy and the worst-case vertex involved, to nail
down whether the bug is:

  (a) frame mismatch between BVH and alpha-wrap envelope
  (b) SDF cache returning wrong data for the wrong probe
  (c) trilinear interp / out-of-bounds artifact
  (d) shank OBBs not capturing the actual silicon shank geometry
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

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.kinematics import pose_from_optimizer_vars
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.sdf_jax import (
    pairwise_signed_clearance_dual,
    trilinear_sdf,
)
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms


def _fcl_pair_signed_clearance(obj_a, obj_b) -> float:
    """Signed clearance: positive = clear, negative = penetration depth."""
    dr = fcl.DistanceResult()
    fcl.distance(obj_a, obj_b, fcl.DistanceRequest(enable_signed_distance=True), dr)
    d = float(dr.min_distance)
    if d > 0.0:
        return d
    cr = fcl.CollisionResult()
    fcl.collide(
        obj_a, obj_b, fcl.CollisionRequest(num_max_contacts=8, enable_contact=True), cr
    )
    if cr.contacts:
        return -float(max(c.penetration_depth for c in cr.contacts))
    return 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument(
        "--polish-pkl", type=Path, default=Path("/tmp/full_polish_patchAB.pkl")
    )
    p.add_argument("--cand", type=int, default=5113)
    p.add_argument("--probe-a", type=str, default="VM")
    p.add_argument("--probe-b", type=str, default="CA1")
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
        p.name: (
            make_fcl_bvh(p.collision_mesh) if p.collision_mesh is not None else None
        )
        for p in probes
    }

    # === Sanity 1: ensure each probe's SDF matches its OWN raw mesh ===
    print("=" * 70)
    print("Sanity 1: probe SDF surface_points should sit AT alpha-wrap envelope")
    print("=" * 70)
    for p in probes:
        if p.name not in (args.probe_a, args.probe_b):
            continue
        sdf = sdf_by_name[p.name]
        # Lookup each surface point in its own SDF — should be ~ 0 mm
        # (these ARE the surface samples).
        surf = jnp.asarray(sdf.surface_points, dtype=jnp.float32)
        grid = jnp.asarray(sdf.grid, dtype=jnp.float32)
        origin = jnp.asarray(sdf.origin, dtype=jnp.float32)
        spacing = jnp.asarray(sdf.spacing, dtype=jnp.float32)
        d = trilinear_sdf(grid, origin, spacing, surf)
        d_np = np.asarray(d)
        print(
            f"  {p.name} (kind={p.kind}): self-SDF on its own surface samples: "
            f"mean={d_np.mean():+.4f}, max|d|={np.max(np.abs(d_np)):+.4f}"
        )

    # === Sanity 2: are SDFs different objects for VM and CA1? ===
    print()
    print("=" * 70)
    print("Sanity 2: VM and CA1 are both kind '2.1' — are their SDFs the SAME object?")
    print("=" * 70)
    sa = sdf_by_name[args.probe_a]
    sb = sdf_by_name[args.probe_b]
    print(f"  same SDF object: {sa is sb}")
    print(f"  same grid array (id): {id(sa.grid) == id(sb.grid)}")
    print(f"  grid shapes: {sa.grid.shape} vs {sb.grid.shape}")
    print(f"  origins: {sa.origin} vs {sb.origin}")

    # === Pull cand polished y from pkl and reconstruct poses ===
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
    arc_aps = np.asarray(jc.reduced_y[:n_arcs], dtype=np.float64)

    # Find the two probes in the statics list
    idx_a = next(i for i, st in enumerate(statics) if st.name == args.probe_a)
    idx_b = next(i for i, st in enumerate(statics) if st.name == args.probe_b)
    sa_static = statics[idx_a]
    sb_static = statics[idx_b]

    # Extract reduced y → (ml, sx, sy) for each, build the *Stage 2* pose
    # (offsets=0, depth=0). Phase 1 was the one introducing the issue; we
    # check at Stage 2's pose first to see if the bug exists there too.
    def _stage2_pose(st, idx):
        off = n_arcs + 3 * idx
        ml = float(jc.reduced_y[off + 0])
        sx = float(jc.reduced_y[off + 1])
        sy = float(jc.reduced_y[off + 2])
        spin_deg = float(np.degrees(np.arctan2(sy, sx)))
        ap = float(arc_aps[st.arc_idx])
        R, t = pose_from_optimizer_vars(
            target_LPS=st.target_LPS,
            ap_deg=ap,
            ml_deg=ml,
            spin_deg=spin_deg,
            offset_R_mm=0.0,
            offset_A_mm=0.0,
            past_target_mm=0.0,
            recording_center_local=st.pivot_local,
        )
        return R, t

    print()
    print("=" * 70)
    print(
        f"At Stage 2 polished pose for cand #{args.cand} "
        f"({args.probe_a} vs {args.probe_b})"
    )
    print("=" * 70)
    Ra, ta = _stage2_pose(sa_static, idx_a)
    Rb, tb = _stage2_pose(sb_static, idx_b)
    print(f"  {args.probe_a}: t={ta.round(3).tolist()}")
    print(f"  {args.probe_b}: t={tb.round(3).tolist()}")
    print(f"  ||t_a − t_b|| = {np.linalg.norm(ta - tb):.3f} mm")

    # FCL on full mesh
    sa_static.bvh_obj.setTransform(
        fcl.Transform(
            np.ascontiguousarray(Ra, dtype=np.float64),
            np.ascontiguousarray(ta, dtype=np.float64),
        )
    )
    sb_static.bvh_obj.setTransform(
        fcl.Transform(
            np.ascontiguousarray(Rb, dtype=np.float64),
            np.ascontiguousarray(tb, dtype=np.float64),
        )
    )
    fcl_d = _fcl_pair_signed_clearance(sa_static.bvh_obj, sb_static.bvh_obj)
    print(f"  FCL (full mesh): {fcl_d:+.4f} mm")

    # JAX dual-rep
    sdf_a = sdf_by_name[args.probe_a]
    sdf_b = sdf_by_name[args.probe_b]
    (hbb, sbb), (hbs, sbs), (hss, sss) = pairwise_signed_clearance_dual(
        jnp.asarray(Ra, dtype=jnp.float32),
        jnp.asarray(ta, dtype=jnp.float32),
        jnp.asarray(Rb, dtype=jnp.float32),
        jnp.asarray(tb, dtype=jnp.float32),
        jnp.asarray(sdf_a.grid, dtype=jnp.float32),
        jnp.asarray(sdf_a.origin, dtype=jnp.float32),
        jnp.asarray(sdf_a.spacing, dtype=jnp.float32),
        jnp.asarray(sdf_b.grid, dtype=jnp.float32),
        jnp.asarray(sdf_b.origin, dtype=jnp.float32),
        jnp.asarray(sdf_b.spacing, dtype=jnp.float32),
        jnp.asarray(sdf_a.surface_points, dtype=jnp.float32),
        jnp.asarray(sdf_b.surface_points, dtype=jnp.float32),
        jnp.asarray(sdf_a.shank_centers, dtype=jnp.float32),
        jnp.asarray(sdf_a.shank_halves, dtype=jnp.float32),
        jnp.asarray(sdf_b.shank_centers, dtype=jnp.float32),
        jnp.asarray(sdf_b.shank_halves, dtype=jnp.float32),
    )
    print(f"  JAX hard: bb={float(hbb):+.4f} bs={float(hbs):+.4f} ss={float(hss):+.4f}")

    # === Direct probe: which point on probe A is deepest inside probe B? ===
    print()
    print("=" * 70)
    print("Direct test: every RAW vertex of probe A → world → B local → SDF")
    print("=" * 70)
    probe_a_info = next(p for p in probes if p.name == args.probe_a)
    asset_a = runtime.asset_catalog.get_geometry(f"probe:{probe_a_info.kind}").raw
    raw_a = np.asarray(asset_a.vertices, dtype=np.float64)
    # Transform raw A vertices to world via probe A's pose
    world_a = raw_a @ Ra.T + ta
    # Transform to B's local frame
    local_in_b = (world_a - tb) @ Rb
    # Lookup B's SDF
    d_raw = np.asarray(
        trilinear_sdf(
            jnp.asarray(sdf_b.grid, dtype=jnp.float32),
            jnp.asarray(sdf_b.origin, dtype=jnp.float32),
            jnp.asarray(sdf_b.spacing, dtype=jnp.float32),
            jnp.asarray(local_in_b, dtype=jnp.float32),
        )
    )
    in_b = d_raw < 1e2
    if in_b.sum() > 0:
        d_in = d_raw[in_b]
        worst_idx = int(np.argmin(d_in))
        worst_idx_global = int(np.argsort(d_raw)[0])
        print(f"  raw-A vertices in B's SDF grid: {in_b.sum()}/{len(d_raw)}")
        print(
            f"  worst raw-A vertex in B: SDF = {d_in.min():+.4f} mm "
            f"(at vertex {worst_idx_global} of probe A's raw mesh)"
        )
        print(f"  10 deepest: {sorted(d_in)[:10]}")

    # === Direct test 2: get FCL contact points, look them up in BOTH SDFs ===
    print()
    print("=" * 70)
    print("Direct test 2: FCL contact points → both SDFs")
    print("=" * 70)
    cr = fcl.CollisionResult()
    fcl.collide(
        sa_static.bvh_obj,
        sb_static.bvh_obj,
        fcl.CollisionRequest(num_max_contacts=8, enable_contact=True),
        cr,
    )
    print(f"  FCL contacts: {len(cr.contacts) if cr.contacts else 0}")
    for k, c in enumerate(cr.contacts or []):
        pos_world = np.asarray(c.pos, dtype=np.float64)
        depth = float(c.penetration_depth)
        # Transform contact to A's local + look up A's SDF
        local_a = (pos_world - ta) @ Ra
        d_in_a = float(
            np.asarray(
                trilinear_sdf(
                    jnp.asarray(sdf_a.grid, dtype=jnp.float32),
                    jnp.asarray(sdf_a.origin, dtype=jnp.float32),
                    jnp.asarray(sdf_a.spacing, dtype=jnp.float32),
                    jnp.asarray(local_a.reshape(1, 3), dtype=jnp.float32),
                )
            )[0]
        )
        # Transform contact to B's local + look up B's SDF
        local_b = (pos_world - tb) @ Rb
        d_in_b = float(
            np.asarray(
                trilinear_sdf(
                    jnp.asarray(sdf_b.grid, dtype=jnp.float32),
                    jnp.asarray(sdf_b.origin, dtype=jnp.float32),
                    jnp.asarray(sdf_b.spacing, dtype=jnp.float32),
                    jnp.asarray(local_b.reshape(1, 3), dtype=jnp.float32),
                )
            )[0]
        )
        print(f"  contact[{k}] world={pos_world.round(3).tolist()} depth={depth:+.4f}")
        print(
            f"    A-SDF at this point: {d_in_a:+.4f} mm "
            f"(local_a={local_a.round(3).tolist()})"
        )
        print(
            f"    B-SDF at this point: {d_in_b:+.4f} mm "
            f"(local_b={local_b.round(3).tolist()})"
        )
        # Check if local is within shank OBB
        for label, sdf, R, t in [("A", sdf_a, Ra, ta), ("B", sdf_b, Rb, tb)]:
            local = (pos_world - t) @ R
            for s_i in range(sdf.shank_centers.shape[0]):
                ctr = np.asarray(sdf.shank_centers[s_i])
                hal = np.asarray(sdf.shank_halves[s_i])
                q = np.abs(local - ctr) - hal
                inside = np.all(q <= 0)
                if inside:
                    margin = -float(q.max())
                    print(
                        f"    INSIDE {label}'s shank OBB #{s_i} "
                        f"(margin={margin:.4f} mm)"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
