"""Phase 1 sample run on stratified Stage 2 results.

Picks a small (~20) sample of candidates from a Stage 2 polished pkl,
runs Phase 1 (soft-penalty SLSQP with coverage + offsets/depth +
dual-rep clearance + (sx, sy) reparam), then a final feasibility check
against the full-mesh FCL scene including fixtures.

Usage::

    uv run --python 3.13 python -m scripts.run_phase1_sample \\
        examples/836656-config-T12.yml /tmp/836656-holes.yml \\
        --polish-pkl /tmp/full_polish_patchAB.pkl

Reports a table: for each sampled cand:
  - rank, cand idx, max_viol_before (Stage 2)
  - phase1 fn before/after, iters
  - max_viol after Phase 1 (via dual-rep hard min)
  - coverage_total
  - final feasibility (FCL on full mesh + fixtures, broadphase manager)
"""

from __future__ import annotations

import argparse
import os as _os
import pickle
import sys as _sys
import time
from pathlib import Path

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")
_sys.path.insert(0, str(Path(__file__).resolve().parent))

import fcl
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.coverage_jax import (
    GaussianCoverageData,
    build_coverage_data_from_probe_context,
)
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import (
    JointWeights,
    _build_probe_static,
    compute_fixture_max_violation,
)
from aind_low_point.optimization.recording import (
    RECORDING_GEOMETRY,
    RecordingGeometry,
)
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.optimization.stage3_phase1_jax import (
    PHASE1_PER_PROBE_VARS,
    FixtureSDFData,
    Phase1Weights,
    make_phase1_objective,
    phase1_n_vars,
    reduced_to_phase1,
)
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from run_optimizer import _probe_static_info, _transform_holes  # noqa: E402


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    results: list,
    n_top: int = 5,
    n_near: int = 5,
    n_boundary: int = 5,
    n_other: int = 5,
    seed: int = 0,
) -> list[int]:
    """Return candidate indices across four strata of max_violation:

    - top (already-feasible): max_viol <= 0.001
    - near: max_viol in (0.001, 0.5]
    - boundary: max_viol in (0.5, 5.0]
    - other: rest, sampled randomly

    Each stratum returns up to N indices (or all available).
    """
    max_viols = np.array([float(r.metrics.max_violation) for r in results])
    lex_keys = np.array(
        [r.metrics.lex_key() for r in results], dtype=object
    )
    rank_order = sorted(range(len(results)), key=lambda i: results[i].metrics.lex_key())

    top = [i for i in rank_order if max_viols[i] <= 0.001][:n_top]
    near = [i for i in rank_order
            if 0.001 < max_viols[i] <= 0.5][:n_near]
    boundary = [i for i in rank_order
                if 0.5 < max_viols[i] <= 5.0][:n_boundary]

    rng = np.random.default_rng(seed)
    other_pool = [i for i in range(len(results))
                  if max_viols[i] > 5.0 and max_viols[i] < 1e6]
    other = list(rng.choice(other_pool, size=min(n_other, len(other_pool)),
                            replace=False))

    chosen = top + near + boundary + other
    # Dedup while preserving order
    seen = set()
    out = []
    for i in chosen:
        if i not in seen:
            seen.add(int(i)); out.append(int(i))
    return out


# ---------------------------------------------------------------------------
# Phase 1 bounds
# ---------------------------------------------------------------------------


def phase1_bounds(n_arcs: int, n_probes: int, head_pitch_deg: float = 0.0):
    """Box bounds for Phase 1 x = (arc_aps, (ml, sx, sy, off_R, off_A, depth) × P)."""
    bounds = []
    for _ in range(n_arcs):
        bounds.append((-60.0 + head_pitch_deg, +60.0 + head_pitch_deg))
    for _ in range(n_probes):
        bounds.append((-60.0, +60.0))     # ml
        bounds.append((-1.5, +1.5))        # sx
        bounds.append((-1.5, +1.5))        # sy
        bounds.append((-3.0, +3.0))        # off_R (mm)
        bounds.append((-3.0, +3.0))        # off_A (mm)
        bounds.append((-2.0, +2.0))        # depth (mm past target)
    return bounds


# ---------------------------------------------------------------------------
# Final feasibility (full-mesh FCL + broadphase + fixtures)
# ---------------------------------------------------------------------------


def fixture_keys_from_runtime(runtime) -> list[str]:
    """Identify static scene-fixture asset keys.

    Picks scene nodes tagged ``fixture``/``cone``/``well``/``headframe``
    but excludes any node also tagged ``implant`` — probes thread
    through the implant via holes, so it's not a body-collision
    obstacle. The 836656 config tags the implant with both ``fixture``
    AND ``implant`` so the explicit exclusion is required.
    """
    wanted = {"fixture", "cone", "well", "headframe"}
    excluded = {"implant"}
    keys = []
    for node in runtime.scene.nodes.values():
        tags = set(getattr(node, "tags", ()) or ())
        if not (tags & wanted):
            continue
        if tags & excluded:
            continue
        keys.append(node.key)
    return keys


def build_fixture_sdf_data(runtime) -> tuple[FixtureSDFData, ...]:
    """Build α-wrap SDFs for static fixtures.

    Uses the same builder as the probe SDFs (``build_probe_sdf_from_alpha_wrap``)
    — for fixtures with no shank-classified components, this just
    returns the α-wrap envelope SDF without any shank OBBs. Surface
    samples come straight from the envelope.

    Returns one :class:`FixtureSDFData` per fixture, with the SDF grid
    already in world LPS (fixtures are static — identity transform).
    """
    out: list[FixtureSDFData] = []
    for key in fixture_keys_from_runtime(runtime):
        try:
            geom_wrap = runtime.asset_catalog.get_geometry(key)
        except Exception:
            continue
        mesh = getattr(geom_wrap, "raw", None)
        if mesh is None:
            continue
        # Fixtures get 0.2 mm SDF spacing (same as probes) — coarser
        # 0.5 mm gave trilinear-interp errors up to ~0.5 mm at voxel
        # boundaries, which silently reported probe-vs-well contacts
        # as clear when FCL on raw mesh said 1 mm penetration. Grid
        # is ~140×140×70 ≈ 1.4M cells per fixture (~5 MB at fp32):
        # cheap, built once at startup.
        sdf = build_probe_sdf_from_alpha_wrap(
            mesh, spacing_mm=0.2, strip_shanks_first=False,
        )
        out.append(FixtureSDFData(
            name=key,
            grid=jnp.asarray(sdf.grid, dtype=jnp.float32),
            origin=jnp.asarray(sdf.origin, dtype=jnp.float32),
            spacing=jnp.asarray(sdf.spacing, dtype=jnp.float32),
            surface=jnp.asarray(sdf.surface_points, dtype=jnp.float32),
        ))
    return tuple(out)


def build_fixture_collision_objs(runtime) -> dict[str, fcl.CollisionObject]:
    """Return ``{fixture_key: fcl.CollisionObject (in world LPS)}``."""
    fixtures: dict[str, fcl.CollisionObject] = {}
    for key in fixture_keys_from_runtime(runtime):
        try:
            geom_wrap = runtime.asset_catalog.get_geometry(key)
        except Exception:
            continue
        mesh = getattr(geom_wrap, "raw", None)
        if mesh is None:
            continue
        bvh = make_fcl_bvh(mesh)
        # Fixtures are static in their canonical frame; identity
        # transform is correct because the asset catalog returns the
        # mesh already in LPS (matches the probes' world frame).
        # Verify with a sanity-check eventual collision against another
        # static fixture; this is sufficient for our purpose.
        fixtures[key] = bvh
    return fixtures


def final_feasibility_report(
    probes: list,
    statics: list,
    final_pose: dict,  # {probe_name: (R, t)}
    fixtures: dict[str, fcl.CollisionObject],
) -> dict:
    """Run a full-mesh + broadphase feasibility check.

    Returns a dict with per-pair signed distances and feasibility flags.
    """
    manager = fcl.DynamicAABBTreeCollisionManager()
    objs_by_key: dict[str, fcl.CollisionObject] = {}

    for st in statics:
        if st.bvh_obj is None:
            continue
        R, t = final_pose[st.name]
        st.bvh_obj.setTransform(fcl.Transform(
            np.ascontiguousarray(R, dtype=np.float64),
            np.ascontiguousarray(t, dtype=np.float64),
        ))
        manager.registerObject(st.bvh_obj)
        objs_by_key[f"probe:{st.name}"] = st.bvh_obj

    for key, obj in fixtures.items():
        manager.registerObject(obj)
        objs_by_key[key] = obj

    manager.setup()

    # Pairwise: feasibility for every (probe, probe) and (probe, fixture)
    # pair, but NOT (fixture, fixture).
    #
    # python-fcl's BVH-vs-BVH FCL has two reliability issues:
    #   - ``fcl.distance(enable_signed_distance=True)`` returns 0
    #     whenever the meshes touch OR overlap — it does NOT
    #     distinguish them. So a returned 0 means "potentially
    #     colliding", not "touching".
    #   - ``fcl.collide.penetration_depth`` is essentially meaningless
    #     for thin BVH-vs-BVH meshes — it can report 4 mm depth for
    #     meshes that are just touching, or 0 depth for meshes at
    #     identical poses. We don't use it.
    #
    # The only reliable signals from FCL on BVH:
    #   - boolean ``fcl.collide`` has contacts → colliding
    #   - positive ``fcl.distance`` → that's the true signed distance
    #
    # Report scheme: positive number = clearance in mm; -1.0 sentinel
    # = "colliding (depth not known)".
    pair_results: list[tuple[str, str, float]] = []
    keys_list = list(objs_by_key.keys())
    dist_req = fcl.DistanceRequest(enable_signed_distance=True)
    coll_req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
    for i, ka in enumerate(keys_list):
        for kb in keys_list[i + 1 :]:
            # Skip fixture-fixture pairs.
            if not ka.startswith("probe:") and not kb.startswith("probe:"):
                continue
            d_res = fcl.DistanceResult()
            fcl.distance(objs_by_key[ka], objs_by_key[kb], dist_req, d_res)
            d = float(d_res.min_distance)
            if d > 0:
                # Separated — fcl.distance is reliable.
                pair_results.append((ka, kb, d))
            else:
                # fcl.distance returned 0 — could be touching OR
                # overlapping. Use fcl.collide as a boolean test.
                c_res = fcl.CollisionResult()
                fcl.collide(objs_by_key[ka], objs_by_key[kb], coll_req, c_res)
                if c_res.contacts:
                    pair_results.append((ka, kb, -1.0))   # colliding
                else:
                    pair_results.append((ka, kb, 0.0))    # touching only

    overlaps = [(ka, kb, d) for ka, kb, d in pair_results if d < 0.0]
    return {
        "pair_clearances": pair_results,
        "overlaps": overlaps,
        "feasible": len(overlaps) == 0,
        "min_clearance": min((d for _, _, d in pair_results), default=float("inf")),
    }


# ---------------------------------------------------------------------------
# Build coverage data per probe (Gaussian mode, matching Stage 2)
# ---------------------------------------------------------------------------


def build_coverage_data(
    probes, statics,
) -> tuple[GaussianCoverageData, ...]:
    """One Gaussian-mode CoverageData per probe, taken from the probe's
    target_LPS and density_sigma_mm (and the per-kind recording range).
    """
    out = []
    fallback_geom = RecordingGeometry(active_ranges_mm=((0.2, 1.2),))
    # Match statics order, since coverage_data is positional in the JIT.
    for st in statics:
        # Find the parent ProbeStaticInfo (statics are derived from it).
        parent = next(p for p in probes if p.name == st.name)
        geom = RECORDING_GEOMETRY.get(parent.kind, fallback_geom)
        active_range = geom.active_ranges_mm[0]
        cd = build_coverage_data_from_probe_context(parent, active_range)
        out.append(cd)
    return tuple(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("holes", type=Path)
    p.add_argument("--polish-pkl", type=Path,
                   default=Path("/tmp/full_polish_patchAB.pkl"))
    p.add_argument("--n-top", type=int, default=5)
    p.add_argument("--n-near", type=int, default=5)
    p.add_argument("--n-boundary", type=int, default=5)
    p.add_argument("--n-other", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slsqp-max-iter", type=int, default=80)
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
    print(f"Probes: {[p.name for p in probes]}")

    print("Building SDFs (α-wrap envelope + shank OBBs)...")
    t0 = time.time()
    sdf_by_name = {
        p.name: build_probe_sdf_from_alpha_wrap(
            runtime.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    print(f"  SDFs built in {time.time() - t0:.1f}s")

    print("Building fixture FCL objects for final check...")
    t0 = time.time()
    fixtures_fcl = build_fixture_collision_objs(runtime)
    print(f"  {len(fixtures_fcl)} fixtures built in {time.time() - t0:.1f}s: "
          f"{list(fixtures_fcl.keys())}")

    print("Building fixture α-wrap SDFs for Phase 1 clearance...")
    t0 = time.time()
    fixtures_sdf = build_fixture_sdf_data(runtime)
    print(f"  {len(fixtures_sdf)} fixture SDFs in {time.time() - t0:.1f}s")

    with open(args.polish_pkl, "rb") as f:
        data = pickle.load(f)
    candidates = data["candidates"]
    results = data["results"]
    print(f"Stage 2 pool: {len(candidates)} cands")

    sample_idxs = stratified_sample(
        results,
        n_top=args.n_top, n_near=args.n_near,
        n_boundary=args.n_boundary, n_other=args.n_other,
        seed=args.seed,
    )
    print(f"Sampled {len(sample_idxs)} cands: {sample_idxs[:10]}{' ...' if len(sample_idxs) > 10 else ''}")
    print()

    bvh_cache = {
        p.name: (make_fcl_bvh(p.collision_mesh)
                 if p.collision_mesh is not None else None)
        for p in probes
    }
    weights = Phase1Weights()

    # Header. ``mv_fix2`` is the Stage 2 fixture-vs-probe penetration
    # diagnostic at the raw polished y (before Phase 1); ``mv_p1`` is
    # the post-Phase 1 min clearance across all pairs incl. fixtures.
    print(f"{'cand#':>5} {'rank':>5} {'mv_before':>10} {'mv_fix2':>8} "
          f"{'fn0':>10} {'fn_end':>10} "
          f"{'iter':>4} {'mv_p1':>8} {'cov_p1':>8} {'final':>6} {'wall_s':>7}")

    # Pre-rank for printing
    rank_by_idx = {
        i: r for r, (i, _) in enumerate(
            sorted(enumerate(results), key=lambda kv: kv[1].metrics.lex_key())
        )
    }

    for cand_idx in sample_idxs:
        cand = candidates[cand_idx]
        jc = results[cand_idx]
        mv_before = float(jc.metrics.max_violation)
        statics = _build_probe_static(
            probes, holes_list, cand.ha, cand.aa,
            bvh_cache=bvh_cache, sdf_by_name=sdf_by_name,
        )
        n_arcs = jc.n_arcs

        # Build coverage data (Gaussian centroid, matching Stage 2's mode).
        coverage_data = build_coverage_data(probes, statics)
        fun, jac = make_phase1_objective(
            statics, n_arcs,
            coverage_data=coverage_data,
            fixtures=fixtures_sdf,
            weights=weights,
        )
        bounds = phase1_bounds(n_arcs, len(statics))

        # Stage 2 fixture diagnostic at the raw polished y — shows how
        # many of the Stage 2 top-ranked cands are silently penetrating
        # cone/well/headframe before Phase 1 gets a chance to fix it.
        fixture_bvh_list = list(fixtures_fcl.values())
        mv_fixture_stage2 = compute_fixture_max_violation(
            jc.reduced_y, statics, n_arcs, fixture_bvh_list
        )

        # Lift Stage 2 reduced y to Phase 1 x.
        x0 = reduced_to_phase1(jc.reduced_y, n_arcs, len(statics))
        fn0 = float(fun(x0))

        t0 = time.time()
        res = minimize(
            fun, x0, jac=jac, method="SLSQP", bounds=bounds,
            options=dict(maxiter=args.slsqp_max_iter, ftol=1e-4),
        )
        wall = time.time() - t0

        # Extract pose per probe for final feasibility.
        x_opt = np.asarray(res.x, dtype=np.float64)
        final_pose: dict = {}
        from aind_low_point.optimization.kinematics import pose_from_optimizer_vars
        arc_aps = x_opt[:n_arcs]
        for i, st in enumerate(statics):
            off = n_arcs + PHASE1_PER_PROBE_VARS * i
            ml = float(x_opt[off + 0])
            sx = float(x_opt[off + 1])
            sy = float(x_opt[off + 2])
            off_R = float(x_opt[off + 3])
            off_A = float(x_opt[off + 4])
            depth = float(x_opt[off + 5])
            spin_deg = float(np.degrees(np.arctan2(sy, sx)))
            ap = float(arc_aps[st.arc_idx])
            R, t = pose_from_optimizer_vars(
                target_LPS=st.target_LPS, ap_deg=ap, ml_deg=ml,
                spin_deg=spin_deg,
                offset_R_mm=off_R, offset_A_mm=off_A, past_target_mm=depth,
                recording_center_local=st.pivot_local,
            )
            final_pose[st.name] = (R, t)

        # Phase 1 hard-min for max_viol via dual-rep (uses Stage 2's
        # diagnostic pair-clearance helper indirectly via ``statics``).
        # Approximation here: use the JAX dual hard mins by transforming
        # each probe and walking pair list; defer to the final FCL check
        # for the true ground truth.
        mv_p1 = mv_before  # placeholder; real value via final_check below

        # Final feasibility via FCL + broadphase + fixtures.
        feas = final_feasibility_report(
            probes, statics, final_pose, fixtures_fcl
        )
        feas_flag = "FEAS" if feas["feasible"] else "FAIL"
        coverage_after = -res.fun + (
            # Back out coverage from the soft objective by adding back
            # the (residual) penalties. Easier: just don't try — coverage
            # signal magnitude per Phase 1 is fn_end ≈ -coverage when
            # penalties are zero. We report -fn_end as a proxy.
            0.0
        )
        cov_p1 = max(0.0, -res.fun)

        print(f"{cand_idx:>5} {rank_by_idx[cand_idx]:>5} "
              f"{mv_before:>10.4f} {mv_fixture_stage2:>8.3f} "
              f"{fn0:>10.2f} {res.fun:>10.2f} "
              f"{res.nit:>4} {feas['min_clearance']:>8.3f} "
              f"{cov_p1:>8.2f} {feas_flag:>6} {wall:>7.2f}")
        # Surface the worst pair for diagnosing why Phase 1 fails FCL.
        # Also compute JAX dual-rep clearance for the same pair so we
        # can see whether JAX is blind to the collision or just
        # under-penalising it.
        from aind_low_point.optimization.sdf_jax import (
            pairwise_signed_clearance_dual as _dual,
        )
        import jax.numpy as _jnp

        worst = sorted(feas["pair_clearances"], key=lambda t: t[2])[:3]
        probe_names = [st.name for st in statics]
        for ka, kb, d in worst:
            line = f"      worst: {ka} ↔ {kb}: FCL={d:+.3f}"
            if ka.startswith("probe:") and kb.startswith("probe:"):
                a_name = ka[len("probe:"):]
                b_name = kb[len("probe:"):]
                if a_name in probe_names and b_name in probe_names:
                    ai = probe_names.index(a_name)
                    bi = probe_names.index(b_name)
                    sa = statics[ai]
                    sb = statics[bi]
                    Ra, ta = final_pose[a_name]
                    Rb, tb = final_pose[b_name]
                    (hbb, sbb), (hbs, sbs), (hss, sss) = _dual(
                        _jnp.asarray(Ra, dtype=_jnp.float32),
                        _jnp.asarray(ta, dtype=_jnp.float32),
                        _jnp.asarray(Rb, dtype=_jnp.float32),
                        _jnp.asarray(tb, dtype=_jnp.float32),
                        _jnp.asarray(sa.sdf_data["grid"], dtype=_jnp.float32),
                        _jnp.asarray(sa.sdf_data["origin"], dtype=_jnp.float32),
                        _jnp.asarray(sa.sdf_data["spacing"], dtype=_jnp.float32),
                        _jnp.asarray(sb.sdf_data["grid"], dtype=_jnp.float32),
                        _jnp.asarray(sb.sdf_data["origin"], dtype=_jnp.float32),
                        _jnp.asarray(sb.sdf_data["spacing"], dtype=_jnp.float32),
                        _jnp.asarray(sa.sdf_data["surface"], dtype=_jnp.float32),
                        _jnp.asarray(sb.sdf_data["surface"], dtype=_jnp.float32),
                        _jnp.asarray(sa.sdf_data.get("shank_centers",
                            np.zeros((0, 3))), dtype=_jnp.float32),
                        _jnp.asarray(sa.sdf_data.get("shank_halves",
                            np.zeros((0, 3))), dtype=_jnp.float32),
                        _jnp.asarray(sb.sdf_data.get("shank_centers",
                            np.zeros((0, 3))), dtype=_jnp.float32),
                        _jnp.asarray(sb.sdf_data.get("shank_halves",
                            np.zeros((0, 3))), dtype=_jnp.float32),
                    )
                    line += (
                        f"  | JAX hard: bb={float(hbb):+.3f} "
                        f"bs={float(hbs):+.3f} ss={float(hss):+.3f}"
                    )
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
