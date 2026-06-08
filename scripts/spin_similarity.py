"""Do the restore and the beam reproduce the durable feasible spin basin?

For each high-quality stage-3 candidate, compare its FINAL (Phase-2, feasible)
per-probe spin to (a) the round-robin restore output and (b) the nearest beam
proposal — pure angle comparison, no ADAM. Tells us whether the restore lands on
the durable basin, and whether the beam's set even contains it.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.spin_similarity
Env:  IDXS=4195,1035,4747,6423,697
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle

import numpy as np

from aind_low_point.optimization.joint_rerank import _build_probe_static
from scripts.restore_seed_compare import facing_seeds
from scripts.restore_well_adam_manual import (
    run_restore,
    setup,
    spins_deg_from_phase1,
    spins_deg_from_reduced,
)
from scripts.spin_heuristic_search import (
    beam_search_assignments,
    build_coupling_graph,
    per_probe_spin_candidates,
)

IDXS = [int(x) for x in _os.environ.get("IDXS", "4195,1035,4747,6423,697").split(",")]
BEAM_B = 64
TOL = 25.0  # within this many degrees counts as "matched"


def _wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def main() -> int:
    _cfg, rt, probes, holes, sdf_by_name, bvh, _fx, well, _fb = setup()
    K = len(probes)
    names = [p.name for p in probes]
    probe_kind = {p.name: p.kind for p in probes}
    mesh_by_kind = {
        k: np.asarray(rt.asset_catalog.get_geometry(f"probe:{k}").raw.vertices)
        for k in set(probe_kind.values())
    }
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    h2 = pickle.load(open("scratch/phase2_handoff.pkl", "rb"))
    pose_by_idx = {
        r["idx"]: (r["pose"], r["n_arcs"], r["coverage"], r["fcl"])
        for r in h2["all"]
        if r["fcl"] >= -0.2
    }

    for idx in IDXS:
        if idx not in pose_by_idx:
            print(f"\ncand {idx}: no feasible stage-3 pose — skipped")
            continue
        pose, n_arcs, dcov, dfcl = pose_by_idx[idx]
        pose = np.asarray(pose, np.float64)
        cand = pool["candidates"][idx]
        st = _build_probe_static(
            probes, holes, cand.ha, cand.aa, bvh_cache=bvh, sdf_by_name=sdf_by_name
        )
        durable = spins_deg_from_phase1(pose, n_arcs, K)
        arc_aps = pose[:n_arcs]
        mls = np.array([pose[n_arcs + 6 * k] for k in range(K)])
        target_LPS = np.array([s.target_LPS for s in st])
        seed = {i: float(cand.spin_seed.get(p.name, 0.0)) for i, p in enumerate(probes)}

        # restore output spins (production seed path)
        y2 = run_restore(
            cand, probes, holes, sdf_by_name, n_arcs, well, with_well=True, n_rounds=2
        )
        rest = spins_deg_from_reduced(y2, n_arcs, K)

        # beam proposal set
        coupling, _t, _c = build_coupling_graph(
            st, arc_aps, mls, target_LPS, mesh_by_kind, probe_kind
        )
        cands = per_probe_spin_candidates(
            st, coupling, target_LPS, arc_aps, mls, probe_kind, seed_spins=seed
        )
        beam = beam_search_assignments(
            st, cands, coupling, target_LPS, arc_aps, mls, probe_kind, beam_B=BEAM_B
        )
        facing = facing_seeds(rt, st, arc_aps, mls, target_LPS, probe_kind, well, names)
        props = [np.array([_wrap(dict(a.spins)[i]) for i in range(K)]) for a in beam]
        props.append(np.array([_wrap(facing[n]) for n in names]))
        props = np.array(props)

        d_rest = np.abs(_wrap(rest - durable))
        # best beam proposal = min over proposals of the worst-probe gap
        worst = np.abs(_wrap(props - durable)).max(axis=1)
        bi = int(worst.argmin())
        d_beam = np.abs(_wrap(props[bi] - durable))
        # per-probe: how many proposals match the durable spin within TOL
        per_probe_hits = (np.abs(_wrap(props - durable)) <= TOL).sum(axis=0)

        print(
            f"\ncand {idx}  (durable stage-3: cov {dcov:.2f}, fcl {dfcl:+.3f})  "
            f"probes={names}"
        )
        print(f"  durable spins : {np.round(durable, 0).astype(int).tolist()}")
        print(
            f"  restore spins : {np.round(rest, 0).astype(int).tolist()}  "
            f"Δ={np.round(d_rest, 0).astype(int).tolist()}  max {d_rest.max():.0f}°"
        )
        print(
            f"  best beam     : {np.round(props[bi], 0).astype(int).tolist()}  "
            f"Δ={np.round(d_beam, 0).astype(int).tolist()}  max {d_beam.max():.0f}°"
        )
        print(
            f"  restore matches durable (≤{TOL:.0f}°): "
            f"{int((d_rest <= TOL).sum())}/{K} probes"
        )
        print(
            f"  beam-set per-probe coverage (≤{TOL:.0f}°): "
            f"{per_probe_hits.tolist()}  of {len(props)} proposals"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
