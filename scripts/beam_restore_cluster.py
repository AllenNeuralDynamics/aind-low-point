"""(B) Does the beam's SET of spin proposals survive the round-robin restore,
or collapse to a few basins?

Generate the beam's spin assignments for the manual candidate (4195), push ALL
of them through the batched restore (with well) in one call, then cluster the
restored spin vectors. Many proposals → few clusters ⇒ restore is a strong
attractor and the beam is redundant. Many proposals → many clusters ⇒ restore
preserves the beam's diversity and the beam is a real generator.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.beam_restore_cluster
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle
from collections import Counter

import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.batched_objective import (
    make_batched_reduced_objective,
)
from aind_low_point.optimization.batched_spin_restore import (
    make_batched_spin_restore_partial,
)
from aind_low_point.optimization.batched_static import build_batched_probe_static
from aind_low_point.optimization.joint_rerank import JointWeights, _build_probe_static
from scripts.restore_seed_compare import facing_seeds
from scripts.restore_well_adam_manual import setup, spins_deg_from_reduced
from scripts.spin_heuristic_search import (
    beam_search_assignments,
    build_coupling_graph,
    per_probe_spin_candidates,
)

IDX = 4195
BEAM_B = 64
CLUSTER_DEG = 25.0   # round each probe spin to this bin when clustering basins


def _bin(sp):
    return tuple(int(round(_wrap(s) / CLUSTER_DEG)) for s in sp)


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def main() -> int:
    _cfg, rt, probes, holes, sdf_by_name, bvh, _fx, well, _fb = setup()
    K = len(probes)
    names = [p.name for p in probes]
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    cand = pool["candidates"][IDX]
    n_arcs = int(pool["results"][IDX].n_arcs)
    st = _build_probe_static(probes, holes, cand.ha, cand.aa,
                             bvh_cache=bvh, sdf_by_name=sdf_by_name)
    probe_kind = {p.name: p.kind for p in probes}

    arc_aps = np.array([float(cand.aa.arc_centroids_deg[a]) for a in range(n_arcs)])
    ml = np.array([float(cand.ml_seed.get(p.name, 0.0)) for p in probes])
    target_LPS = np.array([s.target_LPS for s in st])
    seed_spins = {i: float(cand.spin_seed.get(p.name, 0.0))
                  for i, p in enumerate(probes)}

    # --- beam: a SET of spin proposals -------------------------------------
    mesh_by_kind = {
        k: np.asarray(rt.asset_catalog.get_geometry(f"probe:{k}").raw.vertices)
        for k in set(probe_kind.values())
    }
    coupling, _t, _c = build_coupling_graph(
        st, arc_aps, ml, target_LPS, mesh_by_kind, probe_kind)
    cands = per_probe_spin_candidates(
        st, coupling, target_LPS, arc_aps, ml, probe_kind, seed_spins=seed_spins)
    beam = beam_search_assignments(
        st, cands, coupling, target_LPS, arc_aps, ml, probe_kind, beam_B=BEAM_B)
    proposals = []
    for a in beam:
        d = dict(a.spins)
        proposals.append(np.array([_wrap(d[i]) for i in range(K)]))
    # Append the direct heuristic facing output as the LAST row, to see which
    # restored basin it lands in vs the beam set.
    facing = facing_seeds(rt, st, arc_aps, ml, target_LPS, probe_kind, well, names)
    facing_vec = np.array([_wrap(facing[n]) for n in names])
    proposals.append(facing_vec)
    proposals = np.array(proposals)
    n_prop = len(proposals)
    fac_idx = n_prop - 1
    n_prop_basins = len(set(_bin(p) for p in proposals))
    print(f"cand {IDX}  probes={names}")
    print(f"heuristic facing output: {facing_vec.round(0).astype(int).tolist()}")
    print(f"beam proposals (+facing): {n_prop}  →  {n_prop_basins} distinct "
          f"pre-restore basins (binned {CLUSTER_DEG:.0f}°)\n")

    # --- batched restore of ALL proposals in one call ----------------------
    pairs = [(cand.ha, cand.aa)] * n_prop
    bs = build_batched_probe_static(
        pairs, probes, holes, n_arcs=n_arcs, sdf_by_name=sdf_by_name,
        head_pitch_deg=0.0)
    weights = JointWeights()
    restore = make_batched_spin_restore_partial(
        bs, weights, n_spins=8, n_rounds=2, fixtures=(well,))
    obj, _ = make_batched_reduced_objective(bs, weights, (well,))
    varying = obj.extract_arrays(bs)

    y0 = np.zeros((n_prop, n_arcs + 3 * K), np.float32)
    for b in range(n_prop):
        for a in range(n_arcs):
            y0[b, a] = arc_aps[a]
        for k in range(K):
            sp = np.deg2rad(proposals[b, k])
            y0[b, n_arcs + 3 * k] = ml[k]
            y0[b, n_arcs + 3 * k + 1] = np.cos(sp)
            y0[b, n_arcs + 3 * k + 2] = np.sin(sp)
    y_r = np.asarray(restore(jnp.asarray(y0), *varying))

    restored = np.array([spins_deg_from_reduced(y_r[b], n_arcs, K)
                         for b in range(n_prop)])
    basins = Counter(_bin(r) for r in restored)
    fac_key = _bin(restored[fac_idx])
    print(f"facing restored to: {np.round(restored[fac_idx], 0).astype(int).tolist()}")
    print(f"after restore: {n_prop} proposals → {len(basins)} distinct basins "
          f"(binned {CLUSTER_DEG:.0f}°)\n")
    print("  basin (representative restored spins)            count   facing?")
    reps = {}
    for r in restored:
        reps.setdefault(_bin(r), np.round(r, 0).astype(int).tolist())
    for key, n in basins.most_common():
        tag = "  <-- heuristic facing" if key == fac_key else ""
        print(f"  {str(reps[key]):<48} {n:>4}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
