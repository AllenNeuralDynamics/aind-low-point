"""Prototype: hole-first MRV enumerator with bitset domains + 1-D Helly
clique AP-feasibility + shared greedy-stab (intra-arc ML packing AND inter-arc
AP separation, which also emits the warm-start ml seed).

Consumes the SAME visibility atlas as production's
``enumerate_arc_first_candidates`` (via ``_build_atlas_arrays``), so the
per-(probe,hole) AP intervals and ml-ranges are identical — the only changes
are (a) search order/structure (hole-first MRV + forward-checked uniqueness)
and (b) ML feasibility (greedy joint packing instead of pairwise max-diff).

Validation (main): the enumerated discrete decisions {probe->hole, partition}
MUST be a superset of the 45 FCL-feasible candidates from phase2_handoff.pkl,
and must contain the manual T12 hole-assignment. Any dropped feasible is a
regression (likely the stricter ML pack rejecting an atlas-range-tight tuple
whose true SLSQP ml lies outside the atlas anchors — reported if it happens).

Run:  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.arc_first_mrv
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
import time
from pathlib import Path

from aind_low_point.planning import PoseLimits

CONFIG = "examples/836656-config-T12.yml"
HOLES = "scratch/0283-300-04.holes.yml"
ATLAS_CACHE = "scratch/atlas_0283.pkl"
POOL_PKL = "scratch/full_polish_0283.pkl"
HANDOFF_PKL = "scratch/phase2_handoff.pkl"
MANUAL_H = {"MD": 3, "BLA": 4, "PL": 1, "VM": 7, "RSP": 5, "CA1": 10, "CLA": 12}

# Arc / per-arc caps are KINEMATIC (16° angular exclusion over the AP/ML
# range), not hardware counts — no rail limit, the rig takes >4 per arc.
_POSE_LIMITS = PoseLimits()
MAX_ARCS = _POSE_LIMITS.max_arcs()
MAX_PROBES_PER_ARC = _POSE_LIMITS.max_probes_per_arc()
MIN_ARC_AP_SEP_DEG = 16.0
MIN_ML_SEP_DEG = 16.0
GLOBAL_CAP = 1_000_000


# --------------------------------------------------------------------------
# Atlas (build once, cache).
# --------------------------------------------------------------------------
def build_or_load_atlas():
    if Path(ATLAS_CACHE).exists():
        with open(ATLAS_CACHE, "rb") as f:
            return pickle.load(f)
    from aind_low_point.config import ConfigModel
    from aind_low_point.optimization.holes import load_holes
    from aind_low_point.optimization.visibility_atlas import build_visibility_atlas
    from aind_low_point.runtime import build_runtime_from_config
    from aind_low_point.runtime.transforms import compile_all_transforms
    from scripts.run_optimizer import _probe_static_info, _transform_holes

    cfg = ConfigModel.from_yaml(CONFIG)
    rt = build_runtime_from_config(cfg)
    probes = [_probe_static_info(rt.plan_state, rt, n) for n in rt.plan_state.probes]
    holes = load_holes(Path(HOLES))
    comp = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in comp:
        R, t = comp["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    t0 = time.time()
    atlas = build_visibility_atlas(probes, holes, n_top=128, n_spin=72, verbose=False)
    print(
        f"built visibility atlas in {time.time() - t0:.0f}s "
        f"({len(probes)} probes, {len(holes)} holes)"
    )
    probe_names = [p.name for p in probes]
    payload = (atlas, probe_names)
    with open(ATLAS_CACHE, "wb") as f:
        pickle.dump(payload, f)
    return payload


# --------------------------------------------------------------------------
# Shared greedy stab: place one point per interval, sorted, consecutive gaps
# >= sep. Returns the placed points (a valid assignment / seed) or None.
# Correct for "pairwise |x_i - x_j| >= sep with x_i in interval_i" (sorted ->
# only consecutive gaps bind; earliest-placement is optimal).
# --------------------------------------------------------------------------
def greedy_place(intervals, sep):
    order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
    pts = [0.0] * len(intervals)
    prev = None
    for i in order:
        lo, hi = intervals[i]
        x = lo if prev is None else max(lo, prev + sep)
        if x > hi + 1e-9:
            return None
        pts[i] = x
        prev = x
    return pts


class Enumerator:
    def __init__(
        self, atlas, probe_names, ml_margin_deg: float = 0.0, ml_mode: str = "greedy"
    ):
        from aind_low_point.optimization.arc_first_principled import (
            _build_atlas_arrays,
        )

        self.names = probe_names
        self.K = len(probe_names)
        # SLSQP can push ml beyond the atlas-sampled anchors (threading is
        # soft), so the greedy ML-pack uses ml-windows widened by this margin
        # to avoid false-rejecting tuples that polish to feasible.
        self.ml_margin = ml_margin_deg
        # "greedy" = joint interval-packing (sound for K probes); "pairwise" =
        # production's max-possible-pairwise-diff (necessary but unsound for
        # 3+ probe arcs). Comparing counts isolates the joint-ML prune.
        self.ml_mode = ml_mode
        self.arr = _build_atlas_arrays(atlas, probe_names)

        # Nodes = feasible (probe_idx, hole_id). Index them; record intervals.
        self.nodes = []  # list of (p, h, ap_lo, ap_hi)
        self.node_id = {}  # (p, h) -> node index
        self.domain0 = [0] * self.K  # per-probe hole bitmask
        for (p, h), (lo, hi) in self.arr.ap_min_max.items():
            nid = len(self.nodes)
            self.node_id[(p, h)] = nid
            self.nodes.append((p, h, lo, hi))
            self.domain0[p] |= 1 << h

        # Precompute the interval-graph adjacency as bitmasks over node ids:
        # overlap[N] = {M : AP envelopes of N and M intersect}. By 1-D Helly,
        # an arc (set of nodes, one per probe) is AP-feasible iff its members
        # are pairwise-overlapping == members ⊆ overlap[N] for each added N.
        n = len(self.nodes)
        self.overlap = [0] * n
        for a in range(n):
            _, _, la, ha_ = self.nodes[a]
            for b in range(n):
                if a == b:
                    continue
                _, _, lb, hb = self.nodes[b]
                if la <= hb and lb <= ha_:  # intervals intersect
                    self.overlap[a] |= 1 << b

        self.candidates = []
        self.capped = False

    def ml_window(self, p, h, lo, hi):
        """ml-range of (p,h) anchors restricted to AP in [lo,hi] (matches the
        production ml_ranges computation; falls back to full range if empty)."""
        ap = self.arr.ap_sorted[(p, h)]
        ml = self.arr.ml_sorted[(p, h)]
        mask = (ap >= lo) & (ap <= hi)
        sel = ml[mask] if mask.any() else ml
        return float(sel.min()) - self.ml_margin, float(sel.max()) + self.ml_margin

    def _arc_ml_windows(self, arc):
        lo, hi = arc["lo"], arc["hi"]
        return [self.ml_window(p, arc["holes"][p], lo, hi) for p in arc["members"]]

    def _ml_pack(self, arc):
        """Greedy ML-pack the arc's members over the arc's running AP window.
        Returns placed ml points (parallel to arc['members']) or None."""
        return greedy_place(self._arc_ml_windows(arc), MIN_ML_SEP_DEG)

    def _ml_gate(self, arc) -> bool:
        """Feasibility gate per the selected ML mode."""
        ivals = self._arc_ml_windows(arc)
        if len(ivals) < 2:
            return True
        if self.ml_mode == "pairwise":
            # production's necessary condition: every pair CAN reach sep apart
            mp = min(
                max(abs(ivals[i][1] - ivals[j][0]), abs(ivals[j][1] - ivals[i][0]))
                for i in range(len(ivals))
                for j in range(i + 1, len(ivals))
            )
            return mp >= MIN_ML_SEP_DEG
        return greedy_place(ivals, MIN_ML_SEP_DEG) is not None

    def enumerate(self):
        # arcs: list of dicts {members:[p...], holes:{p:h}, mask:int, lo,hi}
        domain = list(self.domain0)
        used = 0
        self._search([], domain, used, set())
        return self.candidates

    def _search(self, arcs, domain, used, assigned):
        if self.capped:
            return
        if len(assigned) == self.K:
            self._emit(arcs)
            return
        # MRV: unassigned probe with the smallest hole-domain.
        p = min(
            (q for q in range(self.K) if q not in assigned),
            key=lambda q: bin(domain[q]).count("1"),
        )
        if domain[p] == 0:
            return  # forward-check failure (dead end)

        holes = [h for h in range(self.K.bit_length() + 16) if domain[p] >> h & 1]
        for h in holes:
            nid = self.node_id[(p, h)]
            nlo, nhi = self.nodes[nid][2], self.nodes[nid][3]
            # Try joining each existing arc (Helly clique test + ML pack).
            for ai, arc in enumerate(arcs):
                if len(arc["members"]) >= MAX_PROBES_PER_ARC:
                    continue
                if arc["mask"] & ~self.overlap[nid]:
                    continue  # not pairwise-overlapping -> AP-infeasible
                new_lo, new_hi = max(arc["lo"], nlo), min(arc["hi"], nhi)
                trial = {
                    "members": arc["members"] + [p],
                    "holes": {**arc["holes"], p: h},
                    "mask": arc["mask"] | (1 << nid),
                    "lo": new_lo,
                    "hi": new_hi,
                }
                if not self._ml_gate(trial):
                    continue  # ML packing infeasible (mode-dependent)
                new_arcs = arcs[:ai] + [trial] + arcs[ai + 1 :]
                self._recurse_assign(new_arcs, domain, used, assigned, p, h)
            # Try opening a new arc.
            if len(arcs) < MAX_ARCS:
                new_arc = {
                    "members": [p],
                    "holes": {p: h},
                    "mask": (1 << nid),
                    "lo": nlo,
                    "hi": nhi,
                }
                # inter-arc AP separation feasibility (greedy stab on intervals)
                ivals = [(a["lo"], a["hi"]) for a in arcs] + [(nlo, nhi)]
                if greedy_place(ivals, MIN_ARC_AP_SEP_DEG) is not None:
                    self._recurse_assign(arcs + [new_arc], domain, used, assigned, p, h)

    def _recurse_assign(self, arcs, domain, used, assigned, p, h):
        # Forward-check: remove hole h from every unassigned probe's domain.
        nd = list(domain)
        bit = 1 << h
        dead = False
        for q in range(self.K):
            if q == p or q in assigned:
                continue
            nd[q] &= ~bit
            if q != p and nd[q] == 0 and (q not in assigned):
                dead = True
        if dead:
            return
        self._search(arcs, nd, used | bit, assigned | {p})
        if len(self.candidates) >= GLOBAL_CAP:
            self.capped = True

    def _emit(self, arcs):
        # Final inter-arc AP separation + arc AP seeds (greedy stab).
        ivals = [(a["lo"], a["hi"]) for a in arcs]
        arc_pts = greedy_place(ivals, MIN_ARC_AP_SEP_DEG)
        if arc_pts is None:
            return
        probe_to_hole = {}
        partition = []
        ml_seed = {}
        for arc in arcs:
            partition.append(frozenset(self.names[p] for p in arc["members"]))
            mls = self._ml_pack(arc)
            if mls is None:  # pairwise mode admitted a non-greedy-packable arc
                mls = [0.5 * (lo + hi) for lo, hi in self._arc_ml_windows(arc)]
            for k, p in enumerate(arc["members"]):
                probe_to_hole[self.names[p]] = arc["holes"][p]
                ml_seed[self.names[p]] = mls[k]
        self.candidates.append(
            {
                "probe_to_hole": probe_to_hole,
                "partition": frozenset(partition),
                "arc_aps": [0.5 * (a["lo"] + a["hi"]) for a in arcs],
                "ml_seed": ml_seed,
            }
        )


# --------------------------------------------------------------------------
# Validation.
# --------------------------------------------------------------------------
def _partition_of(p2arc):
    groups = {}
    for name, ai in p2arc.items():
        groups.setdefault(ai, set()).add(name)
    return frozenset(frozenset(g) for g in groups.values())


def validate(cands, feas, pool):
    hole_keys = {tuple(sorted(c["probe_to_hole"].items())) for c in cands}
    full_keys = {
        (tuple(sorted(c["probe_to_hole"].items())), c["partition"]) for c in cands
    }
    man = tuple(sorted(MANUAL_H.items()))
    miss_hole, miss_full = [], []
    for r in feas:
        c = pool[r["idx"]]
        hk = tuple(sorted(dict(c.ha.probe_to_hole).items()))
        part = _partition_of(c.aa.probe_to_arc_idx)
        if hk not in hole_keys:
            miss_hole.append(r["idx"])
        elif (hk, part) not in full_keys:
            miss_full.append(r["idx"])
    return man in hole_keys, miss_hole, miss_full


def main() -> int:
    atlas, probe_names = build_or_load_atlas()
    print(f"probes: {probe_names}")
    pool = pickle.load(open(POOL_PKL, "rb"))["candidates"]
    feas = [r for r in pickle.load(open(HANDOFF_PKL, "rb"))["all"] if r["fcl"] >= -0.2]
    n = len(feas)

    margins = [
        float(x) for x in _os.environ.get("ML_MARGINS", "0,0.5,1,2,3").split(",")
    ]
    print(f"\nml-margin sweep (keep ALL {n} FCL-feasibles is the bar):")
    print(
        f"{'margin':>7} {'cands':>8} {'sec':>6} {'manual':>7} "
        f"{'holes_kept':>11} {'full_kept':>10}  dropped"
    )
    for m in margins:
        enr = Enumerator(atlas, probe_names, ml_margin_deg=m)
        t0 = time.time()
        cands = enr.enumerate()
        dt = time.time() - t0
        manual_ok, miss_hole, miss_full = validate(cands, feas, pool)
        kh = n - len(miss_hole)
        kf = n - len(miss_hole) - len(miss_full)
        print(
            f"{m:>7.1f} {len(cands):>8} {dt:>6.1f} "
            f"{'YES' if manual_ok else 'NO':>7} {kh:>8}/{n} "
            f"{kf:>7}/{n}  {miss_hole or ''}"
        )

    # Isolate the joint-ML prune: greedy vs production's pairwise check, same
    # margin. Delta = candidates pairwise wrongly admits (can't actually pack).
    m = float(_os.environ.get("ML_MARGIN", "1.0"))
    g = len(Enumerator(atlas, probe_names, m, "greedy").enumerate())
    pw = len(Enumerator(atlas, probe_names, m, "pairwise").enumerate())
    print(
        f"\njoint-ML prune @ margin {m}°: greedy={g}  pairwise={pw}  "
        f"pruned={pw - g} ({100 * (pw - g) / pw:.1f}% of pairwise admits "
        f"can't joint-ML-pack)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
