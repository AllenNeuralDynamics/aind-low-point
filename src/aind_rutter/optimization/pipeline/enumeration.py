"""Prototype: hole-first MRV enumerator with bitset domains + 1-D Helly
clique AP-feasibility + shared greedy-stab (intra-arc ML packing AND inter-arc
AP separation).

Consumes the visibility atlas through the shared ``_build_atlas_arrays``
packing helper, so the per-(probe,hole) AP intervals and ML ranges used by
seed emission and enumeration are identical.

``_emit`` records only the CHEAP discrete decision (probe->hole, partition +
midpoint arc-AP placeholder). The joint AP/ML/spin seed is expensive (per-arc
MRV/CSP over atlas anchors via the shared ``emit_seed``) and most enumerated
candidates never get polished, so seeding is LAZY: call ``Enumerator.seed(cand)``
on the top-N you actually hand to the optimizer.

The ``Enumerator`` accepts optional search limits: ``max_arcs`` /
``max_probes_per_arc`` (default to the kinematic max) and ``ap_range`` /
``ml_range`` windows (clip each (probe,hole)'s feasible AP envelope and drop
anchors outside the AP/ML windows).
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
from collections.abc import Sequence
from pathlib import Path

from aind_rutter.optimization.enumeration.atlas import Atlas
from aind_rutter.optimization.enumeration.seed_emission import emit_seed
from aind_rutter.optimization.pipeline.contracts import (
    AtlasCachePayload,
    EnumeratorCandidate,
    SeedResult,
)
from aind_rutter.optimization.pipeline.payloads import (
    check_payload_path,
    read_atlas_cache,
    write_atlas_cache,
)
from aind_rutter.optimization.pipeline.settings import PipelineSettings
from aind_rutter.planning import AP_LIMIT_DEG, ML_LIMIT_DEG, PoseLimits

# Subject is config-driven (generalizes across subjects). The visibility atlas
# depends on the subject's targets + implant placement, so its cache is keyed off
# the config stem — different subjects NEVER share an atlas. Override ATLAS_CACHE
# to force a path.
_SHARED = PipelineSettings()  # one resolution of the subject for every stage
CONFIG = _SHARED.config
HOLES = _SHARED.holes
ATLAS_CACHE = _os.environ.get(
    "ATLAS_CACHE", f"scratch/atlas_{Path(CONFIG).stem}.json.gz"
)

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
def build_or_load_atlas() -> AtlasCachePayload:
    cache = check_payload_path(ATLAS_CACHE)
    if cache.exists():
        try:
            return read_atlas_cache(cache)
        except ValueError as e:
            print(f"atlas cache unusable, rebuilding: {e}", flush=True)
    from aind_rutter.optimization.enumeration.visibility_atlas import (
        build_visibility_atlas,
    )
    from aind_rutter.optimization.pipeline.runtime_adapter import (
        OptimizationRuntime,
    )

    opt = OptimizationRuntime.from_config_path(CONFIG, HOLES)
    t0 = time.time()
    atlas = build_visibility_atlas(
        opt.probes, opt.holes, n_top=128, n_spin=72, verbose=False
    )
    print(
        f"built visibility atlas in {time.time() - t0:.0f}s "
        f"({len(opt.probes)} probes, {len(opt.holes)} holes)"
    )
    payload = AtlasCachePayload(atlas, tuple(opt.probe_names), opt.head_pitch_deg)
    write_atlas_cache(cache, payload)
    return payload


# --------------------------------------------------------------------------
# Shared greedy stab: place one point per interval, sorted, consecutive gaps
# >= sep. Returns the placed points (a valid assignment / seed) or None.
# Correct for "pairwise |x_i - x_j| >= sep with x_i in interval_i" (sorted ->
# only consecutive gaps bind; earliest-placement is optimal).
# --------------------------------------------------------------------------
def greedy_place(
    intervals: Sequence[tuple[float, float]], sep: float
) -> list[float] | None:
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
        self,
        atlas: Atlas,
        probe_names: Sequence[str],
        ml_margin_deg: float = 0.0,
        ml_mode: str = "greedy",
        max_arcs: int = MAX_ARCS,
        max_probes_per_arc: int = MAX_PROBES_PER_ARC,
        ap_range: "tuple[float, float] | None" = None,
        ml_range: "tuple[float, float] | None" = None,
    ):
        from aind_rutter.optimization.enumeration.seed_emission import (
            _build_atlas_arrays,
        )

        self.names = tuple(probe_names)
        self.K = len(probe_names)
        # The optimizer can push ml beyond the atlas-sampled anchors (threading is
        # soft), so the greedy ML-pack uses ml-windows widened by this margin
        # to avoid false-rejecting tuples that polish to feasible.
        self.ml_margin = ml_margin_deg
        # "greedy" = joint interval-packing (sound for K probes); "pairwise" =
        # production's max-possible-pairwise-diff (necessary but unsound for
        # 3+ probe arcs). Comparing counts isolates the joint-ML prune.
        self.ml_mode = ml_mode
        # Arc / per-arc caps. Default to the KINEMATIC max (16° exclusion over
        # the AP/ML range); callers can tighten them to restrict the search
        # (e.g. reproduce the old 3-arc / 4-per-arc enumeration).
        self.max_arcs = max_arcs
        self.max_probes_per_arc = max_probes_per_arc
        self.arr = _build_atlas_arrays(atlas, list(probe_names))
        # Always clip to the kinematic rig limits so enumeration only generates
        # within-limit plans (ML is frame-invariant; AP is subject-frame and is
        # shifted by head pitch by the caller). Explicit ap_range/ml_range
        # (e.g. from env) override these defaults.
        if ml_range is None:
            ml_range = (-ML_LIMIT_DEG, ML_LIMIT_DEG)
        if ap_range is None:
            ap_range = (-AP_LIMIT_DEG, AP_LIMIT_DEG)
        self._restrict_atlas(ap_range, ml_range)

        # Nodes = feasible (probe_idx, hole_id). Index them; record intervals.
        self.nodes: list[tuple[int, int, float, float]] = []
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

        self.candidates: list[EnumeratorCandidate] = []
        self.capped = False

    def _restrict_atlas(self, ap_range, ml_range):
        """In-place AP/ML windowing of the atlas arrays (mutates ``self.arr``).

        Clips each (probe,hole) feasible AP envelope to ``ap_range`` and keeps
        only anchors whose AP and ML lie inside both windows. A (probe,hole)
        whose clipped envelope is empty, or whose anchors all fall outside the
        windows, is dropped from the node set entirely (so it never enters an
        arc and is never seeded). Both ranges are ``(lo, hi)`` in atlas degrees;
        ``None`` means unbounded on that axis.
        """
        arr = self.arr
        ap_lo, ap_hi = ap_range if ap_range is not None else (-1e18, 1e18)
        ml_lo, ml_hi = ml_range if ml_range is not None else (-1e18, 1e18)
        for key in list(arr.ap_min_max.keys()):
            lo, hi = arr.ap_min_max[key]
            clo, chi = max(lo, ap_lo), min(hi, ap_hi)  # clipped AP envelope
            ap, ml, sp = arr.ap_sorted[key], arr.ml_sorted[key], arr.spin_sorted[key]
            m = (ap >= ap_lo) & (ap <= ap_hi) & (ml >= ml_lo) & (ml <= ml_hi)
            if clo > chi or not m.any():
                for d in (
                    arr.ap_sorted,
                    arr.ml_sorted,
                    arr.spin_sorted,
                    arr.ap_min_max,
                ):
                    del d[key]
                continue
            arr.ap_sorted[key] = ap[m]  # boolean mask preserves AP-sorted order
            arr.ml_sorted[key] = ml[m]
            arr.spin_sorted[key] = sp[m]
            arr.ap_min_max[key] = (clo, chi)

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

    def enumerate(self) -> list[EnumeratorCandidate]:
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
                if len(arc["members"]) >= self.max_probes_per_arc:
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
            if len(arcs) < self.max_arcs:
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
        # Record only the CHEAP discrete decision (probe->hole, partition) plus
        # a midpoint arc-AP placeholder. The joint AP/ML/spin seed is expensive
        # (per-arc MRV/CSP over atlas anchors) and most enumerated candidates
        # never get polished, so seeding is LAZY: call ``self.seed(candidate)``
        # on the top-N you actually hand to the optimizer.
        ivals = [(a["lo"], a["hi"]) for a in arcs]
        if greedy_place(ivals, MIN_ARC_AP_SEP_DEG) is None:
            return  # arc APs cannot be >=16 deg separated -> reject
        probe_to_hole = {}
        partition = []
        for arc in arcs:
            partition.append(frozenset(self.names[p] for p in arc["members"]))
            for p in arc["members"]:
                probe_to_hole[self.names[p]] = arc["holes"][p]
        self.candidates.append(
            {
                "probe_to_hole": probe_to_hole,
                "partition": frozenset(partition),
                "arc_aps": [0.5 * (a["lo"] + a["hi"]) for a in arcs],
            }
        )

    def seed(self, cand: EnumeratorCandidate) -> SeedResult | None:
        """Lazily compute the joint AP/ML/spin seed for one enumerated candidate.

        Rebuilds the per-arc ``emit_seed`` input from the candidate's discrete
        ``probe_to_hole`` / ``partition`` (each arc's AP window = its members'
        atlas AP-envelope intersection, desired AP = window midpoint) and runs
        the shared source-of-truth :func:`emit_seed` (convex isotonic arc-AP +
        MRV/CSP ML-anchor pick + spin).

        Returns ``(arc_aps, ml_seed, spin_seed, min_ml_gap)`` (ml/spin keyed by
        probe name; ``min_ml_gap < 16`` flags an atlas-limited best-effort seed),
        or ``None`` if some probe has no atlas anchors at all.
        """
        p2h = cand["probe_to_hole"]
        seed_arcs = []
        # Canonical (sorted) arc + member order so the seed is independent of the
        # partition frozensets' hash-seed-dependent iteration order. Otherwise
        # emit_seed's MRV anchor pick varies per process — breaking reproducibility
        # across runs and the parallel seed pool. ``wrap`` (phase1_pool) must use
        # the same arc order so probe_to_arc_idx aligns with the returned arc_aps.
        for group in sorted(cand["partition"], key=sorted):
            members = [
                (self.names.index(name), p2h[name], name) for name in sorted(group)
            ]
            lo = max(self.arr.ap_min_max[(p, h)][0] for p, h, _ in members)
            hi = min(self.arr.ap_min_max[(p, h)][1] for p, h, _ in members)
            seed_arcs.append(
                {
                    "members": members,
                    "ap_lo": lo,
                    "ap_hi": hi,
                    "ap_desired": 0.5 * (lo + hi),
                }
            )
        seed = emit_seed(
            seed_arcs,
            self.arr,
            min_arc_ap_sep_deg=MIN_ARC_AP_SEP_DEG,
            min_ml_sep_deg=MIN_ML_SEP_DEG,
        )
        return None if seed is None else SeedResult(*seed)
