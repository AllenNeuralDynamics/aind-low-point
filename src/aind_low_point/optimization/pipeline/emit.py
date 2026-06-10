"""Emit trame-app plan YAMLs from a Phase-2 handoff.

For each of the top-N MMR-ranked *feasible* plans,
rebuild the plan_state from the saved pose + arc assignment and round-trip it
through ``save_plan_to_config`` — one ready-to-open config per candidate.

Config-driven (works on any subject): CONFIG/HOLES select the subject; HANDOFF is
that subject's Phase-2 output. Each handoff record already carries everything
needed (pose, probe_to_hole, probe_to_arc_idx, arc_centroids_deg, n_arcs), so no
re-optimization happens here — pure reconstruction.

Run:
  CONFIG=examples/837229-config.yml HOLES=scratch/0283-300-04.holes.yml \\
  HANDOFF=scratch/837229_phase2_handoff.pkl N=15 OUTDIR=examples/837229_plans \\
  JAX_PLATFORMS=cpu uv run --python 3.13 alp-emit
"""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import yaml

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.optimizer_vars import _apply_x_to_plan_state
from aind_low_point.optimization.pipeline.contracts import Phase2HandoffPayload
from aind_low_point.runtime import (
    build_plan_state_from_config,
    planning_state_to_plan_model,
)
from aind_low_point.runtime.export import reorder_plan_for_rig

CONFIG = os.environ.get("CONFIG", "examples/836656-config-T12.yml")
HOLES = os.environ.get("HOLES", "scratch/0283-300-04.holes.yml")
HANDOFF = os.environ.get("HANDOFF", "scratch/phase2_handoff.pkl")
N = int(os.environ.get("N", "15"))
OUTDIR = os.environ.get("OUTDIR", "scratch/plans")


def _hole_path(hole, probe_order):
    """Filename-style hole encoding, e.g. bla4_ca16_cla8_md12_pl1_rsp5_vm3."""
    return "_".join(f"{pr.lower()}{hole[pr]}" for pr in probe_order)


def _leaf(p):
    return (
        f"plan-{p['rank']:02d}  cand {p['idx']}  "
        f"cov {p['coverage']:.2f}  fcl {p['fcl']:+.3f}"
    )


def _tree_nodes(plans, remaining):
    """Recursive MRV tree → nested ``(label, children|None)`` nodes. At each node,
    branch on the most-constrained FLEX probe (fewest distinct holes among the
    plans here), recomputed per subtree; locked probes are skipped; a lone leaf is
    inlined onto its branch."""
    if len(plans) == 1:
        return [(_leaf(plans[0]), None)]
    flex = [pr for pr in remaining if len({p["hole"][pr] for p in plans}) > 1]
    if not flex:  # all identical in the remaining probes (duplicate assignments)
        return [(_leaf(p), None) for p in sorted(plans, key=lambda x: -x["coverage"])]
    probe = min(flex, key=lambda pr: (len({p["hole"][pr] for p in plans}), pr))
    rest = [pr for pr in remaining if pr != probe]
    groups: dict = {}
    for p in plans:
        groups.setdefault(p["hole"][probe], []).append(p)
    nodes = []
    for hole, grp in sorted(
        groups.items(), key=lambda kv: -max(x["coverage"] for x in kv[1])
    ):
        lo, hi = min(g["coverage"] for g in grp), max(g["coverage"] for g in grp)
        label = f"{probe}=h{hole}  ({len(grp)}, {lo:.1f}–{hi:.1f})"
        kids = _tree_nodes(grp, rest)
        if len(kids) == 1 and kids[0][1] is None:  # inline a lone leaf
            nodes.append((f"{label}  →  {kids[0][0]}", None))
        else:
            nodes.append((label, kids))
    return nodes


def _render(nodes, lines, prefix=""):
    """Pretty box-drawing render of the nested node list."""
    for i, (label, kids) in enumerate(nodes):
        last = i == len(nodes) - 1
        lines.append(prefix + ("└─ " if last else "├─ ") + label)
        if kids:
            _render(kids, lines, prefix + ("   " if last else "│  "))


def _emit_tree(meta, probe_order, path, fcl_desc):
    ndist = len({_hole_path(m["hole"], probe_order) for m in meta})
    lines = [
        f"Phase-2 feasible decision tree  (FCL ≥ {fcl_desc} mm)",
        f"{len(meta)} feasible plans · {ndist} distinct hole-assignments",
    ]
    trunk = [pr for pr in probe_order if len({m["hole"][pr] for m in meta}) == 1]
    tstr = "  ".join(f"{pr}=h{meta[0]['hole'][pr]}" for pr in trunk) or "none"
    lines.append(f"trunk (locked across all feasibles): {tstr}")
    free = [pr for pr in probe_order if pr not in trunk]
    for na in sorted({m["n_arcs"] for m in meta}):
        grp = [m for m in meta if m["n_arcs"] == na]
        lo, hi = min(g["coverage"] for g in grp), max(g["coverage"] for g in grp)
        lines += ["", f"{na}-arc  ({len(grp)} plans, cov {lo:.1f}–{hi:.1f})"]
        _render(_tree_nodes(grp, free), lines, "")
    Path(path).write_text("\n".join(lines) + "\n")


def _emit_manifest(meta, probe_order, path, fcl_desc):
    rows = sorted(meta, key=lambda m: -m["coverage"])
    lines = [
        "# Phase-2 feasible handoff manifest",
        "",
        f"{len(rows)} plans with FCL ≥ {fcl_desc} mm, sorted by coverage. `plan` is "
        "the MMR-ranked plan-file number (plan-NN); `path` is the hole encoding.",
        "",
        "| cov# | plan | cand | n_arcs | cov | fcl | path | "
        + " | ".join(probe_order)
        + " |",
        "|" + "---|" * (7 + len(probe_order)),
    ]
    for i, m in enumerate(rows, 1):
        cells = " | ".join(str(m["hole"][pr]) for pr in probe_order)
        lines.append(
            f"| {i} | plan-{m['rank']:02d} | {m['idx']} | {m['n_arcs']} | "
            f"{m['coverage']:.3f} | {m['fcl']:+.3f} | "
            f"{_hole_path(m['hole'], probe_order)} | {cells} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> int:
    cfg = ConfigModel.from_yaml(CONFIG)
    # Emission is pure-symbolic: it only writes plan parameters (arc/ml/spin/
    # offsets/depth) from the handoff x-vector into a PlanningModel. It needs NO
    # meshes/SDFs/BVHs — only the base PlanningState (built mesh-free from the
    # config's plan section) plus, per probe, its name + arc_idx (both already in
    # the handoff record). The actual 3D placement happens when the plan is
    # loaded later.
    base_plan_state = build_plan_state_from_config(cfg)
    probe_names = list(base_plan_state.probes)

    H = cast(Phase2HandoffPayload, pickle.load(open(HANDOFF, "rb")))
    ranked = H.get("ranked", [])  # MMR-ranked feasible plans
    fcl_tol = H.get("config", {}).get("fcl_tol", 0.2)
    fcl_desc = f"-{fcl_tol:g}"
    probe_order = sorted(ranked[0]["hole"].keys()) if ranked else []
    out = Path(OUTDIR)
    plans_dir = out / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    n_emit = min(N, len(ranked))
    print(f"{len(ranked)} feasible plans in handoff; emitting {n_emit} + tree/manifest")

    meta = []
    for i, r in enumerate(ranked[:N]):
        n_arcs = r["n_arcs"]
        # Lightweight statics: _apply_x_to_plan_state reads only .name and
        # .arc_idx (the pose values come from the x-vector), both in the handoff.
        # Order matches the optimizer: config probe order == x-block order.
        statics = [
            SimpleNamespace(name=nm, arc_idx=r["probe_to_arc_idx"][nm])
            for nm in probe_names
        ]
        # Deep-copy the mesh-free base plan_state per plan (mutations must not
        # bleed across plans).
        plan_state = copy.deepcopy(base_plan_state)
        _apply_x_to_plan_state(
            plan_state, np.asarray(r["pose"], float), statics, n_arcs
        )
        # Rig-readability ordering: arcs relabelled a=most-+AP, probes sorted by
        # arc then ML-descending. Cosmetic; pose semantics unchanged.
        reorder_plan_for_rig(plan_state)
        # PLAN-ONLY file (just the plan section) so it pairs with the base config
        # via ``--plan``. Filename encodes the hole assignment (matches the tree).
        plan_model = planning_state_to_plan_model(plan_state, cfg.plan)
        hp = _hole_path(r["hole"], probe_order)
        fname = f"plan-{i + 1:02d}-cov{r['coverage']:05.2f}-{hp}.plan.yml"
        with open(plans_dir / fname, "w") as f:
            yaml.safe_dump(
                plan_model.model_dump(mode="json"),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        meta.append(
            dict(
                rank=i + 1,  # MMR/plan-file number: this record is plan-{rank}
                idx=r.get("idx", i),
                n_arcs=n_arcs,
                hole=dict(r["hole"]),
                coverage=float(r["coverage"]),
                fcl=float(r["fcl"]),
                file=fname,
            )
        )
        print(
            f"  [{i + 1:>3}] {n_arcs}arc cov={r['coverage']:.2f} "
            f"fcl={r['fcl']:+.3f} → {fname}"
        )

    _emit_tree(meta, probe_order, out / "tree.txt", fcl_desc)
    _emit_manifest(meta, probe_order, out / "manifest.md", fcl_desc)
    print(f"\nwrote {n_emit} plans → {plans_dir}/")
    print(f"wrote {out}/tree.txt + {out}/manifest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
