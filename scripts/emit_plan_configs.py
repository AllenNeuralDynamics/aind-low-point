"""Emit trame-app config YAMLs from a Phase-2 handoff (the overnight pipeline
output, ``phase2_parallel``). For each of the top-N MMR-ranked *feasible* plans,
rebuild the plan_state from the saved pose + arc assignment and round-trip it
through ``save_plan_to_config`` — one ready-to-open config per candidate.

Config-driven (works on any subject): CONFIG/HOLES select the subject; HANDOFF is
that subject's Phase-2 output. Each handoff record already carries everything
needed (pose, probe_to_hole, probe_to_arc_idx, arc_centroids_deg, n_arcs), so no
re-optimization happens here — pure reconstruction.

Run:
  CONFIG=examples/837229-config.yml HOLES=scratch/0283-300-04.holes.yml \\
  HANDOFF=scratch/837229_phase2_handoff.pkl N=15 OUTDIR=examples/837229_plans \\
  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.emit_plan_configs
"""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.headstages import make_fcl_bvh
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.joint_rerank import _build_probe_static
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.runtime import (
    build_runtime_from_config,
    planning_state_to_plan_model,
)
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import (
    _probe_static_info,
    _transform_holes,
    retro_opts_from_env,
)
from scripts.save_chain_plans import _apply_x_to_plan_state

CONFIG = os.environ.get("CONFIG", "examples/836656-config-T12.yml")
HOLES = os.environ.get("HOLES", "scratch/0283-300-04.holes.yml")
HANDOFF = os.environ.get("HANDOFF", "scratch/phase2_handoff.pkl")
N = int(os.environ.get("N", "15"))
OUTDIR = os.environ.get("OUTDIR", "scratch/plans")


def _hole_path(hole, probe_order):
    """Filename-style hole encoding, e.g. bla4_ca16_cla8_md12_pl1_rsp5_vm3."""
    return "_".join(f"{pr.lower()}{hole[pr]}" for pr in probe_order)


def _leaf(p):
    return f"cand {p['idx']}  cov {p['coverage']:.2f}  fcl {p['fcl']:+.3f}"


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
        f"{len(rows)} plans with FCL ≥ {fcl_desc} mm, sorted by coverage. "
        "`path` is the hole-assignment encoding (also the plan filename).",
        "",
        "| # | cand | n_arcs | cov | fcl | path | " + " | ".join(probe_order) + " |",
        "|" + "---|" * (6 + len(probe_order)),
    ]
    for i, m in enumerate(rows, 1):
        cells = " | ".join(str(m["hole"][pr]) for pr in probe_order)
        lines.append(
            f"| {i} | {m['idx']} | {m['n_arcs']} | {m['coverage']:.3f} | "
            f"{m['fcl']:+.3f} | {_hole_path(m['hole'], probe_order)} | {cells} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> int:
    cfg = ConfigModel.from_yaml(CONFIG)
    rt = build_runtime_from_config(cfg)
    _ro = retro_opts_from_env(rt)
    probes = [
        _probe_static_info(rt.plan_state, rt, n, _ro) for n in rt.plan_state.probes
    ]
    holes = load_holes(Path(HOLES))
    comp = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in comp:
        R, t = comp["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    sdf = {
        p.name: build_probe_sdf_from_alpha_wrap(
            rt.asset_catalog.get_geometry(f"probe:{p.kind}").raw
        )
        for p in probes
    }
    bvh = {
        p.name: make_fcl_bvh(p.collision_mesh) if p.collision_mesh else None
        for p in probes
    }

    H = pickle.load(open(HANDOFF, "rb"))
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
        ha = SimpleNamespace(probe_to_hole=r["hole"])
        aa = SimpleNamespace(
            probe_to_arc_idx=r["probe_to_arc_idx"],
            arc_centroids_deg=list(r["arc_centroids_deg"]),
        )
        statics = _build_probe_static(
            probes, holes, ha, aa, bvh_cache=bvh, sdf_by_name=sdf
        )
        # Deep-copy the base plan_state per plan (mutations must not bleed across
        # plans); the runtime + meshes are built ONCE up front, not rebuilt per
        # plan — rebuilding reloaded every OBJ from disk, ~2-3s/plan.
        plan_state = copy.deepcopy(rt.plan_state)
        _apply_x_to_plan_state(
            plan_state, np.asarray(r["pose"], float), statics, n_arcs
        )
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
