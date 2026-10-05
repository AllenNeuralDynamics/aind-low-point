"""Emit trame-app plan YAMLs from a Phase-2 handoff.

For each MMR-ranked *feasible* plan (all of them unless ``RUTTER_PLANS`` caps it),
rebuild the plan_state from the saved pose + arc assignment and round-trip it
through ``save_plan_to_config`` — one ready-to-open config per candidate.

Config-driven (works on any subject): ``RUTTER_CONFIG``/``RUTTER_HOLES`` select
the subject and ``RUTTER_HANDOFF`` is that subject's Phase-2 output. Each handoff
record already carries everything
needed (pose, probe_to_hole, probe_to_arc_idx, arc_centroids_deg, n_arcs), so no
re-optimization happens here — pure reconstruction. Each plan's manifest entry
also carries the share of every region its active sites reach; see
:mod:`measure`.

Run:
  RUTTER_CONFIG=examples/837229-config.yml \\
  RUTTER_HOLES=scratch/0283-300-04.holes.yml \\
  RUTTER_HANDOFF=scratch/837229_phase2_handoff.json \\
  RUTTER_OUTDIR=examples/837229_plans \\
  JAX_PLATFORMS=cpu uv run --python 3.13 rutter-emit
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from aind_rutter.build.assemble import (
    build_plan_state_from_config,
    build_runtime_from_config,
)
from aind_rutter.config import ConfigModel
from aind_rutter.optimization.objectives.variables import _apply_x_to_plan_state
from aind_rutter.optimization.pipeline.measure import (
    active_segments,
    recorded_mm3,
    recorded_share,
    region_clouds,
    region_names,
    shank_geometry,
    voxel_mm3,
)
from aind_rutter.optimization.pipeline.payloads import read_handoff
from aind_rutter.optimization.pipeline.probe_setup import (
    RetroDensityOpts,
    retro_anchor_LPS,
    retro_opts_from_env,
)
from aind_rutter.optimization.pipeline.settings import EmitSettings
from aind_rutter.plan_io.plan_set import PlanEntry, RecordedMeasure, write_manifest
from aind_rutter.plan_io.rig_export import reorder_plan_for_rig
from aind_rutter.plan_io.roundtrip import planning_state_to_plan_model


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


def run(settings: EmitSettings) -> int:
    """Write the top ranked plans of a handoff as plan-only YAML."""
    cfg = ConfigModel.from_yaml(settings.config, settings.locator())
    # Writing a plan is pure-symbolic: plan parameters (arc/ml/spin/offsets/
    # depth) go from the handoff x-vector into a PlanningModel, from the base
    # PlanningState (built mesh-free from the config's plan section) plus, per
    # probe, its name + arc_idx (both already in the handoff record). Only the
    # recorded measure places the probes, and so loads the meshes.
    base_plan_state = build_plan_state_from_config(cfg)
    probe_names = list(base_plan_state.probes)
    measure, clouds, shanks = None, {}, {}
    runtime = (
        build_runtime_from_config(cfg)
        if settings.recorded_radius_mm is not None or retro_opts_from_env() is not None
        else None
    )
    retro_opts = None if runtime is None else retro_opts_from_env(runtime)
    anchors = {}
    if runtime is not None and retro_opts is not None:
        # The solver aimed each probe here, and the pose's offsets and depth are
        # relative to it; against the config's target the probes land elsewhere.
        anchors = {nm: retro_anchor_LPS(runtime, nm, retro_opts) for nm in probe_names}
    voxels: dict = {}
    voxel_size: dict[str, float] = {}
    if settings.recorded_radius_mm is not None and runtime is not None:
        clouds = region_clouds(cfg, runtime, retro_opts)
        voxels = clouds if retro_opts is None else region_clouds(cfg, runtime, None)
        mask_fmt = (retro_opts or RetroDensityOpts()).region_mask_fmt
        voxel_size = {r: voxel_mm3(runtime, mask_fmt.format(region=r)) for r in voxels}
        shanks = shank_geometry(runtime)
        if clouds:
            measure = RecordedMeasure(
                "voxels" if retro_opts is None else "retro",
                settings.recorded_radius_mm,
            )
            print(
                f"recorded share within {measure.radius_mm:g} mm of {measure.cloud}: "
                + ", ".join(f"{r} {len(c)}" for r, c in clouds.items())
            )
        else:
            print("no region has a structure asset; plans carry no recorded share")
        unscored = [r for r in region_names(runtime.plan_state) if r not in clouds]
        if clouds and unscored:
            # Their absence lowers nothing but hides them from the mean.
            print(f"  not scored, no structure asset: {', '.join(unscored)}")

    H = read_handoff(settings.handoff)
    ranked = H.get("ranked", [])  # MMR-ranked feasible plans
    fcl_tol = H.get("config", {}).get("fcl_tol", 0.2)
    fcl_desc = f"-{fcl_tol:g}"
    probe_order = sorted(ranked[0]["hole"].keys()) if ranked else []
    out = Path(settings.outdir)
    plans_dir = out / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    n_emit = len(ranked) if settings.plans is None else min(settings.plans, len(ranked))
    print(f"{len(ranked)} feasible plans in handoff; emitting {n_emit} + tree/manifest")

    meta = []
    entries: list[PlanEntry] = []
    for i, r in enumerate(ranked[:n_emit]):
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
        for nm, anchor in anchors.items():
            probe = plan_state.probes[nm]
            probe.target_key = None
            x, y, z = (float(v) for v in anchor)
            probe.target_point_LPS = (x, y, z)
        # Rig-readability ordering: arcs relabelled a=most-+AP, probes sorted by
        # arc then ML-descending. Cosmetic; pose semantics unchanged.
        plan_state = reorder_plan_for_rig(plan_state)
        recorded: dict[str, float] = {}
        reached_mm3: dict[str, float] = {}
        if measure is not None and runtime is not None:
            placed = copy.copy(plan_state)
            placed.target_index = runtime.plan_state.target_index
            segments = active_segments(placed, runtime, shanks)
            recorded = {
                region: recorded_share(cloud, segments, measure.radius_mm)
                for region, cloud in clouds.items()
            }
            reached_mm3 = {
                region: recorded_mm3(
                    points, segments, measure.radius_mm, voxel_size[region]
                )
                for region, points in voxels.items()
            }
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
        entries.append(
            PlanEntry(
                file=fname,
                rank=i + 1,
                coverage=float(r["coverage"]),
                fcl_mm=float(r["fcl"]),
                n_arcs=n_arcs,
                candidate=int(r.get("idx", i)),
                holes={k: int(v) for k, v in r["hole"].items()},
                arcs={nm: p.arc_id for nm, p in plan_state.probes.items()},
                recorded=recorded,
                recorded_mm3=reached_mm3,
            )
        )
        print(
            f"  [{i + 1:>3}] {n_arcs}arc cov={r['coverage']:.2f} "
            f"fcl={r['fcl']:+.3f} → {fname}"
        )

    _emit_tree(meta, probe_order, out / "tree.txt", fcl_desc)
    _emit_manifest(meta, probe_order, out / "manifest.md", fcl_desc)
    write_manifest(
        out,
        entries,
        probes=probe_order,
        config=str(settings.config),
        measure=measure,
        region_mm3={r: len(v) * voxel_size[r] for r, v in voxels.items()},
    )
    print(f"\nwrote {n_emit} plans → {plans_dir}/")
    print(f"wrote {out}/tree.txt + {out}/manifest.md + {out}/manifest.json")
    return 0


def main() -> int:
    """Console entry point: settings from the environment, then `run`."""
    return run(EmitSettings())


if __name__ == "__main__":
    raise SystemExit(main())
