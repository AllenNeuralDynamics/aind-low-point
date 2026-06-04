"""Static handoff export: turn the Phase-2 feasible set into reviewable
artifacts under ``scratch/handoff/``.

Tier 1 (always written, no kinematics needed):
  - ``tree.txt``     the decision tree over the feasible set (locked trunk →
                     flex forks MD/BLA/CA1/CLA/VM → leaf plans).
  - ``manifest.md``  one row per feasible plan: decision path, per-probe hole,
                     coverage, FCL margin — sortable/scannable for review.

Tier 2 (``--plans``): one ``plan-NN-*.yml`` per feasible, decoded from the
stored Phase-2 pose, plus a round-trip pose check (reload → resolve → compare
(R,t) to the decode) to PROVE the serialization preserved the geometry.

Run:  uv run --python 3.13 -m scripts.export_handoff [--plans]
Env:  HANDOFF_PKL (default scratch/phase2_handoff.pkl), FCL_TOL (0.2)
"""

from __future__ import annotations

import os as _os
import pickle
from collections import Counter
from pathlib import Path

HANDOFF_PKL = _os.environ.get("HANDOFF_PKL", "scratch/phase2_handoff.pkl")
FCL_TOL = float(_os.environ.get("FCL_TOL", "0.2"))
OUT = Path("scratch/handoff")

# Split order for the tree: locked probes form the trunk, then the remaining
# probes split most-constrained-first so the tree stays narrow at the top and
# fans out at the leaves (matches how a reviewer would decide: big structural
# forks before fine ones).


def _feasibles(pkl_path: str) -> list[dict]:
    d = pickle.load(open(pkl_path, "rb"))
    rows = [r for r in d["all"] if r["fcl"] >= -FCL_TOL]
    rows.sort(key=lambda r: -r["coverage"])
    return rows


def _probes(rows: list[dict]) -> list[str]:
    return sorted(rows[0]["hole"].keys())


def _split_probe(rows, fixed, probes):
    """Un-fixed probe with the fewest distinct holes in this subset (>1)."""
    best = None
    for p in probes:
        if p in fixed:
            continue
        k = len(set(r["hole"][p] for r in rows))
        if k > 1 and (best is None or k < best[1]):
            best = (p, k)
    return best[0] if best else None


def _tree_lines(rows, fixed, probes, depth=0) -> list[str]:
    out: list[str] = []
    p = _split_probe(rows, fixed, probes)
    if p is None:
        for r in sorted(rows, key=lambda r: -r["coverage"]):
            out.append(
                "  " * depth + f"* cand {r['idx']:>4}  "
                f"cov {r['coverage']:.2f}  fcl {r['fcl']:+.3f}"
            )
        return out
    groups = Counter(r["hole"][p] for r in rows)
    for hole, _ in sorted(groups.items(), key=lambda kv: -kv[1]):
        sub = [r for r in rows if r["hole"][p] == hole]
        cs = [r["coverage"] for r in sub]
        out.append(
            "  " * depth + f"{p}=hole{hole}  "
            f"({len(sub)} plans, cov {min(cs):.1f}-{max(cs):.1f})"
        )
        out.extend(_tree_lines(sub, fixed | {p}, probes, depth + 1))
    return out


def _path_label(row, probes, locked) -> str:
    """Decision path through the flex probes (skips the locked trunk)."""
    return "_".join(f"{p.lower()}{row['hole'][p]}" for p in probes if p not in locked)


def write_tier1(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probes = _probes(rows)
    locked = {p for p in probes if len(set(r["hole"][p] for r in rows)) == 1}

    # tree.txt
    trunk = ", ".join(f"{p}->h{rows[0]['hole'][p]}" for p in sorted(locked))
    lines = [
        f"Phase-2 feasible decision tree  (FCL >= -{FCL_TOL})",
        f"{len(rows)} feasible plans, {len(rows)} distinct hole-assignments",
        f"TRUNK (locked across all feasibles): {trunk or '(none)'}",
        "",
    ]
    lines += _tree_lines(rows, locked, probes)
    (OUT / "tree.txt").write_text("\n".join(lines) + "\n")

    # manifest.md
    hdr = ["#", "cand", "coverage", "fcl", "path"] + probes
    md = [
        "# Phase-2 feasible handoff manifest",
        "",
        f"{len(rows)} plans with FCL >= -{FCL_TOL}, sorted by coverage. "
        f"`path` is the decision route through the flex probes "
        f"(trunk {trunk or 'none'} omitted).",
        "",
        "| " + " | ".join(hdr) + " |",
        "|" + "|".join(["---"] * len(hdr)) + "|",
    ]
    for i, r in enumerate(rows, 1):
        cells = [
            str(i),
            str(r["idx"]),
            f"{r['coverage']:.3f}",
            f"{r['fcl']:+.3f}",
            _path_label(r, probes, locked),
        ]
        cells += [str(r["hole"][p]) for p in probes]
        md.append("| " + " | ".join(cells) + " |")
    (OUT / "manifest.md").write_text("\n".join(md) + "\n")

    print(
        f"wrote {OUT / 'tree.txt'} and {OUT / 'manifest.md'} "
        f"({len(rows)} feasibles, {len(locked)} locked probes)"
    )


def main() -> int:
    import sys

    rows = _feasibles(HANDOFF_PKL)
    write_tier1(rows)
    if "--plans" in sys.argv:
        from scripts.export_handoff_plans import write_plans

        write_plans(rows, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
