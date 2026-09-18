#!/usr/bin/env python3
"""Check that a Phase-2 change leaves every solved pose bitwise unchanged.

A refactor of the optimizer is only safe if it is invisible in the output, and
the output is a handoff of float poses that no unit test pins down. This runs the
stage twice over identical inputs — once from a baseline commit, checked out into
a git worktree and reached through ``PYTHONPATH``, once from the working tree —
and compares the two handoffs field by field.

    scripts/parity_phase2.py --baseline HEAD~1

The default subject is written on the spot, so the check needs no data and runs
anywhere; point ``--config``, ``--holes`` and ``--poses`` at a real subject to
compare on production geometry instead.

Both arms read the same subject, the same pool and the same caches: the baseline
worktree supplies code only. An arm that resolves ``aind_rutter`` outside its
intended tree aborts rather than reporting a meaningless parity.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCALARS = ("idx", "fcl", "max_g_thread", "coverage", "nit", "rank")
FLAGS = (
    "kept",
    "strict_feasible",
    "fcl_keep",
    "thread_keep",
    "fcl_strict",
    "thread_strict",
)

# Each arm writes its own handoff, so its recorded destination must differ.
IGNORED_CONFIG_FIELDS = {"settings.out"}

# One arm per run; the tag names the log, the handoff and the tree it imports.
ARM_BASELINE = "baseline"
ARM_CURRENT = "current"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def write_synthetic_inputs(work: Path, n_candidates: int) -> tuple[Path, Path, Path]:
    """Write a subject and a Phase-1 pool; return (config, holes, pool)."""
    sys.path.insert(0, str(REPO))
    from aind_rutter.optimization.pipeline.payloads import write_pool
    from tests.synthetic_subject import write_subject

    subject = write_subject(work / "subject")
    probes = ("P1", "P2")
    records = []
    for i in range(n_candidates):
        # Spread the seeds over the arc's AP range and the probes' ML, so the
        # candidates differ and the ranking is not a tie.
        arc_ap = -8.0 + 4.0 * i
        x = [arc_ap]
        for j, _ in enumerate(probes):
            x += [1.5 * i * (-1) ** j, 1.0, 0.0, 0.0, 0.0, 0.0]
        records.append(
            {
                "n_arcs": 1,
                "probe_to_hole": {"P1": 1, "P2": 2},
                "partition": frozenset(frozenset({name}) for name in probes),
                "probe_to_arc_idx": dict.fromkeys(probes, 0),
                "arc_centroids_deg": [arc_ap],
                "min_ml_gap": 10.0,
                "x": np.asarray(x, dtype=np.float32),
                "x_reduced": np.zeros(len(x), dtype=np.float32),
                "objective": float(i),
                "min_clear": 0.1 - 0.01 * i,
                "min_clear_reduced": 0.0,
                "fcl": float("nan"),
                "idx": i,
            }
        )
    pool = work / "pool.json.gz"
    write_pool(
        pool,
        {
            "records": records,
            "stage1": 0,
            "stage2": 0,
            "n_spins": 0,
            "max_arcs": 1,
            "max_ppa": len(probes),
            "minimizer": "synthetic",
            "well": "thin",
            "coarse_n": 0,
            "reduced_fine": 0,
            "full_fine": 0,
        },
    )
    return subject.config, subject.holes, pool


# ---------------------------------------------------------------------------
# Running one arm
# ---------------------------------------------------------------------------


def run_arm(tag: str, tree: Path, env: dict[str, str], work: Path) -> Path:
    """Run the stage with ``tree/src`` first on the path; return the handoff."""
    out = work / f"handoff_{tag}.json"
    log = work / f"{tag}.log"
    arm_env = {
        **env,
        "OUT": str(out),
        "PYTHONPATH": os.pathsep.join(
            [str(tree / "src"), *filter(None, [env.get("PYTHONPATH")])]
        ),
    }
    code = (
        "import pathlib, sys, aind_rutter;"
        f"expected = pathlib.Path({str(tree / 'src')!r}).resolve();"
        "actual = pathlib.Path(aind_rutter.__file__).resolve();"
        "print('aind_rutter from', actual, flush=True);"
        "sys.exit(f'WRONG TREE: expected {expected}') "
        "if expected not in actual.parents else None;"
        "from aind_rutter.optimization.pipeline.phase2_ipopt import main;"
        "sys.exit(main())"
    )
    print(f"[{tag}] running → {log}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=arm_env,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        print(log.read_text()[-2000:])
        raise SystemExit(f"[{tag}] failed with exit code {result.returncode}")
    print(f"[{tag}] {log.read_text().splitlines()[0]}")
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return bool(a == b)


def compare_handoffs(base_path: Path, new_path: Path) -> int:
    """Print every difference between two handoffs; return how many there were."""
    from aind_rutter.optimization.pipeline.payloads import read_handoff

    base, new = read_handoff(base_path), read_handoff(new_path)
    differences = 0

    for section in ("all", "ranked"):
        if len(base[section]) != len(new[section]):
            print(f"{section}: {len(base[section])} vs {len(new[section])} records")
            differences += 1
    print(f"all: {len(base['all'])}  ranked: {len(base['ranked'])} (baseline)")

    by_rank_base = {r["rank"]: r for r in base["all"]}
    by_rank_new = {r["rank"]: r for r in new["all"]}
    if set(by_rank_base) != set(by_rank_new):
        print(f"rank sets differ: {sorted(set(by_rank_base) ^ set(by_rank_new))}")
        differences += 1

    shared = sorted(set(by_rank_base) & set(by_rank_new))
    identical_poses = 0
    for rank in shared:
        left, right = by_rank_base[rank], by_rank_new[rank]
        for field in (*SCALARS, *FLAGS):
            if not _same(left.get(field), right.get(field)):
                print(
                    f"rank {rank}: {field} {left.get(field)!r} vs {right.get(field)!r}"
                )
                differences += 1
        pose_base = np.asarray(left["pose"], dtype=float)
        pose_new = np.asarray(right["pose"], dtype=float)
        if pose_base.shape == pose_new.shape and np.array_equal(pose_base, pose_new):
            identical_poses += 1
        else:
            gap = (
                np.abs(pose_base - pose_new).max()
                if pose_base.shape == pose_new.shape
                else float("nan")
            )
            print(f"rank {rank}: pose differs, max|delta| {gap:.3e}")
            differences += 1
    print(f"poses bitwise identical: {identical_poses}/{len(shared)}")

    order_base = [r["idx"] for r in base["ranked"]]
    order_new = [r["idx"] for r in new["ranked"]]
    if order_base == order_new:
        print(f"ranked order identical: {order_base}")
    else:
        print(f"ranked order DIFFERS: {order_base} vs {order_new}")
        differences += 1

    return differences + _compare_config(base["config"], new["config"])


def _compare_config(base: dict[str, Any], new: dict[str, Any], path: str = "") -> int:
    """Compare provenance, descending into the nested settings dump."""
    differences = 0
    for label, keys in (
        ("baseline", sorted(set(base) - set(new))),
        ("current", sorted(set(new) - set(base))),
    ):
        if keys:
            print(f"config keys only in {label}: {[f'{path}{k}' for k in keys]}")
            differences += len(keys)
    for key in sorted(set(base) & set(new)):
        if f"{path}{key}" in IGNORED_CONFIG_FIELDS:
            continue
        left, right = base[key], new[key]
        if isinstance(left, dict) and isinstance(right, dict):
            differences += _compare_config(left, right, f"{path}{key}.")
        elif left != right:
            print(f"config {path}{key}: {left!r} vs {right!r}")
            differences += 1
    return differences


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--baseline", help="git ref to compare the working tree against"
    )
    parser.add_argument("--config", type=Path, help="subject config YAML")
    parser.add_argument("--holes", type=Path, help="implant bore YAML")
    parser.add_argument("--poses", type=Path, help="Phase-1 pool")
    parser.add_argument("--candidates", type=int, default=3, help="synthetic pool size")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--p2-iter", type=int, default=60)
    parser.add_argument("--select-by", default="objective")
    parser.add_argument("--well", choices=("thin", "thick"), default="thin")
    parser.add_argument("--pool", choices=("thread", "process"), default="thread")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--work-dir", type=Path, help="default: a new temp directory")
    parser.add_argument(
        "--keep", action="store_true", help="keep the worktree and logs"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BASELINE", "CURRENT"),
        help="compare two existing handoffs and exit",
    )
    args = parser.parse_args(argv)
    if not args.compare and not args.baseline:
        parser.error("--baseline is required unless --compare is given")
    supplied = [args.config, args.holes, args.poses]
    if any(supplied) and not all(supplied):
        parser.error("--config, --holes and --poses go together")
    return args


def stage_env(
    args: argparse.Namespace, config: Path, holes: Path, poses: Path, work: Path
) -> dict[str, str]:
    """The environment both arms run under, identical except for the tree."""
    platform = "cuda" if args.platform == "gpu" else "cpu"
    return {
        **os.environ,
        "CONFIG": str(config),
        "HOLES": str(holes),
        "POSES": str(poses),
        "SELECT_BY": args.select_by,
        "TOPK": str(args.topk),
        "WORKERS": str(args.workers),
        "P2_ITER": str(args.p2_iter),
        "WELL": args.well,
        "SOLVER": "ipopt",
        "POOL": args.pool,
        "PLATFORM": args.platform,
        "JAX_PLATFORMS": platform,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        # Shared, so the second arm pays neither the SDF build nor the compile —
        # and so a change to how the cache directory is chosen cannot be what the
        # comparison measures. Both names are set because the two code
        # generations either side of a change may read different ones.
        "AIND_LOW_POINT_CACHE_DIR": str(work / "sdf_cache"),
        "JAX_CACHE_DIR": str(work / "jax_cache"),
        "AIND_JAX_CACHE_DIR": str(work / "jax_cache"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    if args.compare:
        return 1 if compare_handoffs(*args.compare) else 0

    work = args.work_dir or Path(tempfile.mkdtemp(prefix="parity_phase2_"))
    work.mkdir(parents=True, exist_ok=True)
    worktree = work / "baseline_tree"
    print(f"work dir: {work}")

    if args.config:
        config, holes, poses = args.config, args.holes, args.poses
    else:
        config, holes, poses = write_synthetic_inputs(work, args.candidates)
        print(f"synthetic subject: {config}")

    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), args.baseline],
        cwd=REPO,
        check=True,
    )
    try:
        env = stage_env(args, config, holes, poses, work)
        base_out = run_arm(ARM_BASELINE, worktree, env, work)
        new_out = run_arm(ARM_CURRENT, REPO, env, work)
        differences = compare_handoffs(base_out, new_out)
    finally:
        if not args.keep:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=REPO,
                check=False,
            )

    if differences:
        print(f"\n{differences} DIFFERENCE(S) — the change is not behavior-preserving")
        print(f"logs and handoffs in {work}")
        return 1
    print("\nPARITY OK")
    if not args.keep and not args.work_dir:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
