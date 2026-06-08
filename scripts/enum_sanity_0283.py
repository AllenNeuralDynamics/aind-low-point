"""Sanity: does arc-first enumeration find the manual MANUAL_H assignment
when given the CORRECT 0283-derived bores? (It returned None with the
wrong 0274/27-hole file.)"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.arc_first_principled import (
    enumerate_arc_first_candidates,
    find_target_in_candidates,
)
from aind_low_point.optimization.holes import load_holes
from aind_low_point.optimization.visibility_atlas import build_visibility_atlas
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.transforms import compile_all_transforms
from scripts.run_optimizer import _probe_static_info, _transform_holes

MANUAL_H = {"MD": 3, "BLA": 4, "PL": 1, "VM": 7, "RSP": 5, "CA1": 10, "CLA": 12}
HOLES = "scratch/0283-300-04.holes.yml"


def main() -> int:
    cfg = ConfigModel.from_yaml("examples/836656-config-T12.yml")
    runtime = build_runtime_from_config(cfg)
    probes = [
        _probe_static_info(runtime.plan_state, runtime, n)
        for n in runtime.plan_state.probes
    ]
    holes = load_holes(Path(HOLES))
    compiled = compile_all_transforms(cfg.transforms)
    if "implant_to_lps" in compiled:
        R, t = compiled["implant_to_lps"].rotate_translate
        holes = _transform_holes(holes, R, t)
    print(f"{len(holes)} holes from {HOLES}")

    atlas = build_visibility_atlas(probes, holes, n_top=128, n_spin=72, verbose=False)
    cands = enumerate_arc_first_candidates(
        probes,
        atlas,
        max_arcs=3,
        max_probes_per_arc=4,
        per_arc_max_hole_tuples=50,
        global_max_candidates=200_000,
        verbose=False,
    )
    rank = find_target_in_candidates(cands, MANUAL_H)
    print(f"enumerated {len(cands)} candidates; manual MANUAL_H rank = {rank}")
    if rank is not None:
        _ = cands[rank]
        print(
            f"  manual is reachable in the pool at rank {rank} "
            f"(was None with the wrong 0274 bores)"
        )
    else:
        print("  manual NOT in pool — atlas visibility may need a wider grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
