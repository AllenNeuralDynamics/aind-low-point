"""Classifying solved candidates and recording what produced them.

Kept free of jax so it can be imported and tested without initialising a GPU
backend: the stage module sets JAX platform and memory options at import, which a
unit test has no business triggering.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from aind_rutter.optimization.pipeline.records import Phase2ResultRecord
from aind_rutter.optimization.pipeline.settings import Phase2Settings


def classify_results(
    results: list[Phase2ResultRecord], settings: Phase2Settings
) -> list[Phase2ResultRecord]:
    """Flag both feasibility axes on every result and return the kept band.

    The FCL gate includes the implant, so a plan whose shank pierces the implant
    solid gets fcl < 0 here and no separate threading gate is needed;
    ``max_g_thread`` records how close a kept plan sits to its bore walls.

    FCL (probe/fixture/implant collisions, mm) and threading g (bore fit) are
    independent and both REPORTED rather than hard-gated upstream. STRICT means
    truly feasible on both; the KEEP band also admits mildly-infeasible plans a
    human can salvage. Dropped plans stay in the payload's ``all`` for
    inspection. Under a coverage floor a plan is kept only when every probe's
    target region keeps its fraction of the share that probe records alone, so
    no target is traded away for clearance.
    """
    kept: list[Phase2ResultRecord] = []
    for r in results:
        floor_slack = r.get("coverage_floor_slack")
        coverage_keep = floor_slack is None or floor_slack >= 0.0
        g = r.get("max_g_thread", float("nan"))
        fcl_keep = bool(r["fcl"] >= -settings.fcl_tol)
        thread_keep = bool(np.isfinite(g) and g <= settings.g_tol)
        r["fcl_strict"] = bool(r["fcl"] >= -1e-4)
        r["thread_strict"] = bool(np.isfinite(g) and g <= 0.0)
        r["strict_feasible"] = bool(r["fcl_strict"] and r["thread_strict"])
        r["fcl_keep"] = fcl_keep
        r["thread_keep"] = thread_keep
        r["coverage_keep"] = coverage_keep
        r["kept"] = fcl_keep and thread_keep and coverage_keep
        if r["kept"]:
            kept.append(r)
    return kept


def best_alone_by_probe(
    settings: Phase2Settings, probes, best_alone: Mapping[str, float] | None
) -> tuple[float, ...] | None:
    """The pool's recorded shares alone in ``probes`` order, or None without a
    floor.

    Raises
    ------
    ValueError
        If the settings set a floor and the pool lacks a probe's share, or no
        radius to measure it at.
    """
    if settings.min_coverage_fraction == 0.0:
        return None
    if settings.recorded_radius_mm is None:
        raise ValueError("RUTTER_MIN_COVERAGE_FRACTION needs RUTTER_RECORDED_RADIUS_MM")
    if best_alone is None:
        raise ValueError(
            "RUTTER_MIN_COVERAGE_FRACTION is set but the pool records no share "
            "alone per probe; build the pool with the same floor"
        )
    missing = [p.name for p in probes if p.name not in best_alone]
    if missing:
        raise ValueError(f"the pool records no share alone for {missing}")
    return tuple(float(best_alone[p.name]) for p in probes)


def handoff_config(settings: Phase2Settings) -> dict[str, object]:
    """Provenance recorded beside the results.

    ``settings`` holds every field under its own name, and ``feasibility`` and
    ``objective`` the specs they resolve to. The flat keys beside them repeat a
    subset under the spelling earlier handoffs used, so one reader can load runs
    written either side of the settings refactor.
    """
    pinned = settings.pinned_specs()
    # As in the pool: the default rig's ranges are left out.
    if not pinned["rig"]:
        del pinned["rig"]
    return {
        "settings": settings.model_dump(mode="json"),
        **pinned,
        "subject_config": str(settings.config),
        "topk": settings.topk,
        "select_by": settings.select_by,
        "minclear": settings.min_clearance,
        "lam_clear": settings.lam_clear,
        "tau_clear": settings.tau_clear,
        "p2_iter": settings.p2_iter,
        "ip_tol": settings.ip_tol,
        "acc_tol": settings.ip_acc_tol,
        "fcl_tol": settings.fcl_tol,
        "g_tol": settings.g_tol,
        "mmr_lambda": settings.mmr_lambda,
        "well": settings.well,
        "solver": settings.solver,
        "drop_dead_rows": settings.drop_dead_rows,
        "obb_gain": settings.obb_gain,
        "smooth_reward": settings.smooth_reward,
        "diag": settings.p2_diag,
        "perturb_scale": settings.p2_perturb,
        "perturb_seed": settings.p2_perturb_seed,
        "ranks_file": str(settings.ranks_file or ""),
    }
