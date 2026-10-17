"""One subject's Phase-2 geometry, built once per process, and what reads it.

Phase 2's solver workers and the learned ranker's measuring workers both hold
this state; neither imports the other's stage module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aind_rutter.optimization.pipeline.records import (
    MRVArcAssignment,
    MRVHoleAssignment,
    Phase2InputRecord,
)
from aind_rutter.optimization.pipeline.settings import Phase2Settings

# A pair counts as colliding below this FCL slack (mm), the strict gate's bound.
FCL_COLLIDING_MM = 1e-4


def build_state(settings: Phase2Settings) -> dict[str, Any]:
    """Probes, bores, SDFs and FCL fixtures for ``settings``'s subject.

    SDFs load from the disk cache, so this is cheap after the first build.
    """
    from aind_rutter.optimization.objectives.statics import _build_probe_static
    from aind_rutter.optimization.pipeline.fixtures import build_coverage_data
    from aind_rutter.optimization.pipeline.subject import OptimizationRuntime

    opt = OptimizationRuntime.from_config_path(
        settings.config,
        settings.holes,
        thread_corners=settings.thread_shank == "corners",
        locator=settings.locator(),
        retro_sigma_mm=settings.retro_sigma_mm,
        structure_sigma_mm=settings.structure_sigma_mm
        if settings.structure_coverage
        else None,
    )
    assets = opt.build_problem_assets(well_mode=settings.well, include_brain=True)
    # The FCL gate uses the same fixture names plus the world-frame implant BVH.
    # The implant remains excluded from soft SDF constraints because probes pass
    # through its bored holes; FCL is the ground-truth threading/body gate.
    fcl = opt.fcl_fixture_set(
        assets.fixtures, fixture_bvhs=assets.fixture_bvhs, include_implant=True
    )
    return dict(
        probes=list(opt.probes),
        holes=list(opt.holes),
        sdf=assets.probe_sdfs,
        bvh=assets.probe_bvhs,
        fx=assets.fixtures,
        fbvh=assets.fixture_bvhs,
        fcl_fixtures=fcl.fixtures,
        fcl_fbvh=fcl.bvhs,
        brain_sdf=assets.brain_sdf,
        cov_data=None,
        n_probes=len(opt.probes),
        head_pitch_deg=opt.head_pitch_deg,
        rig=opt.rig,
        build_static=_build_probe_static,
        build_cov=build_coverage_data,
    )


def static_for(state: dict[str, Any], rec: Phase2InputRecord):
    """Rebuild the per-probe static `st` for a Phase-1 pool record. Uses the
    arc assignment SAVED by Phase 1 (probe_to_hole / probe_to_arc_idx /
    arc_centroids_deg) so the geometry matches exactly what the pose `x` was
    optimized against — no re-seed, no frozenset-order ambiguity."""
    ha = MRVHoleAssignment(probe_to_hole=rec["probe_to_hole"])
    aa = MRVArcAssignment(
        probe_to_arc_idx=rec["probe_to_arc_idx"],
        arc_centroids_deg=list(rec["arc_centroids_deg"]),
    )
    return state["build_static"](
        state["probes"],
        state["holes"],
        ha,
        aa,
        bvh_cache=state["bvh"],
        sdf_by_name=state["sdf"],
    )


def measure_pose(
    state: dict[str, Any], rec: Phase2InputRecord
) -> tuple[int, int, float, float, float]:
    """Phase 2's gate and coverage at the record's own pose, unsolved.

    Returns the colliding probe-probe and probe-fixture pair counts, the worst FCL
    slack, the worst threading value and the coverage Phase 2 reports.
    """
    from aind_rutter.optimization.objectives.coverage import (
        coverage_total_over_probes,
    )
    from aind_rutter.optimization.objectives.variables import (
        _poses,
        worst_threading_g,
    )
    from aind_rutter.optimization.validation.fcl import make_fcl_validator

    n_arcs, pose = rec["n_arcs"], np.asarray(rec["pose"], float)
    st = static_for(state, rec)
    if state["cov_data"] is None:
        state["cov_data"] = state["build_cov"](state["probes"], st)
    v = make_fcl_validator(
        st,
        n_arcs,
        fixtures=tuple(state["fcl_fixtures"]),
        fixture_bvhs=state["fcl_fbvh"],
    )
    slacks = np.asarray(v.slacks(pose))
    colliding = slacks < -FCL_COLLIDING_MM
    fixture_suffixes = tuple("↔" + f for f in v.fixture_names)
    n_fixture = sum(
        1
        for name, b in zip(v.pair_names, colliding)
        if b and name.endswith(fixture_suffixes)
    )
    g = float(worst_threading_g(st, pose, n_arcs))
    Rs, ts, tp, mk = _poses(st, pose, n_arcs)
    cov = float(coverage_total_over_probes(Rs, ts, tp, mk, state["cov_data"], 41))
    return int(colliding.sum()) - n_fixture, n_fixture, float(slacks.min()), g, cov
