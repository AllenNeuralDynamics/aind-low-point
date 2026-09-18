"""Runtime-level probe and target context resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from aind_rutter.core import MeshTransformable
from aind_rutter.optimization.geometry.recording import (
    RecordingGeometry,
    pivot_from_shank_tips,
)
from aind_rutter.planning import ProbePlan, probe_asset_key, resolve_target_LPS
from aind_rutter.runtime.build import RuntimeBundle
from aind_rutter.runtime.shanks import detect_shank_tips_local

if TYPE_CHECKING:
    import trimesh


@dataclass(frozen=True)
class ProbeContext:
    """Runtime interpretation of one planned probe, before optimizer conversion."""

    name: str
    kind: str
    target_LPS: NDArray[np.float64]
    shank_tips_local: NDArray[np.float64]
    collision_mesh: "trimesh.Trimesh | None"
    coverage_weight: float
    # Where this probe's electrodes are, as the config resolved it. ``None``
    # means it has no recording array and targets with its tip.
    recording: "RecordingGeometry | None" = None
    pivot_local: NDArray[np.float64] | None = None
    target_points_LPS: NDArray[np.float64] | None = None


def resolve_plan_target_lps(
    plan: ProbePlan,
    target_index: Mapping[str, NDArray[np.floating]],
    *,
    target_points_LPS: NDArray[np.floating] | None = None,
    strict: bool = True,
) -> NDArray[np.float64]:
    """Resolve a probe plan's target point in world LPS coordinates.

    The optimizer's view of :func:`aind_rutter.planning.resolve_target_LPS`,
    which the app shares. The two used to be separate implementations that
    disagreed about which form of target wins.
    """
    return resolve_target_LPS(
        plan,
        target_index,
        points_LPS=target_points_LPS,
        strict=strict,
    )


def coverage_weight_for_probe(
    runtime: RuntimeBundle,
    name: str,
    plan: ProbePlan | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> float:
    """Resolve per-probe coverage weight from env override or target metadata."""
    plan = runtime.plan_state.probes[name] if plan is None else plan
    env = os.environ if environ is None else environ
    for token in env.get("COVERAGE_WEIGHTS", "").split(","):
        if ":" not in token:
            continue
        probe_name, raw_weight = token.split(":", 1)
        if probe_name.strip() == name:
            return float(raw_weight)

    if plan.target_key is not None:
        try:
            spec = runtime.asset_catalog.get_spec(plan.target_key)
            weight = spec.metadata.get("coverage_weight") if spec is not None else None
            if weight is not None:
                return float(weight)
        except Exception:
            pass
    return 1.0


def probe_context_from_runtime(
    runtime: RuntimeBundle,
    name: str,
    *,
    target_points_LPS: NDArray[np.floating] | None = None,
    coverage_environ: Mapping[str, str] | None = None,
) -> ProbeContext:
    """Build runtime-level probe context for one planned probe."""
    plan = runtime.plan_state.probes[name]
    target_lps = resolve_plan_target_lps(
        plan,
        runtime.plan_state.target_index,
        target_points_LPS=target_points_LPS,
    )

    asset_key = probe_asset_key(plan.kind)
    geometry = runtime.asset_catalog.get_geometry(asset_key)
    if isinstance(geometry, MeshTransformable):
        collision_mesh = geometry.raw
        shank_tips_local = detect_shank_tips_local(collision_mesh)
    else:
        collision_mesh = None
        shank_tips_local = np.zeros((1, 3), dtype=np.float64)

    # A configured pivot wins. The app has always honoured AssetSpec.pivot_LPS
    # while the optimizer recomputed its own, so a config that set one moved the
    # drawn probe and not the optimized one.
    spec = runtime.asset_catalog.assets.get(asset_key)
    recording = None if spec is None else spec.recording
    # A probe with no recording array pivots on its tips, which is what a
    # centre of zero gives.
    center_mm = 0.0 if recording is None else float(recording.active_center_mm)
    pivot_local = (
        np.asarray(spec.pivot_LPS, dtype=np.float64)
        if spec is not None and spec.pivot_LPS is not None
        else pivot_from_shank_tips(shank_tips_local, center_mm)
    )

    return ProbeContext(
        name=name,
        kind=plan.kind,
        recording=recording,
        target_LPS=target_lps,
        target_points_LPS=None
        if target_points_LPS is None
        else np.asarray(target_points_LPS, dtype=np.float64),
        shank_tips_local=shank_tips_local,
        pivot_local=pivot_local,
        collision_mesh=collision_mesh,
        coverage_weight=coverage_weight_for_probe(
            runtime, name, plan, environ=coverage_environ
        ),
    )


def probe_contexts_from_runtime(
    runtime: RuntimeBundle,
    *,
    coverage_environ: Mapping[str, str] | None = None,
) -> tuple[ProbeContext, ...]:
    """Build runtime-level context for all planned probes in declaration order."""
    return tuple(
        probe_context_from_runtime(runtime, name, coverage_environ=coverage_environ)
        for name in runtime.plan_state.probes
    )
