"""Runtime-level probe and target context resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from numpy.typing import NDArray

from aind_low_point.core import MeshTransformable
from aind_low_point.planning import ProbePlan
from aind_low_point.runtime.build import RuntimeBundle
from aind_low_point.runtime.shanks import detect_shank_tips_local

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
    target_points_LPS: NDArray[np.float64] | None = None


def resolve_plan_target_lps(
    plan: ProbePlan,
    target_index: Mapping[str, NDArray[np.floating]],
    *,
    target_points_LPS: NDArray[np.floating] | None = None,
    strict: bool = True,
) -> NDArray[np.float64]:
    """Resolve a probe plan's target point in world LPS coordinates."""
    if target_points_LPS is not None:
        return np.asarray(target_points_LPS, dtype=np.float64).reshape(-1, 3).mean(0)

    if plan.target_key is not None:
        target_pts = target_index.get(plan.target_key)
        if target_pts is not None:
            return np.asarray(target_pts, dtype=np.float64).reshape(-1, 3).mean(0)

    if plan.target_point_RAS is not None:
        ras = np.asarray(plan.target_point_RAS, dtype=np.float64).reshape(1, 3)
        return convert_coordinate_system(ras, "RAS", "LPS").reshape(3)

    if strict:
        raise RuntimeError(
            "Probe plan has no target_key or target_point_RAS; "
            "a runtime target point is required."
        )
    return np.zeros(3, dtype=np.float64)


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

    geometry = runtime.asset_catalog.get_geometry(f"probe:{plan.kind}")
    if isinstance(geometry, MeshTransformable):
        collision_mesh = geometry.raw
        shank_tips_local = detect_shank_tips_local(collision_mesh)
    else:
        collision_mesh = None
        shank_tips_local = np.zeros((1, 3), dtype=np.float64)

    return ProbeContext(
        name=name,
        kind=plan.kind,
        target_LPS=target_lps,
        target_points_LPS=None
        if target_points_LPS is None
        else np.asarray(target_points_LPS, dtype=np.float64),
        shank_tips_local=shank_tips_local,
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
