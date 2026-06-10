"""Optimization-facing runtime adapter.

This module centralizes the subject/runtime state that the fast-moving
optimization drivers need before they enter JAX kernels: config/runtime loading,
probe statics, transformed implant holes, probe SDF/BVH caches, fixture SDF/BVH
sets, brain SDF, and the implant-inclusive FCL fixture set used by Phase 2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.holes import Hole, load_holes
from aind_low_point.optimization.optimize import ProbeStaticInfo
from aind_low_point.optimization.pipeline.probe_setup import (
    RetroDensityOpts,
    _probe_static_info,
    _transform_holes,
    retro_opts_from_env,
)
from aind_low_point.runtime import (
    RuntimeBundle,
    build_runtime_from_config,
    head_pitch_deg_from_runtime,
    implant_world_geometry,
    world_geometry_for_node,
)
from aind_low_point.runtime.transforms import CompiledTransforms, compile_all_transforms

if TYPE_CHECKING:
    from aind_low_point.optimization.phase1_objective_jax import (
        BrainSDFData,
        FixtureSDFData,
    )
    from aind_low_point.optimization.sdf import ProbeSDF


@dataclass(frozen=True)
class OptimizationProblemAssets:
    """Heavy reusable geometry/cache objects for one optimization run."""

    probe_sdfs: dict[str, ProbeSDF]
    probe_bvhs: dict[str, Any | None]
    fixtures: tuple[FixtureSDFData, ...]
    well_fixture: FixtureSDFData
    fixture_bvhs: dict[str, Any]
    brain_sdf: BrainSDFData | None = None


@dataclass(frozen=True)
class FCLFixtureSet:
    """Fixture names and BVHs for the ground-truth FCL gate."""

    fixtures: tuple[Any, ...]
    bvhs: dict[str, Any]


def find_well_fixture(fixtures: Sequence[FixtureSDFData]) -> FixtureSDFData:
    """Return the configured well fixture from a fixture SDF sequence."""
    for fixture in fixtures:
        if "well" in fixture.name.lower():
            return fixture
    raise ValueError("No fixture with 'well' in its name was found")


def transform_holes_to_lps(
    holes: Sequence[Hole], compiled_transforms: CompiledTransforms
) -> tuple[Hole, ...]:
    """Apply ``implant_to_lps`` to hole geometry when the config provides it."""
    if "implant_to_lps" not in compiled_transforms:
        return tuple(holes)
    rotation, translation = compiled_transforms["implant_to_lps"].rotate_translate
    return tuple(_transform_holes(list(holes), rotation, translation))


@dataclass(frozen=True)
class OptimizationRuntime:
    """Config/runtime/probe/hole state shared by optimization pipeline stages."""

    cfg: ConfigModel
    runtime: RuntimeBundle
    holes_path: Path
    compiled_transforms: CompiledTransforms
    retro_opts: RetroDensityOpts | None
    probes: tuple[ProbeStaticInfo, ...]
    holes: tuple[Hole, ...]
    head_pitch_deg: float

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        holes_path: str | Path,
        *,
        retro_opts: RetroDensityOpts | None = None,
        use_env_retro: bool = True,
    ) -> "OptimizationRuntime":
        cfg = ConfigModel.from_yaml(config_path)
        return cls.from_config(
            cfg,
            holes_path,
            retro_opts=retro_opts,
            use_env_retro=use_env_retro,
        )

    @classmethod
    def from_env(
        cls,
        *,
        config_env: str = "CONFIG",
        holes_env: str = "HOLES",
        default_config: str = "examples/836656-config-T12.yml",
        default_holes: str = "scratch/0283-300-04.holes.yml",
    ) -> "OptimizationRuntime":
        import os

        return cls.from_config_path(
            os.environ.get(config_env, default_config),
            os.environ.get(holes_env, default_holes),
        )

    @classmethod
    def from_config(
        cls,
        cfg: ConfigModel,
        holes_path: str | Path,
        *,
        retro_opts: RetroDensityOpts | None = None,
        use_env_retro: bool = True,
    ) -> "OptimizationRuntime":
        runtime = build_runtime_from_config(cfg)
        resolved_retro: RetroDensityOpts | None
        if retro_opts is not None:
            resolved_retro = retro_opts
        elif use_env_retro:
            resolved_retro = retro_opts_from_env(runtime)
        else:
            resolved_retro = None
        probes = tuple(
            _probe_static_info(runtime.plan_state, runtime, name, resolved_retro)
            for name in runtime.plan_state.probes
        )
        compiled_transforms = compile_all_transforms(cfg.transforms)
        raw_holes = load_holes(Path(holes_path))
        holes = transform_holes_to_lps(raw_holes, compiled_transforms)
        return cls(
            cfg=cfg,
            runtime=runtime,
            holes_path=Path(holes_path),
            compiled_transforms=compiled_transforms,
            retro_opts=resolved_retro,
            probes=probes,
            holes=holes,
            head_pitch_deg=head_pitch_deg_from_runtime(runtime),
        )

    @property
    def probe_names(self) -> tuple[str, ...]:
        return tuple(probe.name for probe in self.probes)

    def probe_sdfs(self, n_surface_points: int = 5000) -> dict[str, ProbeSDF]:
        from aind_low_point.optimization.sdf import build_sdf_by_name

        return cast(
            "dict[str, ProbeSDF]",
            build_sdf_by_name(self.probes, self.runtime, n_surface_points),
        )

    def probe_bvhs(self) -> dict[str, Any | None]:
        from aind_low_point.optimization.headstages import make_fcl_bvh

        return {
            probe.name: make_fcl_bvh(probe.collision_mesh)
            if probe.collision_mesh is not None
            else None
            for probe in self.probes
        }

    def fixture_sdfs(self, *, well_mode: str = "thin") -> tuple[FixtureSDFData, ...]:
        from aind_low_point.optimization.pipeline.phase1_geometry import (
            build_fixture_sdf_data,
        )

        fixtures = build_fixture_sdf_data(self.runtime)
        if well_mode.lower() != "thick":
            return fixtures

        well_thick = self.thick_well_fixture(find_well_fixture(fixtures))
        return tuple(
            well_thick if fixture.name == well_thick.name else fixture
            for fixture in fixtures
        )

    def thick_well_fixture(self, well_thin: FixtureSDFData) -> FixtureSDFData:
        from aind_low_point.optimization.pipeline.thick_well import (
            fit_well_cone,
            make_thick_well_sdf,
        )

        geometry = world_geometry_for_node(self.runtime, well_thin.name)
        mesh = (
            geometry.raw
            if geometry is not None
            else self.runtime.asset_catalog.get_geometry("well").raw
        )
        return make_thick_well_sdf(mesh, well_thin, cone=fit_well_cone(mesh))

    def fixture_bvhs(
        self, fixtures: Sequence[FixtureSDFData] | None = None
    ) -> dict[str, Any]:
        from aind_low_point.optimization.headstages import make_fcl_bvh

        selected = self.fixture_sdfs() if fixtures is None else fixtures
        out: dict[str, Any] = {}
        for fixture in selected:
            geometry = world_geometry_for_node(self.runtime, fixture.name)
            if geometry is not None:
                out[fixture.name] = make_fcl_bvh(geometry.raw)
        return out

    def brain_sdf(self) -> BrainSDFData | None:
        from aind_low_point.optimization.pipeline.phase1_geometry import (
            maybe_build_brain_sdf,
        )

        return maybe_build_brain_sdf(self.runtime, self.compiled_transforms)

    def fcl_fixture_set(
        self,
        fixtures: Sequence[Any],
        *,
        fixture_bvhs: Mapping[str, Any] | None = None,
        include_implant: bool = True,
    ) -> FCLFixtureSet:
        from aind_low_point.optimization.headstages import make_fcl_bvh

        bvhs = dict(
            self.fixture_bvhs(cast(Any, fixtures))
            if fixture_bvhs is None
            else fixture_bvhs
        )
        fcl_fixtures = tuple(fixtures)
        if not include_implant:
            return FCLFixtureSet(fcl_fixtures, bvhs)

        implant = implant_world_geometry(self.runtime)
        if implant is not None:
            bvhs["implant"] = make_fcl_bvh(implant.raw)
            fcl_fixtures = fcl_fixtures + (SimpleNamespace(name="implant"),)
        return FCLFixtureSet(fcl_fixtures, bvhs)

    def build_problem_assets(
        self,
        *,
        n_surface_points: int = 5000,
        well_mode: str = "thin",
        include_brain: bool = False,
    ) -> OptimizationProblemAssets:
        fixtures = self.fixture_sdfs(well_mode=well_mode)
        return OptimizationProblemAssets(
            probe_sdfs=self.probe_sdfs(n_surface_points),
            probe_bvhs=self.probe_bvhs(),
            fixtures=fixtures,
            well_fixture=find_well_fixture(fixtures),
            fixture_bvhs=self.fixture_bvhs(fixtures),
            brain_sdf=self.brain_sdf() if include_brain else None,
        )
