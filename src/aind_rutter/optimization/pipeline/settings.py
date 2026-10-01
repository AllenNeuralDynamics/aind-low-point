"""Typed settings for the optimizer pipeline stages.

Each stage is an importable function taking one of these objects, so a caller in
Python constructs it directly and the console entry point builds it from the
environment. Resolution order is explicit arguments, then environment, then
defaults — pydantic-settings ranks constructor arguments above environment
sources, so passing only the values a caller actually supplied gives command-line
precedence without the CLI layer knowing about the environment.

Every field reads one environment variable: ``RUTTER_`` followed by the field's
name, upper-cased. The unprefixed spellings the pipeline used to accept are
gone — ``CONFIG``, ``OUT``, ``LIMIT``, ``N`` and ``WORKERS`` are set for
unrelated reasons in ordinary shells, and a stage quietly reading one of those
changed what it optimized.

Booleans are parsed by pydantic rather than compared against ``"1"``, which
accepts ``1/true/yes/on`` and rejects anything it cannot interpret. A value that
used to fall through to False now fails loudly at construction.

Variables read before ``jax`` and ``numpy`` import — ``PLATFORM``, ``POOL``,
``THREADS`` and ``GPU_MEM_FRACTION`` — are deliberately absent. They must be in
the environment before the stage module imports, so they cannot come from an
object built inside ``main``. The same goes for ``JAX_PLATFORMS`` and the
``XLA_PYTHON_CLIENT_*`` pair.

Two result-changing values are still read where they are used rather than here:
``THREADING_MARGIN_MM`` in ``geometry.holes`` and ``RETRO_DENSITY`` in
``pipeline.probe_setup``. Both sit below the pipeline layer, so surfacing them
would mean passing settings down into ``geometry`` and ``objectives``; that is a
layering change rather than a configuration one. ``tests/architecture`` keeps
them visible in the meantime.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aind_rutter.optimization.objectives.feasibility import FeasibilitySpec
from aind_rutter.optimization.objectives.objective_spec import ObjectiveSpec
from aind_rutter.optimization.pipeline.payloads import check_payload_path

# Retired variables, refused by name, and what to do instead.
RETIRED_VARIABLES = {
    "RUTTER_MINCLEAR": "set RUTTER_MIN_CLEARANCE, read by every stage",
    "RUTTER_COARSE_N": "Phase 1 always reads every surface sample",
    "RUTTER_REDUCED_FINE": "each Phase-1 stage runs as one pass",
    "RUTTER_FULL_FINE": "each Phase-1 stage runs as one pass",
}


class PipelineSettings(BaseSettings):
    """Values more than one pipeline stage reads.

    Kept separate so the stages that already share these names — the subject
    config and bore file span five and four modules respectively — cannot drift
    apart as further stages are converted.
    """

    model_config = SettingsConfigDict(
        env_prefix="RUTTER_", extra="forbid", frozen=True, populate_by_name=True
    )

    config: Path = Field(
        Path("examples/836656-config-T12.yml"),
        description="Subject config YAML selecting assets, targets and transforms.",
    )
    holes: Path = Field(
        Path("scratch/0283-300-04.holes.yml"),
        description="Implant bore file, placed by the config's implant_to_lps.",
    )
    well: Literal["thin", "thick"] = Field(
        "thick",
        description="Well SDF mode; thick solidifies the thin-skin envelope.",
    )
    thread_shank: Literal["corners", "centreline"] = Field(
        "corners",
        description=(
            "What must stay inside each bore: the four corner edges of the "
            "shank's cross-section, or its centreline, whose bore inset "
            "(RUTTER_THREADING_MARGIN_MM) must also cover the shank's width."
        ),
    )
    # Clearance every stage requires between probe and fixture surfaces (mm),
    # read as the soft minimum plus its measured shortfall; see feasibility_spec().
    min_clearance: float = Field(0.1)
    # The objective, which every stage minimizes; see objective_spec().
    # Coverage normalization divides each probe's coverage by its achievable
    # ceiling and blends the average and worst region by cov_alpha; cov_weight is
    # the coverage-vs-margin gain.
    cov_norm: bool = Field(True)
    cov_alpha: float = Field(0.2)
    cov_weight: float = Field(7.0)
    # Clearance-margin reward weight and saturation scale (mm), and whether it
    # reads the soft minimum.
    lam_clear: float = Field(ObjectiveSpec.lambda_margin_clear)
    tau_clear: float = Field(ObjectiveSpec.tau_clear_mm)
    smooth_reward: bool = Field(ObjectiveSpec.smooth_clearance_reward)
    # Threading-margin reward weight, and whether its minimum over shanks and its
    # zero gate are smoothed.
    lam_thread: float = Field(ObjectiveSpec.lambda_margin_thread)
    smooth_thread_reward: bool = Field(ObjectiveSpec.smooth_thread_reward)
    # Solve over each probe's spin angle rather than its (cos, sin) pair, whose
    # radius no geometric term constrains and whose origin, inside the variable
    # bounds, is a singularity of arctan2. Each stage converts at its own
    # boundary and the pool keeps the pair, so the stages need not agree.
    spin_param: Literal["unit_vector", "angle"] = Field("angle")

    n_surf: int = Field(
        5000,
        description="Surface points per probe SDF; the clearance query set.",
    )
    well_margin: float = Field(
        0.5,
        description="mm the solidified well cone is grown by, inside and out.",
    )
    atlas_cache: Path | None = Field(
        None,
        description="Visibility atlas cache; defaults to one named for the config.",
    )

    def feasibility_spec(self) -> FeasibilitySpec:
        """What every stage counts as feasible."""
        return FeasibilitySpec(
            min_clearance_mm=self.min_clearance, softmin_correction=True
        )

    def pinned_specs(self) -> dict[str, dict[str, Any]]:
        """The specs a pool is built under, which Phase 2 must share."""
        return {
            "feasibility": asdict(self.feasibility_spec()),
            "objective": asdict(self.objective_spec()),
        }

    def objective_spec(self) -> ObjectiveSpec:
        """The objective every stage minimizes."""
        return ObjectiveSpec(
            coverage_gain=self.cov_weight,
            # Read only with coverage ceilings; zero otherwise so the compile
            # cache does not split on a value the trace ignores.
            cov_alpha=self.cov_alpha if self.cov_norm else 0.0,
            lambda_margin_clear=self.lam_clear,
            tau_clear_mm=self.tau_clear,
            smooth_clearance_reward=self.smooth_reward,
            lambda_margin_thread=self.lam_thread,
            smooth_thread_reward=self.smooth_thread_reward,
        )

    @model_validator(mode="before")
    @classmethod
    def _refuse_retired_variables(cls, data: Any) -> Any:
        # An unknown RUTTER_ variable is ignored, so a run still setting a retired
        # one would silently get the default instead.
        for old, instead in RETIRED_VARIABLES.items():
            if old in os.environ:
                raise ValueError(f"{old} is retired: {instead}")
        return data

    @model_validator(mode="after")
    def _caches_follow_the_config(self) -> "PipelineSettings":
        """Name each cache after the subject, since both are subject-specific.

        Sharing one between subjects silently reuses the wrong geometry.
        """
        if self.atlas_cache is None:
            stem = Path(self.config).stem
            object.__setattr__(
                self, "atlas_cache", Path(f"scratch/atlas_{stem}.json.gz")
            )
        return self


class EmitSettings(PipelineSettings):
    """Inputs to plan emission, which turns a handoff into plan-only YAML."""

    handoff: Path = Field(
        Path("scratch/phase2_handoff.json"),
        description="Phase-2 handoff to emit from.",
    )
    plans: int | None = Field(
        None,
        description="How many of the ranked feasible plans to write; unset writes all.",
    )
    outdir: Path = Field(
        Path("scratch/plans"),
        description="Directory receiving plans/, the tree and the manifests.",
    )
    recorded_radius_mm: float | None = Field(
        0.2,
        description=(
            "How near an active site a region's point must lie to count as "
            "recorded, for the per-region share each plan's manifest entry "
            "carries; unset skips the measure and the mesh load it needs."
        ),
    )


class Phase1Settings(PipelineSettings):
    """Inputs to the Phase-1 pool build."""

    # Optimization schedule. The two stages run the same compiled kernel, so
    # their step counts are runtime arguments rather than separate compiles.
    # stage1 is the reduced stage (offsets and depth pinned, no coverage term);
    # off by default, since the full stage alone delivers the same layouts in half
    # the time.
    stage1: int = Field(0)
    stage2: int = Field(500)
    n_spins: int = Field(16)
    restore_rounds: int = Field(4)
    fixture_c2f: bool = Field(
        True,
        description="Read the probe side of probe-fixture body clearance from the "
        "probes' coarse-to-fine samples rather than their uniform surface samples.",
    )

    # Batch shapes. A full-sample RProp step's largest buffer is ~17 MB per
    # candidate, which at 256 does not fit a 10 GB card beside a whole group's
    # restore; 128 runs as fast.
    chunk: int = Field(128)
    restore_chunk: int = Field(128)
    pipeline_depth: int = Field(2)
    bf16_store: bool = Field(True)

    # Enumeration caps. Defaults reproduce the historical 3-arc pool; the
    # kinematic maxima are 8 each.
    max_arcs: int = Field(3)
    max_probes_per_arc: int = Field(4)
    only_narcs: int = Field(
        0,
        description="Build only this arc-count group; 0 builds all of them.",
    )
    limit: int = Field(
        0,
        description="Cap candidates for a smoke test; disables the seed cache.",
    )
    seed_workers: int = Field(
        default_factory=lambda: min(os.cpu_count() or 1, 16),
    )

    # I/O.
    out: Path = Field(
        Path("scratch/mrv_pool_results.json.gz"),
        description="Where the pool is written; resumable, groups are skipped.",
    )
    seed_cache: Path | None = Field(
        None,
        description="Enumerate+seed cache; defaults to one named for the config.",
    )
    progress_every: int = Field(25)
    # Diagnostic only: the RProp stages say nothing about how they got where they
    # got, so record per-iteration rates for this many iterations of each stage
    # beside the pool. Tracing changes XLA's fusion, so a traced run is
    # comparable to other traced runs, not to an untraced one.
    trace: int = Field(
        0,
        description="Record this many RProp iterations per stage to <out>_trace.npz.",
    )

    @model_validator(mode="after")
    def _seed_cache_follows_the_config(self) -> "Phase1Settings":
        if self.seed_cache is None:
            stem = Path(self.config).stem
            object.__setattr__(
                self, "seed_cache", Path(f"scratch/mrv_seeds_{stem}.json.gz")
            )
        return self

    @field_validator("out")
    @classmethod
    def _json_payload(cls, path: Path) -> Path:
        return check_payload_path(path)


class Phase2Settings(PipelineSettings):
    """Inputs to the Phase-2 constrained pose refinement.

    Solver defaults encode the measured configuration; see
    ``dev/PHASE2_CONDITIONING.md`` for what each one is worth.
    """

    # Selection and I/O.
    topk: int = Field(80)
    select_by: str = Field(
        "learned",
        description="'learned' (see pipeline.learned_rank), or a Phase-1 record "
        "field to rank by; 'objective' sorts ascending, any other field descending.",
    )
    rank_head: int = Field(
        100, description="'learned' only: how many the objective orders first."
    )
    rank_workers: int = Field(
        8, description="'learned' only: CPU processes measuring the pool."
    )
    ranker_table: Path | None = Field(
        None,
        description="'learned' only: the model's training table, a pool Phase 2 "
        "solved whole (pipeline.learned_rank.write_training_table). Needed when the "
        "cut reaches past rank_head.",
    )
    poses: Path = Field(
        Path("scratch/mrv_pool_results.json.gz"),
        description="Phase-1 pool to select from.",
    )
    out: Path = Field(
        Path("scratch/phase2_handoff.json"),
        description="Where the handoff is written.",
    )
    ranks: str = Field(
        "",
        description="Explicit zero-based offsets into the select_by order.",
    )
    ranks_file: Path | None = Field(None)

    @field_validator("poses", "out")
    @classmethod
    def _json_payload(cls, path: Path) -> Path:
        # Checked at construction so a wrong path fails before any solving.
        return check_payload_path(path)

    # Execution.
    workers: int = Field(4)
    warmup: bool = Field(True)

    # Scale of the OBB-based slack categories relative to the mm-native voxel-SDF
    # ones.
    obb_gain: float = Field(100.0)
    # Hand the solver only rows with a gradient; padding keeps compiled shapes
    # uniform but leaves most rows constant, and constant rows at the fill value
    # drive IPOPT's adaptive barrier parameter up by orders of magnitude.
    drop_dead_rows: bool = Field(True)
    # Collision-grid storage. bfloat16 halves grid memory and rounds each stored
    # distance to about 0.4% of its value; interpolation is float32 either way.
    grid_dtype: Literal["bfloat16", "float32"] = Field("bfloat16")

    # Solver.
    # IPOPT's restoration phase reaches feasibility from infeasible starts where
    # trust-constr stalls. It runs limited-memory, so only first-order
    # evaluations reach the GPU; the exact Lagrangian Hessian of this nonconvex
    # problem is indefinite away from the optimum and helps less than it costs.
    solver: Literal["ipopt", "trust-constr"] = Field("ipopt")
    # trust-constr only: none (BFGS), dense (exact n×n Hessian, slow) or hessp
    # (exact Hessian-vector products at about the cost of a gradient).
    hess: Literal["none", "dense", "hessp"] = Field("none")
    p2_iter: int = Field(300)
    # Above the variable count, so the L-BFGS model can represent a full Hessian
    # and a longer history buys nothing.
    ip_hist: int = Field(60)
    ip_mu: Literal["adaptive", "monotone"] = Field("adaptive")
    # Upper bound on the adaptive barrier parameter; IPOPT's own default. With
    # constant rows dropped, clearance rows continuous and multipliers started from
    # μ, the update needs no cap.
    ip_mu_max: float = Field(1e5)
    # IPOPT's bound_mult_init_method. "constant" starts every multiplier at 1, so an
    # inactive row starts with complementarity equal to its value, and a gain-100
    # clearance row at its cap reads 130 against a threading row's ~0.5; the
    # adaptive update then raises μ from the first iterations. "mu-based" starts
    # each at μ over its slack, so every row starts at μ.
    ip_bound_mult_init: Literal["constant", "mu-based"] = Field("mu-based")
    # tol thresholds the overall NLP error, which is built from float32
    # derivatives over bfloat16 collision grids; IPOPT's 1e-6 default asks for
    # more precision than those gradients carry.
    ip_tol: float = Field(1e-4)
    # Measured on the gain-carrying constraint, so it bounds the mm-native rows in
    # mm, inside the FCL gate's -1e-4. It also sets acceptable_constr_viol_tol,
    # whose IPOPT default of 1e-2 would accept 0.01 mm of overlap.
    ip_cvtol: float = Field(1e-4)
    # acceptable_tol bounds the same overall error, which here is the dual
    # infeasibility. Feasible solves stall far above tol, so without this every
    # solve ends at the iteration cap or in restoration; the FCL gate decides the
    # plan either way.
    ip_acc_tol: float = Field(5.0)
    # Consecutive acceptable iterations before stopping; 0 disables the early exit.
    ip_acc_iter: int = Field(8)
    # The acceptable test's separate bounds on its dual-infeasibility and
    # complementarity legs, which ip_acc_tol otherwise reaches only through the
    # overall error. Both hold IPOPT's own defaults, so the test is unchanged
    # until one is lowered, and only a value below ip_acc_tol tightens the dual
    # leg at all.
    ip_acc_du_tol: float = Field(1e10)
    ip_acc_compl_tol: float = Field(1e-2)

    # Keep bands and ranking.
    fcl_tol: float = Field(0.2)
    # Threading keep band in g-units, judged independently of FCL; the separate
    # strict flag marks g <= 0.
    g_tol: float = Field(0.2)
    mmr_lambda: float = Field(0.5)

    # Diagnostics.
    p2_diag: bool = Field(False)
    p2_perturb: float = Field(0.0)
    p2_perturb_seed: int = Field(0)

    @model_validator(mode="after")
    def _spin_angle_needs_ipopt(self) -> "Phase2Settings":
        # trust-constr takes the NonlinearConstraint path, which the angle
        # adapter does not wrap, and its exact Hessians would need the map's
        # curvature too. Asking for it there is an error; inheriting it from the
        # default is not, or no trust-constr run could be configured at all.
        if self.spin_param == "angle" and self.solver != "ipopt":
            if "spin_param" in self.model_fields_set:
                raise ValueError("spin_param='angle' is only wired for solver='ipopt'")
            object.__setattr__(self, "spin_param", "unit_vector")
        return self
