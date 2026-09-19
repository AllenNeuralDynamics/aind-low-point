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
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aind_rutter.optimization.pipeline.payloads import check_payload_path


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
    # Coverage normalization divides each probe's coverage by its achievable
    # ceiling and blends the average and worst region by cov_alpha; cov_weight is
    # the coverage-vs-clearance gain. Both phases must agree for the objective to
    # mean the same thing across stages.
    cov_norm: bool = Field(False)
    cov_alpha: float = Field(0.2)
    cov_weight: float = Field(1.0)

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
    plans: int = Field(
        15,
        description="How many of the ranked feasible plans to write.",
    )
    outdir: Path = Field(
        Path("scratch/plans"),
        description="Directory receiving plans/, the tree and the manifest.",
    )


class Phase1Settings(PipelineSettings):
    """Inputs to the Phase-1 pool build.

    Defaults are the THROUGHPUT preset; ``dev/POOL_RUN_CONFIGS.md`` says what
    each lever was measured to be worth.
    """

    # Optimization schedule. The two stages run the same compiled kernel, so
    # their step counts are runtime arguments rather than separate compiles.
    stage1: int = Field(500)
    stage2: int = Field(500)
    n_spins: int = Field(16)
    restore_rounds: int = Field(4)
    # Coarse surf count, then the fine steps that finish each stage. Running the
    # bulk coarse and finishing fine is a homotopy: it is both faster and finds
    # more feasible candidates. coarse_n at or above 5000 collapses to all-fine.
    coarse_n: int = Field(1000)
    reduced_fine: int = Field(50)
    full_fine: int = Field(50)

    # Batch shapes. VRAM is ~9.5 MB per candidate plus a ~2.2 GB baseline.
    chunk: int = Field(256)
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

    @property
    def two_fidelity(self) -> bool:
        """Whether a coarse pass runs at all."""
        return self.coarse_n < 5000

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
        "min_clear",
        description="Phase-1 record field to rank by; 'objective' sorts ascending.",
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

    # Objective weights.
    minclear: float = Field(0.2)
    # The clearance reward duplicates the min_clearance constraint and takes its
    # gradient from the single closest surface sample, which flips between
    # near-tied samples and makes solves irreproducible. Off, poses repeat
    # bitwise across runs.
    lam_clear: float = Field(0.0)
    tau_clear: float = Field(0.8)
    # Scale of the OBB-based slack categories relative to the mm-native voxel-SDF
    # ones.
    obb_gain: float = Field(100.0)
    # Soft minima instead of hard ones in the clearance reward.
    smooth_reward: bool = Field(False)
    # Hand the solver only rows with a gradient; padding keeps compiled shapes
    # uniform but leaves most rows constant.
    drop_dead_rows: bool = Field(False)

    # Solver.
    # IPOPT's restoration phase reaches feasibility from infeasible starts where
    # trust-constr stalls. It runs limited-memory, so only first-order
    # evaluations reach the GPU; the exact Lagrangian Hessian of this nonconvex
    # problem is indefinite away from the optimum and helps less than it costs.
    solver: Literal["ipopt", "trust-constr"] = Field("ipopt")
    # trust-constr only: none (BFGS), dense (exact n×n Hessian, slow) or hessp
    # (exact Hessian-vector products at about the cost of a gradient).
    hess: Literal["none", "dense", "hessp"] = Field("none")
    p2_iter: int = Field(200)
    # Above the variable count, so the L-BFGS model can represent a full Hessian
    # and a longer history buys nothing.
    ip_hist: int = Field(60)
    ip_mu: Literal["adaptive", "monotone"] = Field("adaptive")
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
