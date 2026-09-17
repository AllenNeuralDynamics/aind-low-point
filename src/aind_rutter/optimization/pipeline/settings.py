"""Typed settings for the optimizer pipeline stages.

Each stage is an importable function taking one of these objects, so a caller in
Python constructs it directly and the console entry point builds it from the
environment. Resolution order is explicit arguments, then environment, then
defaults — pydantic-settings ranks constructor arguments above environment
sources, so passing only the values a caller actually supplied gives command-line
precedence without the CLI layer knowing about the environment.

Every field keeps the environment name the pipeline has always used. The two
generic ones also accept a prefixed spelling, listed first so it wins when both
are set: ``RUTTER_CONFIG`` over ``CONFIG`` and ``RUTTER_OUT`` over ``OUT``.
``CONFIG`` in particular is set for unrelated reasons in many environments.

Booleans are parsed by pydantic rather than compared against ``"1"``, which
accepts ``1/true/yes/on`` and rejects anything it cannot interpret. A value that
used to fall through to False now fails loudly at construction.

Variables read before ``jax`` and ``numpy`` import — ``PLATFORM``, ``POOL``,
``THREADS`` and ``GPU_MEM_FRACTION`` — are deliberately absent. They must be in
the environment before the stage module imports, so they cannot come from an
object built inside ``main``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env(*names: str) -> AliasChoices:
    return AliasChoices(*names)


class PipelineSettings(BaseSettings):
    """Values more than one pipeline stage reads.

    Kept separate so the stages that already share these names — the subject
    config and bore file span five and four modules respectively — cannot drift
    apart as further stages are converted.
    """

    model_config = SettingsConfigDict(
        extra="forbid", frozen=True, populate_by_name=True
    )

    config: Path = Field(
        Path("examples/836656-config-T12.yml"),
        validation_alias=_env("RUTTER_CONFIG", "CONFIG"),
        description="Subject config YAML selecting assets, targets and transforms.",
    )
    holes: Path = Field(
        Path("scratch/0283-300-04.holes.yml"),
        validation_alias=_env("RUTTER_HOLES", "HOLES"),
        description="Implant bore file, placed by the config's implant_to_lps.",
    )
    well: Literal["thin", "thick"] = Field(
        "thick",
        validation_alias=_env("WELL"),
        description="Well SDF mode; thick solidifies the thin-skin envelope.",
    )
    # Coverage normalization divides each probe's coverage by its achievable
    # ceiling and blends the average and worst region by cov_alpha; cov_weight is
    # the coverage-vs-clearance gain. Both phases must agree for the objective to
    # mean the same thing across stages.
    cov_norm: bool = Field(False, validation_alias=_env("COV_NORM"))
    cov_alpha: float = Field(0.2, validation_alias=_env("COV_ALPHA"))
    cov_weight: float = Field(1.0, validation_alias=_env("COV_WEIGHT"))


class Phase2Settings(PipelineSettings):
    """Inputs to the Phase-2 constrained pose refinement.

    Solver defaults encode the measured configuration; see
    ``dev/PHASE2_CONDITIONING.md`` for what each one is worth.
    """

    # Selection and I/O.
    topk: int = Field(80, validation_alias=_env("TOPK"))
    select_by: str = Field(
        "min_clear",
        validation_alias=_env("SELECT_BY"),
        description="Phase-1 record field to rank by; 'objective' sorts ascending.",
    )
    poses_pkl: Path = Field(
        Path("scratch/mrv_pool_results.pkl"), validation_alias=_env("POSES")
    )
    out_pkl: Path = Field(
        Path("scratch/phase2_handoff.pkl"),
        validation_alias=_env("RUTTER_OUT", "OUT"),
    )
    ranks: str = Field(
        "",
        validation_alias=_env("RANKS"),
        description="Explicit zero-based offsets into the select_by order.",
    )
    ranks_file: Path | None = Field(None, validation_alias=_env("RANKS_FILE"))

    # Execution.
    workers: int = Field(4, validation_alias=_env("WORKERS"))
    warmup: bool = Field(True, validation_alias=_env("WARMUP"))

    # Objective weights.
    minclear: float = Field(0.2, validation_alias=_env("MINCLEAR"))
    # The clearance reward duplicates the min_clearance constraint and takes its
    # gradient from the single closest surface sample, which flips between
    # near-tied samples and makes solves irreproducible. Off, poses repeat
    # bitwise across runs.
    lam_clear: float = Field(0.0, validation_alias=_env("LAM_CLEAR"))
    tau_clear: float = Field(0.8, validation_alias=_env("TAU_CLEAR"))
    # Scale of the OBB-based slack categories relative to the mm-native voxel-SDF
    # ones.
    obb_gain: float = Field(100.0, validation_alias=_env("OBB_GAIN"))
    # Soft minima instead of hard ones in the clearance reward.
    smooth_reward: bool = Field(False, validation_alias=_env("SMOOTH_REWARD"))
    # Hand the solver only rows with a gradient; padding keeps compiled shapes
    # uniform but leaves most rows constant.
    drop_dead_rows: bool = Field(False, validation_alias=_env("DROP_DEAD_ROWS"))

    # Solver.
    # IPOPT's restoration phase reaches feasibility from infeasible starts where
    # trust-constr stalls. It runs limited-memory, so only first-order
    # evaluations reach the GPU; the exact Lagrangian Hessian of this nonconvex
    # problem is indefinite away from the optimum and helps less than it costs.
    solver: Literal["ipopt", "trust-constr"] = Field(
        "ipopt", validation_alias=_env("SOLVER")
    )
    # trust-constr only: none (BFGS), dense (exact n×n Hessian, slow) or hessp
    # (exact Hessian-vector products at about the cost of a gradient).
    hess: Literal["none", "dense", "hessp"] = Field(
        "none", validation_alias=_env("HESS")
    )
    p2_iter: int = Field(200, validation_alias=_env("P2_ITER"))
    # Above the variable count, so the L-BFGS model can represent a full Hessian
    # and a longer history buys nothing.
    ip_hist: int = Field(60, validation_alias=_env("IP_HIST"))
    ip_mu: Literal["adaptive", "monotone"] = Field(
        "adaptive", validation_alias=_env("IP_MU")
    )
    # tol thresholds the overall NLP error, which is built from float32
    # derivatives over bfloat16 collision grids; IPOPT's 1e-6 default asks for
    # more precision than those gradients carry.
    ip_tol: float = Field(1e-4, validation_alias=_env("IP_TOL"))
    # Measured on the gain-carrying constraint, so it bounds the mm-native rows in
    # mm, inside the FCL gate's -1e-4. It also sets acceptable_constr_viol_tol,
    # whose IPOPT default of 1e-2 would accept 0.01 mm of overlap.
    ip_cvtol: float = Field(1e-4, validation_alias=_env("IP_CVTOL"))
    # acceptable_tol bounds the same overall error, which here is the dual
    # infeasibility. Feasible solves stall far above tol, so without this every
    # solve ends at the iteration cap or in restoration; the FCL gate decides the
    # plan either way.
    ip_acc_tol: float = Field(5.0, validation_alias=_env("IP_ACC_TOL"))
    # Consecutive acceptable iterations before stopping; 0 disables the early exit.
    ip_acc_iter: int = Field(8, validation_alias=_env("IP_ACC_ITER"))

    # Keep bands and ranking.
    fcl_tol: float = Field(0.2, validation_alias=_env("FCL_TOL"))
    # Threading keep band in g-units, judged independently of FCL; the separate
    # strict flag marks g <= 0.
    g_tol: float = Field(0.2, validation_alias=_env("G_TOL"))
    mmr_lambda: float = Field(0.5, validation_alias=_env("MMR_LAMBDA"))

    # Diagnostics.
    p2_diag: bool = Field(False, validation_alias=_env("P2_DIAG"))
    p2_perturb: float = Field(0.0, validation_alias=_env("P2_PERTURB"))
    p2_perturb_seed: int = Field(0, validation_alias=_env("P2_PERTURB_SEED"))
