"""What Phase 1 resolved from the environment before `Phase1Settings` existed.

Phase 1 read 21 values as module globals at import. Nothing in this suite runs a
Phase-1 pool — it wants a GPU and an hour — so the conversion to a settings
object was checked by resolution rather than outcome, against the two tables
below: each field, from the same variable, to the same value.

The tables were read out of the module before the conversion
(commit `26ec9ed`). They are the record of what the globals did, so
`Phase1Settings` is still being compared against the old behaviour and not
against itself.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# global -> settings attribute
FIELDS: dict[str, str] = {
    "STAGE1": "stage1",
    "STAGE2": "stage2",
    "N_SPINS": "n_spins",
    "RESTORE_ROUNDS": "restore_rounds",
    "CHUNK": "chunk",
    "RESTORE_CHUNK": "restore_chunk",
    "PIPELINE_DEPTH": "pipeline_depth",
    "LIMIT": "limit",
    "MAX_ARCS": "max_arcs",
    "MAX_PPA": "max_probes_per_arc",
    "ONLY_NARCS": "only_narcs",
    "BF16": "bf16_store",
    "PROGRESS_EVERY": "progress_every",
    "OUT": "out",
    "SEED_CACHE": "seed_cache",
    "COARSE_N": "coarse_n",
    "REDUCED_FINE": "reduced_fine",
    "FULL_FINE": "full_fine",
    "WELL_MODE": "well",
    "COV_NORM": "cov_norm",
    "COV_ALPHA": "cov_alpha",
    "COV_WEIGHT": "cov_weight",
    "TWO_FIDELITY": "two_fidelity",
}

# A value per variable that is not its default, so a field that ignores its
# variable is visible. COARSE_N sits at the threshold where the coarse pass
# collapses to all-fine, which is what flips TWO_FIDELITY.
SAMPLE: dict[str, str] = {
    "STAGE1": "300",
    "STAGE2": "400",
    "N_SPINS": "8",
    "RESTORE_ROUNDS": "3",
    "CHUNK": "64",
    "RESTORE_CHUNK": "32",
    "PIPELINE_DEPTH": "3",
    "LIMIT": "7",
    "MAX_ARCS": "5",
    "MAX_PROBES_PER_ARC": "6",
    "ONLY_NARCS": "2",
    "BF16_STORE": "0",
    "PROGRESS_EVERY": "13",
    "OUT": "scratch/other_pool.json.gz",
    "SEED_CACHE": "scratch/other_seeds.json.gz",
    "COARSE_N": "5000",
    "REDUCED_FINE": "100",
    "FULL_FINE": "90",
    "WELL": "thin",
    "COV_NORM": "true",
    "COV_ALPHA": "0.4",
    "COV_WEIGHT": "2.5",
    "CONFIG": "examples/836656-config.yml",
}

DEFAULT_ENV: dict[str, str] = {"CONFIG": "examples/836656-config.yml"}

# What the globals resolved to under SAMPLE.
RESOLVED_SAMPLE: dict[str, str] = {
    "BF16": "False",
    "CHUNK": "64",
    "COARSE_N": "5000",
    "COV_ALPHA": "0.4",
    "COV_NORM": "True",
    "COV_WEIGHT": "2.5",
    "FULL_FINE": "90",
    "LIMIT": "7",
    "MAX_ARCS": "5",
    "MAX_PPA": "6",
    "N_SPINS": "8",
    "ONLY_NARCS": "2",
    "OUT": "scratch/other_pool.json.gz",
    "PIPELINE_DEPTH": "3",
    "PROGRESS_EVERY": "13",
    "REDUCED_FINE": "100",
    "RESTORE_CHUNK": "32",
    "RESTORE_ROUNDS": "3",
    "SEED_CACHE": "scratch/other_seeds.json.gz",
    "STAGE1": "300",
    "STAGE2": "400",
    "TWO_FIDELITY": "False",
    "WELL_MODE": "thin",
}

# And with nothing set but the subject.
RESOLVED_DEFAULT: dict[str, str] = {
    "BF16": "True",
    "CHUNK": "256",
    "COARSE_N": "1000",
    "COV_ALPHA": "0.2",
    "COV_NORM": "False",
    "COV_WEIGHT": "1.0",
    "FULL_FINE": "50",
    "LIMIT": "0",
    "MAX_ARCS": "3",
    "MAX_PPA": "4",
    "N_SPINS": "16",
    "ONLY_NARCS": "0",
    "OUT": "scratch/mrv_pool_results.json.gz",
    "PIPELINE_DEPTH": "2",
    "PROGRESS_EVERY": "25",
    "REDUCED_FINE": "50",
    "RESTORE_CHUNK": "128",
    "RESTORE_ROUNDS": "4",
    "SEED_CACHE": "scratch/mrv_seeds_836656-config.json.gz",
    "STAGE1": "500",
    "STAGE2": "500",
    "TWO_FIDELITY": "True",
    "WELL_MODE": "thick",
}
