"""Phase 2's intake order: the objective's head, then a learned model's order.

The model scores P(kept) × E[coverage percentile among kept plans] from features
that mean the same thing on any subject: Phase-1 coverage and objective as
percentiles within the pool, FCL, threading and clearance in mm, slider geometry,
and per-probe aggregates, never raw pose or hole identity. It is fitted at run
time on a packaged table of one whole pool solved to the end. The objective
reaches the easy best layouts first and the model reaches the ones the objective
buries, so the objective's first ``rank_head`` go first and the model orders the
rest; every prefix of that order is as good as the better of the two at that K.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import numpy as np

from aind_rutter.optimization.objectives.layout import PHASE1_PER_PROBE_VARS
from aind_rutter.optimization.pipeline.records import (
    Phase1PoolRecord,
    Phase2InputRecord,
)
from aind_rutter.optimization.pipeline.settings import Phase2Settings

MEASURED = ("p1cov", "fcl", "pp", "pf", "g")
FEATURES = (
    "p1cov_pct", "p1cov_z", "objective_pct",
    "fcl", "pp", "pf", "g", "min_clear", "min_clear_reduced", "n_arcs",
    "min_ml_gap", "min_ap_gap", "ap_span", "ap_max", "ml_span", "ml_max",
    "n_probes", "max_per_arc", "offset_max", "offset_mean", "depth_max", "depth_mean",
)  # fmt: skip
TRAINING_TABLE = "837772.npz"
MODEL: dict[str, Any] = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=10.0,
    # sklearn's early stopping holds out a random 10%, which splits layouts
    # across the holdout; the iteration count is chosen by grouped CV instead.
    early_stopping=False,
    random_state=0,
)


def percentile(v: np.ndarray) -> np.ndarray:
    """Mid-rank percentile in (0, 1), ties averaged."""
    from scipy.stats import rankdata

    return (rankdata(v) - 0.5) / len(v)


def _pose_features(rec: Phase1PoolRecord) -> list[float]:
    na = rec["n_arcs"]
    x = np.asarray(rec["x"], float)
    aps = np.sort(x[:na])
    probe = x[na:].reshape(-1, PHASE1_PER_PROBE_VARS)
    ml, offset, depth = probe[:, 0], np.hypot(probe[:, 3], probe[:, 4]), probe[:, 5]
    return [
        rec["min_ml_gap"],
        np.diff(aps).min() if na > 1 else np.nan,
        np.ptp(aps), np.abs(aps).max(), np.ptp(ml), np.abs(ml).max(),
        len(probe), max(Counter(rec["probe_to_arc_idx"].values()).values()),
        offset.max(), offset.mean(), depth.max(), depth.mean(),
    ]  # fmt: skip


def feature_matrix(
    recs: Sequence[Phase1PoolRecord], measured: dict[str, np.ndarray]
) -> np.ndarray:
    """Rows of FEATURES for a whole pool; percentiles are within that pool.

    ``measured`` holds MEASURED, from ``measure_pool``, in record order.
    """
    p1cov = measured["p1cov"]
    field = lambda k: np.array([cast(Any, r)[k] for r in recs], dtype=float)  # noqa: E731
    head = [
        percentile(p1cov),
        (p1cov - p1cov.mean()) / p1cov.std(),
        percentile(field("objective")),
        *(measured[k] for k in ("fcl", "pp", "pf", "g")),
        field("min_clear"),
        field("min_clear_reduced"),
        field("n_arcs"),
    ]
    geometry = np.array([_pose_features(r) for r in recs], float)
    return np.column_stack([*head, geometry])


_STATE: dict[str, Any] = {}


def _measure_init(settings: Phase2Settings) -> None:
    # Runs before anything in the worker initialises a jax backend, so the
    # worker stays on the CPU whatever platform the parent was told.
    import jax

    jax.config.update("jax_platforms", "cpu")
    from aind_rutter.optimization.jax_env import configure_compile_cache
    from aind_rutter.optimization.pipeline.phase2_setup import build_state

    configure_compile_cache()
    _STATE.update(build_state(settings))


def _measure_chunk(recs: list[Phase2InputRecord]) -> list[tuple]:
    from aind_rutter.optimization.pipeline.phase2_setup import measure_pose

    return [measure_pose(_STATE, r) for r in recs]


def measure_pool(
    recs: Sequence[Phase1PoolRecord], settings: Phase2Settings
) -> dict[str, np.ndarray]:
    """MEASURED at each record's Phase-1 pose, on ``settings.rank_workers`` CPUs."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    inputs = [cast(Phase2InputRecord, {**r, "pose": r["x"]}) for r in recs]
    workers = settings.rank_workers
    parts = np.array_split(np.arange(len(inputs)), workers * 4)
    with ProcessPoolExecutor(
        workers,
        mp_context=get_context("spawn"),
        initializer=_measure_init,
        initargs=(settings,),
    ) as ex:
        rows = [
            row
            for chunk in ex.map(_measure_chunk, [[inputs[i] for i in p] for p in parts])
            for row in chunk
        ]
    pp, pf, fcl, g, cov = (np.array(c, float) for c in zip(*rows))
    return {"p1cov": cov, "fcl": fcl, "pp": pp, "pf": pf, "g": g}


def training_table() -> dict[str, np.ndarray]:
    """The packaged solved pool: X (FEATURES), kept, and the coverage target."""
    path = files("aind_rutter") / "data" / "ranker" / TRAINING_TABLE
    with path.open("rb") as fh, np.load(fh) as z:
        table = {k: z[k] for k in z.files}
    if tuple(table["features"]) != FEATURES:
        raise ValueError(f"{TRAINING_TABLE} was written for other features")
    return table


def coverage_target(kept: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Coverage percentile among kept plans; NaN elsewhere."""
    t = np.full(len(kept), np.nan)
    t[kept] = percentile(coverage[kept])
    return t


def fit(X: np.ndarray, kept: np.ndarray, target: np.ndarray) -> Callable:
    """Score function Z -> P(kept) × E[target | kept]."""
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    clf = HistGradientBoostingClassifier(**MODEL).fit(X, kept)
    reg = HistGradientBoostingRegressor(**MODEL).fit(X[kept], target[kept])
    return lambda Z: clf.predict_proba(Z)[:, 1] * reg.predict(Z)


def hybrid_order(
    objective_order: Sequence[int], score: np.ndarray, head: int
) -> list[int]:
    """The objective's first ``head``, then the rest by descending score."""
    first = list(objective_order[:head])
    taken = set(first)
    rest = [int(i) for i in np.argsort(-score, kind="stable") if int(i) not in taken]
    return [int(i) for i in first] + rest


def learned_order(
    recs: Sequence[Phase1PoolRecord],
    settings: Phase2Settings,
    objective_order: Sequence[int],
    log: Callable[[str], None] | None = None,
) -> list[int]:
    """Indices of ``recs`` in Phase 2's intake order."""
    import time

    note = log or (lambda _msg: None)
    t0 = time.time()
    measured = measure_pool(recs, settings)
    table = training_table()
    score = fit(table["X"], table["kept"], table["target"])(
        feature_matrix(recs, measured)
    )
    note(
        f"learned ranking: {len(recs)} poses measured and scored in "
        f"{time.time() - t0:.0f}s; the objective's first {settings.rank_head}, "
        "then the model"
    )
    return hybrid_order(objective_order, score, settings.rank_head)


def write_training_table(
    path: Path,
    recs: Sequence[Phase1PoolRecord],
    measured: dict[str, np.ndarray],
    kept: np.ndarray,
    coverage: np.ndarray,
    source: str,
) -> None:
    """Write a training table from a pool whose every record Phase 2 solved.

    ``kept`` and ``coverage`` are Phase 2's, in record order.
    """
    kept = np.asarray(kept, bool)
    np.savez_compressed(
        path,
        X=feature_matrix(recs, measured),
        kept=kept,
        target=coverage_target(kept, np.asarray(coverage, float)),
        features=np.array(FEATURES),
        source=np.array(source),
    )
