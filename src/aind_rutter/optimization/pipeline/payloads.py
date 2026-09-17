"""On-disk form of the Phase-1 pool and the Phase-2 handoff.

Both are JSON, gzip-compressed when the path ends in ``.gz``, and validated
against the models here on every write and read. Arrays are stored with their
dtype and shape, and JSON holds float64 and float32 values exactly (NaN and the
infinities as bare constants), so a read returns the arrays that were written.

In memory the pipeline passes the TypedDicts in ``contracts``; these models are
only the file format, and ``tests/test_payloads.py`` keeps the two in step. A
handoff's ``ranked`` records are entries of ``all``, so the file stores each
record once and ``ranked`` as candidate ids.

Kept free of jax so it can be imported and tested without a GPU backend.
"""

from __future__ import annotations

import gzip
import os
import tempfile
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    ValidationError,
)

from aind_rutter.optimization.pipeline.contracts import (
    Phase1PoolPayload,
    Phase2HandoffPayload,
)

SCHEMA_VERSION = 1
# gzip level 6 is ~3x faster than the default 9 on pool-sized files for a few
# percent more bytes.
_GZIP_LEVEL = 6


def _decode_array(value: object, dtype: np.dtype | None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, Mapping) and {"dtype", "shape", "data"} <= set(value):
        arr = np.asarray(value["data"], dtype=np.dtype(value["dtype"])).reshape(
            value["shape"]
        )
    else:
        raise ValueError(f"expected an ndarray, got {type(value).__name__}")
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"unsupported array dtype {arr.dtype}")
    # Exact match: the file records the dtype, so a producer that drifts to a
    # different one would otherwise change what readers get without any error.
    if dtype is not None and arr.dtype != dtype:
        raise ValueError(f"expected dtype {dtype}, got {arr.dtype}")
    return arr


def _encode_array(arr: np.ndarray) -> dict[str, object]:
    return {
        "dtype": arr.dtype.name,
        "shape": list(arr.shape),
        "data": arr.ravel().tolist(),
    }


_ENCODE = PlainSerializer(_encode_array, when_used="json")
AnyArray = Annotated[
    np.ndarray, BeforeValidator(partial(_decode_array, dtype=None)), _ENCODE
]
Float32Array = Annotated[
    np.ndarray,
    BeforeValidator(partial(_decode_array, dtype=np.dtype(np.float32))),
    _ENCODE,
]
Float64Array = Annotated[
    np.ndarray,
    BeforeValidator(partial(_decode_array, dtype=np.dtype(np.float64))),
    _ENCODE,
]

# Sorted so the file does not depend on set iteration order, which varies with
# the hash seed between processes.
Partition = Annotated[
    frozenset[frozenset[str]],
    PlainSerializer(lambda p: sorted(sorted(g) for g in p), when_used="json"),
]


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        ser_json_inf_nan="constants",
    )


class PoolRecord(_Model):
    n_arcs: int
    probe_to_hole: dict[str, int]
    partition: Partition
    probe_to_arc_idx: dict[str, int]
    arc_centroids_deg: list[float]
    min_ml_gap: float
    x: Float32Array
    x_reduced: Float32Array
    objective: float
    min_clear: float
    min_clear_reduced: float
    fcl: float
    idx: int | None = None


class PoolFile(_Model):
    kind: Literal["phase1_pool"]
    schema_version: Literal[1]
    stage1: int
    stage2: int
    n_spins: int
    max_arcs: int
    max_ppa: int
    minimizer: str
    well: str
    coarse_n: int
    reduced_fine: int
    full_fine: int
    records: list[PoolRecord]


class Perturbation(_Model):
    scale: float
    seed: list[int]


class HandoffRecord(_Model):
    idx: int
    rank: int
    n_arcs: int
    fcl: float
    coverage: float
    pose: Float64Array
    nit: int
    secs: float
    hole: dict[str, int]
    partition: Partition
    probe_to_arc_idx: dict[str, int]
    arc_centroids_deg: list[float]
    min_clear: float | None
    max_g_thread: float | None = None
    pose_in: Float64Array | None = None
    objective_p1: float | None = None
    solver_status: int | None = None
    solver_message: str | None = None
    fcl_strict: bool | None = None
    thread_strict: bool | None = None
    strict_feasible: bool | None = None
    fcl_keep: bool | None = None
    thread_keep: bool | None = None
    kept: bool | None = None
    pose_start: Float64Array | None = None
    perturb: Perturbation | None = None
    # Keyed by slack group or history column; open so a new diagnostic does not
    # fail a write at the end of a long run.
    slack_start: dict[str, AnyArray] | None = None
    slack_end: dict[str, AnyArray] | None = None
    fcl_start: float | None = None
    fcl_pairs_start: list[tuple[str, float]] | None = None
    fcl_pairs_end: list[tuple[str, float]] | None = None
    diag_hist: dict[str, AnyArray] | None = None


class HandoffFile(_Model):
    kind: Literal["phase2_handoff"]
    schema_version: Literal[1]
    config: dict[str, Any]
    all: list[HandoffRecord]
    ranked: list[int]


def check_payload_path(path: str | Path) -> Path:
    """``path`` as a ``Path``, if it names a ``.json`` or ``.json.gz`` file.

    Stages call this before doing any work, so a wrong output path fails at
    startup instead of after the solve.
    """
    path = Path(path)
    if path.suffix != ".json" and path.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError(
            f"{path}: pipeline payloads are .json or .json.gz files; pickled "
            "payloads are no longer read"
        )
    return path


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            if path.suffix == ".gz":
                # mtime=0 so identical payloads give identical files.
                with gzip.GzipFile(
                    fileobj=f, mode="wb", compresslevel=_GZIP_LEVEL, mtime=0
                ) as gz:
                    gz.write(data)
            else:
                f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def _same_value(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_value(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return bool(a == b)


def write_pool(path: str | Path, payload: Phase1PoolPayload) -> None:
    """Validate a Phase-1 pool and write it to ``path``.

    Raises
    ------
    ValueError
        If ``path`` is not ``.json`` or ``.json.gz``, or the payload does not
        match the pool schema.
    """
    path = check_payload_path(path)
    model = PoolFile.model_validate(
        {"kind": "phase1_pool", "schema_version": SCHEMA_VERSION, **payload}
    )
    _write_bytes(path, model.model_dump_json(exclude_unset=True).encode())


def read_pool(path: str | Path) -> Phase1PoolPayload:
    """Read and validate a Phase-1 pool written by ``write_pool``."""
    path = check_payload_path(path)
    try:
        model = PoolFile.model_validate_json(_read_bytes(path))
    except ValidationError as e:
        raise ValueError(f"{path}: not a valid Phase-1 pool") from e
    out = model.model_dump(exclude_unset=True, exclude={"kind", "schema_version"})
    return cast(Phase1PoolPayload, out)


def write_handoff(path: str | Path, payload: Phase2HandoffPayload) -> None:
    """Validate a Phase-2 handoff and write it to ``path``.

    Raises
    ------
    ValueError
        If ``path`` is not ``.json`` or ``.json.gz``, the payload does not match
        the handoff schema, candidate ids in ``all`` repeat, or a ``ranked``
        record differs from the entry of ``all`` with its id.
    """
    path = check_payload_path(path)
    by_idx: dict[int, object] = {}
    for rec in payload["all"]:
        if rec["idx"] in by_idx:
            raise ValueError(f"{path}: candidate {rec['idx']} repeats in 'all'")
        by_idx[rec["idx"]] = rec
    for rec in payload["ranked"]:
        match = by_idx.get(rec["idx"])
        if match is None or not (match is rec or _same_value(match, rec)):
            raise ValueError(
                f"{path}: ranked candidate {rec['idx']} is not an entry of 'all'"
            )
    model = HandoffFile.model_validate(
        {
            "kind": "phase2_handoff",
            "schema_version": SCHEMA_VERSION,
            "config": payload["config"],
            "all": payload["all"],
            "ranked": [rec["idx"] for rec in payload["ranked"]],
        }
    )
    _write_bytes(path, model.model_dump_json(exclude_unset=True).encode())


def read_handoff(path: str | Path) -> Phase2HandoffPayload:
    """Read and validate a Phase-2 handoff written by ``write_handoff``.

    ``ranked`` holds the same record objects as ``all``, as it does in the
    payload the stage produced.
    """
    path = check_payload_path(path)
    try:
        model = HandoffFile.model_validate_json(_read_bytes(path))
    except ValidationError as e:
        raise ValueError(f"{path}: not a valid Phase-2 handoff") from e
    records = [rec.model_dump(exclude_unset=True) for rec in model.all]
    by_idx = {rec["idx"]: rec for rec in records}
    missing = [i for i in model.ranked if i not in by_idx]
    if missing:
        raise ValueError(f"{path}: ranked candidates {missing} are not in 'all'")
    return cast(
        Phase2HandoffPayload,
        {
            "ranked": [by_idx[i] for i in model.ranked],
            "all": records,
            "config": model.config,
        },
    )
