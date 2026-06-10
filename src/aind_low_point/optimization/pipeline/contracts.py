"""Lightweight contracts for optimizer pipeline payloads.

The pipeline still serializes plain Python objects with pickle.  These types make
those boundaries explicit without adding runtime validation or changing payload
shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Protocol, TypedDict

from numpy.typing import NDArray

from aind_low_point.optimization.atlas import Atlas

Array = NDArray[Any]
ProbeName = str
ProbeToHole = dict[ProbeName, int]
ProbeToArcIdx = dict[ProbeName, int]
Partition = frozenset[frozenset[ProbeName]]
SeedMap = dict[ProbeName, float]


@dataclass(frozen=True, slots=True)
class AtlasCachePayload:
    """Normalized visibility-atlas cache payload."""

    atlas: Atlas
    probe_names: tuple[ProbeName, ...]
    head_pitch_deg: float


class SeedResult(NamedTuple):
    """Joint AP/ML/spin seed for one enumerated candidate."""

    arc_aps: list[float]
    ml_seed: SeedMap
    spin_seed: SeedMap
    min_ml_gap: float


class EnumeratorCandidate(TypedDict):
    """Cheap discrete decision emitted by the MRV enumerator."""

    probe_to_hole: ProbeToHole
    partition: Partition
    arc_aps: list[float]


class Phase1PoolRecordRequired(TypedDict):
    """Record saved by phase-1 pool optimization."""

    n_arcs: int
    probe_to_hole: ProbeToHole
    partition: Partition
    probe_to_arc_idx: ProbeToArcIdx
    arc_centroids_deg: list[float]
    min_ml_gap: float
    x: Array
    x_reduced: Array
    objective: float
    min_clear: float
    min_clear_reduced: float
    fcl: float


class Phase1PoolRecord(Phase1PoolRecordRequired, total=False):
    """Phase-1 record with optional legacy/provenance keys."""

    idx: int


class Phase1PoolPayload(TypedDict):
    """Pickled phase-1 pool output."""

    records: list[Phase1PoolRecord]
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


class Phase2InputRecord(TypedDict):
    """Normalized phase-1 record handed to Phase 2."""

    idx: int
    n_arcs: int
    pose: Array
    probe_to_hole: ProbeToHole
    partition: Partition
    probe_to_arc_idx: ProbeToArcIdx
    arc_centroids_deg: list[float]
    min_clear: float | None
    rank: int


class Phase2ResultRecord(TypedDict):
    """Record emitted by the Phase 2 solver and consumed by plan emission."""

    idx: int
    rank: int
    n_arcs: int
    fcl: float
    max_g_thread: float
    coverage: float
    pose: Array
    nit: int
    secs: float
    hole: ProbeToHole
    partition: Partition
    probe_to_arc_idx: ProbeToArcIdx
    arc_centroids_deg: list[float]
    min_clear: float | None


class Phase2HandoffPayload(TypedDict):
    """Pickled handoff consumed by ``pipeline.emit``."""

    ranked: list[Phase2ResultRecord]
    all: list[Phase2ResultRecord]
    config: dict[str, object]


class MRVArcAssignment(SimpleNamespace):
    """Attribute contract for the phase-1 MRV arc assignment stand-in."""

    probe_to_arc_idx: ProbeToArcIdx
    arc_centroids_deg: list[float]


class MRVHoleAssignment(SimpleNamespace):
    """Attribute contract for the phase-1 MRV hole assignment stand-in."""

    probe_to_hole: ProbeToHole


class SpinRestoreFn(Protocol):
    """Callable returned by spin-restore factories."""

    def __call__(self, y: Any, *varying: Any) -> Any: ...


class SpinRestoreWithLosses(SpinRestoreFn, Protocol):
    """Spin restore callable that exposes single-candidate loss diagnostics."""

    def spin_losses(self, y: Any, i: int, *varying: Any) -> Any: ...


BatchedObjectiveFn = Callable[..., Any]
BatchedGradientFn = Callable[..., Any]
ArglistBuilder = Callable[[list[Any]], list[Any]]
AdamRunnerFactory = Callable[..., Callable[..., Any]]
Phase1ObjectiveFns = tuple[BatchedObjectiveFn, BatchedGradientFn]
Phase1ChunkedFns = tuple[
    BatchedObjectiveFn,
    BatchedGradientFn,
    ArglistBuilder,
    AdamRunnerFactory,
    AdamRunnerFactory,
]


class Phase2Problem(TypedDict):
    """Scipy-facing callable bundle returned by ``make_phase2``."""

    fun: Callable[[Array], float]
    jac: Callable[[Array], Array]
    hess: Callable[[Array], Array] | None
    hessp: Callable[[Array, Array], Array] | None
    constraints: list[dict[str, object]]
    constraints_nlc: list[Any]
