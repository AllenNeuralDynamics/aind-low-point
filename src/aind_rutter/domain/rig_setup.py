"""What is mounted on the rig for a plan, and the rework between two plans.

A setup is the arcs in AP order, each holding its probes in slider-ML order.
Probes of one kind are interchangeable, and neither arcs nor the probes on an
arc can pass each other, so going from one setup to the next is an
order-preserving alignment at both levels. Depth and the in-plane offsets are
manipulator travel and are not rework.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from aind_rutter.domain.plan import PlanningState

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Slot:
    """One mounted probe."""

    kind: str
    ml_deg: float
    spin_deg: float


@dataclass(frozen=True, slots=True)
class ArcSetup:
    ap_deg: float
    slots: tuple[Slot, ...]  # by slider ML


RigSetup = tuple[ArcSetup, ...]  # by AP


@dataclass(frozen=True, slots=True)
class SetupChanges:
    """How many of each change a transition takes."""

    arc_moves: int = 0  # an arc kept, slid to a new AP angle
    slider_moves: int = 0  # a probe kept, slid to a new ML angle
    spins: int = 0  # a kept probe turned about its axis
    mounts: int = 0  # a probe put on or taken off, each
    arc_mounts: int = 0  # an arc put on or taken off; its probes count as mounts

    def __add__(self, other: SetupChanges) -> SetupChanges:
        return SetupChanges(
            *(getattr(self, f.name) + getattr(other, f.name) for f in fields(self))
        )


@dataclass(frozen=True, slots=True)
class SetupCosts:
    """The price of each change, in units of sliding one arc."""

    arc_move: float = 1.0
    slider_move: float = 1.0
    spin: float = 1.0
    mount: float = 10.0
    arc_mount: float = 15.0

    def of(self, changes: SetupChanges) -> float:
        return (
            self.arc_move * changes.arc_moves
            + self.slider_move * changes.slider_moves
            + self.spin * changes.spins
            + self.mount * changes.mounts
            + self.arc_mount * changes.arc_mounts
        )


@dataclass(frozen=True, slots=True)
class SetupTolerances:
    """How far an angle may differ before it counts as a change (deg)."""

    ap_deg: float = 0.5
    ml_deg: float = 0.5
    spin_deg: float = 5.0


def rig_setup(state: PlanningState) -> RigSetup:
    """The arcs that carry a probe, and the probes on each."""
    by_arc: dict[str, list[Slot]] = {}
    for plan in state.probes.values():
        if plan.arc_id is not None:
            by_arc.setdefault(plan.arc_id, []).append(
                Slot(plan.kind, float(plan.ml_local), float(plan.spin) % 360.0)
            )
    arcs = [
        ArcSetup(
            state.kinematics.get_arc(arc_id),
            tuple(sorted(slots, key=lambda s: (s.ml_deg, s.kind, s.spin_deg))),
        )
        for arc_id, slots in by_arc.items()
    ]
    return tuple(sorted(arcs, key=lambda a: (a.ap_deg, a.slots)))


def _spin_apart_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _align(
    xs: Sequence[T],
    ys: Sequence[T],
    keep: Callable[[T, T], SetupChanges],
    swap: Callable[[T], SetupChanges],
    costs: SetupCosts,
) -> SetupChanges:
    """Cheapest way to turn ``xs`` into ``ys`` without reordering either.

    ``keep`` is what pairing two items takes, ``swap`` what taking one off or
    putting one on does.
    """
    row = [SetupChanges()]
    for y in ys:
        row.append(row[-1] + swap(y))
    for x in xs:
        above, row = row, [row[0] + swap(x)]
        for j, y in enumerate(ys):
            row.append(
                min(
                    above[j] + keep(x, y),
                    above[j + 1] + swap(x),
                    row[j] + swap(y),
                    key=costs.of,
                )
            )
    return row[-1]


def setup_changes(
    before: RigSetup,
    after: RigSetup,
    costs: SetupCosts = SetupCosts(),
    tolerances: SetupTolerances = SetupTolerances(),
) -> SetupChanges:
    """The cheapest rework that turns one setup into another."""

    def keep_slot(a: Slot, b: Slot) -> SetupChanges:
        if a.kind != b.kind:
            return SetupChanges(mounts=2)
        return SetupChanges(
            slider_moves=int(abs(a.ml_deg - b.ml_deg) > tolerances.ml_deg),
            spins=int(_spin_apart_deg(a.spin_deg, b.spin_deg) > tolerances.spin_deg),
        )

    def keep_arc(a: ArcSetup, b: ArcSetup) -> SetupChanges:
        moved = SetupChanges(
            arc_moves=int(abs(a.ap_deg - b.ap_deg) > tolerances.ap_deg)
        )
        return moved + _align(
            a.slots, b.slots, keep_slot, lambda _: SetupChanges(mounts=1), costs
        )

    def swap_arc(a: ArcSetup) -> SetupChanges:
        return SetupChanges(arc_mounts=1, mounts=len(a.slots))

    return _align(before, after, keep_arc, swap_arc, costs)
