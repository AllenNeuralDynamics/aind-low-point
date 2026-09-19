"""The plan store: one planning state, edited by command, with subscribers."""

from __future__ import annotations

from typing import Callable, List

from aind_rutter.domain.commands import PlanningCommand, apply_planning_command
from aind_rutter.domain.plan import PlanningState

Subscriber = Callable[[PlanningState, List[str]], None]


class PlanStore:
    def __init__(self, initial: PlanningState):
        self._state = initial
        self._subs: List[Subscriber] = []

    @property
    def state(self) -> PlanningState:
        return self._state

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subs.append(fn)

        def _unsub():
            try:
                self._subs.remove(fn)
            except ValueError:
                pass

        return _unsub

    def _notify(self, changed: List[str]) -> None:
        for fn in list(self._subs):
            fn(self._state, changed)

    def dispatch(self, cmd: PlanningCommand) -> None:
        changed = apply_planning_command(self._state, cmd)
        self._notify(changed)
