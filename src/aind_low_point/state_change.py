"""handling events and broadcasting state changes"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import (
    Callable,
    List,
    Optional,
    Set,
)

from aind_low_point.commands import PlanningCommand, apply_planning_command
from aind_low_point.planning import PlanningState

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
        changed = apply_planning_command(
            self._state, cmd
        )  # your existing pure function
        self._notify(changed)


class DebouncedCoalescer:
    """
    Wrap a (state, changed_ids) sink with a debounce/coalesce buffer.
    Thread-safe; suitable for slider drags calling dispatch() rapidly.
    """

    def __init__(self, sink: Subscriber, interval_ms: int = 16):
        self._sink = sink
        self._interval = interval_ms / 1000.0
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._pending_ids: Set[str] = set()
        self._latest_state: Optional[PlanningState] = None

    def __call__(self, state: PlanningState, changed_ids: List[str]) -> None:
        with self._lock:
            self._latest_state = state
            self._pending_ids.update(changed_ids)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._interval, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            ids = list(self._pending_ids)
            self._pending_ids.clear()
            state = self._latest_state
            self._timer = None
        if state is not None and ids:
            self._sink(state, ids)


@dataclass
class StoreSubscriber:
    store: PlanStore
    on_event: Callable[[PlanningState, List[str]], None]

    def __post_init__(self):
        self._unsubscribe = self.store.subscribe(self.on_event)

    def dispose(self):
        try:
            self._unsubscribe()
        except Exception:
            pass
