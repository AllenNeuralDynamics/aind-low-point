"""handling events and broadcasting state changes"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, List

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
        changed = apply_planning_command(self._state, cmd)
        self._notify(changed)


class AsyncLatestWorker:
    """Latest-only background worker implementing the Subscriber protocol.

    Pre-computation (reading PlanningState) runs on the calling thread.
    The expensive work runs in a dedicated background thread. Results
    are delivered back on the main thread via *post_to_main*.
    """

    def __init__(
        self,
        prepare: Callable[[PlanningState, List[str]], Any],
        work: Callable[[Any], Any],
        deliver: Callable[[Any], None],
        post_to_main: Callable[[Callable[[], None]], None],
    ) -> None:
        self._prepare = prepare
        self._work = work
        self._deliver = deliver
        self._post = post_to_main

        self._lock = threading.Lock()
        self._pending_ids: set[str] = set()
        self._pending_plan: PlanningState | None = None
        self._busy = False

        self._current_request: Any = None
        self._shutdown = False
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="async-collision"
        )
        self._thread.start()

    def __call__(
        self, plan: PlanningState, changed_ids: List[str]
    ) -> None:
        """Store subscriber entry point (main thread)."""
        with self._lock:
            self._pending_ids.update(changed_ids)
            self._pending_plan = plan
            if not self._busy:
                self._submit()

    def _submit(self) -> None:
        """Prepare work on main thread and wake worker. Lock held."""
        plan = self._pending_plan  # type: ignore[arg-type]
        ids = list(self._pending_ids)
        self._pending_ids.clear()
        self._busy = True
        self._current_request = self._prepare(plan, ids)
        self._ready.set()

    def _loop(self) -> None:
        """Worker thread: wait → work → deliver → repeat."""
        while not self._shutdown:
            self._ready.wait()
            self._ready.clear()
            if self._shutdown:
                break
            result = self._work(self._current_request)
            self._post(lambda r=result: self._deliver(r))
            with self._lock:
                if self._pending_ids:
                    # Must prepare on main thread (reads PlanningState)
                    self._post(self._resubmit)
                else:
                    self._busy = False

    def _resubmit(self) -> None:
        """Called on main thread to prepare and submit pending work."""
        with self._lock:
            if self._pending_ids:
                self._submit()
            else:
                self._busy = False

    def shutdown(self) -> None:
        self._shutdown = True
        self._ready.set()
        self._thread.join(timeout=2.0)


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
