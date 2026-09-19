"""A background worker that only ever runs the latest job handed to it."""

from __future__ import annotations

import threading
from typing import Any, Callable, List

from aind_rutter.domain.plan import PlanningState


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

    def __call__(self, plan: PlanningState, changed_ids: List[str]) -> None:
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
