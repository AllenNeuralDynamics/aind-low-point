"""Phase-2 process workers each take their own CPUs on the CPU platform.

XLA's CPU pool starts a thread per CPU the process may run on, whatever the
BLAS and OpenMP variables say, so an unpinned pool of N workers runs N times
the core count in compute threads.
"""

from __future__ import annotations

import multiprocessing

import pytest


def test_slices_are_disjoint_while_cpus_last_then_wrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aind_rutter.optimization.pipeline import phase2

    monkeypatch.setattr(phase2._os, "sched_getaffinity", lambda pid: {9, 3, 4, 7, 8})
    assert phase2._cpu_slices(2, 2) == [[3, 4], [7, 8]]
    assert phase2._cpu_slices(3, 2) == [[3, 4], [7, 8], [9, 3]]


def test_each_started_worker_takes_the_next_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aind_rutter.optimization.pipeline import phase2

    pinned: list[list[int]] = []
    monkeypatch.setattr(
        phase2._os, "sched_setaffinity", lambda pid, cpus: pinned.append(cpus)
    )
    slices = [[0, 1], [2, 3]]
    started = multiprocessing.get_context("spawn").Value("i", 0)
    for _ in range(3):
        phase2._pin_worker(slices, started)
    # The third is a replacement worker: it wraps rather than waiting.
    assert pinned == [[0, 1], [2, 3], [0, 1]]
