"""`phase2_ipopt.run` end to end on CPU, over a synthetic subject.

Nothing covered the Phase-2 stage above the level of its helpers: the worker
setup, the per-candidate problem build, the solve, the FCL gate, the keep bands
and the MMR ranking only ran together in an overnight job against meshes under
/mnt. One candidate over the synthetic subject exercises all of them in about
fifteen seconds, most of it the JAX trace and compile.

The seed pose is deliberately FCL-clear, so the keep and ranking paths run too.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from aind_rutter.optimization.pipeline.contracts import (
    Phase2HandoffPayload,
    Phase2InputRecord,
)
from tests.synthetic_subject import SyntheticSubject, write_subject

# x = (arc_aps, then (ml, sx, sy, off_R, off_A, depth) per probe). The unit spin
# vector (1, 0) is zero spin, and every offset starts at zero.
SEED_ARC_AP_DEG = 0.0
PROBE_SEED = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
PROBE_TO_HOLE = {"P1": 1, "P2": 2}
N_PROBES = len(PROBE_TO_HOLE)


@pytest.fixture(scope="module")
def subject(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSubject:
    return write_subject(tmp_path_factory.mktemp("subject"))


@pytest.fixture(scope="module")
def caches(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Keep the SDF and compile caches out of the working tree."""
    cache = tmp_path_factory.mktemp("caches")
    patch = pytest.MonkeyPatch()
    patch.setenv("AIND_LOW_POINT_CACHE_DIR", str(cache / "sdf"))
    patch.setenv("JAX_CACHE_DIR", str(cache / "jax"))
    yield cache
    patch.undo()


@pytest.fixture(scope="module")
def settings(subject: SyntheticSubject, caches: Path):
    from aind_rutter.optimization.pipeline.settings import Phase2Settings

    return Phase2Settings(
        config=subject.config,
        holes=subject.holes,
        poses=caches / "pool.json.gz",
        out=caches / "handoff.json",
        workers=1,
        well="thin",
        p2_iter=20,
    )


def _record(rank: int = 0, idx: int = 0) -> Phase2InputRecord:
    pose = np.array([SEED_ARC_AP_DEG, *PROBE_SEED * N_PROBES], dtype=float)
    record: Phase2InputRecord = {
        "idx": idx,
        "n_arcs": 1,
        "pose": pose,
        "probe_to_hole": dict(PROBE_TO_HOLE),
        "partition": frozenset(frozenset({name}) for name in PROBE_TO_HOLE),
        "probe_to_arc_idx": dict.fromkeys(PROBE_TO_HOLE, 0),
        "arc_centroids_deg": [SEED_ARC_AP_DEG],
        "min_clear": -0.1,
        "rank": rank,
    }
    return record


@pytest.fixture(scope="module")
def solved(
    settings,
) -> tuple[Phase2HandoffPayload, list[tuple[int, int]], list[str]]:
    """Run the stage once; every test below reads the same payload."""
    from aind_rutter.optimization.pipeline import phase2_ipopt

    seen: list[tuple[int, int]] = []
    messages: list[str] = []
    payload = phase2_ipopt.run(
        [_record()],
        settings,
        on_result=lambda k, n, _result: seen.append((k, n)),
        log=messages.append,
    )
    return payload, seen, messages


def test_the_payload_has_the_three_handoff_sections(solved) -> None:
    payload, _, _ = solved
    assert set(payload) == {"ranked", "all", "config"}
    assert len(payload["all"]) == 1


def test_every_result_carries_the_record_contract(solved) -> None:
    payload, _, _ = solved
    (result,) = payload["all"]
    required = {
        "idx",
        "rank",
        "n_arcs",
        "fcl",
        "coverage",
        "pose",
        "nit",
        "secs",
        "hole",
        "partition",
        "probe_to_arc_idx",
        "arc_centroids_deg",
        "min_clear",
        "max_g_thread",
    }
    assert required <= set(result)
    assert result["idx"] == 0
    assert result["hole"] == PROBE_TO_HOLE
    assert result["n_arcs"] == 1


def test_the_solved_pose_keeps_its_shape_and_stays_in_bounds(solved) -> None:
    from aind_rutter.optimization.pipeline.phase1_geometry import phase1_bounds

    payload, _, _ = solved
    pose = np.asarray(payload["all"][0]["pose"], dtype=float)
    assert pose.shape == (1 + 6 * N_PROBES,)
    assert np.isfinite(pose).all()
    bounds = np.asarray(phase1_bounds(1, N_PROBES, head_pitch_deg=14.0), dtype=float)
    assert (pose >= bounds[:, 0] - 1e-6).all()
    assert (pose <= bounds[:, 1] + 1e-6).all()


def test_the_feasible_seed_stays_feasible_and_is_kept(solved) -> None:
    """The seed threads both bores clear of every body, so the bands admit it."""
    payload, _, _ = solved
    (result,) = payload["all"]
    assert result["fcl"] > 0.0
    assert result["kept"] is True
    assert [r["idx"] for r in payload["ranked"]] == [0]


def test_the_classification_flags_follow_the_bands(solved, settings) -> None:
    payload, _, _ = solved
    (result,) = payload["all"]
    assert result["fcl_keep"] == (result["fcl"] >= -settings.fcl_tol)
    assert result["thread_keep"] == (result["max_g_thread"] <= settings.g_tol)
    assert result["kept"] == (result["fcl_keep"] and result["thread_keep"])
    assert result["strict_feasible"] == (
        result["fcl_strict"] and result["thread_strict"]
    )


def test_the_provenance_records_the_settings_that_ran(solved, settings) -> None:
    payload, _, _ = solved
    config = payload["config"]
    assert config["settings"]["p2_iter"] == settings.p2_iter
    assert config["fcl_tol"] == settings.fcl_tol
    assert config["g_tol"] == settings.g_tol


def test_the_callbacks_report_each_candidate_and_the_progress(solved) -> None:
    _, seen, messages = solved
    assert seen == [(1, 1)]
    assert any("Parallel Phase 2" in message for message in messages)


def test_the_payload_round_trips_through_the_handoff_file(solved, tmp_path) -> None:
    from aind_rutter.optimization.pipeline.payloads import read_handoff, write_handoff

    payload, _, _ = solved
    path = tmp_path / "handoff.json"
    write_handoff(path, payload)
    reloaded = read_handoff(path)
    assert [r["idx"] for r in reloaded["all"]] == [r["idx"] for r in payload["all"]]
    np.testing.assert_array_equal(
        np.asarray(reloaded["all"][0]["pose"]), np.asarray(payload["all"][0]["pose"])
    )
    assert reloaded["config"]["fcl_tol"] == payload["config"]["fcl_tol"]
