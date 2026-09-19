"""The pool and handoff files read back exactly what was written."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast, get_type_hints

import numpy as np
import pytest

from aind_rutter.optimization.assignment.atlas import Atlas, AtlasEntry, PoseAnchor
from aind_rutter.optimization.pipeline import records as contracts
from aind_rutter.optimization.pipeline.payloads import (
    HandoffFile,
    HandoffRecord,
    PoolFile,
    PoolRecord,
    read_atlas_cache,
    read_handoff,
    read_pool,
    read_seed_cache,
    write_atlas_cache,
    write_handoff,
    write_pool,
    write_seed_cache,
)
from aind_rutter.optimization.pipeline.records import AtlasCachePayload

_FILE_ONLY = {"kind", "schema_version"}


def _pool_record(idx: int | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "n_arcs": 2,
        "probe_to_hole": {"VM": 7, "MD": 3},
        "partition": frozenset({frozenset({"VM"}), frozenset({"MD"})}),
        "probe_to_arc_idx": {"VM": 0, "MD": 1},
        "arc_centroids_deg": [-12.5, 30.25],
        "min_ml_gap": 16.0,
        "x": np.linspace(-1, 1, 14, dtype=np.float32),
        "x_reduced": np.arange(8, dtype=np.float32) / 3,
        "objective": 1.0 / 3.0,
        "min_clear": -0.125,
        "min_clear_reduced": 0.1,
        # FCL runs only on the top candidates; the rest carry NaN.
        "fcl": float("nan"),
    }
    if idx is not None:
        rec["idx"] = idx
    return rec


def _pool(records: list[dict[str, Any]]) -> Any:
    return {
        "records": records,
        "stage1": 450,
        "stage2": 450,
        "n_spins": 16,
        "max_arcs": 3,
        "max_ppa": 4,
        "minimizer": "rprop",
        "well": "thick",
        "coarse_n": 1000,
        "reduced_fine": 50,
        "full_fine": 50,
    }


def _handoff_record(idx: int, *, diag: bool) -> dict[str, Any]:
    rng = np.random.default_rng(idx)
    rec: dict[str, Any] = {
        "idx": idx,
        "rank": idx + 100,
        "n_arcs": 2,
        "fcl": 0.0125,
        "max_g_thread": float("nan"),
        "coverage": 1.5,
        "pose": rng.standard_normal(14),
        "nit": 42,
        "secs": 12.5,
        "hole": {"VM": 7, "MD": 3},
        "partition": frozenset({frozenset({"VM", "MD"})}),
        "probe_to_arc_idx": {"VM": 0, "MD": 0},
        "arc_centroids_deg": [4.0],
        "min_clear": None,
        "pose_in": rng.standard_normal(14),
        "objective_p1": -2.0,
        "solver_status": 1,
        "solver_message": "Solved To Acceptable Level.",
        "fcl_strict": True,
        "thread_strict": False,
        "strict_feasible": False,
        "fcl_keep": True,
        "thread_keep": False,
        "kept": False,
    }
    if diag:
        rec |= {
            "pose_start": rng.standard_normal(14),
            "perturb": None,
            "slack_start": {
                "thread": rng.standard_normal((3, 2, 4)).astype(np.float32),
                "arc_sep": np.zeros(0, np.float32),
                # Empty along the first axis: nested lists would lose the shape.
                "probe_fixture": np.zeros((0, 2, 3), np.float32),
            },
            "slack_end": {"thread": np.full((3, 2, 4), np.inf, np.float32)},
            "fcl_start": -1.0,
            "fcl_pairs_start": [("VM|well", -0.5)],
            "fcl_pairs_end": [],
            "diag_hist": {
                "iter": np.arange(5, dtype=np.int32),
                "obj": rng.standard_normal(5),
                "restoration": np.array([0, 0, 1, 0, 0], np.int8),
                "group_min": rng.standard_normal((5, 6)).astype(np.float32),
            },
        }
    return rec


def _handoff(n: int = 3, *, diag: bool = True) -> Any:
    records = [_handoff_record(i, diag=diag) for i in range(n)]
    return {
        "all": records,
        "ranked": [records[2], records[0]],
        "config": {"fcl_tol": 0.2, "settings": {"topk": 8, "ranks_file": None}},
    }


def _same(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray):
        return (
            isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
        )
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_same(a[k], b[k]) for k in a)
        )
    if isinstance(a, list | tuple):
        return (
            type(a) is type(b)
            and len(a) == len(cast(Any, b))
            and all(_same(x, y) for x, y in zip(a, cast(Any, b)))
        )
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_a_pool_reads_back_exactly(tmp_path: Path, suffix: str) -> None:
    payload = _pool([_pool_record(), _pool_record(idx=9)])
    path = tmp_path / f"pool{suffix}"
    write_pool(path, payload)
    back = read_pool(path)
    assert _same(dict(payload), dict(back))
    assert "idx" not in back["records"][0], "an absent optional key stays absent"


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_a_handoff_reads_back_exactly(tmp_path: Path, suffix: str) -> None:
    payload = _handoff()
    path = tmp_path / f"handoff{suffix}"
    write_handoff(path, payload)
    back = read_handoff(path)
    assert _same(payload["all"], back["all"])
    assert _same(payload["ranked"], back["ranked"])
    assert _same(payload["config"], back["config"])


def test_ranked_records_are_the_same_objects_as_entries_of_all(tmp_path: Path) -> None:
    path = tmp_path / "handoff.json"
    write_handoff(path, _handoff())
    back = read_handoff(path)
    assert [r["idx"] for r in back["ranked"]] == [2, 0]
    assert back["ranked"][0] is back["all"][2]


def test_a_handoff_without_diagnostics_omits_those_keys(tmp_path: Path) -> None:
    path = tmp_path / "handoff.json"
    write_handoff(path, _handoff(diag=False))
    assert "diag_hist" not in read_handoff(path)["all"][0]


def test_float_bit_patterns_survive_including_nan_inf_and_subnormals(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    f64 = rng.integers(0, 2**63, 4000, dtype=np.uint64).view(np.float64)
    f32 = rng.integers(0, 2**31, 4000, dtype=np.uint32).view(np.float32)
    specials = [np.nan, np.inf, -np.inf, -0.0, 5e-324, 1.7976931348623157e308]
    rec = _handoff_record(0, diag=True)
    rec["pose"] = np.concatenate([f64, specials])
    rec["diag_hist"] = {"f32": np.concatenate([f32, np.array(specials, np.float32)])}
    path = tmp_path / "handoff.json.gz"
    write_handoff(path, {"all": [rec], "ranked": [], "config": {}})
    back = read_handoff(path)["all"][0]
    finite = ~np.isnan(rec["pose"])
    assert np.array_equal(
        back["pose"][finite].view(np.uint64), rec["pose"][finite].view(np.uint64)
    ), "non-NaN float64 values are bitwise identical, sign of zero included"
    assert np.isnan(back["pose"][~finite]).all()
    assert _same(rec["diag_hist"], back["diag_hist"])


def test_identical_payloads_give_identical_files(tmp_path: Path) -> None:
    # Same partition built in different orders iterates differently in memory.
    a = _handoff(diag=False)
    b = _handoff(diag=False)
    for rec in b["all"]:
        rec["partition"] = frozenset(
            frozenset(sorted(g, reverse=True)) for g in reversed(list(rec["partition"]))
        )
    write_handoff(tmp_path / "a.json.gz", a)
    write_handoff(tmp_path / "b.json.gz", b)
    assert (tmp_path / "a.json.gz").read_bytes() == (
        tmp_path / "b.json.gz"
    ).read_bytes()


def test_a_producer_that_changes_an_array_dtype_is_rejected(tmp_path: Path) -> None:
    rec = _pool_record()
    rec["x"] = rec["x"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        write_pool(tmp_path / "pool.json", _pool([rec]))


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    rec = _handoff_record(0, diag=False) | {"surprise": 1}
    with pytest.raises(ValueError, match="surprise"):
        write_handoff(tmp_path / "h.json", {"all": [rec], "ranked": [], "config": {}})


def test_a_pool_is_not_accepted_as_a_handoff(tmp_path: Path) -> None:
    path = tmp_path / "pool.json"
    write_pool(path, _pool([_pool_record()]))
    with pytest.raises(ValueError, match="not a valid Phase-2 handoff"):
        read_handoff(path)


@pytest.mark.parametrize("name", ["handoff.pkl", "handoff", "handoff.gz"])
def test_only_json_paths_are_accepted(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match=r"\.json"):
        write_handoff(tmp_path / name, _handoff())
    with pytest.raises(ValueError, match=r"\.json"):
        read_handoff(tmp_path / name)


def test_a_ranked_record_must_be_an_entry_of_all(tmp_path: Path) -> None:
    payload = _handoff()
    payload["ranked"] = [_handoff_record(7, diag=False)]
    with pytest.raises(ValueError, match="ranked candidate 7"):
        write_handoff(tmp_path / "h.json", payload)
    changed = _handoff()
    changed["ranked"] = [dict(changed["all"][1], coverage=99.0)]
    with pytest.raises(ValueError, match="ranked candidate 1"):
        write_handoff(tmp_path / "h.json", changed)


def test_candidate_ids_in_all_must_be_unique(tmp_path: Path) -> None:
    payload = _handoff()
    payload["all"].append(_handoff_record(0, diag=False))
    with pytest.raises(ValueError, match="repeats"):
        write_handoff(tmp_path / "h.json", payload)


def test_a_failed_write_leaves_no_partial_file(tmp_path: Path) -> None:
    rec = _pool_record()
    rec["x"] = rec["x"].astype(np.float64)
    with pytest.raises(ValueError):
        write_pool(tmp_path / "pool.json", _pool([rec]))
    assert list(tmp_path.iterdir()) == []


def _keys(typed_dict: type) -> tuple[set[str], set[str]]:
    return set(typed_dict.__required_keys__), set(typed_dict.__optional_keys__)


def _fields(model: Any, drop: set[str]) -> tuple[set[str], set[str]]:
    fields = {k: f for k, f in model.model_fields.items() if k not in drop}
    required = {k for k, f in fields.items() if f.is_required()}
    return required, set(fields) - required


@pytest.mark.parametrize(
    ("typed_dict", "model", "drop"),
    [
        (contracts.Phase1PoolRecord, PoolRecord, set()),
        (contracts.Phase1PoolPayload, PoolFile, _FILE_ONLY),
        (contracts.Phase2ResultRecord, HandoffRecord, set()),
        (contracts.Phase2HandoffPayload, HandoffFile, _FILE_ONLY),
    ],
)
def test_file_models_match_the_in_memory_contracts(
    typed_dict: type, model: type, drop: set[str]
) -> None:
    assert _fields(model, drop) == _keys(typed_dict)
    get_type_hints(typed_dict)  # the contract itself still resolves


def _atlas() -> AtlasCachePayload:
    anchors = (
        PoseAnchor(-12.0, 3.5, 90.0, 0.1, -0.2, 4.25, float("nan"), 0.0),
        PoseAnchor(-10.0, 3.0, 270.0, 0.0, 0.0, 4.5, -0.3, 0.125),
    )
    entries = {
        ("VM", 7): AtlasEntry("VM", 7, -12.0, -10.0, anchors),
        # No target-valid pose for this pair.
        ("MD", 3): AtlasEntry("MD", 3, None, None, ()),
    }
    return AtlasCachePayload(Atlas(entries, ("VM", "MD"), (3, 7)), ("VM", "MD"), 14.0)


def test_an_atlas_cache_reads_back_equal(tmp_path: Path) -> None:
    payload = _atlas()
    path = tmp_path / "atlas.json.gz"
    write_atlas_cache(path, payload)
    back = read_atlas_cache(path)
    assert back.probe_names == payload.probe_names
    assert back.head_pitch_deg == payload.head_pitch_deg
    assert list(back.atlas.entries) == list(payload.atlas.entries)
    assert back.atlas.entries[("MD", 3)] == payload.atlas.entries[("MD", 3)]
    got = back.atlas.entries[("VM", 7)].anchors
    want = payload.atlas.entries[("VM", 7)].anchors
    assert got[1] == want[1]
    assert np.isnan(got[0].threading_max_g) and got[0].spin_deg == want[0].spin_deg


def test_an_atlas_cache_for_other_anchor_fields_is_unreadable(tmp_path: Path) -> None:
    import json

    path = tmp_path / "atlas.json"
    write_atlas_cache(path, _atlas())
    doc = json.loads(path.read_text())
    doc["anchor_fields"] = list(reversed(doc["anchor_fields"]))
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="anchors stored as"):
        read_atlas_cache(path)


def test_a_mis_keyed_atlas_entry_is_rejected(tmp_path: Path) -> None:
    payload = _atlas()
    entry = payload.atlas.entries.pop(("MD", 3))
    payload.atlas.entries[("MD", 4)] = entry
    with pytest.raises(ValueError, match="keyed"):
        write_atlas_cache(tmp_path / "atlas.json", payload)


def _seed(n_arcs: int, ml: float) -> dict[str, Any]:
    return {
        "n_arcs": n_arcs,
        "probe_to_hole": {"VM": 7, "MD": 3},
        "partition": frozenset({frozenset({"VM"}), frozenset({"MD"})}),
        "probe_to_arc_idx": {"MD": 0, "VM": 1},
        "arc_centroids_deg": [-20.0, 16.5],
        "ml_seed": {"VM": ml, "MD": -ml},
        "spin_seed": {"VM": 0.0, "MD": 180.0},
        "min_ml_gap": 16.0,
    }


def test_a_seed_cache_keeps_groups_and_candidates_in_order(tmp_path: Path) -> None:
    groups = {3: [_seed(3, 1.5), _seed(3, 2.5)], 1: [_seed(1, 0.25)]}
    path = tmp_path / "seeds.json.gz"
    write_seed_cache(path, groups)
    back = read_seed_cache(path)
    assert list(back) == [3, 1], "group order and integer keys survive JSON"
    assert _same(groups[3], back[3]) and _same(groups[1], back[1])


@pytest.mark.parametrize(
    ("name", "content"),
    [("atlas.json.gz", b"not gzip"), ("atlas.json", b'{"kind": "atlas_cache"')],
)
def test_a_corrupt_cache_raises_value_error(
    tmp_path: Path, name: str, content: bytes
) -> None:
    # Callers treat ValueError as a cache miss and rebuild.
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(ValueError):
        read_atlas_cache(path)


def test_importing_the_module_does_not_initialise_jax() -> None:
    code = (
        "import sys, aind_rutter.optimization.pipeline.payloads;"
        " print('jax' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
