"""Which solved candidates are kept, and what provenance travels with them."""

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from aind_rutter.optimization.pipeline.handoff import (
    classify_results,
    handoff_config,
)
from aind_rutter.optimization.pipeline.settings import Phase2Settings

# Every field carries an environment alias, so ambient values would otherwise
# decide what these assertions compare against.
_ENV = (
    "RUTTER_CONFIG",
    "CONFIG",
    "RUTTER_HOLES",
    "HOLES",
    "WELL",
    "TOPK",
    "SELECT_BY",
    "FCL_TOL",
    "G_TOL",
    "MMR_LAMBDA",
    "RANKS_FILE",
    "IP_TOL",
    "IP_ACC_TOL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _res(fcl: float, **fields: Any) -> Any:
    return cast(Any, {"fcl": fcl, **fields})


def test_a_result_inside_both_bands_is_kept() -> None:
    rows = [_res(0.5, max_g_thread=-0.3)]
    assert classify_results(rows, Phase2Settings()) == rows
    assert rows[0]["kept"] and rows[0]["strict_feasible"]


def test_each_axis_drops_a_result_on_its_own() -> None:
    s = Phase2Settings(fcl_tol=0.2, g_tol=0.2)
    bad_fcl = _res(-0.5, max_g_thread=-0.3)
    bad_g = _res(0.5, max_g_thread=0.9)
    assert classify_results([bad_fcl, bad_g], s) == []
    assert not bad_fcl["fcl_keep"] and bad_fcl["thread_keep"]
    assert bad_g["fcl_keep"] and not bad_g["thread_keep"]


def test_the_keep_band_admits_what_the_strict_flags_reject() -> None:
    # Mildly infeasible on both axes: inside the keep band, outside strict.
    row = _res(-0.1, max_g_thread=0.1)
    assert classify_results([row], Phase2Settings(fcl_tol=0.2, g_tol=0.2)) == [row]
    assert row["kept"]
    assert not row["fcl_strict"] and not row["thread_strict"]
    assert not row["strict_feasible"]


def test_a_missing_or_nan_threading_g_is_never_feasible() -> None:
    missing = _res(0.5)
    nan_g = _res(0.5, max_g_thread=float("nan"))
    assert classify_results([missing, nan_g], Phase2Settings()) == []
    for row in (missing, nan_g):
        assert not row["thread_keep"] and not row["thread_strict"]
        assert row["fcl_keep"], "the FCL axis is judged independently"


def test_the_bands_come_from_settings_not_a_fixed_threshold() -> None:
    row = _res(-0.4, max_g_thread=0.4)
    assert classify_results([row], Phase2Settings(fcl_tol=0.2, g_tol=0.2)) == []
    assert classify_results([row], Phase2Settings(fcl_tol=0.5, g_tol=0.5)) == [row]


def test_provenance_carries_every_field_under_its_own_name() -> None:
    s = Phase2Settings(topk=7, ip_tol=1e-3)
    cfg = handoff_config(s)
    assert cfg["settings"] == s.model_dump(mode="json")
    assert cast(Any, cfg["settings"])["topk"] == 7


def test_the_flat_keys_repeat_the_settings_they_name() -> None:
    s = Phase2Settings(topk=7, ip_acc_tol=3.0, fcl_tol=0.25, minclear=0.15)
    cfg = handoff_config(s)
    assert cfg["topk"] == 7
    assert cfg["acc_tol"] == 3.0, "IP_ACC_TOL keeps its earlier handoff spelling"
    assert cfg["fcl_tol"] == 0.25
    assert cfg["minclear"] == 0.15
    assert cfg["subject_config"] == str(s.config)


def test_an_unset_ranks_file_records_an_empty_string() -> None:
    assert handoff_config(Phase2Settings())["ranks_file"] == ""
    path = "scratch/ranks.txt"
    assert handoff_config(Phase2Settings(ranks_file=path))["ranks_file"] == path


def test_the_provenance_survives_a_round_trip_through_json() -> None:
    import json

    # The payload is pickled, but a settings dump that cannot serialise would
    # mean a Path or enum leaked into it.
    json.dumps(handoff_config(Phase2Settings()))


def test_importing_the_module_does_not_initialise_jax() -> None:
    import subprocess
    import sys

    code = (
        "import sys, aind_rutter.optimization.pipeline.handoff;"
        " print('jax' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.stdout.strip() == "False"
