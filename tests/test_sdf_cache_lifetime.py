"""The two SDF caches key on ``id``, which is only safe while the keyed object
is alive.

An entry outliving its object is read by whatever lands on that address next,
so a later probe would be handed the previous probe's grids. Both caches now
drop their entry as the object is collected, which also bounds them.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from aind_rutter.optimization.clearance.voxel_sdf import ProbeSDF


def _sdf() -> ProbeSDF:
    return ProbeSDF(
        grid=np.zeros((4, 4, 4), np.float32),
        origin=np.zeros(3, np.float32),
        spacing=np.ones(3, np.float32),
        surface_points=np.zeros((8, 3), np.float32),
        shank_centers=np.zeros((1, 3), np.float32),
        shank_halves=np.ones((1, 3), np.float32),
    )


def test_the_payload_cache_drops_an_entry_with_its_probe_sdf() -> None:
    from aind_rutter.optimization.objectives import statics

    before = len(statics._SDF_JNP_CACHE)
    sdf = _sdf()
    payload = statics._sdf_jnp_payload(sdf)
    assert statics._SDF_JNP_CACHE[id(sdf)] is payload
    assert len(statics._SDF_JNP_CACHE) == before + 1

    del sdf, payload
    gc.collect()
    assert len(statics._SDF_JNP_CACHE) == before


def test_the_same_probe_sdf_is_converted_once() -> None:
    from aind_rutter.optimization.objectives import statics

    sdf = _sdf()
    assert statics._sdf_jnp_payload(sdf) is statics._sdf_jnp_payload(sdf)


def test_the_payload_can_be_weakly_referenced() -> None:
    """A plain dict cannot, which is why the packed-table cache could not tell
    that the payloads its key names had gone."""
    import weakref

    from aind_rutter.optimization.objectives import statics

    sdf = _sdf()
    payload = statics._sdf_jnp_payload(sdf)
    assert weakref.ref(payload)() is payload
    assert payload["grid"].shape == (4, 4, 4)
    assert isinstance(payload, dict)


def test_the_packed_table_cache_drops_entries_with_their_payloads() -> None:
    from aind_rutter.optimization.objectives import soft, statics

    key_payload = statics._sdf_jnp_payload(_sdf())
    other = statics._sdf_jnp_payload(_sdf())
    key = ((id(key_payload), id(other)), True)
    soft._SDF_PACK_CACHE[key] = {"packed": "table"}

    soft._drop_pack_entries_for(id(other))
    assert key not in soft._SDF_PACK_CACHE


def test_dropping_leaves_unrelated_entries_alone() -> None:
    from aind_rutter.optimization.objectives import soft

    keep = ((111111, 222222), True)
    soft._SDF_PACK_CACHE[keep] = {"packed": "keep"}
    try:
        soft._drop_pack_entries_for(999999)
        assert keep in soft._SDF_PACK_CACHE
    finally:
        soft._SDF_PACK_CACHE.pop(keep, None)


@pytest.mark.parametrize("n", [2, 5])
def test_neither_cache_grows_without_bound(n: int) -> None:
    """Every ProbeSDF in a loop used to leave its grids on device for good."""
    from aind_rutter.optimization.objectives import statics

    before = len(statics._SDF_JNP_CACHE)
    for _ in range(n):
        statics._sdf_jnp_payload(_sdf())
        gc.collect()
    gc.collect()
    assert len(statics._SDF_JNP_CACHE) == before
