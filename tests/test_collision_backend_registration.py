"""What the FCL backend has to record for a node to be collidable at all.

``rebuild`` and ``sync`` both add objects. When ``sync`` recorded fewer maps
than ``rebuild``, a node added after the initial build — which is what a probe
kind change does — was missing from the group/mask maps, and the filter in
``collide_internal`` reads a missing group as 0, so the node was never reported
as colliding with anything.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import trimesh

from aind_rutter.collision.adapter import ObjSpec, pair_bits
from aind_rutter.collision.fcl import FCLBackend
from aind_rutter.collision.geometry import bvh_from_mesh, rt_to_fcl_transform
from aind_rutter.domain.enums import Role

PROBE_GROUP, PROBE_MASK = 1, 3
FIXTURE_GROUP, FIXTURE_MASK = 2, 1


def _spec(name: str, centre, group: int, mask: int) -> ObjSpec:
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    return ObjSpec(
        node_id=name,
        geom=bvh_from_mesh(box, name=name),
        transform=rt_to_fcl_transform(np.eye(3), np.asarray(centre, float), name=name),
        group=group,
        mask=mask,
    )


def _overlapping_pair():
    return (
        _spec("probe:P", (0.0, 0.0, 0.0), PROBE_GROUP, PROBE_MASK),
        _spec("well", (0.5, 0.0, 0.0), FIXTURE_GROUP, FIXTURE_MASK),
    )


def test_a_node_added_by_sync_is_collidable() -> None:
    probe, well = _overlapping_pair()
    backend = FCLBackend()
    backend.rebuild([well])
    backend.sync([probe])

    pairs = backend.collide_internal()
    assert {(p.id1, p.id2) for p in pairs} == {("probe:P", "well")}


def test_sync_records_what_rebuild_records() -> None:
    """The maps a query reads must not depend on which path added the node."""
    probe, well = _overlapping_pair()

    built = FCLBackend()
    built.rebuild([well, probe])
    synced = FCLBackend()
    synced.rebuild([well])
    synced.sync([probe])

    for attr in ("_node_to_group", "_node_to_mask", "_node_to_geomid"):
        assert set(getattr(built, attr)) == set(getattr(synced, attr)), attr
    assert getattr(synced, "_node_to_group")["probe:P"] == PROBE_GROUP
    assert getattr(synced, "_node_to_mask")["probe:P"] == PROBE_MASK


def test_the_backend_keeps_each_geometry_alive() -> None:
    """``_geomid_to_node`` is keyed on ``id``; a collected geometry frees that
    id for a different object, and the map would then name the wrong node."""
    probe, well = _overlapping_pair()
    backend = FCLBackend()
    backend.rebuild([well])
    backend.sync([probe])
    assert set(backend._node_to_geom) == {"well", "probe:P"}
    for node_id, geom_id in backend._node_to_geomid.items():
        assert id(backend._node_to_geom[node_id]) == geom_id


def test_removing_a_node_leaves_nothing_behind() -> None:
    probe, well = _overlapping_pair()
    backend = FCLBackend()
    backend.rebuild([well, probe])
    backend.remove(["probe:P"])

    for attr in (
        "_node_to_obj",
        "_node_to_geom",
        "_node_to_group",
        "_node_to_mask",
        "_node_to_geomid",
    ):
        assert "probe:P" not in getattr(backend, attr), attr
    assert set(backend._geomid_to_node.values()) == {"well"}
    assert backend.collide_internal() == []


def test_the_pair_rule_still_decides_which_pairs_are_tested() -> None:
    """Two fixtures overlap but neither admits the other's group."""
    a = _spec("well", (0.0, 0.0, 0.0), FIXTURE_GROUP, FIXTURE_MASK)
    b = _spec("implant", (0.5, 0.0, 0.0), FIXTURE_GROUP, FIXTURE_MASK)
    backend = FCLBackend()
    backend.rebuild([a])
    backend.sync([b])
    assert backend.collide_internal() == []


def test_the_bits_under_test_are_the_ones_the_adapter_assigns() -> None:
    """Pin the fixture's constants to the rule, so a change to `pair_bits`
    cannot leave these tests passing against numbers nothing produces."""
    probe = type("S", (), {"collidable": True, "role": Role.PROBE})()
    fixture = type("S", (), {"collidable": True, "role": Role.FIXTURE})()
    assert pair_bits(probe) == (PROBE_GROUP, PROBE_MASK)
    assert pair_bits(fixture) == (FIXTURE_GROUP, FIXTURE_MASK)


def test_the_manager_is_guarded_against_concurrent_use() -> None:
    """The async worker collides on its own thread while the main thread can
    be rebuilding after a probe kind change."""
    probe, well = _overlapping_pair()
    backend = FCLBackend()
    backend.rebuild([well, probe])
    assert isinstance(backend._lock, type(threading.RLock()))

    errors: list[BaseException] = []

    def hammer(fn):
        try:
            for _ in range(60):
                fn()
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(
            target=hammer,
            args=(lambda: backend.collide_internal(),),
        ),
        threading.Thread(target=hammer, args=(lambda: backend.rebuild([well, probe]),)),
        threading.Thread(
            target=hammer,
            args=(
                lambda: backend.update_transforms(
                    [
                        (
                            "probe:P",
                            rt_to_fcl_transform(np.eye(3), np.zeros(3), name="p"),
                        )
                    ]
                ),
            ),
        ),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    assert not any(th.is_alive() for th in threads), "a thread did not finish"
    assert not errors, errors


@pytest.mark.parametrize("n", [1, 4])
def test_repeated_sync_of_the_same_node_does_not_duplicate_it(n: int) -> None:
    probe, well = _overlapping_pair()
    backend = FCLBackend()
    backend.rebuild([well])
    for _ in range(n):
        backend.sync([probe])
    pairs = backend.collide_internal()
    assert len(pairs) == 1
