"""Tests for ccf_overlay module."""

from __future__ import annotations

import json

import numpy as np
import pytest
import SimpleITK as sitk

from aind_low_point.ccf_ontology import CCFOntology
from aind_low_point.ccf_overlay import CCFOverlayManager

pv = pytest.importorskip("pyvista")


# --- Fixtures ---------------------------------------------------------------

ONTOLOGY_DATA = [
    {
        "id": 10,
        "acronym": "AAA",
        "name": "Region A",
        "color_hex_triplet": "FF0000",
        "parent_structure_id": None,
    },
    {
        "id": 20,
        "acronym": "BBB",
        "name": "Region B",
        "color_hex_triplet": "00FF00",
        "parent_structure_id": None,
    },
]


@pytest.fixture
def ontology(tmp_path):
    p = tmp_path / "ont.json"
    p.write_text(json.dumps(ONTOLOGY_DATA))
    return CCFOntology.from_json(p)


@pytest.fixture
def volume_path(tmp_path):
    """Create a small 10x10x10 segmentation volume with labels 10 and 20."""
    arr = np.zeros((10, 10, 10), dtype=np.int32)
    arr[2:5, 2:5, 2:5] = 10  # small cube for label 10
    arr[6:9, 6:9, 6:9] = 20  # small cube for label 20

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))

    path = tmp_path / "seg.nrrd"
    sitk.WriteImage(img, str(path))
    return path


@pytest.fixture
def manager(volume_path, ontology):
    pl = pv.Plotter(off_screen=True)
    mgr = CCFOverlayManager(
        plotter=pl,
        volume_path=volume_path,
        ontology=ontology,
        global_opacity=0.5,
    )
    yield mgr
    pl.close()


# --- Tests -------------------------------------------------------------------


class TestToggle:
    def test_toggle_on_creates_mesh(self, manager):
        result = manager.toggle(10)
        assert result is True
        region = manager._regions[10]
        assert region.mesh is not None
        assert region.visible is True
        assert region.actor is not None

    def test_toggle_off_hides(self, manager):
        manager.toggle(10)  # on
        result = manager.toggle(10)  # off
        assert result is False
        region = manager._regions[10]
        assert region.visible is False
        assert region.actor is None
        # mesh is still cached
        assert region.mesh is not None

    def test_toggle_on_again_reuses_mesh(self, manager):
        manager.toggle(10)
        mesh_id = id(manager._regions[10].mesh)
        manager.toggle(10)  # off
        manager.toggle(10)  # on again
        assert id(manager._regions[10].mesh) == mesh_id


class TestShowHide:
    def test_show_creates_actor(self, manager):
        manager.show(10)
        assert 10 in manager._regions
        assert manager._regions[10].visible is True

    def test_hide_removes_actor(self, manager):
        manager.show(10)
        manager.hide(10)
        region = manager._regions[10]
        assert region.visible is False
        assert region.actor is None

    def test_show_unknown_label_does_nothing(self, manager):
        manager.show(9999)
        assert 9999 not in manager._regions

    def test_show_idempotent(self, manager):
        manager.show(10)
        actor_id = id(manager._regions[10].actor)
        manager.show(10)
        assert id(manager._regions[10].actor) == actor_id


class TestRemove:
    def test_remove_clears_cache(self, manager):
        manager.show(10)
        manager.remove(10)
        assert 10 not in manager._regions

    def test_remove_nonexistent_is_noop(self, manager):
        manager.remove(9999)  # should not raise


class TestOpacity:
    def test_global_opacity(self, manager):
        manager.show(10)
        manager.show(20)
        manager.set_global_opacity(0.8)
        assert manager.global_opacity == 0.8
        for region in manager._regions.values():
            if region.actor is not None:
                expected = region.opacity * 0.8
                assert region.actor.prop.opacity == pytest.approx(expected, abs=0.01)

    def test_region_opacity(self, manager):
        manager.show(10)
        manager.set_region_opacity(10, 0.5)
        region = manager._regions[10]
        assert region.opacity == 0.5
        expected = 0.5 * manager.global_opacity
        assert region.actor.prop.opacity == pytest.approx(expected, abs=0.01)


class TestRegionColor:
    def test_set_color(self, manager):
        manager.show(10)
        manager.set_region_color(10, "#0000FF")
        assert manager._regions[10].color == "#0000FF"


class TestClearAll:
    def test_clear_all(self, manager):
        manager.show(10)
        manager.show(20)
        manager.clear_all()
        assert len(manager._regions) == 0


class TestVisibleRegions:
    def test_returns_visible(self, manager):
        manager.show(10)
        manager.show(20)
        manager.hide(20)
        visible = manager.visible_regions()
        assert len(visible) == 1
        assert visible[0].label_id == 10


class TestAvailableLabels:
    def test_returns_labels(self, manager):
        labels = manager.available_labels()
        assert labels == {10, 20}
