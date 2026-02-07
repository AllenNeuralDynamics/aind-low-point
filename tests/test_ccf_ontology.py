"""Tests for ccf_ontology module."""

from __future__ import annotations

import json

import pytest

from aind_low_point.ccf_ontology import CCFOntology

FIXTURE = [
    {
        "id": 997,
        "acronym": "root",
        "name": "root",
        "color_hex_triplet": "FFFFFF",
        "parent_structure_id": None,
    },
    {
        "id": 8,
        "acronym": "grey",
        "name": "Basic cell groups and regions",
        "color_hex_triplet": "BFDAE3",
        "parent_structure_id": 997,
    },
    {
        "id": 385,
        "acronym": "VISp",
        "name": "Primary visual area",
        "color_hex_triplet": "08858C",
        "parent_structure_id": 8,
    },
    {
        "id": 500,
        "acronym": "MO",
        "name": "Somatomotor areas",
        "color_hex_triplet": "1F9D5A",
        "parent_structure_id": 8,
    },
    {
        "id": 993,
        "acronym": "CA1",
        "name": "Field CA1",
        "color_hex_triplet": "7ED04B",
        "parent_structure_id": 8,
    },
]


@pytest.fixture
def ontology(tmp_path):
    p = tmp_path / "ont.json"
    p.write_text(json.dumps(FIXTURE))
    return CCFOntology.from_json(p)


class TestCCFOntologyFromJson:
    def test_structure_count(self, ontology):
        assert len(ontology.structures) == 5

    def test_get_existing(self, ontology):
        s = ontology.get(385)
        assert s is not None
        assert s.acronym == "VISp"
        assert s.name == "Primary visual area"

    def test_get_missing(self, ontology):
        assert ontology.get(9999) is None

    def test_color_hex_prefix(self, ontology):
        for s in ontology.structures.values():
            assert s.color_hex.startswith("#"), f"{s.acronym}: {s.color_hex}"

    def test_parent_id(self, ontology):
        root = ontology.get(997)
        assert root is not None
        assert root.parent_id is None

        visp = ontology.get(385)
        assert visp is not None
        assert visp.parent_id == 8


class TestCCFOntologySearch:
    def test_search_by_acronym(self, ontology):
        results = ontology.search("VISp")
        assert len(results) >= 1
        assert any(s.id == 385 for s in results)

    def test_search_by_name(self, ontology):
        results = ontology.search("visual")
        assert len(results) >= 1
        assert any(s.id == 385 for s in results)

    def test_search_case_insensitive(self, ontology):
        results = ontology.search("visp")
        assert len(results) >= 1
        assert any(s.id == 385 for s in results)

    def test_search_empty_query(self, ontology):
        assert ontology.search("") == []

    def test_search_no_match(self, ontology):
        assert ontology.search("zzzzz") == []

    def test_search_limit(self, ontology):
        results = ontology.search("a", limit=2)
        assert len(results) <= 2


class TestAutocompleteItems:
    def test_format(self, ontology):
        items = ontology.autocomplete_items("VISp")
        assert len(items) >= 1
        item = items[0]
        assert "title" in item
        assert "value" in item
        assert item["value"] == 385
        assert "VISp" in item["title"]
        assert "Primary visual area" in item["title"]

    def test_empty_query(self, ontology):
        assert ontology.autocomplete_items("") == []


class TestCCFOntologyFromBundled:
    def test_loads_bundled(self):
        ont = CCFOntology.from_bundled()
        assert len(ont.structures) > 1000
        # VISp should be present
        visp = ont.get(385)
        assert visp is not None
        assert visp.acronym == "VISp"
