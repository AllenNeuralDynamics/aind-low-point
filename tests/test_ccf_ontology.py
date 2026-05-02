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

    def test_bundled_is_cached(self):
        a = CCFOntology.from_bundled()
        b = CCFOntology.from_bundled()
        assert a is b


class TestFindByAcronym:
    def test_exact_match(self, ontology):
        s = ontology.find_by_acronym("VISp")
        assert s is not None
        assert s.id == 385

    def test_case_sensitive(self, ontology):
        # "visp" lowercase should not match "VISp"
        assert ontology.find_by_acronym("visp") is None

    def test_unknown_acronym(self, ontology):
        assert ontology.find_by_acronym("ZZZ") is None


class TestDescendantsOf:
    def test_descendants_of_root(self, ontology):
        # root → grey → {VISp, MO, CA1}
        ids = {s.id for s in ontology.descendants_of("root")}
        assert ids == {8, 385, 500, 993}

    def test_descendants_of_intermediate(self, ontology):
        # grey → {VISp, MO, CA1} (no further nesting in fixture)
        ids = {s.id for s in ontology.descendants_of("grey")}
        assert ids == {385, 500, 993}

    def test_descendants_of_leaf(self, ontology):
        assert ontology.descendants_of("VISp") == []

    def test_descendants_of_unknown(self, ontology):
        assert ontology.descendants_of("ZZZ") == []

    def test_descendants_include_self(self, ontology):
        ids = {s.id for s in ontology.descendants_of("grey", include_self=True)}
        assert ids == {8, 385, 500, 993}

    def test_descendants_of_leaf_include_self(self, ontology):
        result = ontology.descendants_of("VISp", include_self=True)
        assert len(result) == 1
        assert result[0].id == 385

    def test_descendants_bfs_order(self, ontology):
        # Closer ancestors should appear before their descendants. Direct
        # children of root come before grandchildren.
        result = ontology.descendants_of("root")
        ids_in_order = [s.id for s in result]
        # 'grey' (id=8) is a direct child of root; the rest are grandchildren.
        assert ids_in_order[0] == 8
        assert set(ids_in_order[1:]) == {385, 500, 993}
