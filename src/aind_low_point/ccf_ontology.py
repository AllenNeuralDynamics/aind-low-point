"""Allen CCF ontology loader and search.

Loads the bundled ``allen_ccf_ontology.json`` (produced by
``scripts/fetch_allen_ontology.py``) and provides fast substring search
over ~1300 brain-region entries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CCFStructure:
    """A single Allen CCF brain structure."""

    id: int
    acronym: str
    name: str
    color_hex: str  # "#RRGGBB"
    parent_id: int | None


@dataclass
class CCFOntology:
    """In-memory index over all Allen CCF structures."""

    structures: dict[int, CCFStructure]
    _search_index: list[tuple[str, int]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self._search_index:
            self._search_index = [
                (f"{s.acronym} {s.name}".lower(), s.id)
                for s in self.structures.values()
            ]

    @classmethod
    def from_bundled(cls) -> CCFOntology:
        """Load the ontology shipped inside the package."""
        ref = resources.files("aind_low_point") / "data" / "allen_ccf_ontology.json"
        with resources.as_file(ref) as p:
            return cls.from_json(p)

    @classmethod
    def from_json(cls, path: str | Path) -> CCFOntology:
        """Load from a flat JSON array (as produced by the fetch script)."""
        raw = json.loads(Path(path).read_text())
        structs: dict[int, CCFStructure] = {}
        for entry in raw:
            color = entry.get("color_hex_triplet", "C8C8C8")
            if not color.startswith("#"):
                color = f"#{color}"
            structs[entry["id"]] = CCFStructure(
                id=entry["id"],
                acronym=entry["acronym"],
                name=entry["name"],
                color_hex=color,
                parent_id=entry.get("parent_structure_id"),
            )
        return cls(structures=structs)

    def search(self, query: str, limit: int = 50) -> list[CCFStructure]:
        """Substring search over acronym and name fields."""
        if not query:
            return []
        q = query.lower()
        hits: list[CCFStructure] = []
        for text, sid in self._search_index:
            if q in text:
                hits.append(self.structures[sid])
                if len(hits) >= limit:
                    break
        return hits

    def get(self, label_id: int) -> CCFStructure | None:
        """Look up a structure by its integer label id."""
        return self.structures.get(label_id)

    def autocomplete_items(self, query: str, limit: int = 50) -> list[dict]:
        """Return dicts suitable for a Vuetify VAutocomplete.

        Each dict has ``title`` (display string) and ``value`` (label id).
        """
        return [
            {
                "title": f"{s.acronym} \u2014 {s.name}",
                "value": s.id,
            }
            for s in self.search(query, limit=limit)
        ]
