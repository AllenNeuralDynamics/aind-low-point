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


_BUNDLED_ONTOLOGY: CCFOntology | None = None


@dataclass
class CCFOntology:
    """In-memory index over all Allen CCF structures."""

    structures: dict[int, CCFStructure]
    _search_index: list[tuple[str, int]] = field(default_factory=list, repr=False)
    _acronym_index: dict[str, int] = field(default_factory=dict, repr=False)
    _children_map: dict[int, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self._search_index:
            self._search_index = [
                (f"{s.acronym} {s.name}".lower(), s.id)
                for s in self.structures.values()
            ]
        if not self._acronym_index:
            self._acronym_index = {s.acronym: s.id for s in self.structures.values()}
        if not self._children_map:
            cm: dict[int, list[int]] = {}
            for s in self.structures.values():
                if s.parent_id is not None:
                    cm.setdefault(s.parent_id, []).append(s.id)
            self._children_map = cm

    @classmethod
    def from_bundled(cls) -> CCFOntology:
        """Load the ontology shipped inside the package (cached singleton)."""
        global _BUNDLED_ONTOLOGY
        if _BUNDLED_ONTOLOGY is None:
            ref = resources.files("aind_low_point") / "data" / "allen_ccf_ontology.json"
            with resources.as_file(ref) as p:
                _BUNDLED_ONTOLOGY = cls.from_json(p)
        return _BUNDLED_ONTOLOGY

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

    def find_by_acronym(self, acronym: str) -> CCFStructure | None:
        """Exact-match (case-sensitive) lookup by acronym."""
        sid = self._acronym_index.get(acronym)
        return self.structures.get(sid) if sid is not None else None

    def descendants_of(
        self, acronym: str, *, include_self: bool = False
    ) -> list[CCFStructure]:
        """All descendants of the structure with the given acronym.

        Walks the parent-id tree breadth-first. Order within the result is
        roughly top-down (closer ancestors before their descendants) but is
        not part of the public contract.

        Returns ``[]`` if the acronym is unknown.
        """
        root = self.find_by_acronym(acronym)
        if root is None:
            return []

        result: list[CCFStructure] = [root] if include_self else []
        queue: list[int] = [root.id]
        seen: set[int] = {root.id}
        while queue:
            next_queue: list[int] = []
            for parent_id in queue:
                for child_id in self._children_map.get(parent_id, []):
                    if child_id in seen:
                        continue
                    seen.add(child_id)
                    child = self.structures.get(child_id)
                    if child is not None:
                        result.append(child)
                        next_queue.append(child_id)
            queue = next_queue
        return result

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
