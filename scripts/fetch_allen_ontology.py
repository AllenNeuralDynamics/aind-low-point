#!/usr/bin/env python3
"""Fetch Allen CCF ontology and flatten to a bundled JSON file.

Usage:
    python scripts/fetch_allen_ontology.py

Writes to src/aind_low_point/data/allen_ccf_ontology.json
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
OUT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "aind_low_point"
    / "data"
    / "allen_ccf_ontology.json"
)


def _flatten(nodes: list[dict], parent_id: int | None = None) -> list[dict]:
    """Recursively flatten the nested Allen ontology tree."""
    result: list[dict] = []
    for node in nodes:
        color = node.get("color_hex_triplet", "C8C8C8")
        result.append(
            {
                "id": node["id"],
                "acronym": node["acronym"],
                "name": node["name"],
                "color_hex_triplet": color,
                "parent_structure_id": parent_id,
            }
        )
        children = node.get("children", [])
        if children:
            result.extend(_flatten(children, parent_id=node["id"]))
    return result


def main():
    print(f"Fetching {URL} ...")
    with urllib.request.urlopen(URL, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    root_nodes = data["msg"]
    flat = _flatten(root_nodes)
    print(f"Flattened {len(flat)} structures")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(flat, indent=1, ensure_ascii=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
