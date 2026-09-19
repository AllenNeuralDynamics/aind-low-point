"""Template merge: how a spec inherits from the templates it names."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

from aind_rutter.config.models_catalog import BaseSpecModel


def as_overlay(model: BaseModel | None) -> dict[str, Any]:
    return {} if model is None else model.model_dump(exclude_unset=True)


def union_list(a: Optional[list[Any]], b: Optional[list[Any]]) -> Optional[list[Any]]:
    if a is None and b is None:
        return None
    seen: set[Any] = set()
    out: list[Any] = []
    for src in (a or []), (b or []):
        for x in src:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def merge_dict_shallow(
    a: Optional[dict[str, Any]], b: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return {**a, **b}


# -----------------------------
# Asset template merge
# -----------------------------


def merge_asset_source_fields(
    base: dict[str, Any],
    over: dict[str, Any],
) -> dict[str, Any]:
    """
    Choose exactly one source mode:
      - (src + loader [+ loader_kwargs])
      - (from_resource + selector)
    If overlay specifies any field of a mode, that mode wins; the other mode is
    cleared.
    """
    out: dict[str, Any] = {}

    over_file = (over.get("src", None) is not None) or (
        over.get("loader", None) is not None
    )
    over_res = (over.get("from_resource", None) is not None) or (
        over.get("selector", None) is not None
    )

    if over_file and over_res:
        raise ValueError("Asset: overlay specifies both file and resource source modes")

    if over_file:
        # file mode wins; overlay values take priority, else fall back to base
        out["src"] = over.get("src", None) or base.get("src", None)
        out["loader"] = over.get("loader", None) or base.get("loader", None)
        out["loader_kwargs"] = merge_dict_shallow(
            base.get("loader_kwargs", None), over.get("loader_kwargs", None)
        )
        out["from_resource"] = None
        out["selector"] = None

    elif over_res:
        out["from_resource"] = over.get("from_resource", None) or base.get(
            "from_resource", None
        )
        out["selector"] = over.get("selector", None) or base.get("selector", None)

        # clear file mode
        out["src"] = None
        out["loader"] = None
        out["loader_kwargs"] = {}
    else:
        # overlay did not change mode → keep base (as-is)
        out["src"] = over.get("src", None) or base.get("src", None)
        out["loader"] = over.get("loader", None) or base.get("loader", None)
        out["loader_kwargs"] = (
            over.get("loader_kwargs", {}) or base.get("loader_kwargs", {}) or {}
        )
        out["from_resource"] = over.get("from_resource", None) or base.get(
            "from_resource", None
        )
        out["selector"] = over.get("selector", None) or base.get("selector", None)

    return out


def merge_asset_template_model_dumps(
    base: dict[str, Any],
    over: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(base)
    out.update(over)

    # unions
    out["tags"] = union_list(base.get("tags"), over.get("tags")) or []

    # metadata shallow merge
    out["metadata"] = merge_dict_shallow(base.get("metadata"), over.get("metadata"))

    # nested merges
    out["material"] = merge_dict_shallow(
        base.get("material"), over.get("material", None)
    )
    out["canonicalization"] = merge_dict_shallow(
        base.get("canonicalization"), over.get("canonicalization", None)
    )
    out["canonicalization_override"] = merge_dict_shallow(
        base.get("canonicalization_override"),
        over.get("canonicalization_override", None),
    )

    # refs (replace-on-write)
    out["material_ref"] = over.get("material_ref", None) or base.get(
        "material_ref", None
    )

    # hints (replace-on-write)
    out["pivot_LPS"] = over.get("pivot_LPS", None) or base.get("pivot_LPS", None)
    out["bbox_hint"] = over.get("bbox_hint", None) or base.get("bbox_hint", None)

    # chem-shift hints (replace-on-write)
    out["chem_shift_ppm"] = (
        over.get("chem_shift_ppm", None)
        if "chem_shift_ppm" in over
        else base.get("chem_shift_ppm", None)
    )

    # source modes
    out.update(merge_asset_source_fields(base, over))

    return out


T = TypeVar("T", bound=BaseSpecModel)


def apply_templates_generic(
    mergefun: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    spec: T,
    template_names: list[str],
    registry: dict[str, Any],
) -> T:
    if not template_names:
        return spec

    # Start with spec
    base = spec.model_dump()
    # fold templates left→right
    merged = base
    for name in template_names:
        t = registry.get(name)
        if t is None:
            continue  # already reported by _check_template_ref
        merged = mergefun(merged, as_overlay(t))

    merged_tmpl = mergefun(merged, as_overlay(spec))  # overlay spec onto templates
    # materialize back to AssetSpecModel; ensure we keep spec's identity fields
    merged_tmpl["key"] = spec.key
    merged_tmpl["templates"] = []
    return spec.__class__(**merged_tmpl)


# -----------------------------
# Target template merge
# -----------------------------


def detect_target_mode(target_dump: dict[str, Any]) -> Optional[str]:
    explicit = (target_dump.get("src", None) is not None) or (
        target_dump.get("loader", None) is not None
    )
    derived = target_dump.get("source_key", None) is not None
    res = (target_dump.get("from_resource", None) is not None) or (
        target_dump.get("selector", None) is not None
    )
    cnt = int(explicit) + int(derived) + int(res)
    if cnt == 0:
        return None
    if cnt == 1:
        return "explicit" if explicit else ("derived" if derived else "resource")
    return "conflict"


def merge_target_source_fields(
    base: dict[str, Any],
    over: dict[str, Any],
) -> dict[str, Any]:
    """
    Exactly one of:
      - explicit: src + loader [+ loader_kwargs]
      - derived : source_key + reducer
      - resource: from_resource + selector
    Overlay choosing any field of a mode selects that mode and clears others.
    """
    out: dict[str, Any] = {}

    mode_base = detect_target_mode(base)
    mode_over = detect_target_mode(over)
    if mode_over == "conflict":
        raise ValueError("Target: overlay specifies conflicting source modes")

    # if overlay picks a mode, use it; else keep base
    mode = mode_over or mode_base

    if mode == "explicit":
        out["src"] = over.get("src", None) or base.get("src", None)
        out["loader"] = over.get("loader", None) or base.get("loader", None)
        out["loader_kwargs"] = merge_dict_shallow(
            base.get("loader_kwargs", None), over.get("loader_kwargs", None)
        )
        out["reducer"] = over.get("reducer", None) or base.get("reducer", None)
        out["reducer_kwargs"] = merge_dict_shallow(
            base.get("reducer_kwargs", None), over.get("reducer_kwargs", None)
        )
        # clear others
        out.update(
            {
                "source_key": None,
                "from_resource": None,
                "selector": None,
            }
        )

    elif mode == "derived":
        out["source_key"] = over.get("source_key", None) or getattr(
            base, "source_key", None
        )
        out["reducer"] = over.get("reducer", None) or base.get("reducer", None)
        out["reducer_kwargs"] = merge_dict_shallow(
            base.get("reducer_kwargs", None), over.get("reducer_kwargs", None)
        )
        # clear others
        out.update(
            {
                "src": None,
                "loader": None,
                "loader_kwargs": {},
                "from_resource": None,
                "selector": None,
            }
        )

    elif mode == "resource":
        out["from_resource"] = over.get("from_resource", None) or getattr(
            base, "from_resource", None
        )
        out["selector"] = over.get("selector", None) or getattr(base, "selector", None)
        # clear others
        out.update(
            {
                "src": None,
                "loader": None,
                "loader_kwargs": {},
                "source_key": None,
            }
        )

    else:
        # neither base nor overlay specified a mode → keep as is (all None/empty)
        out.update(
            {
                "src": base.get("src", None),
                "loader": base.get("loader", None),
                "loader_kwargs": base.get("loader_kwargs", {}) or {},
                "source_key": base.get("source_key", None),
                "from_resource": base.get("from_resource", None),
                "selector": base.get("selector", None),
            }
        )

    return out


def merge_target_template_model_dumps(
    base: dict[str, Any],
    over: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(base)
    out.update(over)

    # unions
    out["tags"] = union_list(base.get("tags"), over.get("tags")) or []

    # metadata shallow merge
    out["metadata"] = merge_dict_shallow(base.get("metadata"), over.get("metadata"))

    # nested merges
    out["material"] = merge_dict_shallow(
        base.get("material"), over.get("material", None)
    )
    out["canonicalization"] = merge_dict_shallow(
        base.get("canonicalization"), over.get("canonicalization", None)
    )
    out["canonicalization_override"] = merge_dict_shallow(
        base.get("canonicalization_override"),
        over.get("canonicalization_override", None),
    )

    # refs (replace-on-write)
    out["material_ref"] = over.get("material_ref", None) or base.get(
        "material_ref", None
    )

    # hints (replace-on-write)
    out["pivot_LPS"] = over.get("pivot_LPS", None) or base.get("pivot_LPS", None)
    out["bbox_hint"] = over.get("bbox_hint", None) or base.get("bbox_hint", None)
    out["approach_vector"] = over.get("approach_vector", None) or base.get(
        "approach_vector", None
    )
    out["uncertainty_mm"] = over.get("uncertainty_mm", None) or base.get(
        "uncertainty_mm", None
    )

    # chem-shift hints (targets rarely need it; still honor if present)
    out["chem_shift_ppm"] = (
        over.get("chem_shift_ppm", None)
        if "chem_shift_ppm" in over
        else base.get("chem_shift_ppm", None)
    )

    # source modes
    out.update(merge_target_source_fields(base, over))

    return out
