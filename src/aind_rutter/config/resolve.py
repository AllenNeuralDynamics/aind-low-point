"""Resolving a spec's effective canonicalization and transform."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from aind_rutter.config.models_common import (
    CanonicalizationDefModel,
    TransformRefModel,
)


# ---------------------------------------
def effective_canon_for_spec(
    spec: Any, canonicalizations: dict[str, "CanonicalizationDefModel"]
) -> Optional["CanonicalizationDefModel"]:
    # merge: ref -> inline -> override (last wins)
    c = None
    if getattr(spec, "canonicalization_ref", None):
        base = canonicalizations.get(spec.canonicalization_ref)
        if base is not None:
            c = deepcopy(base)
    if getattr(spec, "canonicalization", None):
        over = spec.canonicalization
        c = CanonicalizationDefModel(
            **{**(c.model_dump() if c else {}), **over.model_dump(exclude_none=True)}
        )
    if getattr(spec, "canonicalization_override", None):
        over = spec.canonicalization_override
        c = CanonicalizationDefModel(
            **{**(c.model_dump() if c else {}), **over.model_dump(exclude_none=True)}
        )
    return c


def has_transform(ref: Optional["TransformRefModel"]) -> bool:
    # Your TransformRefModel already enforces exactly one of {key | inline} if present.
    return bool(ref and (ref.key is not None or ref.inline is not None))
