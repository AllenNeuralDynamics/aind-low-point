"""Keys for the module-level compile caches the objectives keep.

A traced kernel is reused whenever its key repeats, so a value the trace reads
and the key omits is baked in at whatever it was on the first build and silently
reused at every other. Deriving the key from the dataclass rather than a list of
names is what makes that impossible.

Imports nothing from the package: every objective module needs this, and several
of them import each other.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass


def _key_value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return weights_cache_key(value)
    return round(value, 6) if isinstance(value, float) else value


def weights_cache_key(w) -> tuple:
    """Every weight of a weights dataclass, by name, nested dataclasses (the
    feasibility spec) included field by field.

    Floats round to 6 decimals so float noise in a recomputed weight does not
    force a recompile.
    """
    return tuple((f.name, _key_value(getattr(w, f.name))) for f in fields(w))
