"""Where each variable sits in the optimizer's state vectors.

Two layouts, both `[arc APs (n_arcs)]` followed by one block per probe:

- **full**, `PHASE1_PER_PROBE_VARS` wide — `(ml, cos spin, sin spin, offset_R,
  offset_A, depth)`. What Phase 1's full stage and Phase 2 solve over.
- **reduced**, `REDUCED_PER_PROBE_VARS` wide — `(ml, cos spin, sin spin)`. What
  the spin restore and the reduced pre-pass solve over, with offsets and depth
  pinned.

The first three entries mean the same thing in both, which is what lets a
reduced solution seed a full one.

The reduced stride was written as a bare ``3`` at 21 sites across five modules.
A literal is not wrong until the layout changes, at which point every site that
was missed keeps indexing the old shape and reads a neighbouring variable
instead of failing.

Imports nothing from the package: the modules that need it import each other.
"""

from __future__ import annotations

PHASE1_PER_PROBE_VARS = 6
REDUCED_PER_PROBE_VARS = 3

# Offsets within a probe's block, shared by both layouts.
ML = 0
SPIN_COS = 1
SPIN_SIN = 2


def reduced_block(n_arcs: int, probe_index):
    """Index of this probe's first reduced variable.

    ``probe_index`` may be a traced integer; the arithmetic is the same.
    """
    return n_arcs + REDUCED_PER_PROBE_VARS * probe_index


def reduced_var(n_arcs: int, probe_index, slot: int = ML):
    """Index of a single reduced variable.

    For **one** slot only. Reading several, take ``reduced_block`` once and add
    the offsets — calling this per slot recomputes ``stride * index`` each time,
    because nothing shares it between calls while tracing.

    The ``slot == 0`` test resolves at trace time, since ``slot`` is a Python
    int. It is there because ``reduced_block(...) + ML`` emits a scalar add that
    XLA does not fold, inside `spin_restore`'s ``fori_loop`` where the probe
    index is traced.
    """
    base = reduced_block(n_arcs, probe_index)
    return base if slot == 0 else base + slot


def full_block(n_arcs: int, probe_index):
    """Index of this probe's first full-layout variable."""
    return n_arcs + PHASE1_PER_PROBE_VARS * probe_index


def reduced_n_vars(n_arcs: int, n_probes: int) -> int:
    return n_arcs + REDUCED_PER_PROBE_VARS * n_probes


def full_n_vars(n_arcs: int, n_probes: int) -> int:
    return n_arcs + PHASE1_PER_PROBE_VARS * n_probes
