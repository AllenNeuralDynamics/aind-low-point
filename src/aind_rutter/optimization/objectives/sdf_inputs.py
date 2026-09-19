"""World-frame signed-distance grids an objective is built over.

A leaf, so the pipeline that builds these grids does not import the
objective that consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class FixtureSDFData:
    """Static-in-world fixture body SDF (α-wrap envelope).

    Used for probe-vs-fixture body clearance in Phase 1. Built once
    from the fixture mesh (already canonicalized to world LPS) and
    closure-captured by the JIT'd objective.
    """

    name: str
    grid: jnp.ndarray
    origin: jnp.ndarray
    spacing: jnp.ndarray
    surface: jnp.ndarray


@dataclass(frozen=True)
class BrainSDFData:
    """Static-in-world brain signed-distance grid (negative inside).

    Used for the brain-containment term: each shank tip must stay inside
    the brain (don't puncture through the bottom). Built once from the
    world-frame brain mesh and closure-captured by the JIT'd objective.
    Only the voxel SDF is needed — containment is a point query at the
    tips, not a surface-sampling clearance.
    """

    grid: jnp.ndarray
    origin: jnp.ndarray
    spacing: jnp.ndarray
