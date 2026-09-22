"""Spin as an angle at the solver boundary.

The model carries each probe's spin as ``(cos, sin)`` and reads it through
``arctan2``, so the pair's radius is a direction no geometric term sees, and the
origin, which the variable bounds contain, is a singularity. A solver working
over the angle has neither. :class:`SpinAngle` drops each probe's sine slot,
puts the angle in radians in the cosine slot, and expands back to the unit
circle before every model evaluation, pulling gradients and Jacobians back by
the chain rule. The angle is unbounded: the model reads it only through sine
and cosine, so there is no wrap to guard. NumPy only, and outside the compiled
model, so switching representation costs no recompile.

A compiled first-order loop instead composes :func:`from_angle_layout` into the
objective before differentiating, and keeps the vector's shape: the angle sits
in the cosine slot and the sine slot holds zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from aind_rutter.optimization.objectives.layout import (
    PHASE1_PER_PROBE_VARS,
    SPIN_COS,
    SPIN_SIN,
)


class SpinAngle:
    """Maps between the model's full-layout vector and the solver's."""

    def __init__(self, n_arcs: int, n_probes: int, n_vars: int) -> None:
        blocks = n_arcs + PHASE1_PER_PROBE_VARS * np.arange(n_probes)
        self._cos = blocks + SPIN_COS
        self._sin = blocks + SPIN_SIN
        self._n_vars = n_vars
        # Solver slots in model order; each probe's angle sits where its cosine was.
        self._kept = np.setdiff1d(np.arange(n_vars), self._sin)
        self._angle = np.searchsorted(self._kept, self._cos)

    @property
    def n_solver_vars(self) -> int:
        """Length of the solver's vector."""
        return self._kept.size

    def to_solver(self, x: NDArray) -> NDArray:
        """The solver vector for model vector ``x``, whatever its spin radii."""
        x = np.asarray(x, float)
        z = x[self._kept].copy()
        z[self._angle] = np.arctan2(x[self._sin], x[self._cos])
        return z

    def to_model(self, z: NDArray) -> NDArray:
        """The model vector for solver vector ``z``, spins on the unit circle."""
        z = np.asarray(z, float)
        x = np.empty(self._n_vars)
        x[self._kept] = z
        theta = z[self._angle]
        x[self._cos] = np.cos(theta)
        x[self._sin] = np.sin(theta)
        return x

    def pull_back(self, z: NDArray, d_model: NDArray) -> NDArray:
        """A gradient ``(n_vars,)`` or Jacobian ``(m, n_vars)`` taken at
        ``to_model(z)``, as derivatives with respect to ``z``."""
        d_model = np.asarray(d_model, float)
        theta = np.asarray(z, float)[self._angle]
        d = d_model[..., self._kept].copy()
        d[..., self._angle] = (
            -np.sin(theta) * d_model[..., self._cos]
            + np.cos(theta) * d_model[..., self._sin]
        )
        return d

    def bounds(
        self, bounds: Sequence[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Model-layout ``(lo, hi)`` pairs for the solver's vector."""
        solver_bounds = [tuple(bounds[i]) for i in self._kept]
        for slot in self._angle:
            solver_bounds[slot] = (-np.inf, np.inf)
        return [(float(lo), float(hi)) for lo, hi in solver_bounds]

    def wrap(
        self, fun: Callable, jac: Callable, constraints: Sequence[dict]
    ) -> tuple[Callable, Callable, list[dict]]:
        """Objective, gradient and scipy-style constraint dicts over ``z``."""

        def over_z(model_fun: Callable) -> Callable:
            return lambda z: model_fun(self.to_model(z))

        def pulled_back(model_der: Callable) -> Callable:
            return lambda z: self.pull_back(z, model_der(self.to_model(z)))

        return (
            over_z(fun),
            pulled_back(jac),
            [
                {**c, "fun": over_z(c["fun"]), "jac": pulled_back(c["jac"])}
                for c in constraints
            ],
        )


def spin_slots(n_arcs: int, n_vars: int) -> tuple[NDArray, NDArray]:
    """Every probe's cosine and sine slot in a full-layout vector of ``n_vars``."""
    blocks = n_arcs + PHASE1_PER_PROBE_VARS * np.arange(
        (n_vars - n_arcs) // PHASE1_PER_PROBE_VARS
    )
    return blocks + SPIN_COS, blocks + SPIN_SIN


def to_angle_layout(x: NDArray, n_arcs: int) -> NDArray:
    """Full-layout ``x`` of shape ``(..., n_vars)`` in the angle layout: each
    spin's angle in radians in its cosine slot, zero in its sine slot."""
    cos, sin = spin_slots(n_arcs, x.shape[-1])
    z = np.array(x, copy=True)
    z[..., cos] = np.arctan2(x[..., sin], x[..., cos])
    z[..., sin] = 0.0
    return z


def from_angle_layout(z: Array | NDArray, n_arcs: int) -> Array:
    """The full-layout vector for angle-layout ``z``, spins on the unit circle.

    Traceable; its gradient with respect to each sine slot is zero, since the
    slot is overwritten.
    """
    z = jnp.asarray(z)
    cos, sin = spin_slots(n_arcs, z.shape[-1])
    theta = z[..., cos]
    return z.at[..., cos].set(jnp.cos(theta)).at[..., sin].set(jnp.sin(theta))
