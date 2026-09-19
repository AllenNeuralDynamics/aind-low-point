"""Smooth replacements for ``min`` and ``abs``.

A hard ``min`` has zero gradient for every element but the argmin, which
stalls a solver whose active constraint keeps changing; these spread it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def soft_min_topk(
    values: Array,
    *,
    beta: float = 20.0,
    top_k: int = 16,
) -> Array:
    """Smooth approximation to ``min(values)`` using top-k softmin.

    Selects the ``top_k`` smallest values, then aggregates via
    ``-logsumexp(-β·v)/β`` (= negative scaled LSE of negated values).

    Properties:
      - Returns ``≤ min(values)`` (bias ``log(k)/β`` downward when the
        top-k samples are clustered).
      - C¹ smooth in ``values``.
      - Gradient flows through the smallest k values weighted by
        ``softmax(-β·v_topk)``.

    Defaults β=20/mm and k=16 give a 50 µm smoothing window with
    ~0.14 mm worst-case downward bias — calibrated for sub-mm probe
    clearance gradients (see design discussion).

    For ``len(values) <= top_k`` the function reduces to plain softmin
    over all values.
    """
    n = values.shape[-1]
    if n > top_k:
        # ``-jax.lax.top_k(-x, k)`` returns smallest k.
        smallest, _ = jax.lax.top_k(-values, top_k)
        smallest = -smallest
    else:
        smallest = values
    return -jax.nn.logsumexp(-beta * smallest, axis=-1) / beta


def smooth_abs(x: Array, eps: float = 1e-3) -> Array:
    """Smooth approximation to ``|x|`` via ``sqrt(x² + ε²)``. Continuous
    derivative everywhere (vs ``abs``'s sign-flip at zero).

    Default ε = 1e-3 mm/deg keeps the soft region tight around zero —
    only meaningfully different from ``abs`` when ``|x| < a few ε``.
    """
    return jnp.sqrt(x * x + eps * eps)
