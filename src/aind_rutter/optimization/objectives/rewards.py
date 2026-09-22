"""Saturating rewards on clearance margin.

``1 - exp(-slack/tau)`` gated to zero at or below zero slack: the penalty
terms handle infeasibility, so the reward only fires for real margin, and
saturation stops one very loose probe from paying for a tight one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def saturating_reward_mean(
    slack: jnp.ndarray,
    tau: float,
    valid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Mean of ``1 − exp(−max(slack, 0)/τ)`` over valid entries.

    Saturating per-element reward, gated to zero when ``slack ≤ 0``
    (infeasibility is handled by the penalty terms — the reward only
    fires for actual margin). Mean form for problem-size invariance.
    """
    safe = jnp.maximum(0.0, slack)
    h = 1.0 - jnp.exp(-safe / tau)
    if valid is None:
        return jnp.mean(h)
    h_masked = jnp.where(valid > 0, h, 0.0)
    n_valid = jnp.maximum(jnp.sum(valid), 1.0)
    return jnp.sum(h_masked) / n_valid


def saturating_reward_worst(
    slack: jnp.ndarray,
    tau: float,
    valid: jnp.ndarray,
    beta: float | None = None,
) -> jnp.ndarray:
    """Per-probe worst-shank saturating reward, averaged over probes.

    ``slack`` and ``valid`` are ``(P, K)`` where the K axis spans the
    flattened (section × shank) entries for each probe. Picks the
    *worst* (minimum) valid slack per probe — so the reward fires on
    each probe's tightest shank, not an average. Probes with no valid
    entries contribute 0.

    ``beta`` replaces the minimum with a log-sum-exp soft minimum and the zero
    gate with a softplus, both of that sharpness per unit slack, so the reward
    stays differentiable where two shanks tie or a shank reaches zero slack. The
    soft minimum sits at most ``log(K) / beta`` below the hard one.
    """
    has_valid = jnp.any(valid > 0, axis=1)  # (P,)
    if beta is None:
        worst = jnp.min(jnp.where(valid > 0, slack, jnp.inf), axis=1)
        gated = jnp.maximum(0.0, worst)
    else:
        # A finite fill: with an infinite one, a probe with no valid entry gets
        # NaN log-sum-exp weights, and the final mask does not keep NaN out of
        # the gradient.
        filled = jnp.where(valid > 0, slack, 1e6)
        worst = -jax.nn.logsumexp(-beta * filled, axis=1) / beta
        gated = jax.nn.softplus(beta * worst) / beta
    h = jnp.where(has_valid, 1.0 - jnp.exp(-gated / tau), 0.0)
    return jnp.sum(h) / jnp.maximum(jnp.sum(has_valid), 1.0)
