"""Arc AP placement helpers for optimizer seed emission."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import isotonic_regression


def _qp_chain_box(c: NDArray, lo: NDArray, hi: NDArray, sep: float) -> NDArray:
    """Exact L2 projection onto an ascending chain with box bounds."""
    from scipy.optimize import minimize

    n = len(c)
    cons = [
        {"type": "ineq", "fun": (lambda a, i=i: a[i + 1] - a[i] - sep)}
        for i in range(n - 1)
    ]
    x0 = np.clip(c, lo, hi)
    res = minimize(
        lambda a: float(np.sum((a - c) ** 2)),
        x0,
        method="SLSQP",
        bounds=list(zip(lo.tolist(), hi.tolist())),
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    return np.asarray(res.x, dtype=np.float64)


def bounded_isotonic_arc_aps(
    centroids: NDArray,
    lows: NDArray,
    highs: NDArray,
    min_sep_deg: float,
) -> NDArray:
    """L2-place arc APs subject to ordered separation and per-arc windows.

    The problem is reduced by sorting arcs by preferred centroid and
    substituting ``d_k = ap_k - k * min_sep``. Without active box bounds, the
    exact optimum is standard isotonic regression in ``d``-space. If a bound is
    active, the rare fallback solves the small constrained QP directly.

    Returned APs use the caller's original arc order.
    """
    c = np.asarray(centroids, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    hi = np.asarray(highs, dtype=np.float64)
    n = c.size
    if n == 0:
        return c.copy()
    if n == 1:
        return np.clip(c, lo, hi)

    order = np.argsort(c, kind="stable")
    k = np.arange(n, dtype=np.float64)
    cs, los, his = c[order], lo[order], hi[order]
    d = isotonic_regression(cs - k * min_sep_deg, increasing=True).x
    aps_sorted = np.asarray(d) + k * min_sep_deg

    if not ((aps_sorted >= los - 1e-9).all() and (aps_sorted <= his + 1e-9).all()):
        aps_sorted = _qp_chain_box(cs, los, his, min_sep_deg)

    out = np.empty(n, dtype=np.float64)
    out[order] = aps_sorted
    return out
