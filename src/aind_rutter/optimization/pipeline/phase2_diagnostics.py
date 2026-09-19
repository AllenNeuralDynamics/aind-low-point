"""Diagnostics for Phase-2 solves: per-iteration IPOPT history, start-pose jitter,
and explicit rank lists.

Used by ``phase2`` when the diagnostics settings are set, for
the success-estimator experiments in ``dev/proposals/PIPELINE_PLAN.md``. Kept
free of JAX so it
imports cheaply.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# x layout after the arc APs: (ml, sx, sy, off_R, off_A, depth) per probe.
PER_PROBE_VARS = 6

# Standard deviations drawn per candidate at ``P2_PERTURB=1``.
PERTURB_SIGMA: dict[str, float] = {
    "arc_ap_deg": 0.5,
    "ml_deg": 0.5,
    "spin_deg": 5.0,
    "offset_mm": 0.05,
    "depth_mm": 0.1,
}


def read_ranks(path: str | Path) -> list[int]:
    """Zero-based offsets into the Phase-2 selection order.

    Whitespace- or comma-separated; ``#`` starts a comment.
    """
    ranks: list[int] = []
    for line in Path(path).read_text().splitlines():
        body = line.split("#", 1)[0].replace(",", " ")
        ranks.extend(int(tok) for tok in body.split())
    return ranks


def perturb_pose(
    pose: NDArray,
    n_arcs: int,
    n_probes: int,
    bounds: Sequence[tuple[float, float]],
    scale: float,
    seed: int | Sequence[int],
) -> NDArray:
    """Gaussian jitter of a Phase-1 pose, for label-repeatability runs.

    Spin rotates ``(sx, sy)`` so its magnitude is kept; the result is clipped to
    ``bounds``. ``scale <= 0`` returns the pose unchanged; a given ``seed`` always
    draws the same jitter.
    """
    x = np.asarray(pose, dtype=float).copy()
    if scale <= 0:
        return x
    rng = np.random.default_rng(seed)
    sd = {k: v * scale for k, v in PERTURB_SIGMA.items()}
    x[:n_arcs] += rng.normal(0.0, sd["arc_ap_deg"], n_arcs)
    for i in range(n_probes):
        b = n_arcs + PER_PROBE_VARS * i
        x[b] += rng.normal(0.0, sd["ml_deg"])
        angle = math.radians(rng.normal(0.0, sd["spin_deg"]))
        c, s = math.cos(angle), math.sin(angle)
        x[b + 1], x[b + 2] = c * x[b + 1] - s * x[b + 2], s * x[b + 1] + c * x[b + 2]
        x[b + 3 : b + 5] += rng.normal(0.0, sd["offset_mm"], 2)
        x[b + 5] += rng.normal(0.0, sd["depth_mm"])
    lo = np.array([bd[0] for bd in bounds], dtype=float)
    hi = np.array([bd[1] for bd in bounds], dtype=float)
    return np.clip(x, lo, hi)


class _IterationRecorder:
    """Accumulates IPOPT iteration stats and per-group constraint summaries."""

    def __init__(self, group_sizes: Sequence[int] | None, padded_slack: float):
        self.group_sizes = (
            None if group_sizes is None else [int(n) for n in group_sizes]
        )
        self.padded_slack = padded_slack
        self.rows: list[tuple] = []
        self.group_min: list[list[float]] = []
        self.group_nviol: list[list[int]] = []

    def record(
        self, alg_mod, iter_count, obj, inf_pr, inf_du, mu, alpha_pr, ls_trials, g
    ):
        self.rows.append(
            (iter_count, obj, inf_pr, inf_du, mu, alpha_pr, ls_trials, alg_mod)
        )
        if self.group_sizes is None:
            return
        if g is None:  # Ipopt < 3.14 cannot report the iterate
            self.group_min.append([math.nan] * len(self.group_sizes))
            self.group_nviol.append([-1] * len(self.group_sizes))
            return
        splits = np.cumsum(self.group_sizes)[:-1]
        mins, nviol = [], []
        for part in np.split(np.asarray(g, dtype=float), splits):
            real = part[part < self.padded_slack - 1.0]
            mins.append(float(real.min()) if real.size else math.nan)
            nviol.append(int((real < 0.0).sum()))
        self.group_min.append(mins)
        self.group_nviol.append(nviol)

    def as_arrays(self) -> dict[str, NDArray]:
        cols = list(zip(*self.rows)) if self.rows else [()] * 8
        out: dict[str, NDArray] = {
            "iter": np.asarray(cols[0], dtype=np.int32),
            "obj": np.asarray(cols[1], dtype=np.float64),
            "inf_pr": np.asarray(cols[2], dtype=np.float64),
            "inf_du": np.asarray(cols[3], dtype=np.float64),
            "mu": np.asarray(cols[4], dtype=np.float64),
            "alpha_pr": np.asarray(cols[5], dtype=np.float64),
            "ls_trials": np.asarray(cols[6], dtype=np.int32),
            # 1 while IPOPT is in its feasibility-restoration phase.
            "restoration": np.asarray(cols[7], dtype=np.int8),
        }
        if self.group_sizes is not None:
            out["group_min"] = np.asarray(self.group_min, dtype=np.float32)
            out["group_nviol"] = np.asarray(self.group_nviol, dtype=np.int32)
        return out


def _logged_wrapper_class():
    from cyipopt.scipy_interface import IpoptProblemWrapper

    class LoggedProblemWrapper(IpoptProblemWrapper):
        recorder: _IterationRecorder
        nlp: Any = None  # the cyipopt.Problem, set after construction

        def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials,
        ):
            super().intermediate(
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials,
            )
            g = None
            if self.recorder.group_sizes is not None:
                iterate = self.nlp.get_current_iterate(scaled=False)
                g = None if iterate is None else iterate["g"]
            self.recorder.record(
                alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, alpha_pr,
                ls_trials, g,
            )  # fmt: skip

    return LoggedProblemWrapper


def minimize_ipopt_logged(
    fun,
    x0: NDArray,
    *,
    jac,
    bounds,
    constraints,
    options: dict | None = None,
    group_sizes: Sequence[int] | None = None,
    padded_slack: float = 1e3,
):
    """``cyipopt.minimize_ipopt`` (default Ipopt method) that also records each
    iteration. Returns ``(result, history)``.

    Construction mirrors cyipopt 1.7's ``minimize_ipopt`` line for line so the solve
    is identical; ``tests/test_phase2_diagnostics.py`` checks that. ``history`` holds
    per-iteration ``iter, obj, inf_pr, inf_du, mu, alpha_pr, ls_trials`` and, with
    ``group_sizes``, ``group_min``/``group_nviol`` over the constraint vector split
    into those consecutive groups (entries at ``padded_slack`` or above excluded).
    """
    import cyipopt
    from cyipopt import scipy_interface as si

    (fun, x0, args, kwargs, _method, jac, hess, hessp, bounds, constraints, tol, _cb,
     options) = si._minimize_ipopt_iv(
        fun, x0, (), None, None, jac, None, None, bounds, constraints, None, None,
        options,
    )  # fmt: skip
    _x0 = np.atleast_1d(x0)
    lb, ub = bounds
    cl, cu = si.get_constraint_bounds(constraints, _x0)
    con_dims = si.get_constraint_dimensions(constraints, _x0)
    sparse_jacs, jac_nnz_row, jac_nnz_col = si._get_sparse_jacobian_structure(
        constraints, x0
    )
    hess_tril, hess_nnz_row, hess_nnz_col = si._get_sparse_hessian_structure(
        x0, args, kwargs, hess, constraints, con_dims
    )
    if options is None:
        options = {}
    eps = options.pop("eps", 1e-8)

    problem = _logged_wrapper_class()(
        fun,
        args=args,
        kwargs=kwargs,
        jac=jac,
        hess=hess,
        hessp=hessp,
        constraints=constraints,
        eps=eps,
        con_dims=con_dims,
        sparse_jacs=sparse_jacs,
        jac_nnz_row=jac_nnz_row,
        jac_nnz_col=jac_nnz_col,
        hess_tril=hess_tril,
        hess_nnz_row=hess_nnz_row,
        hess_nnz_col=hess_nnz_col,
    )
    nlp = cyipopt.Problem(
        n=len(x0), m=len(cl), problem_obj=problem, lb=lb, ub=ub, cl=cl, cu=cu
    )
    problem.recorder = _IterationRecorder(group_sizes, padded_slack)
    problem.nlp = nlp

    si.convert_to_bytes(options)
    si.replace_option(options, b"disp", b"print_level")
    si.replace_option(options, b"maxiter", b"max_iter")
    options[b"print_level"] = int(options.get(b"print_level", 0))
    if b"tol" not in options:
        options[b"tol"] = tol or 1e-8
    if b"mu_strategy" not in options:
        options[b"mu_strategy"] = b"adaptive"
    if b"hessian_approximation" not in options and hess is None and hessp is None:
        options[b"hessian_approximation"] = b"limited-memory"
    for option, value in options.items():
        nlp.add_option(option, value)

    try:
        x, info = nlp.solve(x0)
    finally:
        # problem ↔ nlp is a reference cycle; left intact, the objective and its
        # device buffers live until the next cyclic GC and exhaust GPU memory.
        problem.nlp = None
    result = si.OptimizeResult(
        x=x,
        success=info["status"] == 0,
        status=info["status"],
        message=info["status_msg"],
        fun=info["obj_val"],
        info=info,
        nfev=problem.nfev,
        njev=problem.njev,
        nit=problem.nit,
    )
    return result, problem.recorder.as_arrays()
