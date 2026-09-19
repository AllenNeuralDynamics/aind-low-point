"""Which constraint rows are real, and what the padded ones report.

Candidates differ in probe, shank and section count, so every batched row
is padded to a fixed shape. A padded row must never look violated, so it
reports a large positive slack and is masked out of the counts.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

LARGE_SLACK = 1e3  # sentinel for masked-out (padded) constraints


PADDED_SLACK = LARGE_SLACK


# Constraint groups, in the order they are concatenated into the slack vector.
SLACK_GROUPS = ("thread", "probe_pair", "probe_fixture", "brain", "arc_sep", "ml_sep")


def padding_mask(
    packed: dict, labels: dict, *, n_arcs: int, has_brain: bool
) -> NDArray:
    """Which rows of the slack vector constrain something, in ``SLACK_GROUPS``
    order.

    Uniform compiled shapes across candidates need per-probe padding, and those
    rows reach the solver as constants with no gradient. They do not make the KKT
    matrix singular — each inequality carries its own slack, so such a row reads
    ``[0 … 0 | −1]`` — but each still costs a Jacobian row, a slack and a barrier
    term in every factorization. The same masks that create the padding say which
    rows they are, so the mask is a property of the probe geometry alone. Reading
    it off slack *values* instead also drops live rows that merely sit at the
    out-of-grid sentinel, which share the padding value but do carry gradient
    once a pose brings them near a fixture.
    """
    section = np.asarray(packed["section_mask"]) > 0  # (probes, sections)
    shank = np.asarray(packed["shank_mask"]) > 0  # (probes, shanks)
    same_arc = np.asarray(packed["same_arc_mask"]) > 0  # (probes, probes)
    iu, ju = np.triu_indices(shank.shape[0], k=1)
    by_group = {
        "thread": (section[:, :, None] & shank[:, None, :]).reshape(-1),
        "probe_pair": np.ones(
            len(labels["probe_pairs"]) * len(labels["pair_categories"]), bool
        ),
        "probe_fixture": np.ones(
            len(labels["fixtures"])
            * len(labels["fixture_probes"])
            * len(labels["fixture_categories"]),
            bool,
        ),
        "brain": shank.reshape(-1) if has_brain else np.zeros(0, bool),
        "arc_sep": np.ones(n_arcs * (n_arcs - 1) // 2, bool),
        "ml_sep": same_arc[iu, ju],
    }
    return np.concatenate([by_group[name] for name in SLACK_GROUPS])


def live_row_constraints(slacks_fn, slacks_jac, live_rows: NDArray, drop_padded: bool):
    """The constraint callables the solver sees.

    Masking happens outside the traced function, so the compiled slack vector is
    unchanged and only IPOPT's view of it shrinks.
    """
    if not drop_padded or live_rows.all():
        return slacks_fn, slacks_jac

    def live_fn(x: NDArray) -> NDArray:
        return slacks_fn(x)[live_rows]

    def live_jac(x: NDArray) -> NDArray:
        return slacks_jac(x)[live_rows]

    return live_fn, live_jac
