"""Phase-1/2 optimizer variable-vector (``x``) pack/unpack and plan application.

The optimizer state ``x`` is ``[arc_aps (n_arcs)]`` followed by per-probe blocks
of :data:`PHASE1_PER_PROBE_VARS` values ``(ml, cos spin, sin spin, offset_R,
offset_A, depth)``. These helpers build it, read spins back out, reconstruct
world poses, and write a solution onto a ``PlanningState``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from aind_low_point.optimization.objectives.phase1 import PHASE1_PER_PROBE_VARS
from aind_low_point.optimization.sdf.kernels import pose_from_optimizer_vars

PPV = PHASE1_PER_PROBE_VARS


def build_y(
    arc_aps: np.ndarray,
    n_arcs: int,
    mls: np.ndarray,
    spins_deg: np.ndarray,
    offsets_R: np.ndarray,
    offsets_A: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    n_probes = len(mls)
    y = np.zeros(n_arcs + PHASE1_PER_PROBE_VARS * n_probes, dtype=np.float64)
    y[:n_arcs] = arc_aps
    for i in range(n_probes):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        rad = np.deg2rad(spins_deg[i])
        y[off + 0] = mls[i]
        y[off + 1] = np.cos(rad)
        y[off + 2] = np.sin(rad)
        y[off + 3] = offsets_R[i]
        y[off + 4] = offsets_A[i]
        y[off + 5] = depths[i]
    return y


def extract_spins(y: np.ndarray, n_arcs: int, n_probes: int) -> np.ndarray:
    out = np.zeros(n_probes, dtype=np.float64)
    for i in range(n_probes):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        out[i] = np.degrees(np.arctan2(y[off + 2], y[off + 1]))
    return out


def _poses(st, x, n_arcs):
    """Reconstruct (Rs (P,3,3), ts (P,3), tips (P,maxsh,3), mask (P,maxsh))
    from a Phase 1 x at this candidate's statics."""
    arc_aps = x[:n_arcs]
    Rs, ts, tips = [], [], []
    for i, s in enumerate(st):
        off = n_arcs + PPV * i
        ml, sx, sy, oR, oA, dep = x[off : off + 6]
        spin = float(np.degrees(np.arctan2(sy, sx)))
        R, tt = pose_from_optimizer_vars(
            target_LPS=jnp.asarray(s.target_LPS, jnp.float32),
            ap_deg=jnp.float32(arc_aps[s.arc_idx]),
            ml_deg=jnp.float32(ml),
            spin_deg=jnp.float32(spin),
            offset_R_mm=jnp.float32(oR),
            offset_A_mm=jnp.float32(oA),
            past_target_mm=jnp.float32(dep),
            recording_center_local=jnp.asarray(s.pivot_local, jnp.float32),
        )
        Rs.append(R)
        ts.append(tt)
        tips.append(np.asarray(s.shank_tips_local, np.float32))
    P = len(st)
    maxsh = max(len(t) for t in tips)
    tips_p = np.zeros((P, maxsh, 3), np.float32)
    mask_p = np.zeros((P, maxsh), np.float32)
    for i in range(P):
        tips_p[i, : len(tips[i])] = tips[i]
        mask_p[i, : len(tips[i])] = 1.0
    return jnp.stack(Rs), jnp.stack(ts), jnp.asarray(tips_p), jnp.asarray(mask_p)


def worst_threading_g(statics, x, n_arcs) -> float:
    """Worst-case threading ``g`` across all probes/shanks/sections at pose ``x``.

    ``g <= 0`` means the (margin-inset) shank centerline clears its bore oval;
    ``g > 0`` means it pierces the wall. Returns the single worst (max) ``g``
    over every probe — a scalar threading-feasibility summary for a final pose.

    Evaluated at the **full** world pose from :func:`_poses` (offsets and depth
    included), so it is valid for Phase-2 outputs where IPOPT optimizes the
    offsets — unlike :func:`joint_rerank._max_g_threading`, which assumes the
    reduced/atlas convention (offsets pinned to 0). A recorded diagnostic; the
    hard accept/reject gate is the FCL validator (which includes the implant).
    """
    from aind_low_point.optimization.objectives.reduced_jax import threading_g_matrix

    Rs, ts, _tips, _mk = _poses(statics, np.asarray(x, dtype=np.float64), n_arcs)
    Rs = np.asarray(Rs, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    worst = -np.inf
    for i, s in enumerate(statics):
        g = np.asarray(
            threading_g_matrix(
                jnp.asarray(Rs[i], jnp.float32),
                jnp.asarray(ts[i], jnp.float32),
                jnp.asarray(s.shank_tips_local, jnp.float32),
                s.section_axes,
                s.section_centers,
                s.section_e1,
                s.section_e2,
                s.section_cos_theta,
                s.section_sin_theta,
                s.section_a,
                s.section_b,
            )
        )
        if g.size:
            worst = max(worst, float(np.nanmax(g)))
    return worst if np.isfinite(worst) else 0.0


def _apply_x_to_plan_state(plan_state, x, statics, n_arcs):
    """Mutate plan_state to reflect Phase 1/2's 45-dim ``x``.

    Converts (sx, sy) → spin via atan2. Arc letters a/b/c/… in
    arc-idx order.
    """
    arc_aps = x[:n_arcs]
    arc_letters = [chr(ord("a") + i) for i in range(n_arcs)]
    plan_state.kinematics.arc_angles = {
        arc_letters[i]: float(arc_aps[i]) for i in range(n_arcs)
    }
    for i, st in enumerate(statics):
        off = n_arcs + PHASE1_PER_PROBE_VARS * i
        ml = float(x[off + 0])
        sx = float(x[off + 1])
        sy = float(x[off + 2])
        off_R = float(x[off + 3])
        off_A = float(x[off + 4])
        depth = float(x[off + 5])
        spin = float(np.degrees(np.arctan2(sy, sx)))
        plan = plan_state.probes[st.name]
        plan.arc_id = arc_letters[st.arc_idx]
        plan.bind_ap_to_arc = True
        plan.ap_local = 0.0
        plan.ml_local = ml
        plan.spin = spin
        plan.offsets_RA = (off_R, off_A)
        plan.past_target_mm = depth
