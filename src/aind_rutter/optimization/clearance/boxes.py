"""Oriented-bounding-box geometry: point-to-box SDF, box-to-box separation
and the surface samples a shank contributes.

A shank is thin enough that its voxel SDF cannot resolve it, so shank terms
use these analytic forms instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def obb_sdf(
    query_local: Array,  # (..., 3) in box-local frame
    half_extents: Array,  # (3,) box half-extents (must be > 0)
) -> Array:
    """Analytic signed distance from points to an axis-aligned box
    centred at the origin, in the box's own local frame.

    Negative inside, positive outside. Closed-form. To use for an
    oriented box at world pose ``(center, R, half_extents)``, transform
    queries to box-local first: ``q_local = (q_world - center) @ R``.

    Implementation note: the outside branch uses ``sqrt(sum(q⁺²) + ε²)``
    instead of the raw ``jnp.linalg.norm`` — keeps the gradient finite
    when the query is inside the box (``max(q, 0) = 0`` everywhere, raw
    norm grad ``= 0/0 = NaN``). The ε floor adds a sub-micron offset
    that's invisible against the inside branch.

    The inside branch keeps the hard ``jnp.max(q)``. A soft-max with
    ``logsumexp`` would bias the reported inside-distance toward zero
    (less-negative penetration), causing the optimizer to under-report
    collisions — non-conservative and dangerous. The soft-min top-k
    aggregation downstream already smooths face transitions across the
    many shank-corner samples per pair, so per-call C⁰ at the face
    transitions is acceptable.
    """
    q = jnp.abs(query_local) - half_extents
    q_pos = jnp.maximum(q, 0.0)
    outside = jnp.sqrt(jnp.sum(q_pos * q_pos, axis=-1) + 1e-12)
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return outside + inside


EMPTY_CLEARANCE_SENTINEL = 1e3  # mm — "definitely no collision" floor.

SHANK_SAMPLES_PER_BOX = 8  # 8 corners only


# 4 long-edge XY positions (one per long edge of the box)
_SHANK_LONG_EDGE_XY = jnp.array(
    [[-1, -1], [+1, -1], [-1, +1], [+1, +1]], dtype=jnp.float32
)  # (4, 2)


# Just the 2 Z corners (±1). Earlier versions had 4 long-edge interior
# samples (Z ∈ {-0.6, -0.2, 0.2, 0.6}) to catch edge-interior closest-pair
# cases on tilted crossings, but shank-shank now uses exact OBB-OBB SAT so
# the interior samples no longer add coverage — they only inflate the
# body-shank "shank corners in other-body's SDF" cost.
_SHANK_Z_FRACS = jnp.array([-1.0, 1.0], dtype=jnp.float32)  # (2,)


# Build (8, 3) sign matrix: 4 XY positions × 2 Z corners = 8 samples per box.
# Pre-computed module-level so each call avoids the gather setup.
_SHANK_SAMPLE_SIGNS = jnp.stack(
    [
        jnp.broadcast_to(_SHANK_LONG_EDGE_XY[:, None, 0], (4, 2)),
        jnp.broadcast_to(_SHANK_LONG_EDGE_XY[:, None, 1], (4, 2)),
        jnp.broadcast_to(_SHANK_Z_FRACS[None, :], (4, 2)),
    ],
    axis=-1,
).reshape(-1, 3)  # (8, 3)


def _obb_sample_points_local(
    centers: Array,  # (S, 3)
    half_extents: Array,  # (S, 3)
) -> Array:
    """Return ``(S, 24, 3)`` surface samples per axis-aligned box in
    its local frame: 8 corners + 16 interior samples along the 4 long
    (z-axis) edges of each box.

    The long-edge samples catch shank-shank edge-edge closest-pair
    cases that pure corner sampling misses (two tilted 10-mm boxes
    can be closest at an edge interior point, not a vertex).
    """
    # (S, 24, 3) = (S, 1, 3) + (1, 24, 3) * (S, 1, 3)
    return (
        centers[:, None, :] + _SHANK_SAMPLE_SIGNS[None, :, :] * half_extents[:, None, :]
    )


def shank_world_samples(
    R: Array,
    t: Array,
    centers_local: Array,
    halves_local: Array,
) -> Array:
    """Return ``(S * 24, 3)`` world-frame surface-sample positions of
    all shank boxes of one probe at pose ``(R, t)``.
    """
    local = _obb_sample_points_local(centers_local, halves_local)
    S = local.shape[0]
    return (local.reshape(-1, 3) @ R.T + t).reshape(S * SHANK_SAMPLES_PER_BOX, 3)


def obb_sdf_world_to_local(
    query_world: Array,  # (M, 3)
    R: Array,
    t: Array,
    center_local: Array,  # (3,) shank center in probe-local
    half_extents: Array,  # (3,) shank half-extents
) -> Array:
    """SDF of one OBB at world pose ``(R, t)`` with local
    ``(center, half_extents)``, evaluated at world ``query_world``.
    """
    # World->probe-local: q_local = R^T (q_world - t).
    # Then shift to box-local: q_box = q_local - center_local.
    q_local = (query_world - t) @ R - center_local
    return obb_sdf(q_local, half_extents)


def obb_obb_signed_distance(
    R_a: Array,
    t_a: Array,
    center_a: Array,
    halves_a: Array,
    R_b: Array,
    t_b: Array,
    center_b: Array,
    halves_b: Array,
) -> Array:
    """Signed distance between two oriented boxes via Separating Axis
    Theorem. Closed-form, exact, fully differentiable.

    Each box is defined by:
      - world pose ``(R, t)`` (rotation + translation of the box's
        local frame relative to world)
      - ``center_local``: box centre in the probe-local frame
      - ``halves``: half-extents along the box's three axes

    Returns positive when the boxes are separated (distance is the
    smallest positive separation across the 15 SAT candidate axes) and
    negative when interpenetrating (depth = ``-min overlap``).

    For thin perpendicular boxes (e.g., crossing shanks), the cross
    product axes catch interior crossings that surface-sample-based
    OBB SDF queries miss entirely.

    Implementation notes
    --------------------
    - World-frame box centres: ``ca_w = R_a @ center_a + t_a`` and same
      for b. SAT compares projections of both boxes onto candidate
      axes in *world* coords.
    - 15 axes: 3 face normals from A (columns of ``R_a``) + 3 from B +
      9 cross products of all (axis_a, axis_b) pairs.
    - Cross products with near-parallel axes get a tiny norm — guarded
      with ``where`` to substitute a unit-Z fallback (the SAT outcome
      from such an axis is dominated by other valid axes, so the
      fallback's projection is harmless as long as the result is
      *not* min-selected — masking via large ``+1e3`` works).
    """
    # World-frame box centres (centres are stored in probe-local).
    ca_w = R_a @ center_a + t_a
    cb_w = R_b @ center_b + t_b
    diff = cb_w - ca_w  # (3,) world-frame vector between centres

    # Axis bases in world: columns of R_a and R_b.
    axes_a = R_a  # (3, 3): each column is one OBB-A axis in world
    axes_b = R_b

    # ---- 6 face-normal candidate axes (3 from A + 3 from B) ----
    def _separation_along(axis_w: Array) -> Array:
        """Signed separation along ``axis_w`` (a (3,) world vector).

        Negative ⇒ overlap by that depth; positive ⇒ separated.
        """
        # Projection of A's half-box onto axis:
        # r_a = sum_i |halves_a[i] * (axis_a[i] . axis_w)|
        proj_a = jnp.sum(jnp.abs(axes_a * axis_w[None, :]).T * halves_a[:, None].T)
        # Simpler form: r_a = halves_a . |axes_a.T @ axis_w|
        proj_a = jnp.dot(halves_a, jnp.abs(axes_a.T @ axis_w))
        proj_b = jnp.dot(halves_b, jnp.abs(axes_b.T @ axis_w))
        centre_sep = jnp.abs(jnp.dot(diff, axis_w))
        return centre_sep - (proj_a + proj_b)

    sep_face_a = jax.vmap(_separation_along)(axes_a.T)  # (3,)
    sep_face_b = jax.vmap(_separation_along)(axes_b.T)  # (3,)

    # ---- 9 cross-product candidate axes (vmapped over the 3×3 i,j grid) ----
    # Parallel-axis guard: ``jnp.linalg.norm(axis)`` at axis=0 has NaN
    # gradient (0/0) which propagates through autodiff even when the
    # ``where`` masks the forward value out. Use soft-norm
    # ``sqrt(||axis||² + ε²)`` so the gradient stays finite when the
    # cross product collapses (parallel face normals).
    cross_axes = jnp.cross(axes_a.T[:, None, :], axes_b.T[None, :, :]).reshape(
        9, 3
    )  # cross(a_i, b_j), i-major/j-minor (== the old i,j loop order)

    def _cross_axis_sep(axis: Array) -> Array:
        sq = jnp.sum(axis * axis)
        soft_norm = jnp.sqrt(sq + jnp.float32(1e-12))
        is_valid = sq > jnp.float32(1e-12)
        sep = _separation_along(axis / soft_norm)  # finite gradient everywhere
        # When axes are parallel, the cross-product axis is degenerate;
        # substitute a very negative separation so max-selection skips it.
        return jnp.where(is_valid, sep, jnp.float32(-1e6))

    sep_cross = jax.vmap(_cross_axis_sep)(cross_axes)  # (9,)

    all_separations = jnp.concatenate(
        [sep_face_a, sep_face_b, sep_cross], axis=0
    )  # (15,)

    # SAT: the boxes are separated iff some axis has positive separation.
    # The actual signed distance is the MAX over all candidate separations
    # (the strongest separator). When all separations are negative, the
    # boxes overlap and the magnitude of max equals the penetration depth.
    return jnp.max(all_separations)


def boxes_sdf(points_world, R, t, centers, halves, mask):
    """Distance from world points to the nearest of one probe's shank boxes."""
    per_box = jax.vmap(
        lambda c, h: obb_sdf_world_to_local(points_world, R, t, c, h),
    )(centers, halves)  # (S, N)
    if mask is not None:
        per_box = jnp.where(
            mask.astype(bool)[:, None], per_box, EMPTY_CLEARANCE_SENTINEL
        )
    return per_box.min(axis=0)
