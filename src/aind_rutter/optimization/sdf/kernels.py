"""JAX-traceable kernels: pose math, SDF trilinear lookup, pairwise
signed clearance via SDF.

Companion to :mod:`aind_rutter.optimization.sdf` (which builds the
voxel grids). This module does the inner-loop work that needs to be
differentiable for constrained optimizer Jacobians.

The core function is :func:`pairwise_signed_clearance`: for two probes
``(a, b)`` at world poses ``(R_a, t_a)`` and ``(R_b, t_b)``, transform
``b``'s surface samples into ``a``'s canonical local frame, look up
``a``'s SDF at those points, take the min; symmetrise. The result is
signed (negative inside, positive outside) and smooth — including
through overlap, where FCL's BVH distance clamps at zero.

Use ``jax.grad(pairwise_signed_clearance)`` to get analytic gradients
w.r.t. the optimizer's variables; no finite-diff needed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# RAS → LPS sign flip applied to a (3, 3) rotation: R_lps = D R_ras D.
_D_RAS_TO_LPS = jnp.diag(jnp.array([-1.0, -1.0, 1.0]))


# Per-category gains on clearance violation magnitudes. After the
# 2026-05-24 cleanup, every category that uses analytic OBB SDF /
# OBB-OBB SAT shares one bounded-magnitude scaling. Voxel-SDF
# categories stay at 1.0 because their magnitudes are already mm-scale.
#
# Categories in use:
#   probe-probe
#     - body-body voxel-SDF top-k        (mm-native)
#     - body-shank OBB direction (SAT)   (thickness-bounded ≈ 0.024 mm)
#     - shank-shank OBB SAT              (thickness-bounded)
#   probe-fixture
#     - body voxel-SDF vs fixture        (mm-native)
#     - probe OBB vs fixture surface     (thickness-bounded)
#
# Why the gain: in a KKT solver Lagrange multipliers track constraint
# magnitudes; a small-magnitude constraint gets a small multiplier and
# the optimiser under-prioritises it. Shank crossings / OBB violations
# are physically more severe than equally-deep body-body grazing
# (probes can't be inserted) so we boost their KKT weight to keep them
# competitive with body-body in the merit function.
#
# Apply the gain to the violation magnitude (penalty form) or directly
# to the signed slack (constraint form), before any squaring — so
# first-order gradient contributions scale linearly with the gain.
SLACK_GAIN_BODY_BODY = 1.0
SLACK_GAIN_BODY_SHANK_CORNERS = 1.0  # voxel-SDF lookup → mm-native magnitude
SLACK_GAIN_BODY_SHANK_OBB = 100.0
SLACK_GAIN_SHANK_SHANK = 100.0
SLACK_GAIN_FIXTURE_BODY = 1.0
SLACK_GAIN_FIXTURE_OBB = 100.0


def unit_circle_penalty(sx: Array, sy: Array) -> Array:
    """Soft penalty pulling ``(sx, sy)`` toward the unit circle.

    Returns ``sum((sx² + sy² − 1)²)`` across all probes. The Patch B
    reparameterisation uses ``spin_deg = atan2(sy, sx)`` which only
    reads the DIRECTION of ``(sx, sy)``; magnitude is geometrically
    irrelevant but the optimizer can wander away from the unit circle
    (e.g. toward the origin where ``atan2`` gradients are undefined,
    or toward the bounds where the (sx, sy) Hessian is poorly
    conditioned).

    This penalty keeps the magnitude consistent across stages so
    poses are interchangeable between reduced / Phase 1 / Phase 2 and
    so any downstream consumer reading ``(sx, sy)`` as a unit vector
    gets a well-conditioned value.
    """
    radii_sq = sx * sx + sy * sy
    return jnp.sum((radii_sq - 1.0) ** 2)


def _rot_x(angle_rad: Array) -> Array:
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    return jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def _rot_y(angle_rad: Array) -> Array:
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    return jnp.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def _rot_z(angle_rad: Array) -> Array:
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    return jnp.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def arc_angles_to_rotation(ap_deg: Array, ml_deg: Array, spin_deg: Array) -> Array:
    """Convention-matched ``arc_angles_to_affine`` in JAX.

    Mirrors :func:`aind_mri_utils.arc_angles.arc_angles_to_affine` with
    ``invert_AP=True, invert_rotation=True`` (the AIND convention). The
    rotation maps probe canonical local frame to world (LPS), so a
    point ``p_local`` becomes ``R @ p_local`` in world coords.
    """
    # invert_AP=True, invert_rotation=True → angles in deg
    ap = -jnp.deg2rad(ap_deg)
    ml = jnp.deg2rad(ml_deg)
    spin = -jnp.deg2rad(spin_deg)
    # RAS-frame Euler XYZ: RX(ap) RY(ml) RZ(spin)
    R_ras = _rot_x(ap) @ _rot_y(ml) @ _rot_z(spin)
    # RAS → LPS conjugation
    return _D_RAS_TO_LPS @ R_ras @ _D_RAS_TO_LPS


def pose_from_optimizer_vars(
    target_LPS: Array,
    ap_deg: Array,
    ml_deg: Array,
    spin_deg: Array,
    offset_R_mm: Array,
    offset_A_mm: Array,
    past_target_mm: Array,
    recording_center_local: Array,
) -> tuple[Array, Array]:
    """JAX-traceable companion of
    :func:`optimization.kinematics.pose_from_optimizer_vars`.

    Returns ``(R, pose_tip_world)``. ``pose_tip_world`` is the position
    of the probe's local origin (= shank-0 tip in canonical) such that
    the recording-array centre lands at ``target + offset_LPS −
    past_target · shaft_dir``.
    """
    R = arc_angles_to_rotation(ap_deg, ml_deg, spin_deg)
    # off_RAS = (off_R, off_A, 0) → off_LPS = (-off_R, -off_A, 0)
    off_LPS = jnp.stack([-offset_R_mm, -offset_A_mm, jnp.zeros_like(offset_R_mm)])
    adjusted_target = target_LPS + off_LPS
    zero = jnp.zeros_like(past_target_mm)
    insertion_vec = R @ jnp.stack([zero, zero, -past_target_mm])
    pose_tip = adjusted_target + insertion_vec - R @ recording_center_local
    return R, pose_tip


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


def spin_deg_from_sxy(sx: Array, sy: Array) -> Array:
    """Convert ``(sx, sy)`` rotation parameterization to ``spin_deg``.

    The optimizer's reduced/full y vector parameterizes spin as a 2D
    vector ``(sx, sy) ∝ (cos θ, sin θ)`` on the unit circle, avoiding
    the ±180° wraparound discontinuity that bound-clipped scalar-angle
    scalar-angle layout. Internally, the existing
    :func:`pose_from_optimizer_vars` API still takes ``spin_deg`` so we
    convert at the unpacking site via ``atan2(sy, sx)``.

    The norm of ``(sx, sy)`` is irrelevant — only the direction
    matters. JAX's ``arctan2`` is well-defined and differentiable
    everywhere except at the origin; optimizer bounds keep us away from
    it.
    """
    return jnp.degrees(jnp.arctan2(sy, sx))


def smooth_abs(x: Array, eps: float = 1e-3) -> Array:
    """Smooth approximation to ``|x|`` via ``sqrt(x² + ε²)``. Continuous
    derivative everywhere (vs ``abs``'s sign-flip at zero).

    Default ε = 1e-3 mm/deg keeps the soft region tight around zero —
    only meaningfully different from ``abs`` when ``|x| < a few ε``.
    """
    return jnp.sqrt(x * x + eps * eps)


def trilinear_sdf(
    grid: Array,
    origin: Array,
    spacing: Array,
    query_local: Array,
    out_of_bounds_value: Array = jnp.array(1e3),
    n_real: Array | None = None,
) -> Array:
    """Trilinear interpolation of an SDF voxel grid at ``query_local``
    points (which must already be in the probe's canonical local frame).

    Parameters
    ----------
    grid : (Nx, Ny, Nz) array of signed distances (mm).
    origin : (3,) the local-frame position of ``grid[0, 0, 0]``.
    spacing : scalar voxel edge length (mm).
    query_local : (..., 3) query points in the same local frame as the grid.
    out_of_bounds_value : scalar returned for points outside the grid bbox.
        Default 1e3 mm — "definitely far positive", safe for clearance.
    n_real : optional (3,) real grid extent ``(nx, ny, nz)``. When the grid is a
        padded slot of a uniform table (the swept-pair table — see
        ``clearance_sweep``), the true extent is smaller than ``grid.shape``;
        pass it so in-bounds / clip use the real extent and out-of-extent queries
        return ``out_of_bounds_value`` exactly as for the unpadded grid. May be a
        traced (dynamic) value so it works under ``vmap`` gather. ``None`` ⇒ use
        ``grid.shape`` (unchanged behaviour for every existing caller).

    Returns
    -------
    (...,) interpolated signed distances. Differentiable w.r.t.
    ``query_local`` and ``grid``.
    """
    grid = jnp.asarray(grid)
    # Value dtype follows the grid: fp32 normally, bf16 when the grid is
    # stored bf16 (mixed-precision kernel — bf16 gather/blend, fp32 output).
    vdt = grid.dtype
    if n_real is None:
        Nx, Ny, Nz = grid.shape
    else:
        Nx, Ny, Nz = n_real[0], n_real[1], n_real[2]
    # Addressing stays fp32 — voxel indices must be exact regardless of the
    # value precision.
    coords = (
        query_local.astype(jnp.float32) - jnp.asarray(origin, jnp.float32)
    ) / jnp.asarray(spacing, jnp.float32)  # (..., 3) in voxel units
    i0 = jnp.floor(coords).astype(jnp.int32)
    f = coords - i0  # fractional parts

    in_bounds = (
        (i0[..., 0] >= 0)
        & (i0[..., 0] < Nx - 1)
        & (i0[..., 1] >= 0)
        & (i0[..., 1] < Ny - 1)
        & (i0[..., 2] >= 0)
        & (i0[..., 2] < Nz - 1)
    )
    ix = jnp.clip(i0[..., 0], 0, Nx - 2)
    iy = jnp.clip(i0[..., 1], 0, Ny - 2)
    iz = jnp.clip(i0[..., 2], 0, Nz - 2)
    # Interp weights carried at the value dtype (bf16 when the grid is bf16).
    fx, fy, fz = (
        f[..., 0].astype(vdt),
        f[..., 1].astype(vdt),
        f[..., 2].astype(vdt),
    )

    c000 = grid[ix, iy, iz]
    c100 = grid[ix + 1, iy, iz]
    c010 = grid[ix, iy + 1, iz]
    c110 = grid[ix + 1, iy + 1, iz]
    c001 = grid[ix, iy, iz + 1]
    c101 = grid[ix + 1, iy, iz + 1]
    c011 = grid[ix, iy + 1, iz + 1]
    c111 = grid[ix + 1, iy + 1, iz + 1]

    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    # interp value carried at the grid dtype; cast back to fp32 so the
    # cross-point reduction (soft-min / top-k) downstream accumulates fp32.
    interp = (c0 * (1 - fz) + c1 * fz).astype(jnp.float32)
    return jnp.where(in_bounds, interp, jnp.asarray(out_of_bounds_value, jnp.float32))


def pairwise_signed_clearance_probe_obb_fixture_world(
    R_p: Array,
    t_p: Array,
    fixture_surface_world: Array,  # (Nf, 3) fixture envelope samples in world LPS
    shank_centers: Array,  # (S, 3) probe OBB centers in probe local
    shank_halves: Array,  # (S, 3) probe OBB half-extents
    *,
    beta: float = 20.0,
    top_k: int = 8,
    shank_mask: Array | None = None,
) -> tuple[Array, Array]:
    """Probe-OBB vs fixture clearance via fixture surface samples →
    probe OBB analytic SDF.

    Symmetric counterpart to the body-shank-OBB direction in
    :func:`shank_only_pair_clearance`: dense surface samples on one
    side + analytic OBB SDF on the other. Catches probe shank /
    transition-zone OBB vs fixture geometry (e.g. probe body-bottom
    grazing the well bore rim — invisible to the body-vs-fixture-body
    SDF check because the body α-wrap closing cap under-inflates at
    the shank-strip boundary).

    ``shank_mask`` (``(S,)`` bool/0-1, optional) marks which OBB rows are real vs
    padding when the per-kind OBB table is padded to a uniform ``max_S``; invalid
    rows are set to the no-collision sentinel so they never win the min/soft-min.
    ``None`` ⇒ no padding (legacy), identical to before.

    Returns ``(hard_min, soft_min)`` over the pool of all
    (fixture_point, probe_OBB) signed distances. Negative ⇒ at least
    one fixture surface point lies inside the probe OBB.
    """
    S = shank_centers.shape[0]
    if S == 0:
        return (
            jnp.float32(1e3),
            jnp.float32(1e3),
        )
    # For each probe OBB, signed distance of each fixture surface point
    # to that OBB. Shape: (S, Nf).
    d_obbs = jax.vmap(
        lambda c, h: _obb_sdf_world_to_local(fixture_surface_world, R_p, t_p, c, h)
    )(shank_centers, shank_halves)
    if shank_mask is not None:
        d_obbs = jnp.where(
            shank_mask.astype(bool)[:, None], d_obbs, _EMPTY_CLEARANCE_SENTINEL
        )
    pool = d_obbs.reshape(-1)
    return _hard_soft(pool, beta=beta, top_k=top_k)


def pairwise_signed_clearance_probe_fixture_body_world(
    R_p: Array,
    t_p: Array,
    sdf_p_grid: Array,
    sdf_p_origin: Array,
    sdf_p_spacing: Array,
    sdf_f_grid: Array,
    sdf_f_origin: Array,
    sdf_f_spacing: Array,
    world_surface_p: Array,  # (Np, 3) pre-transformed probe envelope samples in world
    surface_f: Array,  # (Nf, 3) fixture envelope samples in world LPS
    *,
    beta: float = 20.0,
    top_k: int = 16,
    n_real_p: Array | None = None,
    n_real_f: Array | None = None,
) -> tuple[Array, Array]:
    """Same as :func:`pairwise_signed_clearance_probe_fixture_body` but
    takes pre-transformed world-frame probe surface samples. Callers
    iterating over ``n_fixtures × n_probes`` pairs should hoist
    ``world_surface[i] = surface[i] @ R[i].T + t[i]`` once per probe
    (per Phase 1's body-body hoist).

    ``n_real_p`` is the probe grid's real (unpadded) extent, for the fixture-
    surface-vs-probe-grid query when ``sdf_p_grid`` is a padded slot of a
    per-kind table (see clearance_sweep). ``n_real_f`` is the same for the
    fixture grid, for the probe-surface-vs-fixture-grid query when the fixtures
    are stacked into a padded table to share a vmap axis. Both ``None`` ⇒
    unchanged (un-padded constant grid).
    """
    kw_p = {} if n_real_p is None else {"n_real": n_real_p}
    kw_f = {} if n_real_f is None else {"n_real": n_real_f}

    d_p_in_f = trilinear_sdf(
        sdf_f_grid,
        sdf_f_origin,
        sdf_f_spacing,
        world_surface_p,
        **kw_f,
    )

    local_f_in_p = (surface_f - t_p) @ R_p
    d_f_in_p = trilinear_sdf(
        sdf_p_grid, sdf_p_origin, sdf_p_spacing, local_f_in_p, **kw_p
    )

    distances = jnp.concatenate([d_p_in_f.reshape(-1), d_f_in_p.reshape(-1)])
    hard_min = jnp.min(distances)
    soft = soft_min_topk(distances, beta=beta, top_k=top_k)
    return hard_min, soft


_SHANK_SAMPLES_PER_BOX = 8  # 8 corners only

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


def _shank_world_samples(
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
    return (local.reshape(-1, 3) @ R.T + t).reshape(S * _SHANK_SAMPLES_PER_BOX, 3)


def _obb_sdf_world_to_local(
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


_EMPTY_CLEARANCE_SENTINEL = 1e3  # mm — "definitely no collision" floor.


def _hard_soft(values: Array, *, beta: float, top_k: int) -> tuple[Array, Array]:
    """Return ``(hard_min, soft_min_topk)`` for a (possibly empty) 1-D
    sample vector. Empty pools return ``(sentinel, sentinel)`` so the
    caller's ReLU penalty is silent.
    """
    if values.size == 0:
        s = jnp.asarray(_EMPTY_CLEARANCE_SENTINEL, dtype=jnp.float32)
        return s, s
    return jnp.min(values), soft_min_topk(values, beta=beta, top_k=top_k)


def shank_only_pair_clearance(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    world_surface_a: Array,
    world_surface_b: Array,
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k_body_shank: int = 8,
    top_k_shank_shank: int = 8,
    shank_mask_a: Array | None = None,
    shank_mask_b: Array | None = None,
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """Shank-related dual-rep categories that DON'T use trilinear SDF.

    Computes:
    - **body-shank, OBB direction only**: ``body_samples @ each-OBB-SDF``
      (analytic). The trilinear "shank corners → other body's SDF"
      direction is omitted — caller computes that per-pair if needed.
    - **shank-shank SAT**: exact OBB-vs-OBB distance via 15 candidate
      axes. No SDF, no gather, fully analytic.

    Returns ``((hbs_obb_only, sbs_obb_only), (hss, sss))``. Both pools
    are exact analytic distances (no soft-min bias from gather discont-
    inuities).

    Vmap-friendly: assumes uniform shank shapes ``(S, 3)`` across pairs
    (post-2026-05-23 OBB-union, S=2 for every probe). The internal
    inner vmaps over (Sa, Sb) compose with an outer vmap over pairs to
    give a single XLA launch.
    """
    Sa = shank_centers_a.shape[0]
    Sb = shank_centers_b.shape[0]

    body_shank_chunks = []
    if Sa > 0:
        d_body_b_vs_a_obbs = jax.vmap(
            lambda c, h: _obb_sdf_world_to_local(world_surface_b, R_a, t_a, c, h)
        )(shank_centers_a, shank_halves_a)  # (Sa, Nbody)
        if shank_mask_a is not None:
            d_body_b_vs_a_obbs = jnp.where(
                shank_mask_a.astype(bool)[:, None],
                d_body_b_vs_a_obbs,
                _EMPTY_CLEARANCE_SENTINEL,
            )
        body_shank_chunks.append(d_body_b_vs_a_obbs.reshape(-1))
    if Sb > 0:
        d_body_a_vs_b_obbs = jax.vmap(
            lambda c, h: _obb_sdf_world_to_local(world_surface_a, R_b, t_b, c, h)
        )(shank_centers_b, shank_halves_b)
        if shank_mask_b is not None:
            d_body_a_vs_b_obbs = jnp.where(
                shank_mask_b.astype(bool)[:, None],
                d_body_a_vs_b_obbs,
                _EMPTY_CLEARANCE_SENTINEL,
            )
        body_shank_chunks.append(d_body_a_vs_b_obbs.reshape(-1))
    body_shank_pool = (
        jnp.concatenate(body_shank_chunks, axis=0)
        if body_shank_chunks
        else jnp.zeros((0,), dtype=world_surface_a.dtype)
    )

    return (
        _hard_soft(body_shank_pool, beta=beta, top_k=top_k_body_shank),
        shank_shank_pair_clearance(
            R_a,
            t_a,
            R_b,
            t_b,
            shank_centers_a,
            shank_halves_a,
            shank_centers_b,
            shank_halves_b,
            beta=beta,
            top_k=top_k_shank_shank,
            shank_mask_a=shank_mask_a,
            shank_mask_b=shank_mask_b,
            dtype=world_surface_a.dtype,
        ),
    )


def shank_shank_pair_clearance(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k: int = 8,
    shank_mask_a: Array | None = None,
    shank_mask_b: Array | None = None,
    dtype=jnp.float32,
) -> tuple[Array, Array]:
    """Shank-shank category: exact OBB-vs-OBB distance over 15 separating axes for
    every shank pair, masked rows reading the no-collision sentinel."""
    Sa, Sb = shank_centers_a.shape[0], shank_centers_b.shape[0]
    if Sa == 0 or Sb == 0:
        return _hard_soft(jnp.zeros((0,), dtype=dtype), beta=beta, top_k=top_k)

    d_sa_vs_sb = jax.vmap(
        lambda ca, ha: jax.vmap(
            lambda cb, hb: obb_obb_signed_distance(R_a, t_a, ca, ha, R_b, t_b, cb, hb)
        )(shank_centers_b, shank_halves_b)
    )(shank_centers_a, shank_halves_a)  # (Sa, Sb)
    if shank_mask_a is not None or shank_mask_b is not None:
        ma = (
            shank_mask_a.astype(bool)
            if shank_mask_a is not None
            else jnp.ones((Sa,), bool)
        )
        mb = (
            shank_mask_b.astype(bool)
            if shank_mask_b is not None
            else jnp.ones((Sb,), bool)
        )
        d_sa_vs_sb = jnp.where(
            ma[:, None] & mb[None, :], d_sa_vs_sb, _EMPTY_CLEARANCE_SENTINEL
        )
    return _hard_soft(d_sa_vs_sb.reshape(-1), beta=beta, top_k=top_k)


def body_shank_corners_pair_clearance(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    sdf_a_grid: Array,
    sdf_a_origin: Array,
    sdf_a_spacing: Array,
    sdf_b_grid: Array,
    sdf_b_origin: Array,
    sdf_b_spacing: Array,
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k: int = 8,
    shank_mask_a: Array | None = None,
    shank_mask_b: Array | None = None,
    n_real_a: Array | None = None,
    n_real_b: Array | None = None,
) -> tuple[Array, Array]:
    """Body-shank "shank corners → other body's SDF" direction ONLY.

    Trilinear-gather path; on CPU its gradient suffers the scatter-
    contention slowdown when vmap'd across pairs (see
    [[vmap-cpu-gpu-polish-arch]]). Kept as its own helper so callers
    can decide whether to per-pair-loop it (CPU) or vmap it (GPU).

    ``shank_mask_{a,b}`` (``(S,)`` bool/0-1, optional) mark which OBB rows
    are real vs padding when the per-kind OBB tables are padded to a uniform
    ``max_S``: each invalid OBB's pooled distances are set to the
    "no-collision" sentinel so they never win the min/soft-min. ``None`` ⇒
    no padding (legacy ragged tables), identical to before.
    """
    kw_a = {} if n_real_a is None else {"n_real": n_real_a}
    kw_b = {} if n_real_b is None else {"n_real": n_real_b}
    Sa = shank_centers_a.shape[0]
    Sb = shank_centers_b.shape[0]
    chunks = []
    if Sa > 0:
        corners_a_world = _shank_world_samples(
            R_a, t_a, shank_centers_a, shank_halves_a
        )
        ca_in_b = (corners_a_world - t_b) @ R_b
        d_corners_a_in_b = trilinear_sdf(
            sdf_b_grid, sdf_b_origin, sdf_b_spacing, ca_in_b, **kw_b
        ).reshape(-1)
        if shank_mask_a is not None:
            m = jnp.repeat(shank_mask_a.astype(bool), _SHANK_SAMPLES_PER_BOX)
            d_corners_a_in_b = jnp.where(m, d_corners_a_in_b, _EMPTY_CLEARANCE_SENTINEL)
        chunks.append(d_corners_a_in_b)
    if Sb > 0:
        corners_b_world = _shank_world_samples(
            R_b, t_b, shank_centers_b, shank_halves_b
        )
        cb_in_a = (corners_b_world - t_a) @ R_a
        d_corners_b_in_a = trilinear_sdf(
            sdf_a_grid, sdf_a_origin, sdf_a_spacing, cb_in_a, **kw_a
        ).reshape(-1)
        if shank_mask_b is not None:
            m = jnp.repeat(shank_mask_b.astype(bool), _SHANK_SAMPLES_PER_BOX)
            d_corners_b_in_a = jnp.where(m, d_corners_b_in_a, _EMPTY_CLEARANCE_SENTINEL)
        chunks.append(d_corners_b_in_a)
    pool = (
        jnp.concatenate(chunks, axis=0) if chunks else jnp.zeros((0,), dtype=R_a.dtype)
    )
    return _hard_soft(pool, beta=beta, top_k=top_k)


def body_body_pair_clearance(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    sdf_a_grid: Array,
    sdf_a_origin: Array,
    sdf_a_spacing: Array,
    sdf_b_grid: Array,
    sdf_b_origin: Array,
    sdf_b_spacing: Array,
    world_surface_a: Array,
    world_surface_b: Array,
    *,
    beta: float = 20.0,
    top_k: int = 16,
    n_real_a: Array | None = None,
    n_real_b: Array | None = None,
) -> tuple[Array, Array]:
    """Body-body category only of the dual-rep clearance. Returns
    ``(hard_min, soft_min_topk)``. Vmap-friendly: shape-static (no
    Python ``if`` branches), uniform per-probe inputs, suitable for
    ``jax.vmap`` across a pair axis.

    Hoisting this out of ``pairwise_signed_clearance_dual_world`` lets
    the caller batch all P probe-pairs into ONE XLA kernel launch
    instead of P separate launches per ``jit_obj`` call — the unrolled
    Python pair loop was responsible for ~50% of the objective wall (per
    2026-05-23 jax.profiler trace).
    """
    kw_a = {} if n_real_a is None else {"n_real": n_real_a}
    kw_b = {} if n_real_b is None else {"n_real": n_real_b}
    local_b_in_a = (world_surface_b - t_a) @ R_a
    d_b_in_a = trilinear_sdf(
        sdf_a_grid, sdf_a_origin, sdf_a_spacing, local_b_in_a, **kw_a
    )
    local_a_in_b = (world_surface_a - t_b) @ R_b
    d_a_in_b = trilinear_sdf(
        sdf_b_grid, sdf_b_origin, sdf_b_spacing, local_a_in_b, **kw_b
    )
    pool = jnp.concatenate([d_b_in_a.reshape(-1), d_a_in_b.reshape(-1)])
    return jnp.min(pool), soft_min_topk(pool, beta=beta, top_k=top_k)


# Clearance read for padded cell slots; large enough never to be the minimum.
_PAD_CLEARANCE_MM = 1e3


def trilinear_sdf_stacked(
    grids: Array,
    origins: Array,
    spacings: Array,
    n_reals: Array,
    which: Array,
    query_local: Array,
    out_of_bounds_value: float = 1e3,
) -> Array:
    """:func:`trilinear_sdf` over a table of same-shape grids, reading each query
    point from grid ``which``.

    One gather serves points bound for different probes' SDFs; separate lookups
    would each have to cover every point. ``grids`` is (K, Nx, Ny, Nz) with real
    extents ``n_reals`` (K, 3), ``origins`` (K, 3) and ``spacings`` (K,); ``which``
    is an integer array broadcastable to ``query_local.shape[:-1]``.
    """
    which = jnp.broadcast_to(which, query_local.shape[:-1])
    n = n_reals[which]
    coords = (
        query_local.astype(jnp.float32) - jnp.asarray(origins, jnp.float32)[which]
    ) / jnp.asarray(spacings, jnp.float32)[which][..., None]
    i0 = jnp.floor(coords).astype(jnp.int32)
    f = (coords - i0).astype(grids.dtype)
    in_bounds = jnp.all((i0 >= 0) & (i0 < n - 1), axis=-1)
    idx = jnp.clip(i0, 0, n - 2)
    ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    def corner(dx: int, dy: int, dz: int) -> Array:
        return grids[which, ix + dx, iy + dy, iz + dz]

    c00 = corner(0, 0, 0) * (1 - fx) + corner(1, 0, 0) * fx
    c01 = corner(0, 0, 1) * (1 - fx) + corner(1, 0, 1) * fx
    c10 = corner(0, 1, 0) * (1 - fx) + corner(1, 1, 0) * fx
    c11 = corner(0, 1, 1) * (1 - fx) + corner(1, 1, 1) * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    interp = (c0 * (1 - fz) + c1 * fz).astype(jnp.float32)
    return jnp.where(in_bounds, interp, jnp.asarray(out_of_bounds_value, jnp.float32))


def body_body_pair_clearance_c2f(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    kind_a: Array,
    kind_b: Array,
    grids: Array,
    origins: Array,
    spacings: Array,
    n_reals: Array,
    outside_min: Array,
    coarse: Array,
    fine: Array,
    cells: Array,
    radius: Array,
    *,
    beta: float = 20.0,
    top_k: int = 16,
    n_cells: int = 16,
) -> tuple[Array, Array]:
    """Body-body clearance from coarse-to-fine surface samples, returning
    ``(hard_min, soft_min_topk)`` like :func:`body_body_pair_clearance`.

    Looks up each probe's coarse points in the other probe's SDF, ranks cells by
    ``coarse value − cell radius`` (no fine point in a cell reads lower, since a
    distance field changes no faster than its query point moves), and looks up the
    fine points of the ``n_cells`` best cells. Per-kind tables in each kind's local
    frame: ``coarse`` (K, C, 3), ``fine`` (K, F, 3), ``cells`` (K, C, W) listing fine
    indices padded with F, ``radius`` (K, C); grids as for
    :func:`trilinear_sdf_stacked`. ``outside_min`` (K,) is the lowest value on each
    grid's boundary: a point off the grid is at least that far from the body, which
    stands in for the out-of-grid sentinel when ranking, so a cell whose coarse point
    is off the grid but whose members are close still ranks by a valid bound. See
    :mod:`aind_rutter.optimization.sdf.surface_samples`.
    """
    n_coarse, n_fine = coarse.shape[1], fine.shape[1]

    def into(points: Array, R_src: Array, t_src: Array, R_dst: Array, t_dst: Array):
        return (points @ R_src.T + t_src - t_dst) @ R_dst

    def lookup(kind: Array, points: Array) -> Array:
        return trilinear_sdf_stacked(grids, origins, spacings, n_reals, kind, points)

    coarse_values = jnp.concatenate(
        [
            lookup(kind_a, into(coarse[kind_b], R_b, t_b, R_a, t_a)),
            lookup(kind_b, into(coarse[kind_a], R_a, t_a, R_b, t_b)),
        ]
    )
    off_grid = coarse_values >= _PAD_CLEARANCE_MM
    floor = jnp.concatenate(
        [
            jnp.broadcast_to(outside_min[kind_a], (n_coarse,)),
            jnp.broadcast_to(outside_min[kind_b], (n_coarse,)),
        ]
    )
    bound = jnp.where(off_grid, floor, coarse_values) - jnp.concatenate(
        [radius[kind_b], radius[kind_a]]
    )
    _, chosen = jax.lax.top_k(-bound, n_cells)
    of_a = chosen >= n_coarse  # the cell holds probe a's points, read in b's SDF
    owner = jnp.where(of_a, kind_a, kind_b)
    members = cells[owner, jnp.where(of_a, chosen - n_coarse, chosen)]  # (n_cells, W)
    points = fine[owner[:, None], jnp.minimum(members, n_fine - 1)]
    R_src = jnp.where(of_a[:, None, None], R_a, R_b)
    t_src = jnp.where(of_a[:, None], t_a, t_b)
    R_dst = jnp.where(of_a[:, None, None], R_b, R_a)
    t_dst = jnp.where(of_a[:, None], t_b, t_a)
    world = jnp.einsum("cwj,ckj->cwk", points, R_src) + t_src[:, None, :]
    local = jnp.einsum("cwk,ckj->cwj", world - t_dst[:, None, :], R_dst)
    fine_values = lookup(jnp.where(of_a, kind_b, kind_a)[:, None], local)
    fine_values = jnp.where(members < n_fine, fine_values, _PAD_CLEARANCE_MM)
    pool = jnp.concatenate([coarse_values, fine_values.reshape(-1)])
    return jnp.min(pool), soft_min_topk(pool, beta=beta, top_k=top_k)


def _boxes_sdf(points_world, R, t, centers, halves, mask):
    """Distance from world points to the nearest of one probe's shank boxes."""
    per_box = jax.vmap(
        lambda c, h: _obb_sdf_world_to_local(points_world, R, t, c, h),
    )(centers, halves)  # (S, N)
    if mask is not None:
        per_box = jnp.where(
            mask.astype(bool)[:, None], per_box, _EMPTY_CLEARANCE_SENTINEL
        )
    return per_box.min(axis=0)


def body_shank_box_clearance_c2f(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    kind_a: Array,
    kind_b: Array,
    coarse: Array,
    fine: Array,
    cells: Array,
    radius: Array,
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k: int = 8,
    n_cells: int = 16,
    shank_mask_a: Array | None = None,
    shank_mask_b: Array | None = None,
) -> tuple[Array, Array]:
    """Body-vs-shank-box category from the coarse-to-fine body samples, returning
    ``(hard_min, soft_min_topk)`` like the body-vs-OBB pool of
    :func:`shank_only_pair_clearance`.

    Each probe's coarse body points are measured against the other probe's shank
    boxes, cells are ranked by ``coarse value − cell radius`` (the box distance is
    also 1-Lipschitz, so no cell member reads lower), and the fine points of the
    ``n_cells`` best cells are measured. Sample tables are those of
    :func:`body_body_pair_clearance_c2f`; masked-out boxes and padded cell slots
    read the no-collision sentinel.
    """
    n_coarse, n_fine = coarse.shape[1], fine.shape[1]

    def to_world(points, R, t):
        return points @ R.T + t

    coarse_values = jnp.concatenate(
        [
            _boxes_sdf(
                to_world(coarse[kind_b], R_b, t_b),
                R_a,
                t_a,
                shank_centers_a,
                shank_halves_a,
                shank_mask_a,
            ),
            _boxes_sdf(
                to_world(coarse[kind_a], R_a, t_a),
                R_b,
                t_b,
                shank_centers_b,
                shank_halves_b,
                shank_mask_b,
            ),
        ]
    )
    bound = coarse_values - jnp.concatenate([radius[kind_b], radius[kind_a]])
    _, chosen = jax.lax.top_k(-bound, n_cells)
    of_a = chosen >= n_coarse  # the cell holds probe a's points, read against b
    owner = jnp.where(of_a, kind_a, kind_b)
    members = cells[owner, jnp.where(of_a, chosen - n_coarse, chosen)]  # (n_cells, W)
    points = fine[owner[:, None], jnp.minimum(members, n_fine - 1)]
    source_R = jnp.where(of_a[:, None, None], R_a, R_b)
    source_t = jnp.where(of_a[:, None], t_a, t_b)
    world = jnp.einsum("cwj,ckj->cwk", points, source_R) + source_t[:, None, :]
    flat = world.reshape(-1, 3)
    against_a = _boxes_sdf(
        flat, R_a, t_a, shank_centers_a, shank_halves_a, shank_mask_a
    )
    against_b = _boxes_sdf(
        flat, R_b, t_b, shank_centers_b, shank_halves_b, shank_mask_b
    )
    fine_values = jnp.where(
        jnp.repeat(of_a, members.shape[1]), against_b, against_a
    ).reshape(members.shape)
    fine_values = jnp.where(members < n_fine, fine_values, _EMPTY_CLEARANCE_SENTINEL)
    pool = jnp.concatenate([coarse_values, fine_values.reshape(-1)])
    return _hard_soft(pool, beta=beta, top_k=top_k)


# ---------------------------------------------------------------------------
# Dual-rep aggregator helpers (reduced / Phase 1 / Phase 2 / validation)
# ---------------------------------------------------------------------------


class PairClearance(NamedTuple):
    """Dual-rep clearance between two probes, all four categories.

    Each field is ``(hard_min, soft_min_topk)`` of that category's
    signed-distance pool. Positive ⇒ clear by that distance; negative
    ⇒ penetration. Soft is the smooth surrogate used in objective /
    constraint slack form; hard is the exact min used for the
    saturating-margin reward.
    """

    body_body: tuple[Array, Array]  # voxel-SDF, mm-native
    body_shank_corners: tuple[Array, Array]  # shank OBB corners → voxel SDF
    body_shank_obb: tuple[Array, Array]  # body samples → shank OBB SDF
    shank_shank: tuple[Array, Array]  # OBB-OBB exact SAT


class FixtureClearance(NamedTuple):
    """Dual-rep clearance between one probe and one fixture.

    Each field is ``(hard_min, soft_min_topk)``.
    """

    body: tuple[Array, Array]  # body voxel-SDF vs fixture surface samples
    obb: tuple[Array, Array]  # probe OBBs SDF vs fixture surface samples


# Per-category gains, ordered to match the NamedTuple fields. Sites
# that scale slacks by per-category importance (reduced penalty,
# Phase 1 penalty, Phase 2 constraint, etc.) can ``zip`` over
# ``pc.softs`` and the appropriate gain tuple without re-inlining the
# constant list.
PROBE_PAIR_SLACK_GAINS = (
    SLACK_GAIN_BODY_BODY,
    SLACK_GAIN_BODY_SHANK_CORNERS,
    SLACK_GAIN_BODY_SHANK_OBB,
    SLACK_GAIN_SHANK_SHANK,
)

FIXTURE_PAIR_SLACK_GAINS = (
    SLACK_GAIN_FIXTURE_BODY,
    SLACK_GAIN_FIXTURE_OBB,
)


def dual_rep_pair_clearance(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    sdf_a_grid: Array,
    sdf_a_origin: Array,
    sdf_a_spacing: Array,
    sdf_b_grid: Array,
    sdf_b_origin: Array,
    sdf_b_spacing: Array,
    world_surface_a: Array,
    world_surface_b: Array,
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k_body_body: int = 16,
    top_k_body_shank: int = 8,
    top_k_shank_shank: int = 8,
) -> PairClearance:
    """All four dual-rep probe-pair clearance categories in one call.

    Equivalent to invoking :func:`body_body_pair_clearance`,
    :func:`body_shank_corners_pair_clearance`, and
    :func:`shank_only_pair_clearance` separately and packing the
    results. JAX-traceable, no behavioural change vs the inline
    pattern shared across reduced / Phase 1 / Phase 2 /
    Phase 3 — purely a packaging convenience that gives every site one
    source of truth for "what counts as dual-rep probe-pair clearance".
    """
    body_body = body_body_pair_clearance(
        R_a,
        t_a,
        R_b,
        t_b,
        sdf_a_grid,
        sdf_a_origin,
        sdf_a_spacing,
        sdf_b_grid,
        sdf_b_origin,
        sdf_b_spacing,
        world_surface_a,
        world_surface_b,
        beta=beta,
        top_k=top_k_body_body,
    )
    body_shank_corners = body_shank_corners_pair_clearance(
        R_a,
        t_a,
        R_b,
        t_b,
        sdf_a_grid,
        sdf_a_origin,
        sdf_a_spacing,
        sdf_b_grid,
        sdf_b_origin,
        sdf_b_spacing,
        shank_centers_a,
        shank_halves_a,
        shank_centers_b,
        shank_halves_b,
        beta=beta,
        top_k=top_k_body_shank,
    )
    body_shank_obb, shank_shank = shank_only_pair_clearance(
        R_a,
        t_a,
        R_b,
        t_b,
        world_surface_a,
        world_surface_b,
        shank_centers_a,
        shank_halves_a,
        shank_centers_b,
        shank_halves_b,
        beta=beta,
        top_k_body_shank=top_k_body_shank,
        top_k_shank_shank=top_k_shank_shank,
    )
    return PairClearance(
        body_body=body_body,
        body_shank_corners=body_shank_corners,
        body_shank_obb=body_shank_obb,
        shank_shank=shank_shank,
    )


def dual_rep_fixture_clearance(
    R_p: Array,
    t_p: Array,
    sdf_p_grid: Array,
    sdf_p_origin: Array,
    sdf_p_spacing: Array,
    fx_grid: Array,
    fx_origin: Array,
    fx_spacing: Array,
    world_surface_p: Array,
    fx_surface: Array,
    shank_centers: Array,
    shank_halves: Array,
    *,
    beta: float = 20.0,
    top_k_body: int = 16,
    top_k_obb: int = 8,
    n_real_p: Array | None = None,
    n_real_f: Array | None = None,
    shank_mask: Array | None = None,
) -> FixtureClearance:
    """Both probe-fixture clearance categories in one call.

    Categories: probe body voxel-SDF vs fixture surface samples; probe
    OBB analytic SDF vs fixture surface samples. The OBB direction
    catches probe shank / transition-zone contact with fixture
    surfaces (well-bore-rim grazing) that the body voxel direction
    misses at the α-wrap closing cap.

    ``n_real_p`` / ``shank_mask`` carry the probe grid's real extent and the OBB
    validity mask when the probe grid/OBB come from a padded per-kind table.
    ``n_real_f`` carries the fixture grid's real extent when the fixtures are
    stacked into a padded table to share a vmap axis (see clearance_sweep). All
    ``None`` ⇒ unchanged behaviour.
    """
    body = pairwise_signed_clearance_probe_fixture_body_world(
        R_p,
        t_p,
        sdf_p_grid,
        sdf_p_origin,
        sdf_p_spacing,
        fx_grid,
        fx_origin,
        fx_spacing,
        world_surface_p,
        fx_surface,
        beta=beta,
        top_k=top_k_body,
        n_real_p=n_real_p,
        n_real_f=n_real_f,
    )
    obb = pairwise_signed_clearance_probe_obb_fixture_world(
        R_p,
        t_p,
        fx_surface,
        shank_centers,
        shank_halves,
        beta=beta,
        shank_mask=shank_mask,
        top_k=top_k_obb,
    )
    return FixtureClearance(body=body, obb=obb)


def pairwise_signed_clearance_dual(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    sdf_a_grid: Array,
    sdf_a_origin: Array,
    sdf_a_spacing: Array,
    sdf_b_grid: Array,
    sdf_b_origin: Array,
    sdf_b_spacing: Array,
    surface_a: Array,  # (Nbody, 3) body envelope samples (a-local)
    surface_b: Array,  # (Nbody, 3) body envelope samples (b-local)
    shank_centers_a: Array,  # (Sa, 3) shank centres in a-local
    shank_halves_a: Array,  # (Sa, 3) shank half-extents
    shank_centers_b: Array,  # (Sb, 3)
    shank_halves_b: Array,  # (Sb, 3)
    *,
    beta: float = 20.0,
    top_k_body_body: int = 16,
    top_k_body_shank: int = 8,
    top_k_shank_shank: int = 8,
) -> tuple[
    tuple[Array, Array],
    tuple[Array, Array],
    tuple[Array, Array],
]:
    """Dual-rep pair clearance: body voxel SDF + analytic shank OBBs.

    Returns three ``(hard_min, soft_min)`` tuples, one per category:

      - ``body_body``  : envelope-sample SDF lookups (both directions)
      - ``body_shank`` : shank corners vs other body's SDF (both ways)
      - ``shank_shank``: shank corners vs other-probe shank OBBs

    The caller should apply ReLU-squared penalties to each soft min and
    sum, so the optimizer gets independent gradient signal per category
    rather than letting sample-count imbalance bias the gradient toward
    body samples in a pooled softmin.

    Hard mins are exposed for diagnostics and feasibility checks (use
    these in lex keys and hard constraints).

    The body SDF is looked up trilinearly. Soft-min top-k aggregation
    across 16 samples absorbs trilinear's C⁰ voxel-edge gradient
    discontinuities, so a C¹ tricubic lookup bought no convergence at
    ~5× the cost per pair. OBB SDFs are closed-form regardless.

    β=20/mm default → 50 µm smoothing window. Per-category top_k caps
    the bias at ``log(top_k)/β`` (~0.14 mm body-body, ~0.10 mm shank).
    """
    # Hoist the local→world surface transforms to ONE per probe by
    # computing them and delegating to the world-frame variant. Callers
    # in tight loops over many pairs should call that variant directly
    # to skip the K(K-1) per-pair redundancy XLA's CSE pass leaves in
    # (verified 2026-05-23 via HLO dump).
    world_surface_a = surface_a @ R_a.T + t_a
    world_surface_b = surface_b @ R_b.T + t_b
    return pairwise_signed_clearance_dual_world(
        R_a,
        t_a,
        R_b,
        t_b,
        sdf_a_grid,
        sdf_a_origin,
        sdf_a_spacing,
        sdf_b_grid,
        sdf_b_origin,
        sdf_b_spacing,
        world_surface_a,
        world_surface_b,
        shank_centers_a,
        shank_halves_a,
        shank_centers_b,
        shank_halves_b,
        beta=beta,
        top_k_body_body=top_k_body_body,
        top_k_body_shank=top_k_body_shank,
        top_k_shank_shank=top_k_shank_shank,
    )


def pairwise_signed_clearance_dual_world(
    R_a: Array,
    t_a: Array,
    R_b: Array,
    t_b: Array,
    sdf_a_grid: Array,
    sdf_a_origin: Array,
    sdf_a_spacing: Array,
    sdf_b_grid: Array,
    sdf_b_origin: Array,
    sdf_b_spacing: Array,
    world_surface_a: Array,  # (Nbody, 3) body envelope samples in WORLD
    world_surface_b: Array,  # (Nbody, 3) body envelope samples in WORLD
    shank_centers_a: Array,
    shank_halves_a: Array,
    shank_centers_b: Array,
    shank_halves_b: Array,
    *,
    beta: float = 20.0,
    top_k_body_body: int = 16,
    top_k_body_shank: int = 8,
    top_k_shank_shank: int = 8,
) -> tuple[
    tuple[Array, Array],
    tuple[Array, Array],
    tuple[Array, Array],
]:
    """Same as :func:`pairwise_signed_clearance_dual` but takes
    pre-transformed world-frame body surface samples. Caller is
    responsible for computing ``world_surface[i] = surface[i] @ R[i].T
    + t[i]`` once per probe outside the pair loop. Avoids O(K(K-1))
    redundant transforms across a probe set with K probes' pairs
    (CSE doesn't catch the duplication — HLO inspection 2026-05-23).
    """
    # 1+2: body-body (per-sample, not min-reduced).
    local_in_a = (world_surface_b - t_a) @ R_a
    d_body_b_in_a = trilinear_sdf(sdf_a_grid, sdf_a_origin, sdf_a_spacing, local_in_a)
    local_in_b = (world_surface_a - t_b) @ R_b
    d_body_a_in_b = trilinear_sdf(sdf_b_grid, sdf_b_origin, sdf_b_spacing, local_in_b)
    body_body_pool = jnp.concatenate(
        [d_body_b_in_a.reshape(-1), d_body_a_in_b.reshape(-1)], axis=0
    )

    Sa = shank_centers_a.shape[0]
    Sb = shank_centers_b.shape[0]
    corners_a_world = (
        _shank_world_samples(R_a, t_a, shank_centers_a, shank_halves_a)
        if Sa > 0
        else jnp.zeros((0, 3), dtype=world_surface_a.dtype)
    )
    corners_b_world = (
        _shank_world_samples(R_b, t_b, shank_centers_b, shank_halves_b)
        if Sb > 0
        else jnp.zeros((0, 3), dtype=world_surface_b.dtype)
    )

    # 3+4: body-shank — both directions (see pairwise_signed_clearance_dual).
    body_shank_chunks = []
    if Sa > 0:
        ca_in_b = (corners_a_world - t_b) @ R_b
        d_corners_a_in_b = trilinear_sdf(
            sdf_b_grid, sdf_b_origin, sdf_b_spacing, ca_in_b
        )
        body_shank_chunks.append(d_corners_a_in_b.reshape(-1))
        d_body_b_vs_a_obbs = jax.vmap(
            lambda c, h: _obb_sdf_world_to_local(world_surface_b, R_a, t_a, c, h)
        )(shank_centers_a, shank_halves_a)
        body_shank_chunks.append(d_body_b_vs_a_obbs.reshape(-1))
    if Sb > 0:
        cb_in_a = (corners_b_world - t_a) @ R_a
        d_corners_b_in_a = trilinear_sdf(
            sdf_a_grid, sdf_a_origin, sdf_a_spacing, cb_in_a
        )
        body_shank_chunks.append(d_corners_b_in_a.reshape(-1))
        d_body_a_vs_b_obbs = jax.vmap(
            lambda c, h: _obb_sdf_world_to_local(world_surface_a, R_b, t_b, c, h)
        )(shank_centers_b, shank_halves_b)
        body_shank_chunks.append(d_body_a_vs_b_obbs.reshape(-1))
    body_shank_pool = (
        jnp.concatenate(body_shank_chunks, axis=0)
        if body_shank_chunks
        else jnp.zeros((0,), dtype=world_surface_a.dtype)
    )

    # 5+6: shank-shank via exact OBB-OBB SAT (no sampling). Returns one
    # signed-distance scalar per (Sa, Sb) OBB pair — closed-form. Catches
    # interior crossings that surface-sample-based queries miss for thin
    # OBBs (the 24/70 µm silicon shanks).
    shank_shank_chunks = []
    if Sa > 0 and Sb > 0:
        # vmap outer over A's OBBs, inner over B's OBBs.
        def _pair_distance(ca, ha, cb, hb):
            return obb_obb_signed_distance(R_a, t_a, ca, ha, R_b, t_b, cb, hb)

        d_sa_vs_sb = jax.vmap(
            lambda ca, ha: jax.vmap(lambda cb, hb: _pair_distance(ca, ha, cb, hb))(
                shank_centers_b, shank_halves_b
            )
        )(shank_centers_a, shank_halves_a)  # (Sa, Sb)
        shank_shank_chunks.append(d_sa_vs_sb.reshape(-1))
    shank_shank_pool = (
        jnp.concatenate(shank_shank_chunks, axis=0)
        if shank_shank_chunks
        else jnp.zeros((0,), dtype=world_surface_a.dtype)
    )

    return (
        _hard_soft(body_body_pool, beta=beta, top_k=top_k_body_body),
        _hard_soft(body_shank_pool, beta=beta, top_k=top_k_body_shank),
        _hard_soft(shank_shank_pool, beta=beta, top_k=top_k_shank_shank),
    )
