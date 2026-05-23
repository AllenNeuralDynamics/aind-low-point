"""JAX-traceable kernels: pose math, SDF trilinear lookup, pairwise
signed clearance via SDF.

Companion to :mod:`aind_low_point.optimization.sdf` (which builds the
voxel grids). This module does the inner-loop work that needs to be
differentiable for SLSQP's Jacobian.

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

import jax
import jax.numpy as jnp
from jax import Array

# RAS → LPS sign flip applied to a (3, 3) rotation: R_lps = D R_ras D.
_D_RAS_TO_LPS = jnp.diag(jnp.array([-1.0, -1.0, 1.0]))


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


def arc_angles_to_rotation(
    ap_deg: Array, ml_deg: Array, spin_deg: Array
) -> Array:
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


def _cubic_kernel(
    t: Array, y0: Array, y1: Array, y2: Array, y3: Array
) -> Array:
    """Catmull-Rom cubic interpolation of 4 samples at fractional ``t``.

    Samples ``y0..y3`` are at integer offsets ``{-1, 0, 1, 2}`` relative
    to ``floor(query)``. ``t`` ∈ [0, 1] is the fractional position past
    ``y1``. Result is C¹ continuous.
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * y1
        + (-y0 + y2) * t
        + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
        + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3
    )


def tricubic_sdf(
    grid: Array,
    origin: Array,
    spacing: Array,
    query_local: Array,
    out_of_bounds_value: Array = jnp.array(1e3),
) -> Array:
    """Tricubic (Catmull-Rom) SDF interpolation. C¹ continuous gradient.

    Uses 64 grid samples per query (4³) vs trilinear's 8. The natural
    cost ratio is ~8× per query; in practice JAX/XLA fuses the gathers
    so the wall-time delta is smaller. The payoff is a smooth gradient
    everywhere — trilinear's gradient is piecewise constant per voxel
    and jumps at voxel faces, which the optimizer's line search can
    chatter on.

    In-bounds requires one cell margin from each face: the 4-sample
    stencil per axis needs ``i0 ∈ [1, N-3]`` where ``i0 = floor(query)``.
    Out-of-bounds queries return ``out_of_bounds_value``.
    """
    grid = jnp.asarray(grid)
    Nx, Ny, Nz = grid.shape
    coords = (query_local - origin) / spacing  # voxel units
    i0 = jnp.floor(coords).astype(jnp.int32)
    f = coords - i0  # fractional in [0, 1)

    in_bounds = (
        (i0[..., 0] >= 1) & (i0[..., 0] <= Nx - 3)
        & (i0[..., 1] >= 1) & (i0[..., 1] <= Ny - 3)
        & (i0[..., 2] >= 1) & (i0[..., 2] <= Nz - 3)
    )
    ix = jnp.clip(i0[..., 0], 1, Nx - 3)
    iy = jnp.clip(i0[..., 1], 1, Ny - 3)
    iz = jnp.clip(i0[..., 2], 1, Nz - 3)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    def _along_z(dx, dy):
        return _cubic_kernel(
            fz,
            grid[ix + dx, iy + dy, iz - 1],
            grid[ix + dx, iy + dy, iz],
            grid[ix + dx, iy + dy, iz + 1],
            grid[ix + dx, iy + dy, iz + 2],
        )

    def _along_y(dx):
        return _cubic_kernel(
            fy,
            _along_z(dx, -1),
            _along_z(dx, 0),
            _along_z(dx, 1),
            _along_z(dx, 2),
        )

    interp = _cubic_kernel(
        fx,
        _along_y(-1),
        _along_y(0),
        _along_y(1),
        _along_y(2),
    )
    return jnp.where(in_bounds, interp, out_of_bounds_value)


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
    the ±180° wraparound discontinuity that bound-clipped SLSQP on the
    scalar-angle layout. Internally, the existing
    :func:`pose_from_optimizer_vars` API still takes ``spin_deg`` so we
    convert at the unpacking site via ``atan2(sy, sx)``.

    The norm of ``(sx, sy)`` is irrelevant — only the direction
    matters. JAX's ``arctan2`` is well-defined and differentiable
    everywhere except at the origin; SLSQP bounds keep us away from
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

    Returns
    -------
    (...,) interpolated signed distances. Differentiable w.r.t.
    ``query_local`` and ``grid``.
    """
    grid = jnp.asarray(grid)
    Nx, Ny, Nz = grid.shape
    coords = (query_local - origin) / spacing  # (..., 3) in voxel units
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
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

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
    interp = c0 * (1 - fz) + c1 * fz
    return jnp.where(in_bounds, interp, out_of_bounds_value)


def pairwise_signed_clearance(
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
    surface_a: Array,  # (N, 3) — a's surface in a's canonical local
    surface_b: Array,  # (N, 3) — b's surface in b's canonical local
) -> Array:
    """Minimum signed distance between two probes via SDF lookup.

    Negative ⇒ overlap; positive ⇒ clear. Smooth + differentiable
    everywhere via trilinear interpolation. Symmetrised by querying
    both ``a → b`` and ``b → a`` (the surface-vs-volume formulation is
    asymmetric otherwise — sampling density differences would make one
    direction tighter than the other).
    """
    # Transform b's surface points (b's local) into a's local frame:
    # world_b = R_b @ surf_b + t_b
    # local_in_a = R_a^T @ (world_b - t_a)
    world_b = surface_b @ R_b.T + t_b  # (N, 3)
    local_in_a = (world_b - t_a) @ R_a  # equivalent to R_a^T @ (world_b - t_a).T
    sd_b_in_a = trilinear_sdf(
        sdf_a_grid, sdf_a_origin, sdf_a_spacing, local_in_a
    )
    # Symmetric direction
    world_a = surface_a @ R_a.T + t_a
    local_in_b = (world_a - t_b) @ R_b
    sd_a_in_b = trilinear_sdf(
        sdf_b_grid, sdf_b_origin, sdf_b_spacing, local_in_b
    )
    return jnp.minimum(jnp.min(sd_b_in_a), jnp.min(sd_a_in_b))


def pairwise_signed_clearance_probe_fixture_body(
    R_p: Array, t_p: Array,
    sdf_p_grid: Array, sdf_p_origin: Array, sdf_p_spacing: Array,
    sdf_f_grid: Array, sdf_f_origin: Array, sdf_f_spacing: Array,
    surface_p: Array,           # (Np, 3) probe envelope samples in probe local
    surface_f: Array,           # (Nf, 3) fixture envelope samples in world LPS
    *,
    beta: float = 20.0,
    top_k: int = 16,
    interp: str = "trilinear",
) -> tuple[Array, Array]:
    """Body-only signed clearance between a moving probe and a static
    fixture (e.g., cone, well, headframe).

    Fixtures are *static* in world LPS — their SDF grid is already in
    the world frame, so no transform is applied to fixture samples.
    The probe surface samples are pushed into world via ``(R, t)``;
    the fixture surface samples are pulled into probe local via
    ``R^T (s − t)``.

    Returns ``(hard_min, soft_min_topk)`` — same shape as the other
    dual-rep clearance functions. Shanks are intentionally *not*
    included: shank-vs-fixture is rare (threading already constrains
    the bore), and per design we only enforce probe-*body* clearance
    against fixtures.
    """
    sdf_lookup = tricubic_sdf if interp == "tricubic" else trilinear_sdf

    world_p = surface_p @ R_p.T + t_p
    d_p_in_f = sdf_lookup(sdf_f_grid, sdf_f_origin, sdf_f_spacing, world_p)

    local_f_in_p = (surface_f - t_p) @ R_p
    d_f_in_p = sdf_lookup(sdf_p_grid, sdf_p_origin, sdf_p_spacing, local_f_in_p)

    distances = jnp.concatenate([d_p_in_f.reshape(-1), d_f_in_p.reshape(-1)])
    hard_min = jnp.min(distances)
    soft = soft_min_topk(distances, beta=beta, top_k=top_k)
    return hard_min, soft


@jax.jit
def pairwise_signed_clearance_jit(
    R_a, t_a, R_b, t_b,
    sdf_a_grid, sdf_a_origin, sdf_a_spacing,
    sdf_b_grid, sdf_b_origin, sdf_b_spacing,
    surface_a, surface_b,
):
    return pairwise_signed_clearance(
        R_a, t_a, R_b, t_b,
        sdf_a_grid, sdf_a_origin, sdf_a_spacing,
        sdf_b_grid, sdf_b_origin, sdf_b_spacing,
        surface_a, surface_b,
    )


_SHANK_SAMPLES_PER_BOX = 24  # 8 corners + 16 long-edge interior samples

# 4 long-edge XY positions (one per long edge of the box)
_SHANK_LONG_EDGE_XY = jnp.array(
    [[-1, -1], [+1, -1], [-1, +1], [+1, +1]], dtype=jnp.float32
)  # (4, 2)
# 6 Z fractions: 2 corners at ±1 plus 4 interior samples
_SHANK_Z_FRACS = jnp.array(
    [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0], dtype=jnp.float32
)  # (6,)
# Build (24, 3) sign matrix: 4 XY positions × 6 Z positions = 24 samples per box.
# Pre-computed module-level so each call avoids the gather setup.
_SHANK_SAMPLE_SIGNS = jnp.stack(
    [
        jnp.broadcast_to(_SHANK_LONG_EDGE_XY[:, None, 0], (4, 6)),
        jnp.broadcast_to(_SHANK_LONG_EDGE_XY[:, None, 1], (4, 6)),
        jnp.broadcast_to(_SHANK_Z_FRACS[None, :], (4, 6)),
    ],
    axis=-1,
).reshape(-1, 3)  # (24, 3)


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
        centers[:, None, :]
        + _SHANK_SAMPLE_SIGNS[None, :, :] * half_extents[:, None, :]
    )


def _shank_world_samples(
    R: Array, t: Array, centers_local: Array, halves_local: Array,
) -> Array:
    """Return ``(S * 24, 3)`` world-frame surface-sample positions of
    all shank boxes of one probe at pose ``(R, t)``.
    """
    local = _obb_sample_points_local(centers_local, halves_local)
    S = local.shape[0]
    return (local.reshape(-1, 3) @ R.T + t).reshape(S * _SHANK_SAMPLES_PER_BOX, 3)


def _obb_sdf_world_to_local(
    query_world: Array,  # (M, 3)
    R: Array, t: Array,
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
    R_a: Array, t_a: Array, center_a: Array, halves_a: Array,
    R_b: Array, t_b: Array, center_b: Array, halves_b: Array,
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
        # Projection of A's half-box onto axis: r_a = sum_i |halves_a[i] * (axis_a[i] . axis_w)|
        proj_a = jnp.sum(jnp.abs(axes_a * axis_w[None, :]).T * halves_a[:, None].T)
        # Simpler form: r_a = halves_a . |axes_a.T @ axis_w|
        proj_a = jnp.dot(halves_a, jnp.abs(axes_a.T @ axis_w))
        proj_b = jnp.dot(halves_b, jnp.abs(axes_b.T @ axis_w))
        centre_sep = jnp.abs(jnp.dot(diff, axis_w))
        return centre_sep - (proj_a + proj_b)

    sep_face_a = jax.vmap(_separation_along)(axes_a.T)  # (3,)
    sep_face_b = jax.vmap(_separation_along)(axes_b.T)  # (3,)

    # ---- 9 cross-product candidate axes ----
    # Parallel-axis guard: ``jnp.linalg.norm(axis)`` at axis=0 has NaN
    # gradient (0/0) which propagates through autodiff even when the
    # ``where`` masks the forward value out. Use soft-norm
    # ``sqrt(||axis||² + ε²)`` so the gradient stays finite when the
    # cross product collapses (parallel face normals).
    def _cross_axis_sep(i: int, j: int) -> Array:
        ai = axes_a[:, i]
        bj = axes_b[:, j]
        axis = jnp.cross(ai, bj)
        sq = jnp.sum(axis * axis)
        soft_norm = jnp.sqrt(sq + jnp.float32(1e-12))
        is_valid = sq > jnp.float32(1e-12)
        axis_n = axis / soft_norm   # finite gradient everywhere
        sep = _separation_along(axis_n)
        # When axes are parallel, the cross-product axis is degenerate;
        # substitute a very negative separation so max-selection skips it.
        return jnp.where(is_valid, sep, jnp.float32(-1e6))

    sep_cross = jnp.stack([
        _cross_axis_sep(i, j) for i in range(3) for j in range(3)
    ])  # (9,)

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


def pairwise_signed_clearance_dual(
    R_a: Array, t_a: Array, R_b: Array, t_b: Array,
    sdf_a_grid: Array, sdf_a_origin: Array, sdf_a_spacing: Array,
    sdf_b_grid: Array, sdf_b_origin: Array, sdf_b_spacing: Array,
    surface_a: Array,           # (Nbody, 3) body envelope samples (a-local)
    surface_b: Array,           # (Nbody, 3) body envelope samples (b-local)
    shank_centers_a: Array,     # (Sa, 3) shank centres in a-local
    shank_halves_a: Array,      # (Sa, 3) shank half-extents
    shank_centers_b: Array,     # (Sb, 3)
    shank_halves_b: Array,      # (Sb, 3)
    *,
    beta: float = 20.0,
    top_k_body_body: int = 16,
    top_k_body_shank: int = 8,
    top_k_shank_shank: int = 8,
    interp: str = "trilinear",
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
    these in lex_key / SLSQP constraints).

    ``interp ∈ {"trilinear", "tricubic"}`` selects the body-SDF
    interpolator. **Default is trilinear**: empirically the soft-min
    top-k aggregation across 16 samples absorbs trilinear's C⁰ voxel-
    edge gradient discontinuities, so tricubic's smoothness gain
    doesn't translate to better SLSQP convergence — but tricubic is
    ~5× slower per pair-clearance call. OBB SDFs are closed-form
    regardless.

    β=20/mm default → 50 µm smoothing window. Per-category top_k caps
    the bias at ``log(top_k)/β`` (~0.14 mm body-body, ~0.10 mm shank).
    """
    sdf_lookup = tricubic_sdf if interp == "tricubic" else trilinear_sdf

    # 1+2: body-body (per-sample, not min-reduced)
    world_b = surface_b @ R_b.T + t_b
    local_in_a = (world_b - t_a) @ R_a
    d_body_b_in_a = sdf_lookup(
        sdf_a_grid, sdf_a_origin, sdf_a_spacing, local_in_a
    )
    world_a = surface_a @ R_a.T + t_a
    local_in_b = (world_a - t_b) @ R_b
    d_body_a_in_b = sdf_lookup(
        sdf_b_grid, sdf_b_origin, sdf_b_spacing, local_in_b
    )
    body_body_pool = jnp.concatenate(
        [d_body_b_in_a.reshape(-1), d_body_a_in_b.reshape(-1)], axis=0
    )

    Sa = shank_centers_a.shape[0]
    Sb = shank_centers_b.shape[0]
    corners_a_world = (
        _shank_world_samples(R_a, t_a, shank_centers_a, shank_halves_a)
        if Sa > 0
        else jnp.zeros((0, 3), dtype=surface_a.dtype)
    )
    corners_b_world = (
        _shank_world_samples(R_b, t_b, shank_centers_b, shank_halves_b)
        if Sb > 0
        else jnp.zeros((0, 3), dtype=surface_b.dtype)
    )

    # 3+4: body-shank — both directions:
    #   (a) shank-corner samples → other body's SDF
    #   (b) body-surface samples → other shank's OBB SDF (analytic)
    # The OBB-surface direction (a) has corner blind spots for crossings,
    # but the body-surface direction (b) is dense (5000 samples) and OBB
    # SDF is exact, so together they catch all body-vs-shank cases.
    body_shank_chunks = []
    if Sa > 0:
        # (a) A's shank corners in B's body
        ca_in_b = (corners_a_world - t_b) @ R_b
        d_corners_a_in_b = sdf_lookup(
            sdf_b_grid, sdf_b_origin, sdf_b_spacing, ca_in_b
        )
        body_shank_chunks.append(d_corners_a_in_b.reshape(-1))
        # (b) B's body surface samples in each of A's shank OBBs
        # (analytic OBB SDF; one query per OBB × body sample).
        # Push B's body samples to world via (R_b, t_b), then evaluate
        # each A OBB's SDF at those world points.
        world_surface_b = surface_b @ R_b.T + t_b
        d_body_b_vs_a_obbs = jax.vmap(
            lambda c, h: _obb_sdf_world_to_local(
                world_surface_b, R_a, t_a, c, h
            )
        )(shank_centers_a, shank_halves_a)  # (Sa, Nbody)
        body_shank_chunks.append(d_body_b_vs_a_obbs.reshape(-1))
    if Sb > 0:
        # (a) B's shank corners in A's body
        cb_in_a = (corners_b_world - t_a) @ R_a
        d_corners_b_in_a = sdf_lookup(
            sdf_a_grid, sdf_a_origin, sdf_a_spacing, cb_in_a
        )
        body_shank_chunks.append(d_corners_b_in_a.reshape(-1))
        # (b) A's body surface samples in each of B's shank OBBs
        world_surface_a = surface_a @ R_a.T + t_a
        d_body_a_vs_b_obbs = jax.vmap(
            lambda c, h: _obb_sdf_world_to_local(
                world_surface_a, R_b, t_b, c, h
            )
        )(shank_centers_b, shank_halves_b)  # (Sb, Nbody)
        body_shank_chunks.append(d_body_a_vs_b_obbs.reshape(-1))
    body_shank_pool = (
        jnp.concatenate(body_shank_chunks, axis=0)
        if body_shank_chunks
        else jnp.zeros((0,), dtype=surface_a.dtype)
    )

    # 5+6: shank-shank via exact OBB-OBB SAT (no sampling). Returns one
    # signed-distance scalar per (Sa, Sb) OBB pair — closed-form. Catches
    # interior crossings that surface-sample-based queries miss for thin
    # OBBs (the 24/70 µm silicon shanks).
    shank_shank_chunks = []
    if Sa > 0 and Sb > 0:
        # vmap outer over A's OBBs, inner over B's OBBs.
        def _pair_distance(ca, ha, cb, hb):
            return obb_obb_signed_distance(
                R_a, t_a, ca, ha, R_b, t_b, cb, hb
            )
        d_sa_vs_sb = jax.vmap(
            lambda ca, ha: jax.vmap(
                lambda cb, hb: _pair_distance(ca, ha, cb, hb)
            )(shank_centers_b, shank_halves_b)
        )(shank_centers_a, shank_halves_a)  # (Sa, Sb)
        shank_shank_chunks.append(d_sa_vs_sb.reshape(-1))
    shank_shank_pool = (
        jnp.concatenate(shank_shank_chunks, axis=0)
        if shank_shank_chunks
        else jnp.zeros((0,), dtype=surface_a.dtype)
    )

    return (
        _hard_soft(body_body_pool, beta=beta, top_k=top_k_body_body),
        _hard_soft(body_shank_pool, beta=beta, top_k=top_k_body_shank),
        _hard_soft(shank_shank_pool, beta=beta, top_k=top_k_shank_shank),
    )
