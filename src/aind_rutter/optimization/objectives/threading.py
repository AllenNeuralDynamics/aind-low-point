"""The bore-threading test: how far each shank's crossing of a hole section sits
from fitting, with the padding sizes the phases pack shanks and sections to."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

MAX_SHANKS_PAD = 4
# A shank's cross-section corners, as signs on its (x, y) half-widths.
_CORNER_SIGNS = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
MAX_SECTIONS_PAD = 8


def softplus_squared(values: jnp.ndarray) -> jnp.ndarray:
    sp = jnp.maximum(0.0, values) + jnp.log1p(jnp.exp(-jnp.abs(values)))
    return jnp.sum(sp * sp)


def threading_g_matrix(
    R: jnp.ndarray,
    pose_tip: jnp.ndarray,
    tips_local: jnp.ndarray,
    s_axes: jnp.ndarray,
    s_centers: jnp.ndarray,
    s_e1: jnp.ndarray,
    s_e2: jnp.ndarray,
    s_cos: jnp.ndarray,
    s_sin: jnp.ndarray,
    s_a: jnp.ndarray,
    s_b: jnp.ndarray,
    shaft_length_mm: float = 10.0,
    *,
    w_normals: jnp.ndarray | None = None,
    w_offsets: jnp.ndarray | None = None,
    shank_half_xy: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Raw (n_sections, n_shanks) bore g values for one probe pose.

    ``g <= tol`` is feasible. No masking, no thresholding — the caller
    applies whichever mask-aware sum-of-squares or hard-constraint slack the
    caller needs.

    ``w_normals`` (W, 3) and ``w_offsets`` (W,) add the hole's wall planes (see
    ``holes.pack_walls``). Each section's g becomes the larger of the oval value and
    ``2 (n·p − offset) / b``: the wall distance scaled to the oval's slope across its
    minor axis, so one g tolerance means about the same distance for both.

    ``shank_half_xy`` (2,) is the shank's cross-section half-width along local x and
    y. Given, the rectangle's four edges are tested in place of the centreline and
    each entry is the worst edge's. A straight prism cuts a section plane in a
    parallelogram whose corners are those edges' crossings, and g is convex in the
    crossing point, so the rectangle clears exactly when its corners do."""
    n_shanks = tips_local.shape[0]
    if shank_half_xy is not None:
        offsets = jnp.concatenate(
            [
                jnp.asarray(_CORNER_SIGNS, tips_local.dtype) * shank_half_xy,
                jnp.zeros((4, 1), tips_local.dtype),
            ],
            axis=1,
        )
        tips_local = (tips_local[:, None, :] + offsets[None]).reshape(-1, 3)
    tip_world = tips_local @ R.T + pose_tip
    shaft_dir = R @ jnp.array([0.0, 0.0, 1.0])
    line_d = shaft_length_mm * shaft_dir
    denom = s_axes @ line_d
    rel_centers = tip_world[None, :, :] - s_centers[:, None, :]
    num = jnp.einsum("skd,sd->sk", rel_centers, s_axes)
    # ``denom`` is the dot product of shaft direction with section axis.
    # For typical bounds (ap, ml ≤ 60°) it's in [0.25, 1.0]; the
    # ``shaft ⊥ section`` case (denom → 0) is geometrically unreachable
    # in our problem. The where here is a defensive guard for exact-zero
    # division; the gradient flows through the ``denom`` branch
    # everywhere ``|denom| ≥ 1e-12`` (i.e., always in practice).
    safe_denom = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    t = -num / safe_denom[:, None]
    pts = tip_world[None, :, :] + t[..., None] * line_d
    rel_to_center = pts - s_centers[:, None, :]
    u_w = jnp.einsum("skd,sd->sk", rel_to_center, s_e1)
    v_w = jnp.einsum("skd,sd->sk", rel_to_center, s_e2)
    u = s_cos[:, None] * u_w + s_sin[:, None] * v_w
    v = -s_sin[:, None] * u_w + s_cos[:, None] * v_w
    # No more ``where(parallel, inf, g)`` override (removed Patch B):
    # the ``+inf`` branch's gradient was NaN, corrupting optimizer gradients.
    # Guard the ellipse-axis divisions too: padded sections arrive with
    # ``s_a = s_b = 0`` from BatchedProbeStatic's zero-fill (the phase1
    # packer fills 1.0), so ``u / s_a`` is ``0/0 = NaN`` there — and the
    # caller's ``valid_g * excess²`` mask does NOT rescue it (``0 * NaN =
    # NaN``, value AND gradient). ``safe_{a,b}`` keep ``g`` finite for
    # padded/degenerate sections (masked out downstream); valid sections
    # have ``|s| ≫ 1e-12`` so the guard is an exact no-op for them.
    safe_a = jnp.where(jnp.abs(s_a) < 1e-12, 1.0, s_a)
    safe_b = jnp.where(jnp.abs(s_b) < 1e-12, 1.0, s_b)
    g = (u / safe_a[:, None]) ** 2 + (v / safe_b[:, None]) ** 2 - 1.0
    if w_normals is not None:
        wall_dist = jnp.einsum("skd,wd->skw", pts, w_normals) - w_offsets
        g = jnp.maximum(g, jnp.max(wall_dist, axis=-1) * (2.0 / safe_b[:, None]))
    if shank_half_xy is None:
        return g
    return jnp.max(g.reshape(g.shape[0], n_shanks, 4), axis=-1)
