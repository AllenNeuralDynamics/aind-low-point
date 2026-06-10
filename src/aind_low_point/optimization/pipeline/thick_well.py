"""Solidify the well SDF: fill the body inside the conical lumen.

The well asset is a non-watertight *surface* mesh of a geometrically THICK
annular body (a ~5 mm-radius funnel lumen in an ~8 mm-thick wall). α-wrapping a
surface shell yields a watertight *skin*, so the production well SDF goes
negative only in a ~0.8 mm band hugging the surface (measured: 2.94% of voxels
< 0, min depth −0.63 mm). A probe penetrating the body reads at most −0.63 mm
before popping out the far surface into free space — a deep false-negative with
a flat gradient, papered over by the 0.07 mm α-wrap offset.

This module rebuilds the well SDF *grid* as a solid: the body ANNULUS (between
the conical lumen and the conical outer wall, within the axial band) reads deeply
negative (distance into the body), with a clean gradient pointing back toward the
lumen. The lumen interior, the true exterior, and everything above/below the band
stay free, and the existing surface zero-crossing (bore wall, top face, outer
wall) is preserved by taking the elementwise ``min`` with the original SDF.

Method (per voxel, in the bore frame fit from the mesh):
    r_inner(z) = a·z + b + margin               # conical lumen wall, fit
    r_outer(z) = ao·z + bo − outer_margin       # conical outer wall, fit
    band_solid = where(z in band & r_inner < r < r_outer, r_inner − r, +inf)
    thick = min(original_sdf, band_solid)

Both cones are FIT from the mesh (per-axial-slice inner/outer boundary), so it
generalizes to other well assets. Safe error directions: r_inner slightly too
LARGE (margin ≥ fit residual) — never marks real lumen solid (a false collision
rejecting plans); r_outer slightly too SMALL — never marks free exterior solid
(the over-fill bug that pushed feasible probes into collision). The thin near-wall
shells either margin leaves fall back to the accurate original SDF.

Only the ``grid`` changes; ``surface`` (well points queried against probe grids)
and the FCL ground-truth mesh are untouched, so this is low-risk by
construction — FCL still gates feasibility.

Run (validation):  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.thick_well_sdf
Env:  MARGIN=0.5
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from aind_low_point.optimization.phase1_objective_jax import FixtureSDFData

MARGIN = float(_os.environ.get("MARGIN", "0.5"))


def fit_well_cone(mesh, *, n_slices: int = 10) -> dict:
    """Fit the conical lumen of a well surface mesh in its PCA bore frame.

    Returns the bore frame + linear lumen-radius profile ``r(z) = a·z + b`` and
    the axial band ``[z_lo, z_hi]`` (along-axis coord relative to the centroid).
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    ctr = V.mean(0)
    Vc = V - ctr
    # PCA: the bore axis is the least-spread principal axis (the well is short
    # along the bore, wide across it). Orient +z-ish for readability.
    _, Q = np.linalg.eigh(Vc.T @ Vc)
    axis = Q[:, 0]
    if axis[2] < 0:
        axis = -axis
    e1 = np.cross(axis, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    z = Vc @ axis
    P = np.stack([Vc @ e1, Vc @ e2], axis=1)  # in-plane coords (centroid origin)

    # Bore center: largest empty disk in the central axial band, where the lumen
    # is actually surrounded by material (the flared ends mislead a global search).
    mid = (z > np.percentile(z, 25)) & (z < np.percentile(z, 75))
    Pm = P[mid]
    gx = np.linspace(Pm[:, 0].min(), Pm[:, 0].max(), 80)
    gy = np.linspace(Pm[:, 1].min(), Pm[:, 1].max(), 80)
    tree = cKDTree(Pm)
    best = (-1.0, 0.0, 0.0)
    for x in gx:
        for y in gy:
            d, _ = tree.query([x, y])
            if d > best[0]:
                best = (d, x, y)
    c0 = np.array([best[1], best[2]])

    # Per-axial-slice INNER boundary (5th-pct radius = lumen wall) and OUTER
    # boundary (95th-pct radius = outer wall) about the bore center → robust to
    # stray verts; linear-fit each. The body is the annulus between them; filling
    # past the outer wall would mark free exterior space solid (false collisions).
    R = np.linalg.norm(P - c0, axis=1)
    zb = np.linspace(z.min(), z.max(), n_slices + 1)
    zc_list, rin_list, rout_list = [], [], []
    for i in range(n_slices):
        sel = (z >= zb[i]) & (z < zb[i + 1])
        if int(sel.sum()) < 8:
            continue
        zc_list.append(0.5 * (zb[i] + zb[i + 1]))
        rin_list.append(float(np.percentile(R[sel], 5)))
        rout_list.append(float(np.percentile(R[sel], 95)))
    zc_arr = np.asarray(zc_list)
    a, b = np.polyfit(zc_arr, rin_list, 1)
    ao, bo = np.polyfit(zc_arr, rout_list, 1)
    resid = float(np.std(np.asarray(rin_list) - (a * zc_arr + b)))
    resid_o = float(np.std(np.asarray(rout_list) - (ao * zc_arr + bo)))

    return dict(
        ctr=ctr,
        axis=axis,
        bore_point=ctr + c0[0] * e1 + c0[1] * e2,  # a point on the bore axis
        cone_a=float(a),
        cone_b=float(b),
        outer_a=float(ao),
        outer_b=float(bo),
        z_lo=float(z.min()),
        z_hi=float(z.max()),
        fit_resid=resid,
        outer_resid=resid_o,
    )


def _voxel_zr(grid_shape, origin, spacing, cone):
    """Per-voxel (along-axis z, radial r) in the bore frame."""
    Nx, Ny, Nz = grid_shape
    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    world = origin[None, None, None, :] + np.stack([ii, jj, kk], -1) * spacing
    d = world - cone["bore_point"]
    zc = d @ cone["axis"]
    perp = d - zc[..., None] * cone["axis"]
    r = np.linalg.norm(perp, axis=-1)
    return zc.astype(np.float32), r.astype(np.float32)


def make_thick_well_sdf(
    mesh,
    fixture_sdf,
    *,
    margin: float = MARGIN,
    outer_margin: float = MARGIN,
    cone=None,
):
    """Return a copy of ``fixture_sdf`` with the body (the conical annulus)
    solidified, leaving the lumen AND the true exterior free.

    The fill is bracketed between two fitted cones,
    ``r_cone(z)+margin < r < r_outer(z)−outer_margin``:

    * inner ``margin`` ADDED → ``r_cone`` sits at-or-just-outside the lumen wall,
      so real lumen is never marked solid (a false collision rejecting plans).
    * outer ``outer_margin`` SUBTRACTED → ``r_outer`` sits at-or-just-inside the
      outer wall, so free exterior space is never marked solid (the over-fill
      bug). The thin shells either margin leaves fall back to the accurate
      original SDF.

    Parameters
    ----------
    mesh
        The well surface mesh (for fitting the cones), in world LPS.
    fixture_sdf
        The original :class:`FixtureSDFData` (its grid is the thin α-wrap skin).
    cone
        Pre-fit cone dict (from :func:`fit_well_cone`); fit from ``mesh`` if None.
    """
    if cone is None:
        cone = fit_well_cone(mesh)
    grid = np.asarray(fixture_sdf.grid, dtype=np.float32)
    origin = np.asarray(fixture_sdf.origin, dtype=np.float64)
    spacing = float(np.asarray(fixture_sdf.spacing))

    zc, r = _voxel_zr(grid.shape, origin, spacing, cone)
    r_inner = cone["cone_a"] * zc + cone["cone_b"] + margin
    r_outer = cone["outer_a"] * zc + cone["outer_b"] - outer_margin
    # Fill only the body annulus within the axial band; everywhere else +inf so
    # the elementwise min keeps the accurate original SDF (lumen, exterior, caps).
    in_body = (
        (zc >= cone["z_lo"]) & (zc <= cone["z_hi"]) & (r > r_inner) & (r < r_outer)
    )
    band_solid = np.where(in_body, r_inner - r, np.inf).astype(np.float32)
    thick = np.minimum(grid, band_solid).astype(np.float32)

    return FixtureSDFData(
        name=fixture_sdf.name,
        grid=jnp.asarray(thick, dtype=jnp.float32),
        origin=fixture_sdf.origin,
        spacing=fixture_sdf.spacing,
        surface=fixture_sdf.surface,
    )


def main() -> int:
    from aind_low_point.config import ConfigModel
    from aind_low_point.optimization.pipeline.phase1_geometry import (
        build_fixture_sdf_data,
    )
    from aind_low_point.runtime import build_runtime_from_config

    cfg = ConfigModel.from_yaml("examples/836656-config-T12.yml")
    rt = build_runtime_from_config(cfg)
    mesh = rt.asset_catalog.get_geometry("well").raw
    well = next(f for f in build_fixture_sdf_data(rt) if f.name == "well")

    cone = fit_well_cone(mesh)
    print(
        f"inner cone: r(z) = {cone['cone_a']:+.3f}·z + {cone['cone_b']:.3f} "
        f"(+margin {MARGIN}), resid {cone['fit_resid']:.2f}mm; "
        f"lumen {cone['cone_a'] * cone['z_lo'] + cone['cone_b']:.2f}"
        f"→{cone['cone_a'] * cone['z_hi'] + cone['cone_b']:.2f}mm"
    )
    print(
        f"outer cone: r(z) = {cone['outer_a']:+.3f}·z + {cone['outer_b']:.3f} "
        f"(−margin {MARGIN}), resid {cone['outer_resid']:.2f}mm; "
        f"wall {cone['outer_a'] * cone['z_lo'] + cone['outer_b']:.2f}"
        f"→{cone['outer_a'] * cone['z_hi'] + cone['outer_b']:.2f}mm; "
        f"band z∈[{cone['z_lo']:.2f},{cone['z_hi']:.2f}]"
    )

    thick = make_thick_well_sdf(mesh, well, cone=cone)
    g0 = np.asarray(well.grid)
    g1 = np.asarray(thick.grid)
    print(
        f"\noriginal SDF: {(g0 < 0).mean() * 100:5.2f}% inside, min {g0.min():+.2f}mm\n"
        f"thick    SDF: {(g1 < 0).mean() * 100:5.2f}% inside, min {g1.min():+.2f}mm"
    )

    # Safety + correctness checks on the bore frame.
    zc, r = _voxel_zr(
        g0.shape, np.asarray(well.origin, float), float(np.asarray(well.spacing)), cone
    )
    r_cone = cone["cone_a"] * zc + cone["cone_b"]
    r_out = cone["outer_a"] * zc + cone["outer_b"]
    in_band = (zc >= cone["z_lo"]) & (zc <= cone["z_hi"])

    # 1) SAFETY (lumen): clearly-inside-lumen voxels must stay FREE (>= 0) — no
    #    false collisions. "Clearly inside" = r < r_cone - 1mm.
    lumen = in_band & (r < r_cone - 1.0)
    n_lum = int(lumen.sum())
    viol = int((g1[lumen] < 0).sum()) if n_lum else 0
    print(
        f"\n[safety lumen] in-lumen voxels: {n_lum}, "
        f"thick<0 (false collisions): {viol}  "
        f"min {g1[lumen].min() if n_lum else float('nan'):+.2f}mm "
        f"({'PASS' if viol == 0 else 'FAIL'})"
    )

    # 2) SAFETY (exterior): the thickening must not make any voxel beyond the
    #    outer wall MORE negative than the original SDF (the over-fill bug). We
    #    compare thick vs original (not thick<0 — the original's own outer-wall
    #    skin is legitimately negative and FCL-honest).
    ext = in_band & (r > r_out + 1.0)
    n_ext = int(ext.sum())
    introduced = (g1[ext] < g0[ext] - 1e-6) if n_ext else np.zeros(0, bool)
    viol_e = int(introduced.sum())
    print(
        f"[safety exter] beyond-wall voxels: {n_ext}, "
        f"thickening-INTRODUCED negatives (over-fill): {viol_e}  "
        f"({'PASS' if viol_e == 0 else 'FAIL'})"
    )

    # 3) FILL: body annulus (r_cone+1 < r < r_outer-1, in band) should be deep.
    body = in_band & (r > r_cone + 1.0) & (r < r_out - 1.0)
    n_body = int(body.sum())
    filled = int((g1[body] < 0).sum())
    print(
        f"[fill]   body-annulus voxels: {n_body}, "
        f"thick<0: {filled} ({100 * filled / max(n_body, 1):.1f}%), "
        f"median depth {np.median(g1[body]):+.2f}mm, deepest {g1[body].min():+.2f}mm"
    )

    # 3) PRESERVE: away from the well (outside band, large |r|) the grid is
    #    untouched (min with +inf) — sanity that we didn't perturb free space.
    far = ~in_band
    dmax = float(np.abs(g1[far] - g0[far]).max()) if int(far.sum()) else 0.0
    print(f"[preserve] max |Δ| outside band: {dmax:.4f}mm (expect 0.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
