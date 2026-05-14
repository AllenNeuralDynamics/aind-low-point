"""Feasibility-map sanity check for the optimizer's threading constraint.

Loads one extracted hole, builds a synthetic 4-shank probe at the
geometric "best fit" pose (along the hole's axis, shank row aligned
with the slot's major axis), then sweeps offset and spin perturbations
and plots the worst-case threading constraint value across
(N_shanks × N_sections).

Validates:
  1. Feasible region (worst g ≤ 0) is a smooth, single-basin blob.
  2. Spin tolerance ≈ ±15° (predicted from the slot ratio).
  3. Top section (chamfer) is more permissive than bottom section
     (straight bore), confirming the section semantics are real.

No kinematic adapter / JAX wiring needed yet — the probe is built
directly as 4 capsules along the hole axis at varying offsets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aind_low_point.optimization import (
    Capsule,
    HoleSection,
    cap_basis,
    find_hole_by_id,
    load_holes,
    shaft_section_oval_value,
)


def make_shanks(
    axis: np.ndarray,
    sections: list[HoleSection],
    *,
    offset_e1: float,
    offset_e2: float,
    spin_rad: float,
    n_shanks: int = 4,
    shank_pitch_mm: float = 0.25,
    shaft_half_len_mm: float = 5.0,
    shank_radius_mm: float = 0.05,
) -> list[Capsule]:
    """Build N shank capsules at the given pose perturbation.

    Slot major-axis direction is taken from the *bottom* section's
    `theta_rad` (the straight bore is the binding constraint). Spin
    rotates the shank row around the hole axis from that reference;
    spin_rad = 0 means perfectly slot-aligned.
    """
    e1, e2 = cap_basis(axis)

    # Slot major-axis direction in world. theta is measured from e1
    # toward e2 in the cap-perpendicular plane.
    theta = sections[-1].theta
    slot_major = np.cos(theta) * e1 + np.sin(theta) * e2
    slot_minor = -np.sin(theta) * e1 + np.cos(theta) * e2

    # Spin rotates the shank row from slot major.
    shank_dir = np.cos(spin_rad) * slot_major + np.sin(spin_rad) * slot_minor

    # Use the middle section's center as the reference point. The
    # offset is applied in (e1, e2) — perpendicular to the hole axis.
    base_center = sections[len(sections) // 2].center.copy()
    base_center = base_center + offset_e1 * e1 + offset_e2 * e2

    # Shank tip offsets along the shank-row direction
    half = (n_shanks - 1) / 2.0
    tip_offsets = [(i - half) * shank_pitch_mm for i in range(n_shanks)]

    shanks: list[Capsule] = []
    for ty in tip_offsets:
        tip = base_center + ty * shank_dir
        # Capsule line passes through `tip` in both directions along
        # the hole axis. The threading constraint only uses the line,
        # not the segment endpoints, so symmetric extension is fine.
        p0 = tip - shaft_half_len_mm * axis
        p1 = tip + shaft_half_len_mm * axis
        shanks.append(Capsule(p0=p0, p1=p1, radius=shank_radius_mm))
    return shanks


def evaluate_threading(shanks: list[Capsule], sections: list[HoleSection]) -> dict:
    """Return per-section worst-shank g and overall worst-of-all g."""
    per_section = []
    for sec in sections:
        gs = [shaft_section_oval_value(sh, sec) for sh in shanks]
        per_section.append(max(gs))
    return {
        "per_section": per_section,  # list, one g per section
        "worst": max(per_section),  # scalar overall worst
    }


def sweep_2d(
    axis, sections, *, e1_axis, e2_axis, spin_deg, **shank_kw
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (worst_g, top_minus_bot) on the (e1, e2) grid at a fixed
    spin. ``top_minus_bot`` is g_top − g_bot — positive means the bottom
    section is tighter (expected for chamfered bores)."""
    spin_rad = np.deg2rad(spin_deg)
    G = np.empty((len(e1_axis), len(e2_axis)))
    TmB = np.empty_like(G)
    for i, e1v in enumerate(e1_axis):
        for j, e2v in enumerate(e2_axis):
            shanks = make_shanks(
                axis,
                sections,
                offset_e1=float(e1v),
                offset_e2=float(e2v),
                spin_rad=spin_rad,
                **shank_kw,
            )
            ev = evaluate_threading(shanks, sections)
            G[i, j] = ev["worst"]
            TmB[i, j] = ev["per_section"][0] - ev["per_section"][-1]
    return G, TmB


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("holes_yaml", type=Path)
    p.add_argument("--hole-id", type=int, default=0)
    p.add_argument("--n-shanks", type=int, default=4)
    p.add_argument("--shank-pitch-mm", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("/tmp/feasibility_map.png"))
    args = p.parse_args()

    hole = find_hole_by_id(load_holes(args.holes_yaml), args.hole_id)
    axis, sections = hole.axis, hole.sections
    print(
        f"Hole #{args.hole_id}: axis={axis.round(3).tolist()}  "
        f"sections={len(sections)}  "
        f"slot_theta_deg={np.rad2deg(hole.slot_theta_rad):.1f}"
    )
    for k, sec in enumerate(sections):
        print(
            f"  section {k}: a={sec.a:.3f}  b={sec.b:.3f}  "
            f"theta_deg={np.rad2deg(sec.theta):.1f}"
        )

    # Sweep ranges. Slot is ~1.20 × 0.70 mm so the major (e1) needs
    # wider offsets than the minor (e2). Use ±0.4 mm both ways for
    # symmetric plots; feasible region is well inside.
    e1_axis = np.linspace(-0.5, 0.5, 51)
    e2_axis = np.linspace(-0.4, 0.4, 41)
    spins = [-30, -20, -15, -10, 0, 10, 15, 20, 30]

    shank_kw = dict(
        n_shanks=args.n_shanks,
        shank_pitch_mm=args.shank_pitch_mm,
    )

    # Center-pose diagnostics (offset 0, spin 0).
    center_shanks = make_shanks(
        axis,
        sections,
        offset_e1=0.0,
        offset_e2=0.0,
        spin_rad=0.0,
        **shank_kw,
    )
    center = evaluate_threading(center_shanks, sections)
    print(
        f"\nAt offset=(0,0), spin=0:  per-section worst-shank g = "
        f"{[round(x, 3) for x in center['per_section']]}  "
        f"overall worst = {center['worst']:.3f}"
    )
    print("  (chamfer should be most negative; straight bore tightest)")

    # Quantitative sweep: feasibility % per spin
    print("\nFeasibility sweep over (e1, e2) at varying spin:")
    print(f"  {'spin':>5}  {'feas%':>6}  {'min_g':>7}  {'max_g':>7}")
    feas_pct_rows = []
    for sp in spins:
        G, _ = sweep_2d(
            axis,
            sections,
            e1_axis=e1_axis,
            e2_axis=e2_axis,
            spin_deg=sp,
            **shank_kw,
        )
        feas_pct = 100.0 * (G <= 0).sum() / G.size
        feas_pct_rows.append((sp, feas_pct, G.min(), G.max()))
        print(f"  {sp:>+4d}°  {feas_pct:>5.1f}%  {G.min():>+7.3f}  {G.max():>+7.3f}")

    # Plot heatmaps: one per spin, with feasibility contour overlay
    fig, axes = plt.subplots(
        2, len(spins), figsize=(2.5 * len(spins), 5.5), sharex=True, sharey=True
    )
    extent = [e1_axis[0], e1_axis[-1], e2_axis[0], e2_axis[-1]]
    for col, sp in enumerate(spins):
        G, TmB = sweep_2d(
            axis,
            sections,
            e1_axis=e1_axis,
            e2_axis=e2_axis,
            spin_deg=sp,
            **shank_kw,
        )
        # Top row: worst-g heatmap
        ax = axes[0, col]
        im0 = ax.imshow(
            np.clip(G.T, -1.0, 2.0),
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="RdYlGn_r",
            vmin=-1.0,
            vmax=2.0,
        )
        ax.contour(
            e1_axis,
            e2_axis,
            G.T,
            levels=[0],
            colors="black",
            linewidths=1.5,
        )
        ax.set_title(f"spin={sp:+d}°")
        if col == 0:
            ax.set_ylabel("Δe2 / minor (mm)")

        # Bottom row: g_top − g_bot (chamfer-vs-bore semantics check)
        ax2 = axes[1, col]
        ax2.imshow(
            TmB.T,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap="coolwarm",
            vmin=-0.5,
            vmax=0.5,
        )
        ax2.set_xlabel("Δe1 / major (mm)")
        if col == 0:
            ax2.set_ylabel("g_top − g_bot")

    fig.colorbar(im0, ax=axes[0, :], label="worst g", shrink=0.7)
    fig.suptitle(
        f"Hole #{args.hole_id}: {args.n_shanks}-shank threading "
        f"feasibility (pitch={args.shank_pitch_mm * 1000:.0f}µm)\n"
        f"Top: worst g across (shanks × sections); black contour = g=0\n"
        f"Bottom: g_top − g_bot (positive ⇒ bottom is tighter, expected)",
        y=0.99,
    )
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nWrote heatmap to {args.out}")


if __name__ == "__main__":
    main()
