"""Diagnose how well alpha-wrap envelope contains the raw probe mesh.

For each probe kind, this:
  1. Loads the raw mesh
  2. Builds the alpha-wrap envelope with the current default params
  3. Looks up the envelope SDF at every vertex of the raw mesh
  4. Reports how many vertices stick *outside* the envelope (positive SDF)
     and how far the worst offender protrudes

Positive distances from envelope = features the envelope misses
(envelope is supposed to be an outer bound, so every raw vertex should
be inside or on the envelope, ie SDF ≤ 0).
"""

from __future__ import annotations

import argparse
import os as _os
import sys as _sys
from pathlib import Path

_os.environ.setdefault("JAX_PLATFORMS", "cpu")
_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from aind_low_point.config import ConfigModel
from aind_low_point.optimization.envelope import build_alpha_wrap_envelope
from aind_low_point.optimization.sdf import build_probe_sdf_from_alpha_wrap
from aind_low_point.runtime import build_runtime_from_config


def _sdf_lookup_numpy(grid, origin, spacing, query_pts):
    """Trilinear SDF lookup, OOB queries return +large."""
    coords = (query_pts - origin) / spacing
    i0 = np.floor(coords).astype(np.int64)
    f = coords - i0
    Nx, Ny, Nz = grid.shape
    in_b = (
        (i0[..., 0] >= 0) & (i0[..., 0] < Nx - 1)
        & (i0[..., 1] >= 0) & (i0[..., 1] < Ny - 1)
        & (i0[..., 2] >= 0) & (i0[..., 2] < Nz - 1)
    )
    ix = np.clip(i0[..., 0], 0, Nx - 2)
    iy = np.clip(i0[..., 1], 0, Ny - 2)
    iz = np.clip(i0[..., 2], 0, Nz - 2)
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
    val = c0 * (1 - fz) + c1 * fz
    return np.where(in_b, val, 1e3)


def diagnose_one(name: str, mesh, alpha: float, offset: float):
    """Build envelope at (alpha, offset) and check raw vertex containment."""
    print(f"\n--- {name} (alpha={alpha}, offset={offset}) ---")
    print(f"  raw mesh: {len(mesh.vertices)} vertices, "
          f"{len(mesh.faces)} faces, bbox extents "
          f"{(mesh.bounds[1] - mesh.bounds[0]).round(2).tolist()} mm")

    env = build_alpha_wrap_envelope(mesh, alpha_mm=alpha, offset_mm=offset)
    print(f"  envelope: {len(env.vertices)} vertices, "
          f"{len(env.faces)} faces, watertight={env.is_watertight}")

    # Build the envelope SDF
    sdf = build_probe_sdf_from_alpha_wrap(
        mesh, alpha_mm=alpha, offset_mm=offset,
    )
    grid = np.asarray(sdf.grid)
    origin = np.asarray(sdf.origin)
    spacing = float(sdf.spacing)

    # Look up envelope SDF at every raw-mesh vertex
    raw_verts = np.asarray(mesh.vertices, dtype=np.float64)
    sdfs = _sdf_lookup_numpy(grid, origin, spacing, raw_verts)

    # In-bounds vertices only
    n_oob = int(np.sum(sdfs >= 1e2))
    in_b = sdfs < 1e2
    sdfs_in = sdfs[in_b]
    n_total = len(sdfs_in)

    # Envelope should contain raw mesh; positive SDF = vertex outside envelope.
    n_outside = int(np.sum(sdfs_in > 0))
    pct_outside = n_outside / max(n_total, 1) * 100
    max_protrusion = float(sdfs_in.max()) if n_total > 0 else 0.0
    p99_protrusion = float(np.quantile(sdfs_in, 0.99)) if n_total > 0 else 0.0
    p50_signed = float(np.median(sdfs_in))

    print(f"  vertices in-bounds: {n_total}/{len(sdfs)} ({n_oob} OOB)")
    print(f"  outside envelope: {n_outside} ({pct_outside:.2f}%)")
    print(f"  max protrusion: {max_protrusion:+.4f} mm "
          f"(p99={p99_protrusion:+.4f}, median signed dist={p50_signed:+.4f})")
    if n_outside > 0:
        worst_idx = int(np.argmax(sdfs))
        worst_pt = raw_verts[worst_idx]
        print(f"  worst protrusion at vertex {worst_idx} "
              f"({worst_pt.round(3).tolist()})")
    return dict(
        n_outside=n_outside,
        max_protrusion=max_protrusion,
        p99=p99_protrusion,
        pct_outside=pct_outside,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument(
        "--probe-kind",
        action="append",
        help="probe kind to diagnose (repeat for multiple)",
    )
    p.add_argument("--alphas", type=str, default="0.5,0.2,0.1")
    p.add_argument("--offsets", type=str, default="0.05,0.2")
    args = p.parse_args()

    cfg = ConfigModel.from_yaml(args.config)
    runtime = build_runtime_from_config(cfg)
    if args.probe_kind is None:
        kinds = sorted(set(
            p.kind for p in runtime.plan_state.probes.values()
        ))
        print(f"Auto-detected probe kinds: {kinds}")
    else:
        kinds = list(args.probe_kind)
    alphas = [float(x) for x in args.alphas.split(",")]
    offsets = [float(x) for x in args.offsets.split(",")]

    for kind in kinds:
        try:
            geom = runtime.asset_catalog.get_geometry(f"probe:{kind}")
            mesh = geom.raw
        except Exception as e:
            print(f"\n[{kind}] could not load mesh: {e}")
            continue
        print(f"\n========== probe:{kind} ==========")
        for alpha in alphas:
            for offset in offsets:
                diagnose_one(f"probe:{kind}", mesh, alpha=alpha, offset=offset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
