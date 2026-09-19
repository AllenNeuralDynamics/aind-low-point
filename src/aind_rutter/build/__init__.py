"""Turning a validated `ConfigModel` into a `RuntimeBundle`.

- `assemble`     — `build_runtime_from_config` and the `RuntimeBundle` it returns
- `transforms`   — config transform recipes and refs compiled to affine chains
- `loaders`      — registry-driven file loaders (trimesh, sitk_volume, csv_points)
- `reducers`     — registry-driven point reducers
- `canonicalize` — orientation flip, scale and transform into canonical LPS mm
- `chem_shift`   — the chemical-shift correction context
- `calibration`  — the probe calibration bank, and the NewScale frame it is in
- `queries`      — semantic scene-node geometry in world LPS
- `probe_context`— a planned probe's target and asset context

Reading a plan back out lives next door, in `aind_rutter.plan_io`.
"""

from aind_rutter.build.assemble import RuntimeBundle, build_runtime_from_config

__all__ = ["RuntimeBundle", "build_runtime_from_config"]
