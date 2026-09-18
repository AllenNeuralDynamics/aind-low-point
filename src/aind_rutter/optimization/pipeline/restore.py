"""Runtime setup and spin readout shared with the Phase-1 pool stage.

``setup`` builds the optimizer assets for the subject selected by ``CONFIG`` and
``HOLES``. ``spins_deg_from_reduced`` reads per-probe spins out of a
reduced-layout pose, whose probe blocks follow the arc angles as
(ml, cos spin, sin spin).
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import numpy as np

from aind_rutter.optimization.pipeline.runtime_adapter import (
    OptimizationRuntime,
)
from aind_rutter.optimization.pipeline.settings import PipelineSettings

# Variables per probe in the full Phase-1 layout.
PPV = 6
N_SURF = int(_os.environ.get("N_SURF", "5000"))
# Subject is config-driven: CONFIG selects the YAML, HOLES the implant-bore file
# (placed into the scene by the config's own implant_to_lps in setup()). Both
# come from PipelineSettings so that every stage resolves them the same way —
# RUTTER_CONFIG ahead of CONFIG — rather than each reading the bare name and
# sending the phases to different subjects.
_SHARED = PipelineSettings()
CONFIG = _SHARED.config
HOLES = _SHARED.holes


def setup_runtime() -> OptimizationRuntime:
    return OptimizationRuntime.from_config_path(CONFIG, HOLES)


def setup(runtime: OptimizationRuntime | None = None):
    """9-tuple setup (cfg, runtime, probes, holes, probe_sdfs, probe_bvhs,
    fixtures, well_fixture, fixture_bvhs) for callers that still unpack it. New
    code should use ``OptimizationRuntime`` + its ``build_problem_assets``
    directly (as ``phase2_ipopt`` does)."""
    opt = setup_runtime() if runtime is None else runtime
    assets = opt.build_problem_assets(n_surface_points=N_SURF)
    return (
        opt.cfg,
        opt.runtime,
        list(opt.probes),
        list(opt.holes),
        assets.probe_sdfs,
        assets.probe_bvhs,
        assets.fixtures,
        assets.well_fixture,
        assets.fixture_bvhs,
    )


def spins_deg_from_reduced(y_red, n_arcs, K):
    out = []
    for k in range(K):
        sx = y_red[n_arcs + 3 * k + 1]
        sy = y_red[n_arcs + 3 * k + 2]
        out.append(float(np.degrees(np.arctan2(sy, sx))))
    return np.array(out)
