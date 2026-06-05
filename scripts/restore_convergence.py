"""Is the round-robin spin restore converged at 2 rounds?

The restore loop is Python-unrolled, so compiling n_rounds=8 directly is
pathological. Instead run ONE round (one compile) and feed its output back as
the seed — each pass is one more coordinate-descent round, reusing the cached
kernel. Reports the per-probe spin and the change vs the previous round;
coordinate descent on the discrete 8-pt grid is converged once a round changes
nothing.

Run:  JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.restore_convergence
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("JAX_PLATFORMS", "cuda")

import pickle

import numpy as np

from scripts.restore_well_adam_manual import (
    run_restore,
    setup,
    spins_deg_from_reduced,
)

IDX = 4195
MAX_ROUNDS = 8


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def main() -> int:
    _cfg, _rt, probes, holes, sdf_by_name, _bvh, _fx, well, _fb = setup()
    K = len(probes)
    names = [p.name for p in probes]
    pool = pickle.load(open("scratch/full_polish_0283.pkl", "rb"))
    cand = pool["candidates"][IDX]
    n_arcs = int(pool["results"][IDX].n_arcs)

    print(f"cand {IDX}  probes={names}\n")
    print(f"{'round':>5}  {'spins (deg)':<46}  max|Δ vs prev|")
    prev = None
    seed = None   # round 1 from the atlas seed
    for rnd in range(1, MAX_ROUNDS + 1):
        y = run_restore(cand, probes, holes, sdf_by_name, n_arcs, well,
                        with_well=True, n_rounds=1, seed_spins_deg=seed)
        sp = spins_deg_from_reduced(y, n_arcs, K)
        if prev is None:
            dmax = "--"
        else:
            dm = float(np.abs(_wrap(sp - prev)).max())
            dmax = f"{dm:.1f}°"
        print(f"{rnd:>5}  {str(np.round(sp, 0).astype(int).tolist()):<46}  {dmax}",
              flush=True)
        if prev is not None and float(np.abs(_wrap(sp - prev)).max()) < 0.5:
            print(f"\nconverged at round {rnd} (no probe moved ≥0.5°)")
            break
        prev = sp
        seed = sp        # feed back as the next round's seed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
