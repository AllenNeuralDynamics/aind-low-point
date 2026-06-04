"""Compare the manually-found T12 plan's per-probe spin against the spin-restore
(+ADAM) result from ``restore_well_adam_manual`` for the manual candidate (4195).

The manual plan's ``spin`` field is in the SAME convention as the optimizer's
``spin_deg`` — both feed ``arc_angles_to_affine(ap, ml, spin)`` (verified here by
a tip round-trip), so the values are directly comparable with circular deltas.

The restore/ADAM arrays below are pasted from a ``restore_well_adam_manual`` run
(restore = 8-pt-grid argmin; ADAM-A = single-basin refined; chain-A = durable
rerank pose). Update them if you re-run with different settings.

Run:  JAX_PLATFORMS=cpu uv run --python 3.13 -m scripts.compare_manual_spin
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

import yaml

from aind_low_point.config import ConfigModel, PlanningModel
from aind_low_point.planning import _resolved_angles
from aind_low_point.runtime import build_runtime_from_config
from aind_low_point.runtime.export import apply_plan_model_to_state
from aind_low_point.state_change import PlanStore

CONFIG = "examples/836656-config-T12.yml"
PLAN = "examples/836656-config-T12.plan.yml"

# --- from restore_well_adam_manual (cand 4195, restore WITH well) -------------
# probe order is rt.plan_state.probes (printed by the experiment).
RESTORE_WELL = {  # 8-pt-grid restore argmin
    "MD": -45.0, "BLA": -180.0, "PL": 135.0, "VM": -90.0,
    "RSP": 0.0, "CA1": 90.0, "CLA": 90.0,
}
ADAM_A_WIN = {  # single-basin ADAM refinement (FCL +0.053)
    "MD": -35.5, "BLA": -177.7, "PL": 130.7, "VM": -95.4,
    "RSP": 10.0, "CA1": 89.8, "CLA": 57.0,
}
CHAIN_A = {  # durable rerank pose (restore->L-BFGS->ADAM, FCL +0.075)
    "MD": -34.4, "BLA": -177.9, "PL": 131.0, "VM": -4.8,
    "RSP": 10.5, "CA1": -74.6, "CLA": 73.7,
}


def circ(a, b):
    """Smallest signed circular difference a-b in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def main() -> int:
    cfg = ConfigModel.from_yaml(CONFIG)
    rt = build_runtime_from_config(cfg)
    template = PlanningModel.model_validate(yaml.safe_load(open(PLAN)))
    store = PlanStore(rt.plan_state)
    apply_plan_model_to_state(template, store)

    names = list(rt.plan_state.probes)

    print(f"manual plan: {PLAN}")
    print(f"probe order: {names}\n")
    print(f"{'probe':>5} {'kind':>17} {'manual':>8} {'restore':>8} "
          f"{'adamA':>8} {'chainA':>8} | {'|Δ man-restore|':>15} "
          f"{'|Δ man-adamA|':>13}")

    rows = []
    for n in names:
        _ap, _ml, spin_m = _resolved_angles(n, store.state)
        d_rest = abs(circ(spin_m, RESTORE_WELL[n]))
        d_adam = abs(circ(spin_m, ADAM_A_WIN[n]))
        rows.append((n, d_rest, d_adam))
        kind = store.state.probes[n].kind
        print(f"{n:>5} {kind:>17} {spin_m:>8.1f} {RESTORE_WELL[n]:>8.1f} "
              f"{ADAM_A_WIN[n]:>8.1f} {CHAIN_A[n]:>8.1f} | "
              f"{d_rest:>15.1f} {d_adam:>13.1f}")

    match_rest = sum(1 for _, dr, _ in rows if dr <= 20)
    match_adam = sum(1 for _, _, da in rows if da <= 20)
    print(f"\nwithin 20° of manual:  restore-grid {match_rest}/{len(rows)},  "
          f"adamA-refined {match_adam}/{len(rows)}")
    far = [n for n, _, da in rows if da > 20]
    print(f"probes where the feasible ADAM plan differs from manual (>20°): {far}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
