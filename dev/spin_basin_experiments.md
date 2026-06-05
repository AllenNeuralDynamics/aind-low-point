# Spin-basin experiments — does heuristic spin generation help?

**Status:** 2026-06-04. Investigated whether heuristic spin proposals (H1/H2 +
beam search) add value over the production round-robin restore for choosing
per-probe spin. **Conclusion: no — the restore already lands on the durable
feasible spin basin; the beam systematically misses it.** Keep the restore;
do not ship the beam/heuristics as a production spin generator.

The optimizer config is `examples/836656-config-T12.yml` +
`scratch/0283-300-04.holes.yml`; the durable artifacts are
`scratch/full_polish_0283.pkl`, `full_rerank_0283.pkl`, `phase2_handoff.pkl`.

## Background: the two ways to pick spin

- **Round-robin restore** (`batched_spin_restore.py`): 8-spin full-circle
  coordinate descent (2 rounds) on the reduced objective — **one basin per
  candidate**. Production seed path. (Now well-aware, see `f3205ce`.)
- **Heuristic beam** (`spin_heuristic_search.py`): H1 (threading slot/flip) +
  H2 (narrow-profile-faces-contact) → a **set** of proposals. Prototype, never
  wired into production.

Spin coupling for the heuristics was fixed this session: the old 15 mm
target-distance test (≈ whole mouse brain → fully coupled) was replaced by
**swept-volume intersection** — revolve each probe about its insertion axis,
overlap = spins can interact (`8dee98f`, `4d69cd8`).

## The experiments (scripts in `scripts/`)

| script | question | finding |
|---|---|---|
| `restore_convergence` | is the 2-round restore converged? | **No** — converges at round 4 for cand 4195; VM/CLA *oscillate* (180° ping-pong) before settling. 2-round is a transient. |
| `restore_rounds_adam` (**A**) | does converged seed beat 2-round → ADAM? | **No** — different VM/CLA basins, but *equivalent* quality (cov 16.29 vs 16.32). The loose probes have multiple equivalent feasible basins. |
| `beam_restore_cluster` (**B**) | do the beam's proposals survive the restore? | **No** — 65 diverse proposals (beam + heuristic) → **1 basin**. The restore is a total attractor; the beam is redundant downstream of it. |
| `beam_adam_cluster` (**C**) | is ADAM also an attractor? | **No** — ADAM is *local*: 65 seeds → 65 basins. But seeding ADAM directly (no restore) gives mostly-infeasible / lower-coverage basins than the restore basin. |
| `spin_similarity` | do restore / beam reproduce the *durable feasible* spins? (5 cands) | **Restore yes, beam no** (see below). |

## The decisive result: `spin_similarity` across 5 candidates

Compare each candidate's final Phase-2 (feasible) per-probe spin to the restore
output and the nearest beam proposal:

```
cand   durable cov   restore→durable      beam-best→durable
                      match   max Δ        max Δ (best of 65)
4195    15.96         5/7     166°         178°
1035    17.44         7/7      13°         176°
4747    17.34         7/7      23°         178°
6423    16.95         7/7      23°         152°
 697    16.66         4/7      59°         174°
```

- **Restore reproduces the durable feasible basin** — 7/7 probes within ~13–23°
  on 3/5 candidates; the 4195/697 misses are the under-determined VM/CA1 probes
  (equivalent basins, per A).
- **The beam never does** — its *best* of 65 proposals is always ≥152° off on
  some probe, and for many probes **zero** proposals match the durable spin
  (its candidate generation doesn't contain it).

## Why

The restore optimizes *actual* reduced clearance + the well, which tracks
feasibility. The beam optimizes a geometric facing proxy
(narrow-profile-faces-contact) that reproduces *human-sensible* spins (it
nearly regenerated the manual plan's spins, within ~14°) but **not** the
*feasible-optimal* ones. So the heuristics explain the human's reasoning but
don't beat the restore.

## Takeaways

- **Keep the restore** as the spin-basin finder. Do not wire the beam/heuristics
  into production as a spin generator.
- **2 rounds is fine** — convergence doesn't improve final quality.
- **Diversity for the handoff comes from different *candidates*** (hole/arc
  assignments), not from spin-basin generation on one candidate.
- The swept-volume coupling + facing analysis remain useful for *understanding*
  (and the coupling fix is committed); they are not a production spin pipeline.
- Compile caveat: the restore loop is Python-unrolled (`n_rounds × K` copies of
  the 8-way-vmapped heavy objective), so high `n_rounds` compiles pathologically
  — use `lax.scan` if more rounds are ever needed (they aren't).
