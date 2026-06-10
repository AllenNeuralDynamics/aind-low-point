# Placement-optimizer — sharpened pipeline proposal & action plan

**Status:** proposal + open work, 2026-06-04. Companion to `dev/PIPELINE.md`
(the *read-verified current* pipeline). This file is the *target* and the
*to-do*. When the two disagree, PIPELINE.md describes what runs today; this
file describes where we're going and what's unvalidated.

---

## Target pipeline (the proposal)

```
1. visibility atlas              (unchanged; samples ap/ml/spin, offset/depth=0, oval_slack)
2. IMPROVED enumeration          (MRV bitset enumerator replaces arc-first; KEEP spin seeds)
3. round-robin spin restore      (unchanged; the primary spin search)
   + beam/heuristic spin basins  (H1-H4; CURRENTLY UNWIRED — see open Q)
4. batched ADAM                  (coverage ON; multi-fidelity early-cull) — NO L-BFGS upstream
5. soft-SDF feasibility floor    (cheap one-sided cut; the real feasibility floor)
6. coverage+diversity selection  (MMR over coverage; pick diverse top-N for the expensive stage)
7. trust-constr Phase 2          (hard constraints + coverage maximization)
8. FCL as REPORTED confidence    (annotation, NOT a gate)
9. MMR rank + handoff export     (the terminus: ~10-15 diverse review-ready plan YAMLs)
```

### Deltas vs the current pipeline (PIPELINE.md)
- **Drop the L-BFGS reduced polish** (current Stage 0 step 4). ADAM polishes
  directly from the round-robin restore output (`y0_restored`). **This is the
  central unvalidated bet — see Open Questions.**
- **MRV enumerator** (`optimization.pipeline.enumeration.Enumerator`) owns the
  production candidate pool and uses `enumeration.seed_emission.emit_seed` for
  lazy AP/ML/spin seeds.
  *NB: this does NOT shrink the search space — the MRV set ≈ the same ~8908.
  Its value is legibility, sound joint-ML feasibility, and the decision-tree.*
- **Multi-fidelity ADAM (early-cull)** — the real throughput lever (see below).
- **Soft-SDF cut is the feasibility floor; FCL is reporting.** Coverage stays ON
  in ADAM (mandatory — it anchors recording-center→target and prevents the
  depth-retraction packing-cheat; verified `coverage:True` in the durable run).

---

## Multi-fidelity ADAM (early-cull) — the throughput design

The rerank is the bottleneck (~8908 cands × 3 basins × 150 steps). The cull must
happen *inside* ADAM. Two axes, both using the **soft SDF-min already in the
loss (free signal)**:

1. **Early basin-select (instant ~3×):** run all 3 basins ~20-30 steps, pick the
   leading basin per candidate, finish only that one. (Today basin-select is
   after the full 150 steps.) *Validate: basin-rank stability — does the step-20
   leader stay the leader?*
2. **Early infeasibility cull (successive halving):** after the low-fidelity
   pass, drop candidates with deeply-negative, non-improving soft-min; run only
   the *uncertain* survivors to full steps.

- **Cutoff anchor:** soft-min `< -0.25 mm` ⇒ guaranteed FCL collision (one-sided;
  soft *over*-reports clearance so a deep-negative reject is safe). From
  `sdf_vs_fcl_proxy` work.
- **Fits the existing chunked structure** (`CHUNK=64`): run all chunks K₀ steps →
  gather survivors into a smaller batch → re-run at full steps (gathering
  reclaims compute vs masking).
- **Validation gate:** does early-cull drop any *eventual* feasibles? Compare
  full-150 vs early-cull feasible sets on a sample; cutoff must be conservative.
- **Report how many the cut dropped** — keep the floor honest (no silent
  truncation).

---

## Load-bearing findings (do NOT re-derive — this cost us a whole session)

1. **L-BFGS reduced polish is LIVE and load-bearing**, not superseded by ADAM.
   `_slsqp_reduced` is **L-BFGS-B** despite the name. Across the pool it moves
   spin ~91° mean off the enum seed.
2. **Every ADAM evaluation ever run seeded from L-BFGS output**
   (`augmented_phase1_x`). No ADAM run has started from `y0_restored` or the raw
   enum seed. So "ADAM replaces L-BFGS" is **untested**.
3. **The pilot (`pilot_cheap_basins`) actively contradicts the replacement
   claim.** With L-BFGS upstream, the L-BFGS *incumbent* spin basin won:
   - cand 4195: incumbent **+0.147** > cheap4 +0.100 ≈ beam4 +0.099
   - cand 1035: incumbent **+0.127** (feasible); cheap4/beam4 **0/4 feasible**
   "Beam unnecessary" held *only because L-BFGS already nailed the spin*.
4. **The round-robin spin restore is the primary spin search**
   (`batched_spin_restore.py`: 8 spins × 2 rounds, full circle, coordinate
   descent). The H1-H4 **beam search is a prototype, UNWIRED**
   (`spin_heuristic_search.main`); only its helpers `is_four_shank` /
   `spin_to_align_y_with` are used live.
5. **The "four-shank doesn't flip" concern is largely moot** — the restore
   sweeps the full circle for all probes (incl. quadbase), so quadbase 180°
   flips are already searched. The ADAM rerank's 3 cheap basins are secondary
   refinement around the restore output.
6. **MRV enumerator validated** (`scripts/arc_first_mrv.py`): superset of the 45
   FCL-feasibles (44/45 at margin 0, **45/45 at ml-margin ≥0.5°**), manual
   reachable. Joint-ML greedy-pack prunes ~350 (4%) that can't actually ML-pack
   (sound, vs the unsound pairwise-max-diff). At the production-matched setting
   it reproduces ~8908. Helly clique = pairwise AP-overlap (1-D), so AP
   feasibility is a one-bitmask test.
7. **Atlas freedom profile = the decision tree, pre-polish.** PL (2 holes) and
   RSP (5) are near-locked; VM (14)/CA1 (12)/MD (10)/CLA/BLA flex. The feasible
   set has a fixed PL/RSP/BLA skeleton + ~7-11 distinct basins (MD bimodal
   12-vs-3, VM the wild card). **Handoff target is ~10-15 diverse plans, not 80.**
8. **Atlas samples ap/ml/spin only**; offset/depth pinned 0; 20% `oval_slack`
   compensates. Multi-shank threading IS modeled (all shanks through bore;
   3 quadbase MD/BLA/PL are 4-shank, 4 NP2.1 single-shank).

---

## Open questions / unvalidated bets

1. **Can ADAM replace L-BFGS?** — the central question. Pilot leans NO.
2. **Does the beam add value without L-BFGS?** — untested (pilot was contaminated
   by L-BFGS upstream). C below tests it.
3. **Does early-cull preserve eventual-feasibles?** — needs the cutoff validated.
4. **Should the MRV enumerator actually go to production?** — only if its
   legibility/decision-tree value is worth the wiring; it won't help throughput.

### The deciding experiment (NEVER RUN) — build ONE rig for all of it
Seed ADAM from `y0_restored` (the restore output, *before* L-BFGS — exposed
cleanly inside `polish_all_with_batched_spin_restore` step 3) with **per-step
soft-min logging**, and run:

| config | chain | tests |
|---|---|---|
| A (baseline) | restore → L-BFGS → ADAM | current |
| B | restore → ADAM, cheap basins | can ADAM replace L-BFGS? |
| C | restore → ADAM, beam basins | does beam matter w/o L-BFGS? |

**Null hypothesis = keep L-BFGS.** B/C must match A at scale (n≫2) on FCL
feasibility + rank (Spearman) + the recovered handoff set. The per-step soft-min
log simultaneously answers the early-cull question (when does the
feasible/infeasible split become decidable). **One rig, both answers.**

---

## Action items (ordered)

1. **Build the restore-seeded ADAM experiment rig** with per-step soft-min
   logging (hook = `y0_restored` in `polish_all_with_batched_spin_restore`).
2. **Run A/B/C** on a representative set (the 45 feasibles + manual #4195 +
   stratified sample), then scale if promising. Decide L-BFGS keep/drop.
3. **Analyze the soft-min trajectories** → set the early-cull step K₀ and cutoff;
   validate no eventual-feasibles are culled.
4. **If dropping L-BFGS:** wire `restore → ADAM` into the pool build; if keeping
   beam, wire `beam_search_assignments` into the basin construction.
5. **Implement multi-fidelity ADAM** (early basin-select + soft-min cull) in the
   rerank.
6. **Loosen `export_handoff`:** FCL becomes an annotated column (clean/marginal/
   violating), not a `>= -0.2` filter; soft-SDF cut is the floor; MMR-by-coverage
   selects ~10-15 diverse plans; report drop counts.
7. **MRV production hardening:** keep seed emission lazy and preserve the
   enumerator's decision-tree diagnostics when changing pool ranking.

---

## Key parameters / data points (so they survive compaction)

- Durable rerank config: `steps:150, flip_degs:[0,180] (3 basins w/ inc),
  n_surf:5000, bf16_store:True, chunk:64, fcl_topk:100, coverage:True,
  soft_fixtures:[well], fcl_fixtures:[headframe,cone,well]`.
- Spin restore: `n_spins:8, n_rounds:2, spin_restore_chunk:100`.
- Reduced polish: `reduced_slsqp_max_iter:50` (L-BFGS-B).
- Soft-SDF reject anchor: `< -0.25 mm` ⇒ guaranteed FCL collision (one-sided).
- MRV ml-margin: `0.5°` keeps all 45 feasibles (0.4° was the single miss).
- Per-probe atlas hole freedom: PL=2, RSP=5, MD=10, CLA=12, CA1=12, BLA=13, VM=14.
- Config: `examples/836656-config-T12.yml` + `scratch/0283-300-04.holes.yml`.
- Durable artifacts: `scratch/full_polish_0283.pkl` (pool),
  `full_rerank_0283.pkl` (ADAM), `phase2_handoff.pkl`, `scratch/handoff/`.

## Repo state after the 2026-06-04 cleanup
- Committed: bf16 trilinear, vmap coverage, points_in_region reducer, 837229
  config, `alp-plan` CLI (tyro + startup plan-apply), spin orbit-basis refactor,
  live pipeline scripts tracked, 73 stale diagnostics deleted, `dev/PIPELINE.md`.
- `CLAUDE.md` is gitignored in this repo (edits are local-only).
- Live scripts tracked; active spin/ADAM/enumerator exploration scripts kept;
  experiment-output configs + `.claude/` gitignored.
