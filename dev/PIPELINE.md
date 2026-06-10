# Placement-optimizer pipeline — verified map

**Status:** read-verified against code 2026-06-04. This document exists because
the pipeline's docstrings, function names, and prior session notes drifted out
of sync with the code and repeatedly misled investigation. **When in doubt,
trust this file's "verified" sections (traced through imports + the actual
call sites), not docstrings.** Stale-docstring hazards are listed at the end.

The optimizer is an **offline batch pipeline**: each stage is a script that
writes a durable `scratch/*.pkl` consumed by the next. It runs on
`examples/836656-config-T12.yml` + `scratch/0283-300-04.holes.yml`.

---

## Verified current pipeline

### Stage 0 — pool build → `scratch/full_polish_0283.pkl`
Driver: `scripts/run_full_polish.py`. Steps (all verified):

1. **Visibility atlas** — `optimization/visibility_atlas.py::build_visibility_atlas`
   (n_top=128, n_spin=72). Multi-shank threading atlas: for each `(probe, hole)`
   it projects *all* shank tips through the bore sections and requires every
   shank to thread, swept over an AP×spin grid. **Samples (ap, ml, spin) only —
   `offset_R/A` and `past_target/depth` are pinned at 0**; a 20% `oval_slack`
   expands the bore ellipse to compensate for those unsampled DOFs. (The *other*
   atlas, `optimization/atlas.py::build_atlas`, is LEGACY — see below.)
2. **Arc-first enumeration** — `optimization/arc_first_principled.py::enumerate_arc_first_candidates`.
   Discrete decision unit = `(arc-partition, hole-tuple)`. Emits ~8908
   `ArcFirstCandidate` for this config, each carrying `ml_seed`/`spin_seed`
   (nearest-AP atlas anchor) and a single principled arc-AP seed. AP/ML/arc
   feasibility uses per-`(probe,hole)` AP-envelope intersection + 16° seps.
3. **Spin restore + reduced polish** —
   `optimization/parallel_stage2.py::polish_all_with_batched_spin_restore`:
   - Build `y0` from the **enum seed** (`arc_centroids_deg`, `ml_seed`,
     `spin_seed → (sx,sy)`).
   - **Round-robin spin restore** (`optimization/batched_spin_restore.py`):
     8 spins × 2 rounds, full circle, GPU-batched **coordinate descent** on the
     reduced objective (for each probe in turn, sweep the 8-point spin grid,
     take argmin, fix it). → `y0_restored`.
   - **Reduced polish** — `polish_all(..., y0_per_candidate=y0_restored,
     skip_spin_restore=True, reduced_slsqp_max_iter=50)` →
     `joint_rerank.score_joint` → `joint_rerank._slsqp_reduced`. **This is
     scipy `method="L-BFGS-B"` despite the `_slsqp_` name.** Returns
     `JointCandidate` with the polished `reduced_y`.
4. **Offset augment** — `scripts/augment_polish_with_offsets.py`: lift each
   `results[i].reduced_y` to the Phase-1 layout, run an offset-only L-BFGS-B
   (~20 iter) → `augmented_phase1_x`.
5. **Violation eval** — `scripts/eval_violation_at_augmented.py` → `violation_fn`.

Pool dict keys: `candidates` (8908 `ArcFirstCandidate`, have `ml_seed`, **no**
`reduced_y`), `results` (8908 `JointCandidate`, **have** `reduced_y`),
`augmented_phase1_x`, `violation_fn`, `coverage_at_aug`, `offset_polish_fn`,
`manual_rank` (=4195). *Both lists are present — that dual structure is what
caused the "candidates have no reduced_y" confusion. `results[i].reduced_y` is
the L-BFGS output; `augmented_phase1_x` is built from it.*

### Stage 1 — batched-ADAM rerank → `scratch/full_rerank_0283.pkl`
Driver: `scripts/batched_full_rerank.py` (GPU).
- **Seeds each candidate from `augmented_phase1_x[idx]`** (= the L-BFGS output;
  offsets are then zeroed).
- **Spin basins = 3:** `[inc, h1, 1-shank-flip]` where `inc = extract_spins(aug)`
  (the L-BFGS-refined spin), `h1 = spin_to_align_y_with(slot_major)`, and the
  flip applies only to single-shank probes (`FLIP_DEGS=[0,180]`). **No beam
  search.** Four-shank probes are not flipped in the basins, but the Stage-0
  round-robin restore already searched their full-circle spin.
- One vmapped ADAM pass (150 steps, 5000 surface pts, bf16 grid storage),
  per-candidate basin-select, FCL top-K (`optimization/stage3_phase3_fcl.py`).
- Output: `records=[{idx, n_arcs, viol, pose, ...}]` + `source_pool=...`.

### Stage 2 — feasible ingestion → `feasibles_by_coverage.pkl`, `ingest_top_enriched.pkl`
`scripts/ingest_feasibles.py` (FCL all → coverage-rank feasibles) and
`scripts/ingest_analysis.py` (coverage/FCL/diversity over the rerank top-K;
also imported as a library by the Phase-2 scripts).

### Stage 3 — Phase 2 polish + handoff ranking → `scratch/phase2_handoff.pkl`
Driver: `alp-phase2` (`optimization/pipeline/phase2_ipopt.py`). Runs `optimization/stage3_phase2_jax.py`
(trust-constr, hard constraints + coverage) on the selected top-N, FCL-gates
(`stage3_phase3_fcl.py`), MMR-diversity-ranks. `scripts/phase2_throughput.py`
is the **diagnostic** timing variant (not the durable producer).

### Stage 4 — handoff export → `scratch/handoff/{tree.txt, manifest.md, plans/*.yml}`
Driver: `scripts/export_handoff.py` (`--plans` → `scripts/export_handoff_plans.py`,
which decodes each Phase-2 pose into a `PlanningModel` YAML with a pose
round-trip check).

### One-line summary
```
visibility atlas → arc-first enumerate → round-robin spin restore
  → L-BFGS reduced polish (reduced_y) → offset augment (augmented_phase1_x)
  → batched ADAM rerank (3 cheap spin basins) → ingest → trust-constr Phase 2
  → FCL gate → handoff export
```

---

## Spin handling (verified)

- **Primary spin search = the round-robin restore** in Stage 0 (full circle, all
  probes, coordinate descent). This is where the spin basin is actually chosen;
  L-BFGS then refines continuously within it. Across the pool, the combined
  restore+L-BFGS moves spin ~91° (mean) off the enum seed.
- **ADAM rerank uses 3 cheap basins**, not a beam search.
- **The H1–H4 beam search** (`scripts/spin_heuristic_search.py::main`,
  `dev/spin_search_heuristics.md`) is a **prototype, NOT wired into production**.
  Only its helpers `is_four_shank` and `spin_to_align_y_with` are imported live.

---

## ⚠️ The L-BFGS-vs-ADAM caveat (the thing we kept re-discovering)

**Every ADAM evaluation to date — exploratory and the durable pool — seeds from
`augmented_phase1_x`, i.e. L-BFGS output. No ADAM run has ever started from the
raw restore (`y0_restored`) or enum seed.** (Verified: all of
`batched_full_rerank`, `batched_basin_select_run`, `batched_adam_test`,
`basin_select_prototype`, `pilot_cheap_basins`, `try_fulldof_4195` read
`data["augmented_phase1_x"][idx]`.)

Consequences:
- The headline "batched-ADAM basin-select un-buries manual #4195 (pool-rank
  4641 → #1)" is a **re-ranking** result *on top of* L-BFGS poses, **not**
  evidence that ADAM can **replace** L-BFGS.
- The `pilot_cheap_basins` "cheap basins ≈ beam, beam unnecessary" verdict was
  reached **with L-BFGS upstream**. Its own data (n=2):
  ```
  cand 4195: incumbent(L-BFGS) +0.147  > cheap4 +0.100 ≈ beam4 +0.099
  cand 1035: incumbent(L-BFGS) +0.127  ; cheap4 -1.000 (0/4) ; beam4 -1.000 (0/4)
  ```
  The **L-BFGS incumbent basin won both**, and on 1035 was the *only* feasible
  basin. So "beam unnecessary" holds *only because L-BFGS already nailed the
  spin*. Pull L-BFGS and that winning basin disappears.

**Bottom line: "ADAM (or beam) replaces L-BFGS" is UNSUPPORTED and mildly
contradicted by the pilot's own data. Treat the L-BFGS reduced polish as
load-bearing until proven otherwise.**

---

## Intended pipeline (the goal — NOT yet validated)

Collapse Stage 0's per-candidate scipy L-BFGS reduced polish into the batched
GPU ADAM, seeding ADAM from `y0_restored` (the round-robin restore output,
*before* L-BFGS). That keeps the valuable spin search (the restore) and removes
the slow serial scipy stage — the original `vmap_cpu_gpu_polish_arch` intent.

**The deciding experiment (never run):** seed ADAM from `y0_restored` and compare

| config | chain |
|---|---|
| A (current) | restore → L-BFGS → ADAM |
| B | restore → ADAM, cheap basins |
| C | restore → ADAM, beam basins |

**Null hypothesis = keep L-BFGS.** B/C must *prove* they match A at scale
(n≫2), on FCL feasibility and rank. The restore output is exposed cleanly inside
`polish_all_with_batched_spin_restore` (step 3, before the `polish_all` call),
so wiring B/C is a small hook. Until this runs, do not remove L-BFGS.

---

## Retired code (deleted)

The old SLSQP Stage-3 "inner-solve" chain was **removed** (commit `a0d4ef3`),
having been superseded by the Phase-1/2/3 pipeline above. Deleted:

- `optimization/optimize.py::_inner_solve_one`, `optimize`, `polish_seed`, … —
  the whole module is now just the `ProbeStaticInfo` dataclass.
- `optimization/atlas.py::build_atlas`, `atlas_stage1` — superseded by
  `visibility_atlas` + arc-first. *Kept:* `Atlas`/`AtlasEntry`/`PoseAnchor`
  dataclasses (imported by `arc_first_principled`).
- `optimization/joint_rerank.py::optimize_joint`, `_inner_solve_worker*` — the
  LSAP/atlas_stage1 reranker driver. *Kept:* `score_joint`, `_slsqp_reduced`,
  `_build_probe_static`, `JointWeights`, `JointCandidate`.
- `optimization/sdf_clearance.py`, `optimization/stage3_jax.py` — only reached
  from `_inner_solve_one`. Whole modules deleted.

The shared probe-setup helpers that the legacy `run_optimizer.py` used to host
(`_probe_static_info`, `_transform_holes`, `retro_opts_from_env`) now live in
`optimization/pipeline/probe_setup.py` and are imported by the live pipeline.

---

## Stale-docstring / naming hazards

- `run_full_polish.py` docstring says it runs `polish_all_adaptive`; it actually
  calls `polish_all_with_batched_spin_restore`.
- `_slsqp_reduced` is **L-BFGS-B**, not SLSQP.
- `stage3_phase1_jax.py` / `stage3_phase2_jax.py` / `stage3_phase3_fcl.py` are the
  **live** Phase-1/2/3 modules; `stage3_jax.py` + `optimize._inner_solve_one`
  are the **legacy** "Stage 3 chain" — overlapping `stage3_` prefix, different
  things.
- `scripts/test_h1_chain_cand4195.py` is misnamed — it is a **live library**
  (`build_y`, `extract_spins` used by the rerank), not a test.

## Vocabulary

Use: **enumerate → Stage-0 pool polish (restore + L-BFGS) → ADAM rerank →
Phase 2 (trust-constr) → FCL gate → handoff.** Avoid the bare "Stage 2 / Stage 3"
labels — they mean different things in different modules.
