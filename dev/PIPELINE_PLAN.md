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
5. **Is the top-200 cull discarding feasible plans, and what cheap signal
   predicts Phase-2 success?** — see *Cheap estimates of Phase-2 success* below.

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
8. **Run the Phase-2 success-estimator experiments** (section below), starting
   with the labelled benchmark.

---

## Cheap estimates of Phase-2 success (planned 2026-09-15; tooling built, GPU runs pending)

Phase 2 (IPOPT, `pipeline/phase2_ipopt.py`) costs about 18 s of compute per
candidate, roughly 90× a Phase-1 candidate, so the pipeline solves only the 200
best Phase-1 candidates by total objective (`SELECT_BY=objective`, commit
`4606a21`). That cutoff was never validated, and Phase-1 rank is the only signal
used to decide where solves go. This plan looks for a cheap estimate of whether a
candidate will come out feasible, good enough to choose which candidates get full
solves.

### Evidence so far

Data: the 837772 run (2026-06-25) and the 837229 rerun (2026-06-15), both on
implant 0283-300-04 (`scratch/0283-300-04.holes.yml`). Artifacts per config stem:
`scratch/<stem>_rerun_pool.pkl` (Phase-1 pool), `scratch/<stem>_rerun_phase2_handoff.pkl`
(the 200 solves under `all`), `scratch/mrv_seeds_<stem>.pkl` (enumeration seeds:
dict of arc count → list of `MRVCand`, joined to the pool by
`(probe_to_hole, partition)`) and `scratch/atlas_<stem>.pkl`. The atlas and seed
pickles predate the package rename and load only through an `Unpickler` that maps
`aind_low_point` to `aind_rutter`.

- The cutoff sits where solves still succeed. Feasible plans per 40 Phase-1
  positions within the top 200: 837772 27/23/21/19/21, 837229 31/28/30/24/27.
- Every failure (89 and 60) is a collision at the FCL −1 mm sentinel; none fails
  threading.
- Within the top 200, nothing tested predicts feasibility well (AUC, 837772 /
  837229): Phase-1 objective 0.58 / 0.56; Phase-1 min clearance 0.54 / 0.61;
  probe-kind layout 0.50 / 0.61; visibility-atlas stats per probe–hole pair ≈0.5,
  because the atlas tests threading only; distance travelled from the enumeration
  seeds 0.36–0.59; per-probe hole choice in a logistic model 0.62 / 0.72, the best.
  These labels are range-restricted, which weakens every predictor.
- Start-pose features already beat Phase-1 rank on the same labels
  (`scratch/estimators/start_features.py`, ~0.14 s per candidate on CPU). The number
  of pairs colliding (FCL) at the Phase-1 pose reaches AUC 0.67 / 0.74, and ranking
  by it puts 70 / 88 feasible plans in the first 100 solves against 61 / 73 in
  Phase-1 order. A two-feature model (colliding pairs + probe-pair slack violation)
  trained on one subject scores AUC 0.74 / 0.69 on the other. Hardware and brain
  slacks carry no signal (AUC 0.42–0.54), and hole choices add nothing once start
  collisions are known. The feature ranks rather than prunes: plans with one
  colliding pair at the start still end feasible 46% / 53% of the time (no
  collision: 75% / 88%).
- Across the full pools (13,454 and 14,726 candidates), single probe–hole choices
  explain 35–37% of the held-out variance in Phase-1 min clearance, and adding
  pairs of choices raises that to 53–55%. Pair effects correlate r = 0.62 between
  the two subjects, and the worst pairs (written probe@hole) repeat in both:
  BLA@h1+PL@h2, BLA@h11+MD@h3 on one arc, CA1@h3+MD@h8, CLA@h1+MD@h3 and
  BLA@h13+MD@h6 on one arc, each 0.25–0.56 mm below what the single choices predict.

### Experiments (in order)

Existing hooks: `RANKS` in `phase2_ipopt.py` solves an explicit list of offsets
into the selection order (zero-based, so position 201 is 200); cyipopt 1.7
`minimize_ipopt(callback=...)` sees every iteration; `make_phase2(...).slacks_fn(x)`
returns the full constraint slack vector at any pose; and
`FCLValidator.violating_pairs(x)` names the colliding pairs and fixtures.

0. **Setup and label noise (~20 min GPU).** Run the test suite. Re-solve ~40
   existing 837772 plans unchanged, which confirms the hole-wall change is a no-op
   for wall-free holes, and ~40 from slightly perturbed starts. If outcomes agree
   less than ~85% of the time, fix solver consistency first: no estimator can beat
   the label noise.
1. **Logging in `phase2_ipopt.py` (CPU, small).** Record objective and constraint
   violation per iteration, slack summaries by constraint type at start and end,
   and the colliding pairs at start and end. *Built 2026-09-15:* `P2_DIAG=1`,
   `P2_PERTURB` / `P2_PERTURB_SEED` and `RANKS_FILE` in `phase2_ipopt.py`; the
   logged solve lives in `pipeline/phase2_diagnostics.py` and reproduces
   `minimize_ipopt` bit for bit (`tests/test_phase2_diagnostics.py`); `make_phase2`
   exposes `slack_parts` and `slack_labels`. A CPU smoke run on two 837772
   candidates recorded every field. IPOPT's per-iteration `inf_pr` measures its
   internal slack-variable formulation and stayed at 5–37 while every constraint
   held, so early-iteration features should use the recorded per-group minimum
   slack and violation counts instead.
2. **Labelled benchmark (~1.5 h GPU).** Per subject, ~360 solves through `RANKS`,
   stratified over Phase-1 positions 201–1,000, 1,001–3,000, 3,001–7,000 and
   7,001–end, and spread over probe-kind layouts and the rare holes of constrained
   targets (PL h2/h3, MD outside h6/h8, RSP h5). Use production settings
   (`P2_ITER=1000`); low ranks may run to the iteration cap. With the existing 400
   solves this gives ~1,100 labels and the feasibility-versus-rank curve over the
   whole pool.
3. **Score cheap estimators on those labels (no new solves).** Candidates: Phase-1
   objective and clearance (baseline); constraint slacks at the Phase-1 pose; FCL
   at the Phase-1 pose (count and type of collisions, with CPU cost measured);
   constraint violation after 10/25/50/100 iterations; and a single-choice + pair
   model fitted on one subject and tested on the other. Measure feasible plans and
   distinct kind layouts per 100 solves when picking by each score, weighted by
   stratum, against Phase-1 rank, plus cost per candidate.
4. **Geometry tables (once per implant, ~30 min GPU and the most new code).** Test
   the five worst pairs in isolation first; if they clear alone, the pair effect is
   crowding and pair tables drop in priority. Then build, per probe–hole choice,
   the fraction of atlas poses (over a few depths) clear of the well, headframe and
   implant, and per pair of choices (same or different arc) the best clearance over
   sampled pose pairs. Accept pruning only if no labelled feasible plan is excluded.
5. **Prospective test (~25 min GPU).** Solve 200 candidates chosen by the best cheap
   score, spread across kind layouts, and compare with the existing rank 1–200 run
   on feasible count, distinct layouts, best coverage and total time including the
   estimator.

Prototype tooling lives in `scratch/estimators/` (gitignored):

- `sample_ranks.py <stem>` writes `scratch/estimators/<stem>/benchmark_ranks.txt`,
  `repeat_ranks.txt` and `manifest.csv`. Each position band is half uniform random,
  half chosen for kind-layout and rare-hole diversity; the manifest's `selection`
  column records which, because stratum weights are valid only for the random half
  and the top-200 census.
- `run_benchmark.sh <stem> <ranks_file> <out_pkl> [VAR=value ...]` runs production
  Phase-2 settings with `P2_DIAG=1` on an MPS process pool (3 workers, 2 on retry).
  Step 0 is the repeat set run twice: as is, and with `P2_PERTURB=1 P2_PERTURB_SEED=1`.
- `start_features.py <handoff> <out>` adds start-pose slack groups and FCL collisions
  to an existing handoff on CPU, in the same fields a `P2_DIAG=1` run records.
- `score.py <stem> --handoff ... [--manifest ...] [--other-handoff ...] [--png ...]`
  reports stratum-weighted AUC and yield per estimator. On the existing top-200
  labels a logistic model on per-probe hole choices already reaches AUC 0.63
  (837772) and 0.71 (837229), and 0.66 / 0.70 when trained on the other subject.

Decision rules: adopt an estimator that reaches ≥1.3× the feasible yield of
Phase-1 rank for the same number of solves. If constraint violation at ~50
iterations predicts the outcome at AUC ≥ 0.85, a two-stage Phase 2 (short solves
on ~1,000 candidates, full solves on the best) is the simplest change.

---

## Phase-2 solver conditioning and convergence (2026-09-16)

Measured state of the IPOPT build, the conditioning numbers, and the formulation
defects are in `dev/PHASE2_CONDITIONING.md`. Next steps, in order. The first one
decides how much the rest matter, so run it before tuning anything.

1. **One instrumented run (minutes, CPU). DONE 2026-09-16** — findings and the
   measured corrections are in `dev/PHASE2_CONDITIONING.md`; it refuted the
   regularization-thrash hypothesis and showed status 2 firing at strictly
   feasible points. `print_level=5`,
   `print_info_string="yes"` and `output_file=...` on ~5 candidates that exit
   status 2. Production runs at `print_level=0`, so nothing is visible today. The
   tag column discriminates between the competing explanations instead of leaving
   them to a sweep:
   - `!` — restoration tightened its tolerance because the original problem was
     only slightly infeasible, i.e. IPOPT gave up while nearly feasible → (2).
   - `Nj` / `L` / `S` / `a` — perturbation and singular-system tags, i.e.
     regularization thrash → (4) and (5).
   - `Ws` / `We` / `WS` — L-BFGS skipped an update (tiny step, non-positive
     `sᵀy`) → revisit `limited_memory_max_history`.
   - `e` — evaluation error: the JAX constraint path returned NaN or Inf at a
     trial point. That is a modelling bug, not a solver setting.
2. **Tolerances. DONE 2026-09-16.** `IP_ACC_TOL` is wired into
   `phase2_ipopt.py`, echoed into the run config, and pins
   `resto.acceptable_iter=0` whenever it is loosened; it defaults to IPOPT's 1e-6,
   so production is unchanged until set. On the 40-candidate pair,
   `IP_ACC_TOL=5 IP_ACC_ITER=8` cut median iterations 673 → 187 with **0 of 80
   reaching the cap** (baseline 24 of 120) and 79 of 80 exiting "acceptable"
   where the baseline gave 96 local-infeasibility plus 24 iteration-limit. Kept
   averaged 70% against the baseline's 71.3%, inside its own 57–85% spread, and
   the FCL margin of surviving plans was unchanged (+0.0092 vs +0.0087). It buys
   cost, not determinism — divergence stays at 4.5.

   **The composite is now the `phase2_ipopt.py` default:** `LAM_CLEAR=0 IP_HIST=60
   IP_ACC_TOL=5 IP_ACC_ITER=8` holds kept at 88%/88%, agreement at 100% and all
   40 poses bitwise identical, while cap hits fall 70/80 → 42/80 with 34 clean
   acceptable exits. The remaining 42 are unexplained: the same setting zeroed cap
   hits under defaults, so something specific to the no-bonus arm blocks
   acceptance. Complementarity at exit was 8.9e-3 / 1.0e-1 / 1.6e-3 there against
   ~1e-9 under defaults. `acceptable_compl_inf_tol` defaults to 0.01, so one of
   those three exceeds it tenfold and a second sits at 89% of it, while defaults
   run three orders inside. Testing it needs a new env knob; not done.

   `tol` is exposed as `IP_TOL` and defaults to 1e-4. The one FCL failure in the
   probe came from `tol=1e-4` *without* the acceptable exit; combined with it the
   arm was 6/6 clear. On top of `IP_ACC_TOL=5` the change is close to inert —
   median iterations 100 against 104 — because the acceptable exit fires first and
   `tol` is no longer the binding criterion. It rests on six candidates and wants
   the 40-candidate pair. Not adopted: `acceptable_obj_change_tol=1e-5`, which
   cannot stop these solves because the relative objective keeps moving by more
   than that. `acceptable_tol=1e-3` was
   the originally planned value and never fires — the Overall NLP error equals the
   dual infeasibility, measured at 0.64–3.05. Keep `acceptable_constr_viol_tol` at
   1e-4: loosening the dual tolerance under L-BFGS is documented practice,
   loosening the feasibility tolerance would discard the real gate. The `resto.`
   prefix matters, since prefixed lookups fall back to the unprefixed value and a
   global loosening would also relax the branch that reports local infeasibility.
3. **Match `tol` to the arithmetic. DONE 2026-09-16** — `IP_TOL` exposes `tol`
   and defaults to 1e-4. Derivatives are float32 (epsilon 1.2e-7) over bfloat16
   grids, so IPOPT's 1e-6 default asked for roughly one digit more precision than
   the gradients carry, putting the threshold at their noise floor. The change is
   close to inert on top of `IP_ACC_TOL=5` because the acceptable exit fires first
   and `tol` no longer binds; it can only matter where neither exit fires, which
   is the remaining cap-hitters. It rests on six candidates and wants the
   40-candidate pair. The alternative — evaluating objective and Jacobian in
   float64 — is untested and would cost GPU throughput.
4. **`linear_system_scaling="slack-based"`, `linear_scaling_on_demand="no"`.**
   Verified available without HSL. It scales the slack block of the augmented
   system, where an all-inequality problem degenerates as slacks approach their
   bounds. Expect a modest effect: published reliability across 386 CUTEr
   problems is 92.8% unscaled against 92.0% with MC19.
5. **Two-sided user scaling.** `nlp_scaling_method="user-scaling"` with
   `set_problem_scaling(obj_scaling, x_scaling, g_scaling)`, injected in
   `minimize_ipopt_logged` between the option loop and `solve`
   (`pipeline/phase2_diagnostics.py`). This is the only route to variable
   scaling — every automatic method sets `dx = NULL`. `minimize_ipopt` exposes no
   hook, so the non-diagnostic path needs the same treatment. Unsettled: whether
   to scale through IPOPT or by an affine change of variables in the model, since
   IPOPT relaxes the unscaled bounds before applying scaling.
6. **Reparameterize spin as an angle.** The largest change and the only one with
   a mechanism rather than a correlation behind it: `(sx, sy)` reaches the model
   only through `arctan2`, making the radial direction an exact null direction of
   all 402 constraint rows. Removes the penalty, its weight, the origin
   singularity inside the bounds, and 7 variables.

Independent of the ordering, and cheap:

- `lambda_unit_circle` is missing from `_weights_key` in `objectives/phase2.py`,
  so it never invalidates `_JIT_CACHE`. Any past tuning of it on a cache hit was
  a no-op.
- The `_padding_mask` docstring claims the padded rows leave the KKT matrix
  rank-deficient. They do not — each inequality carries its own slack, so the row
  reads `[0 … 0 | −1]`. Their cost is size. Fix the docstring.
- Run `OBB_GAIN=1` with `IP_CVTOL=1e-6` if the OBB-gain arms matter: because
  `constr_viol_tol` is absolute on the gain-carrying constraint, every gain arm
  already run changed the physical tolerance and the row scaling together.

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
  config, `rutter-plan` CLI (tyro + startup plan-apply), spin orbit-basis refactor,
  live pipeline scripts tracked, 73 stale diagnostics deleted, `dev/PIPELINE.md`.
- `CLAUDE.md` is gitignored in this repo (edits are local-only).
- Live scripts tracked; active spin/ADAM/enumerator exploration scripts kept;
  experiment-output configs + `.claude/` gitignored.
