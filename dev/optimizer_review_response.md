# Response to architectural review

This file responds to the staged-optimization review of `aind-low-point`'s
placement optimizer. The review is well-reasoned and directionally
correct, but some of its critique applies to a previous version of the
code — several of the recommended changes are already implemented. This
document reconciles the review against the current state, summarises what
today's diagnostic work found, and lays out the highest-leverage next
moves.

The code references below are all on branch `finish-refactor` at HEAD
(or close to it — pull and re-check `git log` if anything looks off).

---

## TL;DR

- The review's critique of "soft-penalty SLSQP" and "single-pose hard
  reject" applies to a prior version. Native `ineq` constraints, a
  feasibility-first inner solve, multi-pose feasibility scoring,
  target-anchored coverage in the LSAP cost, lex-ranked reporting with
  feasibility-threshold ε, and infeasibility certificates are all
  already implemented. References below.
- The architectural critique that still applies — and that today's
  diagnostic confirms — is that **the discrete stages (LSAP + arc
  partition) score per-pair / per-probe, not jointly**. A known-feasible
  manual plan (which the inner solve converges to with coverage *higher*
  than manual when warm-started from it) is below rank 50 in both
  discrete stages. The fix is the review's "Stage 2: joint (H, A)
  reranking with arc-aware pairwise terms."
- Two orthogonal model-fidelity gaps the review doesn't address:
  - the optimizer treats subject anatomical LPS as the rig coordinate
    frame (no `subject_to_rig` rotation captures the mounted head
    pitch);
  - the headstage clearance constraint compares 2 mm-radius placeholder
    capsules, not the actual probe meshes.

---

## State of the implementation

Layout:

```
src/aind_low_point/optimization/
├── geometry.py          ← Capsule, capsule_capsule_dist, HoleSection, shaft_section_oval_value
├── holes.py             ← Hole, load_holes
├── recording.py         ← RecordingGeometry, RECORDING_GEOMETRY (per kind)
├── density.py           ← gaussian_density, coverage, Simpson integration
├── kinematics.py        ← pose_from_optimizer_vars, pose_at_hole_best_fit,
│                          shank_capsules_from_pose, required_ap_deg
├── hole_assignment.py   ← AssignmentProbe, CostWeights, build_cost_matrix,
│                          _build_pose_bank, multi_pose_evaluate,
│                          solve_top_k_assignments (Murty's k-best)
├── arc_assignment.py    ← required_aps_deg_for_assignment, enumerate_partitions,
│                          solve_top_k_arc_assignments, project_centroids_min_sep
├── objective.py         ← OptimizerContext, ProbeContext, VariableLayout,
│                          evaluate_probe, evaluate_objective, evaluate_constraints,
│                          coverage_objective, feasibility_violation_squared
└── optimize.py          ← 3-level driver: optimize(), polish_seed(),
                           best_fit_hole_id_at_pose,
                           _inner_solve_one, _feasibility_solve,
                           _slsqp_polish_constrained, _multistage_cma_es
```

Driver runs:

```
1. solve_top_k_assignments(probes, holes, k_holes)              ← LSAP + Murty
       ↓ for each hole assignment ha:
2. solve_top_k_arc_assignments(ha.probe_to_hole, holes, k_arcs) ← brute-force enumerate partitions
       ↓ for each (ha, aa):
3. _inner_solve_one:
     CMA-ES multi-stage homotopy (penalty multipliers 0.1, 1.0, 10.0)
     Stage A: _feasibility_solve     (min Σ ReLU(-slack_j)² over all groups)
     Stage B: _slsqp_polish_constrained  (max coverage s.t. native ineq slack ≥ 0)
     Stage C: _feasibility_solve again, keep whichever lex-better
       ↓
4. PlanCandidate per (ha, aa) with feasibility / coverage / min-separations
       ↓
5. Sort by PlanCandidate.lex_key(feasibility_threshold)
6. Return OptimizationResult.best + alternatives = full ranked tuple
```

---

## Review claims vs current implementation

### Already implemented (review's "incremental implementation plan" items)

| # | Recommendation | Status / where |
|---|---|---|
| 1 | "Add hard inequality constraints to SLSQP" | **Done**. `optimize._slsqp_polish_constrained` uses native scipy `ineq` constraints; `slsqp_constrained=True` is default; soft penalties opt-in via `--slsqp-soft` |
| 2 | "Add a feasibility-only continuous solve before coverage optimization" | **Done**. `optimize._feasibility_solve` minimises `feasibility_violation_squared` as Stage A, before Stage B coverage polish |
| 4 | "Add approximate coverage to LSAP cost" | **Done**. `CostWeights.delta_coverage = 5.0`, dominant term. After 2026-05-14 bank re-anchoring (see below), coverage is computed at target-anchored poses so it's meaningful per-pair, not zero for off-axis targets |
| 5 | "Replace one-pose static threading hard reject with a small pose-feature bank" | **Done**. `hole_assignment._build_pose_bank` + `multi_pose_evaluate`; hard reject is `min_violation_sq > 1.0` across the bank, not single-pose |
| 8 | "Return the top 3-5 distinct plans rather than only the lowest-cost plan" | **Done**. `OptimizationResult.alternatives`; `format_plan_table` Markdown table; `PlanCandidate.lex_key(ε)`; infeasibility certificate printed when 0/N feasible |
| 9 | "Add infeasibility diagnostics" | **Partial**. `PlanCandidate` carries `max_violation_threading / clearance / arc_ap_sep / intra_arc_ml_sep` + `dominant_violation_group`. Missing: per-probe, per-section identification of the limiting constraint |
| 11 | "Move from SLSQP to trust-constr" | **Partial**. `_slsqp_polish_constrained(method="trust-constr")` is wired; `--polish-method trust-constr` flag exists. Finite-diff gradients only — full JAX migration still pending |

### Other current features the review doesn't mention but are worth knowing

- **Multi-stage CMA-ES homotopy** (`_multistage_cma_es`) with penalty multipliers `(0.1, 1.0, 10.0)` across stages, sigma halving between stages. Lets early generations explore broadly, later generations enforce feasibility.
- **Two-stage inner solve (Stage A + Stage B)** with empirical win on 836656 / 7 probes: cost 1.70M → 0.86M; threading 519K → 101K; coverage 0.72 → 3.88.
- **Stage C feasibility cleanup** (`final_feasibility_cleanup`, default on): after Stage B, re-runs `_feasibility_solve` from Stage B's output and keeps the lex-better candidate. Empirically pulls SLSQP best max_viol from 0.73 → 0.10 on 836656.
- **Feasibility-threshold lex ranking** (`PlanCandidate.lex_key(feasibility_threshold)`): plans with `max_viol ≤ ε` collapse to the same first-tier rank so coverage decides among "feasible enough" plans. Default ε=0 (strict feasibility-first); CLI flag `--feasibility-threshold` exposes it.
- **Tolerance knobs**: `threading_oval_tolerance` (allow `g_thread ≤ tol` instead of `≤ 0`) and `clearance_overlap_allowance_mm` (allow headstage capsules to overlap up to N mm). For modelling-vs-reality slop, not as a substitute for the constraint structure.
- **Soft arc AP-separation in middle layer**: `enumerate_partitions` no longer hard-filters centroids closer than 16°; scores them with `Σ_{i<j} max(0, 16 − Δc_ij)²` weighted by `arc_sep_shortfall_weight` (default 10.0). Replaced the legacy hard gate which collapsed 7-probe / 3-arc problems too aggressively.
- **PAV centroid projection** in `project_centroids_min_sep`: enforces the chained ≥16° AP-min-sep on every enumerated partition's centroids, then ranks by `Σ (required_ap_p − projected_centroid_p)²` ("tilt cost") rather than raw within-cluster variance.

### 2026-05-14 changes (today)

Driven by the seed-polish diagnostic (see next section). Two LSAP changes:

- **`_build_pose_bank` re-anchored on target**. The bank previously placed pose_tip such that the shank-row centroid sat at the slot bottom — fine for threading, but for off-axis targets the recording bank floated up at the slot rather than reaching the target Gaussian, so `max_coverage` was zero. Bank now uses `pose_tip = target_LPS − R @ pivot_local` (same anchoring as `pose_from_optimizer_vars` in the inner loop), so the recording bank center lands on the target.

  Also dropped slot-aligned and halfway poses from the bank. New 5-pose set: target-aligned + 4 perturbations (±2° AP, ±2° ML around target-aligned).

  Threading values are unchanged by the re-anchoring — the shift is along `R[:, 2]` (the shaft direction), and section `(u, v)` projections of the shaft line are invariant under shifts parallel to the shaft.

- **`CostWeights.alpha_target_angle: 1.0 → 0.0`**. The bore-vs-target angle was the wrong axis to penalise: once the bank scores feasibility at *target-oriented* poses, an off-bore angle is double-counted (the probe can tilt to reach the target; the bank captures that directly). A proper "extreme rig-frame angle" penalty would be useful, but it needs the rig vs subject anatomical convention first (see Open Questions).

After today's changes, per-probe LSAP cost is dominated by `−δ·max_coverage` (≈ −21 for deep targets, ≈ −6 for shallow), with `β·max_g` as the tiebreaker and `γ·interference` as a row-wide static penalty. Coverage is nearly constant per probe across hole choices (every hole CAN reach every probe's target at a target-aligned pose, modulo threading feasibility), so the LSAP cost is now effectively threading-clearance-ordered.

---

## Diagnostic findings (2026-05-14)

Two new scripts:

- `scripts/score_manual_plan.py` — scores a manually-authored plan against
  the optimizer's constraint model. Reports the same `PlanCandidate`
  metrics the optimizer reports for its own candidates.
- `scripts/diagnose_search.py` — for a given (config, holes, plan)
  triple, reports where the seed plan's hole assignment ranks in the
  LSAP top-K, and where the seed plan's arc partition ranks in
  `solve_top_k_arc_assignments`.

Subject: `examples/836656-config-T12.yml` + `examples/836656-config-T12.plan.yml` (a known-feasible 7-probe insertion authored by the user).

### Inner solve from manual seed: healthy

Added `polish_seed` to `optimize.py` and `--seed-plan PATH` to
`scripts/run_optimizer.py`. Skips LSAP + arc partition entirely; runs
just the inner solve from the manual plan's (probe→hole auto-detected
at the seed pose, probe→arc, x0 = manual plan's per-probe variables).

At `clearance_overlap_allowance_mm=2.1`:

```
                     Manual seed   After polish (no CMA)
feasible (strict)         no            yes
max_violation             —             0.0
coverage                  14.96         17.91
arc AP centroids          (-43,-10,+13) (-43,-11,+14)
arc-AP span               56°           56°
per-probe (ml,spin,off)   manual        ≈ manual (no drift)
past_target_mm            small neg.    ≈ 0
Stage A violation²        0 → 0         (already feasible)
```

The polish from manual seed lands on a strictly-feasible plan with
*higher coverage than manual*. Per-probe ml/spin/offsets barely move;
the only systematic change is past_target_mm sliding to ≈0 (the polish
prefers to center the recording bank on the target centroid, which the
manual workflow deliberately under-inserts from — a coverage-objective
question, not an optimizer correctness issue).

**Conclusion: the inner continuous solve is correct.** Given the manual
plan's (hole, arc, x0), it finds a strictly-feasible-and-better point.
The 6× coverage gap between full-optimizer (cov 2.65) and manual (cov
14.96) is therefore entirely a discrete-search failure: the inner solve
never gets a warm start close to the manual basin.

### LSAP: seed pattern not in top-50

After today's bank re-anchoring and α=0:

- Seed total LSAP cost: **−86.52**
- LSAP top-1: **−86.71**
- Gap: **0.19** (0.2% of `|cost|`)
- Seed assignment rank in Murty top-50: **NOT FOUND**

The seed pattern is essentially tied with top-1 by LSAP score, but
Murty enumerates 50 near-identical perturbations of the top-1 (e.g.
shuffling RSP among {4, 5, 9}, BLA among {12, 13}) before reaching it.
Wider Murty enumeration alone doesn't help — the gap isn't depth but
direction: the seed is a *different joint structure* whose value the
per-pair LSAP cost can't see.

Specifically: the seed's BLA assignment is hole **4** (right-column
posterior). At the target-aligned pose, BLA@4 has `max_g = +0.00` —
literally grazing the slot wall — while BLA@12 (left-column anterior)
has `max_g = −0.22`. The LSAP correctly downgrades BLA@4 on threading.
But the manual plan uses BLA@4 anyway, because the joint plan (BLA on
arc b with PL, CA1 at slider_ml=+27°, off-axis tilt) makes it work.
The LSAP cost can't see "BLA@4 works *if* the arc-AP and ML choices
support it."

### Arc partitioner: seed partition not in top-50

The partitioner's score is `Σ_p (required_ap_p − projected_centroid_p)²`
("tilt cost") plus the AP-min-sep shortfall. Manual seed:

```
probe  required_AP   seed_arc_AP   tilt
  PL     -17.33°        -10.00°    +7.33°
  MD      -7.06°        +13.00°   +20.06°
 CLA      -4.75°        -43.00°   -38.25°
 BLA      +0.94°        -10.00°   -10.94°
 RSP      +6.61°        -43.00°   -49.61°
 VM      +11.79°        -43.00°   -54.79°
 CA1     +12.69°        -10.00°   -22.69°

Σ tilt² ≈ 8068
```

Top-1 partition: centroids `(-19.5, -3.5, +12.5)` — 32° AP span, tilt²
≈ 167. **50× smaller.** Seed partition rank in top-50: NOT FOUND.

The partitioner's objective directly opposes the manual strategy: the
manual spreads arcs wide (56° span) so 4 probes can share each arc with
≥16° ML separation; the partitioner clusters arcs around the
required-AP centroids and treats wide tilts as expensive.

### Implication

Both discrete layers' costs are **per-pair / per-probe**. Neither can
express joint multi-probe feasibility constraints (shared-AP coupling
across same-arc probes; ML-separation feasibility within an arc; AP
separation across arcs; pairwise headstage clearance at specific
hole-arc combinations). The inner solve has all of these — but only
sees the candidates the discrete layers feed it.

---

## Where the review's recommendations stand

In the order the review proposes them, mapped to current state:

### Stage 0: probe-hole pose-feature bank — partial

Conceptually present inside `multi_pose_evaluate`, which evaluates a
5-pose bank per (probe, hole) and returns `(min_violation_sq,
min_max_g, max_coverage)`. The review's richer features ("preferred AP
angle or interval, plausible depth range, headstage proxy, quadbase
fit score, pose robustness") aren't precomputed; they're recomputed
ad-hoc.

**Worth lifting** into a `precompute_pose_features(probes, holes) →
Dict[(probe_name, hole_id), PoseFeatures]` table if and when the joint
reranker (Stage 2) needs them. Doing it before Stage 2 isn't required.

### Stage 1: broad probe-hole candidate generation — partial

LSAP + Murty exists (`solve_top_k_assignments`). Cost matrix uses
target-anchored coverage, threading clearance, multi-pose feasibility
slack, and a row-wide interference heuristic. Defaults: `k_holes=5`.

The review suggests `k_holes = 50 to 500`. Today's diagnostic argues
against the unconditional widening: the LSAP top-1 to top-50 are
near-tied (cost spread ≈ 0.5 / |cost| 86), so widening just enumerates
permutations of the same joint structure. Widening becomes useful
*paired with* Stage 2 reranking, when each LSAP candidate can be
re-scored on different joint dimensions.

### Stage 2: joint (H, A) discrete scoring — **NOT in current code**

This is the critical missing piece. Proposed by the review and
confirmed by today's diagnostic.

Sketch:

```
For each top-K_h LSAP hole assignment H:
  For each enumerated arc partition A in solve_top_k_arc_assignments(H, ..., k=K_a):
    score(H, A) =
        sum_i  C_per_pair[i, H[i]]                          # existing LSAP cost
      + sum_{i<j, A[i] == A[j]}  Q_same[i, j, H[i], H[j]]   # same-arc pairwise
      + sum_{i<j, A[i] != A[j]}  Q_diff[i, j, H[i], H[j], A[i], A[j]]   # cross-arc pairwise
      + arc_capacity_penalty(A)
      + arc_count_penalty(A)
```

Same-arc pairwise terms (`Q_same`) need to capture:
- AP feasibility of sharing one rig AP — the intersection of each
  probe's tolerable AP interval given its bore and the rig limits.
- ML separation feasibility — `≥16°` between any two same-arc probes.
- Headstage clearance at the joint pose (capsule-vs-capsule signed
  distance, using existing primitives).

Cross-arc pairwise terms (`Q_diff`) need:
- `≥16°` AP separation between arc centroids.
- Headstage clearance between probes on different arcs.

These map onto the existing primitives:
- `pairwise_headstage_clearances` in `objective.py` (capsule-capsule).
- `kinematic_separations` in `objective.py` (AP and ML pair seps).
- Plus a new "shared-AP feasibility" surrogate — e.g. intersect each
  same-arc probe's `[required_AP - slack, required_AP + slack]`
  intervals; non-empty intersection means there's an AP at which all
  probes on the arc could thread.

All of these are static per (H, A) candidate, so the joint score is
~`O(K_h · K_a · K²)` evaluations where K = num_probes. For K_h=50,
K_a=10, K²=49: ~25,000 pairwise evaluations per run. Each is a few
ms — total ≈ a minute. Cheap compared to the inner solve.

**This is the recommended next architectural addition.** It would
directly close the diagnosed seed-not-in-top-50 gap.

### Stage 2.5: cheap continuous arc-feasibility screening — would be useful with Stage 2

Reduce variables to `(arc AP, probe ML)` only; hold spin / offsets /
depth at warm-start; minimise `threading + kinematic + rough clearance
violation`. Hard-reject candidates whose violation can't be driven near
zero.

Only worth implementing if Stage 2 widens the candidate pool enough
that running the full inner solve on every candidate is too expensive.
At current `K_h=5, K_a=3` (~15 inner solves at ~25s = ~6 min) there's
no need; if Stage 2 grows the pool to 50+ candidates, Stage 2.5 is
worth ~1-2s per candidate to filter.

### Stage 3: full continuous feasibility projection — done

Maps onto current `_inner_solve_one`. Stage A `_feasibility_solve`
minimises `feasibility_violation_squared`; Stage B
`_slsqp_polish_constrained` runs hard-constrained coverage maximisation;
Stage C re-projects if Stage B drifted.

### Stage 4: hard-constrained coverage polish — done

Same as above; `_slsqp_polish_constrained` is the implementation.

### Stage 5: ranked candidate reporting and infeasibility diagnostics — partial

`format_plan_table` + `PlanCandidate.lex_key(ε)` + dominant-group
diagnostics + infeasibility certificate ("0/15 candidates feasible —
best is INFEASIBLE") are in.

Missing: per-probe / per-section identification of the limiting
constraint. The data is available in `evaluate_constraints` — would
need `PlanCandidate` to carry the argmax indices alongside the magnitudes.

### Gradient interface — partial

`coverage_objective(x, ctx)` and `evaluate_constraints(x, ctx)` are
already separable. SLSQP gets them as separate `fun` and `constraints`
arguments today; finite-diff Jacobians.

JAX migration is on the active-next list in `dev/optimizer_plan.md`.
Highest payoff once `evaluate_objective` and `evaluate_constraints` are
JAX-traceable: `trust-constr` with exact Jacobians, which interior-points
better through tight feasibility tubes than SLSQP's active-set.

---

## Open issues the review doesn't address

These are real and orthogonal to the discrete-search critique.

### Subject-to-rig coordinate frame

`arc_angles_to_affine(0, 0, 0) = identity` in LPS — i.e., the pipeline
treats "rig vertical pose" as "probe vertical in subject anatomical
LPS". `required_ap_deg(hole.axis_LPS)` projects onto LPS (y, z) — the
subject's AP plane, which the code implicitly identifies with the
rig's AP rotation axis.

If the mouse's head is mounted at a non-trivial pitch / roll / yaw
relative to rig vertical (the user reports a 14° head pitch about
the R-axis, with unclear sign convention), that rotation isn't in
the code. The optimizer's "AP=0" then doesn't match the rig's
mechanical "AP=0", and `required_ap_deg` is reporting subject-frame
angles, not rig-frame ones.

To fix:
1. Add `subject_to_rig: AffineTransform` to `Kinematics` (or wherever
   feels natural). Defaults to identity.
2. Replace `required_ap_deg(hole_axis_LPS)` with
   `required_ap_ml_rig(hole_axis_LPS, R_subject_to_rig) → (ap_rig, ml_rig)`
   that transforms the bore axis into rig coords first, then projects.
3. The discrete reranker (Stage 2) and the inner-solve bounds
   (`_default_bounds`) consume rig-frame angles.

This is a pre-req for any "extreme angle" penalty to mean what the
review intends. Without it, "the rig has to set AP to 50°" can't be
distinguished from "the subject's anatomical AP at this bore is 50°."

### Headstage geometry

`objective.headstage_capsule()` builds a generic 2 mm-radius × 5 mm-long
capsule positioned 10 mm above the probe tip — *identical for every
probe kind*. The code comment explicitly flags it as a placeholder.
At the manual T12 plan, the headstage-capsule-vs-headstage-capsule
clearance reports `−2.04 mm` (collision); the actual probe meshes
don't touch.

Fix: at runtime build, derive a per-kind bounding capsule (or set of
capsules, or OBB) from the canonicalised probe mesh's vertices above
the recording-array top. Cache per kind. Use in
`objective.headstage_capsule()`.

The review's `Q_same` / `Q_diff` headstage-clearance terms would
inherit the same model fidelity issue until this is fixed.

### Coverage objective shape

`gaussian_density(target_LPS, sigma=0.5mm)` is the same density per
probe. In production usage, targets are tracer-label centroids and the
real density volumes are noisier than a Gaussian, often spilling into
adjacent regions or white matter. A masked-by-brain-region voxel
density would be more faithful. Mentioned in
`coverage_retro_target_masking.md` memory; not a blocker for the
search-quality fix, but worth knowing.

### ML bound asymmetry

`_default_bounds` clamps `ml_local` to `(-30, +30)`, while
`PoseLimits.ml_deg` is `(-60, +60)`. The manual T12 plan sits at
`PL.ml = -30°` (right at our bound) and `BLA.ml = +27°` (near it). So
the optimizer is artificially constraining ML tighter than the rig.
Two-line fix in `optimize._default_bounds`.

---

## Recommended next moves

In rough order of leverage relative to current state:

1. **Settle the subject-to-rig convention** (head tilt). Smallest code
   change, but a pre-req for steps 2 and 3 to be physically meaningful.

2. **Joint (H, A) reranking with arc-aware pairwise terms** (review
   Stage 2). Architecturally the biggest move, but contained — a new
   module that sits between `solve_top_k_assignments` and
   `solve_top_k_arc_assignments`. Closes the diagnosed seed-not-in-top-50.

3. **Per-kind headstage capsule from mesh**. Independent of the
   discrete-search work but blocks "is this plan actually
   collision-free?". Without it the reranker's headstage-clearance
   terms inherit the bad capsule.

4. **Widen LSAP and add cheap continuous screening** (review Stages 1,
   2.5). Only pays off after step 2.

5. **Per-probe / per-section infeasibility diagnostics**. Small change,
   ergonomic win, useful for any debugging from here on.

6. **JAX + `trust-constr`**. Already on the roadmap. Best landed after
   the discrete side is stable so we don't churn the objective surface
   while moving to autodiff.

Items 2, 3 are roughly independent and can ship in parallel. Item 1
gates the rig-frame angle treatment that items 2 needs in its
`Q_same` / `Q_diff` AP-feasibility terms.

---

## Inputs to verify if continuing this work

- Branch: `finish-refactor` (not yet merged to main).
- Subject config: `examples/836656-config-T12.yml`.
- Subject plan: `examples/836656-config-T12.plan.yml` (manual, known
  feasible, gold standard for tuning).
- Hole extraction: `/tmp/836656-holes.yml` (regenerable via
  `scripts/extract_implant_holes.py /mnt/vast/scratch/.../0283-300-04.obj`).
- Score the manual plan:
  `uv run --python 3.13 python scripts/score_manual_plan.py examples/836656-config-T12.yml /tmp/836656-holes.yml --plan examples/836656-config-T12.plan.yml --clearance-overlap-allowance-mm 2.1`
- Diagnose discrete-search ranking:
  `uv run --python 3.13 python scripts/diagnose_search.py examples/836656-config-T12.yml /tmp/836656-holes.yml --plan examples/836656-config-T12.plan.yml --k-holes 50 --k-arcs 50 --max-num-arcs 3 --min-num-arcs 3`
- Polish from manual seed:
  `uv run --python 3.13 python scripts/run_optimizer.py examples/836656-config-T12.yml /tmp/836656-holes.yml --seed-plan examples/836656-config-T12.plan.yml --clearance-overlap-allowance-mm 2.1`
- Run full optimizer (now with target-anchored LSAP bank):
  `uv run --python 3.13 python scripts/run_optimizer.py examples/836656-config-T12.yml /tmp/836656-holes.yml --max-num-arcs 3 --min-num-arcs 3 --clearance-overlap-allowance-mm 2.1`
