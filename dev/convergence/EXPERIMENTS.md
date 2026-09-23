# Convergence experiments — results

One entry per measurement, appended, never rewritten. An entry records what was
asked, how it was measured, what came back, and what may not be concluded from
it. Corrections are new entries that cite the old one, so a wrong reading stays
visible next to its retraction, and the corrected entry gets a one-line banner
pointing forward — the reader who stops at the old entry is the one being
protected.

**State a statistic's scope before it carries an argument.** Every wrong reading
in this file came from the same move: a recorded field used without checking what
it measures *over*.

- **Final iterate vs trajectory.** "Infeasible at the cap" describes the last
  iterate, not the solve. The capped solves reached feasibility at iteration 24
  and held runs of 60 (X25 → X26).
- **Trailing window vs total.** "The trailing feasible run is 8" is not "the
  solve is feasible for 8 iterations". The total was 245 of 570 (X25 → X27).
- **Scaled vs unscaled.** `diag_hist`'s `inf_pr`/`inf_du` are IPOPT's *scaled*
  iteration-log values and `mu` is the barrier parameter, not complementarity.
  Three of the acceptable test's five legs bound unscaled quantities, so none
  of those three may be compared to a `diag_hist` field; `acceptable_tol` is the
  exception and *is* scaled. `group_min`/`group_nviol` are the only unscaled
  records (X25, narrowed by X28).
- **Unresolvable vs null.** A metric the prior art calls unresolvable licenses
  "unresolved", never "no effect" (X19 → X26).
- **Raw counts vs rates.** A count that grows with solve length (crossings,
  iterations spent anywhere) must be normalised before two arms are compared on
  it (X27).

So: name the axis a figure runs over, and say whether it is scaled, in the
sentence that first uses it.

Prior art lives in `dev/PHASE2_CONDITIONING.md` (measured state of the Phase-2
solve as of 2026-09-16: build capabilities, constraint scaling, why solves exit
locally infeasible, the tolerance settings, formulation defects). Read it before
adding an entry — several questions below are refinements of its findings, and
an entry that merely re-measures one of them should say so.

`dev/convergence/PLAYBOOK.md` gives the item IDs (`D1`, `I2`, `F1` …) used here.

---

## X1 — Threading reward off vs float32 grid blend, 837772 capped-10

**Date:** 2026-09-22 · **Item:** D1 · **Status:** provisional, control missing

**Question.** Of the two interventions that were meant to address non-smoothness,
which changes Phase-2 behaviour, and in which direction?

**Method.** Ten candidates from `scratch/estimators/837772-config/capped10_ranks.txt`,
chosen because they capped at 1000 iterations in production. Three CPU arms, three
workers each, 1000-iteration cap: `base` (production), `nothread`
(`RUTTER_LAM_THREAD=0`, threading reward removed from the objective entirely) and
`f32` (`RUTTER_GRID_DTYPE=float32`, grid storage and blend in float32 instead of
bf16). Compared with `scratch/estimators/compare_arms.py`.

**Result.**

| arm | exits | med it | med inf_du | med mu end | kept | med obj | med s |
|---|---|---|---|---|---|---|---|
| `nothread` | acceptable 6, infeas 3, cap 1 | 547 | 0.344 | 0.0014 | 4/10 | −5.355 | 2246 |
| `f32` | acceptable 7, cap 3 | 390 | 1.53 | 0.004 | 9/10 | −5.771 | 1518 |

Per candidate, `f32` returns a positive FCL margin on 9 of 10; `nothread` returns
`−1.000` (FCL reject) on 6 of 10. `f32` produces no locally-infeasible exits;
`nothread` produces three.

**Reading.** `f32` is better on every outcome that matters — kept plans, objective,
wall clock, iterations, no infeasible exits — while being **worse** on `inf_du`,
the stationarity measure whose stalling started this investigation. The two
measures point in opposite directions.

If that survives the control, it is the most consequential finding so far: it
says `inf_du` is not tracking solution quality here, and that tuning the solver
to drive it down may buy nothing. It is the direct motivation for D2, which asks
which half of IPOPT's `acceptable` test is actually failing.

**Do not conclude.**
- Nothing about `base`. The control had to be restarted (see STATUS) and is not
  finished, so neither arm has been compared against production.
- Nothing comparing the two arms *to each other* as a pair. They differ in two
  unrelated ways; each is meaningful only against `base`.
- Nothing about the threading reward's value. `nothread` deletes the term, so its
  objective is not on the same scale as the others', and `med obj` across arms is
  not a like-for-like comparison.
- Ten candidates, all selected for capping. This is the hard tail, not the
  population.

---

## X2 — Which leg of the `acceptable` test fails, and which constraint group holds it

**Date:** 2026-09-22 · **Item:** D2 · **Status:** solid on the logged legs; fires the playbook gate

**Question.** Of IPOPT's `acceptable` criteria, which one do the capping solves
fail — stationarity (`inf_du ≤ ip_acc_tol = 5.0`) or feasibility
(`inf_pr ≤ ip_cvtol = 1e-4`), for `ip_acc_iter = 8` consecutive iterations?

**Method.** No replay was needed: the arms already run under
`minimize_ipopt_logged`, so every candidate carries a per-iteration
`inf_pr`/`inf_du`/`mu`/`alpha_pr`/`ls_trials`/`restoration` trace plus
`group_min`/`group_nviol` over the six `SLACK_GROUPS`. Analysed with
`scratch/estimators/acceptable_forensics.py` over the last 200 iterations.

**Result.** Across all four capped candidates in the two finished arms:

| arm | tail fraction meeting dual leg | tail fraction meeting feasibility leg | longest run meeting both |
|---|---|---|---|
| `nothread` | 0.60 | **0.00** | 0 |
| `f32` | 0.90 | **0.00** | 0 |

Eight consecutive iterations are required; the longest run achieved is zero.
`inf_pr` never once drops under `1e-4` in the tail of any capped solve.

The line search is healthy throughout: median `alpha_pr` 0.19–0.5, median
`ls_trials` 1–3, restoration entered on 0–2% of iterations.

End-state worst slack per group says which constraint is responsible. In `f32`
every geometric group finishes with a comfortable positive margin — thread +0.08
to +0.29, probe_pair +0.008 to +0.32, probe_fixture +0.24 to +0.59, brain +0.005
to +0.69 — while `ml_sep` is the only group ever negative, and the only one
violated worse than `-1e-4`. It is negative in **every capped candidate of both
arms** and in no candidate that exited acceptable.

**Reading.**

1. **Feasibility, not stationarity, is what caps these solves.** The dual leg is
   satisfied most of the time; the feasibility leg is never satisfied.
2. **Nonsmoothness is not preventing progress.** Healthy step lengths, few
   line-search trials and almost no restoration are not the signature of a solver
   fighting kinks. This substantially weakens the premise the F1/C2 work rests on.
3. **One constraint group is doing all of it.** `ml_sep` — intra-arc ML separation,
   `min_intra_arc_ml_sep_deg = 16.0` at `constrained.py:116`, built at
   `constrained.py:590` as `ml_diff - min_intra_ml` — is the only group that ever
   ends violated. The solve satisfies every clearance constraint with room to
   spare and then spends a thousand iterations failing to close an `ml_sep`
   residual of 0.02–0.03.
4. It also explains X1 without appeal to smoothness: `f32` keeps more plans
   because it ends with far larger geometric margins, not because it converges
   better.

**Not established — the mechanism.** Three candidate explanations, untested:
`ml_sep` rows are in **degrees** while thread/pair/fixture/brain rows are in
**millimetres**, and one `constr_viol_tol` covers all of them; or the 16°
separation is genuinely unachievable for these crowded candidates and IPOPT is
grinding at a gap it cannot close; or that row is badly conditioned. Note the same
candidate (1567) ends at `-5.7e-4` in `nothread` and `-0.023` in `f32`, so it is
not a fixed geometric infeasibility. Deciding between these is the next
measurement, and `dev/PHASE2_CONDITIONING.md`'s constraint-scaling section plus
`scratch/estimators/conditioning.py` are where it starts.

**Also noticed.** `arc_sep` reports `+nan` worst slack in all 20 candidate-runs —
the group appears to be empty or entirely padded. Unrelated to the above, but it
means one of the six constraint groups is not constraining anything.

**Do not conclude.** Four capped candidates, from a set hand-picked for capping.
The complementarity leg of the `acceptable` test is not logged, so "the dual leg
held" is necessary but not sufficient. Nothing here yet touches `base`.

---

## X5 — `inf_pr` cannot reach its tolerance, and disagrees with the model

**Date:** 2026-09-22 · **Item:** D5 · **Status:** solid; this is the mechanism

**Question.** X2 showed the feasibility leg never holds. Is it narrowly missed, or
unreachable — and is `inf_pr` even measuring the constraint violation we think?

**Method.** Over the full 1000-iteration trace of each capped candidate: the
minimum `inf_pr` ever reached, and how many iterations the model itself calls
feasible (`group_nviol` zero across all six groups).

**Result.**

| arm | cand | min `inf_pr` over the run | iterations model-feasible |
|---|---|---|---|
| `f32` | 11367 | 0.0283 (it 327) | 334 / 1001 |
| `f32` | 1567 | 0.0690 (it 907) | 131 / 1002 |
| `f32` | 2140 | 0.0112 (it 89) | 416 / 1003 |
| `nothread` | 1567 | 0.0104 (it 975) | 148 / 1003 |

**Reading.** Two things, and the second is the important one.

1. **The feasibility leg is unreachable, not narrowly missed.** `ip_cvtol` is
   `1e-4`; the best `inf_pr` any capped solve ever achieves is `0.0104`. That is
   100–700× away. No number of extra iterations closes it, which is why the cap is
   hit rather than the acceptable test firing.
2. **`inf_pr` and the model disagree.** A third of iterations satisfy every
   constraint group — `group_nviol` is zero — while `inf_pr` at those same
   iterations is still ≥ 0.01. Whatever IPOPT is measuring is not the constraint
   violation the model reports, and it is roughly two orders of magnitude larger.

That points at constraint scaling: with `nlp_scaling_method` at its
gradient-based default, rows are rescaled, and a physically trivial violation can
present as an `inf_pr` of 1–9. `ip_cvtol = 1e-4` is then effectively demanding
something nearer `1e-6` physical, which nothing in this problem needs.
`dev/PHASE2_CONDITIONING.md`'s constraint-scaling section and
`scratch/estimators/conditioning.py` are where to confirm it.

**Do not conclude.** Which of scaled-vs-unscaled `inf_pr`, variable-bound
contributions, or the `group_nviol` tolerance explains the gap is not yet pinned
down — only that a gap of ~100× exists and that it, not smoothness, is what caps
these solves. Four candidates.

**Blocked on.** `gmin.min(axis=1)` is unusable because `arc_sep` is `NaN` in every
candidate (see X2). Any further work on group slacks should exclude that column
until the empty-group defect is fixed.

---

## X4 — Active-set rank and the float32 stationarity floor

**Date:** 2026-09-22 · **Items:** D3, D4 · **Status:** both answered, both negative

**Method.** `scratch/estimators/solution_rank_and_noise.py`, rebuilding each capped
candidate's problem under the arms' exact environment and evaluating at the
returned point. Rank from the SVD of the rows active within a tolerance; noise from
the change in `∇f` under a one-float32-ulp perturbation of `x`.

**D3 — the active Jacobian is not rank-deficient.** At the returned point of all
three `f32` capped candidates, **no** constraint row is within `1e-3` of its bound,
and only 1–3 rows are within `1e-2`. Those are full row rank with condition numbers
of 1–2.55. The returned points sit in the interior of the constraint set with
almost nothing active, so LICQ failure cannot be what stalls these solves.
**One sentence: the active set is tiny and well conditioned, so degeneracy is not
the cause.**

**D4 — the float32 floor is far below the tolerance.** `|∇f|∞` is 1.23–1.30, and
one ulp in `x` moves the gradient by a median of `6.0e-6` to `1.2e-5`. Against
`ip_tol = 1e-4` and `ip_acc_tol = 5.0`, the precision floor is an order of
magnitude below the tighter of the two. **One sentence: float32 gives a
stationarity floor near `1e-5`, comfortably under `ip_tol`, so precision is not
the binding constraint and C1 (float64) is not warranted.**

**Do not conclude.** Three candidates, one arm, evaluated at the returned point
rather than the last iterate — and X5 shows those differ substantially.

---

## X3 — Phase-1 RProp trace: parity and first readings

**Date:** 2026-09-22 · **Item:** I1 · **Status:** built and verified

**Question.** Can the RProp loop report how it behaves without changing what it
computes, and do the channels discriminate?

**Method.** `trace` added to `make_staged_rprop` as a static Python flag
(`search/minimizers.py`), wired through `Phase1Settings.trace` / `RUTTER_TRACE`
to an `<out>_trace_<n>arc.npz` beside the pool. Commits `abeb9f0`, `81fbefe` on
`smooth-model-p1trace` in the `-p1angle` worktree. Verified by running the
synthetic Phase-1 smoke against the pre-change code.

**Result.** Default-off is **bitwise identical** across `x`, `x_reduced`,
`objective`, `min_clear` and `min_clear_reduced` — checked twice, once for the
minimizer change alone and again with the pipeline wiring in place. 904 tests
pass, 2 skip; ruff, codespell and mypy clean on the changed files. Six new unit
tests, including that tracing does not move the iterate.

First readings, synthetic subject, 4 candidates, 40 iterations per stage:

| channel | `reduced_fine` | `full_fine` |
|---|---|---|
| `flip` | 0 → 0.29, peak 0.86 | 0 → 0.15, peak 0.62 |
| `clipped` | 0 of 40 iterations | 18 of 40 |
| `eta_ceil` | 0 | 7 of 40 |
| `eta_floor` | 0 | 0 |
| `step` | 0.92 → 0.016 | 0.55 → 0.012 |

**Reading.** The channels discriminate. Sign-flip churn reaching 62–86% of movable
coordinates is high for a loop whose step rule halves `η` on every flip, which is
the mechanism behind the η-collapse hypothesis. `eta_floor` stays at zero here
only because 40 iterations is too few to halve `η` from its start to `1e-6`;
production stages run far longer, which is what I2 measures.

**Noticed while verifying.** A coordinate pinned by `lo == hi` starts at `η = 0`
but `jnp.maximum(eta * shrink, eta_min)` lifts it to `eta_min` on its first sign
flip, after which it steps by `1e-6` and is clipped straight back. Harmless, and
excluded from the channels by the movable mask, but it means a pinned coordinate's
stored `η` is not always zero.

**Do not conclude.** Four candidates on the synthetic subject, 40 iterations. This
establishes that the instrument works, not anything about production Phase 1.
Tracing changes XLA fusion, so traced runs compare to each other, not to untraced.

---

## X0 — Prior measurements carried in

Recorded before this document existed; kept here because later entries cite them.

**Phase-2 derivatives are correct; the model is non-smooth** (2026-09-22,
`scratch/estimators/derivative_check.py`). A central-difference sweep showed the
supplied derivatives match the functions IPOPT sees. The stalling is therefore a
property of the model at the solve's points, not a gradient bug. Sources measured:
the hard `min` over shanks and `max(0,·)` gate in `objectives/rewards.py`; bf16
interpolation *arithmetic* in `clearance/grids.py`; the spin `(sx, sy)` radial null
direction and its arctan2 origin singularity.

**bf16 storage with float32 blend matches float32 storage exactly for smoothness**
(`scratch/estimators/bf16_storage_f32_blend.py`). Grid footprint 296 MiB at bf16
against 593 MiB at float32 per worker. float64 storage is not needed.

**Synthetic parity for the smooth-model opt-ins** (60 iterations, three
candidates): all opt-ins off reproduces baseline poses bitwise 3/3. `blend`
against settings-only moved kept from 0/3 to 2/3. `angle` plus `smooth` against
`blend` was mixed, 2/3 to 1/3. Three candidates on a synthetic subject — direction
only, no weight.

**Phase-1 spin-angle default-off is bitwise identical** (commit `a4826e8`) across
`x`, `x_reduced`, `objective`, `min_clear` and `min_clear_reduced` on the synthetic
subject. With angles on, the run completes and spin radii stay at 1.000 in both
arms, so the unit-circle penalty is not what the angle parameterization is
relieving. Read as trajectory sensitivity, not as improvement.

**Capped solves end at median mu 8e-3** (acceptable ones at 2.6e-3). The barrier
parameter is not being driven down — the solve is stuck at a barrier subproblem
rather than at the final one.

---

## X6 — `inf_pr` is not the quantity any stopping test reads; X2 and X5 are retracted

**Date:** 2026-09-22 · **Item:** D5 · **Status:** settled — **supersedes X2 and X5**

**Question.** D5 asked why the feasibility leg of the `acceptable` test can never
fire, given X5's finding that `inf_pr` floors two orders of magnitude above
`ip_cvtol`. The premise turned out to be false, so the answered question is
instead: which quantity does the test actually read, and what really gates it?

**Method.** All from `diag_hist` in the two finished arms, no compute. Four steps:
correlate `inf_pr` against the model's own worst violation `max(0, -group_min)`
over the groups, `arc_sep` excluded; repeat at lags −2…+2 to test whether the
callback's `inf_pr` and its `get_current_iterate` refer to the same point; score
the `acceptable` legs against the model's violation rather than `inf_pr`; and
predict each run's stopping iteration from a three-leg model of the test.

**Result.**

1. **The ratio is not constant** — `inf_pr / max(0, -group_min)` spans p10 1.0 to
   p90 2.4e4 *within* single runs. No scaling factor of any kind reproduces it.
2. **Same point, loose relationship.** Lag 0 maximizes the rank correlation in
   all 20 runs, so the off-by-one-iteration explanation is dead; but at lag 0
   Spearman is only 0.32–0.70 (mean 0.56). The two are not the same measure.
3. **The test fires with `inf_pr` enormous.** All 13 runs that returned
   `Solved_To_Acceptable_Level` had the model's violation at exactly 0.0 for
   every one of their last 8 iterations, while `inf_pr` at the final iteration
   was 0.0039 to 6.23 — up to 6e4 × `acceptable_constr_viol_tol`. IPOPT does not
   return "acceptable" with the leg it tests out of tolerance, so the leg it
   tests is the unscaled violation, which `group_min` reports and `inf_pr` does not.
4. **The exit mix was never a convergence failure.** Across the 20 runs: 13
   acceptable, 4 at the iteration cap, 3 locally infeasible, and **0 at
   `Solve_Succeeded`** — the same mix X1 recorded per arm.
5. **A third leg gates it.** Modelling the test as 8 consecutive iterations with
   violation ≤ `ip_cvtol`, `inf_du` ≤ `ip_acc_tol` and `mu` ≤ 1e-2 — IPOPT's
   default `acceptable_compl_inf_tol`, which this project never sets — predicts
   the stopping iteration within ±2 for 8 of the 13 acceptable runs, and
   correctly predicts "never fires" for all 3 locally-infeasible runs and 2 of
   the 4 capped ones. Two legs alone do not: 17 of 20 runs have an 8-iteration
   stretch satisfying violation and `inf_du` while the solver runs on.

**Reading.** `inf_pr` is IPOPT's internal primal infeasibility, which measures the
mismatch between a constraint value and its slack variable and is not bounded by
the constraint violation; `ip_cvtol` governs the unscaled violation instead. The
two coincide only at a converged point, which is why comparing them mid-solve
produced X5's phantom 100× gap. The `acceptable` test is half-configured: this
project sets `acceptable_tol` and `acceptable_constr_viol_tol` but leaves
`acceptable_compl_inf_tol` at 1e-2 and `acceptable_dual_inf_tol` at 1e10, so
complementarity is the leg in control and `ip_acc_tol = 5.0` is close to inert.
This is the same mechanism X0 recorded and the ranking then set aside: capped
solves end at median `mu` 8e-3 against 2.6e-3 for acceptable ones, stuck on a
barrier subproblem.

**Do not conclude.** That the identification of `inf_pr` as the internal measure
is verified — it is inferred from points 1–3 and from the option defaults, not
from a run with `inf_pr_output` set both ways. That `mu` equals complementarity;
it is a proxy, which is why 5 of 13 predictions fire early. That constraint
scaling is or is not a problem — X5's evidence for it has evaporated, but nothing
here measures row conditioning. That the 4 capped runs are explained: 2 of them
are, 2 are not.

**Retractions.** X2 concluded "feasibility is the binding leg on every capped
solve" from `inf_pr` never reaching `ip_cvtol`. It measured a quantity no
stopping test consults; the conclusion does not hold. X5 concluded the model and
`inf_pr` "disagree by ~100×, which points at constraint scaling". They disagree
because they measure different things. **C3 was promoted to the critical path on
X5's evidence and that evidence is withdrawn.**

---

## X7 — The per-group constraint split is misaligned; `arc_sep`'s NaN is the symptom

**Date:** 2026-09-22 · **Item:** the `arc_sep` defect · **Status:** cause found, fix deferred

**Question.** Why does `arc_sep` report `+nan` as its worst slack in every
candidate, when the group has 3 rows carrying real values?

**Method.** Read the layout by hand: `slack_start` key order against
`SLACK_GROUPS`, the per-group sizes, `padding_mask`, `live_row_constraints`,
`_live_group_sizes` and the `drop_dead_rows` default. No compute.

**Result.** Key order matches and sizes are 224/84/42/28/3/21 = 402, so the
layout itself is sound and `padding_mask` marks all 3 `arc_sep` rows live. The
defect is a mismatch between two conditions:

- `live_row_constraints` shrinks the constraint vector only when
  `drop_padded_rows and hessian == "none"`, and `drop_dead_rows` **defaults to
  False**. IPOPT therefore receives all 402 rows.
- `_live_group_sizes` returns *live* counts whenever any row is padded,
  unconditionally. X4's build reports 168–172 live rows.

So the recorder splits a 402-row vector at boundaries that sum to ~168. Every
group after the first padded one reads the wrong offsets; the `arc_sep` slice
lands inside the padded `thread` block, where every value is `PADDED_SLACK` and
the recorder's `part < padded_slack - 1.0` filter empties it, giving `NaN`; and
the trailing group absorbs rows ~147 through 402.

**Reading.** Any *per-group attribution* taken from `group_min`/`group_nviol` is
unreliable on the default settings, and the trailing group is the worst affected
because it absorbs the remainder. X2's "`ml_sep` is the only group ever violated"
is an artifact of that absorption, which is a second independent reason that
finding does not stand.

**What this does not touch.** The six slices still *partition* all 402 rows, so a
max or a sum taken over the union is unaffected by the mislabeling. X6's
`max(0, -group_min)` is a max over the union and its numbers stand — the excluded
`arc_sep` slice is padded rows that cannot be violated. The compl-forensics run
re-derives the same quantity from IPOPT's own `g_violation` independently.

**Fix deferred, deliberately.** It is a source change in a module the running
arms import, and spawn workers re-import on respawn. It waits for the arms.

---

## X8 — The prior art already had half of X6, and it answers C6

**Date:** 2026-09-22 · **Item:** D5/C5/C6 · **Status:** correction to X6's framing

**Question.** What does `dev/PHASE2_CONDITIONING.md` already establish about the
tolerances, and does X6 add anything to it? Asked late — this file's own preamble
says to read that document *before* adding an entry, and X6 did not.

**Already established there (2026-09-16), not by X6.**

- **The displayed and the tested infeasibility are different quantities.** With
  `inf_pr_output="internal"` the column reads 1.5e-01 through the regular
  iterations while the final summary's unscaled `Constraint violation` stays
  exactly 0, and the returned point has a minimum slack of +0.0205 mm over all
  402 rows. X6's central identification was documented six days earlier.
- **`constr_viol_tol`, `dual_inf_tol` and `compl_inf_tol` are checked on
  unscaled quantities**, which is precisely why the `acceptable` test fires while
  `inf_pr` is large.
- **`acceptable_tol = 5.0` is a deliberate, measured choice**, not an oversight.
  At the observed dual-infeasibility range of 0.64–3.05 no smaller value can
  fire; 5 fires on every candidate and cuts iterations roughly fivefold at 6/6
  FCL kept. Before it, status 1 never appeared at all.
- **`solver_status` is already ignored by the pipeline.** `kept` is
  `fcl_keep and thread_keep`, and 24 of 33 status-2 exits in one 40-candidate run
  passed independent FCL validation. Exit code is not the success measure.
- **Objective claims need the 40-candidate benchmark.** Run-to-run
  nondeterminism moved one candidate's objective by 1.7 under identical settings,
  so six candidates cannot resolve a 0.13 mean difference.

**What X6 does add.** That the value recorded in `diag_hist["inf_pr"]` comes from
IPOPT's *intermediate callback*, which passes the internal quantity regardless of
`inf_pr_output` — the prior run had to set that option explicitly to see it. So
every reading this project takes from the recorded column inherits the internal
measure, which is what made X2 and X5 wrong. And that complementarity, left at
`acceptable_compl_inf_tol = 1e-2`, is the remaining leg on the candidates that
still cap.

**Correction to the framing.** C6 as posed — "0 of 20 reach `Solve_Succeeded`, is
acceptable the target?" — is largely answered: **yes, by design.** Stopping at
acceptable was chosen knowingly, it rests on the independent FCL gate, and the
pipeline already ignores exit codes. The open part is narrower: whether the four
capped candidates can be brought into the same early exit, which is C5's
question. **A defaults recommendation must therefore be scored by `kept` and
coverage on the 40-candidate protocol, not by exit status** — which is also why
X1's per-arm objective medians on ten candidates cannot carry a recommendation.

**Process lesson.** X6 spent a session re-deriving a documented result because
the prior art was not read first. The instruction to read it exists at the top of
this file; following it would have started from the complementarity leg, which is
the only genuinely open part.

**Evidence level for X8's callback claim.** That the intermediate callback passes
the internal quantity *regardless of* `inf_pr_output` is an inference from two
measurements, not a verified fact: the prior run had to set the option to
`internal` to see 1.5e-01 in the printed column, while our callback records
values of the same character with the option unset. Consistent, but the direct
test — one solve with `inf_pr_output="original"` set explicitly, comparing the
recorded column against the unset run — has not been done. It is cheap and worth
adding to the next forensics run.

---

## X9 — The goal's criterion-2 answers, consolidated

**Date:** 2026-09-22 · **Item:** D2/D3/D4 + the `inf_du` question · **Status:** answered

Criterion 2 asks for one sentence per diagnostic and a yes or no on `inf_du`.
D3 and D4 are answered in X4 and stand unchanged. D2's answer in X2 was wrong and
is replaced here.

**D2 — which leg of the `acceptable` test fails.** *Not feasibility.* Against the
unscaled violation the test actually reads, the feasibility leg is satisfied
routinely — 13 of 20 runs exit `Solved_To_Acceptable_Level` with the model's
violation at exactly zero — and what still gates the four that cap is
complementarity, held at IPOPT's `acceptable_compl_inf_tol` default of 1e-2 that
this project never set (X6, X8; X2 retracted).

**D3 — active-Jacobian rank.** The active set is tiny and well conditioned, so
degeneracy is not the cause (X4).

**D4 — float32 noise floor.** float32 gives a stationarity floor near 1e-5,
comfortably under `ip_tol`, so precision is not binding and C1 is not warranted
(X4).

**Is `inf_du` the right target? — No.** It is a gate but not a target, and the
distinction matters:

- It *is* a gate. The prior art measures the overall NLP error as equal to the
  dual infeasibility, so `acceptable_tol = 5.0` is in practice a bound on
  `inf_du`, and X6's model of the exit needs that leg.
- It is *not* a target. `acceptable_dual_inf_tol` sits at its 1e10 default, so
  the dual leg proper is off. `kept` is `fcl_keep and thread_keep` and ignores
  `solver_status` entirely, and 24 of 33 status-2 exits in a 40-candidate run
  produced FCL-valid plans. Most directly, X1 has `f32` keeping 9 of 10 against
  `nothread`'s 4 of 10 **while being roughly four times worse on median
  `inf_du`** (1.53 against 0.344).

So the one comparison in hand runs opposite to the target: optimizing `inf_du`
would have preferred the arm that produced fewer usable plans. **Yield — `kept`
and coverage — is the target; `inf_du` is a stopping-path parameter to be set,
not minimized.** This is the answer criterion 2 asks for, and it is also why
criterion 3 insists the recommendation be scored on the 40-candidate protocol.

---

## ID note — C6 and C7 were re-lettered on 2026-09-22

Entries above cite item IDs that have since changed. Nothing in them is
withdrawn; only the labels moved, and the entries are left as written because
this file is appended, not rewritten.

- **C7 → B1.** The misaligned per-group split (X7). `B` is a new prefix for a
  defect found in passing. It was never conditional and never gated on a
  diagnostic, so `C` was the wrong letter for it.
- **C6 → no ID.** "What should Phase 2 be converging to" is a decision for the
  user rather than work, so it lives in `STATUS.md` under "Blocked on the user".
  X8's framing correction stands unchanged.
- **C5 unchanged** — genuinely conditional, and still the open item.

The playbook now carries a legend for the prefixes and the rule that an ID must
be defined there before it is used anywhere else.

---

## X10 — Complementarity is what caps the solve, measured from IPOPT directly
> **CONTESTED BY X25, RESTORED BY X26.** X25 argued the capped solves never reach
> feasibility so no acceptance tolerance could reach them. That is false on direct
> measurement, and this entry's method — `get_current_violations(scaled=False)`,
> the unscaled quantities the acceptable test compares against — is the right one.

**Date:** 2026-09-22 · **Item:** C5 · **Status:** first arm settled; `cfg` arm running

**Question.** X6 fingered complementarity using `mu` as a stand-in. Is it really
the leg that blocks the capped solves, measured against IPOPT's own quantities?

**Method.** `scratch/estimators/compl_forensics.py` re-solves candidate 11367
from its Phase-1 start under production options, recording
`get_current_violations(scaled=False)` beside every logged iteration — IPOPT's own
unscaled complementarity, constraint violation and Lagrangian gradient, which are
what `acceptable_compl_inf_tol`, `acceptable_constr_viol_tol` and
`acceptable_dual_inf_tol` are compared against. One process, two pinned cores.

**Result.** The arm reproduced the recorded exit exactly: max-iter at 1000.

| quantity | reading |
|---|---|
| `max｜g_violation − max(0, −group_min)｜` | **0** |
| median `inf_pr / g_violation` | **230** |
| median `｜inf_du − grad_lag_x｜` | **0** |
| terminal compl / viol / grad_lag | 0.0121 / 0.00143 / 1.07 |
| three legs, compl tol **1e-2** (IPOPT default) | compl ok 0.08, longest run **6** — needs 8 |
| three legs, compl tol **5.0** | compl ok 0.93, longest run **10** — fires |

**Reading.** Four things, in order of how load-bearing they are:

1. **Complementarity is the blocking leg.** At IPOPT's unset default of 1e-2 the
   longest qualifying stretch is 6 iterations against the 8 required, so the run
   cannot exit and burns the full 1000. Raise that one tolerance to the value
   `ip_acc_tol` already carries and the stretch reaches 10. This is C5's premise,
   confirmed against IPOPT's own numbers rather than the `mu` proxy.
2. **X6's violation measure was exact.** IPOPT's unscaled `g_violation` equals
   `max(0, −group_min)` to the last bit, which also confirms X7's prediction that
   the misaligned split does not disturb a max taken over the union.
3. **`inf_pr` is confirmed as a different quantity** — 230× the unscaled
   violation — and this is now close to direct evidence for X8's inference:
   `inf_pr_output` was at its default `"original"` throughout, yet the callback's
   value is not the original problem's violation. The callback does not honour
   the option that names the printed column.
4. **`inf_du` is honest.** It equals the unscaled Lagrangian gradient exactly, so
   unlike `inf_pr` it can be read at face value. X9's "gate but not target" verdict
   rests on what it measures, not on whether it measures it correctly.

**Do not conclude.** That the fix works — the three-leg count above is *my model*
of IPOPT's test, not IPOPT's test. The `cfg` arm, which sets
`acceptable_compl_inf_tol` and `acceptable_dual_inf_tol` and asks the solver
itself, is still running. One candidate, one arm; a default change needs the
40-candidate pair.

---

## X11 — Setting `acceptable_compl_inf_tol` turns a 1000-iteration cap into a 120-iteration exit
> **CONTESTED BY X25, RESTORED BY X26.** See X10's note. Caveat unchanged and
> independent of that dispute: this arm moved `acceptable_dual_inf_tol` to 5.0
> alongside the complementarity bound, so it is a two-factor change on one
> candidate, and the 1000 → 120 result cannot be attributed to the complementarity
> bound alone. `compl5_a` at n=40 moves only the one setting.

**Date:** 2026-09-22 13:43 · **Item:** C5 · **Status:** mechanism settled on one candidate

**Question.** X10 showed complementarity is the blocking leg by *modelling* the
`acceptable` test. Does the solver itself agree?

**Method.** The second arm of `compl_forensics.py` on candidate 11367: identical
problem and start, production options plus `acceptable_compl_inf_tol = 5.0` and
`acceptable_dual_inf_tol = 5.0`.

**Result.**

| arm | exit | iterations | terminal viol | terminal compl |
|---|---|---|---|---|
| `asis` | max-iter | **1000** | 0.00143 | 0.0121 |
| `cfg` | **`Solved_To_Acceptable_Level`** | **120** | **0** | 0.153 |

An 8.3× reduction, ending strictly feasible where the capped run did not.

**The confound resolves itself.** The arm moved two tolerances, but only one can
explain the result: `acceptable_dual_inf_tol` went from its default **1e10** to
5.0, which makes that leg strictly *harder* to satisfy and cannot produce an
earlier exit. The entire effect is `acceptable_compl_inf_tol` going 1e-2 → 5.0.
Terminal complementarity of 0.153 confirms it — fifteen times the default
threshold, so under production options this point is unreachable as an exit.

**Reading.** The capped solves are not failing to converge. They reach a strictly
feasible point with acceptable stationarity and then keep iterating for 880 more
iterations because one tolerance nobody set holds the exit shut. On the benchmark
that is **42 of 80 solves at the cap**, so the ceiling on this is large.

**Do not conclude.** That the default should change. This is one candidate and it
measures *iterations*, not yield: nothing here reports `kept`, coverage or the
objective, and the prior art's warning applies — the earlier exit is a less
stationary point (`grad_lag` 2.68 against 1.07), and a 0.13 mean objective
difference was not resolvable at six candidates. Whether the saved compute costs
plan quality is exactly what the 40-candidate × 2 protocol exists to answer, and
it is criterion 3's job, not this entry's.

---

## X12 — The control wins on yield: no arm beats `base`

**Date:** 2026-09-22 13:59 · **Item:** D1 · **Status:** four of six arms, control in

**Question.** With `base` finally finished, what is each intervention's signed
effect on kept-plan yield?

**Method.** `compare_arms.py` over the four finished handoffs on the capped-10 set.
`blend` (9/10) and `blend_smooth` (6/10) are still running.

**Result.**

| arm | exits | med it | med `inf_du` | **kept** | med obj |
|---|---|---|---|---|---|
| **`base`** | acceptable 6, cap 3, infeas 1 | 656 | 1.39 | **10/10** | −5.631 |
| `nothread` | acceptable 6, infeas 3, cap 1 | 547 | 0.344 | **4/10** | −5.355 |
| `f32` | acceptable 7, cap 3 | 390 | 1.53 | **9/10** | −5.771 |
| `blend_angle` | acceptable 9, cap 1 | 368 | 1.46 | **7/10** | −5.975 |

Signed effects against the control: `nothread` **−6**, `blend_angle` **−3**,
`f32` **−1**. Every intervention loses on yield.

**The stopping rule is met, and cannot be unmet.** `base` keeps 10 of 10, so the
best the two unfinished arms can do is tie. No arm beats the control on kept-plan
yield, whatever `blend` and `blend_smooth` return.

**This retracts X1's headline.** X1 read `f32`'s 9/10 against `nothread`'s 4/10
as `f32` winning "on every outcome that matters", flagged as provisional for
exactly this reason. With the control in, `f32` *loses* to it. The caution in the
goal — that 9/10 could not justify flipping `RUTTER_GRID_DTYPE` — was
well placed, and the flip now has no support at all on this set.

**Reading, with the metric's limit stated plainly.** Kept saturates: `base` is at
the ceiling on ten candidates, and `dev/PHASE2_CONDITIONING.md` already warns that
kept rate is not resolvable at this size, since every kept candidate clears by a
median of just +0.021 mm. So "no arm beats the control" is partly a statement
about a metric with no headroom on a hand-picked tail. It is still the honest
reading of the only yield evidence in hand, and it is the condition the stopping
rule names.

Two things point the other way and are *not* yield: `blend_angle` and `f32` both
cut median iterations by ~44% and ~40%, and `blend_angle` nearly eliminates
capping (9 acceptable exits against the control's 6). Both also reach a better
median objective than the control while keeping fewer plans, which is a warning
that objective and yield are not aligned on this set.

**Do not conclude.** That the interventions are harmful in production — ten
hand-picked capping candidates cannot support that either. The direction of the
evidence is what matters: it does not support adopting any of them.

## X13 — `kept` is a feasibility gate, not a score; and what `f32` actually buys

Raised by the user against X12: *"Keeping a plan does not mean it's feasible"*, and
*"f32 should not push plans out of feasibility, should it?"* Both land. This entry
corrects X12's framing and answers the dtype question with a measurement.

**X12's metric was described wrongly.** `kept = fcl_keep and thread_keep` is a
pass/fail feasibility gate, so a signed effect of **−1** does not mean "slightly
worse" — it means one plan is *not usable*. Calling that "kept-plan yield" and
ranking arms by it reads a gate as if it were a score. The counts in X12 stand;
the word does not.

**The gate also discards the margin.** Comparing the FCL margin instead, on the
nine candidates `base` and `f32` both keep, `f32` clears by a **median +0.0071 mm
more**, and by more on **6 of 9**. On yield `f32` is −1; on clearance it is
mostly ahead. A gate count cannot show that.

**`f32`'s one loss is a different basin, not eroded geometry.** Candidate 1282,
both arms from the same start, same hole assignment, same partition:

| | exit | FCL | thread | coverage | pose |
|---|---|---|---|---|---|
| `base` | acceptable/620 | **+0.0102** | −0.0453 | 3.820 | — |
| `f32` | acceptable/570 | **reject** (`CA1↔implant`) | −0.0053 | 3.200 | 11.32 away |

Both *started* at `fcl −1.0`; `base` repaired that collision and `f32` did not.
`−1.0` is the reject sentinel, not a measured depth. `f32` ended 11.3 away in pose
space at worse coverage — a different solution, not a margin that wobbled over a
line.

**And at that point its own model said the pose was clear.** `f32`'s terminal
slacks on 1282 are all positive — `probe_fixture` **+0.0619 mm**, `probe_pair`
+0.0016, `brain` +0.0160, `thread` +0.0054 — while FCL finds a real collision on
exactly the probe/fixture pair. The solver satisfied its constraints; the
constraints were wrong there.

**How often does each arm's model disagree with FCL:**

| arm | model says clear | FCL agrees | **model optimistic** | FCL stricter only |
|---|---|---|---|---|
| `base` | 6 | 6 | **0** | 4 |
| `nothread` | 8 | 4 | **4** | 0 |
| `f32` | 8 | 7 | **1** | 2 |
| `blend_angle` | 8 | 6 | **2** | 1 |

"Model optimistic" is the dangerous direction: constraints satisfied, FCL still
colliding. `base` never does it, but only because it *ends with residual
violations* on 4 of 10 — it is conservative by failing to converge, not by being
right.

**Measured: what bf16 storage costs, on the actual grids.** Voxel pitch 0.2 mm;
error of the bf16 blend against the fp32 blend over the same fp32-addressed
voxels, by distance band (all seven probe grids agree to the digits shown):

| band | median | p99 | max |
|---|---|---|---|
| `|d| < 0.05 mm` (at contact) | **1.5e-4** | 6.4e-4 | 1.1e-3 mm |
| 0.05–0.2 mm | 3.2e-4 | 1.6e-3 | 2.6e-3 mm |
| 0.2–1 mm | 1.3e-3 | 6.6e-3 | 1.3e-2 mm |
| ≥ 1 mm | 1.4e-2 | 9.0e-2 | **2.3e-1 mm** |

So the user's intuition is right where it is checkable: **at the zero level set
bf16's noise (median 1.5e-4 mm) sits at or above `ip_cvtol` = 1e-4 mm.** Under
bf16 the solver is driving a violation to a tolerance finer than its own
constraint function can resolve. That is a real defect, and `f32` removes it.

**But `f32` is not more accurate collision detection.** The model-vs-FCL
disagreement on 1282 is at least **0.062 mm** — about **400×** the bf16 noise at
contact. That gap is the 0.2 mm voxel sampling and the envelope offset, and dtype
does not touch either; it is also the band `PHASE2_CONDITIONING.md` already
identifies, where kept margins sit (median +0.021 mm). Raising storage precision
cannot fix a sampling error 100× larger than the storage error.

**Conclusion.** `f32` buys a **better-conditioned problem**, not better geometry:
median iterations 390 vs 656, constraints actually satisfied 8/10 vs 6/10, and
noise pushed below the feasibility tolerance instead of above it. Those are
mechanisms, and they are consistent with each other. It does not buy agreement
with FCL, and X12's −1 is one candidate that fell into a worse basin.

**This does not flip the default.** Ten hand-picked capping candidates still
cannot justify it, and the memory cost (296 → 593 MiB per worker) is unchanged.
It does move the question: the case for `f32` should be argued on conditioning —
noise vs `ip_cvtol`, iterations, constraint satisfaction — and tested on the
40-candidate × 2 protocol, not on the kept count.

## X14 — Phase 2 never converges; the acceptable test certifies feasibility only

Asked by the user against X13: *"the base is still not converging then?"* No — and
neither does any arm. This is the settled answer to the goal's standing question
of what Phase 2 should be converging to.

**Not one solve in either arm reached `Solve_Succeeded`.** Twenty solves, zero.

| arm | Succeeded | Acceptable | MaxIter | Infeasible |
|---|---|---|---|---|
| `base` | **0** | 6 | 3 | 1 |
| `f32` | **0** | 7 | 3 | 0 |

**What `Solved_To_Acceptable_Level` is actually testing here.** With
`ip_acc_tol = 5.0` against `ip_tol = 1e-4`, the overall-NLP-error leg is 5e4x
looser and never binds; `acceptable_dual_inf_tol` sits at its unset IPOPT default
of **1e10**, so the stationarity leg is not checked at all. What remains is
constraint violation <= `ip_cvtol` 1e-4 held for `ip_acc_iter` 8 iterations, and
complementarity <= 1e-2. **An acceptable exit certifies a feasible point, not an
optimal one.**

**And the points are nowhere near stationary.** Terminal `inf_du` on every
acceptable exit:

| arm | median | min | max | as multiples of `ip_tol` |
|---|---|---|---|---|
| `base` | 0.93 | 0.71 | 2.31 | 7,000x – 23,000x |
| `f32` | 1.30 | 0.82 | 4.97 | 8,000x – 50,000x |

X6 established `inf_du` is honest — it equals the unscaled Lagrangian gradient
exactly — so these are not a scaling artifact.

**Exit status and terminal feasibility line up exactly.** Every candidate ending
with a real violation is a cap; every acceptable exit is feasible within
`ip_cvtol`. `base`'s residuals at cap are `probe_pair` 6.7e-3, 8.2e-3, 1.6e-2 mm;
`f32`'s are 2.4e-2 and 2.8e-2 mm.

**This revises X13's reading of `f32`.** X13 credited `f32` with a
better-conditioned problem on 390 vs 656 median iterations. Against stationarity
that is partly **stopping sooner**: `f32` exits 40% earlier and at a *higher*
median `inf_du` (1.30 vs 0.93). Cleaner constraints trip a feasibility-gated test
before the objective is worked down. Candidate 1282 is the clearest case —
`f32` stopped at 570 iterations with `inf_du` 4.97, the worst value in either
arm, at coverage 3.20 against `base`'s 3.82, and in a basin FCL rejects.

**Minor correction to X13's agreement table.** It counted "model says clear" as
every slack strictly >= 0, but the tolerance is `ip_cvtol` 1e-4. `base`'s 1274
ends at 7e-5, within tolerance, so `base` is 7 clear / 7 agree / **0 optimistic**
/ 3 FCL-stricter-only. `f32` is unchanged at 8 / 7 / **1** / 2, and the
conclusion is unchanged.

**What this means for the defaults question.** Phase 2 is a feasibility finder
with an objective bias, not an optimizer that converges, and its stopping test
cannot distinguish the two. Any recommendation that makes exits *earlier* —
including C5's `acceptable_compl_inf_tol`, which turned a 1000-iteration cap into
a 120-iteration exit (X11) — is buying compute by stopping further from
stationarity, and must be validated on objective and coverage, not only on
feasibility and yield.

## X15 — The arms are two incomparable groups, and X13 re-derived a fix already committed

Found while auditing what of X13/X14 had actually been implemented. Two errors,
both mine, both structural.

**The six arms did not run the same code.** Launch times against the commit
timeline:

| arms | launched | tree | `bab0970` (fp32 blend)? |
|---|---|---|---|
| `base`, `nothread`, `f32` | 08:48 | main checkout @ `b3b6bbe` + uncommitted | **no** — `bab0970` landed 09:07 |
| `blend`, `blend_angle`, `blend_smooth` | 11:05 | smooth @ `3b83c55` | **yes**, plus `c4fc8ab`, `d969bd7`, `a4826e8` |

So there is no single control. `base` is the control for the first group and
`blend` — same settings, fixed tree — is the control for the second.
**X12's signed effect for `blend_angle` (−3 against `base`) is invalid**; it must
be taken against `blend`, which is still running. The `f32`-vs-`base` comparison
in X13 and X14 is sound, since both are in the first group.

**The goal's criterion 1 cannot be met as written.** "All six arms read, control
included, every intervention with a signed effect against `base`" assumes one
control. Either the first group is re-run on the fixed tree, or the two groups are
reported separately, each against its own baseline.

**X13 re-derived `bab0970`.** That commit, made at 09:07 today, fixes exactly the
defect X13 reported as a finding — trilinear weights carried at the grid dtype, so
a bf16 grid steps as the query point moves — and its message states the same
mechanism, that IPOPT's line search reads those steps as noise. X13 measured the
main checkout, which does not have it. Re-measured on the fixed tree:

| band | before `bab0970` | after |
|---|---|---|
| at contact (`\|d\| < 0.05 mm`) | median **1.5e-4**, p99 6.4e-4, max 1.1e-3 | median **3.9e-5**, p99 2.0e-4, max 3.2e-4 |
| 0.05–0.2 mm | median 3.2e-4 | median 1.0e-4 |
| >= 1 mm | median 1.4e-2, max 2.3e-1 | median 4.1e-3, max 5.8e-2 |

The fix cuts bf16 noise about 3.8x. The contact-band median moves from **1.5x
`ip_cvtol`** to **0.39x** — from straddling the feasibility tolerance to sitting
under it.

**Which reverses X13's reading of `f32`.** X13 argued `f32` buys a
better-conditioned problem because bf16 noise sat above `ip_cvtol`. On the fixed
tree that premise is gone: bf16 is already below tolerance at contact, and the
`f32` arm was largely measuring `bab0970`'s defect rather than dtype. The test is
`blend` — bf16, baseline settings, fixed tree. If `blend` matches `f32` on
iterations and constraint satisfaction, dtype was never the lever.

**Process.** Both errors trace to the same omission: I did not read the branch's
log before running or before concluding. `git log b3b6bbe..<branch>` would have
shown `bab0970` and `619cb3f` immediately. The playbook's bootstrap says to read
the prior art; the branch history is prior art.

## X16 — `bab0970`, not dtype, fixed conditioning; and stationarity at exit tracks feasibility

`blend` finished at 14:29, giving the second group its own control and letting
X15's prediction be tested.

**X14's overclaim, corrected first.** X14 said stationarity is "effectively
unchecked" because `acceptable_dual_inf_tol` sits at 1e10. Wrong: `acceptable_tol`
bounds the *overall* NLP error, which here is dominated by dual infeasibility, so
it binds that leg at 5.0. **All 28 acceptable exits across four arms have terminal
`inf_du` below 5.0, max 4.970** — pressed against the bound, never over. The
1e10 setting is redundant, not a hole, and `settings.py` already documented the
intent. Tightening the dual leg therefore means setting it *below* `ip_acc_tol`,
which isolates it instead of tightening all three legs at once.

**The fp32-blend fix beats the dtype flip.** Same settings, same candidates:

| arm | tree | grid | med it | caps | constraints satisfied | kept | med `inf_du` |
|---|---|---|---|---|---|---|---|
| `base` | pre-`bab0970` | bf16 | 656 | 3 | 7/10 | 10/10 | 0.93 |
| `f32` | pre-`bab0970` | fp32 | 390 | 3 | 8/10 | 9/10 | 1.30 |
| `blend` | **fixed** | bf16 | 474 | **1** | **10/10** | 8/10 | 1.39 |
| `blend_angle` | fixed | bf16 | 368 | 1 | 9/10 | 7/10 | 1.62 |

bf16 on the fixed tree satisfies its constraints on every candidate and caps once;
fp32 on the broken tree manages 8/10 and caps three times. **X15's prediction
holds — the lever was `bab0970`, not `RUTTER_GRID_DTYPE`.** This closes the dtype
question: the flip has no support, and it never had a mechanism the fix does not
already supply. X13's argument for `f32` is withdrawn.

**But every conditioning gain costs kept, and the ordering is exact.**

| arm | med `inf_du` | kept |
|---|---|---|
| `base` | 0.93 | 10/10 |
| `f32` | 1.30 | 9/10 |
| `blend` | 1.39 | 8/10 |
| `blend_angle` | 1.62 | 7/10 |

Spearman rho **−1.000** on four arms. Pooling candidates, lost solves stopped at a
median `inf_du` of 2.12 against 1.26 for kept ones (Mann-Whitney p = 0.033), but
those 40 are 10 candidates x 4 arms and not independent. The paired test that
controls for candidate — rank each arm's terminal `inf_du` within a candidate — is
weaker but monotone: of the ten solves at each rank, **rank 1 lost 0, rank 2 lost
1, rank 3 lost 2, rank 4 lost 3**. Mean rank of a lost solve 3.33 against 2.5 by
chance, on six losses.

**The mechanism this suggests.** Better-conditioned constraints trip a
feasibility-gated acceptable test sooner, so the solve exits further from
stationarity with less of the objective worked down, and the plan is likelier to
fail FCL. On that reading `kept` was never measuring the interventions; it was
measuring how early each one let the solver stop. It also explains why `base`,
the worst-conditioned arm, keeps the most: it is forced to keep iterating.

**This is suggestive, not established** — six losses on ten hand-picked capping
candidates. It is a hypothesis with a sharp prediction, which is what the arms
launched at 14:36 test.

**Prediction under test.** `du_ctl` / `du_1` / `du_p1` hold everything fixed and
set `acceptable_dual_inf_tol` to 1e10, 1.0 and 0.1 on the fixed tree. If
stationarity at exit is what drives feasibility, the tightened arms should **keep
more plans than `du_ctl` while spending more iterations**. If kept does not move,
the relationship above is an artifact of four arms that happen to order that way,
and the whole line should be dropped.

## X17 — Group B complete: `blend_smooth` beats its control, and the stationarity ordering holds at five arms

`blend_smooth` exited 0 at 15:04 — it had not crashed. The `BrokenPipeError`s in
its log were a pool worker that died and was replaced; the truncation damage of
14:13 destroyed the surrounding context, not the run.

**Group B against its own control.** All three on the fixed tree:

| arm | med it | caps | accep | constraints satisfied | kept | med `inf_du` | med cov |
|---|---|---|---|---|---|---|---|
| `blend` (control) | 474 | 1 | 9 | 10/10 | 8/10 | 1.39 | 3.721 |
| `blend_angle` | 368 | 1 | 9 | 9/10 | **7/10** (−1) | 1.62 | 3.861 |
| `blend_smooth` | 643 | 3 | 7 | 8/10 | **10/10** (+2) | 0.94 | 3.646 |

**`blend_smooth` is the first arm in this whole series to beat its control on
kept.** Under the goal's stopping rule that is the condition for not stopping —
but see below, because it may not be an intervention effect at all.

**The stationarity ordering now spans five arms, two trees and three unrelated
interventions:**

| arm | group | med `inf_du` | med it | kept |
|---|---|---|---|---|
| `base` | A | 0.93 | 656 | 10/10 |
| `blend_smooth` | B | 0.94 | 643 | 10/10 |
| `f32` | A | 1.30 | 390 | 9/10 |
| `blend` | B | 1.39 | 474 | 8/10 |
| `blend_angle` | B | 1.62 | 368 | 7/10 |

Spearman(med `inf_du`, kept) = **−0.975, p = 0.0048**. Spearman(med iterations,
kept) = +0.872, p = 0.054.

The two arms at 10/10 are `base` and `blend_smooth` — different trees, different
models, one with the threading reward hardened and one with it smoothed — and they
agree to two digits on both `inf_du` (0.93 / 0.94) and iterations (656 / 643).
Nothing about the interventions predicts that; where the solve stopped does.

**So `blend_smooth`'s +2 is confounded with its cost.** The smooth threading
reward makes the problem *harder* to converge — 643 iterations against 474, three
caps against one, and constraint satisfaction *down* to 8/10. On the reading in
X16 it keeps more plans because the solver could not stop early, not because the
reward is better. A cheaper model that simply refused to stop should do the same.

**That is exactly what the running arms test.** If `du_1` or `du_p1` reaches 10/10
on the plain model purely by tightening `acceptable_dual_inf_tol`, then
`blend_smooth`'s win is a stopping-time artifact and the smooth reward buys
nothing the tolerance does not. If tightening does *not* recover kept, the
stationarity reading is wrong and `blend_smooth` is a real effect worth keeping.

**Either way, the series' headline changes.** X12 concluded "no arm beats the
control." With the groups separated and the second control in, one arm does, and
the metric it wins on is the one X16 suspects of measuring exit time. The
defaults question cannot be answered until that is resolved.

## X18 — The stationarity ordering was an artifact: I dropped the arm that breaks it

Prompted by completing the `nothread` read, which criterion 1 required and which
X16 and X17 did not include. It breaks their central claim.

**`nothread` has the lowest `inf_du` of any arm and the worst kept.**

| arm | med `inf_du` | kept | objective / parameterization |
|---|---|---|---|
| `nothread` | **0.13** | **4/10** | CHANGED — no threading reward |
| `base` | 0.93 | 10/10 | stock |
| `blend_smooth` | 0.94 | 10/10 | CHANGED — smooth reward |
| `f32` | 1.30 | 9/10 | stock |
| `blend` | 1.39 | 8/10 | stock |
| `blend_angle` | 1.62 | 7/10 | CHANGED — spin reparameterized |

| arms included | Spearman rho | p |
|---|---|---|
| **all six** | **−0.116** | **0.827** |
| five, as reported in X17 | −0.975 | 0.0048 |
| three stock-objective only | −1.000 | — (n=3, and across two trees) |

**X16 and X17's headline is withdrawn.** The rho of −0.975 came from excluding
`nothread`, and I gave no reason for excluding it. With all six arms there is no
relationship. This was not a judgement call that went the wrong way; it was a
selection I did not notice myself making, then reported with a p-value.

**There is also a reason the comparison was never valid.** `inf_du` is the norm of
the Lagrangian gradient of *that arm's* objective, in *that arm's* coordinates.
`nothread` deletes a term from the objective, `blend_smooth` replaces it with a
smoothed one, and `blend_angle` changes the variables the gradient is taken in.
Their `inf_du` are not the same quantity, so ranking six arms by it compares
numbers that do not share units of meaning. Only `base`, `f32` and `blend` are
commensurable, and those three span two trees.

**`nothread`'s real signature is the model-optimistic count**, which is the
diagnostic that does survive: 9/10 constraints satisfied, `inf_du` 0.13, and
**5 of 10 cases where the constraints are satisfied and FCL still rejects**. It
converges beautifully to the wrong problem. That is a formulation result, not a
conditioning one, and it is the strongest evidence in the series that the
threading reward is load-bearing.

**D2 answer, with a yes or no behind it, per criterion 2.** *Is `inf_du` the right
target?* **No, not as a cross-arm ranking metric.** X6 established it is honest —
it equals the unscaled Lagrangian gradient exactly, so within one fixed objective
and parameterization it means what it says, and the `du` arms may legitimately use
it. But it cannot rank interventions that change the objective or the variables,
which is what three of six arms do, and using it that way produced X17's spurious
result.

**What survives, and what still tests it.** The `du` arms hold the objective,
the parameterization and the tree fixed and vary only
`acceptable_dual_inf_tol`. That is the one comparison in which `inf_du` is
commensurable, so they remain a valid test of whether stopping earlier costs
kept — now the *only* evidence that can settle it. `m_smooth` against `du_1`
compares a changed objective at a matched tolerance; their `inf_du` are still
incomparable, but `kept` at a matched stopping criterion is a fair test of the
intervention, which is what it was launched for.

**Addendum — X9 already said this, and X16/X17 contradicted it unnoticed.** X9,
written earlier the same day, answers the `inf_du` question **No** and gives the
reason X18 re-derives: "optimizing `inf_du` would have preferred the arm that
produced fewer usable plans", citing `f32` at 9/10 kept against `nothread` at 4/10
*while four times worse on median `inf_du`*. It concludes "`inf_du` is a
stopping-path parameter to be set, not minimized". X16 and X17 then proposed
ranking arms by exactly that quantity, using `nothread`'s absence to make the
ordering work — against a recorded answer built on `nothread`'s presence. Two
entries stood in the file contradicting each other for about half an hour because
I did not re-read the answer before extending it. Criterion 2 was already met by
X9; X18 changes none of its conclusions and only supplies the arm-level
arithmetic.

## X19 — The prior art says `kept` cannot resolve these differences; most of the series is noise
> **QUALIFIED BY X26.** The core claim — kept cannot resolve a few candidates at
> this scale — is right and confirmed. What does not follow is the use made of it
> here and in X25: an unresolvable metric licenses "unresolved", not "no effect".
> `smooth` and `angle` were dropped on that reasoning, and `angle` is the prior
> art's own prescribed fix for a defect in the current default. Score formulation
> changes on the continuous measures instead.

Found while preparing criterion 3's benchmark. `dev/PHASE2_CONDITIONING.md`
states it outright, and stated it before any arm in this series ran:

> Kept rate is not resolvable by this benchmark. Every kept candidate clears by
> p10 +0.002 mm, median +0.021 mm, p90 +0.040 mm — the same order as the measured
> sampling gap and envelope offset. **Differences of a few candidates between
> settings are noise.**

**Applying that standard to the six arms:**

| arm | kept effect | verdict |
|---|---|---|
| `nothread` | **−6** | **real** — and corroborated by an independent signal, 5/10 model-optimistic |
| `blend_smooth` | +2 | **noise** |
| `f32` | −1 | **noise** |
| `blend_angle` | −1 | **noise** |

**So `blend_smooth`'s +2 is not evidence**, and the stopping rule's premise —
whether any arm beats its control on kept — is being evaluated on a metric its
own prior art says cannot resolve a two-candidate difference on ten hand-picked
candidates. X17 treated that +2 as the first real win of the series. It is not.

**Only `nothread` clears the bar**, at six candidates and with a second,
independent signature: constraints satisfied yet FCL rejecting, in 5 of 10 cases.
That is a formulation result and it is the one durable finding of the arm series.

**The metric that does have resolution is run-to-run pose divergence**, which is
what the 40-candidate x 2 protocol measures — "It has resolution where the
kept/lost label does not." And on that benchmark the **current defaults are
already at its optimum**: 0.000 divergence, 40/40 bitwise identical, 100% label
agreement. Every alternative measured is worse, and the smooth reward is among
the worst — 3.932 divergence, 0/40 bitwise, 80% agreement, and at memory 60 the
single worst configuration recorded at 6.989.

**Which sets up the defaults recommendation, and a real tension.** Removing the
threading reward wins on determinism (100% agreement) and loses catastrophically
on validity (5/10 model-optimistic, 4/10 kept). Smoothing it loses on both. The
benchmark ranks by reproducibility, which is necessary but not sufficient — a
setting can be perfectly reproducible and reproducibly wrong. Any recommendation
has to state both axes, and `nothread` is the case that proves the second one
matters.

**Process.** This is the second time this document has overturned a conclusion I
reached without consulting it (X8 was the first). Its findings are not background;
they are results that bind. The playbook's bootstrap says to read it, and reading
it once at the start is evidently not enough — it needs re-reading whenever a new
metric is about to carry an argument.

## X20 — the CPU path is reproducible, and `6380f2d` is parity-clean on real candidates

`du_ctl` is the baseline configuration with no overrides, launched 14:36 from the
p1angle tree onto cores 10,11. `blend` is the same configuration launched 11:05
from the smooth tree onto a different core slice. They share the capped-10 rank
file, and at 16:01 `du_ctl` had finished the first six ranks — all six also in
`blend`'s rescued log.

| rank | `blend` fcl / g / cov | `du_ctl` fcl / g / cov |
|---|---|---|
| 35 | +0.038 / -0.143 / 3.810 | +0.038 / -0.143 / 3.810 |
| 39 | +0.008 / -0.000 / 3.748 | +0.008 / -0.000 / 3.748 |
| 51 | -1.000 / -0.003 / 2.627 | -1.000 / -0.003 / 2.627 |
| 57 | +0.004 / -0.000 / 3.932 | +0.004 / -0.000 / 3.932 |
| 70 | +0.005 / -0.077 / 3.900 | +0.005 / -0.077 / 3.900 |
| 78 | +0.025 / -0.137 / 3.776 | +0.025 / -0.137 / 3.776 |

Six of six agree to every printed digit. This is agreement at 1e-3, the print
precision — not a bitwise claim, which needs the handoff poses. Wall time per
candidate moved substantially between the two runs (rank 57: 4300s against
3502s, rank 51: 3146s against 1235s), so machine contention differed while the
results did not, which is the signature of a deterministic path under varying
load rather than of two runs that happened to be scheduled alike.

Two things follow.

**The repeat leg of the 40-candidate protocol can be dropped on CPU.** Its
purpose is the divergence and bitwise columns, and those measure nondeterminism
that the protocol's GPU/MPS runs had and this path does not. A second repeat here
would return 0.000 and 40/40 by construction. The `compl_ctl` / `compl_5` pair is
consequently 40 candidates once per setting, scored on median iterations, cap
rate, exit distribution and kept.

**`6380f2d`'s parity claim holds on real candidates.** The two trees differ by
that commit, whose two new fields default to IPOPT's own values so that the
acceptable test is unchanged until one is lowered. Six candidates solving
identically across the boundary is a stronger check of that than the harness run,
which exercises one subject written on the spot.

The caveat that matters: this shows the path is reproducible, not that it is
correct, and it says nothing about the GPU path, where the protocol's divergence
numbers came from and where MPS reintroduces the nondeterminism being measured.

## X21 — what a finite `acceptable_dual_inf_tol` costs, and what it buys

`du_ctl` (`ip_acc_du_tol` at IPOPT's 1e10) against `du_1` (1.0), capped-10, same
tree, 2 workers each. Two ranks are complete in both arms at 16:01.

| rank | arm | fcl | max_g_thread | coverage | secs |
|---|---|---|---|---|---|
| 35 | `du_ctl` | +0.038 | -0.143 | 3.810 | 861 |
| 35 | `du_1` | +0.034 | **-0.249** | **3.904** | **3176** |
| 39 | `du_ctl` | +0.008 | **-0.000** | 3.748 | 230 |
| 39 | `du_1` | +0.017 | **-0.146** | **3.852** | **3898** |

`g` here is `max_g_thread`, the largest threading constraint value, where <= 0 is
satisfied and more negative is more margin — it is not the objective. Read that
way the two arms differ in a way the kept/lost label does not show.

**The cost is wall time, and it is large:** 3.7x on rank 35 and 17x on rank 39.

**What it buys is margin.** Both candidates stay kept, and under the finite bound
both land with substantially more threading margin and better coverage. Rank 39
is the sharp case: `du_ctl` exits at `max_g_thread` = -0.000, sitting exactly on
the threading constraint with no margin at all, while `du_1` reaches -0.146.

This is consistent with X16 rather than against it. X16 established that
`acceptable_tol` = 5.0 binds the dual leg, so solves may exit with `inf_du`
anywhere up to 4.97 — and an exit taken that far from stationarity is exactly
where a zero-margin solution comes from. The acceptable test stops at *feasible*;
tightening the dual bound makes the solver keep descending, and the descent goes
into margin.

It also corrects the framing in X16's throughput note. "du_1 and du_p1 returned 0
of 10 in 38 minutes against du_ctl's 2" measured only that the arms are slower.
It is not evidence that the bound is useless; the bound is a **cost/margin
trade**, not a defect.

Weight to give this: **n = 2 shared candidates.** Direction is consistent on both
and the rank-39 margin gap is far too large to be print noise, but two candidates
cannot establish a rate, and neither arm is finished. It does not change the
recommendation — 4x to 17x is prohibitive for a production sweep over hundreds of
candidates — but it makes the knob a deliberate trade rather than a redundancy,
and it is the strongest evidence in this series that the acceptable exit's
distance from stationarity has a physical cost.

## X22 — the platform is a larger lever than most of the settings under test

`repeat_ranks.txt` and `capped10_ranks.txt` overlap on ranks 35 and 39, and both
index the same pool under the same config, so the CPU and GPU baseline runs solve
the same two problems with the same settings.

| rank | CPU `du_ctl` | GPU `compl_ctl_a` |
|---|---|---|
| 35 | fcl +0.038, g -0.143, cov 3.810 | fcl +0.014, g -0.028, cov 3.936 |
| 39 | fcl +0.008, g **-0.000**, cov 3.748 | fcl +0.029, g **-0.196**, cov 3.858 |

Both stay kept, but nothing else carries over: threading margin, FCL clearance and
coverage all move, and they do not move in a consistent direction — rank 35 loses
margin on the GPU while rank 39 gains it. These gaps are larger than the effect of
every setting in the arm series except `nothread`.

**This punctures X21's sharpest claim.** X21 highlighted rank 39 exiting at
`max_g_thread` = -0.000 — feasible with no margin at all — and read it as the
acceptable test's distance from stationarity being paid in margin. On the GPU the
same candidate under the same settings exits at -0.196. The zero-margin exit is
therefore at least partly a CPU-path artifact, not a clean property of the
acceptable test. X21's cost measurement (3.7x and 17x wall time) stands, since
both arms there ran on the same platform, but the margin story needs re-testing on
the GPU before it can carry any weight.

The general point is the one to keep: **cross-platform comparisons in this series
are invalid**, in the same way the two-tree comparison in X15 was. The prior art
is GPU, so the GPU is the reference platform, and any CPU result quoted against
the doc's table — including everything in the arm series, which ran on CPU — is
indicative only.

## X23 — the baseline has moved since the doc's table, so the table is not a reference

`compl_ctl_a` is the baseline configuration at n=40 on `repeat_ranks.txt`, the
protocol's own candidate list, run on the card. The doc's table has a row for the
same configuration, `adopted defaults`, from 2026-09-16. Confirmed first that the
two runs address the same problems: the handoffs share all 40 candidate indices
with `837772_repeat_iptol_a`, so the pool and rank file still index what they did
then, despite the pool changing from pickle to JSON in between.

| | doc, 2026-09-16 | `compl_ctl_a`, today |
|---|---|---|
| median iterations | 1000 | **737** |
| at the 1000 cap | 42/80 (52%) | **11/40 (27%)** |
| exits | — | acceptable 27, cap 11, infeas 2 |
| kept | ~85-88% | 31/40 (78%) |
| median `inf_du` at exit | — | 1.36 |

Same 40 candidates, same nominal settings, and the cap rate halves. Ruled out the
obvious confound: `619cb3f` only *exposed* the threading-reward weight as a
setting, so the reward itself was active and hardcoded in the doc-era runs, and
`lam_thread=1.0` is not a change against them.

What is left is the code between the two dates, and the prime suspect is
`bab0970`, whose effect on conditioning X15 and X16 already measured on the
capped-10 — this is the same direction at n=40 on the protocol's candidates,
which is far better evidence for that fix than anything in the arm series. It is
not isolated here, though: the JSON pool and handoff, `a5420cc` building the
warmup problem the way the solve does, and the rest of the p1angle branch all sit
in the same gap.

**The consequence for the recommendation is procedural and matters more than the
number.** `PHASE2_CONDITIONING.md`'s table cannot be used to score anything run on
today's tree — not the pending `compl_5`, and not the four intervention arms. Every
comparison needs its own control from the same tree, which is what `compl_ctl_a`
is and why the batch was built around one. Median `inf_du` at exit of 1.36 is also
a direct n=40 confirmation of X16: `acceptable_tol` = 5.0 is what bounds the dual
leg, and exits land well inside it.

## X24 — the dual-bound question closes against the tightened bound

`du_ctl` completed 10/10 (handoff written): acceptable 9, cap 1, median 474
iterations, median terminal `inf_du` 1.29, kept 8/10, median 1700s. Kept 8/10
matches `blend` exactly, as X20 predicted. `du_1` was stopped at 4/10 and has no
handoff, so it is log lines only — no iteration counts or exit statuses for it.

Four shared ranks, against X21's two:

| rank | `du_ctl` (1e10) | `du_1` (1.0) |
|---|---|---|
| 35 | +0.038, g -0.143, 861s | +0.034, g -0.249, 3176s |
| 39 | +0.008, g -0.000, 230s | +0.017, g -0.146, 3898s |
| 51 | -1.000, g -0.003, 2701s | -1.000, g -0.002, 3636s |
| 57 | **+0.004 kept**, g -0.000, 3502s | **-1.000 reject**, g **+0.000**, 3534s |

**Kept: `du_ctl` 3/4, `du_1` 2/4.** Rank 57 is the case X21 could not see: under the
tightened bound the solve ends with the threading constraint marginally
*violated* — g positive, where <= 0 is required — and FCL rejects a plan the
control keeps.

The cost structure is also clearer than X21 made it. The 3.7x and 17x multipliers
came from the two candidates where the control exited early (230s, 861s); where
the control already ran long the arms are comparable (2701 to 3636s, 3502 to
3534s). So the tightened bound is not uniformly slower — it *removes the early
acceptable exits* and pushes nearly everything toward the cap, landing all four
candidates in a 3176-3898s band.

**Verdict: keep `ip_acc_du_tol` at 1e10.** X21 framed this as a cost/margin trade
worth considering. It is not a trade: the margin gain appears on 2 of 4, X22
showed that gain does not survive a platform change, and the fourth candidate
loses a plan outright. Paying 1.4x to 17x to lose yield is not a trade-off, and
the knob's value is as a diagnostic — it is what proved the acceptable exit's
early stopping is doing real work — rather than as a production setting.

Weight: n=4, one platform, and `du_1` has no handoff. Enough to close the question
against the change, not enough to support any positive claim.

## X25 — both throughput candidates die on the control's own diagnostics (2026-09-22 17:23)
> **CORRECTED BY X26 — both verdicts below are withdrawn.** This entry reads two
> trailing-window statistics as statements about whole trajectories, and both
> inferences are false on direct measurement. "10 of 11 caps are infeasible at
> the cap" does not mean those solves never reached feasibility: they reach it at
> iteration 24 and hold runs of 60. "The median solve is feasible for 8 of its
> 570 iterations" reports the *trailing* run as if it were the total; the total
> is 245 of 570. The 98.5%/1.5% split and the "adopt neither" verdict both rest
> on those errors, and this entry's overturning of X10 and X11 does not stand.
> The measurements in the tables are correct; the readings taken from them are
> not.

Ran against `837772_g40_compl_ctl_a` alone, no card needed. Two false starts
first, and they matter for anything that reads `diag_hist` later.

**`diag_hist` does not record what the acceptable test uses.** `inf_pr`/`inf_du`
arrive from IPOPT's `intermediate_callback`, so they are the *scaled* iteration-log
values, while the acceptable test runs on unscaled quantities; `mu` is the barrier
parameter, not complementarity. Comparing either to `ip_cvtol` or
`ip_acc_compl_tol` is a category error, and a first pass here did exactly that.
The one unscaled feasibility measure recorded is `group_min`/`group_nviol`, built
from `get_current_iterate(scaled=False)`. Everything below uses those, thresholded
at the real `ip_cvtol` of 1e-4; a strict-zero threshold gives identical counts.

**Rank 1, `ip_acc_compl_tol` 0.01 -> 5.0: at most 1 of 40 solves can move.**

| exit | n | feasible on all of the last `ip_acc_iter` iterations |
|---|---|---|
| acceptable | 27 | 26/27 |
| cap | 11 | **1/11** |
| infeas | 2 | 0/2 |

Ten of the eleven caps are not feasible at the cap — five still hold strictly
violated constraint entries at the final iterate, worst margin from -0.012 to
-4.911, and the rest flicker in and out inside the window. They are still looking
for a feasible point, and no acceptance tolerance reaches them. Only rank 11356 is
feasible throughout with margin in hand (+0.038), which makes it the single
candidate a complementarity bound could plausibly be gating.

**Rank 2, `ip_acc_iter` 8 -> 3-5: binding, and worth under 2%.** On acceptable
exits the trailing feasible run is median 8, and exactly 8 on 17 of 27 — so
`acc_iter` genuinely is what ends those solves. But the median solve is 570
iterations and is feasible for 8 of them: **1.5% of the solve**. Dropping to 3
saves 5 iterations where it binds. Six of 27 have runs above 12 (up to 91), where
another leg gated the exit and lowering `acc_iter` saves nothing at all.

**The structural finding, which is worth more than either knob.** A Phase-2 solve
spends ~98.5% of its iterations *reaching* a feasible point and ~1.5% proving the
point acceptable. Throughput is therefore not in the acceptance tolerances — the
whole family is tuning the last 1.5%. It is in how fast the solver reaches
feasibility, which is a formulation and basin question, not a tolerance one.

**Verdict: adopt neither.** `ip_acc_compl_tol` stays at IPOPT's 0.01 and
`ip_acc_iter` at 8. This also retires the `throughput` batch in
`resume_gpu_arms.sh` — its two arms now have predicted effects of at most 1
converted cap and ~1% wall time, which the benchmark cannot resolve.

## X26 — the control is bitwise reproducible here, `nothread` confirms on the card, and X25's structural finding is wrong (2026-09-23 08:16)

Three results from one GPU batch on `repeat_ranks.txt`, all scored against
`compl_ctl_a` with `scratch/estimators/compare_protocol.py`. Note that
`compl_ctl_a`/`compl_ctl_b` are the **baseline pair** — both run with no
overrides, byte-identical headers. The `compl_` prefix is the batch they were
born in, not their configuration.

**The reproducibility property holds on this tree.** `compl_ctl_b` against
`compl_ctl_a`: pose divergence 0.000, 40/40 bitwise identical, 100% label
agreement, and every summary figure matches — acceptable 27, cap 11, infeas 2,
median 737 iterations, kept 31/40. The pre-registered doubt was that the doc
measured this when the median solve ran to the 1000 cap, where two runs take the
same number of iterations, while here the median exits acceptable at 737, so a
reduction-order difference could flip which iteration first satisfies the
8-iteration window. It does not. `ip_hist`, `ip_acc_tol`, `ip_cvtol` and
`drop_dead_rows` now rest on a same-tree measurement rather than the 2026-09-16
table.

**`nothread` reproduces on the reference platform, larger than on CPU.**

| | `compl_ctl_a` | `nothread_a` |
|---|---|---|
| kept | 31/40 | **21/40** |
| threading satisfied, FCL rejects | 8/40 | **19/40** |
| exits | acc 27, cap 11, infeas 2 | acc 28, infeas 9, cap 3 |
| median iterations | 737 | 574 |
| median seconds | 43 | 27 |

The flips carry the mechanism: 1274, 2020 and 2161 exit with threading satisfied
by a hair (g -0.002, -0.010, -0.007) and `fcl -1.000`, a hard collision, where
the control lands the same candidates at -0.106, -0.103, -0.164 with FCL
positive. Removing the reward parks the solve on the constraint boundary and
calls it done — faster, and wrong. X19's CPU reading (-6/10 kept, 5/10
model-optimistic) was directionally right.

**X25's structural finding is wrong, and the error is an inference, not a
measurement.** Measuring where feasibility is *first* reached, from the same
unscaled `group_min`:

| exit | n | median first feasible | % of iterations feasible | feasible runs | longest run | median nit |
|---|---|---|---|---|---|---|
| acceptable | 27 | 33 | 48.9% | 30 | 38 | 572 |
| cap | 11 | **24** | **53.0%** | **62** | **60** | 1001 |
| infeas | 2 | 26 | 55.8% | 40 | 44 | 606 |

No solve in the set fails to reach feasibility. A capped solve reaches it at
iteration 24, is feasible for 53% of its iterations, crosses in and out 62
times, and holds a sustained feasible run of 60 iterations against an
`acc_iter` of 8. X25's table — 10 of 11 caps infeasible on the last
`acc_iter` iterations — is correct and still stands. What does not follow, and
what X25 asserted, is that those solves are "still looking for a feasible
point". Infeasible *at the final iterate* is not the same claim as never
feasible.

**What this overturns.** The 98.5%/1.5% split is wrong: solves reach feasibility
in roughly 4% of their iterations and then fail to *hold* it while improving the
objective. `ip_acc_compl_tol` is therefore un-retired — a cap that satisfies the
constraint-violation leg for 60 consecutive iterations was stopped by a different
leg, which is exactly the complementarity hypothesis `PHASE2_CONDITIONING.md`
pre-registered. "Throughput is a formulation question, not a tuning one" may
still be true but is no longer established.

`diag_hist` cannot say which leg gated those runs: `mu` is the barrier parameter
and `inf_pr`/`inf_du` are scaled. Settling it needs the arm run, not more
analysis of this payload.

**Method note.** Kept is a binary gate on plans clearing by a median +0.021 mm,
so it cannot support a directional claim about a formulation change in either
direction — including the absence of one. X19 used that to call `smooth` and
`angle` noise; this file used it to retire them. That reasoning only licenses
"unresolved", not "no effect". Iterations to first feasible, feasible fraction,
crossing count and the margin distribution are continuous and are what a
better-conditioned parameterization should move.

## X27 — how a solve holds feasibility, and X25's second arithmetic error (2026-09-23 08:29)

Follows X26. Measured with `scratch/estimators/compare_protocol.py` on
`compl_ctl_a`, from `group_min`, the only unscaled feasibility record.
Definitions, since raw counts mislead here: **reach** is the iteration of the
first feasible iterate; **held** is the feasible fraction of the iterations after
that, which drops the pre-feasible transient; **osc/100** is feasible runs per
100 iterations, normalised because a 1000-iteration cap crosses more often than a
200-iteration solve without tracking the constraint surface any worse;
**runlen** is the mean feasible run.

| arm | reach | held | osc/100 | runlen | feasible at exit | kept | model-optimistic |
|---|---|---|---|---|---|---|---|
| `compl_ctl_a` | 28 | 60% | 7.5 | **7.1** | 85% | 31/40 | 8 |
| `nothread_a` | 46 | 40% | 9.3 | 4.0 | 92% | 21/40 | 19 |

**The mean feasible run is 7.1 iterations against an `acc_iter` of 8.** The solve
repeatedly achieves feasibility in runs just short of the window the acceptable
test demands, so it exits at roughly the first run that survives to 8.

Fraction of iterations each constraint group spends violated:

| arm | thread | probe_pair | probe_fixture | brain | arc_sep | ml_sep |
|---|---|---|---|---|---|---|
| `compl_ctl_a` | 20.6% | **39.1%** | 12.9% | 0.9% | 0.0% | 0.0% |
| `nothread_a` | 34.1% | **59.0%** | 20.3% | 1.3% | 0.0% | 0.0% |

`probe_pair` is the group the solver cannot hold, not threading. Removing the
threading reward makes every group worse, which is a second reading of why
`nothread` converges cleanly to the wrong problem.

**X25's second error, on the 27 acceptable exits.**

| | |
|---|---|
| median iterations | 570 |
| median **total** feasible iterations | **245 (43% of the solve)** |
| median **trailing** feasible run | 8 |
| median iteration of the first feasible run reaching 8 | **77** |
| median iteration of the first feasible run reaching 3 | 40 |

X25 wrote "the median solve is 570 iterations and is feasible for 8 of them:
1.5% of the solve". The 8 is the trailing run; the total is 245. The figure is
wrong by a factor of about 30, and `ip_acc_iter` is un-retired with
`ip_acc_compl_tol`.

**What actually gates the exit.** The feasibility leg is satisfied for 8
consecutive iterations by iteration 77, and the median solve exits at 570. So
the constraint-violation leg qualified 490 iterations before the exit, and some
other leg — the overall NLP error bound of `ip_acc_tol` = 5.0, or complementarity
at 0.01 — was unsatisfied for roughly 87% of the solve. That is where the
throughput is, and it is the hypothesis X10 and X11 measured directly before X25
displaced them.

**Not concluded.** Which leg binds over those 490 iterations is not measured
here: `diag_hist` records no unscaled complementarity, and `mu` is the barrier
parameter. `compl5_a` at n=40 is the direct test.

## X28 — the IPOPT options reference, read at last, and three errors it corrects (2026-09-23 08:38)

Nothing in this investigation had consulted IPOPT's own options documentation;
the acceptable test's semantics were reconstructed from `settings.py` and the
prior art. Reading the reference corrects three things.

**`ip_acc_iter = 8` is not "IPOPT's 8".** IPOPT's `acceptable_iter` default is
**15**. X25 and `RECOMMENDATION.md` both call 8 the stock value; it is a project
setting already *below* default, so 8 → 3 continues a move already made rather
than departing from stock.

**The acceptable test has five legs, not four.** `acceptable_obj_change_tol`
(default 1e20) — "if the relative change of the objective function is less than
this value, this part of the acceptable tolerance termination is satisfied". We
do not set it and at 1e20 it is inert, so no conclusion moves, but "all four
legs" appears throughout these documents and is wrong.

**`acceptable_tol` is on the *scaled* error; the other three legs are unscaled.**
The reference is explicit: `acceptable_tol` "determines which (scaled) overall
optimality error is considered to be 'acceptable'", while
`acceptable_constr_viol_tol`, `acceptable_dual_inf_tol` and
`acceptable_compl_inf_tol` each bound a "max-norm of the (unscaled)" quantity.
X25's trap note — that comparing `diag_hist` to any tolerance is a category error
— is therefore too broad. It holds for the three unscaled legs. It does not hold
for `acceptable_tol`, whose leg is exactly what the scaled iteration-log values
feed.

**None of the twelve options this project sets is deprecated**, checked against
the reference: `tol`, `constr_viol_tol`, `acceptable_tol`, `acceptable_iter`,
`acceptable_constr_viol_tol`, `acceptable_dual_inf_tol`,
`acceptable_compl_inf_tol`, `limited_memory_max_history`,
`hessian_approximation`, `mu_strategy`, `max_iter`, `resto.acceptable_iter`.

**A documented knob nothing has tested.** *(QUALIFIED BY X30: the acceptable
leg was tested, in the solver bake-off that chose IPOPT. Loosening it was
measured to return colliding plans, so it is a rejected candidate rather than an
unexplored one.)* `ip_cvtol` drives both
`constr_viol_tol` and `acceptable_constr_viol_tol`, at 1e-4 against IPOPT's
acceptable default of 0.01 — 100x tighter. `settings.py` gives the reason
("whose IPOPT default of 1e-2 would accept 0.01 mm of overlap"), so the coupling
is deliberate. It is still load-bearing on X27: the 7.1-iteration mean feasible
run is measured at 1e-4, and a looser acceptable-leg threshold would lengthen
those runs directly. Decoupling the two is a throughput candidate that has never
been run, and its cost is returning plans with up to 0.01 mm of model overlap.

## X29 — `acceptable_compl_inf_tol` 0.01 → 5.0 converts every cap and gains six plans (2026-09-23 08:38)

> **CORRECTED BY X31.** This entry scores the arm on counts and never looks
> at the objective. With the objective measured, `compl5` is a **trade**: it
> gains six feasible plans and loses 0.92 of objective on 27 of the 31
> candidates both arms solve. The premise that capped solves were wasting
> iterations is also false — `waste` is 0%.

`compl5_a` at n=40 on `repeat_ranks.txt`, GPU, against `compl_ctl_a`. This is
the arm X25 retired and X26/X27 un-retired, and the hypothesis
`PHASE2_CONDITIONING.md` pre-registered.

| | `compl_ctl_a` | `compl5_a` |
|---|---|---|
| exits | acceptable 27, cap 11, infeas 2 | **acceptable 40** |
| median iterations | 737 | **164** |
| median seconds | 43 | **6** |
| kept | 31/40 | **37/40** |
| model-optimistic | 8 | **3** |
| feasible at exit | 85% | **100%** |
| reach / held / runlen | 28 / 60% / 7.1 | 28 / 63% / 6.4 |

**Every cap and both infeasible exits become acceptable exits**, wall time drops
about 7x, and yield goes *up*. All six kept-label flips run the same way —
control rejects, `compl5` keeps — with none lost. Among them is idx 1684, which
X25 cited as proof that caps were hopelessly infeasible at `g +4.911`; under the
loosened bound it lands at `g -0.137` and is kept. Complementarity was the
blocking leg, as X10 and X11 measured directly on one candidate and as the prior
art predicted.

**Not concluded: reproducibility.** The 1.942 divergence and 4/40 bitwise above
are `compl_ctl_a` against `compl5_a` — two different settings, so that is effect
size, not run-to-run reproducibility. `compl5_b` is needed for the latter, and
the doc's table has `acceptable tol 5` at divergence 4.536 and 0/40, so the
expected cost of this change is the bitwise determinism the control has at 40/40.
That is the trade `RECOMMENDATION.md` already named a policy question rather than
a tuning one, now with a measured price on both sides.

**Also confirmed here: the rebase is clean.** `compl_ctl_c`, the baseline re-run
from the merged checkout, is bitwise identical to `compl_ctl_a` at 40/40 with
every summary figure equal, so results from the former worktree carry over.

## X30 — the Hessian and the violation tolerance were both chosen by measurement (2026-09-23 08:45)

Two of the solver settings this investigation has been reasoning about from
first principles were in fact decided by a head-to-head run when IPOPT was
adopted. The record is the solver bake-off on the tuned-545 set, probe
`scratch/ipopt_exact_vs_lm.out`, and it corrects X28 on one point and closes a
question this investigation had not asked out loud.

**`hessian_approximation = limited-memory` beat the exact Hessian on every
axis.** Thirteen candidates — eleven that had stalled under BFGS plus two
feasible controls — were solved both ways on CPU:

| | exact Hessian @400 | limited-memory @1000 |
|---|---|---|
| stalled candidates rescued | 8 of 11 | **9 of 11** |
| regressions | lost 1221; drove control 506 from `+0.200` to `-1.000` | none |
| seconds per candidate | 255–798 | **29–166** |
| GPU | broken — the constraint HVP kernel crashes XLA autotuning | runs |

The reason is structural rather than incidental: this NLP is nonconvex — soft-min
over clearances, OBB-SAT, the well SDF — so the exact Lagrangian Hessian is
indefinite away from the optimum. IPOPT regularizes it, and that regularized
direction is worse than L-BFGS's guaranteed positive-definite approximation. The
bake-off's own summary is that restoration does more for reaching feasibility
from an infeasible start than curvature does, which is the same conclusion X26
and X29 reached from the other end: what gates these solves is the acceptance
test, not the quality of the step.

This also gives `ip_hist` its weight. With no exact Hessian,
`limited_memory_max_history` is the only curvature information the solver holds,
which is why 60 — above `n` at 44–45 — is not the idle setting it looks like.

**X28's `ip_cvtol` candidate is rejected, not untested.** X28 listed decoupling
`acceptable_constr_viol_tol` from `constr_viol_tol` as a throughput candidate
nothing had run. The bake-off had run it, and recorded the 1e-2 default as
*dangerous*: the constraint vector is mixed-unit — body and voxel rows are
mm-native at gain 1, OBB rows are scaled by 100 — so `constr_viol_tol` is set for
the mm rows at 1e-4, and accepting 1e-2 there means accepting 0.01 mm of overlap,
which FCL then rejects as a collision. That is the same failure X29 measures as
the model-optimistic count, and X29 moved it the right way (8 → 3) by loosening a
*different* leg. Loosening this one would move it back, by construction.

The general lesson is the one the header states: before a setting is called
untested, check whether it was chosen. Both of these were, in a run that
predates this investigation and is not in `STATUS.md`.

## X31 — the solves waste no iterations, and `compl5` buys feasibility with objective (2026-09-23 08:47)

`compl_ctl_a` and `compl5_a` rescored on measures aimed at what the solver is
supposed to do rather than at how many plans survive the keep gate: find a
feasible point if one exists, get a good objective there, and stop once it has.

Two measures are new. `settle` is the earliest iteration after which the solve is
feasible and its objective stays within 1e-3 relative of its final value for the
rest of the run — the first point it could have stopped without changing what it
returned. `waste` is the fraction of the solve past that point. Both mask
restoration iterations, where the callback's `obj_value` is the restoration
subproblem's objective and not f(x).

| | `compl_ctl_a` | `compl5_a` |
|---|---|---|
| median iterations | 737 | 164 |
| `settle` | 742 | 165 |
| **`waste`** | **0%** | **0%** |
| strictly feasible | 31/40 | 37/40 |
| model-optimistic | 8 | 3 |
| median final objective | **-5.4488** | -4.3237 |
| paired objective change | — | **+0.9186 median, +2.6869 p90, worse on 27 of 31** |

**`waste` is 0% everywhere, so there are no wasted iterations to reclaim.** The
objective is still descending when each solve exits. The capped solves are not
spinning on a converged answer; they are still improving it. This retires the
whole class of stopping-rule interventions as a source of *gain*: any of them now
costs objective by construction.

**`compl5` is therefore a trade, not a win.** It gains six strictly feasible
plans and gives up 0.92 of objective on almost every candidate both arms solve.
X29 reported only the gain because it scored counts.

**A hypothesis the flips support.** The six plans `compl5` gains were hard
failures for the control — `fcl -1.000` — at tiny model violations (`g -0.006`,
`-0.008`, `-0.078`). Read with the model-optimistic count falling 8 to 3, the
reading is that running longer drives plans harder against a model boundary that
sits slightly outside FCL's, so the extra objective is partly bought by packing
into collision. If that holds, the defect is model fidelity and stopping early
only masks it. Untested: it predicts the control's losses sit at systematically
smaller FCL clearance than its keeps, which the handoffs can answer without a
run.

**Method note.** `obj` had to be masked before any of this could be said. The
callback records `obj_value` verbatim and `restoration` is `alg_mod`, so entries
where it is 1 measure the restoration subproblem instead. Masking changed none of
the figures here — the median restoration fraction is 0.2% — but 2 of 40 control
solves end inside restoration, so their raw final `obj` is not an NLP objective
at all. Checked rather than assumed.

## X32 — `spin_param=angle` improves feasibility and objective together (2026-09-23 08:55)

`angle_a` at n=40 on `repeat_ranks.txt`, GPU, against `compl_ctl_a`. This is the
arm X19 dropped on kept at n=10 and the fix `PHASE2_CONDITIONING.md` prescribes
for a defect in the current default.

| | `compl_ctl_a` | `angle_a` |
|---|---|---|
| exits | acceptable 27, cap 11, infeas 2 | acceptable 33, infeas 6, cap 1 |
| strictly feasible | 31/40 | **33/40** |
| median final objective | -5.4488 | **-5.9910** |
| paired objective change | — | **-0.1143 median, worse on 9 of 26** |
| `reach` | 28 | **14** |
| feasible at exit | 85% | **100%** |
| model-optimistic | 8 | 7 |
| median iterations / seconds | 737 / 43 | 431 / 25 |
| thread group violated | 20.6% | **10.7%** |

**It improves both goals at once**, which no other arm in this investigation has
done: more feasible plans *and* a better objective, with iterations to first
feasible halved. `compl5` by contrast bought feasibility with objective (X31).

**The improvement is where the mechanism predicts.** `spin_deg_from_sxy` is
`arctan2(sy, sx)` (`clearance/poses.py:127`), reached on every objective
evaluation in both phases, and the `(sx, sy)` box is `+/-1.1` on both axes
(`pipeline/fixtures.py:45-46`), so the singular origin is interior and nothing
but `unit_circle_penalty` keeps the iterate away from it. Spin is what drives a
probe through its bore, and **thread** is the one constraint group whose
violated-iteration share fell — 20.6% to 10.7%. `probe_pair` and `probe_fixture`
rose, so this is a redistribution toward the group the reparameterization acts
on, not a uniform improvement.

**Three qualifications.**

- **It loses 5 plans and gains 7**, net +2. Not uniformly better per candidate.
  The five losses are at small model violations (`g -0.001` to `-0.086`).
- **The 6 "infeasible" exits are not failures.** All six ended inside
  restoration with a feasible last iterate; five of six are strictly feasible
  with positive FCL clearance. The status code describes the restoration
  subproblem. The control has 2 of the same. Do not read `solver_status == 2`
  as a lost plan without checking the iterate.
- **Bitwise 0/40 and divergence 5.560 are expected and are not a determinism
  result.** The arms solve in different coordinates, so poses are not comparable
  run-to-run. Scoring this the way `PHASE2_CONDITIONING.md`'s table scores a
  tolerance change would reject the best arm measured here. Reproducibility for
  `angle` needs `angle_b`, against `angle_a`.

**Unresolved:** whether the gain survives a repeat, and whether it holds in
Phase 1, where the setting is also wired but by an entirely different mechanism
and has never been measured (I2).

## X33 — `smooth_thread_reward` regresses Phase 2 on every goal measure (2026-09-23 08:55)

`smooth_a` at n=40 on `repeat_ranks.txt`, GPU, against `compl_ctl_a`, scored
beside `angle_a` from X32.

| | `compl_ctl_a` | `smooth_a` | `angle_a` |
|---|---|---|---|
| strictly feasible | 31/40 | **27/40** | 33/40 |
| median final objective | -5.4488 | **-5.3511** | -5.9910 |
| paired objective vs control | — | **+0.0270, worse on 14 of 24** | -0.1143, worse on 9 of 26 |
| `reach` | 28 | **40** | 14 |
| feasible at exit | 85% | **78%** | 100% |
| model-optimistic | 8 | **10** | 7 |
| caps | 11 | **15** | 1 |
| **thread group violated** | 20.6% | **26.9%** | 10.7% |

**Six independent measures move the wrong way**, so unlike X19's n=10 verdict this
is directionally negative rather than unresolved. Head to head, `angle` beats
`smooth` by -0.3118 median objective, worse on only 4 of 23, and 33 feasible
against 27.

**It fails on the group it acts on.** The reward's whole subject is threading,
and `smooth_a` spends *more* of its iterations violating the thread group than
the control does. A mechanism that would explain it, untested: the smooth branch
replaces `jnp.maximum(0.0, worst)` with `softplus(beta*worst)/beta`
(`objectives/rewards.py:53-64`), and softplus is strictly positive at negative
slack where the hard gate is exactly zero. The smooth form therefore pays a
reward for being *near* the bore while still outside it, weakening the pressure
to get inside. If that is right, the fix is a gate that is smooth but still zero
below zero, not a return to the hard clamp.

**This does not settle `smooth` for Phase 1, where it is also wired and has never
been run.** The argument for smoothing is strongest exactly where it is untested:
RProp steps on the sign of each partial derivative, so a kink reverses its step
and decays that coordinate's step size, whereas IPOPT uses magnitudes and can
average one out. A Phase-2 regression is evidence about IPOPT, not about RProp.
I2 is the test.

**Same qualification as X32:** 0/40 bitwise at divergence 6.303 is expected for a
changed objective and is not a determinism result.
