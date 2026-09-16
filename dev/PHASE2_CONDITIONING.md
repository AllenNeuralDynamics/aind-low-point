# Phase-2 solver conditioning — measured state, IPOPT behaviour, open defects

**Status:** findings as of 2026-09-16. Companion to `dev/PIPELINE.md` (what runs
today) and `dev/PIPELINE_PLAN.md` (the to-do, which carries the ordered next
steps from this work).

Phase 2 is the constrained pose refinement in
`src/aind_rutter/optimization/pipeline/phase2_ipopt.py`, built by
`make_phase2` in `src/aind_rutter/optimization/objectives/phase2.py`. It solves,
per candidate, a nonlinear program of **45 continuous variables and 402
inequality constraints, with no equality constraints**, using IPOPT 3.14.16
through cyipopt 1.7.0 with the MUMPS 5.7.3 linear solver and a limited-memory
BFGS Hessian approximation. Objective gradient and constraint Jacobian are dense
and come from JAX.

Variables are `n_arcs` arc angles in degrees followed by six per probe: ml in
degrees, the pair `(sx, sy)` parameterizing spin, two offsets in millimetres, and
depth past target in millimetres.

Every claim below is labelled **[measured]** (run on this machine),
**[docs/source]** (IPOPT documentation or 3.14 source), or **[unverified]**.

---

## What this build can and cannot do — [measured]

Probed by solving a one-variable problem per option against the project venv.

| capability | state |
|---|---|
| `linear_solver=mumps` | works (the default here) |
| `ma27`, `ma57`, `ma86`, `ma97`, `pardiso` | registered, return −12 at runtime: no library present |
| `pardisomkl`, `spral` | not valid option values; not compiled in |
| `linear_system_scaling=mc19` | −12 — MC19 absent |
| `nlp_scaling_method=equilibration-based` | −12 — needs MC19 |
| `linear_system_scaling=slack-based` | **works, no HSL required** |
| `nlp_scaling_method=user-scaling` + `set_problem_scaling` | works end to end |
| prefixed options, e.g. `resto.acceptable_iter` | accepted |

**No rebuild is required to add HSL.** Every HSL entry is described as loading
the routine from a library at runtime, and IPOPT's install documentation states
that obtaining a solver after compiling needs no recompile — drop a `libhsl.so`
in place and set `hsllib`. Acquiring any HSL library also unlocks MC19, and with
it both `linear_system_scaling=mc19` and `nlp_scaling_method=equilibration-based`.
Coin-HSL Archive (MA27, MC19) is free for personal use; the Full set (MA57,
MA97) is free for academic use.

---

## Where solve time goes — [measured]

IPOPT's own timing statistics, one candidate, 200 iterations, CPU platform:

| | wall seconds | share |
|---|---|---|
| NLP function evaluations | 123.2 | 95% |
| IPOPT internals, everything else | 3.7 | 3% |
| of which the entire linear-solver stack | 1.3 | ~1% |
| of which matrix factorization | 0.000 | — |

At this size the augmented system is roughly 450×450 and factorizes in time too
small to register. **A different linear solver is not a speed lever**; its only
argument is pivoting robustness. `LinearSystemScaling` also reads 0.000, which
independently confirms that no KKT scaling is active.

The cost is 195 Jacobian evaluations for 200 iterations, each a forward-mode
differentiation over 45 inputs, plus line-search trial evaluations: 850 objective
evaluations for those 200 iterations, about 4.25 trials per iteration. Removing
the clearance reward drops that to 622, or 3.1 trials per iteration, though wall
time barely moves because Jacobians dominate.

---

## Constraint scaling — [measured] + [docs/source]

Hand-picked gains multiply six constraint categories in
`src/aind_rutter/optimization/sdf/kernels.py`: 1.0 for the millimetre-native
voxel-SDF rows, 100.0 for the three OBB rows. Row infinity-norms at a starting
pose, with the gains divided back out to expose each category's native
sensitivity:

| category | gain | ‖∇c‖∞ | gain that would equalize to body_body |
|---|---|---|---|
| thread | 1 | 2.66 | 0.59 |
| pair/body_body | 1 | 1.24 | 1.00 |
| pair/body_shank_obb | **100** | 91.6 | 1.25 |
| pair/shank_shank | **100** | 90.6 | 1.49 |
| fixture/obb | **100** | 88.3 | 2.17 |
| arc_sep, ml_sep | 1 | 1.00 | 1.49 |
| pair/body_shank_corners | 1 | 0.90 | 1.25 |
| fixture/body | 1 | 0.86 | 1.89 |
| brain | 1 | 0.79 | 2.39 |

Natively every category lies within 4.1× of every other, and the OBB rows sit in
the middle of that range. Equilibration would give them factors near 1–2. As
IPOPT receives them the spread is **117×**, and dropping the gains to 1 would
make it 3.4×.

**IPOPT's automatic scaling does not undo this.** The default
`nlp_scaling_method=gradient-based` computes `d_j = min(1, 100/‖∇c_j‖∞)` at the
starting point [docs/source]. Three consequences, all confirmed by measurement:

- The `min` makes it **one-sided** — steep rows are capped, shallow rows are
  never lifted. Only 27 of 695 rows exceed the cutoff, so 96% of the problem is
  scaled by exactly 1.0 and the 117× spread survives untouched.
- The gained rows land at 88–92, just under the cutoff of 100. Even if they
  cleared it, capping them at 100 would leave 100/0.79 ≈ 127×.
- **Variables are never scaled by any automatic method** — the gradient-based
  scaler sets `dx = NULL` in source. The measured 10.3× column-norm spread
  across degrees, millimetres and the spin coordinates is reachable only through
  `user-scaling`, or `equilibration-based`, which needs MC19.

Column norms by variable kind [measured]: `off_A_mm` 280, `off_R_mm` 232,
`spin_sy` 155, `past_mm` 131, `spin_sx` 59, `arc_ap_deg` 38, `ml_deg` 27.

### The gains are not a pure scaling device

Mathematically `g(x) ≥ 0 ⟺ 100·g(x) ≥ 0`, so a gain cannot change the feasible
set. But `constr_viol_tol` is an absolute test applied to the **unscaled**
constraint, meaning the function as supplied, after the gains [docs]. So
`constr_viol_tol=1e-4` bounds the millimetre rows at 1e-4 mm and the gain-100 OBB
rows at **1e-6 mm**. Changing a gain silently changes the physical feasibility
standard on those rows by the same factor.

`constr_viol_tol`, `dual_inf_tol` and `compl_inf_tol` are all checked on unscaled
quantities, so adding `g_scaling` through `user-scaling` does **not** move the
feasibility gate [docs].

---

## Why solves exit "locally infeasible" — [docs/source]

Status 2 is thrown from two sites in IPOPT 3.14, and both compare the original
problem's primal infeasibility against a multiple of **`tol`, not
`constr_viol_tol`**. It fires when the restoration *subproblem* reaches its own
stopping criterion while the original infeasibility remains above roughly
`100 × tol`. Restoration is never required to prove anything about the original
problem; it only has to stop making progress on its own KKT system.

At `tol=1e-6` that bar is 1e-4, tightening once to 1e-6. Phase 2's engineering
tolerance is 1e-4 mm and real clearances at accepted poses are about 0.02 mm, so
**IPOPT judges infeasibility against a bar two orders tighter than anything the
pipeline cares about.** IPOPT's author left a disabled heuristic in
`IpIpoptData.cpp` noting the restoration tolerance "became too tight by default…
I originally probably put it in to avoid that a claim of infeasibility is made
prematurely."

This is why status 2 is not a reliable failure signal here: in one 40-candidate
run, 24 of 33 status-2 exits produced plans that passed independent FCL
validation. The pipeline already ignores it — `kept` is `fcl_keep and
thread_keep` and `solver_status` is only recorded — which is the correct
handling and should stay that way.

### The acceptable path cannot currently fire

`acceptable_iter=25` and `acceptable_constr_viol_tol=1e-4` are set, but
`acceptable_tol` is left at its default **1e-6**. All five acceptable criteria
must hold simultaneously and the counter resets on any miss [source], so with
L-BFGS stalling the scaled error around 1e-2–1e-4 there is never one acceptable
iterate, let alone 25 consecutive. Status 1 therefore never appears in the logs.

Note the trap: prefixed option lookups fall back to the unprefixed value
[source], so loosening `acceptable_tol` globally also loosens the restoration
subproblem's exit — the branch that throws status 2. Pin it with
`resto.acceptable_iter=0`.

### Measured: status 2 fires at strictly feasible points

An instrumented run (`print_level=5`, `print_info_string="yes"`) over five
candidates that exit status 2 under production defaults and three that hit the
iteration cap under memory 60 with no bonus settles the mechanism.

- **The returned point satisfies every constraint.** Re-solving candidate 13304
  and evaluating the constraints at the returned `x` gives a minimum slack of
  **+0.0205 mm over all 402 rows, none violated**, while IPOPT reports
  "Converged to a point of local infeasibility".
- **The displayed and the tested infeasibility are different quantities.** With
  `inf_pr_output="internal"` the column reads 1.5e-01 through the regular
  iterations and 0.4–1.3 during restoration, while the final summary's
  `Constraint violation` stays exactly 0. The restoration throw site tests the
  internal ‖d(x) − s‖ against about `100 × tol`, so the verdict comes from slack
  variables that have not caught up to the constraint values, not from geometry.
- **Feasibility arrives early; stationarity never does.** `inf_pr` reaches 0
  within tens of iterations and holds for hundreds, while dual infeasibility
  stalls at 0.64–3.05 against a `tol` of 1e-6.
- **The line search collapses at that stall.** `ls` climbs to 20–29 backtracks
  with `alpha_pr` between 1e-7 and 1e-10 and `||d||` at 1e2–1e3; restoration is
  entered afterwards and runs 18–28 iterations.
- **No regularization thrash.** Across eight logs, 0–7 iterations out of
  400–1000 carry a non-dash `lg(rg)`.

Consequence for the acceptable criteria: the Overall NLP error equals the dual
infeasibility, so at 0.64–3.05 an `acceptable_tol` of 1e-3 cannot fire. Making
that path reachable needs a value near 5, which retires the stationarity test and
rests entirely on the independent FCL gate. Defensible, since `kept` already
ignores `solver_status`, but it is a deliberate choice rather than a tweak.

### Measured: what the tolerance settings actually buy

Six candidates that exit status 2 under production defaults, re-solved under each
setting and scored by FCL rather than by exit code
(`scratch/estimators/tolerance_probe.py`):

| arm | median iterations | mean objective | FCL kept |
|---|---|---|---|
| baseline | 509 | −9.45 | 6/6 |
| `acceptable_tol=1e-3` | 682 | −9.72 | 6/6 |
| `acceptable_tol=5` | 100 | −9.32 | 6/6 |
| `tol=1e-4` | 535 | −9.76 | 5/6 |
| `tol=1e-4` + `acceptable_tol=5` | 104 | −9.34 | 6/6 |
| `acceptable_tol=5` + `acceptable_obj_change_tol=1e-5` | 825 | −9.75 | 6/6 |

- `acceptable_tol=1e-3` never fires on any candidate, as predicted from the
  0.64–3.05 error range. `acceptable_tol=5` fires on every one and cuts
  iterations roughly fivefold.
- **35 of 36 solves end FCL-clear.** The one failure is `tol=1e-4` alone on
  candidate 2426. No setting compromised feasibility.
- The objective cost of stopping early is about 0.13 on the mean, which is **not
  resolvable at this sample size**: run-to-run nondeterminism moves a single
  candidate's baseline objective by up to 1.7 — candidate 11356 gave −10.077 in
  one probe and −8.328 in another under identical settings. Any per-arm objective
  claim needs the 40-candidate benchmark, not six candidates.
- `acceptable_obj_change_tol=1e-5` does not deliver the targeted stop it promised:
  the relative objective keeps changing by more than that, so the arm runs to
  800–1000 iterations and reaches the acceptable exit in only 3 of 6.

### `tol=1e-6` against float32 derivatives

Every entry point in `make_phase2` casts to float32, whose epsilon is 1.2e-7, and
collision grids are stored as bfloat16, yet IPOPT is asked for `tol=1e-6`. That
requests convergence about one digit above the noise floor of our own
derivatives, and is an independent source of run-to-run variation.

---

## Formulation defects

**The clearance reward carried four defects at once** and its removal
(`LAM_CLEAR=0`) is what made solves exactly reproducible. It is redundant with a
hard constraint that already enforces 0.2 mm; its `max(slack,0)` gate is C0 but
not C1; in production it is built from the **hard** minimum over sampled surface
points, so its gradient is one sample's and jumps when the closest sample
changes; and that argmin over near-tied float32 values on GPU is exactly where
XLA's nondeterministic reductions flip an answer. IPOPT itself has no explicit
randomness, so run-to-run variation enters through evaluation.

Its legitimate purpose — keeping the gradient meaningful where coverage is
locally flat — is better served by a proximal term pulling toward the Phase-1
seed, which is smooth, has a derived rather than tuned weight, and selects a
unique point from a flat set. [unverified: not implemented or tested]

**The spin parameterization has an exact null direction.** Spin reaches the model
only through `spin_deg_from_sxy = degrees(arctan2(sy, sx))`, and nothing
normalizes `(sx, sy)` first. The radial direction is therefore an exact null
direction of all 402 constraint rows and of every objective term except
`unit_circle_penalty`. L-BFGS cannot learn curvature where the gradient is
identically zero, so IPOPT supplies it through inertia correction — adding
`δ_w·I`, which is basis-dependent, and is the channel through which the mixed
variable units actually reach the solver.

**Measured, and that prediction does not hold.** The instrumented logs show
essentially no regularization — 0–7 perturbed iterations out of 400–1000 — so
inertia correction is not how this null direction hurts. A direct test also rules
out the penalty's radial gradient as the floor on dual infeasibility: on
candidate 5660 the radial component is 0.003 while dual infeasibility is 2.151,
the largest in the set, and across five candidates dual infeasibility tracks
`max|∇f|` (1.34–2.52) rather than the radial component (0.003–0.668). The
redundant parameterization remains a structural defect and the origin remains
reachable, but neither is the measured cause of the stall.

The bounds do not protect the origin: `sx, sy ∈ [−1.1, +1.1]` **contains
(0, 0)** [measured, `pipeline/phase1_geometry.py`], where `arctan2` gradients
diverge as 1/r while the penalty's restoring gradient `4r(r²−1)` goes to zero.

The fix is to use the spin angle directly, removing the null direction, the
penalty and its weight, the origin singularity, and 7 variables. A cheaper
version imposes `sx² + sy² = 1` as a hard equality. Normalizing inside the model
is worse than either: it converts the null direction into exact gauge freedom.

**Softmin bias consumes part of the clearance budget.** `soft_min_topk` satisfies
`min(v) − log(k)/β ≤ softmin ≤ min(v)`, so at β=20 and k=16 the worst-case bias
is 0.139 mm against a 0.2 mm requirement. The bias is downward, so no collision
is ever admitted, but configurations whose true clearance lies between 0.2 and
0.339 mm are rejected. Measured typical bias is 0.03 mm, so this is a bound
rather than the norm.

**Padded rows cost size, not rank.** `_LARGE_SLACK` rows are constants with
identically zero gradient. They do **not** make the KKT matrix rank-deficient:
IPOPT rewrites every inequality as `d(x) − s = 0` with a bounded slack, so such a
row reads `[0 … 0 | −1]` and the augmented system stays nonsingular; the
multipliers simply go to `μ/1000`. Their cost is 58% of a dense 402×45 Jacobian
plus needless slacks and barrier terms. **The `_padding_mask` docstring in
`objectives/phase2.py` states the rank-deficiency claim and is wrong.**

---

## Known defects to fix

- `lambda_unit_circle` is absent from `_weights_key` in `objectives/phase2.py`,
  so it does not participate in the `_JIT_CACHE` key. Tuning it on a cache hit is
  silently a no-op. [measured]
- The `_padding_mask` docstring asserts rank deficiency; correct it to describe
  size. [measured]
- A single `constr_viol_tol` spans four unit systems: threading g-units,
  millimetres, hundredths of a millimetre on the gained rows, and degrees.
- **A spin pair reached exactly `(0, 0)` in production data** — one of 280 spins
  in the 837772 repeat set at memory 60 with no clearance bonus. `arctan2(0, 0)`
  returns 0, so that probe's spin silently became 0° instead of raising. The
  bounds `[−1.1, +1.1]` do not exclude the origin, and the penalty's restoring
  gradient vanishes there, so this is a reachable state rather than a theoretical
  one. Radius at final poses is otherwise tight: median 0.9993–0.9996, with
  `|r − 1|` median 0.002–0.003 and p90 0.0075–0.012. [measured]

---

## Experimental results — 837772, 40 candidates, each setting solved twice

Run-to-run pose divergence is the median over candidates of the largest component
change between two identical runs. It mixes degrees and millimetres, so it is a
trajectory-divergence proxy, not a physical distance. It has resolution where the
kept/lost label does not.

| setting | pose divergence | bitwise identical | label agreement | median iters | at 1000 cap |
|---|---|---|---|---|---|
| memory 60 + no bonus | 0.000 | 40/40 | 100% | 1000 | 70/80 |
| **memory 60 + no bonus + acc tol 5** | 0.000 | 40/40 | 100% | 1000 | 42/80 |
| memory 60 + no bonus + dead rows dropped | 0.000 | 40/40 | 100% | 1000 | 66/80 |
| memory 60 + no bonus + obb gain 1 | 0.000 | 32/40 | 98% | 1000 | 69/80 |
| no clearance bonus (memory 6) | 0.000 | 28/40 | 90% | 944 | 37/80 |
| obb gain 1 | 1.207 | 0/40 | 82% | 559 | 12/80 |
| obb gain 10 | 1.961 | 0/40 | 82% | 669 | 21/80 |
| dead rows dropped | 2.211 | 0/40 | 70% | 571 | 20/80 |
| tau 0.3 | 3.251 | 0/40 | 75% | 611 | 20/80 |
| memory 20 | 3.523 | 0/40 | 78% | 780 | 28/80 |
| smooth reward | 3.932 | 0/40 | 80% | 638 | 17/80 |
| acceptable tol 5 | 4.536 | 0/40 | 70% | 187 | 0/80 |
| memory 60 + smooth + tau 0.3 | 4.677 | 0/40 | 75% | 1000 | 60/80 |
| memory 60 + smooth + 3000 iters | 4.711 | 0/40 | 70% | 1545 | 53/80 |
| tau 0.8 (current default) | 4.924 | 0/40 | 62% | 673 | 24/120 |
| memory 60 + smooth | 6.989 | 0/40 | 72% | 1000 | 60/80 |

What this establishes:

- **Removing the clearance bonus is categorical.** Every no-bonus arm reaches
  exactly 0.000 divergence; nothing else gets below 1.2. Memory 60 does not
  create determinism — bonus removal alone gives 28/40 bitwise identical at
  memory 6, and memory 60 completes it to 40/40.
- **`limited_memory_max_history=60` exceeds n=45**, so the curvature model can
  represent a full Hessian and "limited memory" stops restricting anything.
  Raising it further should buy nothing.
- **Kept rate is not resolvable by this benchmark.** Every kept candidate clears
  by p10 +0.002 mm, median +0.021 mm, p90 +0.040 mm — the same order as the
  measured sampling gap and envelope offset. Differences of a few candidates
  between settings are noise.
- **Smoothing the reward is not a substitute for removing it.** At memory 60,
  smoothed gives 72% agreement against 100% for removal. Smoothing at memory 60
  is in fact the worst configuration measured by divergence.
- **Dropping dead rows halves divergence** at default settings (4.924 → 2.211)
  but changes no exit characteristics and no yield. Consistent with the rows
  costing size rather than rank.

### What is confounded — do not over-read the gain arms

Because `constr_viol_tol` is absolute on the gain-carrying constraint, each OBB
gain arm changed the effective physical tolerance by the same factor it changed
the row scaling. The 100 → 10 → 1 series therefore does **not** cleanly isolate
conditioning. The control run that would separate them is `OBB_GAIN=1` with
`IP_CVTOL=1e-6`, holding the physical tolerance fixed. It has not been run.

Stacking gain 1 onto memory 60 + no bonus did not reduce cap hits (69/80 against
70/80) and cost 18 points of kept rate; the 9 lost candidates all failed on
implant contacts, whose margins under the baseline were 0.002–0.039 mm.

### The dominant unexplained behaviour

Every no-bonus arm sits at the iteration cap — median 1000, about 70 of 80 runs.
The best configuration available does not converge; it times out, and no
conditioning knob moved that.

The instrumented run explains the mechanism: these solves reach feasibility early
and then stall on stationarity, so the `tol=1e-6` exit is unreachable and the cap
is simply where they stop. `acceptable_tol=5` converts that into a clean exit, and
the 40-candidate pair settles the cost question: median iterations fall 673 → 187,
**nothing reaches the cap (0 of 80 against 24 of 120)**, and exits become 79
acceptable out of 80 where the baseline produced 96 local-infeasibility and 24
iteration-limit. Kept averages 70% against the baseline's 71.3%, inside the
baseline's own 57–85% spread, and the FCL margin of surviving plans is unchanged
at +0.0092 versus +0.0087. The objective worry from the six-candidate probe was
noise.

It buys cost, not determinism: divergence stays at 4.536 against 4.924 and no two
runs agree bitwise. Removing the clearance bonus remains the only lever that
reaches that.

The two compose. `LAM_CLEAR=0` with `IP_HIST=60` and `IP_ACC_TOL=5` keeps kept at
88%/88%, agreement at 100% and all 40 poses bitwise identical — the tolerance does
not disturb the determinism — while cap hits fall from 70 of 80 to 42 of 80 and 34
runs reach a clean acceptable exit. These are now the `phase2_ipopt.py` defaults
— `LAM_CLEAR=0`, `IP_HIST=60`, `IP_ACC_TOL=5`, `IP_ACC_ITER=8` — so the pipeline
drivers pick them up without overriding anything. Every measurement behind them
comes from a single subject; a confirmation run on a second subject is still
outstanding.

The remaining 42 are not explained. In the defaults arm the same setting took cap
hits to zero, so something specific to the no-bonus arm blocks acceptance. Its
complementarity at exit was 8.9e-3, 1.0e-1 and 1.6e-3 across the instrumented
logs, against roughly 1e-9 under defaults, and acceptance requires all five
criteria to hold simultaneously for `acceptable_iter` consecutive iterations.
`acceptable_compl_inf_tol` defaults to **0.01**, so of those three exits one
exceeds the threshold tenfold, a second sits at 89% of it, and only the third is
comfortably inside; the defaults arm runs three orders inside. That fits the split
between 34 acceptable exits and 42 that still cap, but it rests on three logs and
is untested as a fix — the option is not currently exposed as an env knob.

---

## Reading the iteration log

Production runs at `print_level=0`, so none of this is visible today. For a
diagnostic run set `print_level=5`, `print_info_string="yes"` and an
`output_file`. Level 6 additionally prints the scaling factors IPOPT chose and
what MUMPS did; level 8 prints the `x`/`c`/`d` scaling vectors element by
element, which is how to confirm a `g_scaling` array landed.

Columns [docs]: `iter` (a trailing `r` means the restoration phase), `inf_pr`
(**unscaled** original constraint violation), `inf_du` (**scaled** dual
infeasibility — during restoration this belongs to the restoration problem, not
yours), `lg(mu)`, `||d||` (primal step, including internal slacks), `lg(rg)`
(log₁₀ of the Hessian regularization δ_w; a dash means none was needed),
`alpha_du` / `alpha_pr`, `ls` (backtracking steps).

Trailing letter on `alpha_pr` [docs]: `f`/`F` filter f-type step without/with a
second-order correction, `h`/`H` h-type, `R` restoration just started, `w`
watchdog, `s`/`S` accepted in soft restoration, `t`/`T` tiny step accepted
without a line search, `r` a previous iterate was restored.

`print_info_string` tags are **not** in the public documentation; these were read
from the IPOPT 3.14 source, with the emitting file:

| tag | meaning | file |
|---|---|---|
| `!` | restoration tightened its tolerance because the original problem was only slightly infeasible | `IpRestoConvCheck.cpp` |
| `A` | this iterate satisfies the acceptable criteria (count these to see whether `acceptable_iter` is reachable) | `IpOptErrorConvCheck.cpp` |
| `e` | evaluation error — step cut back after NaN/Inf from the model | `IpBacktrackingLineSearch.cpp` |
| `Nh`, `Nj`, `Dh`, `Dj`, `L`, `l`, `dx` | KKT perturbation handler: new or decreased perturbation of the Hessian block (`h`), the Jacobian/(2,2) block (`j`), or both | `IpPDPerturbationHandler.cpp` |
| `S`, `s`, `q`, `a` | linear system singular; singular with small residual; asked the solver for better quality; wrong inertia | `IpPDFullSpaceSolver.cpp` |
| `Wr`, `WS`, `We` | limited-memory quasi-Newton events: reset, update skipped on a tiny step, non-positive `sᵀy` | `IpLimMemQuasiNewtonUpdater.cpp` |
| `F+`, `F-` | filter reset after repeated filter rejections; reset wanted but `max_filter_resets` exhausted | `IpFilterLSAcceptor.cpp` |
| `C`, `M`, `W`, `w` | second-order correction; magic step; watchdog accepted; in watchdog | `IpBacktrackingLineSearch.cpp` |

Telling the three failure modes apart:

- **Stuck in restoration** — iteration numbers carry `r`, the original `inf_pr`
  plateaus, and `inf_du` refers to the restoration problem so its falling means
  nothing about yours. A `!` anywhere is near-proof that the problem was almost
  feasible when IPOPT gave up.
- **Regularization thrash** — `lg(rg)` non-dash on most iterations and climbing,
  with `Nh`/`Nj`/`L`/`S`/`a` tags, small `||d||`, and the objective barely moving.
- **Line-search failure** — `ls` at 10 or more with `alpha_pr` collapsing below
  1e-4, then an `R`. An `e` here means the JAX constraint path returned NaN or
  Inf at a trial point, which is a modelling bug rather than a solver setting.

Worth running once: `derivative_test="first-order"` with
`check_derivatives_for_naninf="yes"`. A wrong dense Jacobian produces exactly the
locally-infeasible signature, and this rules it out cheaply.

---

## Tooling

Analysis prototypes live in `scratch/estimators/` (gitignored):
`conditioning.py` (row norms per category with gains divided out, column norms
per variable kind), `ipopt_scaling.py` (what IPOPT's gradient-based scaling would
compute, and the spread that survives it), `check_padding_mask.py` (structural
padding mask against the rows that carry gradient at a solved pose),
`compare_tuning.py` (kept rate, agreement, iterations and exits per setting) and
`run_benchmark.sh` (a Phase-2 pair on an MPS process pool).

---

## Sources

- [IPOPT options reference](https://coin-or.github.io/Ipopt/OPTIONS.html)
- [IPOPT output reference](https://coin-or.github.io/Ipopt/OUTPUT.html) — iteration columns, `alpha_pr` tags, exit codes
- [IPOPT special features](https://coin-or.github.io/Ipopt/SPECIALS.html) — derivative checker
- IPOPT 3.14 source: `IpGradientScaling.cpp`, `IpRestoConvCheck.cpp`, `IpRestoMinC_1Nrm.cpp`, `IpOptErrorConvCheck.cpp`, `IpPDPerturbationHandler.cpp`, `IpIpoptData.cpp`, `IpOptionsList.cpp`
- [Wächter & Biegler 2006, Math. Prog. 106(1) 25–57](https://link.springer.com/article/10.1007/s10107-004-0559-y) — §3.1 inertia correction, §3.8 scaling
- [Wächter, Short Tutorial: Getting Started With Ipopt in 90 Minutes](https://drops.dagstuhl.de/storage/16dagstuhl-seminar-proceedings/dsp-vol09061/DagSemProc.09061.16/DagSemProc.09061.16.pdf) — restoration phase, L-BFGS termination advice
- [Hogg & Scott, arXiv:1301.7283](https://arxiv.org/pdf/1301.7283) — scaling reliability over 386 CUTEr problems: unscaled 92.8%, MC19 92.0%
- [Tasseff, Coffrin, Wächter & Laird, arXiv:1909.08104](https://arxiv.org/pdf/1909.08104) — linear solver comparison
- [CasADi, On the importance of NLP scaling](https://web.casadi.org/blog/nlp-scaling/)
- [JuMP, Tolerances and numerical issues](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/tolerances/)
- [OpenXLA GPU determinism](https://openxla.org/xla/determinism)
- [cyipopt reference](https://cyipopt.readthedocs.io/en/stable/reference.html) — `set_problem_scaling`
