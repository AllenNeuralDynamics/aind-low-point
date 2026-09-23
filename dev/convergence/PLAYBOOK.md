# Convergence playbook — Phase 1 and Phase 2

The ordered work to make Phase-2 IPOPT solves converge and Phase-1 RProp behave,
with what counts as done, what counts as enough, and where the user has to decide.

## Goal — set 2026-09-23 (supersedes the two-criterion goal, both answered by X26)

**Make both solvers reach a feasible point when one exists, and get a good
objective there. Judge every change on feasibility and objective — never on
`kept`.**

The three things a solve should do, in the user's order:

1. **Find a feasible solution if one exists.**
2. **Maximize the objective where it can.**
3. **Not spend iterations after the best solution is already found.**

**Goal 3 is closed and has no headroom.** X31 measures `waste` — the fraction of
a solve running past the earliest iteration it could have stopped unchanged — at
**0% on both arms**. The objective is still descending at the exit of every
solve. Capped solves are not spinning; they are still improving. So every
stopping-rule knob is off the table as a *win*: shortening a solve now costs
objective by construction, which is exactly what `compl5` does (X31: +0.92 median
objective regression, worse on 27 of 31).

**That leaves one lever: the landscape.** Only a change that makes the problem
easier to solve can serve goals 1 and 2 together. Those are the conditioning and
smoothness changes — `angle`, `smooth`, and anything F-tier — and they apply to
both phases, because Phase 1's RProp steps on the **sign** of each partial
derivative alone, so a kink does not merely slow it but reverses the step and
decays that coordinate's step size.

### Done when

| | test | measure |
|---|---|---|
| Phase 2 | `angle` and `smooth` at n=40 on the card, against `compl_ctl_a` | strictly feasible count; **paired objective** on candidates both arms solve; `reach`; model-optimistic count |
| Phase 1 | **answered by I2/X36 (three arms; `f32` closed).** Both interventions null; the finding is that flips *rise* while η collapses | flip rate and η-at-floor **split by iteration** — the pooled mean is ambiguous, since RProp flips near a minimum by design |
| both | a named verdict per change: improves, regresses, or unresolved | recorded in `EXPERIMENTS.md` with its scope stated |

A change is adopted only if it does not regress the paired objective and does not
raise the model-optimistic count. A change that gains feasible plans while losing
objective is a **trade**, is reported as one, and goes to the user.

**Stopping rule.** If neither `angle` nor `smooth` moves feasibility or objective
outside run-to-run noise, the landscape hypothesis is not supported at this scale
and the next move is the user's — not another knob.

**Test on the GPU.** Production runs on the card, so the card is the reference
platform. A CPU result is a hint, not an answer.

**Why not `kept`.** It is a binary gate on plans clearing by a median +0.021 mm,
which `PHASE2_CONDITIONING.md` calls unresolvable at this scale. It licenses
"unresolved", never "no effect", and using it for the latter is how `smooth` and
`angle` were dropped once already. This supersedes the older blockquoted scoring
rule below, which said to score a recommendation on `kept` and coverage.

**Non-goals:** `f32` (bf16 storage with float32 blending is better and is what
`bab0970` gave — do not reopen), `ip_acc_iter`, `ip_cvtol` (X30), any further
stopping-rule knob, flipping any default, any push or PR.

### Reading a trace before it carries a verdict

Four verdicts have been withdrawn for want of one sentence about scope. The
specific traps, all in `diag_hist`:

- `inf_pr`/`inf_du` are IPOPT's **scaled** iteration-log values; `mu` is the
  barrier parameter, not complementarity. Only `group_min`/`group_nviol` are
  unscaled, and only they may be compared against `ip_cvtol`.
- `obj` is the **restoration** objective wherever `restoration` is 1 — a
  violation norm, not f(x). Mask it, and take the final objective from the last
  non-restoration iteration; 2 of 40 control solves end inside restoration.
- A terminal or trailing-window figure describes that window, not the trajectory.
- Before calling a setting untested, check whether it was *chosen* (X30).

## The three documents

| file | holds | written when |
|---|---|---|
| `PLAYBOOK.md` (this) | the plan: items, order, success and stopping conditions, commands | only when the *plan* changes — a re-rank, a new item, a changed gate |
| `STATUS.md` | where we are: item board, what is running, tree state, append-only log | every session that changes anything |
| `EXPERIMENTS.md` | what we measured: one appended entry per result | whenever a measurement finishes |

**Which to open.** Starting a session: `STATUS.md`, then this file. Finishing a
measurement: append to `EXPERIMENTS.md`, then update the `STATUS.md` board and
log. Deciding what to do next: this file's ranking, filtered by the board.
Wondering why something is the way it is: the `STATUS.md` log.

The split exists so that the plan stays readable as the record grows. Do not put
results in this file, and do not put plans in `EXPERIMENTS.md`.

**Prior art.** `dev/PHASE2_CONDITIONING.md` is the measured state of the Phase-2
solve as of 2026-09-16 — build capabilities, constraint scaling, the tolerance
settings, formulation defects. It is not superseded by these documents; several
items below sharpen questions it left open. Read it before starting D2, D3 or D4.

---

## Bootstrap from a cold start

```bash
cd /home/galen.lynch/Documents/Code/aind-low-point
cat dev/convergence/STATUS.md            # where we are; read the log at the bottom
git status --short && git log --oneline -3
```

**One tree since 2026-09-23.** `implant-hole-walls` was rebased onto
`smooth-model-p1trace`, so the main checkout carries every solver commit this
investigation produced and the `-p1angle` and `-smooth` worktrees were removed.
Run everything from here. The `-motion` and `-rig` worktrees are other work,
hold uncommitted changes on the pre-`f986af6` package layout, and must be left
alone.

**Before a measured number becomes a verdict, say what it measures over.** Three
verdicts in `EXPERIMENTS.md` had to be withdrawn for want of that one sentence —
a terminal value read as a trajectory, a trailing window read as a total, a
scaled log field read as the unscaled quantity a tolerance is compared against.
Its header lists the specific confusions. When a diagnostic is about to carry an
argument, read the code that writes it first.

Everything runs on 3.13, and the tools below are on PATH rather than in the venv:

```bash
uv run --python 3.13 pytest -q     # the suite
ruff check && ruff format          # lint, format
mypy && codespell && interrogate   # types, spelling, docstring floor
uv run --python 3.13 python scripts/parity_phase2.py --baseline HEAD~1
```

**The parity harness is mandatory for any change to a solver path.** It writes a
synthetic subject, runs both arms and exits non-zero on any difference.

**`scratch/` is gitignored.** Every estimator script named below is untracked and
exists only on this machine. If `scratch/estimators/` is empty, the scripts are
gone and must be rebuilt before the items that use them; the docstring at the top
of each one states its question and its invocation.

Machine rules, both non-negotiable:

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # must be < 2.5 GiB to use the GPU
```

CPU work runs at `nice 10` with about nine workers total. XLA's CPU backend starts
one thread per CPU the process may run on and ignores `OMP_NUM_THREADS`, so
workers must be pinned — `3b83c55` does this, and a pool that is already running
cannot be fixed, only restarted.

### Running and comparing Phase-2 arms

```bash
# one tree since the 2026-09-23 rebase; TREE= would run another checkout's code
scratch/estimators/run_arms_cpu.sh 837772-config \
  scratch/estimators/837772-config/capped10_ranks.txt \
  scratch/estimators/837772_cpu 3 \
  base= nothread=RUTTER_LAM_THREAD=0 f32=RUTTER_GRID_DTYPE=float32

taskset -c 6-17 \
  scratch/estimators/run_arms_cpu.sh 837772-config \
  scratch/estimators/837772-config/capped10_ranks.txt \
  scratch/estimators/837772_cpu 2 \
  blend= blend_smooth=RUTTER_SMOOTH_THREAD_REWARD=true blend_angle=RUTTER_SPIN_PARAM=angle

uv run --python 3.13 python scratch/estimators/compare_arms.py \
  scratch/estimators/837772_cpu base nothread f32 blend blend_smooth blend_angle
```

`blend=` is empty because the worktree already carries the float32 blend, so it is
the branch's own baseline. Settings are `RUTTER_` plus the field name upper-cased;
`pipeline/settings.py` is the authority, not this file.

**Score on `compare_protocol.py`, not `compare_arms.py`** — it is the tool that
reports the goal's measures:

```bash
uv run --python 3.13 python scratch/estimators/compare_protocol.py \
  scratch/estimators/837772_g40 compl_ctl_a angle_a smooth_a
```

| column | what it answers |
|---|---|
| `feas` | goal 1 — strictly feasible plans, not `kept` |
| `obj`, and the pair's `objective change` | goal 2 — paired, on candidates both arms solve feasibly, restoration masked |
| `settle`, `waste` | goal 3 — the earliest iteration a solve could have stopped unchanged, and how much ran past it |
| `reach` | how fast feasibility is found, the conditioning indicator |
| `opt` | model says feasible, FCL disagrees — must not rise |

**Never edit a shell script while an invocation of it is running** — bash reads
scripts incrementally and the running copy will corrupt.

---

## Re-ranked 2026-09-22 after Tier 0

The gate fired and the user closed it. Tier 0 finished and did not support the
plan's premise: the line search is healthy (X2), the active set is tiny and well
conditioned (X4), and the float32 floor is an order below `ip_tol` (X4). What caps
these solves is a feasibility leg that cannot fire, on a quantity ~100× the
model's own violation (X5).

**Decided:** C3 becomes the critical path. **F1 and C2 are dropped**, not
deferred — neither has evidence behind it now.

The ordering below is superseded by this. Read D5 and C3 first; the older items
remain for their reasoning and for what they ruled out.

> **Overtaken the same day by X6 — read this before acting on D5 or C3.** The
> paragraph above rests on X5, and X6 retracts it. `inf_pr` is IPOPT's internal
> primal infeasibility, not the unscaled violation `ip_cvtol` governs, so there
> is no ~100× gap to explain and no evidence for a scaling defect: 13 of 20 runs
> exit `Solved_To_Acceptable_Level` with `inf_pr` up to 6.23 and the model's
> violation at exactly zero. **D5 is answered and C3 is unsupported.** What the
> evidence now points at is the leg nobody configured — complementarity, left at
> IPOPT's `acceptable_compl_inf_tol` of 1e-2 — and the fact that no run reaches
> `Solve_Succeeded` at all. Raised as C5, plus a decision for the user; the
> critical path is back with the user. F1, C2 and C1 stay dropped regardless.
>
> **Then narrowed again by X8.** `dev/PHASE2_CONDITIONING.md` had already
> established the `inf_pr` identification, the unscaled basis of the three
> tolerance tests, and `acceptable_tol = 5.0` as a deliberate measured choice
> that works. So *what Phase 2 should converge to* is largely answered — stopping
> at acceptable is the design, and `kept` ignores `solver_status`. That is a
> decision, not work, so it carries no ID and sits in `STATUS.md`.
> **One item is genuinely open:**
>
> ### C5 — Set `acceptable_compl_inf_tol` · *the open item*
> `acceptable_compl_inf_tol` (1e-2) and `acceptable_dual_inf_tol` (1e10) were
> never set while the other two `acceptable_*` tolerances were chosen
> deliberately, and complementarity is the leg that still blocks the candidates
> that cap. The benchmark puts **42 of 80 solves at the 1000 cap under adopted
> defaults**, so the prize is up to half of Phase-2's compute.
> *Done when:* the 40-candidate pair shows the share at cap falling with pose
> divergence still 0.000, bitwise 40/40 and label agreement 100%.
> *Stop when:* determinism moves at all, or `kept` falls outside the noise the
> prior art records (every kept candidate clears by p10 +0.002 mm).
> **Ask the user before changing any production default.**
>
> ### B1 — Misaligned per-group constraint split · **fixed, `d790432`**
> `_live_group_sizes` counted live rows while `drop_dead_rows=False` left every
> row in the vector, so each group was read at the wrong offset — `arc_sep` fell
> into padding and reported `NaN`, and the trailing group absorbed the remainder.
> Per-group attribution was unreliable; a max or sum over the groups was not.
> Fixed with a regression test on the idle worktree, parity clean. Merging waits
> for the arms, which import the module.
>
> **Scoring rule for the whole goal:** a defaults recommendation is scored by
> `kept` and coverage on the 40-candidate x 2 protocol, never by exit codes, and
> never on ten candidates — run-to-run noise moved one candidate's objective by
> 1.7 under identical settings.

### D5 — Why the feasibility leg cannot fire · *critical path, diagnosis*

Added mid-flight from X2 and defined here. `ip_cvtol` is `1e-4`; the best
`inf_pr` any capped solve reaches is `0.0104`. Worse, a third of iterations
satisfy every constraint group (`group_nviol` zero) while `inf_pr` at those same
iterations exceeds `0.01`. Whatever IPOPT measures is not what the model reports.

Establish which it is, cheapest first:

1. **From data in hand, no compute.** Per iteration, correlate `inf_pr` against
   the model's own worst violation, `max(0, -group_min)` over the groups —
   excluding `arc_sep`, which is `NaN` everywhere. A constant ratio names the
   scaling factor outright.
2. If the ratio is not constant, the gap is not pure row scaling: check whether
   `inf_pr` includes variable-bound violations, and whether `group_nviol` uses a
   tolerance that hides small violations the solver still counts.
3. Confirm by re-solving one capped candidate with `nlp_scaling_method=none`.

**Done when** the ~100× gap has a named cause and it is known whether
`ip_cvtol = 1e-4` is asking for something physically meaningless.

**Stop when** that is known — the fix is C3, and it needs the user first.

### C3 — Constraint scaling · *critical path, fix*

Promoted from the conditional tier by X5. The groups mix units — `ml_sep` in
degrees against a 16° requirement, the clearance groups in millimetres — under one
shared `constr_viol_tol`, and `dev/PHASE2_CONDITIONING.md` already measured the
per-category Jacobian row norms and the hand-picked gain of 100 that sits on them.
`scratch/estimators/conditioning.py` regenerates those numbers.

Options, to be chosen with the user once D5 reports: set `ip_cvtol` to something
physically defensible; supply explicit row scaling through `user-scaling` and
`set_problem_scaling`, which `dev/PHASE2_CONDITIONING.md` records as working end
to end; or non-dimensionalize the rows at construction.

**Done when** a capped candidate exits `acceptable` on physically equivalent
geometry, with the 40-candidate × 2 protocol confirming no regression.

**Ask the user before changing `ip_cvtol` or any other production default.**

### Also surfaced, small and worth doing

- **`arc_sep` is `NaN` in all 20 candidate-runs** — one of six constraint groups
  constrains nothing, and it poisons any min-over-groups. Diagnose before relying
  on group statistics again.
- **Is the cap costing anything?** Returned points are interior with 1–3 active
  rows and good margins. If the last several hundred iterations do not improve the
  returned plan, an early stop on the best feasible iterate is worth more than any
  solver tuning. Measurable from the traces already in hand.

## Item IDs

The prefix says what *kind* of work an item is, not its priority or its order.
An item gets an ID only if it is work someone does; decisions belong to the user
and live in `STATUS.md` under "Blocked on the user", with no ID.

| prefix | kind | rule |
|---|---|---|
| `D` | Phase-2 diagnostic — reads the solver, changes nothing | free to start |
| `I` | Phase-1 instrumentation — build or run the trace | free to start |
| `F` | Formulation change to the model or the RProp itself | needs a `D` to point at it |
| `C` | Conditional — expensive or speculative | **do not start unsolicited**; needs a `D` to report first |
| `B` | Defect found in passing, with a reproduction | fix it, but never in a tree a run imports |
| `X` | Not an item — a result entry in `EXPERIMENTS.md`, numbered in order | append only |

Every ID must be *defined here* before it is used anywhere else. D5 was put on
the status board and cited by an experiment before it existed in this file, which
is the thing this section prevents.

## The work, in order

Tier 0 exists because three of its four items are nearly free and can invalidate
everything below them. Do not start Tier 2 before Tier 0 reports.

### D1 — Interpret the six Phase-2 arms · *first*

Read `base`, `nothread`, `f32`, `blend`, `blend_smooth`, `blend_angle` against each
other with `compare_arms.py`. Two arms are already in (X1); the rest are running.

**Done when** every arm has 10/10, and each intervention has a signed effect
against `base` on exits, kept count, objective, iterations and `inf_du`.

**Stop when** that table exists. Do not expand the candidate set to chase
significance — ten hand-picked capping candidates is a tail sample and cannot
carry more weight than direction.

**Watch for** the tension X1 already shows: `f32` better on every outcome that
matters and worse on `inf_du`. If the control confirms it, `inf_du` is not
tracking quality and D2 becomes the decisive item.

### D2 — Instrumented replay of capping candidates · **done, see X2**

**No replay was needed.** The arms already run under `minimize_ipopt_logged`, so
every handoff record carries a `diag_hist` with per-iteration `inf_pr`, `inf_du`,
`mu`, `alpha_pr`, `ls_trials`, `restoration`, plus `group_min`/`group_nviol` over
the six `SLACK_GROUPS`. `scratch/estimators/acceptable_forensics.py` reads them.
Check what the pipeline already records before building a driver to record it.

Still unrecorded, and needing a real replay if ever wanted: IPOPT's own scaling
factors (`print_level 5`), and counters for the jump discontinuities — `is_valid`
flips at `clearance/boxes.py:210`, out-of-bounds fallbacks at
`clearance/grids.py:97` and `objectives/coverage.py:214`, and `top_k` churn.
X2 makes these much less interesting: the line search is not failing.

**Done when** you can say which half of the `acceptable` test fails — `inf_du ≤ 5`
or `inf_pr ≤ 1e-4`, for `ip_acc_iter=8` consecutive iterations — and whether
`alpha_pr` collapses, how often restoration is entered, and what IPOPT's own
scaling factors are.

**Stop and re-rank if** `inf_pr` is the blocker: the smoothness program is then
aimed at the wrong criterion and C3/C1 move up. Likewise if `alpha_pr` stays near
1 throughout — nonsmoothness is then not what binds, and F1 drops to tidying.

Queue behind the running arms; it wants two or three cores.

### D3 — Active-Jacobian rank at a capped solution

Load a capped `x`, build the Jacobian, keep rows active within tolerance, take the
SVD. No solve; seconds. `derivative_check.py` already instantiates the real model
at a saved point, so this is a variation on a script that works.

**Done when** three capped points have a rank verdict, and if deficient, the
dependent rows are named.

**This is the strongest untested hypothesis.** Capped solves ending at median
`mu` 8e-3 is more characteristic of degeneracy than of nonsmoothness: kinks show
up as line-search failure, a rank-deficient active set shows up as duals blowing
up and the barrier refusing to descend. If LICQ fails, no amount of smoothing
converges and F1 becomes urgent for a different reason than the one it was
proposed for.

### D4 — float32 gradient noise floor

Perturb `x` slightly at a solution, look at the scatter in `∇f`, compare to
`ip_tol=1e-4` and `ip_acc_tol=5.0`.

**Done when** the floor is a number sitting next to those two tolerances.

**Stop when** it is known. If the floor is above `ip_tol`, we are asking for a
tolerance the precision cannot deliver; C1 goes live and the tolerance settings
need the user.

### I1 — Build the Phase-1 RProp trace · **done, see X3**

Built on branch `smooth-model-p1trace` in the `-p1angle` worktree, commits
`abeb9f0` and `81fbefe`. `RUTTER_TRACE=N` records N iterations of each stage to
`<out>_trace_<n>arc.npz`. Default-off verified bitwise twice; 904 tests pass.
The spec it was built to follows.



Add `trace=False` to `make_staged_rprop` as a Python-level flag, the same pattern
as `periodic=None` and `spin_angle=False`, so default-off emits today's graph and
parity is bitwise by construction.

Free channels, all already in the loop carry: flip fraction (`prod < 0`,
`search/minimizers.py:159`), fraction of coordinates at `eta_min` and at `eta_max`,
clip rate, wrap rate on periodic coordinates, mean `|Δx|`. Checkpoint channels,
about ten times per run: full `η` and `x`, plus objective and `‖∇f‖` — these are
checkpoint-only because `vmapped_grad_cw` is grad-only, so a value costs an extra
forward pass and per-iteration recording would roughly double Phase 1.

**Done when** default-off is bitwise identical on the synthetic subject across
`x`, `x_reduced`, `objective`, `min_clear`, `min_clear_reduced`; the suite and
checks are green; and the channels produce sane values on a four-candidate run.

**Stop immediately if** default-off parity fails — fix that before anything else.

Note for later reading: tracing changes fusion, so a traced run will not match an
untraced one in the last bits. All arms must be traced to be comparable.

### I2 — Instrumented Phase-1 replay · **done, see X36**

Ran as three arms, not four: `f32` is closed and B2 means a Phase-1 dtype arm
would not reach the gradient anyway. **Both interventions are null** and the
finding is elsewhere: flips *rise* late while the step collapses, and 57% of
movable coordinates end `reduced_coarse` pinned at `eta_min` still reversing
sign a third of the time. `angle` does nothing here because `arctan2`'s defect
is a magnitude singularity and RProp reads only the sign.

**Next is a per-term ablation, not another whole-pipeline arm** — F4's hole-wall
`max`, the hard `min`s at `objectives/soft.py:533`/`:561`, and the OBB-SAT `max`
at `clearance/boxes.py:222`. `clipped` at 0.4% rules out F2's mechanism.

The spec it ran to follows.



`base`, smooth reward, f32 blend, angle — on a real subject, matching the Phase-2
arms. **Gate on `nvidia-smi` under 2.5 GiB.**

**Flip rate is the headline measure, not a side channel.** RProp steps on the
sign of each partial derivative and nothing else, so a kinked objective does not
merely slow it: the sign reverses, RProp reads that as overshoot and cuts that
coordinate's η by `eta_minus`. A coordinate that keeps meeting kinks ratchets
down to `eta_min` and stops moving while its gradient is still large. That makes
the flip fraction (`prod < 0`, `search/minimizers.py:159`) a direct readout of
how kinked the landscape is, and it is why `smooth` and `angle` should show up
more clearly in Phase 1 than in Phase 2 — L-BFGS and IPOPT use magnitudes and can
average a kink out; RProp cannot.

**Done when** each arm reports flip rate, η-at-floor and η-at-ceiling fractions,
clip rate, how long candidates sit frozen at the cap, and `‖∇f‖` at the handoff —
plus VRAM and throughput for the f32 blend, which is still unmeasured on the card.

**Adopt on** a lower flip rate *with* an objective at handoff no worse than
`base`. A smoother objective that reaches a worse point is a trade, not a fix.

**Both flags are wired into Phase 1 and have never been measured there.**
`spin_param` and `smooth_thread_reward` both live on the shared
`PipelineSettings`, so `Phase1Settings` inherits them; `RECOMMENDATION.md`'s
"keep `unit_vector`" and "do not enable" verdicts come from Phase-2 CPU arms at
n=10 and say nothing about Phase 1. The two phases implement `angle` by
different mechanisms — Phase 2 wraps the model in NumPy outside the trace
(`objectives/spin_angle.py:34-106`), while Phase 1 composes `from_angle_layout`
*into* the traced objective (`search/minimizers.py:266-268`) and then treats θ as
a periodic RProp coordinate with a rescaled η — so a Phase-2 result does not
transfer.

**The trace already exists on this branch**: `RUTTER_TRACE=N` records N
iterations per stage to `<out>_trace_<n>arc.npz`, channels `flip`, `eta_floor`,
`eta_ceil`, `clipped`, `wrapped`, `step` (`search/minimizers.py:119-121`). All
arms must be traced to be comparable — tracing changes fusion.

**Note the asymmetry** the arms cannot control for: `smooth_reward`, which swaps
hard clearances for soft ones in the reward, is Phase-2 only. Phase 1's identical
`jnp.min` at `objectives/soft.py:533` and `:561` has no counterpart flag.

This single run closes five open items: the η-collapse question, the sizing for
F2 and F3, the handoff-quality question (C4), the Phase-1 spin-angle A/B that has
been blocked since `a4826e8`, and the f32-blend VRAM numbers.

**If the card stays busy**, run a reduced version on CPU against a small real
subject and label it underpowered. Do not run it on the synthetic subject and
present it as an answer — four candidates cannot characterize this.

### A1 — Which term owns Phase 1's flips · *running, pre-registered*

I2 measured that the flips are there and rise as η collapses; it did not say
which term produces them. A1 zeroes one weight at a time and reads the flip rate
back off the same trace, which attributes the flips without perturbing any other
part of the objective the way a smoothing would. Solution quality is expected to
degrade in every arm — every weight here is load-bearing, and this measures the
landscape rather than a configuration worth shipping.

Seven arms on 837772, 512 candidates, `chunk=128`, `trace=500`:

| arm | weight zeroed | the kink it carries |
|---|---|---|
| `base` | — | control |
| `mclear` | `lambda_margin_clear`, `..._fixture` | hard `min` over dual-rep pair clearances (`soft.py:533`/`:561`) |
| `mthread` | `lambda_margin_thread` | hard `min` across a probe's shanks |
| `thread` | `lambda_thread` | the hole-wall `max` (F4) |
| `clear` | `lambda_clearance`, `..._fixture` | OBB-SAT `max` over separating axes |
| `brain` | `lambda_brain` | one-sided hinge at the margin |
| `unit` | `lambda_unit_circle` | smooth — the control **for the ablation itself** |

```bash
scratch/estimators/run_p1_ablate_gpu.sh 837772-config scratch/estimators/p1abl 512 500 \
  base mclear mthread thread clear brain unit
uv run --python 3.13 python scratch/estimators/compare_p1_arms.py \
  scratch/estimators/p1abl base mclear mthread thread clear brain unit
```

**Pre-registered prediction: `mclear`.** Phase 2 runs this same reward at
`lam_clear = 0.0`, and the field's own comment gives the reason — it "takes its
gradient from the single closest surface sample, which flips between near-tied
samples". Phase 1 hard-codes `lambda_margin_clear = 1.0` in `Phase1Weights` and
exposes no setting to turn it off, so the phase that is most sensitive to a
sign flip is the one still paying for it.

**Read it against `unit`, not against zero.** Dropping any term changes where
the optimiser goes, so some flip-rate movement is the ablation and not the kink.
`unit` is smooth, so whatever it moves is the method's own floor; only a term
that beats that floor by a clear margin has been shown to own the flips.

**This closes nothing on its own.** A term identified here still needs a fix that
keeps what the term is for — F5's shape for a reward, not a deletion.

**Score on `min_clear`, not on the objective.** Each arm minimizes a different
function, so its `objective` field is not comparable to any other arm's — the
medians rank nothing. `min_clear` is safe: `make_min_clear_one` reads only
`softmin_beta` and the `top_k`s, none of which A1 touches, so every arm's
clearance is measured by the same evaluator. `compare_p1_arms.py` refuses the
objective comparison for an ablated arm rather than printing a number that
invites the X29 mistake in new clothes.

**The restore stage is identical across arms.** It builds `JointWeights`, not
`Phase1Weights`, so every arm enters RProp from the same restored seeds and the
comparison is properly paired.

### F6 — Phase 1's clearance reward, if A1 names it · *conditional, needs the user*

Prepared so a confirmed A1 has a fix ready, **not** to be applied before A1
lands. Both options change a production default.

Phase 1 already computes the smooth clearance values it would need. One call
returns both and the objective uses each for a different thing:
`_hard, _soft = swept_pair_clearances(...)`, then the *penalty* takes `_soft`
while the *reward* takes `jnp.min(_hard, axis=1)` (`objectives/soft.py:533`, and
`:561` for fixtures). So the kinked min is a choice, not a cost.

| option | what it does | what it gives up |
|---|---|---|
| **smooth it** | `_soft` in place of `_hard` at `soft.py:533`/`:561`, behind a `Phase1Weights.smooth_clearance_reward` plumbed from a setting | nothing computed — `_soft` is already in hand. Mirrors `constrained.py:407`/`:428` exactly |
| **zero it** | `lambda_margin_clear = 0.0`, which is what Phase 2 ships | the margin incentive that keeps Phase-1 poses off surfaces rather than merely clear of them |

**Phase 2 chose the second**, and note what that means for the first: its
`smooth_clearance_reward` flag is *inert* there, because `lam_clear = 0` has
already switched the reward off. So Phase 2's default is evidence that zeroing
works in Phase 2, and no evidence at all about smoothing — that path is
genuinely untested in either phase.

**Whichever is chosen, score it on `min_clear` and on the Phase-2 outcome**, not
on the flip rate it was picked to lower. A1 measures a mechanism; only the
handoff says whether fixing it produced better plans.

### F1 — Split the hard-`min` constraint rows · **DROPPED 2026-09-22**

Dropped after Tier 0: the line search is healthy and the active set is tiny and
well conditioned, so neither the kink nor the tied-`min` degeneracy this was meant
to fix is in evidence. The reasoning is kept because it was sound given X1.

`objectives/constrained.py:400` collapses four dual-rep categories per pair into
one row with `jnp.min`; `constrained.py:421` does the same for fixtures;
`objectives/threading.py:169` takes `max(oval, max over walls)`. Emitting one row
each is the identical feasible set with no kink, and independent rows also remove
the tied-`min` path to rank deficiency — so it addresses both D2's and D3's
hypotheses at once.

**Done when** the split rows provably reproduce the old row (for sampled points,
the max over the split equals the old single value), the parity harness passes,
and an arms re-run shows the effect.

**Ask the user before implementing.** Row counts change, so slack layout, handoff
and metrics bookkeeping move with them; that is a design decision, not a tidy-up.

### F4 — Smooth the hole-wall `max` · *the newest kink, and it has no flag*

`objectives/threading.py:169` returns
`jnp.maximum(g, jnp.max(wall_dist, axis=-1) * (2.0 / safe_b))` — two stacked hard
maxima, kinked wherever a wall takes over from the oval and wherever two walls
tie. **Both phases reach it on every objective evaluation**, it is this branch's
own addition, and unlike every other smoothing in the package it has neither a
smooth variant nor a setting.

It is the prime suspect for goal 1. X32 gives circumstantial support: the one
constraint group `angle` improved was **thread**, 20.6% to 10.7% of iterations
violated, which is the group this `max` sits in.

**Done when** a softmax variant behind a setting reproduces the hard value within
its stated bias, parity passes, and an arms pair scores it on the goal's
measures in both phases.

**Ask the user before implementing** — it changes a constraint row's value, so
it is a formulation change, not a tidy-up.

### F5 — A threading reward that is smooth *and* zero below zero

X33 rejected `smooth_thread_reward`, but it rejected one particular smoothing,
not the idea. `objectives/rewards.py:53-64` makes two separate substitutions and
both look wrong for this term:

- **The gate.** `softplus(beta*w)/beta` is `ln(2)/beta = 0.035` at *zero* slack
  and still positive at a violation, where `jnp.maximum(0.0, w)` is exactly 0.
  It pays for sitting at or just outside the bore wall. A quadratic hinge is C1
  and still exactly zero for every `w <= 0`:

  ```python
  c = jnp.clip(w, 0.0, delta)
  g = 0.5 * c * c / delta + jnp.maximum(w - delta, 0.0)
  ```

  `g'` runs 0 -> w/delta -> 1 continuously, so the kink at zero slack goes
  without the offset. `delta` sets the smoothing width.
- **The minimum.** A log-sum-exp softmin over all K section-shank entries
  spreads gradient across every near-tied section, and threading needs the
  *worst* section to clear. `clearance/smooth.py:17-46` already has
  `soft_min_topk`, which blends only the k smallest — smoothing real ties
  without diffusing over K.

**Done when** the pair is measured against both the hard reward and the current
smooth one, on the goal's measures, in both phases.

**Ask the user before implementing** — it changes an objective term.

### B3 — Phase 1's chunk is fixed, not sized to the card

`RUTTER_CHUNK` defaults to 256, which needs 6.59 GiB for the batch alone and
fails on a 10 GiB card once grids, atlas, coverage and restore buffers are
resident — before any tracing. `RUTTER_RESTORE_CHUNK` has the same problem one
stage earlier and is 16 spins wide. Nothing consults available VRAM, so a
smaller card gets `RESOURCE_EXHAUSTED` rather than more chunks.

Measured on this card: `chunk=64` completes 256 candidates in 115 s; `chunk=256`
fails identically with tracing on and off, so the trace is not the cause.

### B2 — Phase 1's gradient ignores `grid_dtype` · *defect, reproduction in hand*

`phase1.py:353-362` builds the objective through `make_batched_phase1_chunked`
honouring `grid_dtype`, but the *gradient* comes from `build_cw_fns`, which
hard-codes `jnp.bfloat16` at `search/minimizers.py:251` and `:271`. With
`RUTTER_BF16_STORE=false` the reported objective runs on fp32 grids while the
descent direction still runs on bf16 — the flag silently does not reach the
thing that moves the optimizer.

Phase 2 has no such split; `grid_dtype` flows straight through at
`phase2.py:514`.

**Fix in a tree no run imports**, with a test that the two paths agree on dtype.

### F2 — RProp bound-clip η reset

A coordinate pressed against a bound keeps its gradient sign, so η inflates to
half the bound range while making no progress, then takes a half-range step when
it finally leaves. Treating a clipped step as a sign flip is the standard remedy.
The periodic wrap fixed exactly this for spin; ML and the arc angles still have it.

**Done when** A/B shows clip inflation reduced with no regression, and parity
passes. **Gated on I2** sizing whether it matters.

### F3 — iRprop+, per-candidate freeze, ηmax ablation

Backtracking on a sign flip; a mask that zeroes the step for candidates that have
stopped improving; and an ablation of `ηmax = 0.5·range`, which is a large step to
take right after a kink-induced flip. Run as one comparison, not three.

**Ask before flipping any default.**

### C1–C4 — conditional, do not start unsolicited

- **C1** ~~float64 for Phase 2~~ **RULED OUT** by X4: the float32 floor is ~1e-5, an order below `ip_tol`.
- **C2** ~~tricubic or grid pre-smoothing~~ **DROPPED 2026-09-22** — — trilinear is C0 and its gradient jumps
  across every voxel face; the f32 blend fixed arithmetic noise, not this. Live
  only if D2 shows active rows sitting on faces. Expensive; measure first.
- **C3** explicit constraint and variable scaling — **PROMOTED to the critical path**, defined above. `dev/PHASE2_CONDITIONING.md` has the measured row norms and
  `scratch/estimators/conditioning.py` regenerates them.
- **C4** Phase-1 handoff quality and an L-BFGS tail — RProp is sign-only so it
  never converges to a stationary point, it hovers. Live once I2 gives `‖∇f‖` at
  the handoff.

---

## When to stop and ask

1. **Before changing any production default** in `pipeline/settings.py`. Opt-ins
   land off; flipping one is the user's call.
2. **Before implementing F1**, which changes the constraint layout.
3. **On the threading reward's future.** Already open: the constraint only
   requires being inside the 0.07 mm-inset bore, so a margin may belong in
   `RUTTER_THREADING_MARGIN_MM` rather than in the objective. `nothread` is
   evidence toward that decision, not the decision.
4. **If a Tier-0 result invalidates the plan's premise** — for instance D2 finding
   `inf_pr` is the blocker. Re-rank with the user rather than silently re-planning.
5. **If two consecutive items return null results.** That is a sign the model of
   the problem is wrong, and more items will not fix it.
6. **Always before pushing or opening a PR.** Nothing has been pushed.

## Housekeeping owed

- **Done 2026-09-23.** `implant-hole-walls` rebased onto `smooth-model-p1trace`;
  the `-p1angle` and `-smooth` worktrees are removed and arms run from here.
