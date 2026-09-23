# Convergence playbook — Phase 1 and Phase 2

The ordered work to make Phase-2 IPOPT solves converge and Phase-1 RProp behave,
with what counts as done, what counts as enough, and where the user has to decide.

## Goal — set 2026-09-23

**Put the two load-bearing legs of the defaults recommendation on the reference
platform: the control's own reproducibility, and `nothread`.**

Done when both hold:

1. Pose divergence, bitwise-identical count and label agreement for the baseline
   configuration, measured on this tree, recorded in `EXPERIMENTS.md`.
2. `nothread` at n=40 on the card, scored against `compl_ctl_a` on kept and on
   model-feasible-but-FCL-rejected.

**Stopping rule.** Kept is unresolvable at n=40 by the prior art's own standard,
so no arm changes a default on kept alone. Absent a divergence result that beats
the control without losing kept, the outcome is "confirmed on the reference
platform" and `RECOMMENDATION.md` stands unchanged — which is a success here, not
a null result.

**Non-goals:** `f32`, `smooth`, `angle`, `ip_acc_du_tol`, flipping any default,
any push or PR.

### Why these two and not the others

The 2026-09-22 goal is closed — its recommendation is written and its answer is
*change no default* (`RECOMMENDATION.md`, X25). What that answer lacks is
evidence on the reference platform: the arm series ran on CPU at n=10 across two
trees, and the reproducibility numbers four `keep` rows cite come from the
2026-09-16 table that X23 ruled out for this code. The conclusion is probably
right; almost none of it was measured where it counts.

- **The control repeat is the gap that matters.** Divergence and bitwise
  agreement have *never* been measured on this tree — `compl_ctl_a` is a single
  run, and a repeat is the only way to get them, since X20 showed the CPU path is
  deterministic and would return 0.000 by construction. `ip_hist`, `ip_acc_tol`,
  `ip_cvtol`, `drop_dead_rows` and the rejection of the smooth reward all cite
  that measurement. One arm, ~10 min.
- **There is a mechanism for it to fail, not just an absence of evidence.** The
  property is code-dependent — the doc's own table ranges from 40/40 to 0/40 —
  and the code moved. The table also measured it when the median solve ran to the
  1000 cap, where two runs take exactly 1000 iterations; here the median exits
  acceptable at 737, so a reduction-order difference can flip which iteration
  first satisfies the 8-iteration window and return a different pose.
- **`nothread` is the only arm worth the card.** X19 applied the prior art's
  standard to the series and only `nothread` clears it: −6/10 kept with a second,
  independent signature at 5/10 model-feasible-but-FCL-rejected. `f32`, `smooth`
  and `angle` are ±1–2 candidates, which that standard calls noise. If
  `nothread`'s effect does not reproduce on the card, the one durable finding of
  the arm series was a CPU artifact.

**Prerequisite.** `compare_arms.py` reports exits, iterations, `inf_du`, kept and
objective — not divergence. `compare_repeat.py` computes all three protocol
metrics (pose max|Δx| median, identical count, kept agreement) but reads `.pkl`
on a fixed naming scheme; it needs to read the `_g40_*_{a,b}.json.gz` handoffs
before criterion 1 can be scored.

**Retired, do not restart:** the `throughput` batch and `compl_5_a`/`compl_5_b`
(X25). `compl_ctl_b` was retired with that batch by mistake — it is the control
repeat criterion 1 needs, and answers reproducibility, not complementarity.

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

Worktrees. The main checkout is `implant-hole-walls` and is where production runs
from; branch work is on `smooth-model`:

```bash
git -C /home/galen.lynch/Documents/Code/aind-low-point-smooth log --oneline -3
```

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
# from the main checkout; TREE= runs a worktree's code instead
scratch/estimators/run_arms_cpu.sh 837772-config \
  scratch/estimators/837772-config/capped10_ranks.txt \
  scratch/estimators/837772_cpu 3 \
  base= nothread=RUTTER_LAM_THREAD=0 f32=RUTTER_GRID_DTYPE=float32

TREE=/home/galen.lynch/Documents/Code/aind-low-point-smooth taskset -c 6-17 \
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

### I2 — Four-arm instrumented Phase-1 replay · *GPU-gated*

`base`, smooth reward, f32 blend, angle — on a real subject, matching the Phase-2
arms. **Gate on `nvidia-smi` under 2.5 GiB.**

**Done when** each arm reports η-at-floor and η-at-ceiling fractions, flip rate,
clip rate, how long candidates sit frozen at the cap, and `‖∇f‖` at the handoff —
plus VRAM and throughput for the f32 blend, which is still unmeasured on the card.

This single run closes five open items: the η-collapse question, the sizing for
F2 and F3, the handoff-quality question (C4), the Phase-1 spin-angle A/B that has
been blocked since `a4826e8`, and the f32-blend VRAM numbers.

**If the card stays busy**, run a reduced version on CPU against a small real
subject and label it underpowered. Do not run it on the synthetic subject and
present it as an answer — four candidates cannot characterize this.

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

- Discard the three uncommitted files in the main checkout once `base` finishes;
  they duplicate `smooth-model` and exist only to feed the running arm.
- The `smooth-model-p1angle` worktree is redundant now that `a4826e8` is folded in.
