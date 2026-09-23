# Convergence work — current status

Where the work in `dev/convergence/PLAYBOOK.md` stands. The top of this file is
overwritten as things change and always describes now; the log at the bottom is
appended and never edited, and carries what a reader needs to understand how the
current state came about.

**As of:** 2026-09-22 12:30 PDT

> **The gate has re-opened, and the last decision rested on a misread.** X6
> retracts X2 and X5: `inf_pr` is IPOPT's *internal* primal infeasibility and is
> not the quantity any stopping test reads, so the "100× disagreement" that
> promoted C3 was two different measures being compared, not a scaling defect.
> Against the measure the solver actually tests, 13 of 20 runs already exit
> `Solved_To_Acceptable_Level` with the violation at exactly zero, 3 exit locally
> infeasible and 4 cap — and **none reaches `Solve_Succeeded`**. What gates the
> test is complementarity: `acceptable_compl_inf_tol` is left at its 1e-2 default
> while `ip_acc_tol = 5.0` is close to inert, which is the barrier stall X0 saw.
> Degeneracy, precision and the line search remain ruled out. **C3's evidence is
> withdrawn; the ranking needs the user again.**

---

## Item board

| ID | Item | State |
|---|---|---|
| D1 | Interpret the six Phase-2 arms | **4 of 6 read (X12)** — control in and it wins: kept `base` 10/10, `f32` 9, `blend_angle` 7, `nothread` 4 |
| D2 | Which leg of `acceptable` fails | **answered (X9)** — not feasibility; complementarity gates the four that cap. X2 retracted |
| D3 | Active-Jacobian rank at a capped solution | **done (X4)** — not rank-deficient; active set is 1–3 rows, cond ≤ 2.55 |
| D4 | float32 gradient noise floor vs `ip_tol` | **done (X4)** — floor ~1e-5, an order under `ip_tol`; C1 not warranted |
| D5 | Why the feasibility leg cannot fire | **done (X6)** — false premise; it fires in 13 of 20. Complementarity gates the test |
| I1 | Build the Phase-1 RProp trace, verify default-off parity | **done (X3)** — `abeb9f0`, `81fbefe`; bitwise, 904 tests pass |
| I2 | Four-arm instrumented Phase-1 replay | **ready, blocked** — GPU held at 9.5 of 10.2 GiB |
| F1 | Split the hard-`min` constraint rows | **DROPPED** — no evidence after Tier 0 |
| F2 | RProp bound-clip η reset | not started — gated on I2 sizing it |
| F3 | iRprop+, per-candidate freeze, ηmax ablation | not started — gated on I2 |
| C1 | float64 for Phase 2 | **ruled out by X4** — precision is not binding |
| C2 | tricubic / grid pre-smoothing | **DROPPED** — line search is healthy |
| C3 | Explicit constraint scaling | **evidence withdrawn (X6)** — no longer supported as critical path |
| C4 | Phase-1 handoff tail | not started — gated on I2 |
| C5 | Set `acceptable_compl_inf_tol` so the capped solves can exit | **mechanism settled (X10, X11)** — 1000-iteration cap becomes a 120-iteration acceptable exit, strictly feasible. Needs the 40-candidate pair for yield |
| B1 | Misaligned per-group constraint split | **fixed (X7)** — `d790432` on `smooth-model-p1trace`, test + parity; not merged |

## Running now

| what | where | since | state |
|---|---|---|---|
| `base` arm | main checkout, 3 workers, CPUs 0–5 | 10:47 | 0 of 10 — healthy, 3 workers at ~97% CPU on one core each, ~56 min into their first 1000-iteration candidates |
| `blend`, `blend_smooth`, `blend_angle` | `-smooth` worktree, 2 workers each, CPUs 6–17 | 11:05 | 2/1/1 of 10 |

Finished: `nothread` 10/10 (4 feasible), `f32` 10/10 (9 feasible) — recorded as X1.

Nine workers total at `nice 10`, which is the agreed ceiling for this machine.

## Tree state

- **Main checkout** `/home/galen.lynch/Documents/Code/aind-low-point`, branch
  `implant-hole-walls` at `b3b6bbe`, with three files modified and uncommitted:
  `pipeline/phase2.py`, `pipeline/settings.py`, `tests/test_pipeline_settings.py`.
  These duplicate work already committed on `smooth-model` and exist only so the
  running `base` arm has the settings and the CPU pinning. **Discard them once
  `base` finishes** — do not commit them here.
- **`smooth-model`** worktree `/home/galen.lynch/Documents/Code/aind-low-point-smooth`,
  HEAD `3b83c55`. Carries `619cb3f` settings, `bab0970` f32 blend, `c4fc8ab`
  smooth threading reward, `d969bd7` Phase-2 spin angle, `a4826e8` Phase-1 spin
  angle, `3b83c55` CPU worker pinning. Own venv, built with the `ipopt` extra.
- **`smooth-model-p1trace`** worktree `/home/galen.lynch/Documents/Code/aind-low-point-p1angle`,
  HEAD `d790432`, branched off `3b83c55`. Carries I1's RProp trace (`abeb9f0`,
  `81fbefe`) and the group-split fix (`d790432`). Idle, own venv — the tree to
  use for source work while the arms hold the other two.

Nothing pushed. No PR.

## Blocked on the user

**THE STOPPING RULE HAS FIRED (X12).** `base` keeps 10 of 10 on the capped-10 set;
`f32` 9, `blend_angle` 7, `nothread` 4. No arm beats the control on kept-plan
yield, and since the control is at the ceiling, the two unfinished arms can at
best tie. The rule says: stop tuning, and bring F1 and the threading-reward
question to the user as a decision.

Read it with its limit attached — kept saturates at ten candidates and the prior
art says the rate is not resolvable at this size. The direction is still the only
yield evidence there is, and it does not support adopting any intervention.


**The critical path needs re-deciding.** C3 was made the critical path on X5, and
X6 withdraws X5. X8 then narrowed what is actually open, because
`dev/PHASE2_CONDITIONING.md` had already settled most of it. What remains:

1. **C5 — the one genuinely open item, and the prize is large.** `ip_acc_tol =
   5.0` was chosen deliberately and works. But the 40-candidate benchmark shows
   **42 of 80 solves still hit the 1000 cap under adopted defaults** — over half
   the Phase-2 compute is spent on candidates that will stop at acceptable
   anyway. `acceptable_compl_inf_tol` was never set and complementarity is what
   blocks those. If it is, setting it converts a large share of the cap into an
   early exit. Confirmation is running on candidate 11367.
2. **What Phase 2 should converge to — a decision, not an item, and largely
   already made (X8).** Stopping at acceptable is the
   design: `kept` is `fcl_keep and thread_keep` and ignores `solver_status`.
   The consequence for the goal is that **the defaults recommendation must be
   scored by `kept` and coverage on the 40-candidate protocol, not by exit
   codes** — X1's ten-candidate objective medians cannot carry it. The protocol
   is 40 candidates, each setting solved twice, reported as pose divergence,
   bitwise-identical count, label agreement, median iterations and share at the
   cap (`dev/PHASE2_CONDITIONING.md`, and `run_benchmark.sh` runs the pair).
3. **Fix the misaligned group split (X7).** `_live_group_sizes` returns live
   counts while `drop_dead_rows=False` leaves all 402 rows in the vector, so the
   groups are read at the wrong offsets — that is the `arc_sep` `NaN`, and it
   makes per-group attribution unreliable. Max-over-groups is unaffected.
   **Fixed in `d790432`** on the idle worktree, with a regression test; merging
   waits for the arms, since they import the module.

F1, C2 and C1 stay dropped — X6 changes nothing about them.

---

# Log

Append only. Newest last. One entry per thing a future reader needs in order to
make sense of the state above.

**2026-09-22 08:48** — Launched three Phase-2 CPU arms on the ten candidates that
cap in production (`base`, `nothread`, `f32`), three workers each, from the main
checkout with the settings diff applied uncommitted.

**2026-09-22 ~09:30** — Machine found badly oversubscribed. Root cause: XLA's CPU
backend starts one compute thread per CPU the process is *allowed* to run on and
ignores `OMP_NUM_THREADS` and friends, so nine unpinned workers each ran a
24-thread pool. Measured 81–106 threads per worker, ~1.5M context switches/s,
49–55% system time, load 95–105. CPU affinity is the only lever jaxlib 0.10.0
exposes, and it works when set before the XLA client is created. Fixed in
`3b83c55` by giving each process-pool worker its own CPUs.

**2026-09-22 ~10:20** — Pinning the *already running* workers made it worse (load
63 → 105) because a live pool cannot be shrunk; reverted. Only a restart applies
the fix. The user chose to restart `base` alone and let `nothread` and `f32`
finish on their old pools, so those two arms carry the old thread behaviour. The
parity harness showed poses are bitwise identical between a 6-thread and a
1-thread pool, so thread count does not change results and the arms stay
comparable.

**2026-09-22 10:47** — `base` restarted from scratch on CPUs 0–5. Its earlier log
is preserved as `scratch/estimators/837772_cpu_base_run1.log`. After the restart:
12 threads per worker, 72k context switches/s, 28% system time, load 40.

**2026-09-22 ~10:50** — Pruned the JAX compile cache in `/tmp` of everything unread
for two days: 3.8 GB across 1791 entries down to 538 MB across 441. `/tmp` is
tmpfs, so this freed RAM directly and took swap from full to 3 GB free. Swap has
since refilled to ~6 GB as `base` rebuilds its cache; `si/so` are zero, so nothing
is paging.

**2026-09-22 11:05** — The arms script exited 2 with a syntax error. Cause: the
script was edited on disk while bash was still reading it, which corrupted the
tail of the running copy. Nothing real was lost — `nothread` and `f32` had already
completed and saved, and the branch arms launched correctly at 11:05. The script
on disk is valid. **Do not edit a shell script while an invocation of it is
running.**

**2026-09-22 11:10** — First results read: `nothread` and `f32` compared, recorded
as X1. `f32` wins on every outcome that matters while losing on `inf_du`. Control
(`base`) still missing, so the finding is provisional.

**2026-09-22 11:20** — This document, `EXPERIMENTS.md` and `PLAYBOOK.md` created.

**2026-09-22 11:30** — Goal set (recorded at the top of `PLAYBOOK.md`): a
defensible defaults recommendation plus a settled answer on what Phase 2 should
converge to, by Friday 2026-09-26.

**2026-09-22 11:40** — D2 turned out to need no replay. The arms already run under
`minimize_ipopt_logged`, so the per-iteration traces were sitting in the handoffs
the whole time; `scratch/estimators/acceptable_forensics.py` reads them. This is
worth remembering for future items — check what the pipeline already records
before building a driver to record it.

**2026-09-22 11:45** — X2: feasibility is the binding leg on every capped solve
(`inf_pr` under `1e-4` on 0.00 of tail iterations, against 0.60–0.90 for the dual
leg), the line search is healthy, and `ml_sep` is the only constraint group ever
violated at the end. **Gate fired** — stopped for a re-rank rather than continuing
into D3/D4 or the smoothness items. D3 deprioritized and D5 added as the likely
new critical path.

**2026-09-22 11:50** — I1 done (X3), on branch `smooth-model-p1trace` in the
`-p1angle` worktree, which was idle and at the same commit as `smooth-model`. It
was used deliberately rather than `-smooth`: the branch arms are running from
`-smooth`, and multiprocessing spawn workers re-import on respawn, so editing that
tree's source mid-run is the same hazard as editing a running shell script.

**2026-09-22 11:52** — `base` confirmed healthy after an hour with nothing in its
log: three workers at ~97% CPU, each on exactly one core, which is also the first
live confirmation that the pinning fix does what it claims. It is slow because
every candidate runs the full 1000 iterations, not because it is stuck.

**2026-09-22 12:05** — D3 and D4 answered (X4), both negative: the active set at a
returned point is 1-3 well-conditioned rows, so no degeneracy; and the float32
stationarity floor is ~1e-5, an order below `ip_tol`, so C1 is off the list.

**2026-09-22 12:10** — X5 found the mechanism's shape. `inf_pr` never gets below
0.0104 in any capped run against a `ip_cvtol` of 1e-4, so the feasibility leg is
unreachable rather than narrowly missed; and a third of iterations satisfy every
constraint group while `inf_pr` at those same iterations is still above 0.01. The
two disagree by ~100x, which points at constraint scaling (C3, promoted).

**2026-09-22 12:20** — User closed the gate: C3 is the critical path, F1 and C2
dropped. D5 was found to have been added to this board and cited by X5 without
ever being defined in the playbook; it is defined there now, as the diagnosis C3
depends on. Adding an item to the board without writing it into the plan is the
failure mode the three-document split exists to prevent.

**2026-09-22 12:25** — D5 answered, and it retracted the two findings that set the
current ranking (X6). `inf_pr` is IPOPT's *internal* primal infeasibility — the
constraint-to-slack mismatch — not the unscaled violation `ip_cvtol` governs. The
proof needs no re-solve: 13 of the 20 runs returned `Solved_To_Acceptable_Level`
with `inf_pr` between 0.0039 and 6.23 at the final iteration, which IPOPT will not
do if the leg it tests is out of tolerance; at those same iterations the model's
own violation is exactly 0.0. So X2's "feasibility is the binding leg" and X5's
"~100× disagreement points at constraint scaling" both fall, and **C3 lost its
evidence one message after the user promoted it**. The real gate is
complementarity, left at IPOPT's 1e-2 default — the barrier stall X0 had already
recorded and the ranking set aside. Lesson worth keeping: every reading so far has
come from a column whose definition was assumed rather than checked; X2 would not
have survived one look at what the solver returned as its exit status.

**2026-09-22 12:30** — The `arc_sep` `NaN` turned out to be a misaligned split, not
an empty group (X7): `_live_group_sizes` reports live rows while `drop_dead_rows`
defaults to False and leaves all 402 in the vector. Per-group attribution has been
wrong on default settings the whole time, which is the second reason X2 does not
stand. Max-over-groups is unaffected, so X6 survives it. Fix deferred — the module
is imported by the running arm workers.

**2026-09-22 12:40** — Read `dev/PHASE2_CONDITIONING.md` properly and found it had
already established the `inf_pr` identification, the unscaled basis of the three
tolerance tests, and the deliberate choice of `acceptable_tol = 5.0` with its
measured effect (X8). X6 re-derived that; what it genuinely adds is that the
*callback's* `inf_pr` is internal regardless of `inf_pr_output`, so every reading
taken from `diag_hist` inherits it, plus the complementarity leg. The preamble of
`EXPERIMENTS.md` says to read that document first and I did not — the cost was a
session spent on a settled question. X7's fix passed the parity harness (poses
bitwise identical 3/3, ranked order identical).

**2026-09-22 12:40** — Goal criterion 2 closed (X9). D3 and D4 stood from X4; D2's
sentence was rewritten now that X2 is retracted; and *is `inf_du` the right
target* is answered **no** — it is a gate through `acceptable_tol`, not a target,
because `kept` ignores exit codes and X1's `f32` keeps 9/10 against `nothread`'s
4/10 while being four times worse on median `inf_du`. Yield is the target, which
is also why criterion 3 requires the 40-candidate protocol.

**2026-09-22 12:50** — Gave the item IDs a legend in the playbook and re-lettered
the three added today. `D`/`I`/`F`/`C` kept their meanings, `B` added for a defect
found in passing, and `X` noted as a result entry rather than an item. C7 became
**B1** — it was a bug fix, never conditional and never gated, so the `C` was
wrong. C6 was a decision for the user, not work, so it lost its ID and lives in
"Blocked on the user"; C5 is unchanged. The legend also states that an ID must be
defined in the playbook before it is used, which is what D5 violated.

**2026-09-22 13:35** — X10 confirmed C5's premise against IPOPT's own
`get_current_violations`, not the `mu` proxy: on candidate 11367 the three legs
reach a run of 6 against the 8 needed at the unset `acceptable_compl_inf_tol`
default of 1e-2, and 10 at 5.0. Also settled three side questions — IPOPT's
unscaled violation equals `max(0, -group_min)` exactly (so X7's misalignment
cannot disturb a max over the union), `inf_pr` is 230x that violation while
`inf_pr_output` sits at its default "original" (so the callback ignores the
option, which X8 could only infer), and `inf_du` equals the unscaled Lagrangian
gradient exactly, so it is the one column that can be read at face value.

**2026-09-22 13:43** — X11 closed the mechanism: with `acceptable_compl_inf_tol`
at 5.0 instead of its unset 1e-2 default, candidate 11367 exits
`Solved_To_Acceptable_Level` at 120 iterations against 1000 capped, at a point
with violation exactly 0. The arm also set `acceptable_dual_inf_tol`, but that
went 1e10 -> 5.0, which only tightens and so cannot cause an earlier exit — the
effect is complementarity alone. What remains for C5 is not the mechanism but the
yield question: whether the 880 saved iterations cost `kept`, coverage or
objective, which only the 40-candidate pair can answer.

**2026-09-22 13:59** — `base` finished and inverted the picture (X12). The control
keeps 10/10, so every intervention has a negative signed effect: `f32` -1,
`blend_angle` -3, `nothread` -6. **X1's headline is retracted** — it read `f32` as
winning because the control was missing, which is exactly the provisional caveat
it carried. The stopping rule is met and cannot be unmet, since no arm can beat a
control at the ceiling. C5 is unaffected: it is a compute saving to be validated
as yield-neutral, not a yield improvement, and the arms say nothing about it.

**2026-09-22 14:20** — X13 corrects X12's framing, on the user's challenge.
`kept` is a feasibility gate, not a score, so "-1" means one plan is unusable,
not slightly worse; the counts stand, the word does not. Measured on the real
grids: bf16's interpolation noise at the zero level set is median 1.5e-4 mm
against `ip_cvtol` 1e-4 — under bf16 the solver chases a tolerance finer than the
constraint can resolve, and `f32` removes that. But `f32` is not more accurate
geometry: its one loss (1282) is a different basin 11.3 away in pose space where
its own model claimed +0.0619 mm of clearance and FCL found a collision, a gap
~400x the bf16 noise and set by the 0.2 mm voxel pitch, which dtype cannot touch.
The case for `f32` is conditioning (390 vs 656 median iterations, constraints
satisfied 8/10 vs 6/10), and it still needs the 40-candidate pair, not the gate.

**2026-09-22 14:22** — Process error (the event itself was 14:13), mine, no result affected. `blend` and
`blend_smooth` were never killed at 13:40; they were mid-candidate and are still
running from 11:05. I misread their logs as crashed and launched duplicates onto
the same output paths, which truncated both live logs before I killed them four
minutes later. The handoffs are written at exit and are untouched; the lost
per-candidate lines are preserved in `837772_cpu_blend_rescued.log`. The lesson is
the one already on the board in the other direction: check `ps` start times before
concluding a run is dead, not just the log's last line.

**2026-09-22 14:30** — X14 answers what Phase 2 converges to: nothing. Zero of 20
solves across `base` and `f32` reached `Solve_Succeeded`. `ip_acc_tol` 5.0 against
`ip_tol` 1e-4 never binds, and `acceptable_dual_inf_tol` is at its unset 1e10
default, so the acceptable test enforces only violation <= 1e-4 for 8 iterations
plus complementarity <= 1e-2 — it certifies feasibility, not optimality. Terminal
`inf_du` on acceptable exits runs 0.7–5.0, i.e. 7,000x–50,000x `ip_tol`. This
revises X13: `f32`'s 390-vs-656 iteration saving is partly stopping sooner, since
it exits at a *higher* median `inf_du` (1.30 vs 0.93). C5 is now suspect for the
same reason and must be judged on objective and coverage, not yield alone.

**2026-09-22 14:35** — X15, from auditing what had actually been implemented. The
six arms are two groups on different code: `base`/`nothread`/`f32` launched 08:48
from the main checkout, which lacks `bab0970` (09:07); `blend`/`blend_angle`/
`blend_smooth` launched 11:05 from smooth@`3b83c55`, which has it. There is no
common control, so criterion 1 is unmeetable as written and `blend_angle`'s -3 is
invalid — its control is `blend`. X13 also re-derived `bab0970`, which already
fixes the bf16 blend defect it reported; re-measured, the fix cuts contact-band
noise 3.8x (1.5e-4 -> 3.9e-5), putting bf16 under `ip_cvtol` rather than over it,
which removes X13's stated reason to prefer `f32`. Nothing from X13/X14 had been
implemented; B1 is still stranded on `smooth-model-p1trace`.

**2026-09-22 14:37** — Log hygiene: three entries above were stamped from
assumption rather than the clock, twice landing in the future, and are corrected
to 14:20 / 14:22 / 14:30 / 14:35. The stamps are the only ordering this log has,
so a guessed one is worse than none.

**2026-09-22 14:45** — X16. `blend` finished, giving group B its own control, and
it settles the dtype question against the flip: bf16 on the fixed tree satisfies
constraints 10/10 with 1 cap, beating fp32 on the broken tree (8/10, 3 caps). The
lever was `bab0970`. X14's "stationarity is unchecked" is also corrected —
`acceptable_tol` binds it at 5.0, and all 28 acceptable exits sit under it (max
4.970). New and more interesting: terminal `inf_du` orders `kept` exactly across
the four arms (rho −1.000), and within candidates the stationarity rank is
monotone in loss rate (0/1/2/3 of ten). Hypothesis: conditioning gains buy earlier
exits, and earlier exits cost feasibility. The 14:36 arms (`ip_acc_du_tol` 1e10 /
1.0 / 0.1) test it directly — tightened arms should keep more and iterate more.

**Implemented this session** (was: nothing). `6380f2d` on `smooth-model-p1trace`
adds `ip_acc_du_tol` and `ip_acc_compl_tol`, both at IPOPT's defaults so the
acceptable test is unchanged; parity bitwise on poses and ranking. The two scratch
callers of `_live_group_sizes` are moved to B1's 3-arg signature. Still open: B1
itself is unmerged to `implant-hole-walls`, and `run_arms_cpu.sh` still truncates
a live log with `>` — unfixable while it is running.

**2026-09-22 15:12** — X17. `blend_smooth` exited 0 at 15:04 (not crashed; a pool
worker died and was replaced), completing group B. It beats its control `blend` by
**+2 on kept, 10/10** — the first arm in the series to beat a control, so the
stopping rule has NOT fired within group B. But it gets there with 643 median
iterations against 474, three caps against one, and constraint satisfaction down
to 8/10, which is the X16 mechanism rather than an intervention effect. The
stationarity ordering now spans five arms, two trees and three interventions at
rho −0.975 (p 0.0048); the two 10/10 arms, `base` and `blend_smooth`, agree to two
digits on `inf_du` and iterations despite sharing nothing else. The `du` arms
decide it: if tightening alone reaches 10/10 on the plain model,
`blend_smooth`'s win is a stopping-time artifact.

**2026-09-22 15:20** — X18 withdraws X16/X17's headline. Completing the `nothread`
read (criterion 1) breaks the stationarity ordering: it has the lowest `inf_du`
of any arm (0.13) and the worst kept (4/10). With all six arms Spearman is
**−0.116, p 0.827**; X17's −0.975 came from dropping `nothread`, an exclusion I
made without noticing or justifying it. The comparison was also never valid —
`inf_du` is the Lagrangian gradient of each arm's own objective in its own
coordinates, and three arms change one or the other. D2 is answered: `inf_du` is
honest within a fixed objective (X6) but is **not** a cross-arm ranking metric.
`nothread`'s durable result is instead 5/10 model-optimistic cases — it converges
well to the wrong problem, which is the series' strongest evidence that the
threading reward is load-bearing. The `du` arms stay valid: they vary only the
tolerance, so their `inf_du` are commensurable.

**2026-09-22 15:14** — Criterion 4 re-verified end to end. I1 was already built
and bitwise-parity-checked (X3, `abeb9f0`/`81fbefe`), but that predated `6380f2d`
on the same branch, so the suite was re-run with it in: **906 passed, 2 skipped**
(904 before; B1's parametrized regression test adds the two). Phase-2 parity for
`6380f2d` is separately bitwise on poses and ranking. I2 stays GPU-blocked —
`nvidia-smi` read 5267 MiB at 14:35, above the 2.5 GiB bar.

**Criterion status.** 1 met, with X15's caveat that the six arms are two groups
with separate controls. 2 met by X9, confirmed by X18. 4 met. **3 is the only
open criterion**: the 40-candidate x 2 protocol has not run and cannot start
while eight workers are busy. The setting it should test is C5's
`acceptable_compl_inf_tol` (now `ip_acc_compl_tol`, wired in `6380f2d`), which is
the one live candidate for a default change — the dtype flip is ruled out by X16
and the threading reward is ruled in by `nothread`'s 5/10 model-optimistic rate.

**2026-09-22 15:20** — X19, from preparing criterion 3. `PHASE2_CONDITIONING.md`
states that kept rate is **not resolvable** at this scale — kept candidates clear
by a median +0.021 mm, the order of the sampling gap, so "differences of a few
candidates between settings are noise". By that standard only `nothread`'s −6 is
real; `blend_smooth` +2, `f32` −1 and `blend_angle` −1 are all noise, and X17's
claim that `blend_smooth` was the series' first genuine win is withdrawn. The
metric with resolution is run-to-run pose divergence, and on it the **current
defaults already sit at the optimum** (0.000 divergence, 40/40 bitwise, 100%
agreement), with the smooth reward among the worst arms measured. The open work
for criterion 3 is therefore narrow: run the 40x2 on `ip_acc_compl_tol`, the one
setting the benchmark has never covered. Second time this document has overturned
a conclusion reached without consulting it.

**2026-09-22 15:55** — Criterion 3 started. `du_p1` and `m_smooth` stopped on the
user's call (whole trees killed, verified no reparented orphans this time), which
freed four workers; `du_ctl` and `du_1` continue. The benchmark runs the two
`ip_acc_compl_tol` settings — default 1e-2 and C5's 5.0 — over 40 candidates from
`bench40_ranks.txt` (the first 40 of `benchmark_ranks.txt`, which holds 360; the
prior art does not record which 40 it used, so this set is stated rather than
matched). Outputs `scratch/estimators/837772_b40_<arm>.json.gz`.

**Methodological caveat, recorded before the result.** The protocol's headline
metric is run-to-run pose divergence, but that column was built on GPU with MPS,
where the nondeterminism is real. These runs are CPU with single-threaded pinned
workers and look deterministic — `du_ctl` reproduced `blend` exactly on its first
two candidates, across different trees. If CPU runs are bitwise reproducible, the
divergence column is trivially 0.000 for every setting and discriminates nothing,
so the pair is scored on what retains resolution here: median iterations, cap
rate, exit distribution and kept. That is also what C5 claims to improve, so the
substitution does not dodge the question. `run_benchmark.sh` could not be used —
it predates the `RUTTER_` env migration and the JSON payload migration, and is
GPU/MPS, which is blocked at 5267 MiB.

**16:01** — X20: `du_ctl` reproduced `blend` on all six ranks it has finished,
to every printed digit, across two launches 3.5h apart on different cores and
different trees. Two consequences: the CPU path is reproducible, so the 40x2
protocol's repeat leg is dropped with a reason rather than by omission; and
`6380f2d` is parity-clean on six real candidates, a stronger check than the
harness subject.

Also confirmed the b40 is protocol-conformant on the iteration cap —
`run_arms_cpu.sh` hardcodes `RUTTER_P2_ITER=1000` and the conditioning doc's
table is at a 1000 cap, reference `adopted defaults` = median 1000 iters, 42/80
at cap. Pre-registered the compl prediction in RECOMMENDATION.md before the data
lands: the doc's own unexplained 42 cap hits are attributed there to
`acceptable_compl_inf_tol` = 0.01 on three instrumented logs, called "untested
as a fix - the option is not currently exposed as an env knob". `6380f2d`
exposes it; this pair is that test.

Machine at 16:01: 8 compute workers, load 9.45. b40 `compl_ctl`/`compl_5` past
warmup, first candidates in flight. `du_ctl` 6/10, `du_1` 2/10. `nothread_fx`
waiter (pid 2275070) alive and still parked on `du_ctl`'s handoff.

Projection, from observed per-candidate times of 400-3900s: the b40 is 20
candidates per worker at 2 workers per arm, so it lands overnight, not today.

**16:04** — X21: the dual-bound pair has two ranks complete in both arms, and the
answer to "what does a finite `acceptable_dual_inf_tol` cost" is 3.7x and 17x
wall time — buying threading margin and coverage, with rank 39 moving off a
zero-margin exit (`max_g_thread` -0.000 to -0.146). Corrected a reading error in
passing: `g` in the per-candidate line is `max_g_thread`, the threading
constraint, not the objective. Recommendation still keeps 1e10, now on cost
grounds rather than on the knob being inert.

Mechanism note for the du arms: their `all arms done` line at 15:53 is spurious.
Killing the `du_p1`/`m_smooth` subshells let the parent's `wait` return while the
python pools (2181709, 2181711) reparented and kept running — the same orphan
pattern as the 14:13 duplicates. `du_ctl`/`du_1` are unaffected and will write
their handoffs, so the `nothread_fx` waiter still fires correctly, but that log
line cannot be used as a completion signal. The b40 runner tree was never killed
and is intact; its waiter was switched to depend on both output files existing,
with a guard that exits if the runner disappears first.

**16:24** — Moved the criterion-3 benchmark to the GPU and killed the CPU
version. The card read 684 MiB with no compute apps, well inside the 2.5 GiB
gate; one anomalous sample showed 8370 MiB / 100%, but the full process table
and five consecutive samples all read 684 MiB with only gnome-shell, Xwayland
and firefox holding it.

Two things this fixes, beyond speed.

**The CPU run was on the wrong candidates.** I had built `bench40_ranks.txt` as
the first 40 of `benchmark_ranks.txt`, on the belief that the prior art does not
record which 40 it used. It does: `837772-config/repeat_ranks.txt`, 40 entries,
and it is a spread starting 12 15 22 35 39 49 where mine started at 202. The CPU
run would not have been comparable to the table it was meant to be scored
against. The GPU run uses `repeat_ranks.txt`.

**The repeat leg is back.** X20 justified dropping it on CPU because that path is
deterministic. The protocol's divergence and bitwise columns come from GPU/MPS,
so on the card the second run is meaningful again and the run is the literal
40x2: `compl_ctl` and `compl_5`, each solved twice.

Speed, from the protocol's own logs: GPU candidates take 10-64s against CPU's
400-3900s, about 60x. Four runs of 40 should land in well under an hour rather
than overnight.

Killed the b40 tree whole - all 15 pids collected by descent from the five
runner parents, verified zero survivors and no reparented python. `du_ctl` and
`du_1` are untouched and still progressing on cores 10-13.

New driver `scratch/estimators/run_arms_gpu.sh`: same arm syntax as the CPU one,
but arms run sequentially because one pool at 3 workers x 0.28 already claims
most of the card, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` is exported in the
launch env, which `phase2.py` hard-checks.

**16:27** — X22, and a replan off the back of it. Ranks 35 and 39 are in both
rank files, so the CPU and GPU baselines solve the same two problems under the
same settings — and they disagree on threading margin, FCL clearance and
coverage by more than most of the settings under test move them. Two
consequences.

First, X21's sharpest claim is punctured: the zero-margin exit at
`max_g_thread` = -0.000 that it built on does not reproduce on the GPU, where
the same candidate reaches -0.196. X21's cost figures stand, since both its arms
were on one platform, but the margin story needs the card before it means
anything.

Second, the whole arm series is CPU and the reference table is GPU, so every
effect in it is indicative only. With the card at ~60x this is cheap to fix
properly, so a second GPU batch is chained behind the compl one: `nothread`,
`f32`, `smooth` and `angle` at n=40 on `repeat_ranks.txt`, against
`compl_ctl_a` as the baseline. That takes the series from CPU n=10 on a
hand-picked tail to GPU n=40 on the protocol's own candidates, which is what
criterion 1 actually wants. Verified the four override names against
`settings.py` first (`lam_thread`, `grid_dtype`, `smooth_thread_reward`,
`spin_param`); the diag header prints all four, so each arm's log confirms the
override landed rather than silently running the baseline.

Retired the queued CPU `nothread_fx` waiter (2275070), superseded by the GPU arm.
Left `du_ctl`/`du_1` running: both are on one platform so their cost comparison
stays internally valid, and finishing gives a complete n=10 for it.

Throughput: 11 of 40 in about 2.5 min of solve time. Four compl arms land near
17:00, the four intervention arms near 17:40.

**16:35** — PAUSED all GPU work at the user's request; the card is wanted for a
higher-priority test. Killed the `run_arms_gpu` tree whole (13 pids by descent,
zero survivors), shut down the MPS daemon the script had started, since a
mid-run kill skips its own cleanup, and stopped both waiters including the
chained intervention batch so nothing fires later and takes the card back.
Card reads 8816 MiB free, 0% utilization, no MPS.

Correction to the 16:22 entry. That anomalous 8370 MiB / 100% sample was
probably real, not a misread: a compute process from another project is on the
card now, with a pid in the same range as the ones just killed, so it was
spinning up at the time. Five consecutive 684 MiB samples caught idle moments
between its bursts, and the full process table showed only graphics clients
because the compute app had not attached yet. Sampling a few seconds apart is
not a safe way to clear the 2.5 GiB gate; the process table and a longer window
are.

Banked before the pause:

- `compl_ctl_a` — **complete**, 40 candidates, 31 feasible ranked, handoff
  written. This is a full baseline arm at n=40 on the protocol's own ranks.
- `compl_5_a` — 2 of 40, killed mid-arm, no handoff. Its per-candidate lines
  survive in the log.
- `compl_ctl_b`, `compl_5_b`, and all four intervention arms — never started.

Still running, CPU only and not touching the card: `du_ctl` and `du_1` on
cores 10-13.

To resume, both batches relaunch as they were; `compl_ctl_a` is done and can be
dropped from the arm list, so the compl batch restarts as
`compl_5_a`, `compl_ctl_b`, `compl_5_b`.

**16:39** — GPU-free work while the card is reserved.

X23, from the one arm that completed: the baseline at n=40 on the protocol's own
candidates does **not** reproduce the doc's row for the same configuration —
median iterations 737 against 1000, cap 11/40 (27%) against 42/80 (52%). Checked
the obvious confounds first: the handoffs share all 40 candidate indices with
`837772_repeat_iptol_a`, so the pool and ranks still index the same problems
across the pickle-to-JSON change; and `619cb3f` only exposed the threading-reward
weight as a setting, so the reward was active and hardcoded in the doc-era runs
and `lam_thread=1.0` is not a change against them. What is left is the code
between 2026-09-16 and now, `bab0970` foremost. The procedural consequence is the
important one and is now in RECOMMENDATION.md: the table cannot score anything
run on today's tree, so every comparison needs its own same-tree control. Median
`inf_du` at exit of 1.36 also confirms X16 at n=40.

Criterion 2 verified rather than assumed: D2 ("not feasibility"), D3 ("not
rank-deficient") and D4 are each answered in EXPERIMENTS.md, and the `inf_du`
question carries an explicit "No, not as a cross-arm ranking metric". Neither
gate condition is tripped — D2 did not find `inf_pr` at fault and D3 found the
active Jacobian well conditioned — so the smoothness program is not aimed at the
wrong thing and no re-rank is owed.

Criterion 4: I1 is built and committed (`abeb9f0`, `81fbefe`), p1angle is clean
at `6380f2d`. Test suite running to confirm the checks-green half.

Criteria 1 and 3 remain open and cannot be closed without the card. Everything
needed to resume is unchanged; `compl_ctl_a` is the control the rest will be
scored against.

**16:41** — Added `scratch/estimators/resume_gpu_arms.sh`, which refuses to
start if any compute app holds the card or if it reads over 2500 MiB across a
30s window. Verified the guard by dry-running it now: it correctly refuses
against the job currently on the card. It encodes the lesson that a few samples
seconds apart do not clear the gate. `compl_ctl_a` is excluded from the resume
list since it completed.

Stopping here. Criteria 2 and 4 are verified done. Criteria 1 and 3 need the
GPU, which is on hold at the user's request, so they cannot be closed in this
window and no CPU substitute is worth running — X22 established that CPU results
cannot be compared against the GPU reference anyway.

One item for the user, because it changes a success criterion rather than a
result: criterion 3 says the recommendation must be "validated on the
40-candidate x 2 protocol from dev/PHASE2_CONDITIONING.md". X23 shows that
document's table no longer scores this tree. The protocol itself is still the
right instrument and is set up and half-run; what is dead is using the doc's
recorded numbers as the comparison. Criterion 3 should be re-worded to validation
against an own same-tree control, which is what `compl_ctl_a` is.

**17:11** — X24 closes the dual-bound question. `du_ctl` finished 10/10
(acceptable 9, cap 1, median 474 it, kept 8/10, median `inf_du` 1.29) — kept
8/10 matches `blend`, as X20 predicted. With `du_1`'s 4 log-only candidates
there are now four shared ranks, and rank 57 is the one X21 could not see: under
the tightened bound the solve ends with the threading constraint *violated*
(g +0.000) and FCL rejects a plan the control keeps. Kept 3/4 against 2/4.

The cost structure also reads differently: the 3.7x/17x multipliers came from the
two candidates where the control exited early, and where it already ran long the
two arms are comparable. The tightened bound removes the early acceptable exits
and pushes everything toward the cap.

Verdict: keep `ip_acc_du_tol` at 1e10. Corrected the recommendation, which had
carried X21's cost/margin-trade framing, and the convergence section, which had
claimed tightening stationarity buys margin — X22 and X24 both contradict it.
Margin is a formulation question, not a solver-effort one.
