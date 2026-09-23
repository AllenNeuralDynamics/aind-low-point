# Phase-2 production defaults — recommendation

**Status 2026-09-22 17:15:** the yield recommendation is settled. Two throughput
candidates, `ip_acc_compl_tol` and `ip_acc_iter`, are unmeasured — the arms are
built and unrun, paused with the GPU handed to another job. See `RESUME.md`.

## Executive summary

**For plan yield: change no default.** The one real improvement found here,
`bab0970`, is a bug fix already committed, not a setting. Four interventions
(`nothread`, `f32`, `smooth`, `angle`) land inside a noise floor the benchmark
cannot resolve, and two knobs that looked live — `grid_dtype` and
`ip_acc_du_tol` — each closed *against* change once measured. Yield is not where
the remaining gains are.

**For throughput, they are.** Throughput is a separate question with better
evidence behind it, and it is the one worth spending the next hour of card time
on. Three candidates, ranked:

| rank | change | expected effect | evidence |
|---|---|---|---|
| 1 | `ip_acc_compl_tol` 0.01 → 5.0 | converts capped solves into acceptable exits | The conditioning doc's own unexplained 42-of-80 caps, with measured complementarity at exit of 8.9e-3, 1.0e-1 and 1.6e-3 against the 0.01 bound. X11 saw one candidate go 1000 → 120 iterations. **Untested** — the arm is built and unrun |
| 2 | `ip_acc_iter` 8 → 3–5 | exits sooner once all four legs hold | **Untested and unexplored.** A solve that has satisfied the acceptable test still runs 8 more iterations to prove it. Nothing before the exit changes, so determinism is untouched. Risk is exiting on a transient dip |
| 3 | `ip_hist` 60 → 20 | median 1000 → 780, caps 42/80 → 28/80 | Measured, but it is a **trade, not a win**: memory 60 is what completes bitwise reproducibility to 40/40, and memory 20 gives divergence 3.523 and 0/40 bitwise |

**Do not touch these for throughput**, despite how the table reads in isolation:
`drop_dead_rows` looks good at legacy settings (571 median) but in *this*
configuration gives 66/80 caps against 42/80; the smooth reward's 638 median
comes with 0/40 bitwise, and at memory 60 it is the worst row in the table.
`ip_acc_tol` is already at 5.0 and `ip_tol` is inert.

**The structural lever is determinism itself, and that is a policy call, not a
tuning one.** From the doc's table: `acceptable tol 5` alone gives median **187**
iterations and **0/80** caps at divergence 4.536, while the reproducibility
package this project adopted — no clearance bonus plus memory 60 — gives median
**1000** and **42/80** at divergence 0.000. Bitwise reproducibility is costing
roughly **5× the solve time**. Whether that is worth it is a question about how
this pipeline is used, and nobody has been asked it explicitly.

**Caveat on every number above.** They come from the 2026-09-16 tree, which X23
shows no longer reproduces: today's baseline is median 737 and 11-of-40 caps
against the table's 1000 and 42-of-80. The absolute figures have already moved
favourably and the *relative* structure is assumed, not verified, on this code.

**One footgun, unrelated to tuning.** `p2_iter` defaults to 200 in
`settings.py`, while `scripts/run_subject_overnight.sh` sets 1000. Measured
median iterations to an acceptable exit are 474–737, so anything calling
`phase2.run()` without setting the cap truncates well before most solves become
acceptable, and silently.

## What Phase 2 converges to

It does not converge. Across 20 solves on the capped-10, **zero reached
`Solve_Succeeded`** (X14). Solves exit on the acceptable test, whose legs here are
constraint violation ≤ `ip_cvtol` held for `ip_acc_iter` iterations, plus
complementarity, plus an overall-error bound of `ip_acc_tol` = 5.0 that in
practice binds the dual leg — all 28 acceptable exits sit under it, max 4.970
(X16). So an acceptable exit **certifies a feasible point, not an optimal one**,
and terminal `inf_du` of 0.7–5.0 is the expected output, not a symptom.

This is deliberate and `settings.py` documents it: feasible solves stall far above
`tol`, the acceptable path converts that into a clean exit, and the FCL gate
decides the plan. `inf_du` is a **gate to be set, not a target to be minimised**
(X9, X18) — it is honest within one objective and parameterization (X6) but is not
comparable across arms that change either.

**What that exit does not certify is margin.** Solves do land on the threading
constraint with nothing in hand — `max_g_thread` = -0.000 appears on more than one
candidate, feasible and kept but with no margin at all. That is a real property of
a feasibility-gated exit, on a gate whose kept candidates already clear by a
median of only +0.021 mm.

What does *not* follow, and what X21 wrongly inferred, is that tightening
stationarity recovers that margin. X22 showed the same candidate reaches -0.196 on
the GPU under the ordinary settings, so the zero-margin exit was substantially a
CPU-path artifact; and X24 found the tightened bound ending with the constraint
marginally *violated* and the plan rejected. Margin comes from the formulation —
the threading reward, or an explicit `RUTTER_THREADING_MARGIN_MM` — not from
making the solver work longer on the same objective.

## Per-setting

| setting | recommendation | evidence |
|---|---|---|
| `grid_dtype` | **keep `bfloat16`** | `blend` (bf16, fixed tree) beats `f32` (fp32, pre-fix): constraints satisfied 10/10 vs 8/10, caps 1 vs 3. The lever was `bab0970`, not dtype. Flipping also doubles grid memory, 296 → 593 MiB per worker (X16) |
| `lam_thread` | **keep `1.0`** | `nothread` is the only arm whose effect clears the noise floor: −6 kept, and 5/10 cases where constraints are satisfied and FCL still rejects. It converges cleanly to the wrong problem (X18, X19) |
| smooth threading reward | **do not enable** | +2 kept is noise at n=10 by the prior art's own standard; on the metric with resolution it is among the worst settings measured — 3.932 divergence and 0/40 bitwise on its own, and paired with memory 60 it is *the* worst configuration in the table at 6.989 (X19) |
| `spin_param` | **keep `unit_vector`** | `blend_angle` is −1 kept, inside noise, with no resolvable benefit |
| `ip_acc_du_tol` | **keep `1e10`** | `acceptable_tol` already bounds the dual leg at 5.0, so the field is not a hole (X16). Tightening it to 1.0 is not a cost/margin trade as X21 read it: over four shared candidates it keeps 2/4 against the control's 3/4, losing one outright to a *violated* threading constraint, and the margin gain it shows on two others did not survive a platform change (X22, X24). It removes the early acceptable exits and pushes nearly everything to the cap. Valuable as a diagnostic, not as a default |
| `ip_acc_tol`, `ip_cvtol`, `ip_hist`, `drop_dead_rows` | **unchanged** | Chosen on the 40×2 benchmark, where they gave 0.000 divergence, 40/40 bitwise and 100% label agreement against every alternative measured. That evidence is from the 2026-09-16 tree and X23 shows the baseline has moved since, so it justifies *keeping* them — the burden is on a change — but it is no longer a live measurement of today's code |
| `ip_acc_compl_tol` | **candidate, unmeasured — rank 1 for throughput** | The one setting the benchmark never covered. X11 showed it turns a 1000-iteration cap into a 120-iteration exit on one candidate. The arm is built and unrun; note it is a throughput claim, since kept is not resolvable at n=40 |
| `ip_acc_iter` | **candidate, unmeasured — rank 2 for throughput** | Holds at IPOPT's 8: the acceptable test must pass on 8 consecutive iterations before the solve stops. Those 8 are pure overhead on every solve that exits acceptably — 27 of 40 in the control arm. Nothing before the exit changes, so unlike `ip_hist` it cannot cost determinism. Never varied in this investigation or the prior art |

### The pending test, pre-registered

`PHASE2_CONDITIONING.md` ends its cap analysis with an explicit loose end: of the
adopted defaults' 80 runs, 42 still hit the cap, and "the remaining 42 are not
explained." It measures complementarity at exit of 8.9e-3, 1.0e-1 and 1.6e-3
against a default `acceptable_compl_inf_tol` of 0.01 — one exit ten times over the
bound, one at 89% of it, one inside — and concludes the fit "rests on three logs
and is untested as a fix: the option is not currently exposed as an env knob."

`6380f2d` exposes that knob. The running pair is therefore a direct test of a
hypothesis the doc stated and could not run, and its prediction is fixed in
advance:

- **If complementarity is what blocks acceptance**, `compl_5` cuts the cap rate
  well below `compl_ctl`'s and drops median iterations far under 1000.
- **If it is not**, both arms cap at the same rate and the hypothesis is dead —
  in which case the recommendation stands unchanged with no setting left to flip.
- **Either way**, the decision turns on whether kept survives, and kept is not
  resolvable at n=40. A cap-rate win with kept flat is a cost improvement, not a
  yield improvement, and is worth adopting only on that narrower claim.

Reference points from the table for `compl_ctl`: median iterations 1000, 42 of 80
runs at the cap. This run is 40 candidates once per setting, so the comparable
figure is out of 40.

## What this recommendation is not backed by

- **`kept` on the capped-10.** `PHASE2_CONDITIONING.md` states kept rate is not
  resolvable at this scale — kept candidates clear by a median +0.021 mm, the
  order of the sampling gap, so differences of a few candidates are noise. Only
  `nothread`'s −6 clears that bar. Every "+2" and "−1" in the arm series is noise
  and is not load-bearing here (X19).
- **The doc's table, as a reference for today's code.** X23 ran the baseline at
  n=40 on the protocol's own candidates and it does not reproduce the table's row
  for the same configuration: median iterations 737 against 1000, cap rate 27%
  against 52%, on candidate-for-candidate identical problems. The code moved
  between the two dates. So nothing run on today's tree can be scored against that
  table, and every comparison needs its own control from the same tree — which is
  what `compl_ctl_a` is, and why the batch was built around one.
- **Cross-platform comparison.** The whole arm series ran on CPU; the prior art's
  table is GPU/MPS. X22 shows the platform moves threading margin, FCL clearance
  and coverage on the same candidate under the same settings, by more than most of
  the settings under test move them. So every arm-series number here is indicative
  only against the doc's table, and the GPU is the reference platform. Re-running
  the interventions on the card at n=40 is cheap now and is the right way to
  settle this properly.
- **Cross-tree comparison.** The six arms ran on two trees split by `bab0970`
  (X15). Group A is `base`/`nothread`/`f32`; group B is `blend`/`blend_angle`/
  `blend_smooth`. `blend` is `base`'s configuration on the fixed tree, so group
  B's effects are effects against the baseline configuration; `nothread` is being
  re-run on the fixed tree to close the remaining gap.
- ~~The divergence column, and the missing repeat.~~ **Resolved by moving to the
  GPU.** An earlier CPU attempt had to drop the protocol's second run, because on
  CPU the path is deterministic (X20) and a repeat would return 0.000 and 40/40 by
  construction. The card turned out to be free, and the protocol's own runs were
  GPU/MPS — which is where the nondeterminism the divergence column measures comes
  from. The pending pair is therefore the literal protocol: both settings solved
  twice, on the protocol's own 40 ranks, with the divergence and bitwise columns
  intact. They are comparable **to each other and to `compl_ctl_a`**, the control
  from the same tree and platform — not to the doc's recorded table, which X23
  ruled out as a reference for this code.

## Open decisions for the user

1. **The threading reward.** Keeping it is the recommendation, but `nothread`
   shows the reward is what keeps the model honest rather than merely better —
   removing it produces plans the model calls feasible and FCL rejects. Whether
   the margin belongs in the objective or in `RUTTER_THREADING_MARGIN_MM` is a
   design question this investigation did not settle.
2. **F1**, unstarted by design; it needs approval before any work.
3. **B1** (`d790432`) is still unmerged to `implant-hole-walls`.
