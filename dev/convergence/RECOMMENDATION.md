# Phase-2 production defaults — recommendation

**Status 2026-09-23 08:16:** settled for yield; **reopened for throughput.** X26
found X25's structural finding wrong — caps are not solves that never reached
feasibility, they reach it at iteration 24 and hold 60-iteration feasible runs —
so `ip_acc_compl_tol` is un-retired and needs its arm run. The yield half is
now *stronger*: the reproducibility property four `keep` rows cite reproduces on
this tree (0.000 divergence, 40/40 bitwise), and `nothread` confirms on the
reference platform at -10 kept.

**Read every "measured ceiling" and "98.5%" claim below as retracted pending
that arm.** They are left in place so the reasoning that failed stays legible.

## Executive summary

**For plan yield: change no default.** The one real improvement found here,
`bab0970`, is a bug fix already committed, not a setting. Four interventions
(`nothread`, `f32`, `smooth`, `angle`) land inside a noise floor the benchmark
cannot resolve, and two knobs that looked live — `grid_dtype` and
`ip_acc_du_tol` — each closed *against* change once measured. Yield is not where
the remaining gains are.

**For throughput, they are not either — and that is the new finding.** The two
candidates this investigation had ranked first and second were measured against
the control on 2026-09-22 (X25) and both collapsed:

| was | change | measured ceiling |
|---|---|---|
| rank 1 | `ip_acc_compl_tol` 0.01 → 5.0 | **at most 1 of 40 solves.** Ten of the eleven capped solves are not feasible at the cap — five still hold strictly violated constraints, worst margin −4.911. They are still searching for a feasible point; no acceptance tolerance reaches them |
| rank 2 | `ip_acc_iter` 8 → 3–5 | **~1.5% of wall time.** It is genuinely binding — the trailing feasible run is exactly 8 on 17 of 27 acceptable exits — but the median solve is 570 iterations and is feasible for 8 of them |
| rank 3 | `ip_hist` 60 → 20 | unchanged: a **trade, not a win** — memory 60 is what completes bitwise reproducibility to 40/40, memory 20 gives divergence 3.523 and 0/40 |

**Why they collapsed is the useful part.** A Phase-2 solve spends roughly **98.5%
of its iterations reaching a feasible point and 1.5% proving that point
acceptable.** Every acceptance tolerance — `ip_acc_compl_tol`, `ip_acc_iter`,
`ip_acc_tol`, `ip_acc_du_tol` — is tuning that last 1.5%. There is no throughput
left in the family, and the reason the doc's cap analysis looked promising is that
it was written when caps ran at 42/80; on today's tree they run at 11/40 and are
overwhelmingly infeasibility, not complementarity.

**So throughput is a formulation question, not a tuning one.** What would move it
is reaching feasibility faster — the basin the Phase-1 seed lands in, the
conditioning of the constraint set, the threading reward's shape. That is the same
place X24 located margin, and it is outside what a settings sweep can reach.

**Do not touch these**, despite how the table reads in isolation: `drop_dead_rows`
looks good at legacy settings (571 median) but in *this* configuration gives 66/80
caps against 42/80; the smooth reward's 638 median comes with 0/40 bitwise.
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
| smooth threading reward | **do not enable in Phase 2; untested in Phase 1** | X33 at n=40 on the card moves six measures the wrong way: strictly feasible 27 against 31, paired objective +0.027 worse on 14 of 24, `reach` 40 against 28, feasible-at-exit 78% against 85%, model-optimistic 10 against 8, and — on the group its own reward acts on — thread violated 26.9% against 20.6%. Directionally negative, not unresolved. The earlier n=10 divergence argument is withdrawn: 0/40 bitwise is expected for a changed objective and says nothing. **The flag is also wired into Phase 1 and has never been run there**, where the case for it is strongest — RProp reads only the gradient's sign, so a kink reverses its step, while IPOPT can average one out. I2 |
| `spin_param` | **`angle` leads; awaiting its repeat** | X32 at n=40 on the card is the only change measured here that improves both goals at once: strictly feasible 33 against 31, paired objective −0.114 with only 9 of 26 worse, `reach` halved 28 → 14, caps 11 → 1, feasible-at-exit 100%. The gain lands in **thread**, 20.6% → 10.7%, which is where `arctan2(sy, sx)` acts — its singular origin is interior to the ±1.1 box. The old −1-kept verdict came from n=10 on `kept`, the metric that cannot resolve this. Not yet adopted: `angle_b` is needed for run-to-run reproducibility, and Phase 1 implements the setting by a different mechanism and is untested (I2) |
| `ip_acc_du_tol` | **keep `1e10`** | `acceptable_tol` already bounds the dual leg at 5.0, so the field is not a hole (X16). Tightening it to 1.0 is not a cost/margin trade as X21 read it: over four shared candidates it keeps 2/4 against the control's 3/4, losing one outright to a *violated* threading constraint, and the margin gain it shows on two others did not survive a platform change (X22, X24). It removes the early acceptable exits and pushes nearly everything to the cap. Valuable as a diagnostic, not as a default |
| `ip_acc_tol`, `ip_cvtol`, `ip_hist`, `drop_dead_rows` | **unchanged** | Chosen on the 40×2 benchmark, where they gave 0.000 divergence, 40/40 bitwise and 100% label agreement against every alternative measured. That evidence is from the 2026-09-16 tree and X23 shows the baseline has moved since, so it justifies *keeping* them — the burden is on a change — but it is no longer a live measurement of today's code |
| `ip_acc_compl_tol` | **a trade, and the user's call — do not adopt silently** | X29 measured 5.0 at n=40: all 11 caps and both infeasible exits become acceptable exits, median iterations 737 → 164 and wall time ~7x, strictly feasible 31 → 37. X31 then measured the cost X29 omitted — the paired objective is **+0.92 worse on 27 of the 31** candidates both arms solve. It buys feasibility with objective. Its premise is also gone: `waste` is 0%, so the long solves were still improving, not spinning. Earlier reasoning: X25 retired it on the reading that caps never reach feasibility. X26 measured that directly and it is false: caps reach feasibility at iteration 24 and hold runs of 60 against an `acc_iter` of 8, so the constraint-violation leg was satisfied and something else stopped them. Complementarity is the pre-registered candidate and `diag_hist` cannot settle it — `mu` is the barrier parameter and `inf_pr` is scaled |
| `ip_acc_iter` | **unresolved** | X25's retirement is withdrawn (X27): "feasible for 8 of its 570 iterations" reported the *trailing* run, and the total is 245 of 570, so the ~1.5% figure is wrong by about 30x. Note also that 8 is **not** IPOPT's default, which is 15 (X28) — this project already set it below stock |
| `ip_hist` | **keep `60`** | 50 would also clear the variable count (44 at two arcs, 45 at three), so truncation is not what separates them — and the cost of the extra 10 pairs is ~7 KB of storage and an O(mn) two-loop recursion that is invisible behind a 402×45 Jacobian. 60 is the value the 40/40 bitwise result was measured at, and that result is empirical and all-or-nothing: the same m ≥ n reasoning would have called memory 20 a mild degradation, and it measured 0/40 bitwise. Do not trade a measured property for a theoretical one to save nothing |

### The pre-registered test, resolved

`PHASE2_CONDITIONING.md` ended its cap analysis with an explicit loose end: of the
adopted defaults' 80 runs, 42 hit the cap and "the remaining 42 are not
explained." It measured complementarity at exit of 8.9e-3, 1.0e-1 and 1.6e-3
against a default `acceptable_compl_inf_tol` of 0.01, and concluded the fit "rests
on three logs and is untested as a fix: the option is not currently exposed as an
env knob." `6380f2d` exposed the knob, and the prediction was fixed in advance:
either `compl_5` cuts the cap rate well below the control's, or the hypothesis is
dead.

**It came out dead, and without needing the arm.** X25 answered it from the
control's own unscaled constraint diagnostics: 10 of the 11 caps are infeasible at
the cap, so they would run to 1000 iterations under any complementarity bound. The
doc's three logs were a real observation on a tree where caps ran at 52%; on this
tree caps run at 27% and are dominated by solves that never reached feasibility.

One caveat worth carrying forward, since it cost two false starts here: `inf_pr`
and `inf_du` in `diag_hist` come from IPOPT's `intermediate_callback` and are the
**scaled** iteration-log values, while the acceptable test runs on unscaled
quantities — and `mu` is the barrier parameter, not complementarity. The only
unscaled feasibility measure recorded is `group_min`/`group_nviol`. Anything
comparing `diag_hist` to a tolerance must use those.

## What this recommendation is not backed by

- **`kept` on the capped-10.** `PHASE2_CONDITIONING.md` states kept rate is not
  resolvable at this scale — kept candidates clear by a median +0.021 mm, the
  order of the sampling gap, so differences of a few candidates are noise. Only
  `nothread`'s −6 clears that bar, and X26 confirms it at −10 of 40 on the card.

  **This cuts both ways, which X19 and X25 both missed.** An unresolvable metric
  licenses "unresolved", not "no effect". `smooth` and `angle` were retired on
  ±1–2 kept at n=10, and that is not evidence they do nothing. `angle` in
  particular is `PHASE2_CONDITIONING.md`'s own prescribed fix for a defect it
  documents in the *current default*: `(sx, sy)` is an exact null direction of
  all 402 constraint rows, and the bounds contain the origin where `arctan2`
  gradients diverge as 1/r. Judging a conditioning fix by a binary gate on
  marginal plans measures the wrong thing. The continuous measures — iterations
  to first feasible, feasible fraction, crossing count, margin distribution —
  are what should decide it.
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
