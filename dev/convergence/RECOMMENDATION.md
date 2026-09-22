# Phase-2 production defaults — recommendation

**Status 2026-09-22 15:55:** complete except `ip_acc_compl_tol`, which the
40-candidate benchmark is measuring. Every other line below is settled.

## The recommendation in one line

**Change no default.** The one real improvement found in this investigation,
`bab0970`, is a bug fix that is already committed, not a setting.

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

**What that exit does not certify is margin.** X21 caught a candidate exiting at
`max_g_thread` = -0.000 — feasible, kept, and sitting exactly on the threading
constraint with nothing in hand. Re-solved under a tightened dual bound the same
candidate reaches -0.146. So the distance between the acceptable exit and
stationarity is not merely cosmetic: it is paid in margin, on a gate whose kept
candidates already clear by a median of only +0.021 mm. This is the strongest
argument in the series for keeping the threading reward, which is what pushes a
solution off that boundary while the solve is still running.

## Per-setting

| setting | recommendation | evidence |
|---|---|---|
| `grid_dtype` | **keep `bfloat16`** | `blend` (bf16, fixed tree) beats `f32` (fp32, pre-fix): constraints satisfied 10/10 vs 8/10, caps 1 vs 3. The lever was `bab0970`, not dtype. Flipping also doubles grid memory, 296 → 593 MiB per worker (X16) |
| `lam_thread` | **keep `1.0`** | `nothread` is the only arm whose effect clears the noise floor: −6 kept, and 5/10 cases where constraints are satisfied and FCL still rejects. It converges cleanly to the wrong problem (X18, X19) |
| smooth threading reward | **do not enable** | +2 kept is noise at n=10 by the prior art's own standard; on the metric with resolution it is among the worst settings measured — 3.932 divergence and 0/40 bitwise on its own, and paired with memory 60 it is *the* worst configuration in the table at 6.989 (X19) |
| `spin_param` | **keep `unit_vector`** | `blend_angle` is −1 kept, inside noise, with no resolvable benefit |
| `ip_acc_du_tol` | **keep `1e10`**, but as a cost decision, not because the knob is inert | `acceptable_tol` already bounds the dual leg at 5.0, so the field is not a hole (X16). Setting it to 1.0 is a real trade: on the two candidates complete in both arms it costs 3.7x and 17x wall time and buys threading margin and coverage, one of them moving off a zero-margin exit at `max_g_thread` = -0.000 (X21). At n=2 that is suggestive only, and 4-17x is prohibitive for a production sweep |
| `ip_acc_tol`, `ip_cvtol`, `ip_hist`, `drop_dead_rows` | **unchanged** | Chosen on the 40×2 benchmark, where they gave 0.000 divergence, 40/40 bitwise and 100% label agreement against every alternative measured. That evidence is from the 2026-09-16 tree and X23 shows the baseline has moved since, so it justifies *keeping* them — the burden is on a change — but it is no longer a live measurement of today's code |
| `ip_acc_compl_tol` | **pending** | The one setting the benchmark never covered, and the only live candidate. X11 showed it turns a 1000-iteration cap into a 120-iteration exit on one candidate; whether that costs yield is what the running pair decides |

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
  intact and directly comparable to the doc's table.

## Open decisions for the user

1. **The threading reward.** Keeping it is the recommendation, but `nothread`
   shows the reward is what keeps the model honest rather than merely better —
   removing it produces plans the model calls feasible and FCL rejects. Whether
   the margin belongs in the objective or in `RUTTER_THREADING_MARGIN_MM` is a
   design question this investigation did not settle.
2. **F1**, unstarted by design; it needs approval before any work.
3. **B1** (`d790432`) is still unmerged to `implant-hole-walls`.
