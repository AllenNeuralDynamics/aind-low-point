# Resume here — Phase-2 defaults investigation

Updated 2026-09-23 08:22. Read this file, then `RECOMMENDATION.md`. `STATUS.md`
is the running log and `EXPERIMENTS.md` the numbered findings (X1–X26).

## One paragraph of state

**Yield is settled; throughput is reopened.** X26 measured two things on the
card that the recommendation had been resting on indirectly, and both came back
clean: the baseline is bitwise reproducible on this tree (0.000 divergence,
40/40 identical, 100% label agreement), and `nothread` reproduces at −10 kept of
40 with the model-optimistic count more than doubled, so the threading reward
stays. The same run overturned X25's structural finding: capped solves are *not*
still searching for a feasible point — they reach feasibility at iteration 24,
are feasible for 53% of their iterations and hold runs of 60 against an
`acc_iter` of 8. So something other than feasibility stops them,
`ip_acc_compl_tol` is un-retired, and "throughput is a formulation question" is
no longer established.

All the investigation's solver work is now on `implant-hole-walls` and the
experiment worktrees are gone.

## What is open

| | state |
|---|---|
| `ip_acc_compl_tol` 0.01 → 5.0 | **un-retired (X26).** `resume_gpu_arms.sh compl`. `diag_hist` cannot settle which leg gated those feasible runs — `mu` is the barrier parameter and `inf_pr` is scaled — so it needs the arm, not more analysis |
| `angle`, `smooth`, `f32` | **unrun on GPU.** Score on the continuous measures, not kept. `angle` is `PHASE2_CONDITIONING.md`'s own prescribed fix for a defect in the current default, not a tuning knob |
| `ip_acc_iter` 8 → 3 | **un-retired (X27).** X25's "1.5% of wall time" read the trailing feasible run as the total and is wrong by ~30x. Note the mean feasible run is 7.1 against an `acc_iter` of 8 |
| what gates the exit | **the open question.** The feasibility leg qualifies by iteration 77; the median solve exits at 570. Another leg — `ip_acc_tol` = 5.0 or complementarity at 0.01 — is unsatisfied for ~87% of the solve. X10/X11 measured this directly and X25 displaced them; X26 restores them |

```bash
scratch/estimators/resume_gpu_arms.sh compl           # compl5_a
scratch/estimators/resume_gpu_arms.sh interventions   # angle, smooth, f32
uv run --python 3.13 python scratch/estimators/compare_protocol.py \
  scratch/estimators/837772_g40 compl_ctl_a compl5_a angle_a smooth_a f32_a
```

Each refuses to start if any compute app holds the card, or if it reads over
2500 MiB across a 30-second window. `compl_ctl_a` is the control everything is
scored against and is already done — do not re-run it. An arm is ~10 minutes.

## Five traps that cost time

- **Kept cannot retire a formulation change.** It is a binary gate on plans
  clearing by a median +0.021 mm. `PHASE2_CONDITIONING.md` calls it unresolvable
  at this scale, and that licenses "unresolved", not "no effect" — X19 and X25
  both used it for the latter, which is how `smooth` and `angle` were dropped.
- **"Infeasible at the cap" is not "never feasible"** (X26 against X25). The
  measurement was right and the inference from it was wrong, and it propagated
  into the headline of `RECOMMENDATION.md`. Check what a trailing-window
  statistic implies about the whole trajectory before generalising it.
- **`diag_hist`'s `inf_pr`/`inf_du` are IPOPT's *scaled* iteration-log values**,
  and `mu` is the barrier parameter, not complementarity (X25). The acceptable
  test runs on unscaled quantities, so comparing any of the three to `ip_cvtol`
  or `ip_acc_compl_tol` is a category error. `group_min`/`group_nviol`, built
  from `get_current_iterate(scaled=False)`, are the only unscaled measures.
- **Never score a run on this tree against the doc's table** (X23). Same
  configuration, identical problems, median 737 iterations against the table's
  1000 and 11/40 at the cap against 42/80. Use `compl_ctl_a`.
- **Use `repeat_ranks.txt`**, the protocol's own 40 (starts 12 15 22 35 39 49).
  Not the first 40 of `benchmark_ranks.txt` — those start at rank 202 and are a
  different, much worse-ranked set.

Also: sampling `nvidia-smi` a few seconds apart does not clear the 2.5 GiB gate.
A single high reading here was dismissed as an anomaly and was probably another
project's job spinning up between idle moments. Check the compute-app table over
a window; `resume_gpu_arms.sh` does.

## Trees

**One tree.** On 2026-09-23 `implant-hole-walls` was rebased onto
`smooth-model-p1trace` (`0a17bf7`), so the main checkout carries every solver
commit the investigation produced — `bab0970`, B1 (`d790432`), the spin-angle
and smooth-reward options, the settings exposure, the RProp trace, CPU pinning.
`src/`, `tests/`, `pyproject.toml` and `uv.lock` are byte-identical to what
p1angle had, on the same jax/cyipopt/numpy, and the suite passes 906 with 2
skipped. The `-p1angle` and `-smooth` worktrees were removed; `smooth-model` and
`smooth-model-p1trace` remain as refs and are ancestors of HEAD.
`implant-hole-walls-prerebase` is the pre-rebase backup.

The `-motion` and `-rig` worktrees are unrelated to this work and **must not be
removed**: both hold uncommitted changes, `-rig` including three untracked new
source files, and both sit on the pre-`f986af6` `aind_low_point` package layout,
so their work needs porting rather than merging. `rig-panel`'s committed history
is already in `main`; `motion` is 6 commits ahead.

The main checkout's untracked `examples/`, `figures/`, `graphify-out/`,
`launcher-todo.md` and `scripts/render_plan_figure.py` predate this
investigation and are not its work. `main-checkout-wip-2026-09-22.patch` banks a
WIP diff that was discarded once its content was confirmed to be `619cb3f` +
`3b83c55` plus two comment lines.
