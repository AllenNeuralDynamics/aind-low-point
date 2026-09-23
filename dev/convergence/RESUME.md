# Resume here — solver convergence investigation

Updated 2026-09-23 10:00. Read this file, then `RECOMMENDATION.md`. `STATUS.md`
is the running log and `EXPERIMENTS.md` the numbered findings (X1–X36).
`PLAYBOOK.md` opens with the goal this is all scored against.

## One paragraph of state

**Phase 2 is close to settled; Phase 1 is where the work is now.** The goal has
three legs — find a feasible point if one exists, maximise the objective, and
stop once the answer is found — and the third is **closed with no headroom**:
`waste` is 0% on every arm and the objective is still descending at exit (X31),
so every stopping-rule knob costs objective by construction. That retired X29's
headline, which had scored `ip_acc_compl_tol` on counts alone. On the first two
legs `spin_param=angle` is the clear win (X32, X34) and `smooth_thread_reward`
is a clear loss (X33). Phase 1 got the same treatment and came back different:
both interventions are null there, and the real finding is that flips *rise*
while the RProp step collapses (X36) — so Phase 1, unlike Phase 2, does have
iterations to reclaim. A1 is now running to find which objective term owns those
flips.

All the investigation's solver work is on `implant-hole-walls` and the
experiment worktrees are gone.

## What is settled

| | verdict |
|---|---|
| goal 3, wasted iterations | **closed (X31).** `waste` 0% everywhere. No stopping-rule knob is a win; each one trades objective |
| `spin_param=angle`, Phase 2 | **adopt — awaiting the user's approval to flip a production default.** feas 33/40 vs 31, `reach` 14 vs 28, caps 1 vs 11, 100% feasible at exit, 25s vs 43s, and bitwise reproducible against itself (X32, X34) |
| `smooth_thread_reward`, Phase 2 | **do not enable (X33).** Regresses six measures including the thread group its own reward targets, and contributes 0 unique candidates to the satisfiable union |
| `ip_acc_compl_tol` 0.01 → 5.0 | **a trade, and the user's call (X31).** Buys feasibility with +0.92 objective, worse on 27/31. Its premise — wasted iterations — is false |
| `ip_cvtol`'s acceptable leg | **closed (X30).** Mixed-unit constraint vector; 1e-2 admits 0.01 mm of overlap that FCL then calls a collision. Do not loosen |
| exact Hessian | **closed (X30).** Rescued 8/11 against limited-memory's 9, drove a feasible control into collision, 2–5x slower, and its constraint-HVP kernel crashes XLA autotuning on the card |
| `f32` | **closed.** bfloat16 storage with float32 blending is what `bab0970` gave and beats float32 storage, which also doubles grid memory per worker |
| Phase 1 `chunk` | **closed (X35).** 256 needs 6.59 GiB for the RProp batch alone and does not fit the arena on a 10 GiB card. Not tracing, not dtype, not restore overlap. 128 works |

## What is open

| | state |
|---|---|
| **A1 — which term owns Phase 1's flips** | **running.** Seven arms, one weight zeroed each, prediction pre-registered in `PLAYBOOK.md`: the clearance margin reward, which Phase 2 already runs at zero for this exact mechanism and Phase 1 hard-codes on |
| `spin_param` per-phase | **a design question for the user.** It sits on the shared `PipelineSettings`; Phase 2 gains a lot and Phase 1 pays a little (X32, X36) |
| F4 / F5 — the kink fixes | **designed, not implemented.** Both change an objective term, so both need the user first |
| B2 — Phase 1's gradient ignores `grid_dtype` | **verified defect, unfixed.** `build_cw_fns` hard-codes `jnp.bfloat16` |
| `angle`+`compl5` combination | **unrun.** The two best arms, with disjoint unique candidates |
| `compl5_b` | **unrun.** Reproducibility of the complementarity setting |

```bash
# Phase 1 term ablation (A1)
scratch/estimators/run_p1_ablate_gpu.sh 837772-config scratch/estimators/p1abl 512 500 \
  base mclear mthread thread clear brain unit
uv run --python 3.13 python scratch/estimators/compare_p1_arms.py \
  scratch/estimators/p1abl base mclear mthread thread clear brain unit

# Phase 2 arms, scored against the control that is already done
uv run --python 3.13 python scratch/estimators/compare_protocol.py \
  scratch/estimators/837772_g40 compl_ctl_a compl5_a angle_a smooth_a
```

`compl_ctl_a` is the control everything is scored against and is already done —
do not re-run it. A Phase-2 arm is about ten minutes, a Phase-1 arm about five.

## The satisfiable union

All 40 protocol candidates are **proven satisfiable** — each is solved, and
FCL-validated, by at least one arm. That is a ground truth the control cannot
give. Recall against it: `compl5` 37/40, `angle` 33/40, control 31/40, `smooth`
27/40, `nothread` 21/40. Only 14/40 are solved by every arm; `compl5` and
`angle` each uniquely solve one, and `smooth` uniquely solves none.

## Traps that cost time here

- **Score on the objective, not on counts.** X29 read "+6 kept, 7x faster" as a
  win without measuring solution quality; X31 showed it is a trade. `kept` is a
  binary gate on plans clearing by a median +0.021 mm, so it licenses
  "unresolved", never "no effect" — which is how `smooth` and `angle` were both
  dropped once already.
- **A trailing-window statistic is not the trajectory** (X26 against X25). A
  solve infeasible at the cap was typically feasible for half its iterations.
- **`diag_hist`'s `inf_pr`/`inf_du` are IPOPT's *scaled* iteration-log values**
  and `mu` is the barrier parameter, not complementarity. Only
  `group_min`/`group_nviol` are unscaled, so only they may be compared against
  `ip_cvtol` or `ip_acc_compl_tol` (`ip_acc_tol` is the scaled exception).
- **`obj_value` in the IPOPT callback is the restoration subproblem's objective**
  wherever `alg_mod` is 1. Mask it, or an arm's objective is a mix of two
  functions. `solver_status == 2` likewise describes the restoration
  subproblem, not the returned iterate.
- **0/40 bitwise against a *different* configuration is expected**, not a
  determinism failure — different coordinates give different roundoff. Test an
  arm against a second run of itself. That category error buried `angle` once.
- **Never score a run on this tree against the doc's table** (X23). Use
  `compl_ctl_a`.
- **Check whether a setting was *chosen* before calling it untested** (X30). The
  pre-investigation solver bake-off is in the project's memory, not `STATUS.md`;
  its probe is `scratch/ipopt_exact_vs_lm.out`.
- **Use `repeat_ranks.txt`**, the protocol's own 40. Not the first 40 of
  `benchmark_ranks.txt`, which start at rank 202 and are much worse-ranked.
- **Phase 1's seed pool spawns workers that re-execute `__main__` by path.** A
  driver importing the objective package at module scope makes every worker open
  its own CUDA context against the arena the parent already holds, and the
  failure surfaces as a `BrokenPipeError` from the pool rather than as an OOM.
  Seed serially, or keep the driver's top-level imports free of `aind_rutter`.
- Sampling `nvidia-smi` a few seconds apart does not clear the 2.5 GiB gate;
  check the compute-app table over a window, as the runners do.

## Trees

**One tree.** On 2026-09-23 `implant-hole-walls` was rebased onto
`smooth-model-p1trace` (`0a17bf7`), so the main checkout carries every solver
commit the investigation produced. The suite passes 906 with 2 skipped.
`smooth-model` and `smooth-model-p1trace` remain as refs and are ancestors of
HEAD; `implant-hole-walls-prerebase` is the pre-rebase backup.

The `-motion` and `-rig` worktrees are unrelated to this work and **must not be
removed**: both hold uncommitted changes, `-rig` including three untracked new
source files, and both sit on the pre-`f986af6` `aind_low_point` package layout,
so their work needs porting rather than merging.

The main checkout's untracked `examples/`, `figures/`, `graphify-out/`,
`launcher-todo.md` and `scripts/render_plan_figure.py` predate this
investigation and are not its work.
