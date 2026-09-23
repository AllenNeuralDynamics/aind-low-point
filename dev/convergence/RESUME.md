# Resume here — Phase-2 defaults investigation

Paused 2026-09-22 17:26 with the GPU held by another job. Read this file, then
`RECOMMENDATION.md`. `STATUS.md` is the running log and `EXPERIMENTS.md` the
numbered findings (X1–X25); neither needs reading to restart.

## One paragraph of state

**The investigation has reached its answer: change no default**, for yield and
for throughput alike. The one real improvement found, `bab0970`, is a bug fix
already committed, not a setting. X25 retired the last two live candidates,
`ip_acc_compl_tol` and `ip_acc_iter`, from the existing control's own
diagnostics without needing the card, and found the reason they had looked
promising: a solve spends ~98.5% of its iterations *reaching* a feasible point
and ~1.5% proving that point acceptable, so the whole acceptance-tolerance family
tunes the wrong 1.5%. Throughput is a formulation question, not a settings one.

Nothing is running and all three trees are clean. Picking this up again is
optional rather than owed — see "If you do pick it up" below.

## What is left, and why none of it is load-bearing

| | state |
|---|---|
| four intervention arms (`nothread`, `f32`, `smooth`, `angle`) on GPU | unrun. They measure **kept**, which `PHASE2_CONDITIONING.md` states is not resolvable at n=40 — kept candidates clear by a median +0.021 mm, the order of the sampling gap. Running them buys a cleaner cross-platform record, not a decision |
| the `throughput` batch | **retired.** `resume_gpu_arms.sh throughput` refuses to run; its two candidates are worth at most 1 converted solve in 40 and ~1.5% of wall time |
| the `compl` batch (`compl_5_a`, `compl_ctl_b`, `compl_5_b`) | **retired** by X25 — the hypothesis it was built to test is answered and dead |

### If you do pick it up

```bash
scratch/estimators/resume_gpu_arms.sh interventions   # ~4 arms, GPU, from p1angle
uv run --python 3.13 python scratch/estimators/compare_arms.py \
  scratch/estimators/837772_gi nothread f32 smooth angle
```

It refuses to start if any compute app holds the card, or if it reads over
2500 MiB across a 30-second window. `compl_ctl_a` is the control everything is
scored against and is already done — do not re-run it.

The `/goal` that framed this was cleared by the user partway through, so its four
criteria are a record, not an active gate. Two were verified done (D2/D3/D4 plus
the `inf_du` question; I1 at `abeb9f0`+`81fbefe`, 906 passed / 2 skipped). The
other two wanted the intervention arms above. Neither diagnostic gate tripped: D2
did not find `inf_pr` at fault, D3 found the active Jacobian well conditioned.

## Four traps that cost time

- **Never score a run on this tree against the doc's table** (X23). Same
  configuration, candidate-for-candidate identical problems, median 737
  iterations against the table's 1000 and 11/40 at the cap against 42/80. Use a
  control from the same tree; that is what `compl_ctl_a` is for.
- **Never compare across platforms** (X22). CPU and GPU reach different solutions
  on the same candidate under the same settings, by more than most settings move
  them. The prior art is GPU, so GPU is the reference and the entire CPU arm
  series is indicative only.
- **`diag_hist`'s `inf_pr`/`inf_du` are IPOPT's *scaled* iteration-log values**,
  and `mu` is the barrier parameter, not complementarity (X25). The acceptable
  test runs on unscaled quantities, so comparing any of the three to `ip_cvtol`
  or `ip_acc_compl_tol` is a category error — it produced a confident wrong
  answer here before the callback source was checked. `group_min`/`group_nviol`,
  built from `get_current_iterate(scaled=False)`, are the only unscaled measures
  recorded.
- **Use `repeat_ranks.txt`**, the protocol's own 40 (starts 12 15 22 35 39 49).
  Not the first 40 of `benchmark_ranks.txt` — those start at rank 202 and are a
  different, much worse-ranked set.

Also: sampling `nvidia-smi` a few seconds apart does not clear the 2.5 GiB gate.
A single high reading here was dismissed as an anomaly and was probably another
project's job spinning up between idle moments. Check the compute-app table over
a window; `resume_gpu_arms.sh` now does.

## Trees

| path | branch | note |
|---|---|---|
| `aind-low-point` | `implant-hole-walls` @ `fcf27f7` | main checkout; clean |
| `aind-low-point-p1angle` | `smooth-model-p1trace` @ `6380f2d` | **the tree all arms ran from**; clean |
| `aind-low-point-smooth` | `smooth-model` @ `3b83c55` | clean |

**Still run arms from p1angle**, which is where every result in `EXPERIMENTS.md`
came from. The main checkout has none of the investigation's solver commits.

The main checkout had carried an uncommitted diff to `phase2.py`, `settings.py`
and `test_pipeline_settings.py`, which was `619cb3f` + `3b83c55` — verified, the
only content not in those two commits was two comment lines. It was discarded
after banking a copy at `main-checkout-wip-2026-09-22.patch`, because leaving it
armed the two-tree confound that invalidated the first arm series (X15): running
from the main checkout gave the settings knobs and CPU pinning *without*
`bab0970`.

Its untracked `examples/`, `figures/`, `graphify-out/`, `launcher-todo.md` and
`scripts/render_plan_figure.py` predate this investigation and are not its work.

`d790432` (B1, the constraint-vector group split) is still unmerged to
`implant-hole-walls`.
