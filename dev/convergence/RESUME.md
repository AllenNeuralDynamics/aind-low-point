# Resume here — Phase-2 defaults investigation

Paused 2026-09-22 16:44 with the GPU handed back to another job. Read this file,
then `RECOMMENDATION.md`. `STATUS.md` is the running log and `EXPERIMENTS.md` the
numbered findings (X1–X23); neither needs reading to restart.

## One paragraph of state

The recommendation is written and is **change no default** — the one real
improvement found, `bab0970`, is a bug fix already committed, not a setting. Two
of the four goal criteria are verified done. The other two need the card: seven
of eight GPU arms are unrun, and the one that completed (`compl_ctl_a`) is the
control everything else is scored against.

## Restart command

```bash
scratch/estimators/resume_gpu_arms.sh both     # or: compl | interventions
```

It refuses to start if any compute app holds the card, or if it reads over
2500 MiB across a 30-second window. Roughly an hour for all seven arms.
`compl_ctl_a` is deliberately not in the list — it is done.

Then read them against the control:

```bash
uv run --python 3.13 python scratch/estimators/compare_arms.py \
  scratch/estimators/837772_g40 compl_ctl_a compl_5_a compl_ctl_b compl_5_b
uv run --python 3.13 python scratch/estimators/compare_arms.py \
  scratch/estimators/837772_gi nothread f32 smooth angle
```

## Criteria

| | state |
|---|---|
| 1 — six arms read, signed effect against base | **open**, needs the card |
| 2 — D2/D3/D4 + the `inf_du` question | **done**; `inf_du` answered "No, not as a cross-arm ranking metric" |
| 3 — recommendation validated by running the 40×2 protocol against a same-tree, same-platform control | **open**, needs the card |
| 4 — I1 built, parity bitwise, checks green | **done**; `abeb9f0`+`81fbefe`, 906 passed / 2 skipped |

Neither gate tripped: D2 did not find `inf_pr` at fault, D3 found the active
Jacobian well conditioned. The stopping rule has not fired only because the
pending arms are unrun — nothing so far beats `base` on kept yield except noise,
and the one effect clearing the noise floor, `nothread`'s −6, is a worsening.

## Criterion 3, as amended 2026-09-22

Criterion 3 originally required validation *on* the 40×2 protocol from
`dev/PHASE2_CONDITIONING.md`, which read as scoring against that document's
recorded table. X23 showed the table no longer scores this tree — same
configuration, candidate-for-candidate identical problems, median 737 iterations
against 1000 and 11/40 at the cap against 42/80.

It now requires **running** the protocol against a control solved on the same
tree and the same platform. The protocol was always the right instrument; only
its recorded numbers stopped being the reference. `compl_ctl_a` is that control,
which is why the batch was built around one. `PLAYBOOK.md` carries the full
wording, including the platform exclusion from X22.

Nothing else about the criterion changed: the capped-10 is still excluded as a
hand-picked tail, and `f32`'s 9/10 still does not on its own justify flipping
`RUTTER_GRID_DTYPE`.

## Three traps that cost time today

- **Never score a run on this tree against the doc's table** (X23). Use a
  same-tree control.
- **Never compare across platforms** (X22). CPU and GPU reach different solutions
  on the same candidate with the same settings, by more than most settings move
  them. The prior art is GPU, so the GPU is the reference and the entire CPU arm
  series is indicative only.
- **Use `repeat_ranks.txt`**, the protocol's own 40 (starts 12 15 22 35 39 49).
  Not the first 40 of `benchmark_ranks.txt` — those start at rank 202 and are a
  different, much worse-ranked set.

Also: sampling `nvidia-smi` a few seconds apart does not clear the 2.5 GiB gate.
A single high reading here was dismissed as an anomaly and was probably another
project's job spinning up between idle moments. Check the compute-app table over
a window; `resume_gpu_arms.sh` now does.

## Trees and uncommitted work

| path | branch | note |
|---|---|---|
| `aind-low-point` | `implant-hole-walls` @ `b3b6bbe` | main checkout |
| `aind-low-point-p1angle` | `smooth-model-p1trace` @ `6380f2d` | **the tree all arms ran from**; clean |
| `aind-low-point-smooth` | `smooth-model` @ `3b83c55` | |

The main checkout is clean. It had carried an uncommitted diff to `phase2.py`,
`settings.py` and `test_pipeline_settings.py`, which was `619cb3f` + `3b83c55` —
verified, the only content not in those two commits was two comment lines. It was
discarded on 2026-09-22 after banking a copy at
`main-checkout-wip-2026-09-22.patch`, because leaving it armed the two-tree
confound that invalidated the first arm series (X15): running from the main
checkout gave the settings knobs and CPU pinning *without* `bab0970`.

**Still run arms from p1angle**, which is where every result in `EXPERIMENTS.md`
came from. The main checkout at `b3b6bbe` has none of the investigation's commits.

`d790432` (B1, the constraint-vector group split) is still unmerged to
`implant-hole-walls`.
