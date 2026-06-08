# MRV pool run — tuned configurations

`scripts/mrv_pool_run.py` runs the full MRV candidate pool (≈19k configs: 3 arcs,
≤4 probes/arc) through **restore → reduced → full** optimization and a
per-candidate FCL gate, saving one record per candidate for the downstream
cull → diversity-select → trust-constr (Phase 2) → FCL (Phase 3) handoff.

The optimizer is **tuned** (each lever measured on the 545-config calibration set;
see `dev/` memory `well_sdf_thin_skin_thickening`, `adam_moment_restart_schedule`,
`coarse_fine_surf_tuning`):

| lever | tuned value | why |
|---|---|---|
| **well SDF** | `thick` | the well asset is a thin *surface* shell → α-wrap SDF is a ~0.8 mm skin (min −0.63 mm) that under-reports body penetration. `make_thick_well_sdf` solidifies the conical-annulus body. FCL still uses the **true thin mesh** (honest gate). |
| **minimizer** | `rprop` (iRprop−) | ADAM's 2nd moment `v` inflates from early collision-gradient spikes and stalls long continuous runs (measured). RProp is sign-based → immune, and the right tool for this deterministic ill-conditioned problem. |
| **surf schedule** | coarse→fine, both stages | surf points = the DRAM-bound gather count. Running the bulk at coarse surf then a short fine finish is a *homotopy* — coarse smooths collision walls → RProp reaches more basins → fine finish + FCL validate. Win-win: faster **and** more feasibles. The fine finish on **both** stages (reduced too) is load-bearing. |

## The two presets

Both run the same pipeline; they trade wall-time vs feasible yield. Numbers are
on the 545 calibration set (`feasible / known-good-winners`; all-fine RProp
baseline = 91/17).

### THROUGHPUT (default) — least wall time

`coarse_N=1000`, 50 fine steps finishing each stage → **~2.16× faster, 545: 105/20.**
Beats all-fine on *everything* at <½ the surf wall-time.

```bash
JAX_PLATFORMS=cuda uv run --python 3.13 -m scripts.mrv_pool_run
# (defaults: MINIMIZER=rprop WELL=thick COARSE_N=1000 REDUCED_FINE=50 FULL_FINE=50)
```

### YIELD — most feasibles

`coarse_N=3000`, 100 fine steps finishing each stage → **~1.31× faster, 545: 123/21.**
Maximizes the feasible handoff set.

```bash
JAX_PLATFORMS=cuda COARSE_N=3000 REDUCED_FINE=100 FULL_FINE=100 \
  uv run --python 3.13 -m scripts.mrv_pool_run
```

### BASELINE — reproduce the old 165-feasible run

```bash
JAX_PLATFORMS=cuda MINIMIZER=adam_const WELL=thin COARSE_N=5000 \
  REDUCED_FINE=0 FULL_FINE=0 uv run --python 3.13 -m scripts.mrv_pool_run
```

## Knobs

| env | default | meaning |
|---|---|---|
| `MINIMIZER` | `rprop` | `rprop` \| `moment_restart` (ADAM, equivalent) \| `adam_const` (old) |
| `WELL` | `thick` | `thick` \| `thin` (soft side only; FCL always true mesh) |
| `COARSE_N` | `1000` | coarse-pass surf count; `5000` ⇒ single-fidelity (no coarse) |
| `REDUCED_FINE` | `50` | fine (@5000) steps ending the **reduced** stage; rest @`COARSE_N` |
| `FULL_FINE` | `50` | fine (@5000) steps ending the **full** stage; rest @`COARSE_N` |
| `STAGE1`/`STAGE2` | `500` | total steps in the reduced / full stage |
| `COARSE_N=5000` or `RED/FULL_FINE=STAGE` | — | degenerate cases collapse to all-fine |
| `OUT` | `scratch/mrv_pool_results.pkl` | output; **resumable** — re-running skips already-saved n_arcs groups |
| `SEED_CACHE` | `scratch/mrv_seeds.pkl` | enumerate+seed is cached (~14 min); reused on restart |
| `FCL_TOPK` | `300` | FCL is run on the top-K by soft clearance per n_arcs group |
| `LIMIT` | `0` | cap candidates (smoke testing; disables seed cache + resume) |

## Output

`OUT` is a pickle `{records: [...], minimizer, well, coarse_n, reduced_fine,
full_fine, stage1, stage2, n_spins, ...}`. Each record:

- `probe_to_hole`, `partition`, `min_ml_gap` — the discrete decision + seed margin
- `x` — full-stage final pose; `x_reduced` — reduced-stage checkpoint pose
- `objective` — full Phase-1 objective (coverage on)
- `min_clear` — full@end soft dual-rep clearance (**the cull metric**, thick well)
- `min_clear_reduced` — reduced@end soft clearance (checkpoint)
- `fcl` — true-mesh FCL min slack for the top-`FCL_TOPK` by clearance (else `nan`);
  `>= -1e-4` ⇒ FCL-feasible. `-1.0` is a boolean "≥1 pair collides" sentinel.

**Next step from here:** cull on `min_clear` (≈ −0.35 mm gate, see
`early_cull_trajectory_calibration`), diversity-select the survivors, run Phase-2
trust-constr + Phase-3 FCL on them.

## Notes / gotchas

- **Resumable + incremental save**: results are written after each n_arcs group;
  a crash keeps completed groups, and re-running skips them. Largest group runs
  first (cleanest GPU for the heaviest spin-restore).
- **Compile**: RProp + two fidelities compiles several kernels up front
  (~3–4 min); amortized over the ~300 chunks of the big 3-arc group.
- **VRAM**: ~9.5 MB/candidate marginal + ~2.2 GB baseline; `CHUNK=64` peaks
  ~2.8 GB. The thick-well + coarse SDFs are shared (broadcast), not per-cand.
- Don't chain a background waiter with `until ! pgrep -f "[m]rv_pool_run"` whose
  own argv contains the pattern — it matches itself and loops forever. Launch
  directly or match a unique token.
