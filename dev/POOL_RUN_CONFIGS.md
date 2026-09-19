# MRV pool run — tuned configurations

`rutter-phase1` (`src/aind_rutter/optimization/pipeline/phase1.py`) runs the
full MRV candidate pool (about 19k configs: 3 arcs,
≤4 probes/arc) through **restore → reduced → full** optimization, saving one
record per candidate for the downstream Phase-2 IPOPT/trust-constr polish,
FCL/threading gate, and MMR handoff ranking. FCL runs once, in Phase 2.

The optimizer is **tuned** (each lever measured on the 545-config calibration set;
see `dev/` memory `well_sdf_thin_skin_thickening`, `adam_moment_restart_schedule`,
`coarse_fine_surf_tuning`):

| lever | tuned value | why |
|---|---|---|
| **well SDF** | `thick` | the well asset is a thin *surface* shell → α-wrap SDF is a ~0.8 mm skin (min −0.63 mm) that under-reports body penetration. `make_thick_well_sdf` solidifies the conical-annulus body. FCL still uses the **true thin mesh** (honest gate). |
| **minimizer** | iRprop− | ADAM's 2nd moment `v` inflates from early collision-gradient spikes and stalls long continuous runs (measured). RProp is sign-based → immune, and the right tool for this deterministic ill-conditioned problem. It is now the only minimizer; the two ADAM variants it replaced are recorded below. |
| **surf schedule** | coarse→fine, both stages | surf points = the DRAM-bound gather count. Running the bulk at coarse surf then a short fine finish is a *homotopy* — coarse smooths collision walls → RProp reaches more basins → fine finish + FCL validate. Win-win: faster **and** more feasibles. The fine finish on **both** stages (reduced too) is load-bearing. |

## The two presets

Both run the same pipeline; they trade wall-time vs feasible yield. Numbers are
on the 545 calibration set (`feasible / known-good-winners`; all-fine RProp
baseline = 91/17).

### THROUGHPUT (default) — least wall time

`coarse_N=1000`, 50 fine steps finishing each stage → **~2.16× faster, 545: 105/20.**
Beats all-fine on *everything* at <½ the surf wall-time.

```bash
JAX_PLATFORMS=cuda uv run --python 3.13 rutter-phase1
# (defaults: RUTTER_WELL=thick RUTTER_COARSE_N=1000 RUTTER_REDUCED_FINE=50 RUTTER_FULL_FINE=50)
```

### YIELD — most feasibles

`coarse_N=3000`, 100 fine steps finishing each stage → **~1.31× faster, 545: 123/21.**
Maximizes the feasible handoff set.

```bash
JAX_PLATFORMS=cuda RUTTER_COARSE_N=3000 RUTTER_REDUCED_FINE=100 RUTTER_FULL_FINE=100 \
  uv run --python 3.13 rutter-phase1
```

### The two ADAM variants, and why they are gone

Phase 1 once took `MINIMIZER=rprop|moment_restart|adam_const`. Both ADAM
branches were deleted once the comparison settled; neither is reachable now.

| variant | result on the 545 calibration set |
|---|---|
| `adam_const` | flat learning rate. The 2nd moment `v` accumulates from early collision-gradient spikes and the effective step decays before the basin floor, so long continuous runs stall. This produced the old 165-feasible pool, with `RUTTER_WELL=thin RUTTER_COARSE_N=5000 RUTTER_REDUCED_FINE=0 RUTTER_FULL_FINE=0`. |
| `moment_restart` | resets `m, v` every 50 steps, which restores a full `lr·sign(g)` step and roughly doubles `adam_const`'s feasible count — landing level with RProp. Equivalent, so never worth selecting. |

RProp reaches the same place as `moment_restart` without the moment state, and
both ADAM branches only had to keep compiling.

## Knobs

| env | default | meaning |
|---|---|---|
| `WELL` | `thick` | `thick` \| `thin` (soft side only; FCL always true mesh) |
| `COARSE_N` | `1000` | coarse-pass surf count; `5000` ⇒ single-fidelity (no coarse) |
| `REDUCED_FINE` | `50` | fine (@5000) steps ending the **reduced** stage; rest @`COARSE_N` |
| `FULL_FINE` | `50` | fine (@5000) steps ending the **full** stage; rest @`COARSE_N` |
| `STAGE1`/`STAGE2` | `500` | total steps in the reduced / full stage |
| `RUTTER_COARSE_N=5000`, or the fine counts equal to the stage counts | — | degenerate cases collapse to all-fine |
| `OUT` | `scratch/mrv_pool_results.json.gz` | output; **resumable** — re-running skips already-saved n_arcs groups |
| `SEED_CACHE` | `scratch/mrv_seeds_<config-stem>.json.gz` | enumerate+seed is cached (~14 min); subject-specific and reused on restart |
| `LIMIT` | `0` | cap candidates (smoke testing; disables seed cache + resume) |

## Output

`OUT` is gzipped JSON, written through `pipeline/payloads.py`:
`{records: [...], minimizer, well, coarse_n, reduced_fine, full_fine, stage1,
stage2, n_spins, ...}`. Each record:

- `probe_to_hole`, `partition`, `min_ml_gap` — the discrete decision + seed margin
- `x` — full-stage final pose; `x_reduced` — reduced-stage checkpoint pose
- `objective` — full Phase-1 objective (coverage on)
- `min_clear` — full@end soft dual-rep clearance (**the cull metric**, thick well)
- `min_clear_reduced` — reduced@end soft clearance (checkpoint)

Pools written before Phase 1 stopped running its own FCL check also carry `fcl`;
nothing reads it.

**Next step from here:** run `rutter-phase2`, which ranks by `SELECT_BY`
(`min_clear` by default), polishes the top `TOPK` with IPOPT by default
(`RUTTER_SOLVER=trust-constr` remains available), applies the final FCL/threading gate,
and MMR-ranks the feasible handoff set.

## Notes / gotchas

- **Resumable + incremental save**: results are written after each n_arcs group;
  a crash keeps completed groups, and re-running skips them. Largest group runs
  first (cleanest GPU for the heaviest spin-restore).
- **Compile**: RProp + two fidelities compiles several kernels up front
  (~3–4 min); amortized over the ~300 chunks of the big 3-arc group.
- **VRAM**: ~9.5 MB/candidate marginal + ~2.2 GB baseline; `RUTTER_CHUNK=64` peaks
  ~2.8 GB. The thick-well + coarse SDFs are shared (broadcast), not per-cand.
- Don't chain a background waiter with `until ! pgrep -f "[a]lp-phase1"` whose
  own argv contains the pattern — it matches itself and loops forever. Launch
  directly or match a unique token.
