# Probe → Hole Assignment as MINLP: Brief for Help

**Audience:** AI agent or engineer joining mid-stream to help fix the optimizer's discrete-search layer. Read top-to-bottom; everything you need to act is here.

**Date:** 2026-05-18 (updated; original 2026-05-17)
**Branch:** `finish-refactor` (HEAD ~e0be750 + uncommitted work). Not yet merged to main.

> **CONTINUATION NOTE (added 2026-05-18, before context compaction).** The
> sections **"Update 2026-05-18: Visibility Atlas + Arc-First Search"** and
> **"Immediate Next Implementation: Local-Arc-Config Rewrite"** at the bottom
> of this document are the canonical pickup point for the next agent. Read
> those before any of the earlier sections (which capture history).
>
> **UPDATE (after compaction, same day).** Local-arc-configs diagnostic
> implemented as `scripts/diagnose_local_arc_configs.py`. Manual now
> achievable per-arc at **`--ap-tol 5 --n-top 128`** (n_spin=72). The exact
> manual MLs (VM=-8, CLA=-24) are NOT in the atlas — closest atlas pairs
> are (VM=-10.2, CLA=-26.6) with |diff|=16.4 just over the 16° threshold.
> Stage 2/3 polish has to move ML by ~2° to reach manual; well within
> SLSQP basin. **Next step:** rank top-K from arc-first pool + run Stage 2
> on top candidates to confirm they polish to manual-quality. **Avoid full
> global-assembly enumeration** — earlier attempts OOM'd at the unbounded
> cartesian product of `(partition, AP triple, local configs)`. Use
> bounded sampling or direct top-K ranking via the cell metric.

---

## TL;DR

The probe-placement optimizer's discrete layer (which probe lands in which implant hole) is **search-bound, not polish-bound**. The known-good manual plan for `836656/T12` is at **rank 49028 / 792000 feasible by raw LSAP cost** — far past anything Murty's top-K or diversified Murty can enumerate. Reweighting `CostWeights` doesn't fix it: per-cell costs simply don't encode the joint information that makes the manual config good.

The problem is naturally MINLP: 7 discrete probe→hole assignments × continuous arc/ml/spin/offset/depth per probe. Current decomposition (LSAP for discrete → SLSQP for continuous) leaks because LSAP can't see joint structure (true pairwise interference, arc-feasibility, spatial spread).

**Proposed fix:** for K ≤ 8 probes, **brute-force enumerate feasible probe→hole assignments and score with a true joint pre-cost** (exact FCL BVH distances at best-fit poses, precomputed in a 4-tuple cache). Then pass top-K to the existing Stage 2/3 pipeline unchanged.

---

## What This Project Does

`aind-low-point` ("Pinpoint in Python") plans neuropixel probe placement for in-vivo recordings at AIND. A user picks targets in MRI/CCF space; the system finds a set of (arc AP, probe ml, probe spin, offset_R, offset_A, depth) configurations that:

- Thread each probe through its assigned implant hole (no contact with hole sections)
- Keep probe bodies / headstages from colliding with each other and the implant
- Satisfy rig kinematic constraints (≥ 16° AP separation between distinct arcs, ≤ 3-4 arcs)
- Maximize total recording coverage of targets

There are two frontends sharing a runtime: K3D + ipywidgets (Jupyter) and Trame + PyVista (web app). The optimizer is independent of either.

**Invariants** (don't violate these):
- Internal canonical space is **LPS millimeters**. RAS only at user-facing boundaries.
- **Python 3.13 only** (`python-fcl` no 3.14 wheels). All commands: `uv run --python 3.13 ...`.
- **Pydantic models are source of truth.** When tests disagree with `config.py`, fix the tests.
- Branch `finish-refactor`; not merged to main.

**Quick orientation reading** (in repo, in this order):
1. `CLAUDE.md` — project root file layout + invariants
2. `dev/CORE_CONCEPTS.md` — conceptual tour
3. `dev/optimizer_plan.md` — active optimizer design doc

---

## Current Architecture: 3-Stage Optimizer

Entry point: `scripts/run_optimizer.py`, function `optimize_joint` in `src/aind_low_point/optimization/joint_rerank.py:1527`.

### Stage 0: Pre-compute pose features
- Per (probe, hole): evaluate a 7-pose bank (slot-aligned, target-aligned, halfway, ±AP/ML wobbles)
- Aggregate: `min_max_g`, `min_violation_sq`, `max_coverage`
- Used by LSAP cost matrix and reranker scoring
- Cost: ~1-2 s for our problem size

### Stage 1: LSAP probe → hole assignment (`hole_assignment.py`)

Build per-cell cost matrix:
```
cost[probe_i, hole_j] = α·angle + β·max_g + γ·interference + η·violation - δ·coverage
                      = 0·angle + 0.3·max_g + 0.5·interf + 2.0·viol - 5.0·cov
```

Then Murty's top-k enumeration of (probe → hole) matchings. Default `k_holes_pool = 50`.

The Murty implementation lives in `solve_top_k_assignments` (`hole_assignment.py:617`). Standard partition-and-LSAP-subproblems. Recently added `min_hamming_distance` for diversified enumeration (see Findings below).

### Stage 2: Joint reranker reduced SLSQP (`joint_rerank.py`)

For each (H assignment, A arc partition) pair (k_arcs_pool=20 partitions per H ⇒ 1000 candidates):
- Build statics (per-probe target, pivot, shank tips, hole sections)
- Run **reduced SLSQP** (vars: `[arc_aps, (ml, spin) per probe]` — 17 vars for 7 probes, 3 arcs)
- Soft penalties: threading, AP-sep, intra-arc ML-sep, soft bounds, **pairwise clearance**
- Three warm starts per candidate (random, identity, prior-best)
- Each warm start prefixed with **spin restoration** (8×8 FCL sweep, now JAX vmap — see below)

Returns top-15 lex-ranked candidates → Stage 3.

### Stage 3: Full inner solve (`optimize.py`)

For each of 15 survivors:
- Full-x SLSQP polish: 5 continuous vars per probe + arc APs = 38 vars for 7 probes, 3 arcs
- Hard ineq constraints (not soft): threading, clearance, arc-AP-sep, intra-arc ML-sep
- Two-stage: feasibility (minimise violation²) → coverage (minimise -coverage, ineqs held)
- Runs in 15 spawn-mode ProcessPool workers in parallel (fork breaks JAX-CUDA contexts; spawn workers force `JAX_PLATFORMS=cpu`)

---

## Recent JAX Ports (Background — already done)

These are all complete and validated:

### `src/aind_low_point/optimization/sdf_jax.py` — pose math + SDF lookup
- `arc_angles_to_rotation` — JAX of `aind_mri_utils.arc_angles.arc_angles_to_affine` with `invert_AP=True, invert_rotation=True`
- `pose_from_optimizer_vars` — JAX companion
- `trilinear_sdf` — voxel SDF interpolation
- `pairwise_signed_clearance` — symmetric SDF clearance with `jax.grad`-able output

### `src/aind_low_point/optimization/sdf.py` — voxel SDF generation
- `build_probe_sdf(mesh, spacing_mm=0.2, pad_mm=2.0)` using `libigl` FAST_WINDING_NUMBER
- Caches to `~/.cache/aind_low_point/sdfs/` keyed on mesh hash

### `src/aind_low_point/optimization/joint_rerank_jax.py` — Stage 2 JAX backend
- `make_jax_reduced_objective(statics, n_arcs, weights)` returns scipy-compatible `(fun, jac)`
- **Module-level JIT cache** keyed on (n_probes, n_arcs, shape padding, has_sdf, sdf grid shapes, weights). Across all 1000 Stage 2 candidates in a run, **1 compile + 2999 cache hits**.
- Padded to MAX_SHANKS_PAD=4, MAX_SECTIONS_PAD=8.
- Per-probe SDF data stays as tuples-of-arrays (not stacked) because per-kind grid shapes differ.

### `src/aind_low_point/optimization/stage3_jax.py` — Stage 3 constraint Jacobians
- Threading, AP-sep, intra-arc-ML-sep with analytic `jax.jacrev` Jacobians
- Wired into `_slsqp_polish_constrained` when SDF backend is active
- (Coverage objective itself still NumPy MC — could be ported)

### `src/aind_low_point/optimization/spin_restore_jax.py` — JAX spin sweep
- Replaces FCL 8×8 brute spin sweep with vmapped JAX SDF over coarse 4×4 + fine 4×4 grid
- Same module-level JIT cache pattern; 1 compile per process
- Wall: **24.16s → 10.21s** on the 15-candidate diagnostic

### Stage 3 ProcessPool: spawn, not fork
- Stage 2 puts JAX-CUDA into the parent process; fork() inherits a broken CUDA context → OOM cascade
- Workers do `JAX_PLATFORMS=cpu` in `_inner_solve_worker_init` to avoid GPU contention
- 15 workers × ~5s JAX compile each = ~75 s overhead, but the parallelism wins anyway

---

## What's Failing

The optimizer **cannot find the manual plan's configuration** in any feasible time budget on `836656/T12`.

### Manual plan (`examples/836656-config-T12.plan.yml`)
```
Manual probe → (arc, AP, ml, spin) :
  MD   arc=a  AP=+13.0  ml=-12.00  spin=-34.0   hole=3
  BLA  arc=b  AP=-10.0  ml=+27.00  spin=+0.0    hole=4
  PL   arc=b  AP=-10.0  ml=-30.00  spin=+131.0  hole=1
  VM   arc=c  AP=-43.0  ml= -8.00  spin=-180.0  hole=7
  RSP  arc=c  AP=-43.0  ml= +8.00  spin=+4.0    hole=5
  CA1  arc=b  AP=-10.0  ml= +3.50  spin=+87.0   hole=10
  CLA  arc=c  AP=-43.0  ml=-24.00  spin=+171.0  hole=12

Result: feasible at strict 0 mm clearance, coverage=17.91.
```

### Optimizer result, strict 0 mm allowance (`examples/836656-config-T12_opt_alternatives/`)
```
0 / 15 feasible. Best plan: max_viol=0.31, coverage=5.20.
Arc APs: a=-3°, b=+13°, c=-19°  (completely different basin)
Probe→hole: only PL matches manual.
```

### Optimizer result, 2.1 mm allowance
```
5 / 15 feasible. Best: coverage=17.09, max_viol=0.   
(Closer to manual quality, but still doesn't recover the manual H assignment.)
```

### When run with `--seed-plan` (seed-polish path)
```
Polished output equals the manual: coverage 17.91, max_viol 0, feasible.
```

So Stage 3 polish *can* converge to the manual config when handed it. The gap is purely **Stage 1 + Stage 2 search**.

---

## Diagnostic Findings

Tooling: `scripts/diagnose_lsap_rank.py` — pipe manual probe→hole mapping on stdin:
```bash
printf 'MD 3\nBLA 4\nPL 1\nVM 7\nRSP 5\nCA1 10\nCLA 12\n' | \
  uv run --python 3.13 python -m scripts.diagnose_lsap_rank \
    examples/836656-config-T12.yml /tmp/836656-holes.yml \
    --manual-plan examples/836656-config-T12.plan.yml --k 50
```

### LSAP cost breakdown — manual vs LSAP top-1

```
LSAP-best (rank 1) at default weights:
  MD→8, BLA→12, PL→1, VM→3, RSP→5, CA1→13, CLA→11  cost=-86.713

Manual:
  MD→3, BLA→4, PL→1, VM→7, RSP→5, CA1→10, CLA→12  cost=-86.523

Gap = 0.189   (manual is 0.2% worse on per-cell LSAP)
```

Per-cell components are stored in `(angle, max_g, violation, coverage, interference)` matrices. The pattern that emerges:
- Coverage is **saturated**: 4.306 for wide probes (4-shank NP 2.0), 1.248 for narrow probes (NP 1.0), basically the same for any viable hole.
- The 0.19 cost gap is entirely in **`max_g` differences** of ~0.01–0.3 per cell.
- Manual's BLA→hole 4 has max_g=+0.006 (borderline). LSAP-top picks BLA→13 with max_g=-0.329. Tiny but compounding.

### Cost reweighting sweep (manual rank stays out)

```
default         (δ=5.0, β=0.3): NOT in 200  manual_cost=-86.523  gap=+0.189
more clearance  (δ=5.0, β=1.0): NOT in 200  manual_cost=-88.943  gap=+0.631
less coverage   (δ=1.0, β=0.3): NOT in 200  manual_cost=-14.883  gap=+0.189
less cov + clr  (δ=1.0, β=1.0): NOT in 200  manual_cost=-17.302  gap=+0.631
clr dominates   (δ=0.5, β=2.0): NOT in 200  manual_cost=-11.803  gap=+1.262
coverage off    (δ=0.0, β=1.0): NOT in 200  manual_cost=  0.608  gap=+0.631
```

**Reweighting alone never works.** The cost gap stays small (0.19–1.26) and Murty trivial-swap variants always crowd out the manual.

### Diversified Murty (Hamming distance threshold)

```
hamming ≥ 2, exp ×200:  pool 50/50  cost range [-86.713, -86.685]  manual NOT in pool
hamming ≥ 3, exp ×500:  pool 50/50  cost range [-86.713, -86.654]  manual NOT in pool
hamming ≥ 4, exp ×1000: pool 50/50  cost range [-86.713, -86.579]  manual NOT in pool
hamming ≥ 5, exp ×2000: pool 21/50  cost range [-86.713, -86.474]  manual NOT in pool
```

(`hamming ≥ M` requires at least M of 7 probes to map to different holes vs all prior accepted; `exp ×E` means up to `k × E` candidates explored.)

Diversification widens the cost range and breaks trivial swaps, but **the manual is so deep in raw LSAP cost ordering that even high-Hamming exploration can't reach it efficiently.**

### Brute-force ground truth

For K=7, N=14: `itertools.permutations` enumerates 17.1 M arrangements. Of these, 792 K are feasible (after `violation > 1.0` hard reject). 13 s total wall.

```
Brute-force enumerated 792000 feasible arrangements in 13.0 s
Manual TRUE LSAP cost rank: 49028 / 792000
```

**The manual is in the bottom 94% by per-cell LSAP cost.** Murty would need to enumerate 49000+ candidates to surface it — far past anything practical. Diversified Murty narrows the search but doesn't escape this fundamental.

### Wider LSAP pool

```
LSAP top-5000 (default weights): manual NOT in pool. Pool worst cost: -86.618.
                                 Manual cost: -86.523. Gap from pool worst: +0.095.
```

Even k=5000 doesn't reach the manual. Estimated pool size to surface manual: > 10K, and Murty enumeration cost grows.

---

## Diagnosis: Why LSAP is the Wrong Discrete Layer

The LSAP cost is a **sum of per-cell terms** — `cost = Σ_i cost[i, σ(i)]`. It cannot encode information that depends on the joint structure of the assignment:

1. **True pairwise interference.** The `γ·interference` term in the cost matrix is a per-cell approximation: "probe P at hole H interferes with other probes at *their best-fit* holes". The actual joint pairwise interference depends on (probe_i, hole_i, probe_j, hole_j) — not encodable in a per-cell matrix.

2. **Spatial spread.** Per-cell cost happily piles all wide probes onto the same few central holes (3, 8, 11, 12, 13) because each gives best per-cell coverage. The manual's "free up high-value holes" sacrifice logic is invisible per-cell.

3. **Arc-feasibility coupling.** Holes have AP positions. The 16°-min-AP-separation rig constraint plus ≤ 3-4 arcs means certain hole combinations force a 4th arc. Per-cell LSAP can't check this.

This is a MINLP:
- Discrete: probe→hole assignment (x ∈ {0,1}^{K×N}, with permutation constraint)
- Continuous: arc APs, per-probe ml/spin/off_R/off_A/depth
- Coupling constraints: threading (continuous depends on which hole the probe is in), pairwise clearance (continuous depends on multiple holes), kinematic separation
- Objective: maximize Σ coverage(continuous, discrete)

The current pipeline approximates this as LSAP (discrete-only) → SLSQP (continuous-only). That works when per-cell LSAP cost ranks correctly, but fails on `836656/T12` because the manual's joint advantage isn't visible per-cell.

---

## Manual Plan vs LSAP Top: Why the Manual Wins Jointly

Looking at the manual's "unusual" choices:

- **BLA → hole 4** (borderline max_g=+0.006). LSAP-top picks hole 13 (max_g=-0.329). But the manual's choice frees up hole 13 for CA1, hole 12 for CLA. Trade per-cell quality for joint allocation.
- **VM → hole 7**. LSAP-top picks hole 3 for VM (best max_g=-0.999) and hole 8 for MD. But MD's per-cell-best is hole 3 too. Manual gives MD → hole 3 (its preferred), VM → hole 7 (second-best in its target region). LSAP arbitrates the contention with a sub-optimal-for-both swap; manual sacrifices VM cleanly.
- **CA1 → hole 10**. LSAP-top picks hole 13. Same story: free up hole 13 for someone else.

The pattern: manual makes per-probe sacrifices that enable the *whole assignment* to be jointly viable on a real rig. LSAP can't see this; it sums per-cell costs and arbitrates contention with whichever permutation has the lowest per-cell-sum.

---

## Proposed Path: Brute-Force + True Joint Pre-Cost

### The structural insight

**Pairwise interference at best-fit poses depends only on the 4-tuple `(probe_i, hole_i, probe_j, hole_j)`** — not on the other probes in the assignment. So we can **precompute** an exact joint-interference table once, then per-assignment evaluation is just a sum over 21 cached pair values.

This bypasses the per-cell limitation of LSAP without giving up exact joint signal.

### Numbers (K=7, N=14)

```
4-tuple cache size:  C(K,2) × N² = 21 × 196 = 4116 entries
Cost per entry:      one FCL BVH distance call at best-fit poses ≈ 50 µs
Total precompute:    4116 × 50 µs ≈ 200 ms

Per-assignment scoring: 21 dict lookups + sum ≈ 1 µs
Score all 792K feasible:                       ≈ 1 s
Sort + take top-K:                             instant
```

End-to-end: brute enumerate 792K feasible (13 s) + precompute (0.2 s) + score (1 s) + top-K to Stage 2 = **~15 s** to replace the current LSAP layer.

### Scaling

| K probes | N=14 holes | N=20 holes | Cache size (4-tuple) |
|---|---|---|---|
| 7  | 17M (~13 s feasible filter) | 390M (~5 min) | 4 K–8 K |
| 8  | 121M (~90 s) | 5.1B (infeasible) | 5.5 K–11 K |
| 9  | 727M (~10 min) | infeasible | 7 K–14 K |
| 10 | 3.6 B (infeasible) | infeasible | 9 K–18 K |

Brute force works at **K ≤ 8 for N ≤ 14**. Beyond that, need smarter pruning (e.g. per-probe hole shortlist by per-cell cost percentile, then brute-force over the cartesian product of shortlists).

### What the joint pre-cost should contain

Composite (weights need tuning):
- `LSAP_per_cell_sum` (cheap, already known) — keeps coverage / threading / approx interference signal
- `+ λ_joint · Σ pairwise_interference_at_best_fit` (precomputed cache, **the new term**)
- `+ λ_arc · arc_spread_penalty(hole_AP_coords)` — encourages hole AP positions to be clusterable into ≤ 3 arcs with ≥ 16° AP separation. Cheap closed-form.
- *(optional, more expensive)* `+ λ_thread_joint · max_threading_max_g` — flags assignments where the worst-fit probe is borderline.

Tuning these weights: validate that the manual config lands in top-K = 50. Then sweep weights and check that manual stays in top-K under reasonable perturbations. If λ is sensitive, refactor.

### Implementation plan

1. **`src/aind_low_point/optimization/joint_pre_cost.py`** (new module)
   - `precompute_pair_table(probes, holes)` → `dict[(pi, hi, pj, hj), float]` of best-fit pose pairwise FCL signed distances. Use same `make_fcl_bvh` + best-fit pose code already in `optimize.py:_inner_solve_one` / `hole_assignment.py:static_coverage`.
   - `score_assignment(assignment, lsap_cost, pair_table, hole_aps, weights) -> float`
   - `enumerate_and_score(probes, holes, weights, top_k=50)` → `list[HoleAssignment]` ranked by joint score

2. **`src/aind_low_point/optimization/joint_rerank.py:optimize_joint`**
   - Add `--use-joint-pre-cost` flag (default OFF for now)
   - When enabled, replace `solve_top_k_assignments(...)` with `enumerate_and_score(...)` for the Stage 1 → Stage 2 handoff

3. **Validation harness** (`scripts/diagnose_lsap_rank.py` extension)
   - Add a section that runs `enumerate_and_score` and reports manual's rank under the joint pre-cost
   - Target: manual in top-50

4. **End-to-end smoke test**
   - Run with `--use-joint-pre-cost --max-num-arcs 3 --min-num-arcs 3` (strict 0 mm clearance)
   - Verify the optimizer produces a feasible plan with coverage close to 17.91 (manual reference)

### Open questions

1. **Arc-spread penalty form.** Simplest is `var(hole_APs) > threshold ⇒ penalty`. Better: actual arc clustering check (can the K hole APs be 3-clustered with each cluster's spread < 16°?). The latter is a 1D k-means with constraints — fast but more code.

2. **Best-fit pose definition.** `static_coverage` uses `pose_at_hole_best_fit(hole)` shifted by centroid of shank tips. Is this the right pose to evaluate pairwise interference at? For our purposes (ranking) it's fine, but worth double-checking that the cache values match what `pairwise_headstage_clearances` would compute at the same pose.

3. **Filtering before brute-force at K=8+ and N=20+.** Need a per-probe hole shortlist (e.g. top-5 holes by per-cell max_g). Cartesian product of shortlists is much smaller than full N-permutations. Worth implementing now if anyone wants this to scale.

4. **Validation data points beyond `836656/T12`.** Have we got any other (config, manual plan) pairs to verify the joint pre-cost generalises? If only one data point, weights might overfit.

---

## State to Pick Up From

### Branch & files modified recently (this session)

```
modified:
  scripts/run_optimizer.py                          # --device, --profile, save-alternatives default-on
  src/aind_low_point/optimization/joint_rerank.py   # _sdf_jnp_payload cache, _STAGE2_TIMINGS,
                                                    # JAX spin restore wiring, spawn-mode Stage 3 pool
  src/aind_low_point/optimization/hole_assignment.py # min_hamming_distance + explore_multiplier
                                                    # added to solve_top_k_assignments
  src/aind_low_point/optimization/optimize.py       # JAX Stage 3 constraint wiring
  src/aind_low_point/optimization/joint_rerank_jax.py # padding + module-level JIT cache
                                                    # threading_g_matrix lifted to module-level

untracked (new):
  src/aind_low_point/optimization/sdf.py            # libigl SDF generation + cache
  src/aind_low_point/optimization/sdf_jax.py        # JAX pose math + SDF lookup
  src/aind_low_point/optimization/sdf_clearance.py  # JAX SDF clearance for Stage 3
  src/aind_low_point/optimization/joint_rerank_jax.py # JAX reduced objective + grad
  src/aind_low_point/optimization/stage3_jax.py     # JAX Stage 3 constraint Jacobians
  src/aind_low_point/optimization/spin_restore_jax.py # JAX spin sweep
  scripts/verify_fcl_clearances.py                  # FCL ground-truth verifier
  scripts/diagnose_lsap_rank.py                     # LSAP rank diagnostic
  scripts/validate_jax_reduced.py                   # JAX vs NumPy parity test
  scripts/bench_jax_reduced.py                      # GPU vs CPU bench
  scripts/bench_jax_recompile.py                    # JIT cache hit test
  examples/836656-config-T12_opt_alternatives/      # latest strict run alternatives
  refactor-plan.md                                  # unrelated, ignore
```

### Key file references

- `src/aind_low_point/optimization/hole_assignment.py:413` — `CostWeights` dataclass
- `src/aind_low_point/optimization/hole_assignment.py:483` — `build_cost_matrix`
- `src/aind_low_point/optimization/hole_assignment.py:617` — `solve_top_k_assignments` (Murty)
- `src/aind_low_point/optimization/hole_assignment.py:443` — `static_coverage` (best-fit pose evaluation pattern)
- `src/aind_low_point/optimization/optimize.py:80` — `ProbeStaticInfo` dataclass
- `src/aind_low_point/optimization/optimize.py:230` — `_build_inner_context`
- `src/aind_low_point/optimization/objective.py:317` — `pairwise_headstage_clearances` (FCL kernel for clearance)
- `src/aind_low_point/optimization/headstages.py` — `make_fcl_bvh`
- `src/aind_low_point/optimization/kinematics.py` — `pose_from_optimizer_vars` (NumPy version)
- `src/aind_low_point/optimization/sdf_jax.py:82` — `pose_from_optimizer_vars` (JAX version)
- `src/aind_low_point/optimization/joint_rerank.py:1527` — `optimize_joint` entry
- `scripts/run_optimizer.py:138` — `_probe_static_info` builder
- `scripts/diagnose_lsap_rank.py` — the diagnostic that surfaced the manual rank
- `examples/836656-config-T12.yml` — input config
- `examples/836656-config-T12.plan.yml` — manual ground truth
- `/tmp/836656-holes.yml` — hole spec for 836656

### Current Stage 2 component timing (small-pool diagnostic)

```
[profile] Stage 2 component breakdown (15 candidates):
slsqp                  22.84s  (65.1%)
spin_restore           10.21s  (29.1%)   ← was 24s with FCL; JAX port saves 14s
metric_eval             1.46s  ( 4.2%)
build_probe_static      0.62s  ( 1.2%)   ← SDF jnp cache fix helping
```

### Current full-run wall (1000 (H,A) candidates × 3 starts)

```
Stage 0  : 1.3 s
Stage 1  : 0.8 s
Stage 2  : 1900 s (~31.7 min) at 1.9 s/cand
Stage 3  : 270 s (~4.5 min) parallel across 15 spawn workers
Total    : ~36 min
```

### Reproducing the diagnostic

```bash
# Brute-force + Murty diagnostics
printf 'MD 3\nBLA 4\nPL 1\nVM 7\nRSP 5\nCA1 10\nCLA 12\n' | \
  uv run --python 3.13 python -m scripts.diagnose_lsap_rank \
    examples/836656-config-T12.yml /tmp/836656-holes.yml \
    --manual-plan examples/836656-config-T12.plan.yml --k 50

# Strict-clearance optimizer run with current pipeline (~36 min)
uv run --python 3.13 python scripts/run_optimizer.py \
    examples/836656-config-T12.yml /tmp/836656-holes.yml \
    --joint-rerank --sdf-clearance --device auto --profile \
    --max-num-arcs 3 --min-num-arcs 3 --verbose

# Seed-polish path (validate Stage 3 polish on the manual plan)
uv run --python 3.13 python scripts/run_optimizer.py \
    examples/836656-config-T12.yml /tmp/836656-holes.yml \
    --joint-rerank --sdf-clearance --max-num-arcs 3 --min-num-arcs 3 \
    --seed-plan examples/836656-config-T12.plan.yml \
    --clearance-overlap-allowance-mm 0 --verbose
```

---

## Things That Aren't Solutions

For the record, so future agents don't redo these:

1. **Reweighting `CostWeights`.** Tried six configurations including coverage off, clearance dominates, both extremes. Manual stays past rank 200 in every case. The cost-gap to LSAP-best ranges 0.19 to 1.26, but Murty trivial swaps always fill that gap.

2. **Wider Murty pool (k=5000).** Manual still NOT FOUND. Murty packs 5000 trivial-swap variants within a 0.095 cost band of LSAP-best. Going to k=50K would surface manual but Murty enumeration cost grows superlinearly.

3. **Diversified Murty (Hamming distance enforcement).** Widens the cost range explored (from 0.028 spread at h≥2 to 0.24 spread at h≥5) but still doesn't reach manual within practical explore budgets. Manual is genuinely deep in cost-ordered exploration.

4. **Joint pre-score on Murty pool.** Doesn't help because the manual isn't IN the Murty pool — re-ranking can't surface what isn't enumerated.

5. **Local 2-opt search from Murty seeds.** Plausible but manual differs from LSAP-top by 5 of 7 probes (Hamming = 5). 2-swap would need 2-3 steps with intermediate states improving in joint cost. Unverified whether such a monotone path exists.

6. **Feeding `--seed-plan` to the optimizer.** This *works* but defeats the purpose: the goal is for the optimizer to discover manual-quality plans without being told the answer.

7. **AABB / convex hull / capsule approximations of probe geometry for clearance.** All tried earlier and rejected — they miss the silicon body region of NP 2.1 and similar features. Use full-mesh FCL BVH everywhere.

---

## Glossary

- **LSAP** — Linear Sum Assignment Problem. Min-cost bipartite matching of K probes to N holes.
- **Murty's algorithm** — top-k enumeration of LSAP solutions via partition.
- **SLSQP** — Sequential Least-Squares Programming (scipy.optimize). The continuous-NLP solver used in Stages 2/3.
- **SDF** — Signed Distance Field. Voxel grid of signed distance to mesh surface. Differentiable; supports `jax.grad`.
- **MINLP** — Mixed Integer Non-Linear Programming. Discrete + continuous variables, non-linear constraints.
- **best-fit pose** — pose where the shank-row centroid lands at the slot bottom with shank row aligned with slot major axis. See `pose_at_hole_best_fit` in `kinematics.py`.
- **max_g** — threading oval value. ≤ 0 = inside the oval (feasible). > 0 = outside (violation).
- **arc** — kinematic structure on the rig. Each probe is mounted on one arc; arcs have a fixed AP angle. Probes on the same arc must share that AP. Rig has 2–4 arcs available.
- **headstage** — the electronics board on the back of the probe. Larger than the silicon; the dominant collision source between adjacent probes.

---

## Update 2026-05-18: Visibility Atlas + Arc-First Search

### Where we landed

The brute-force + joint pre-cost approach above was tried and **falsified by data**. We then built two structural reframings:

1. **Target-aligned, JAX-vmapped visibility atlas** (`src/aind_low_point/optimization/visibility_atlas.py`). Replaces the SLSQP-anchored atlas with closed-form ray-cast geometry: per (probe, hole, top-ellipse-sample, spin), build pose from chord direction, test whether all K shanks thread all hole sections. JAX `vmap` over (top × spin), hole-section parameters closure-captured per JIT. Builds in **3.5 s** vs 156 s for the SLSQP atlas. Manual present in all 7 (probe, hole) pairs with `n_top=96, n_spin=72`, interior-of-top sampling (not boundary), oval slack 0.2.

2. **Arc-first enumeration** (`scripts/diagnose_arc_first.py`). Inverts the search: instead of `LSAP → top-K H → enumerate arc partitions`, do `partition → arc-AP triple → bipartite match probes to holes`. Each emitted candidate is joint `(H, partition, arc-AP triple)` already arc-feasible by construction.

### What additional joint-pre-cost diagnostics established

On the SLSQP-atlas pool (43 K H, manual rank 11 645) and visibility-atlas pool (41 K H, manual rank 12 779), we ran a sweep of joint-cost variants (see `scripts/diagnose_joint_pre_cost.py`). **No per-H ranking signal moves the manual into Stage-2-tractable range:**

```
LSAP only                       11 645
LSAP + best-fit pair clearance  65 102 (worse)
LSAP + arc-AP feas at best-fit 147 421 (much worse)
LSAP + pair + arc              164 747 (much worse)
pair + arc, no LSAP            330 230 (terrible)
minimax + pair + arc           206 914 (worse)
–coverage + pair + arc         330 231 (terrible)
sum min target_miss             24 421
sum min thread max_g            16 798
sum interval widths             23 729
joint ML-sep (greedy AP placement) 5 035
```

Best per-H signal ranks manual at ~5K. **Joint quality emerges only after polish**, not from any cell-local property.

### Atlas correctness validation (manual coverage)

With current visibility-atlas settings (`n_top=96, n_spin=72, oval_slack=0.2, interior sampling`):

```
MD→3   ✓  AP interval [+6.2, +17.2]
BLA→4  ✓  [-12.1, -6.1]
PL→1   ✓  [-16.2, +1.7]
VM→7   ✓  [-45.2, -38.0]
RSP→5  ✓  [-60.5, -10.6]
CA1→10 ✓  [-23.7, -7.7]
CLA→12 ✓  [-43.3, -40.7]
```

All 7 manual pairs present in atlas with manual planned AP inside the interval.

### Arc-first attempts and the bug

First pass: stopped at the first matching per (partition, AP-triple). Manual not found in 72K candidates.

Second pass: enumerated all matchings (cap 50 per combo). Still no manual in 280K candidates.

Focused diagnostic on the **manual partition + manual AP triple exactly** revealed the actual bug. At arc c (VM, RSP, CLA at AP=-43°):

```
nearest-anchor MLs:
  VM→7   ml = -12.8  (manual is  -8.0)
  RSP→5  ml = +24.9  (manual is  +8.0)
  CLA→12 ml = -26.3  (manual is -24.0)

pairwise:
  |ml_VM  - ml_RSP| = 37.7  ✓
  |ml_VM  - ml_CLA| = 13.5  ✗  FAIL  (manual: 16.0  ✓)
  |ml_RSP - ml_CLA| = 51.3  ✓
```

The atlas anchors don't include the exact manual MLs, and the **nearest-anchor-by-AP picks ML values that happen to violate the 16° intra-arc separation**. With a different anchor at the same AP (different spin), MLs would be different and could satisfy.

So the bug is in my matching code, not the atlas: I picked **one** ML per (probe, hole) at the chosen AP, instead of enumerating **all** ML options across all anchors at that AP.

### The correct architecture (per agent feedback)

The natural search unit is `(partition, AP triple, local arc configs)` not `(partition, AP triple, H)`:

```
for each unordered partition of K probes into ≤ max_arcs groups:
    for each arc group G and AP bin a:
        LocalConfigs[G, a] = all probe→(hole, anchor) assignments where:
            - each probe's atlas has an anchor with AP within tolerance of a
            - holes are distinct within the arc
            - intra-arc ml separation ≥ min_ml_sep_deg
        (per-arc support: a is "supported" if LocalConfigs[G, a] is non-empty)

    for each arc-AP triple (a0, a1, a2) with pairwise ≥ min_arc_ap_sep:
        if a_i in support[group_i] for every group i:
            for each combination of one local config per arc:
                if cross-arc hole uniqueness:
                    emit GlobalCandidate(partition, AP_tuple, per-arc local config)

rank GlobalCandidates by margin / coverage / robustness (NOT LSAP cost)
top-K → reduced SLSQP preview → full Stage 3 polish
```

Critical points:

- **ML separation is local to the arc** — it's a constraint that drives which anchors per probe to pick, not a global filter applied after a matching.
- **Per probe at AP, enumerate ALL (hole, anchor) pairs** with anchor AP within tolerance — there are typically many anchors per (probe, hole) at varying spins, and they have different MLs.
- **Per-arc support is a bitset**: efficient AP-triple enumeration via bitset of supported AP bins per arc, then check pairwise separation.
- **Ranking** uses joint robustness (margins, coverage, AP/ML slack); LSAP cost is at most a weak tie-breaker. Never penalize wide AP span — the manual uses a 56° span.
- **Manual-membership diagnostic** runs first, before any caps. If manual is not emitted at no-caps, fix the atlas (anchor density, oval slack, AP tolerance) rather than the search.

The earlier diagnostic `scripts/diagnose_arc_first.py` is **the wrong shape** — it does single-anchor matching at AP bins and caps by global backtracking order. Do not extend it; rewrite per above.

---

## Immediate Next Implementation: Local-Arc-Config Rewrite

### Files to add / edit

1. **`scripts/diagnose_local_arc_configs.py` (new)** — diagnostic prototype, not production wiring yet. The diagnostic should:
   - Build the visibility atlas once.
   - Define `gather_probe_choices(probe_idx, arc_ap, ap_tol)` → `list[(hole_id, ml, spin, anchor_ap)]`. Crucially: collect ALL anchors at AP within tolerance, not just the nearest.
   - Define `local_arc_configs(group_probes, arc_ap, min_ml_sep)` → `list[dict{probe_idx: (hole, ml, spin)}]` via backtracking with intra-arc distinct-hole + ml-sep constraints.
   - **Manual-membership check (no caps)**: at the exact manual partition `[{0}, {1, 2, 5}, {3, 4, 6}]` and manual AP triple `(+13, -10, -43)`, enumerate local configs per arc; check whether the manual (probe → hole) is in each arc's config list. Report per-arc local-config count and pass/fail.
   - **Global assembly**: enumerate partitions × AP triples × cross-arc local configs with global hole uniqueness; report total candidate count and manual rank.

2. **If diagnostic passes**: promote to `src/aind_low_point/optimization/arc_first_search.py` (production module), wire as Stage 1 in `optimize_joint` behind `--arc-first` flag.

### Concrete pseudocode for the diagnostic

```python
# Builds on visibility_atlas.Atlas; each entry has anchors: tuple[PoseAnchor, ...]
def gather_probe_choices(atlas, probe_name, arc_ap, ap_tol):
    choices = []
    for hid in atlas.hole_ids:
        e = atlas.entries[(probe_name, hid)]
        if e.ap_min is None:
            continue
        for a in e.anchors:                 # ← ALL anchors, not nearest
            if abs(a.ap_deg - arc_ap) <= ap_tol:
                choices.append((hid, a.ml_deg, a.spin_deg))
    return choices

def local_arc_configs(probe_indices, arc_ap, atlas, probe_names, min_ml_sep, ap_tol):
    per_probe = [
        gather_probe_choices(atlas, probe_names[i], arc_ap, ap_tol)
        for i in probe_indices
    ]
    out = []
    def search(idx, used_holes, mls, current):
        if idx == len(probe_indices):
            out.append(dict(current)); return
        i = probe_indices[idx]
        for hid, ml, spin in per_probe[idx]:
            if hid in used_holes: continue
            if any(abs(ml - m) < min_ml_sep for m in mls.values()): continue
            used_holes.add(hid); mls[i] = ml; current[i] = (hid, ml, spin)
            search(idx + 1, used_holes, mls, current)
            used_holes.discard(hid); del mls[i]; del current[i]
    search(0, set(), {}, {})
    return out

# Manual partition for 836656/T12
manual_partition = [(0,), (1, 2, 5), (3, 4, 6)]   # MD | BLA, PL, CA1 | VM, RSP, CLA
manual_ap_triple = (13.0, -10.0, -43.0)

# Per-arc local-config count + manual membership
for arc_idx, (group, ap) in enumerate(zip(manual_partition, manual_ap_triple)):
    cfgs = local_arc_configs(group, ap, atlas, probe_names,
                             min_ml_sep=16.0, ap_tol=3.0)
    manual_in = any(
        all(cfg[i][0] == manual_h[probe_names[i]] for i in group)
        for cfg in cfgs
    )
    print(f"arc {arc_idx}: {len(cfgs)} configs, manual {'IN' if manual_in else 'NOT IN'}")
```

### Settings to start from

```
n_top         = 96      # already validated for manual coverage
n_spin        = 72      # 5° spin grid
oval_slack    = 0.2
ap_tol        = 3.0     # anchor AP must be within this of arc AP
min_arc_ap_sep_deg = 16.0
min_intra_arc_ml_sep_deg = 16.0
max_arcs      = 3
```

If any manual arc returns zero local configs, the failure modes to check in order (per agent feedback):

1. Atlas edge missing → check `atlas.entries[(probe, hole)].ap_min is None`
2. AP tolerance too tight → widen `ap_tol` from 3 to 5
3. Anchor density too sparse at manual AP → bump `n_spin` to 144
4. ML sep too strict → temporarily relax `min_ml_sep` and see if it appears
5. Oval slack too tight → bump from 0.2 to 0.3
6. Top-sample grid missed the manual ML basin → bump `n_top` to 128

Do not tighten the atlas. Do not cap matchings. Manual must appear at full enumeration before optimizing.

### Files to know

```
src/aind_low_point/optimization/visibility_atlas.py    # ATLAS: production
src/aind_low_point/optimization/atlas.py               # earlier SLSQP atlas (kept)
src/aind_low_point/optimization/joint_rerank.py        # optimize_joint entry
scripts/diagnose_arc_first.py                          # BUGGY — do not extend
scripts/diagnose_atlas_pass1.py                        # AP-interval atlas debug
scripts/diagnose_joint_pre_cost.py                     # joint-ranking sweep
scripts/diagnose_lsap_rank.py                          # LSAP-rank brute force
examples/836656-config-T12.yml                         # input config
examples/836656-config-T12.plan.yml                    # manual ground truth
/tmp/836656-holes.yml                                  # hole spec
```

Manual probe→hole for 836656/T12: `{MD: 3, BLA: 4, PL: 1, VM: 7, RSP: 5, CA1: 10, CLA: 12}`.
Manual arc APs: `a=+13.0, b=-10.0, c=-43.0`. Coverage 17.91, max_viol 0 (strict feasible).

### What success looks like

Manual-membership diagnostic emits manual H as a local-arc-config triple at the manual partition + AP triple at no caps. Global assembly count: target ≤ 10K candidates with manual in pool. Top-K ranking by joint metric surfaces manual within top ~50–200.

If those hold, wire as Stage 1 and run end-to-end. The remaining Stage 2/3 pipeline is unchanged.

#### Resolution (2026-05-18 post-compaction)

Implemented in `scripts/diagnose_local_arc_configs.py`. Two fixes vs. the brief's pseudocode were needed to avoid OOM:

1. **Dedup `gather_probe_choices` by 0.5° ML bins per (probe, hole)**. Each atlas entry has up to `n_top × n_spin = ~7k` anchors; without dedup, the per-probe choice list at one AP can run to tens of thousands and the 3-arc backtracker blows up.
2. **Direct manual-achievability check** instead of "enumerate all local configs and filter for manual". At fixed `manual_h_for_arc`, run a small recursive search over `(restrict_holes={manual_h})`-filtered choices that just returns whether ML-sep is satisfiable. O(deduped-bins^|arc|) per arc — small.

With those, the diagnostic runs in ~15s at the working settings:

```
--ap-tol 5 --n-top 128 --n-spin 72 --min-ml-sep 16 --min-arc-sep 16
```

Result: all 3 manual arcs ACHIEVABLE. Atlas representatives sit ~2° off the exact manual MLs (VM=-10.2 vs manual -8; CLA=-26.6 vs manual -24); Stage 2/3 polish moves them into place.

What **did not** help: `--n-spin 144` (no change to the failing pair).

What did help: `--ap-tol 5` widened RSP/PL bins; `--n-top 128` shifted VM and CLA basin edges by ~0.3° each — enough to clear the 16° pairwise sep.

#### Pitfall: do NOT run full global assembly unbounded

The current `--global-assembly` path enumerates `partitions × AP grid × per-arc local configs × cross-arc combinations`. At 7 probes this is huge. Earlier attempts OOM'd. Until a proper top-K / streaming ranker is in place, **leave `--global-assembly` off** — the manual-achievability check is the gating diagnostic and runs cheaply.

The right next step is one of:
- **(a) Direct top-K-per-cell ranking.** For each `(partition, AP triple)` cell where manual achievability holds, compute one cheap cell-score (e.g. per-probe target_miss at best-fit pose), keep top-K cells, and emit one representative local config per cell.
- **(b) Sample-based assembly.** Stream a bounded sample of `(partition, AP triple, local config)` tuples from the inner loop, accumulate top-K by joint metric, and never materialise the full cartesian product.

Both decouple "manual is in the *implicit* pool" (proven) from "manual is in the *materialised* top-K" (the real test). Stage 2 then polishes the top-K.

### Things explicitly ruled out by data (do not redo)

Everything in the "Things That Aren't Solutions" section above, PLUS:

- Nearest-anchor-by-AP per (probe, hole) at fixed arc AP. Confirmed to drop the manual ML basin.
- Capping matchings per (partition, AP triple) by global backtracking order.
- Using LSAP cost as anything other than a weak tie-breaker in arc-first ranking.
- Picking arc APs by best-fit-pose AP clustering.
- Penalising wide AP span (manual uses 56° span across 3 arcs).
- Tightening visibility-atlas thresholds enough to shrink the pool — manual sits at the threshold and will drop out.

### Open questions for the next agent

1. **Local-config count at manual partition + manual AP triple.** Critical: this is the gating diagnostic.
2. **Total global candidate count** under no-caps arc-first enumeration. If ≤ 10K, we're done with discrete-layer search; ranking sorts out the rest.
3. **Ranking metric that surfaces manual-quality.** With every candidate already arc + ML feasible, simpler joint metrics (sum_min_threading_margin, min_arc_ap_sep_margin, total_coverage_at_anchors, robustness_over_neighbouring_AP_bins) might work where they failed on the un-pre-filtered pool.
4. **Performance of full enumeration**: with 365 partitions × ~60 AP bins per arc, the per-(partition, AP-triple) backtracking is the inner loop. Profile + vectorise / JAX if it's too slow.



