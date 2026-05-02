# Probe Placement Optimizer (planned)

Status: **deferred — to be picked up after manual-placement MVP lands.**

## Goal

Given:
- a fixed set of **target brain regions** (one per probe) with associated
  per-region density volumes (e.g. retrograde tracer density, or just a
  uniform-on-mask binary volume),
- an **implant fixture** with N (~14) openings,
- a fleet of **K (~7) probes** with known shaft and headstage geometry,

automatically find the joint placement (per-probe angles, entry offset, depth,
and probe→hole assignment) that:

1. **Feasibility (hard)** — every probe shaft passes through *some* implant
   hole without grazing the hole walls; bulky probe headstages above the
   implant don't collide with each other.
2. **Coverage (primary objective)** — maximise total integrated target-region
   density along each probe's shaft (i.e., "as much of the right tissue as
   possible per probe").
3. **Margin (secondary objective)** — once feasible and coverage-good,
   maximise min clearance: shaft-vs-hole-edge and headstage-vs-headstage.

Probe→hole assignment is **not specified by the user** — the optimizer picks.

## Algorithm structure (two-level)

```
                    ┌──────────────────────────┐
                    │  enumerate top-K          │
                    │  probe→hole assignments   │  ← Hungarian / LSAP on
                    │  (ranked by heuristic)    │     a feasibility-aware
                    └──────────┬───────────────┘     pair cost
                               │
                ┌──────────────┴──────────────┐
                ▼                              ▼
   continuous opt (assignment 0)    ...    continuous opt (assignment K-1)
   ┌─────────────────────────────┐
   │ vars: per-probe              │
   │   (ap, ml, spin,             │
   │    entry_offset_2D_in_hole,  │
   │    past_target_mm)           │
   │   ~5 DOF × K probes = ~35    │
   │                              │
   │ stages:                      │
   │   1. CMA-ES global  (cma)    │
   │   2. SLSQP local    (scipy)  │  ← gradient-based polish
   └─────────────────────────────┘
                               │
                    pick best across assignments
```

### Outer (combinatorial)

- Build a (K × N) cost matrix where `cost[i][j]` is a heuristic feasibility
  score for routing probe i through hole j to target i. Cheap proxies:
  angle from the hole's nominal axis to the line `(hole_center → target_i)`,
  plus a soft penalty if the cone of valid angles intersects another probe's
  required cone.
- `scipy.optimize.linear_sum_assignment` solves the optimal assignment in O(KN²).
- For robustness, enumerate top-K assignments via LSAP variants (Murty's
  algorithm) and run the inner stage on each. Empirically K=5–10 is plenty.

### Inner (continuous)

Variables per probe (5 each, K probes total):
- `ap_local, ml_local, spin` — angles, in degrees, respecting arc coupling
  if `bind_ap_to_arc=True`.
- `entry_offset_(R, A)` — 2D offset within the assigned hole, mm. Bounded by
  the hole radius minus the shaft radius.
- `past_target_mm` — how far past the target centroid the tip extends.

Stages:
1. **CMA-ES** (`cma` library): population-based, derivative-free, handles
   bounds. Starts from broad initial sigma; converges to a feasible-and-good
   region in 50–200 generations on this dimensionality.
2. **SLSQP** (`scipy.optimize.minimize`) for local polish: smooth objective,
   gradient via JAX autodiff, constraint formulation `g_i(x) ≥ 0` for each
   feasibility metric (clearance min ≥ ε).

## Geometric primitives

All built once at config-load time and cached:

| Primitive | What | Library |
|---|---|---|
| `FixtureSDF` | voxel signed-distance field per static collider (implant, headframe, well). Negative inside material; **positive in the hole tunnels** (this is the key — convex decomposition can't represent holes; SDF can). | `mesh_to_sdf`, `igl.signed_distance`, or custom `trimesh.proximity` over a grid |
| `TargetDensity` | per-probe volume of "where to record." Uniform-on-CCF-mask, or a scalar density (e.g. retrograde tracer count). Voxel grid + trilinear interp. | numpy / JAX |
| `ProbeCapsule(p0, p1, r)` | shaft model; signed distance to a point is closed-form. | hand-rolled |
| `HeadstageCapsule(p0, p1, r)` | bulky-top model above the shaft, same primitive. | hand-rolled |
| `KinematicJacobian` | `∂(p0, p1)/∂(ap, ml, spin, entry, depth)` — closed form via `arc_angles_to_affine` derivative. | analytical or JAX autodiff |

Per-evaluation cost (rough):
- 7 probes × 30 shaft samples × 2 SDF lookups (implant, target density) ≈ 420 trilinear lookups, ~10 µs each → ~4 ms.
- C(7, 2) = 21 headstage-headstage capsule queries, ~µs each → negligible.
- Forward + gradient via JAX: ~2× forward → ~8 ms.
- CMA-ES population 30 × 100 generations = 3000 evals × 8 ms ≈ 24 s per assignment.

## Objective function

Lexicographic, but smoothly stitched via penalties for tractability:

```
J(x) = -coverage(x)
     + λ_feas · max(0, -min_clearance_shaft_hole(x))²
     + λ_feas · max(0, -min_clearance_headstage(x))²
     - λ_margin · softmin(clearances(x), β)
```

- `coverage(x)`: sum over probes of `∫₀ᴸ density_i(p_i(s)) ds` evaluated by
  sampling.
- `min_clearance_shaft_hole(x)`: min over (probe, shaft sample) of
  `implant_sdf(p_i(s))`. **Positive** inside the hole, **negative** in
  fixture material.
- `min_clearance_headstage(x)`: min over (i, j) pairs of capsule-capsule
  signed distance.
- `softmin(...)`: `-β · log(Σ exp(-d_k / β))` — smooth approximation of min
  for the margin term.

`λ_feas` is large; we use a homotopy schedule (start moderate, ramp up) to
keep CMA-ES from getting trapped in deeply infeasible regions early.

## Library plan

- **JAX** for the differentiable inner-loop (kinematic chain → capsule
  positions → SDF / density lookups → coverage, clearances). Trilinear
  interpolation over voxel grids is `jax.scipy.ndimage.map_coordinates` or a
  hand-rolled 8-tap.
- **`cma`** for CMA-ES (`pip install cma`) — global stage.
- **`scipy.optimize.minimize(method="SLSQP")`** for local polish with
  constraints; jacobian provided via `jax.jacrev`.
- **`scipy.optimize.linear_sum_assignment`** for outer LSAP.
- **`mesh_to_sdf`** or libigl bindings for fixture SDF generation
  (one-time per fixture; cached to disk keyed by mesh hash).

## Where the code goes

Probably its own package or a sibling module:

```
src/aind_low_point/
└── optimization/
    ├── __init__.py
    ├── sdf.py               # FixtureSDF + voxelizer + cache
    ├── density.py           # TargetDensity + CCF-mask helpers
    ├── geometry.py          # Capsule, capsule-capsule SDF
    ├── kinematics.py        # JAX-friendly probe pose Jacobian
    ├── objective.py         # coverage / clearance / penalties (JAX)
    ├── assignment.py        # LSAP + Murty
    └── optimize.py          # CMA-ES → SLSQP wrapper
```

Tests:
- Unit tests for each primitive (capsule SDF closed-form vs. brute-force
  numerical, SDF gradient consistency, kinematic Jacobian vs. finite-diff).
- One small end-to-end smoke test on a synthetic 2-probe / 4-hole problem
  with known optimum.

## Open questions to resolve before starting

1. **Headstage geometry.** Is it well-approximated by a single capsule, or
   does it need a small convex hull / box? The current package has the
   probe meshes loaded as a single object — probably need to split shaft
   from headstage in the asset, or model the headstage as a separate
   collidable spec attached to the same probe.
2. **Hole modelling.** Are the implant holes truly cylindrical (so we can
   reduce "entry within hole" to a 2D bounded offset), or shaped (so we
   need the full SDF tunnel)? If cylindrical, the entry-offset variable is
   bounded by `hole_radius - shaft_radius`. If shaped, we just clip to the
   set `{x : implant_sdf(x) > shaft_radius}` projected onto the entry plane.
3. **Density representation pipeline.** Where does each probe's
   `TargetDensity` come from? Likely a CCF region mask warped into the
   working frame, possibly weighted by a tracer density volume. Need a
   small builder that takes a CCF acronym + (optional) density nrrd and
   returns a `TargetDensity`.
4. **Which frame the optimizer runs in.** Internal canonical is LPS mm;
   anything imported (CCF density, fixture mesh) needs to be in that frame
   first. The existing canonicalization pipeline handles this but needs to
   apply to density volumes too.

## Out of scope (for v1)

- Calibration uncertainty → robust optimization (mean ± noise).
- Multi-day staged insertion plans.
- Time-of-day / temperature drift compensation.
- Probe re-use across days.

## Picking this up later

Sequence to follow:
1. Build `geometry.py` (capsule + capsule-capsule) — small, pure, easy to
   test.
2. Build `sdf.py` with disk caching — verify on the real implant mesh that
   holes show up as positive-SDF tunnels (visualize a slice).
3. Build `kinematics.py` — verify the JAX-autodiff Jacobian matches
   `arc_angles_to_affine` finite-difference at a few points.
4. Build `objective.py` end-to-end on a synthetic problem; visualize the
   loss landscape on a 2D slice (vary one probe's AP × ML, hold rest
   fixed).
5. Add `assignment.py` heuristic + LSAP.
6. Wire CMA-ES, then SLSQP polish.
7. Hook into the runtime: add a "Run optimizer" button to TrameController
   that takes the current PlanningState, runs the optimizer, returns a
   `PlanningState` to apply.

Before any of that: **manual placement MVP must be solid**, both because
(a) the user is the gold-standard "optimizer" we'll compare against and
(b) a good warm-start dramatically reduces CMA-ES population/generations.
