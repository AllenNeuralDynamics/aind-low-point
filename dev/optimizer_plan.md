# Probe Placement Optimizer (planned)

Status: **inner-loop primitives landing now. Outer driver still deferred.**

## Glossary (acronyms used below)

| Term | Expansion | What it means in this context |
|---|---|---|
| **AP / ML** | Anterior-Posterior / Mediolateral | Anatomical tilt axes of the probe holder. AP is roughly the rostral-caudal angle; ML is the lateral angle perpendicular to AP. |
| **CCF** | Common Coordinate Framework | Allen Brain Atlas's reference brain space. Used for naming and bounding target regions. |
| **CMA-ES** | Covariance Matrix Adaptation Evolution Strategy | A derivative-free, population-based global optimizer. It samples a population of candidates from a Gaussian, ranks them by the objective, and updates the Gaussian's mean + covariance to bias future samples toward better candidates. Good at escaping local minima; converges slowly compared to gradient methods. |
| **DOF** | Degrees of Freedom | Number of free continuous variables describing a probe's pose. |
| **LPS** | Left-Posterior-Superior | This codebase's canonical anatomical axis convention. Internal everything is in LPS millimetres. |
| **LSAP** | Linear Sum Assignment Problem | "Given an N×N cost matrix, find the minimum-cost permutation." Solvable in polynomial time by the Hungarian algorithm; produces the best probe→hole pairing. |
| **PCA** | Principal Component Analysis | Used in the hole extractor to find each bore's axis. |
| **SDF** | Signed Distance Field/Function | Distance to the nearest surface, with sign denoting inside/outside (negative inside material). Two flavours: *analytical* (closed-form for primitives like capsules) and *voxel* (precomputed grid for arbitrary meshes, queried by trilinear interpolation). |
| **SLSQP** | Sequential Least SQuares Programming | A local, gradient-based, constrained optimizer. Each iteration approximates the objective and constraints quadratically and solves a small QP for the step. Excellent at polishing a feasible-region warm start; needs gradients. |
| **JAX** | (not an acronym) | Library for differentiable numerical Python. Provides `jax.grad` / `jax.jacrev` for autodiff — we use it so SLSQP gets gradients for free, given a JAX-compatible objective. |

## Goal

Given:
- a fixed set of **target brain regions** (one per probe) with associated
  per-region density volumes (e.g. retrograde tracer density, or just a
  uniform-on-mask binary volume),
- an **implant fixture** with N (~14) bores (the build5 implant has 15
  by direct extraction; 1.20 × 0.70 mm slot holes with chamfered tops
  and per-hole axes that fan up to ~24° from vertical),
- a fleet of **K (~7) probes** with known shaft and headstage geometry,
  including 4-shank Neuropixels 2.0 probes (4 shanks at 250 µm pitch
  → 750 µm total span — must thread the slot's *major* axis),

automatically find the joint placement (per-probe arc *id*, arc and
probe angles, entry offset, depth, and probe→hole assignment) that:

1. **Feasibility (hard).** Every probe shaft (or every shank, for
   multi-shank probes) passes through *some* implant hole without
   grazing the hole walls; bulky probe headstages above the implant
   don't collide with each other or with non-implant fixtures.
2. **Coverage (primary objective).** Maximise total integrated
   target-region density along each probe's shaft — i.e., "as much of
   the right tissue as possible per probe."
3. **Margin (secondary objective).** Once feasible and coverage-good,
   maximise the minimum clearance: shaft-vs-hole-edge and
   headstage-vs-headstage.

**Both probe→hole and probe→arc are optimizer outputs, not user
inputs.** Arc assignment is empirically the hardest part of manual
placement and the optimizer should re-derive it from scratch — the
``arc_id`` field on each ``ProbePlan`` in the user's config is treated
as advisory at most, ignored entirely on first solve. ``num_arcs`` (2-4
on the AIND rig) and the rig's per-arc kinematic limits are config
inputs; everything else about arcs is computed.

## Algorithm structure (three-level)

```
       ┌──────────────────────────────┐
       │ enumerate top-K_h             │  ← LSAP / Murty on
       │ probe→hole assignments        │     hole-target geometry
       └──────────────┬────────────────┘
                      │  for each hole assignment:
       ┌──────────────┴────────────────┐
       │ enumerate top-K_a             │  ← cluster probes by
       │ probe→arc assignments         │     required-AP, filter
       │ (cluster on required AP)      │     by capacity + AP-sep
       └──────────────┬────────────────┘
                      │  for each (hole, arc) pair:
                      ▼
           ┌──────────────────────────────┐
           │ continuous opt               │
           │ vars: per-arc ap (num_arcs), │
           │       per-probe ml, spin,    │
           │            entry offset 2D,  │
           │            past_target_mm    │
           │ stages:                      │
           │   1. CMA-ES (global)         │
           │   2. SLSQP (local polish)    │
           └──────────────────────────────┘
                      │
        pick best across (hole × arc) combinations
```

**Why three levels?** Discrete vs. continuous separates as before, but
the discrete part has its own structure: hole assignment is dominated
by *target-line geometry* (which bore points at which target), while
arc assignment is dominated by *AP-coupling kinematics* (which probes
need similar AP angles, so they can share an arc). You can rank holes
without knowing arcs; you cannot rank arcs without knowing holes
(because the hole's bore axis is what tells you the required AP).
Hence: holes first, arcs second, continuous third.

### Outer level (probe→hole assignment)

Build a (K × N) cost matrix where `cost[i][j]` is a heuristic
feasibility-and-fit score for routing probe *i* through hole *j* to
target *i*. The cost combines three components, each computed once
per pair (no per-pair optimization required):

1. **Target-line alignment.** Angle between the hole's bore axis and
   the line `(hole_center → target_i)`. Smaller angle = easier to
   thread the probe with its tip on target. Dominant term.
2. **Static threading clearance.** At the geometric "best-fit pose"
   for the pair — probe's shank row centered on the hole, spin
   aligned to the slot's major axis (modulo 180°), shaft along the
   hole's bore axis — evaluate the threading constraint:

   ```
   max_g(probe, hole) = max over (shanks × sections) of
                        oval_value(shank_tip projected onto section)
   ```

   This is a **static SDF-style clearance** — *how much room does
   this probe have inside this hole, assuming the geometrically
   optimal pose?* Smaller (more negative) = more clearance.
   `max_g > 0` means the probe physically doesn't fit through the
   slot at all (e.g. 4-shank probe with span > slot major
   diameter) — **hard reject** that pair from the LSAP. Among
   feasible pairs, smaller `max_g` is better. For 4-shank NP 2.0
   through a 1.20 × 0.70 build5 slot at perfect alignment,
   `max_g ≈ −0.61`; a 15° spin misalignment costs ~0.08 in margin.
3. **Soft pairwise interference penalty.** If two probes' valid-angle
   cones (from criterion 1) overlap significantly, their headstages
   may collide. Catches obvious joint-infeasibility before reaching
   the inner loop.
4. **Coarse arc-feasibility check.** Compute `required_ap` for every
   probe under this assignment; reject if those APs can't be
   partitioned into ≤num_arcs feasible groups (capacity ≤4 per arc,
   centroids ≥16° apart pairwise). Don't waste a middle-layer run on
   an unworkable hole assignment.

Combined cost (lexicographic via weighted sum, weights tuned so
target-line alignment dominates and clearance is a tiebreaker):

```
cost[i][j] = α * angle_to_target(i, j)         # primary
           + β * max_g(probe_i, hole_j)        # tiebreaker, β < α
           + γ * pairwise_interference(i, j)   # soft penalty
        +∞ if max_g > 0 OR arc-feasibility check fails  # hard reject
```

`scipy.optimize.linear_sum_assignment` solves the optimal assignment
in O(KN²). For robustness, enumerate top-K_h assignments via Murty's
algorithm (a standard k-best variant of LSAP) and run the middle
layer on each. Empirically K_h = 5–10 is plenty for this problem size.

**Why static `max_g` and not a dynamic per-pose evaluation?** The
LSAP needs scalar costs per pair — running a small CMA-ES inside the
cost matrix would be circular and expensive. The geometric best-fit
pose is closed-form (center + axis + slot-aligned spin) and gives a
fair upper bound on clearance: any actual optimized pose can only do
*as good or better* once probe-probe constraints are also at play.
So `max_g` is a valid (and gradient-free) ranking signal.

### Middle level (probe→arc assignment)

Given a probe→hole assignment, every probe *i* has an extracted hole
axis `axis_i_LPS`. Project that axis onto the rig's AP plane to get a
**required-AP** angle:

```
required_ap(i) = atan2(axis_i · ê_AP, axis_i · ê_S)
```

(``ê_AP, ê_S`` are the rig's AP and superior unit vectors.) This
single number per probe summarises "what AP must the arc sit at to
align this probe to its bore." Probes with similar required-AP
naturally pair on the same arc.

Arc assignment is then a 1D constrained partition of K required-APs
into num_arcs labelled groups:

- **Cluster.** k-means (or a small enumeration since K = 7 is tiny)
  partitions the K values into num_arcs groups.
- **Filter — per-arc capacity.** With ML range ±30° and 16° pairwise
  minimum, an arc holds at most `floor(60°/16°) + 1 = 4` probes.
  Reject partitions exceeding this.
- **Filter — inter-arc AP separation.** Cluster centroids must be
  ≥16° apart pairwise. With num_arcs = 4 the centroids must span
  ≥48° of AP — flag at config load if the rig can't.
- **Symmetry quotient.** Two assignments differing only by a permutation
  of arc labels are equivalent. Order arcs ascending in AP centroid;
  reduces the assignment count by `num_arcs!`.

Rank surviving partitions by total within-cluster variance (sum of
squared distances to centroid). Take top-K_a (≈ 5).

For K = 7 probes into 2-4 arcs, the unfiltered Stirling-style count is
S(7, 2) = 63, S(7, 3) = 301, S(7, 4) = 350. After capacity + AP-sep
filters, typically 5–20 valid partitions per hole assignment — mostly
covered by K_a = 5.

### Combined enumeration cost

| Layer | Multiplicand | Notes |
|---|---|---|
| Hole assignments (K_h) | ~10 | from Murty's k-best LSAP |
| Arc assignments per hole assignment (K_a) | ~5 | after capacity + AP-sep filters |
| Inner runs total | **~50** | each ~25 s with CMA-ES + SLSQP |
| Wall-clock budget | **~20 min** | for a full sweep |

Many arc partitions are infeasible at filter time and don't reach
continuous opt — the layer's cost is dominated by the runs that
survive filtering, not the raw enumeration count.

### Inner level (continuous)

Variables (`num_arcs + 4 × K_probes` total — 30 to 32 for 2-4 arcs ×
7 probes):
- **per-arc** `ap` — one AP angle per arc id, **seeded from the
  middle-layer cluster centroid for that arc**. Probes carrying
  `bind_ap_to_arc=True` inherit it; this is the rig's structural
  coupling baked into the data model.
- **per-probe** `ml_local`, `spin` — angles in degrees. Spin is *not*
  a free degree of freedom for slot-shaped holes: it's tightly
  constrained to align the shank row with the hole's major axis (see
  "4-shank threading" below). Initialise spin to the hole's
  ``theta_rad`` of its bottom (straight-bore) section.
- **per-probe** `entry_offset_(R, A)` — 2D offset within the assigned
  hole, mm. Bounded by the hole oval minus the shaft radius.
- **per-probe** `past_target_mm` — how far past the target centroid
  the tip extends.

Hard pairwise constraints (rig hardware), **convexified by ordering**:

The kinematic constraints `|x_i − x_j| ≥ 16°` are non-convex
(disjunction of half-spaces). Within an ordering, they become convex
chained constraints `x_{σ(i+1)} ≥ x_{σ(i)} + 16°`. The middle layer
already produces an ordering for arcs (by AP centroid); the inner
layer fixes that ordering and treats AP separation as chained:

- `ap_arc_{σ(j+1)} ≥ ap_arc_{σ(j)} + 16°` for `j = 1..num_arcs-1`
  — `num_arcs - 1` chained inequalities (default
  `min_arc_ap_separation_deg` on `PoseLimits`).
- `ml_{σ_a(i+1)} ≥ ml_{σ_a(i)} + 16°` for each arc *a* and each
  consecutive pair within it — at most `4 - 1 = 3` chained
  inequalities per arc (default `min_within_arc_ml_separation_deg`).
  The within-arc ordering ``σ_a`` is determined from the warm-start
  required-ML angles (computed similarly to required-AP).

Convexification halves the ML constraint count and removes a major
source of CMA-ES inefficiency (sampling across the disjunction's
infeasible gap).

These are SLSQP-friendly inequality constraints and become
hard-feasibility filters for CMA-ES. `planning.kinematic_violations`
already implements the check; the optimizer reuses it rather than
reimplementing.

Stages:
1. **CMA-ES** (`cma` library): population-based, derivative-free,
   handles bounds. Starts from broad initial sigma; converges to a
   feasible-and-good region in 50–200 generations on this dimensionality.
2. **SLSQP** (`scipy.optimize.minimize`) for local polish: smooth
   objective, gradient via JAX autodiff, constraint formulation
   `g_i(x) ≥ 0` (clearance min ≥ ε).

### Why this combo?

- CMA-ES is robust to non-smooth, multi-modal landscapes — exactly
  what we expect when probe-hole assignment changes implicitly via
  collision-driven repulsion. It'll find the right basin.
- SLSQP needs a *feasible warm start* and gradients, but inside a
  basin it converges to high precision in handfuls of iterations. Use
  it to refine, not search.
- Skipping CMA-ES and going straight to SLSQP would risk getting
  stuck in the first local minimum the gradient walks to. Skipping
  SLSQP and stopping at CMA-ES would leave us with sub-mm imprecision
  on a sub-mm geometry — not acceptable.

## Threading constraint (the implant-specific feasibility check)

This is the structural innovation that simplifies the rest of the
plan. Each implant bore is extracted to per-hole spec:

```
hole:
  axis_LPS:      [unit vector]      # the bore's own axis
  ref_point_LPS: [3-vec]            # any point on the axis
  sections: [                       # cap planes perpendicular to axis
    {s_mm, center_LPS, a_mm, b_mm, theta_rad},   # top (chamfer)
    {s_mm, center_LPS, a_mm, b_mm, theta_rad},   # mid (straight bore)
    {s_mm, center_LPS, a_mm, b_mm, theta_rad},   # bottom
  ]
```

For a probe whose shaft is parameterised as a line, the threading
constraint at one section is:

1. Intersect the shaft axis line with the section plane → 3D point.
2. Project the point into the section's local 2D frame (basis built
   from `axis`; `theta_rad` rotates the oval major axis from `e1`).
3. Evaluate `g = (u/a)² + (v/b)² − 1`. `g ≤ 0` ⇒ inside the oval ⇒
   shaft passes through this section.

Stack one inequality per (probe, hole-section) pair. With `K` probes,
3 sections per hole, 4 shanks per probe (worst case, 4-shank Neuropixels
2.0), that's `K × 4 × 3 = 12K` threading inequalities. They are smooth
and analytical — gradient-friendly for SLSQP.

**Why this replaces a fixture SDF for the implant.** A voxel SDF of
the implant would tell you "shaft is at distance d from the implant
material; positive in tunnels." Querying it along the shaft would
verify the shaft stays in tunnels. The threading constraint does the
same thing analytically and per-section: it certifies the shaft passes
through *each* hole section's oval, which by construction means it
isn't in implant material. No grid build, no caching, no resolution
tradeoffs — and the constraint has clean derivatives without hitting
voxel-grid artifacts.

### 4-shank threading

For multi-shank probes, every shank must pass each section, not just
the probe's centerline. With 4 shanks at 250 µm pitch (NP 2.0), the
shanks span 750 µm. The build5 slot is 1.20 × 0.70 mm, so the shank
row fits **only along the slot's major axis** — spin is forced to
align ±arccos(750/1200) ≈ ±15° of the slot major axis (modulo 180°).

Implementation: get shank tip positions in local probe frame from
`runtime/shanks.detect_shank_tips_local` (already auto-detected),
transform to world via the probe pose, build one capsule per shank,
evaluate `shaft_section_oval_value` for each shank against each
section, take the worst (maximum). That's the constraint.

## Do we need voxel SDFs at all?

The plan originally treated voxel SDFs as central. With the threading
constraint replacing the implant SDF, the question becomes *what
remaining static colliders need a voxel SDF, and what can use cheaper
analytical approximations?*

| Fixture | Geometry | Recommended representation |
|---|---|---|
| Implant (the body, not the holes) | Plate with bores | **Threading constraint** — no SDF needed. |
| Headframe | Curved, bulky, no holes | Capsule approximation (one or two capsules along its main mass) is probably enough. SDF only if probes routinely come close to surface detail. |
| Well | Cylindrical | Capsule. |
| Probe-guard | Skirt-like, wraps around the probe area | Probably capsule(s). SDF only if it has internal structure that probes might collide with. |
| Brain mesh (over-insertion) | Closed surface | Already handled lazily by `mesh.ray.intersects_location` in the manual mode. Could reuse, no SDF needed. |
| Other probes' shafts/headstages | Capsules per `Capsule(p0, p1, r)` | Capsule-capsule analytical distance — already done in `optimization.geometry`. |

So the v1 optimizer can run **without a single voxel SDF**:
- Threading constraints handle implant interaction.
- Capsule approximations handle headframe / well / probe-guard.
- Capsule-capsule SDF handles probe-probe.
- Brain ray-intersection (already in trame controller) handles
  over-insertion.

If empirical accuracy on a real plan turns out to be insufficient
(e.g. a capsule approximation flags a false collision on the
headframe's curve), *then* upgrade that specific fixture to a voxel
SDF. Build the SDF infrastructure on demand, not preemptively.

## Geometric primitives

All built once at config-load time and cached:

| Primitive | What | Status |
|---|---|---|
| `Capsule(p0, p1, r)` | Probe shaft / shank / headstage / fixture-approximation. Closed-form analytical SDF. | ✅ landed in `optimization/geometry.py` |
| `capsule_capsule_dist` | Two-capsule signed distance via segment-segment + radii. | ✅ landed |
| `HoleSection` + threading | Per-section oval threading constraint + projection math. | ✅ landed |
| `extract_implant_holes.py` | Loads a real implant mesh and emits per-hole YAML with axis, sections, oval params. | ✅ landed (15 holes recovered from build5 implant) |
| `TargetDensity` | per-probe scalar volume of "where to record." Uniform-on-CCF-mask, or weighted by tracer density. Voxel grid + trilinear interp. | ❌ not yet (can start with a Gaussian on the CCF region centroid) |
| `KinematicJacobian` | `∂(p0, p1)/∂(ap, ml, spin, entry, depth)` — closed form via `arc_angles_to_affine` derivative. | ❌ not yet (use JAX autodiff on the existing kinematics) |
| `FixtureSDF` (voxel) | Only built on demand for fixtures whose geometry capsules can't approximate. | ❌ not yet — likely defer indefinitely |

## Objective function

Lexicographic, but smoothly stitched via penalties for tractability.

```
J(x) = -coverage(x)
     + λ_feas · max(0, -min_clearance_shaft_hole(x))²
     + λ_feas · max(0, -min_clearance_headstage(x))²
     - λ_margin · softmin(clearances(x), β)
```

- `coverage(x)`: sum over probes of `∫₀ᴸ density_i(p_i(s)) ds`,
  evaluated by sampling along the shaft.
- `min_clearance_shaft_hole(x)`: with the threading constraint,
  this becomes `min over sections of -g(x, section)` (negative of the
  worst oval value). Positive when every section is satisfied.
- `min_clearance_headstage(x)`: min over `(i, j)` pairs of capsule-capsule
  signed distance.
- `softmin(...)`: `-β · log(Σ exp(-d_k / β))` — smooth approximation
  of the discrete `min`. Important for SLSQP because plain `min` has
  a kink and breaks gradients.
- **Homotopy schedule**: `λ_feas` ramps up over CMA-ES generations.
  Early generations explore broadly with mild penalties (so infeasible
  candidates still inform the search direction); later generations
  enforce strict feasibility. Standard trick to avoid the population
  collapsing prematurely onto an artefactual local minimum.

## Library plan

- **JAX** for the differentiable inner loop (kinematic chain → capsule
  positions → density lookups → coverage, clearances). Trilinear
  interpolation is `jax.scipy.ndimage.map_coordinates` or a hand-rolled
  8-tap.
- **`cma`** for CMA-ES (`pip install cma` — pure Python, ~5k lines).
- **`scipy.optimize.minimize(method="SLSQP")`** for local polish with
  constraints; jacobian provided via `jax.jacrev`.
- **`scipy.optimize.linear_sum_assignment`** for outer LSAP. Murty
  variant either rolled by hand (the standard recursive partitioning
  is short) or `lap` package.
- **No `mesh_to_sdf` / libigl** for v1 — see "Do we need voxel SDFs"
  above.

## Where the code goes

```
src/aind_low_point/
└── optimization/
    ├── __init__.py
    ├── geometry.py    ✅ Capsule, capsule-capsule SDF, threading constraint
    ├── holes.py       ❌ load extracted YAML into HoleSection lists
    ├── density.py     ❌ TargetDensity + CCF-mask helpers
    ├── kinematics.py  ❌ JAX-friendly probe pose Jacobian
    ├── objective.py   ❌ coverage / clearance / penalties (JAX)
    ├── hole_assignment.py  ❌ outer LSAP + Murty (probe→hole)
    ├── arc_assignment.py   ❌ middle layer (probe→arc clustering on
    │                          required-AP, capacity + AP-sep filters)
    └── optimize.py    ❌ three-level driver (CMA-ES → SLSQP per
                           hole×arc pair, pick global best)
```

Tests:
- Unit tests for each primitive (capsule SDF closed-form vs.
  brute-force numerical, kinematic Jacobian vs. finite-diff).
- One small end-to-end smoke test on a synthetic 2-probe / 4-hole
  problem with known optimum.

## Open questions to resolve before going further

1. **Headstage geometry.** Single capsule, two stacked capsules, or a
   small convex hull? The current package has the probe meshes loaded
   as a single object — need to either split shaft from headstage in
   the asset or model the headstage as a separate collidable spec
   attached to the same probe.
2. **Fixture clearance representation in practice.** Does a single
   capsule for the headframe actually cover the relevant collision
   surface? Empirical check needed before committing.
3. **Density representation pipeline.** Where does each probe's
   `TargetDensity` come from? Likely a CCF region mask warped into
   the working frame, possibly weighted by a tracer density volume.
   Need a small builder that takes a CCF acronym + (optional) density
   nrrd and returns a `TargetDensity`.
4. **Which frame the optimizer runs in.** Internal canonical is
   LPS-mm; anything imported (CCF density, fixture mesh) needs to be
   in that frame first. The existing canonicalization pipeline
   handles meshes; needs to apply to density volumes too.
5. **Required-AP from hole axis.** The middle-layer cluster key needs
   the rig's AP/superior axis directions in the same frame as the
   extracted hole axes. The arc rotation plane is rig-specific —
   should come from rig config, not be hardcoded. Verify via the
   manual-mode kinematics what direction `arc_angles_to_affine`
   actually rotates around.
6. **`num_arcs` as a config field vs. enumerated choice.** Treating
   `num_arcs` as fixed by the rig is the simple path and matches
   physical reality (the user instantiates a fixed set of arcs). If
   later we want to optimize across "use 2 arcs vs. 3 vs. 4," that's
   another outer enumeration layer — defer.
7. **AP span feasibility check at config load.** With ``num_arcs = 4``
   and 16° pairwise minimum, total AP span ≥ 48°. If the rig's AP
   travel is narrower than `(num_arcs - 1) × 16°`, the configuration
   is unworkable before any optimization runs. Surface as a clean
   error at config load, not deep in CMA-ES.

## Out of scope (for v1)

- Calibration uncertainty → robust optimization (mean ± noise).
- Multi-day staged insertion plans.
- Time-of-day / temperature drift compensation.
- Probe re-use across days.
- Voxel SDFs for any fixture (defer until empirically necessary).

## Picking this up later

Sequence to follow (updated to reflect what's already done):

1. ✅ **`geometry.py`** — capsule + capsule-capsule + threading.
2. ✅ **Hole extraction tool** — extracts per-hole specs from the
   real implant mesh.
3. ⏭ **Feasibility-map sanity check** — sweep `(offset, offset, spin)`
   for one probe through one extracted hole, plot threading constraint
   values, verify the feasible region is a smooth connected blob with
   the predicted ±15° spin tolerance. *Lightweight: ~50 LOC, no
   kinematic adapter needed.*
4. **`holes.py`** — small loader that turns the YAML into a list of
   `Hole` objects with `HoleSection` lists.
5. **`kinematics.py`** — JAX-compatible probe pose. Verify autodiff
   Jacobian matches `arc_angles_to_affine` finite-difference at a
   few points. Also exposes a `required_ap(hole_axis)` helper for the
   middle layer.
6. **`density.py`** — Gaussian-on-centroid baseline; keep the
   interface `density(p_LPS) → scalar` so we can swap in a voxel
   density later.
7. **`objective.py`** — assemble coverage + threading + capsule
   clearances. Visualize the loss landscape on a 2D slice (vary one
   probe's AP × ML, hold the rest fixed).
8. **`hole_assignment.py`** — heuristic feasibility cost + LSAP + Murty,
   producing top-K_h ranked probe→hole assignments. Includes the
   "can required APs be partitioned into ≤num_arcs feasible groups?"
   coarse check.
9. **`arc_assignment.py`** — given a hole assignment, compute
   required-AP per probe, cluster into num_arcs labelled groups,
   filter by capacity + AP-sep, return top-K_a partitions ranked by
   within-cluster variance.
10. **`optimize.py`** — three-level driver: for each hole assignment,
    for each arc assignment, run CMA-ES → SLSQP polish; return the
    best result with its (hole, arc) labels and the continuous values.
11. **Plumb arc reassignment into the data model.** A small
    `ReassignProbeToArc` command + handler so the optimizer's chosen
    `arc_id` per probe can be applied via the existing dispatch path.
12. **Hook into the runtime** — add a "Run optimizer" button to
    `TrameController` that takes the current `PlanningState`, ignores
    its `arc_id` assignments, runs the optimizer, returns a new
    `PlanningState` to apply (with possibly different `arc_id` values
    per probe).

Before any of that beyond step 3: the **feasibility-map sanity check**
is the cheapest way to validate the strategy with what we have.
