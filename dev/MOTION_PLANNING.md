# Multi-Probe Manipulator Motion Planning — design sketch

**Status:** design exploration, **not implemented**. This is a scoping document
so a future agent can reason about a solution. No code exists yet.

**One-line framing:** the probe manipulators can now be driven *automatically and
safely* because `aind-rutter` finally holds the scene knowledge (meshes,
poses, angles, a collision oracle, and a differentiable clearance field) that a
motion planner needs. The task is a **multi-robot motion-planning** problem, and
most of the hard primitives already exist in this repo.

---

## 1. The physical problem

- Each Neuropixels-style **probe sits on its own 3-DOF Cartesian (XYZ)
  manipulator**. The probe's **orientation is fixed** (set by its arc/manipulator
  angle); the stage only **translates** it. So each probe's configuration space
  is ℝ³ (its manipulator coordinates), and the whole rig is a **3·N-dimensional
  composite C-space** for N probes (today N ≈ 7–8).
- The probes share a tight workspace and can collide with **each other, the
  implant, the headstages/fixtures, the well, and the brain**.
- **Shanks are thin and fragile** — any shank contact breaks the probe. This is a
  safety-critical, low-tolerance-for-error setting. Conservative clearance and
  verified (not best-effort) paths matter more than optimality or speed.
- **Goal motions** (both directions):
  - **Disperse:** move all probes out of each other's way (e.g., to stow/retract
    for access or removal).
  - **Converge:** move probes from a parked/stowed state to their **planned final
    poses** (the poses this package already computes).
  - In both cases: either find **collision-free paths** for simultaneous motion,
    or **schedule/sequence** the moves so no two collide at any instant.
- **Control interface:** manipulators are commanded over **network sockets**
  (protocol details TBD — see Open Questions). Planning and execution must be
  cleanly separated so the planner can be dry-run/simulated before any motor moves.

---

## 2. What this project already provides (the key enabler)

A motion planner is mostly a search loop that repeatedly asks *"is this
configuration / this motion collision-free?"* and a forward map from actuator
coordinates to placed geometry. `aind-rutter` already has both, plus a
differentiable clearance field that makes trajectory *optimization* feasible too.

**Coordinate convention (must obey):** the internal canonical frame is **LPS
millimeters**. RAS appears only at named user-facing boundaries and is converted
at the planning boundary. See `dev/COORDINATES.md`. A planner should work
entirely in canonical LPS mm.

**Forward kinematics (actuator coords → placed meshes):**
- `planning.py` — `Kinematics`, `ProbePose`, `PoseResolver`, `PlanningState`,
  `PoseLimits` (`ML_LIMIT_DEG = 42`, `AP_LIMIT_DEG = 75`; single source shared by
  the UI *and* the optimizer). This is the runtime/UI-facing kinematics.
- `optimization/geometry/kinematics.py` and `optimization/sdf/kernels.py` —
  `pose_from_optimizer_vars`, `arc_angles_to_rotation`, `shank_capsules_from_pose`
  (the optimizer-side kinematics; verified to agree with the UI/app kinematics to
  ~1e-7, CI-guarded in `tests/test_export_kinematics_parity.py`).
- `core.py` — `AffineTransform`, `TransformChain`, `Transformable`; `scene.py` —
  `Scene`, `NodeInstance`. The placed-mesh world state lives here.

**Collision oracle (the validity check a planner calls):**
- `collisions.py` — `CollisionAdapter`, `CollisionHandler` (sync + async).
- `fcl_backend.py` — `FCLBackend`: exact mesh collision/distance via `python-fcl`,
  with per-pair callbacks and group/mask filtering (so "ignore probe-vs-its-own-
  fixture" style rules are already expressible). **This is the ground-truth
  collision/clearance query** and is the natural state/edge validity primitive.
- `optimization/objectives/fcl_validator.py` — returns graded positive clearance
  or a flat −1.0 collision **sentinel** (no graded penetration). Good for
  pass/fail gating; not for gradient.

**Differentiable clearance field (enables trajectory optimization):**
- `optimization/sdf/` — `build.py`, `kernels.py` (`trilinear_sdf`),
  `clearance_sweep.py`, `envelope.py`. The whole placement optimizer is built on
  JAX SDF clearance kernels that are **differentiable, batched (vmap), and GPU-
  accelerated**. A CHOMP-style trajectory optimizer (§7) can reuse these directly.
- `optimization/objectives/` — `reduced_jax.py`, `phase1.py`, `phase2.py`,
  `batched_static.py`, `spin_restore.py`: examples of building/optimizing against
  these SDFs at scale (per-pair capsule clearance, OBB-SAT, swept volumes for spin
  coupling — the swept-volume code is relevant to continuous-collision checking).

**Established solver pattern to imitate:** the placement pipeline runs
**soft/differentiable optimization (Phase-1 SDF) → polish (Phase-2 IPOPT) → FCL
ground-truth gate (Phase-3)**. A trajectory planner should mirror this: plan/
optimize against the fast SDF, then **validate the final trajectory with FCL**
before anything is sent to hardware. See `dev/PIPELINE.md`.

**Visualization for dry-runs:** `rendering.py` (`RendererAdapter`,
`RenderBackend`), `k3d_backend.py`, `pyvista_backend.py`. A planned trajectory can
be animated/inspected in the existing viewers before execution.

---

## 3. Algorithm families (the solution space)

This is **multi-robot motion planning**, which factors into two subproblems.

**(a) Single-probe path planning** — each probe is only 3-DOF, so this part is
*easy* and many methods work:
- **Sampling-based:** PRM (build a reusable roadmap of free workspace once,
  multi-query), RRT / RRT-Connect / RRT* / BIT* (single-query). Mature, off-the-
  shelf (e.g., OMPL).
- **Grid/graph search:** voxelize the 3D workspace, run A*/Dijkstra. Complete,
  deterministic, *inspectable* — attractive for a safety-critical system because
  paths are repeatable and auditable (unlike randomized RRT).
- **Trajectory optimization:** CHOMP / STOMP / TrajOpt / GPMP — optimize a smooth,
  clearance-maximizing trajectory against an SDF cost. **Best fit here** (§7).

**(b) Multi-probe coordination** — the genuinely hard part:
- **Coupled/centralized:** plan in the full 3·N-D space. Complete but scales
  poorly; overkill at N≈7.
- **Decoupled + prioritized planning:** order the probes; plan each one avoiding
  the already-planned higher-priority probes (treated as moving obstacles).
  Simple, fast, usually sufficient.
- **Velocity tuning / coordination diagrams:** fix each probe's *path*, then only
  adjust *timing* along those fixed paths to avoid collisions — turns geometry
  into scheduling.
- **MAPF:** Conflict-Based Search (CBS, the well-known optimal one), Priority-
  Based Search (PBS), M* — plan individually, detect conflicts, add constraints,
  replan.
- **Reactive (ORCA/velocity obstacles):** *avoid* for fragile hardware — you want
  a verified plan up front, not reactive avoidance.

Related framings: **assembly/disassembly sequencing** (what order to insert/
extract) and **Task-and-Motion Planning (TAMP)** if move *order* interacts with
geometric feasibility (it does here — see §6).

---

## 4. Why this instance is favorable

1. **Fixed insertion axes** ⇒ a natural safe primitive: *retract each probe along
   its own axis to a parked height → translate laterally → re-insert.* This
   **decomposes the problem into phases** (§6), and within a phase moves are
   near-1-D. Phasing + a simple schedule sidesteps most of the hard coupling. The
   boring, safe answer is often "fixed retract/insert paths + a collision-aware
   schedule of when each probe moves" — no joint planner required.
2. **Low per-robot dimension (3-DOF)** ⇒ even brute-force grid search is tractable.
3. **The collision oracle, kinematics, and a differentiable SDF clearance field
   already exist** (§2) — the planner is "glue + search," not new geometry.

---

## 5. Proposed architecture (how a solution coexists with this package)

Mirror the existing **core-vs-backend / adapter** split (planning produces
artifacts; a hardware backend executes them). Suggested new subpackages:

```
optimization/...            # EXISTING: kinematics, SDF, FCL, clearance  ← reuse
motion/                     # NEW planning core — pure, UI-independent, testable
  ├── cspace.py             # manipulator XYZ ⇄ ProbePose ⇄ placed meshes
  │                         #   (thin wrapper over planning.Kinematics + scene)
  ├── validity.py           # state/edge collision via FCLBackend + SDF margins;
  │                         #   continuous (swept) checking, not just waypoints
  ├── planner.py            # single-probe: PRM / RRT-Connect / grid-A*  (or CHOMP)
  ├── coordinate.py         # multi-probe: prioritized planning / schedule / CBS
  └── trajectory.py         # Trajectory dataclass: per-probe timed waypoints
drivers/manipulator.py      # NEW hardware executor — the ONLY socket-touching code
```

- **Planning core (`motion/`)** depends only on the kinematics + collision +
  (optionally) SDF layers. It emits a **verified `Trajectory`** and touches no
  hardware. Fully unit-testable and dry-runnable in the existing renderer.
- **Executor (`drivers/manipulator.py`)** consumes a `Trajectory`, streams
  waypoints over sockets, enforces feed-rate limits, **synchronization barriers
  between phases** (or a coordinated schedule), and an **abort/e-stop** path. Keep
  it as isolated from planning as `fcl_backend` is from `collisions`.
- **Goal source:** "move to planned poses" = "goal config = the `ProbePose` this
  package already computed" (from `PlanningState` / emitted plan YAML). No new
  goal representation needed.

**Data flow:** `PlanningState` (current + target poses) → `cspace` maps to
manipulator coords → `planner`/`coordinate` search using `validity` (FCL/SDF) →
`Trajectory` → FCL re-validation gate → (dry-run render) → `drivers` execution.

---

## 6. The phased decomposition (recommended backbone)

Because orientations are fixed, decompose any disperse/converge motion into:

1. **Retract** every probe along its own insertion axis to a collision-free
   "parked" band (above/clear of the brain and of each other). Retraction along a
   fixed axis is nearly collision-free by construction if parked heights are
   chosen so axes don't intersect in the parked zone.
2. **Reposition** laterally in the parked band (this is where the real
   coordination/scheduling happens — but in a safer, shank-clear region).
3. **Insert** along axes to final poses, in a collision-safe order.

This converts a scary 21-D joint problem into a sequence of well-separated, mostly
low-D moves with clear safety invariants. Sequencing the order of retract/insert
is the assembly-sequencing subproblem; prioritized planning or a coordination
schedule handles the lateral phase.

---

## 7. Trajectory-optimization tie-in (the strongest fit)

CHOMP/TrajOpt is "gradient-descend a time-discretized trajectory against an SDF
collision cost + a smoothness term." **This package already built exactly that SDF
machinery** for *pose* optimization. So a manipulator trajectory optimizer can:

- represent each probe trajectory as waypoints `[x₀ … x_T]` in manipulator coords,
- map to swept shank/body geometry via the existing kinematics,
- score with the existing **JAX SDF clearance kernels** (`optimization/sdf`,
  `trilinear_sdf`, the per-pair capsule/OBB clearance in
  `optimization/objectives/reduced_jax.py` / `batched_static.py`),
- add a smoothness/path-length term and (soft) min-clearance margin,
- **vmap across probes**, optimize, then **FCL-gate** the result —
  the same soft-SDF → polish → FCL-ground-truth pattern as the placement pipeline.

This reuses the most expensive, already-tuned part of the codebase and yields
smooth, clearance-maximizing, *verifiable* trajectories.

---

## 8. Safety requirements (because shanks break)

- **Continuous collision checking**, not just waypoint sampling — check swept
  volumes or finely subdivide edges. FCL has continuous-collision queries, and the
  spin-coupling code already does swept-volume intersection.
- **Inflated margins** — reuse the existing α-wrap / `offset_mm` / envelope
  machinery (`optimization/sdf/envelope.py`); treat the **shank** as the protected
  entity with extra clearance. Note the documented α-wrap precision floor — FCL is
  the ground-truth check, SDF is the fast guide.
- **Deterministic, inspectable paths** preferred over raw randomized RRT for
  hardware that breaks. Prefer grid-A* / fixed PRM roadmap / optimize-then-verify,
  and make the plan reviewable and replayable identically.
- **FCL is the final gate.** Never send a trajectory to hardware that hasn't
  passed FCL ground-truth validation end-to-end (mirror Phase-3).
- **Execution safety:** sequential moves (one probe at a time) is the simplest,
  safest baseline; coordinate concurrency only with a verified schedule. Always
  wire an abort path in the executor.

---

## 9. Minimal starting interfaces (illustrative, not prescriptive)

```python
# motion/trajectory.py
@dataclass(frozen=True, slots=True)
class Waypoint:
    probe: str
    xyz_mm: tuple[float, float, float]  # manipulator coords, canonical LPS mm
    t: float  # seconds from motion start


@dataclass(frozen=True, slots=True)
class Trajectory:
    waypoints: tuple[Waypoint, ...]  # per-probe, time-ordered
    # invariant: FCL-validated collision-free over the full swept motion


# motion/cspace.py
def manipulator_to_pose(probe: str, xyz_mm, state: PlanningState) -> ProbePose: ...
def placed_meshes(poses: Mapping[str, ProbePose]) -> Scene: ...  # for FCL


# motion/validity.py
def edge_is_clear(
    probe, a_xyz, b_xyz, others: Mapping[str, ProbePose], margin_mm: float
) -> bool: ...  # swept/continuous FCL check
```

The executor (`drivers/manipulator.py`) is intentionally *out of scope* for the
planning core: it consumes a `Trajectory` and is the only module aware of the
socket protocol.

---

## 10. Open questions for the next agent

- **Manipulator socket protocol**: command/feedback format, units, coordinate
  frame the hardware expects, whether it reports live position, latency, and
  whether it supports synchronized multi-axis moves or only per-axis. (Unknown —
  must be obtained before the executor can be written.)
- **Where is "current" manipulator state?** Does `PlanningState` already hold live
  manipulator coordinates, or only target poses? The planner needs a verified
  *start* config, not just the goal.
- **Parked-zone geometry**: define collision-free retract heights per probe such
  that axes don't intersect in the parked band (informs §6).
- **Coordination policy**: is sequential (one-at-a-time) acceptable for throughput,
  or is concurrent motion required? That decides whether you need scheduling/CBS or
  just an ordering.
- **Manipulator ↔ probe-pose mapping**: confirm the exact transform from stage XYZ
  to `ProbePose` (the manipulator's mounting/zero relative to the arc geometry).
  The optimizer-vs-app kinematics parity test is the model to extend.

---

## 11. Pointers

- Scene/kinematics/collision: `planning.py`, `core.py`, `scene.py`,
  `collisions.py`, `fcl_backend.py`.
- Differentiable clearance / SDF: `optimization/sdf/` (`build`, `kernels`,
  `clearance_sweep`, `envelope`), `optimization/objectives/` (`reduced_jax`,
  `phase1`, `phase2`, `fcl_validator`, `batched_static`, `spin_restore`).
- Coordinate convention: `dev/COORDINATES.md`. Solver pattern (soft → polish →
  FCL gate): `dev/PIPELINE.md`. Architecture tour: `dev/CORE_CONCEPTS.md`,
  `dev/MODULE_MAP.md`.
- Suggested external libs to evaluate: **OMPL** (sampling-based planning), or a
  JAX/CHOMP-style optimizer built on the existing SDF kernels.

**Summary for a fresh reader:** it's multi-robot motion planning; each probe is
3-DOF with a fixed angle; this repo already supplies forward kinematics, an FCL
collision oracle, and a differentiable JAX SDF clearance field. The natural
solution is a **phased retract → reposition → insert** plan, with single-probe
paths from sampling-based/grid search **or** a CHOMP-style trajectory optimizer
reusing the existing SDF kernels, coordinated across probes by prioritized
planning/scheduling, **FCL-gated** before execution, with a socket **executor**
split off as an isolated hardware backend.
