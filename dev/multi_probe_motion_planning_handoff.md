# Multi-Probe Manipulator Motion Planning Handoff

**Audience:** execution / implementation team  
**Status:** design recommendation, not yet implemented  
**Scope:** autonomous safe motion planning and execution for multi-probe Neuropixels manipulator rigs  
**Primary recommendation:** build a mode-specific, phase-based planner that always produces a verified serial-safe plan, then optionally compresses that plan into validated concurrent batches.

---

## 1. Executive summary

This system should be treated as a safety-critical multi-robot motion-planning problem with unusually favorable structure:

- Each probe/manipulator is effectively a fixed-orientation 3-DOF translating robot.
- The full rig is a 3N-dimensional composite configuration space, with expected N around 12-15 probes.
- Shanks are fragile, so verified clearance matters more than optimality or speed.
- The existing codebase already has the most important building blocks: probe poses, manipulator kinematics, scene meshes, FCL collision checking, differentiable JAX/SDF clearance, and rendering.

The best first implementation is **not** a centralized 36-45 DOF multi-robot planner. Instead:

1. Generate simple, mechanically interpretable per-probe paths.
2. Search over probe order using assembly/disassembly sequencing.
3. Validate every state and edge conservatively with FCL.
4. Execute serially by default.
5. Add simultaneous motion only as a post-processing scheduling layer over already-valid paths.

The autonomous insertion target should be **hover**, not final brain insertion. The planner should move each probe to approximately **3 mm above the brain surface along its planned insertion axis**, then park for manual or supervised final insertion.

---

## 2. Updated operating assumptions

These assumptions incorporate the latest product/experiment constraints.

### Probe count

Expected maximum probe count is approximately **12-15 probes**.

This is still small enough for exact or near-exact subset search over probe insertion/retraction order. For 15 probes, a full bitmask state space has:

```text
2^15 = 32768 subset states
15 * 2^14 = 245760 possible remove-one/add-one transitions before pruning
```

That is feasible if collision checks are cached and generated lazily.

### Motion concurrency

Serial movement is acceptable, but simultaneous movement is preferred where safe.

Design implication:

- The planner must always be able to emit a valid **serial fallback**.
- Concurrent execution should be treated as optional schedule compression, not as the foundation of safety.

### Manipulator command interface

Likely interface:

- Send **absolute XYZ setpoints**.
- Receive online streamed position readback.
- Commands are issued over network sockets.

Design implication:

- Planning and execution must be strictly separated.
- The planner emits a validated timed trajectory artifact.
- The socket driver only consumes validated artifacts.
- The execution monitor enforces readback tracking tubes, phase barriers, and abort conditions.

Important unresolved controller detail:

- Does firmware move in a straight line between absolute XYZ setpoints?
- If not guaranteed, the driver must stream small validated micro-waypoints so actual motion stays inside the certified safety tube.

### Insertion policy

Fully automatic final insertion into the brain is not the default.

Recommended autonomous target:

```text
hover_pose_i = final_insertion_pose_i retracted along probe axis
               until the tip is 3 mm above the brain surface
```

Final insertion can be handled by a supervised/manual `INSERT_ASSIST` mode.

### Goal validation

Final planned insertion poses are already FCL-validated and physically validated before insertion.

Design implication:

- The motion planner can assume target geometries are plausible.
- It still must validate the path to the autonomous hover target.
- It should fail clearly if the target or hover geometry is invalid under motion-planning margins.

### Axial retraction

Pure axial retraction is often safe, but not guaranteed.

Design implication:

- Axial retraction should be tried first.
- It must be validated per plan.
- Edge cases should produce a structured hazard report.
- Automatic escape maneuvers should be optional and conservative.

### Parked/stowed pose definition

The canonical parked/stowed position should be the insertion position, axially retracted.

Definitions:

```text
final_i = planned final insertion pose
hover_i = final_i retracted along insertion axis to 3 mm above brain surface
park_i  = hover_i retracted farther along the same axis
```

This avoids assignment complexity: each probe's parked pose belongs to its own planned tract.

### Automation level

Fully automated execution is acceptable once validation passes.

Design implication:

- The validation certificate is the safety boundary.
- No certificate, no hardware execution.
- Dry-run rendering should be available before execution, but operator approval does not replace validation.

### Optimization objective

Optimize for a balance of:

1. Clearance / risk reduction.
2. Total travel time.
3. Path length and smoothness.
4. Determinism and auditability.

Safety ordering should be:

```text
validity > clearance > deterministic/auditable behavior > time
```

### Additional motion mode

The scattered/parked configuration should support a **calibration sweep** mode:

- Select one probe.
- Move it around inside a fixed validated volume.
- Use observations to calibrate photogrammetry / closed-loop position feedback.
- Keep all other probes parked.

---

## 3. Recommended motion modes

Implement four first-class motion modes.

### 3.1 `DISPERSE`

Purpose:

Move probes from current or near-target configurations into safe retracted/scattered states.

Typical use cases:

- Clearing the workspace.
- Preparing for access/removal.
- Moving from hover/insertion-side arrangements into parked states.

Core behavior:

```text
current_i -> park_i
```

Each probe's path is validated against stationary probes, fixture geometry, implant/well geometry, and forbidden brain geometry.

### 3.2 `CONVERGE_TO_HOVER`

Purpose:

Move probes from parked/scattered states to their planned insertion axes, stopping at hover.

Autonomous target:

```text
tip_i is approximately 3 mm above the brain surface along the planned insertion axis
```

Core behavior:

```text
park_i -> hover_i
```

This should be the main autonomous pre-insertion mode.

### 3.3 `INSERT_ASSIST`

Purpose:

Provide supervised/manual assistance from hover to final insertion depth.

Default policy:

- Not fully automatic.
- Permit microsteps, readback monitoring, and guardrails.
- Use planned final poses and physical validation as reference.

Potential behavior:

```text
hover_i -> final_i
```

but only under an explicit insertion policy.

### 3.4 `CALIBRATION_SWEEP`

Purpose:

Move one selected probe through a validated 3D volume to calibrate photogrammetry and closed-loop feedback.

Preconditions:

```text
all non-active probes are parked
active probe has a declared calibration volume
calibration volume has been validated with margins
brain/implant/fixture keep-out zones are respected
```

Output:

A dataset aligning commanded positions, manipulator readback, photogrammetry estimates, timestamps, and residuals.

---

## 4. Core algorithm

Use this algorithmic stack:

```text
phase-based assembly/disassembly sequencing
+ deterministic single-probe path generation
+ FCL-certified state and edge validation
+ optional JAX/SDF trajectory optimization
+ optional concurrent batch scheduling
```

### 4.1 Why not start with a centralized planner?

A centralized planner over all probe positions would operate in 36-45 dimensions for 12-15 probes. It may work in theory, but it has poor engineering properties for this system:

- Hard to debug.
- Hard to explain to operators.
- Hard to validate conservatively.
- Likely to exploit tiny clearances.
- More complex than necessary because the probe orientations are fixed.

The fixed insertion axes give us a much better structure: retract, park, move, hover, insert-assist.

### 4.2 Phase decomposition

For most motion requests, decompose into three physical phases:

```text
1. Retract / disassemble
2. Reposition while parked
3. Converge / assemble to hover
```

This turns a high-dimensional multi-robot problem into mostly single-probe path validation plus sequence search.

### 4.3 Reverse-disassembly sequencing

For `CONVERGE_TO_HOVER`, compute insertion order by solving the reverse problem:

```text
Can the final hover arrangement be safely disassembled back to parked poses?
```

If yes, execute the reverse of that disassembly sequence as the converge-to-hover plan.

Why this works:

- Retraction from hover to park is easier to test than insertion into a crowded scene.
- The reverse path is valid under the same geometry if the path and margins are symmetric.
- The resulting sequence is interpretable and easy to certify.

State representation:

```text
S = set of probes already removed / parked
```

Transition:

```text
Try retracting probe i not in S while:
    probes in S are at park_i
    probes not in S and not i remain at hover_i
```

Search:

```text
start: S = empty set
 goal: S = all probes
```

For `CONVERGE_TO_HOVER`, execute the found path in reverse.

### 4.4 Sequence-search objective

Use Dijkstra or A* over subset states.

Transition cost:

```text
edge_cost =
    duration_seconds
  + lambda_risk * clearance_penalty(min_clearance_mm)
  + lambda_motion * path_length_mm
  + lambda_complexity * path_complexity_penalty
```

Recommended clearance penalty:

```text
clearance_penalty(d) = softplus(required_margin - d)^2
```

Use hard rejection when:

```text
d < required_margin
```

Useful tie-breakers:

- Prefer pure axial paths over doglegs.
- Prefer larger minimum clearance.
- Prefer shorter total duration.
- Prefer deterministic/repeatable path families.

### 4.5 Caching

Cache validity checks aggressively.

Suggested cache keys:

```text
state_validity_cache[(scene_hash, config_hash, margins_hash)]
edge_validity_cache[(scene_hash, q0_hash, q1_hash, moving_probe_set, margins_hash)]
pair_clearance_cache[(scene_hash, probe_i, pose_i_hash, object_j, pose_j_hash)]
```

The subset search becomes much faster if repeated axial moves and pair checks are reused.

---

## 5. Single-probe path generation

Each probe has only 3 translational degrees of freedom, so single-probe planning should be simple and deterministic.

Use a three-tier planner.

### Tier 1: templates

Try mechanically interpretable paths first.

Examples:

```text
axial retract
axial descend to hover
straight-line parked-band translation
dogleg: retract -> translate -> descend
```

Most normal moves should succeed with these templates.

### Tier 2: deterministic lattice A*

If a template segment fails, search in the active probe's 3D manipulator space.

Recommended properties:

- Bounded local search region.
- Deterministic grid resolution.
- Weighted A*.
- FCL edge validity for candidate edges.
- SDF clearance as a heuristic or soft cost.

Edge cost:

```text
cost(q -> q_prime) =
    nominal_time(q, q_prime)
  + lambda_clearance * soft_clearance_penalty(q, q_prime)
  + lambda_turn * direction_change_penalty
```

Use cases:

- Parked-band repositioning.
- Edge cases where pure axial retraction is blocked.
- Recovery from non-canonical current states.

### Tier 3: trajectory optimization

Use trajectory optimization to smooth and improve clearance after a feasible path exists.

Do not use optimization as the sole source of safety.

Candidate objective:

```text
J =
    w_time      * estimated_duration
  + w_length    * path_length
  + w_smooth    * squared_acceleration
  + w_clearance * sum(softplus(required_margin - clearance)^2)
  + w_min_clear * max(0, desired_margin - min_clearance)^2
```

The existing JAX/SDF machinery is a good fit for this layer because it already supports differentiable, batched clearance evaluation.

Final optimized trajectories must still pass the FCL validation gate.

---

## 6. Multi-probe coordination

### 6.1 Baseline: serial execution

Always produce a serial-safe plan.

```text
move probe A
barrier
move probe B
barrier
move probe C
barrier
...
```

Advantages:

- Simplest validation.
- Easiest to debug.
- Easiest to explain.
- Best fallback when concurrency fails.

### 6.2 Concurrent batch compression

After a serial plan exists, try to compress independent segments into simultaneous batches.

Build a conflict graph:

```text
vertex = one planned probe segment
edge   = these two segments cannot safely run at the same time
```

Two segments conflict if their simultaneous execution fails continuous validation with margins.

Then greedily or optimally color the graph into batches:

```text
batch 1: probes 1, 4, 9 move together
barrier
batch 2: probes 2, 5 move together
barrier
batch 3: probes 3, 6, 7 move together
```

This gives useful concurrency without requiring a full high-dimensional coupled planner.

### 6.3 Fixed-path timing search

If batch concurrency is not enough, keep paths fixed and optimize timing only.

Each probe gets a monotone path coordinate:

```text
s_i(t) in [0, 1]
```

The scheduler may insert waits or slowdowns, but should not alter geometry unless it triggers a new validation cycle.

Recommended later-stage algorithms:

- Priority-Based Search (PBS) over motion priorities.
- Conflict-Based Search (CBS) over pairwise trajectory conflicts.
- Coordination diagrams for fixed path timing.

Do not use reactive velocity-obstacle behavior as the main safety mechanism.

---

## 7. Calibration-sweep mode

The calibration sweep should be a first-class mode, not manual jogging.

### 7.1 Request object

Suggested request schema:

```python
@dataclass(frozen=True)
class CalibrationSweepRequest:
    active_probe_id: str
    volume_geometry: VolumeSpec  # AABB, convex polytope, or mesh volume
    sample_spacing_mm: float
    max_points: int
    dwell_time_s: float
    max_speed_mm_s: float
    required_clearance_mm: float
    return_to_park: bool = True
```

### 7.2 Preconditions

```text
all non-active probes are parked
active probe starts from a valid pose
calibration volume is inside manipulator limits
calibration volume avoids forbidden geometry with margins
photogrammetry visibility constraints are satisfied, if available
```

### 7.3 Path generation

Recommended algorithm:

```text
1. Sample candidate points inside the calibration volume.
   Use grid, Sobol/Halton sequence, or corners plus interior lattice.

2. Reject invalid waypoints and invalid edges.

3. Choose a waypoint order.
   For grid volumes, use a lawnmower pattern.
   For irregular volumes, use nearest-neighbor plus 2-opt.

4. Add dwell times at observation points.

5. Return active probe to park if requested.
```

### 7.4 Logged observations

Suggested observation schema:

```python
@dataclass(frozen=True)
class CalibrationObservation:
    timestamp_s: float
    probe_id: str
    commanded_xyz_lps_mm: tuple[float, float, float]
    readback_xyz_lps_mm: tuple[float, float, float]
    predicted_pose_lps: ProbePose
    photogrammetry_pose_lps: ProbePose | None
    image_ids: tuple[str, ...]
    residual_after_fit_mm: float | None
```

### 7.5 Closed-loop rule

Photogrammetry may correct motion only inside a prevalidated safety tube.

If feedback indicates the probe has left the tube:

```text
stop / hold position
freeze the plan
log deviation
require a validated recovery plan
```

The feedback system should not invent new collision-untested geometry online.

---

## 8. Validation and safety model

### 8.1 Ground-truth validator

FCL should be the final ground-truth collision and distance validator.

Use the existing collision stack:

```text
collisions.py
fcl_backend.py
optimization/objectives/fcl_validator.py
```

SDF/JAX clearance is for fast planning, optimization, and heuristics. FCL is the final gate.

### 8.2 State validity

A state is valid if:

```text
all active and stationary probe geometries are placed correctly
no forbidden collision exists
all required pairwise clearances exceed margins
manipulator limits are respected
brain/implant/well keep-out rules are respected
```

### 8.3 Edge validity

Waypoint-only checking is not enough.

Each edge must be validated continuously or conservatively.

Implement two edge validators:

```text
FCL_CCD_EDGE_VALIDATOR
    Use FCL continuous collision detection when reliable for the relevant pair.

CONSERVATIVE_TRANSLATION_EDGE_VALIDATOR
    Recursively subdivide the edge.
    At each interval, compare sampled FCL clearance against:
        safety_margin
      + calibration_error
      + tracking_error
      + mesh/model_uncertainty
      + bound_on_relative_motion_within_interval
```

Because probe orientations are fixed, the conservative translation bound is straightforward:

```text
moving probe vs stationary geometry:
    relative_motion_bound = max_translation_over_interval

moving probe vs moving probe:
    relative_motion_bound = translation_bound_i + translation_bound_j
```

Accept an interval only if:

```text
sampled_clearance > required_margin + relative_motion_bound
```

Otherwise subdivide or reject.

### 8.4 Safety margins

Margins should be explicit in the request and recorded in the certificate.

Suggested margin fields:

```python
@dataclass(frozen=True)
class SafetyPolicy:
    shank_clearance_margin_mm: float
    body_clearance_margin_mm: float
    brain_surface_margin_mm: float
    calibration_uncertainty_mm: float
    online_tracking_tolerance_mm: float
    mesh_uncertainty_mm: float
    controller_interpolation_margin_mm: float
    max_edge_length_mm: float
    max_speed_mm_s: float
    max_accel_mm_s2: float
```

Treat shanks as the most protected geometry.

### 8.5 Retraction hazard reporting

If pure axial retraction fails, do not hide the failure.

Suggested report:

```python
@dataclass(frozen=True)
class RetractionHazard:
    probe_id: str
    segment_id: str
    worst_pair: tuple[str, str]
    min_clearance_mm: float
    first_invalid_fraction_along_edge: float
    suggested_action: Literal[
        "FLAG_FOR_OPERATOR",
        "TRY_ESCAPE_PLANNER",
        "MANUAL_RECOVERY_REQUIRED",
    ]
```

For near-brain moves, default to `FLAG_FOR_OPERATOR` or `MANUAL_RECOVERY_REQUIRED` unless an explicit policy allows automatic escape.

### 8.6 Validation certificate

Every accepted trajectory should emit a certificate.

Suggested fields:

```python
@dataclass(frozen=True)
class ValidationCertificate:
    trajectory_id: str
    trajectory_hash: str
    scene_hash: str
    software_commit: str
    coordinate_frame: Literal["LPS_mm"]
    motion_mode: str
    probe_count: int
    active_probe_ids: tuple[str, ...]
    safety_policy_hash: str
    endpoint_validity: bool
    edge_validity_method: str
    min_clearance_mm: float
    worst_offending_pair: tuple[str, str] | None
    segment_reports: tuple[SegmentValidationReport, ...]
    dry_run_render_path: str | None
    created_at_utc: str
```

Execution rule:

```text
No certificate, no hardware execution.
```

---

## 9. Software stack recommendation

### 9.1 Keep the planner in the existing Python/JAX/FCL ecosystem

The existing package already has the key abstractions and validators. The motion planner should reuse them rather than duplicate them in a separate robotics stack.

Recommended core dependencies:

```text
Python
JAX / jaxlib for differentiable clearance and vectorization
python-fcl / FCL backend for final validation
NumPy / SciPy for geometry and search utilities
Existing rendering backends for dry-run playback
```

### 9.2 OMPL as optional plugin, not core safety dependency

OMPL can be useful for offline fallback planning, especially RRT-Connect or PRM in single-probe 3D spaces.

However, do not make OMPL the safety-critical center of the system if every validity query must cross from C++ into Python and call the existing scene/FCL abstractions. Keep the core deterministic planner and validator native to the repository's planning stack.

### 9.3 MoveIt / ROS 2 only if required by hardware integration

Do not migrate this planner into MoveIt unless the broader hardware/control environment is already ROS 2-based.

Reasons:

- The repo already has scene geometry, kinematics, FCL, SDF optimization, and rendering.
- MoveIt would duplicate many concepts.
- Additional integration complexity does not buy much for fixed-orientation 3-DOF probes.

---

## 10. Proposed package structure

Recommended new modules:

```text
motion/
  __init__.py

  request.py
    MotionRequest
    MotionMode
    SafetyPolicy
    ExecutionPolicy
    CalibrationSweepRequest

  cspace.py
    ProbeConfig
    CompositeConfig
    manipulator_xyz_to_probe_pose(...)
    probe_pose_to_manipulator_xyz(...)
    place_probe_geometry(...)

  hover.py
    compute_hover_pose(...)
    compute_park_pose(...)
    brain_surface_intersection(...)

  validity.py
    StateValidityOracle
    EdgeValidityOracle
    FCLValidityOracle
    SDFClearanceOracle
    ConservativeTranslationEdgeValidator
    ClearanceReport
    RetractionHazard

  templates.py
    axial_retract_path(...)
    axial_descend_path(...)
    straight_line_path(...)
    dogleg_path(...)

  single_probe.py
    TemplatePlanner
    LatticeAStarPlanner
    SingleProbePlanner

  sequence.py
    DisassemblySequencePlanner
    SubsetSearchState
    precedence_graph_from_pair_tests(...)

  optimize.py
    smooth_path_with_sdf(...)
    clearance_optimize_path(...)

  schedule.py
    SerialScheduler
    BatchCompressor
    ConflictGraph
    FixedPathTimingScheduler

  calibration.py
    CalibrationVolume
    CalibrationSweepPlanner
    CalibrationObservation
    CalibrationDataset

  trajectory.py
    ProbeWaypoint
    ProbeSegment
    TimedWaypoint
    TimedTrajectory
    MotionBarrier

  certificate.py
    ValidationCertificate
    SegmentValidationReport
    write_certificate(...)

  render.py
    render_trajectory_preview(...)
```

Hardware-facing modules:

```text
drivers/
  manipulator_socket.py
    ManipulatorSocketClient
    send_absolute_xyz(...)
    read_position_stream(...)

  execution_monitor.py
    TrajectoryExecutor
    TrackingTubeMonitor
    AbortController
    ExecutionLog

  mock_manipulator.py
    MockManipulatorClient
    SimulatedReadbackStream
```

Hard boundary:

```text
motion/ emits validated artifacts and never opens sockets
drivers/ consumes validated artifacts and never plans new geometry
```

---

## 11. Core request and trajectory schemas

### 11.1 Motion request

```python
class MotionMode(Enum):
    DISPERSE = "disperse"
    CONVERGE_TO_HOVER = "converge_to_hover"
    INSERT_ASSIST = "insert_assist"
    CALIBRATION_SWEEP = "calibration_sweep"


@dataclass(frozen=True)
class MotionRequest:
    mode: MotionMode
    current_state: PlanningState
    target_state: PlanningState
    active_probe_ids: tuple[str, ...]
    safety_policy: SafetyPolicy
    execution_policy: ExecutionPolicy
    hover_clearance_mm: float = 3.0
    park_retraction_mm: float = 15.0
    allow_concurrency: bool = False
    require_serial_fallback: bool = True
```

### 11.2 Execution policy

```python
@dataclass(frozen=True)
class ExecutionPolicy:
    allow_hardware_execution: bool
    require_dry_run_render: bool
    require_operator_approval: bool
    allow_insert_below_hover: bool
    allow_auto_escape_near_brain: bool
    max_concurrent_probes: int
    stop_on_tracking_error: bool = True
```

### 11.3 Trajectory

```python
@dataclass(frozen=True)
class ProbeWaypoint:
    probe_id: str
    xyz_lps_mm: tuple[float, float, float]
    t_s: float | None = None


@dataclass(frozen=True)
class ProbeSegment:
    segment_id: str
    probe_id: str
    waypoints: tuple[ProbeWaypoint, ...]
    phase: Literal[
        "retract", "parked_reposition", "converge", "calibration", "insert_assist"
    ]
    max_speed_mm_s: float
    max_accel_mm_s2: float


@dataclass(frozen=True)
class TimedTrajectory:
    trajectory_id: str
    segments: tuple[ProbeSegment, ...]
    barriers: tuple[MotionBarrier, ...]
    certificate: ValidationCertificate
```

---

## 12. Planner pipeline

Recommended pipeline:

```text
Input:
    current PlanningState
    target PlanningState
    scene meshes
    manipulator limits
    safety policy
    execution policy

1. Normalize to canonical LPS millimeters.

2. Compute final, hover, and park poses.
       final_i = planned insertion pose
       hover_i = final_i retracted to 3 mm above brain surface
       park_i  = hover_i retracted farther along the same insertion axis

3. Validate endpoints.
       current state valid or explain existing collision
       target hover state valid with required margin
       parked poses valid with required margin

4. Generate candidate per-probe paths.
       try templates first
       fallback to deterministic lattice A* if needed
       optionally optimize/smooth using JAX SDF

5. Search sequence.
       DISPERSE: current/hover -> park
       CONVERGE_TO_HOVER: reverse of valid hover -> park disassembly sequence
       CALIBRATION_SWEEP: active probe path through validated volume

6. Produce serial-safe trajectory.
       one moving probe at a time
       barriers between segments

7. Optional concurrency.
       build conflict graph over fixed segments
       form validated simultaneous batches
       keep serial fallback

8. Final validation.
       FCL state checks
       FCL CCD or conservative continuous edge checks
       inflated margins
       tracking and calibration uncertainty

9. Emit artifacts.
       trajectory JSON/YAML
       validation certificate
       dry-run render/animation
       hazard report if applicable

10. Hardware execution.
       preflight readback checks
       stream absolute setpoints or micro-waypoints
       monitor tracking tube
       enforce barriers
       stop on deviation
       log execution trace
```

---

## 13. Pseudocode

### 13.1 High-level planning

```python
def plan_motion(request: MotionRequest) -> TimedTrajectory:
    scene = normalize_scene_to_lps(request.current_state.scene)
    poses = compute_final_hover_and_park_poses(request, scene)

    validate_endpoints(scene, poses, request.safety_policy)

    if request.mode == MotionMode.DISPERSE:
        serial = plan_disperse_serial(scene, poses, request)

    elif request.mode == MotionMode.CONVERGE_TO_HOVER:
        disassembly = plan_hover_to_park_disassembly(scene, poses, request)
        serial = reverse_disassembly_to_converge(disassembly)

    elif request.mode == MotionMode.CALIBRATION_SWEEP:
        serial = plan_calibration_sweep(scene, poses, request)

    elif request.mode == MotionMode.INSERT_ASSIST:
        serial = plan_insert_assist(scene, poses, request)

    else:
        raise ValueError(f"Unsupported motion mode: {request.mode}")

    validated_serial = validate_full_trajectory(scene, serial, request.safety_policy)

    if request.allow_concurrency:
        concurrent = try_batch_compression(
            scene, validated_serial, request.safety_policy
        )
        if concurrent is not None:
            return validate_full_trajectory(scene, concurrent, request.safety_policy)

    return validated_serial
```

### 13.2 Reverse-disassembly search

```python
def plan_hover_to_park_disassembly(scene, poses, request):
    start = frozenset()  # no probes parked yet
    goal = frozenset(request.active_probe_ids)

    pq = PriorityQueue()
    pq.push(start, priority=0.0)
    best_cost = {start: 0.0}
    parent = {}

    while pq:
        S = pq.pop()
        if S == goal:
            return reconstruct_sequence(parent, S)

        for probe_id in request.active_probe_ids:
            if probe_id in S:
                continue

            candidate = plan_single_retraction(
                probe_id=probe_id,
                parked_set=S,
                scene=scene,
                poses=poses,
                safety_policy=request.safety_policy,
            )

            if not candidate.valid:
                continue

            S_next = S | {probe_id}
            new_cost = best_cost[S] + candidate.cost

            if new_cost < best_cost.get(S_next, float("inf")):
                best_cost[S_next] = new_cost
                parent[S_next] = (S, candidate)
                pq.push(S_next, priority=new_cost + heuristic(S_next, goal))

    raise PlanningFailure("No valid hover-to-park disassembly sequence found")
```

### 13.3 Concurrent batch compression

```python
def try_batch_compression(scene, serial_trajectory, safety_policy):
    segments = serial_trajectory.segments
    graph = ConflictGraph()

    for a, b in all_pairs(segments):
        if not can_run_simultaneously(scene, a, b, safety_policy):
            graph.add_conflict(a.segment_id, b.segment_id)

    batches = color_or_greedily_pack_conflict_graph(graph, segments)
    batched_trajectory = build_batched_trajectory(batches)

    if validate_full_trajectory(scene, batched_trajectory, safety_policy):
        return batched_trajectory

    return None
```

---

## 14. Hardware execution model

### 14.1 Preflight

Before sending any motion command:

```text
verify certificate exists and matches trajectory
verify scene_hash matches current planning scene
verify software version / commit if required
read current manipulator positions
confirm readback is close to planned start pose
validate current measured state in FCL if possible
confirm no stale calibration
confirm emergency stop path is armed
```

### 14.2 During motion

The executor should:

```text
send absolute XYZ setpoints or validated micro-waypoints
monitor streamed readback
check tracking error against safety tube
enforce speed and acceleration limits
enforce phase barriers
stop all motion if any probe exceeds tolerance
log commands, readback, timestamps, and state transitions
```

### 14.3 After motion

After each segment or batch:

```text
verify final readback is within tolerance
optionally validate measured final state with FCL
log final state
only then release the next barrier
```

### 14.4 Tracking tube

The tracking tube should be derived from the validated trajectory and safety policy.

```text
tracking_tube_radius_mm = online_tracking_tolerance_mm
                        + calibration_uncertainty_mm
                        + controller_interpolation_margin_mm
```

If readback exits the tube:

```text
stop / hold
log deviation
do not continue trajectory
require validated recovery
```

---

## 15. Brain and insertion geometry policy

For `CONVERGE_TO_HOVER`:

- The brain is a forbidden or high-margin geometry.
- The tip must stop at the configured hover clearance, default 3 mm above surface.
- The hover point should be computed along the planned insertion axis.

For `INSERT_ASSIST`:

- The planned shank tract may be an allowed volume.
- Contact with brain outside the planned tract remains forbidden.
- The policy should explicitly encode whether automatic motion below hover is allowed.

Suggested representation:

```python
@dataclass(frozen=True)
class BrainMotionPolicy:
    hover_clearance_mm: float
    surface_uncertainty_mm: float
    allowed_tract_volumes: dict[str, MeshOrCapsuleVolume]
    allow_auto_below_hover: bool
    require_operator_gate_below_hover: bool
```

Default:

```text
allow_auto_below_hover = False
require_operator_gate_below_hover = True
```

---

## 16. Testing and validation plan

### 16.1 Unit tests

Test the following without hardware:

```text
LPS coordinate normalization
final -> hover -> park pose computation
state validity on known collision / non-collision scenes
edge validity on analytic cases
axial retraction path generation
reverse-disassembly sequence search
lattice A* fallback
trajectory serialization and hashing
certificate generation
tracking tube logic
```

### 16.2 Property tests

Useful invariants:

```text
reverse(reverse(disassembly)) preserves sequence geometry
validated serial trajectory remains valid after serialization round-trip
increasing safety margin cannot turn an invalid path valid
moving from hover to park and reversing gives identical geometry with reversed time
```

### 16.3 Simulation tests

Use synthetic multi-probe scenes:

```text
non-interacting probes: all axial moves valid
simple blocked retraction: sequence planner finds correct order
cyclic blockage: planner reports failure or requires fallback
near-miss case: margin increase rejects path
concurrent batch case: independent probes move together
conflicting batch case: probes remain serialized
```

### 16.4 Dry-run rendering

Every planned trajectory should be renderable.

Render views should include:

```text
all probes and fixtures
brain / implant / well keep-out geometry
hover plane / surface target
parked poses
motion paths
worst-clearance segment highlights
```

### 16.5 Hardware-in-the-loop tests

Before real probes:

```text
mock manipulator socket server
recorded readback replay
intentional readback deviation to test abort
latency injection
dropped packets
wrong starting pose
stale certificate / mismatched scene hash
```

Then with dummy probes or non-fragile stand-ins:

```text
single probe axial move
single probe calibration sweep
multi-probe serial disperse
multi-probe converge-to-hover
batch concurrency with large clearances
```

---

## 17. Implementation milestones

### Milestone 1: serial converge-to-hover in simulation

Deliverables:

```text
compute hover and park poses
validate endpoints
plan pure axial paths
sequence probes using reverse-disassembly search
emit serial trajectory
emit validation certificate
render dry-run animation
```

Acceptance criteria:

```text
works on representative 12-15 probe scenes
rejects invalid axial paths with clear hazard reports
trajectory passes final FCL validation
no hardware code involved
```

### Milestone 2: serial disperse

Deliverables:

```text
current/hover -> park planning
same validation/certificate pipeline
failure reports for blocked retractions
```

Acceptance criteria:

```text
safe serial path found when one exists under template moves
clear failure mode when none exists
```

### Milestone 3: calibration sweep

Deliverables:

```text
calibration volume specification
waypoint sampling
valid edge ordering
dwell points
observation logging schema
return-to-park behavior
```

Acceptance criteria:

```text
active probe covers requested volume
all other probes remain parked
trajectory passes validation
observation dataset can be consumed by calibration fitting code
```

### Milestone 4: deterministic fallback planner

Deliverables:

```text
bounded 3D lattice A*
clearance-aware edge costs
fallback from failed template paths
```

Acceptance criteria:

```text
solves representative blocked-retraction and parked-band repositioning cases
paths remain deterministic and certifiable
```

### Milestone 5: JAX/SDF path smoothing

Deliverables:

```text
trajectory smoothing objective
clearance optimization objective
FCL gate after optimization
before/after clearance report
```

Acceptance criteria:

```text
smoothing improves path quality without reducing certified clearance below margin
optimization failure falls back to unsmoothed valid path
```

### Milestone 6: hardware executor

Deliverables:

```text
socket client abstraction
mock manipulator client
absolute setpoint streaming
position readback monitor
tracking tube monitor
barrier execution
abort/e-stop path
execution log
```

Acceptance criteria:

```text
executor refuses uncertified trajectories
executor refuses wrong starting pose
executor stops on readback deviation
executor logs every command and readback sample
```

### Milestone 7: concurrent batch scheduling

Deliverables:

```text
conflict graph over fixed trajectory segments
batch packing
continuous validation of simultaneous batches
serial fallback preserved
```

Acceptance criteria:

```text
independent moves batch together
conflicting moves remain serialized
batched trajectory passes final validation
failure falls back to serial trajectory
```

---

## 18. Failure modes and required behavior

### Invalid current state

Behavior:

```text
report existing collision or margin violation
refuse autonomous motion unless recovery policy is explicitly enabled
```

### Invalid target hover state

Behavior:

```text
report offending geometry pair and minimum clearance
recommend revising target pose, hover clearance, or safety margin
```

### Blocked axial retraction

Behavior:

```text
try alternate sequence first
if sequence cannot avoid blockage, emit RetractionHazard
only run escape planner if policy allows it
```

### Lattice planner failure

Behavior:

```text
fall back to serial/template alternatives if available
otherwise report bounded search region, resolution, and closest failure pair
```

### Concurrency validation failure

Behavior:

```text
remove that concurrent pair from batch
fall back to serial if needed
```

### Readback leaves tracking tube

Behavior:

```text
stop all motion
log deviation
do not continue plan
require validated recovery
```

### Scene hash mismatch

Behavior:

```text
refuse execution
require replanning or explicit scene update
```

---

## 19. Open questions for the implementation team

These should be resolved before hardware execution.

1. **Controller interpolation:** between absolute XYZ setpoints, is the actual path guaranteed to be straight in manipulator coordinates?
2. **Setpoint mode:** does the controller accept streamed setpoints at a fixed rate, or only blocking move-to commands?
3. **Velocity and acceleration limits:** what are safe per-axis and vector limits for each manipulator?
4. **Tracking error:** what readback deviation is normal during motion?
5. **Backlash/repeatability:** what calibration uncertainty should be included in safety margins?
6. **Latency:** what is the worst-case command/readback latency?
7. **Brain surface model:** is the 3 mm hover distance measured from mesh intersection, measured surface, planned tract entry, or operator-updated surface?
8. **Photogrammetry visibility:** does calibration sweep need camera visibility constraints or just volume coverage?
9. **Allowed recovery:** may the planner automatically perform escape maneuvers near the brain, or should those always be manual-gated?
10. **Operator workflow:** should dry-run render inspection be required for every motion, or only during development and flagged hazard cases?

---

## 20. Recommended defaults

Use these defaults unless experiment requirements override them.

```text
motion modes allowed for fully autonomous execution:
    DISPERSE
    CONVERGE_TO_HOVER
    CALIBRATION_SWEEP

motion mode requiring manual/supervised gate:
    INSERT_ASSIST below hover

hover clearance:
    3.0 mm above brain surface along insertion axis

park pose:
    hover pose retracted farther along same insertion axis

execution:
    serial-safe plan always required
    concurrency optional and only after validation

planning:
    templates first
    deterministic lattice A* fallback
    SDF smoothing optional
    FCL final gate mandatory

safety:
    shank gets highest protection margin
    continuous edge validation required
    waypoint-only validation forbidden
    no certificate means no hardware motion
```

---

## 21. Acceptance criteria for first production-capable version

A first production-capable version should satisfy:

```text
1. Plans DISPERSE and CONVERGE_TO_HOVER for representative 12-15 probe scenes.
2. Stops at hover, not final brain insertion, by default.
3. Computes park poses from each probe's own insertion axis.
4. Validates pure axial retractions and reports hazards.
5. Produces a serial-safe trajectory for every successful plan.
6. Emits a validation certificate with scene hash, trajectory hash, margins, and minimum clearance.
7. Runs final FCL state and continuous/conservative edge validation.
8. Renders a dry-run preview.
9. Refuses hardware execution without a valid certificate.
10. Uses absolute XYZ setpoints with readback tracking and abort-on-deviation.
11. Logs all commanded positions, readback positions, timestamps, and barrier transitions.
12. Supports calibration sweep for one active probe while all other probes are parked.
13. Keeps serial fallback when concurrency is enabled.
```

---

## 22. Bottom line

Build the planner around **certified simple motion**, not clever high-dimensional search.

The recommended system is:

```text
Mode-specific request
-> compute final/hover/park poses
-> reverse-disassembly sequence search
-> template or deterministic single-probe paths
-> optional SDF smoothing
-> mandatory FCL continuous validation
-> serial-safe trajectory
-> optional validated concurrent batches
-> certificate and dry-run render
-> socket executor with readback tracking and abort
```

This should be safe, inspectable, feasible for 12-15 probes, and aligned with the existing `aind-rutter` geometry/optimization stack.
