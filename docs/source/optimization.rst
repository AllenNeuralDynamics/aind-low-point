Optimization Developer Guide
============================

This guide documents the current offline placement-optimizer pipeline. It is
intended for developers who need to run the optimizer, change one of its stages,
or understand why the implementation is split across discrete search, batched
JAX polishing, and a final constrained handoff.

The optimizer is not part of the interactive trame planner loop. It is an
offline batch workflow that reads a subject ``ConfigModel`` YAML and an implant
hole YAML, writes pickle artifacts under ``scratch/``, and emits plan-only YAML
files that can be opened later with ``alp-plan --plan``.

The live production entry points are:

* ``alp-phase1`` -> ``aind_low_point.optimization.pipeline.phase1_pool``
* ``alp-phase2`` -> ``aind_low_point.optimization.pipeline.phase2_ipopt``
* ``alp-emit`` -> ``aind_low_point.optimization.pipeline.emit``
* ``scripts/run_subject_overnight.sh`` -> unattended wrapper around those three


Install And Inputs
------------------

Install the optional optimizer stack before running the pipeline:

.. code-block:: bash

   uv sync --extra optimization

The two subject inputs are:

``CONFIG``
   Full aind-low-point config YAML. It defines probes, targets, fixture meshes,
   the implant transform, and the runtime plan skeleton.

``HOLES``
   Implant-bore YAML loaded by ``optimization.holes.load_holes``. If the config
   defines ``implant_to_lps``, the holes are transformed into world LPS before
   optimization.

Example one-subject run:

.. code-block:: bash

   CONFIG=examples/837229-config.yml scripts/run_subject_overnight.sh

The wrapper writes subject-keyed outputs:

.. code-block:: text

   scratch/<config-stem>_pool.pkl
   scratch/<config-stem>_phase2_handoff.pkl
   scratch/<config-stem>_plans/


Pipeline Strategy
-----------------

At a high level, the optimizer solves a mixed discrete/continuous placement
problem by separating cheap combinatorics from expensive continuous geometry.

.. code-block:: text

   config + holes
      |
      v
   runtime adapter
      |
      v
   visibility atlas -> MRV enumeration -> lazy AP/ML/spin seed
      |
      v
   spin restore
      |
      v
   Phase 1 pool: reduced RProp -> full RProp -> soft clearance ranking
      |
      v
   Phase 2: IPOPT/trust-constr constrained polish -> FCL/threading gate
      |
      v
   MMR ranking -> plan YAML emission

The important design choice is that the discrete stage should over-generate
plausible hole/arc decisions, while the continuous stages decide which of those
decisions actually produce a high-coverage, feasible pose.

The split also keeps each algorithm in the regime where it works best:

* The atlas answers a binary reachability question with closed-form geometry.
* Enumeration reasons over discrete hole/arc choices and interval feasibility.
* Phase 1 runs a fast, batched, differentiable soft objective over many
  candidates.
* Phase 2 spends more solver effort on a much smaller set and promotes
  feasibility terms from penalties into explicit inequality constraints.


Runtime Adapter
---------------

``optimization.pipeline.runtime_adapter.OptimizationRuntime`` is the subject
setup boundary. It loads the config, builds the normal runtime, compiles
transforms, loads and transforms holes, and exposes the optimizer-specific
geometry caches.

Key responsibilities:

* ``from_config_path(CONFIG, HOLES)`` builds the shared runtime state.
* ``probes`` are ``ProbeStaticInfo`` records in config probe order.
* ``holes`` are implant holes in world LPS.
* ``head_pitch_deg`` shifts subject AP bounds into the rig-reachable AP window.
* ``build_problem_assets()`` builds probe SDFs/BVHs, fixture SDFs/BVHs, the
  optional brain SDF, and the tuned thick-well fixture.
* ``fcl_fixture_set(..., include_implant=True)`` adds the implant mesh to the
  final FCL gate, even though the soft Phase-1 SDF excludes it so probes can
  thread through bored holes.


Visibility Atlas
----------------

``build_or_load_atlas()`` creates or loads a subject-specific visibility atlas.
The atlas is built from the current runtime and hole file with
``optimization.visibility_atlas.build_visibility_atlas``. It records which
``(probe, hole)`` pairs have sampled AP/ML/spin anchors that can thread all
shanks through the bore.

The atlas is a visibility test, not an optimizer. For each ``(probe, hole)``
pair it:

* Samples interior points on the top ellipse of the bore. Interior rings matter
  for multi-shank probes because the shank-row centroid must stay away from the
  edge if the offset shanks are also going to fit.
* Sweeps a spin grid, usually full-circle.
* Builds a candidate pose from the target point to each sampled top point.
  AP/ML come from that line direction; spin comes from the spin grid.
* Projects every shank line through every section plane of the implant bore.
* Tests each section crossing against the section's rotated ellipse.
* Keeps only anchors where every real shank passes every real section.

The implementation is JAX-vmapped over ``top_sample x spin``. That makes atlas
construction mostly one batched geometric kernel per hole signature rather than
a Python loop over possible poses. A valid anchor stores AP, ML, and spin with
``off_R_mm = off_A_mm = depth_mm = 0``. Offsets and depth are intentionally not
sampled in the atlas; later continuous phases can move those degrees of freedom.

Two details matter when interpreting atlas output:

* The ellipse test uses a small ``oval_slack``. This keeps the atlas from
  false-rejecting poses that are just outside the strict sampled chord but can
  be recovered when the continuous optimizer moves offsets or depth.
* The atlas is allowed to be conservative or approximate, but it should not be
  the final feasibility authority. The final FCL/threading checks happen after
  continuous polish.


Enumeration
-----------

Discrete search lives in ``optimization.pipeline.enumeration``. It consumes the
visibility atlas and emits cheap candidate decisions before any expensive JAX
polish runs.

Then ``Enumerator`` performs hole-first MRV search:

* Each node is a feasible ``(probe, hole)`` pair with an AP interval.
* A candidate assigns every probe to a unique hole and partitions probes into
  arcs.
* AP feasibility uses interval overlap; by the 1-D Helly property, pairwise
  overlap is enough to keep an arc's AP window non-empty.
* Inter-arc AP separation is enforced with greedy interval placement.
* Intra-arc ML separation is enforced with greedy interval packing rather than
  a weaker pairwise-only check.
* ``max_arcs``, ``max_probes_per_arc``, AP range, and ML range can restrict the
  search.

The search order is MRV: pick the unassigned probe with the smallest remaining
hole domain first. After assigning a hole, the search forward-checks uniqueness
by removing that hole from every other unassigned probe. This cuts off dead
branches early instead of discovering duplicate-hole conflicts only after a
full assignment.

AP feasibility is handled as an interval problem:

* Each ``(probe, hole)`` atlas entry contributes an AP interval.
* Probes on the same arc must have a non-empty common AP intersection.
* In one dimension, pairwise interval overlap is enough to guarantee a common
  intersection for the arc.
* Different arcs must be separated by at least 16 degrees. The implementation
  uses greedy interval placement to check whether one AP value per arc can be
  placed with that separation.

ML feasibility is also interval-based. For probes sharing an arc, the enumerator
computes each probe's ML window from atlas anchors within the arc's AP window
and greedily places one ML value per probe with 16 degree separation. This is
stronger than a pairwise "some pair can be far enough apart" check: it verifies
that the whole arc can be packed at once.

Arc Partitions And Hole Tuples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current Phase-1 enumerator is not simply "choose top hole assignment, then
choose arcs." It searches the joint discrete decision:

.. code-block:: text

   probe -> hole
   probe -> arc
   arc -> feasible AP window

The search state is a list of arcs. Each arc stores its current member probes,
their chosen holes, the intersection of their AP intervals, and a bitmask of
atlas nodes in the arc. For each unassigned probe, the search tries two classes
of move:

* join an existing arc, if the new ``(probe, hole)`` node keeps the arc AP
  intersection non-empty and the arc still ML-packs;
* open a new arc, if the total arc set can still be placed with the AP
  separation constraint.

Hole uniqueness is global across all arcs. Assigning a hole removes it from
every remaining probe domain. That makes the hole assignment part of the
search, not a post-hoc filter.

``arc_first_principled.enumerate_arc_first_candidates`` implements the same
broad idea from the opposite direction: enumerate unordered arc partitions,
enumerate feasible hole tuples per arc, then take a Cartesian product across
arcs and reject global hole conflicts. It attaches cheap ranking signals such
as AP intersection width, ML slack, number of atlas anchors, and AP
centeredness. The ``alp-phase1`` console entry point uses
``optimization.pipeline.enumeration.Enumerator`` for the production pool.


Bounded Isotonic AP Placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a candidate fixes ``probe -> hole`` and ``probe -> arc``, each arc has:

* a preferred AP, usually the center of its feasible AP window or a centroid of
  member preferences;
* a lower/upper feasible AP window from the atlas;
* a pairwise 16 degree separation requirement against neighboring arcs.

The seed AP problem is therefore:

.. code-block:: text

   minimize    sum_k (ap_k - preferred_k)^2
   subject to  ap_sorted[k+1] - ap_sorted[k] >= sep
               low_k <= ap_k <= high_k

This is solved by ``arc_placement.bounded_isotonic_arc_aps``. The key trick is
to sort arcs by preferred AP and substitute:

.. code-block:: text

   d_k = ap_sorted[k] - k * sep

The chain constraint becomes monotone:

.. code-block:: text

   d_0 <= d_1 <= ... <= d_n

Without the per-arc boxes, the exact least-squares solution is isotonic
regression, solved by the pool-adjacent-violators algorithm via
``scipy.optimize.isotonic_regression``. If the result also falls inside every
arc's feasible AP window, it is already the constrained optimum. If a window is
active, the code solves the small boxed chain QP with SLSQP.

This is better than greedy AP placement because it splits unavoidable AP
separation deficits across the squeezed arcs instead of arbitrarily pushing all
error into whichever arc is placed last.

The enumerator emits only the cheap discrete decision:

.. code-block:: python

   {
       "probe_to_hole": {...},
       "partition": frozenset(...),
       "arc_aps": [...],
   }

The expensive joint seed is lazy. ``Enumerator.seed(candidate)`` calls
``arc_first_principled.emit_seed`` to compute arc APs, ML seeds, spin seeds,
and the minimum ML gap for only the candidates that will be optimized.

Seed emission has its own two-level algorithm:

* APs are projected into their bounded arc intervals while preserving the
  required inter-arc separation.
* For each arc, ``ml_anchors_mrv`` runs a bounded MRV/CSP over atlas anchors to
  choose one ``(ml, spin, ap)`` anchor per probe. It tries to satisfy the 16
  degree ML gap with anchors close to the chosen arc AP. If the sampled atlas
  cannot satisfy the gap, it returns the max-min-gap best effort and records
  ``min_ml_gap`` as a quality flag rather than dropping the candidate.

That best-effort behavior is intentional: the atlas does not sample offsets and
depth, so a strict seed-level rejection can discard plans that the continuous
optimizer can recover.

The anchor CSP uses the same MRV idea at a smaller scale. Each probe on an arc
has many atlas anchors ``(ml, spin, ap)`` for its assigned hole. Values are
ordered by closeness to the selected arc AP, then the search assigns the probe
with the fewest still-viable ML anchors first. It first tries to meet the 16
degree ML separation exactly. If that fails within the bounded search budget, it
binary-searches the largest achievable separation and returns that best-effort
seed with ``min_ml_gap`` set below threshold.


Spin Restore
------------

Phase 1 does not trust atlas spin seeds as final basins. It first runs batched
round-robin spin restore over the reduced objective:

* ``phase1_pool.restore_group`` builds seed rows from arc APs, ML seeds, and
  spin seeds.
* ``batched_spin_restore.make_batched_spin_restore_partial`` sweeps a full
  circle of spin proposals for each probe.
* The default production knobs are ``N_SPINS=16`` and ``RESTORE_ROUNDS=4`` in
  ``alp-phase1``.
* The restore uses the well-aware reduced clearance objective and returns one
  spin vector per candidate.

This restore is the production spin-basin finder. Heuristic beam-search spin
proposal scripts are exploratory tools and are not part of the production
pipeline.


Phase 1 Pool
------------

``alp-phase1`` runs the full MRV pool through batched continuous optimization
and writes ``Phase1PoolPayload``.

The continuous variable layout is:

.. code-block:: text

   x = (arc_aps, (ml, sx, sy, off_R, off_A, depth) x probe)

Spin is represented by ``(sx, sy)`` during optimization; the objective includes
a unit-circle penalty so the pair behaves as a continuous spin angle without
angle wrap discontinuities.

Phase 1 has two continuous passes:

``reduced``
   Offsets and depth are pinned to zero and coverage is off. The goal is to
   obtain a physically plausible threaded/clear pose.

``full``
   Offsets and depth are free and coverage is on. The goal is to improve target
   coverage while preserving clearance.

The reduced pass is also the "limited" objective: it uses the same x-vector and
same compiled objective machinery as the full pass, but the bounds pin
``off_R``, ``off_A``, and ``depth`` to zero and the runtime ``cov_weight`` is
zero. In code this lets one JAX kernel serve both passes:

* Reduced/limited: ``lo == hi == 0`` for offsets/depth and
  ``cov_weight = 0``.
* Full: offsets/depth use their real bounds and ``cov_weight = COV_WEIGHT``.

The Phase-1 scalar objective is a soft-penalty objective:

.. code-block:: text

   minimize
       - coverage
       + threading_penalty
       + probe_probe_clearance_penalty
       + probe_fixture_clearance_penalty
       + arc_ap_and_intra_arc_ml_penalty
       + soft_bounds_penalty
       + spin_unit_circle_penalty
       + optional_brain_containment_penalty
       - clearance_margin_reward
       - threading_margin_reward

The penalties are one-sided squared penalties: they are zero when a condition is
satisfied and grow smoothly when it is violated. Margin rewards are saturating
bonuses inside the feasible region; they encourage slack without letting slack
dominate coverage. The clearance model is dual-representation: probe bodies use
alpha-wrap SDFs and shank/headstage representations use analytic OBB/section
terms, with soft-min/top-k aggregation so the gradients are not controlled by a
single brittle closest point.

Phase 1 remains soft even when penalties have large weights. A candidate with a
small collision, a slightly bad threading value, or a cramped AP/ML separation
can still move through the batch. That is deliberate: the pool builder is a
throughput-oriented ranking stage, not the final feasibility judge.

Coarse Versus Fine Fidelity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The coarse/fine schedule changes the SDF surface-sample count used by the
clearance terms.

``COARSE_N``
   Number of surface samples for the coarse pass. The default ``1000`` is much
   cheaper than the fine 5000-sample representation and tends to smooth narrow
   collision features.

``REDUCED_FINE`` and ``FULL_FINE``
   Number of final steps in each pass that rerun at fine fidelity. The preceding
   steps run at ``COARSE_N`` when ``COARSE_N < 5000``.

The practical effect is a homotopy:

* Coarse steps are faster and often easier to optimize because they smooth the
  collision landscape.
* Fine finish steps reintroduce the higher-resolution geometry before the pose
  is scored and handed downstream.
* Both the reduced and full passes get a fine finish by default; skipping the
  reduced fine finish changes the basin handed to the full pass.

The current tuned defaults are:

* ``MINIMIZER=rprop``: sign-based iRprop-, chosen because ADAM's second moment
  can freeze after large collision-gradient spikes.
* ``WELL=thick``: soft SDF uses a solidified well body; final FCL still checks
  the true mesh.
* ``COARSE_N=1000``, ``REDUCED_FINE=50``, ``FULL_FINE=50``: coarse-to-fine SDF
  surface schedule.
* ``STAGE1=500``, ``STAGE2=500`` total reduced/full steps.
* ``FCL_TOPK=300`` when running the command directly; the overnight wrapper sets
  ``FCL_TOPK=0`` and leaves final FCL gating to Phase 2.

Run directly:

.. code-block:: bash

   CONFIG=examples/837229-config.yml \
   HOLES=scratch/0283-300-04.holes.yml \
   OUT=scratch/837229_pool.pkl \
   JAX_PLATFORMS=cuda uv run --python 3.13 alp-phase1

Useful Phase-1 environment knobs:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Meaning
   * - ``MAX_ARCS``
     - ``3``
     - Maximum arcs in the MRV search.
   * - ``MAX_PROBES_PER_ARC``
     - ``4``
     - Per-arc cap used by the search.
   * - ``ONLY_NARCS``
     - ``0``
     - Restrict a run to one arc-count group; the overnight wrapper uses this
       for one GPU process per group.
   * - ``SEED_CACHE``
     - ``scratch/mrv_seeds_<config-stem>.pkl``
     - Subject-specific enumerate/seed cache.
   * - ``OUT``
     - ``scratch/mrv_pool_results.pkl``
     - Resumable pool output.
   * - ``LIMIT``
     - ``0``
     - Candidate cap for smoke tests.
   * - ``CHUNK``, ``RESTORE_CHUNK``
     - ``256``, ``128``
     - Batched JAX chunk sizes.
   * - ``COV_NORM``, ``COV_ALPHA``, ``COV_WEIGHT``
     - ``0``, ``0.2``, ``1.0``
     - Optional normalized/weighted coverage objective.

Each Phase-1 record contains the discrete decision, explicit arc assignment,
full pose ``x``, reduced checkpoint ``x_reduced``, objective, soft clearance
metrics, and optional top-K FCL slack. Phase 2 rebuilds statics from the saved
assignment; it does not re-run enumeration or infer arc ordering from a
``frozenset``.


Phase 2 Handoff
---------------

``alp-phase2`` loads a Phase-1 pool, selects top candidates by a chosen metric,
runs a constrained continuous polish, gates the results, and writes
``Phase2HandoffPayload``.

Default selection is ``SELECT_BY=min_clear``. ``SELECT_BY=objective`` is also
supported and sorts ascending because lower objective is better.

The default solver is ``SOLVER=ipopt`` using ``cyipopt.minimize_ipopt`` with a
limited-memory Hessian approximation. ``SOLVER=trust-constr`` keeps the scipy
path available. Phase 2 can run in a thread pool sharing one GPU context
(``POOL=thread``, default) or a process pool when needed.

Phase 2 uses the same Phase-1 x-vector and mostly the same JAX geometry kernels,
but it changes the mathematical contract. Feasibility terms become scipy
inequality constraints, each expressed as ``slack(x) >= 0``:

* Threading: ``threading_oval_tolerance - g_thread`` for each real
  probe/section/shank tuple.
* Probe-probe clearance: soft dual-rep clearance minus ``min_clearance_mm`` for
  each pair/category.
* Probe-fixture clearance: fixture clearance minus ``min_clearance_mm``.
* Brain containment, when present: negative signed distance margin at shank
  tips.
* Arc AP separation: ``abs_smooth(ap_i - ap_j) - 16 deg``.
* Same-arc ML separation: ``abs_smooth(ml_i - ml_j) - 16 deg``.

The Phase-2 objective is therefore smaller:

.. code-block:: text

   minimize
       - coverage
       + soft_bounds_penalty
       + spin_unit_circle_penalty
       - clearance_margin_reward
       - threading_margin_reward

This is the core Phase-1/Phase-2 distinction. Phase 1 asks, "which thousands of
discrete decisions can be moved into promising low-penalty basins quickly?"
Phase 2 asks, "can this smaller selected set satisfy the constraints while
retaining or improving coverage?"

Run directly:

.. code-block:: bash

   SOLVER=ipopt \
   CONFIG=examples/837229-config.yml \
   HOLES=scratch/0283-300-04.holes.yml \
   POSES=scratch/837229_pool.pkl \
   OUT=scratch/837229_phase2_handoff.pkl \
   TOPK=200 P2_ITER=1000 \
   PLATFORM=gpu POOL=thread WORKERS=4 \
   JAX_PLATFORMS=cuda uv run --python 3.13 alp-phase2

Phase 2 reports two feasibility axes:

* ``fcl``: final probe/fixture/implant FCL slack in mm.
* ``max_g_thread``: worst bore-threading constraint value.

Strict feasibility is ``fcl >= -1e-4`` and ``max_g_thread <= 0``. The handoff
``kept`` band is intentionally looser by default: ``FCL_TOL=0.2`` and
``G_TOL=0.2`` admit mildly fixable plans while retaining all results in the
``all`` list for inspection.

The final ``ranked`` list is MMR-ranked: high post-Phase-2 coverage is balanced
against similarity to already selected hole assignments. This keeps the handoff
set useful to a human reviewer instead of returning many nearly identical
plans.


Emit Plans
----------

``alp-emit`` is pure reconstruction. It loads the handoff, applies each saved
pose to a mesh-free planning state, reorders arcs/probes for rig readability,
and writes plan-only YAML files plus a decision tree and manifest.

Run directly:

.. code-block:: bash

   CONFIG=examples/837229-config.yml \
   HOLES=scratch/0283-300-04.holes.yml \
   HANDOFF=scratch/837229_phase2_handoff.pkl \
   N=15 OUTDIR=scratch/837229_plans \
   JAX_PLATFORMS=cpu uv run --python 3.13 alp-emit

Outputs:

.. code-block:: text

   scratch/837229_plans/
     manifest.md
     tree.txt
     plans/
       plan-01-cov17.44-....plan.yml

Open a generated plan with the interactive planner:

.. code-block:: bash

   uv run alp-plan examples/837229-config.yml \
     --plan scratch/837229_plans/plans/plan-01-cov17.44-....plan.yml


Key Modules
-----------

* ``optimization.pipeline.contracts``: TypedDict/dataclass boundaries for
  pickle payloads and callable bundles.
* ``optimization.pipeline.enumeration``: visibility-atlas cache handling and MRV
  hole/arc enumeration.
* ``optimization.pipeline.phase1_pool``: production Phase-1 driver: seed cache,
  spin restore, batched RProp, resume, and Phase-1 payload writing.
* ``optimization.pipeline.phase1_build``: batched and chunked JAX objective
  builders. This is where per-candidate packed statics become reusable vmapped
  kernels.
* ``optimization.pipeline.phase1_geometry``: bounds, fixture/brain SDF
  construction, coverage data, and shared utility helpers.
* ``optimization.pipeline.phase2_ipopt``: Phase-2 selection, solver dispatch,
  FCL/threading gate, and MMR ranking.
* ``optimization.pipeline.emit``: handoff-to-plan reconstruction and
  manifest/tree emission.
* ``optimization.pipeline.runtime_adapter``: subject/runtime setup facade used
  by the pipeline.
* ``optimization.assignment_contracts``: lightweight discrete assignment
  carriers shared by the enumerator, static builders, and batched objectives.
* ``optimization.arc_placement``: bounded isotonic AP placement for separated
  arc seeds.
* ``optimization.pose_bank``: target-oriented per-pair pose-bank scoring used
  by pose feature precomputation.
* ``optimization.probe_static``: per-candidate static geometry builder and
  optimization weight contract.
* ``optimization.phase1_objective_jax`` and ``optimization.phase2_objective_jax``:
  differentiable objectives and constraints used by Phase 1 and Phase 2.
* ``optimization.fcl_validator``: ground-truth FCL validation.
* ``scripts/run_subject_overnight.sh``: recommended unattended production
  wrapper.
* ``scripts/staged_adam.py``, ``scripts/manual_mrv_chain.py``,
  ``scripts/coarse_fine_surf.py``, and ``scripts/instrument_adam_freeze.py``:
  diagnostic and experimental scripts. Do not treat them as production entry
  points unless their module docstring says so.
