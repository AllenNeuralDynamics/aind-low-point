Optimization Developer Guide
============================

This guide documents the current offline placement-optimizer pipeline. It is
intended for developers who need to run the optimizer, change one of its stages,
or understand why the implementation is split across discrete search, batched
JAX polishing, and a final constrained handoff.

The optimizer is not part of the interactive trame planner loop. It is an
offline batch workflow that reads a subject ``ConfigModel`` YAML and an implant
hole YAML, writes gzipped JSON artifacts under ``scratch/``, and emits plan-only
YAML files that can be opened later with ``rutter-plan --plan``.

The three stages run in order:

.. code-block:: text

   RUTTER_CONFIG + RUTTER_HOLES
     -> rutter-phase1  (MRV enumerate, spin restore, batched RProp pool)
     -> rutter-phase2  (IPOPT/trust-constr polish, FCL/threading gate, MMR ranking)
     -> rutter-emit    (plan-only YAMLs, tree.txt, manifest.md)

The live production entry points are:

* ``rutter-phase1`` -> ``aind_rutter.optimization.pipeline.phase1``
* ``rutter-phase2`` -> ``aind_rutter.optimization.pipeline.phase2``
* ``rutter-emit`` -> ``aind_rutter.optimization.pipeline.emit``
* ``scripts/run_subject_overnight.sh`` -> unattended wrapper around those three

Name the stages after those commands: **Phase 1 pool**, **Phase 2 handoff**,
**emit**. Older notes and a few docstrings say "Stage 2" and "Stage 3", which
numbered something else and match nothing in the code today.

Three working notes sit beside this guide and go stale faster, so they carry
dates: ``dev/POOL_RUN_CONFIGS.md`` (tuned Phase-1 presets, with measurements),
``dev/spin_basin_experiments.md`` (why the round-robin spin restore is still the
production spin-basin finder) and ``dev/proposals/PIPELINE_PLAN.md`` (proposed
hardening,
not yet built).


Install And Inputs
------------------

Install the optional optimizer stack before running the pipeline:

.. code-block:: bash

   uv sync --python 3.13 --extra optimization

The two subject inputs are:

``RUTTER_CONFIG``
   Full aind-rutter config YAML. It defines probes, targets, fixture meshes,
   the implant transform, and the runtime plan skeleton. Its ``paths:`` names
   are located as for every command (see the config model guide); paths files
   in ``RUTTER_PATHS``, separated by ``os.pathsep``, rank first. Pack files
   it names are fetched at load; ``RUTTER_OFFLINE=1`` fetches none, so a long
   run can ``rutter-data pull`` its packs first and fail fast on a gap.

``RUTTER_HOLES``
   Implant-bore YAML loaded by ``optimization.geometry.holes.load_holes``. Each
   hole has
   oval cross-sections along its axis and optional ``walls``: planes that cut into
   the bore, such as an implant edge running through the channel. A shank must
   cross every section inside its oval and on the open side of every wall. If the
   config defines ``implant_to_lps``, the holes are transformed into world LPS
   before optimization. Fit it from the implant's STEP, which needs the ``step``
   extra. A flat face cutting into a bore's oval (a D-shaped bore) is written as a
   wall through the face's point nearest the axis. Bores are numbered in reading
   order unless ``--diagram`` names a file of the manufacturer's numbering, which
   also drops bores no probe passes through:

   .. code-block:: bash

      uv run --python 3.13 rutter-bores 0283-300-04.step 0283-300-04.holes.yml \
          --diagram 0283-300-04.diagram.yml

Example one-subject run:

.. code-block:: bash

   RUTTER_CONFIG=examples/837229-config.yml scripts/run_subject_overnight.sh

The wrapper writes outputs keyed by the config's stem:

.. code-block:: text

   scratch/<config-stem>_pool.json.gz
   scratch/<config-stem>_phase2_handoff.json
   scratch/<config-stem>_plans/

With ``RUTTER_RUN_DIR`` set, everything the run writes goes under that
directory instead, the geometry caches included, so two configs with one stem
do not share any of it:

.. code-block:: text

   <run dir>/pool.json.gz
   <run dir>/phase2_handoff.json
   <run dir>/plans/
   <run dir>/manifest.json


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
   Phase 1 pool: full RProp -> soft clearance ranking
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

``optimization.pipeline.subject.OptimizationRuntime`` is the subject
setup boundary. It loads the config, builds the normal runtime, compiles
transforms, loads and transforms holes, and exposes the optimizer-specific
geometry caches.

Key responsibilities:

* ``from_config_path(config_path, holes_path)`` builds the shared runtime state.
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
* Tests each section crossing against the section's rotated ellipse and the
  hole's wall planes.
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

``optimization.pipeline.enumeration.Enumerator`` implements this as the
production search for ``rutter-phase1``. It carries only cheap discrete decisions
in the candidate pool, then asks the shared seed emitter for AP, ML, and spin
starts only for candidates that are actually optimized.


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

This is solved by
``optimization.assignment.arc_placement.bounded_isotonic_arc_aps``. The key
trick is to sort arcs by preferred AP and substitute:

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
``optimization.assignment.seed_emission.emit_seed`` to compute arc APs, ML
seeds, spin seeds, and the minimum ML gap for only the candidates that will be
optimized.

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

Phase 1 does not trust atlas spin seeds as final basins. It first runs a
batched round-robin spin restore:

* ``phase1.restore_group`` builds seed rows from arc APs, ML seeds, and
  spin seeds.
* ``optimization.search.spin_restore.make_batched_spin_restore_partial``
  sweeps a full circle of spin proposals for each probe.
* The default production knobs are ``RUTTER_N_SPINS=16`` and
  ``RUTTER_RESTORE_ROUNDS=4`` in ``rutter-phase1``.
* Each proposal is charged Phase 1's penalty on its threading and clearance
  shortfall: the spec of `Feasibility`_, read through the same swept clearance
  query and gains, with the well as the only fixture. The restore returns one
  spin vector per candidate.

This restore is the production spin-basin finder. Heuristic beam-search spin
proposal scripts are exploratory tools and are not part of the production
pipeline.


Feasibility
-----------

What a stage counts as a feasible plan is a ``FeasibilitySpec``
(``optimization.objectives.feasibility``): the clearance margin, whether the
soft minimum is corrected, the threading tolerance and shaft length, the arc AP
and same-arc ML separations, the brain margin, and the soft minimum's ``beta``
and top-k. ``optimization.objectives.slacks`` turns each condition into a slack
that is ``>= 0`` where it is met. Phase 2 constrains ``gain * slack >= 0``;
Phase 1 and the spin restore penalise ``(gain * shortfall)^2``. A pose that
Phase 1 charges nothing for is therefore one that Phase 2 accepts under the same
spec. The gains (``clearance.aggregate``) scale the solver's view of a row and
never change its sign. Phase 2 and the spin restore lift the OBB categories by
``OBB_SLACK_GAIN`` (100); Phase 1 reads every category at gain 1
(``Phase1Weights.obb_slack_gain``), since at 100 grazing shank contact outweighs
coverage and clears no more layouts.

The soft minimum reads below the exact one by up to ``ln(k)/beta``. With
``softmin_correction`` the spec adds each category's typical shortfall
(``PROBE_PAIR_SOFTMIN_BIAS_MM``, ``FIXTURE_PAIR_SOFTMIN_BIAS_MM``) back, so the
margin reads as a distance between surfaces. The offsets were measured at
``beta`` 20 and top-k 16/8/8, and a spec with the correction refuses any other.

Every stage reads one spec, ``PipelineSettings.feasibility_spec``: a clearance
margin of ``RUTTER_MIN_CLEARANCE`` (default 0.1 mm), corrected, which
``RUTTER_MIN_BODY_CLEARANCE`` replaces, when set, wherever a probe's body (the
probe less its shanks) meets another body, and ``RUTTER_MIN_BODY_FIXTURE_CLEARANCE``
wherever it meets a fixture. All three must stay below the 1.5 mm the clearance rows read to. The retired
``RUTTER_MINCLEAR`` is refused by name, since an unknown variable would
otherwise be ignored and the run would get the default.

Coverage floor
~~~~~~~~~~~~~~

``RUTTER_MIN_COVERAGE_FRACTION`` (default 0, off) requires each probe to reach
that fraction of its *threaded best*: the most coverage it reaches alone
through a bore it threads. It turns trading a target away for clearance into a
failure rather than a lower score. Before the first group, Phase 1 solves every
probe by itself in each bore the visibility atlas says it passes, from eight of
the atlas's anchors spread over spin, with the full stage and neither the
normalization nor the floor shaping the solve, and keeps the best coverage
among the results whose threading, well and brain rows hold
(``pipeline.coverage_floor``). The log prints each probe's best per bore. The
bests are an upper bound on what any layout gives that probe, and the pool
records them as ``coverage_floor_ceilings``; Phase 2 reads them from there and
refuses to run a floor against a pool without them. A probe with no feasible
pose through any bore stops the run.

The floor is one more slack, ``coverage / threaded best − fraction``, so its
rows sit at the same scale for every probe. Phase 1 penalises its squared
shortfall at ``Phase1Weights.lambda_coverage_floor`` (1e4: a probe a tenth
short pays what the other penalties do at their order 100), scaled by the
coverage weight, so the reduced stage ignores it. The penalty is continuously
differentiable and its gradient goes to zero at the floor. Phase 2 constrains a
``coverage`` group of one row per probe, empty when the floor is off. The
threaded bests are separate from the open-space ceilings ``RUTTER_COV_NORM``
divides by.

A pool records the spec and the objective it was built under, as its header's
``feasibility`` and ``objective``. Phase 1 resumes only a pool whose record
matches its settings, and Phase 2 refuses one whose record differs from its
own; a pool that records neither is taken on trust, with a warning.

Where a stage departs from the spec, the site says why:

* The spin restore, Phase 1 and the pool's ``min_clear`` check the well alone.
  The headframe and cone cost VRAM and seldom decide whether a candidate
  clears.
* ``min_clear`` is the worst clearance slack, before gains and without Phase
  2's 1.5 mm cap (`Phase 2 Handoff`_), so a candidate scoring ``>= 0`` meets
  Phase 2's clearance rows on the well.
* Phase 1's straight-through SAT changes the gradient, not the value.


Objective
---------

What every stage minimizes is an ``ObjectiveSpec``
(``optimization.objectives.objective_spec``), evaluated by one function,
``objectives.terms.objective_terms``:

.. code-block:: text

   objective = - coverage_gain * coverage
               + lambda_bounds * soft_bounds_penalty
               + lambda_unit_circle * spin_unit_circle_penalty
               - lambda_margin_clear * clearance_margin_reward
               - lambda_margin_thread * threading_margin_reward

Phase 2 minimizes it subject to the constraints of `Feasibility`_; Phase 1
minimizes it plus the penalties on those constraints' shortfall. At a pose Phase
2 finds feasible the two values are equal, so Phase 1 ranks candidates by what
Phase 2 will optimize; ``tests/test_feasibility_spec.py`` checks that, and that
away from feasibility they differ by exactly the penalty on Phase 2's rows.
Phase 1's clearance reward, like its penalty, reads the well alone.

Both stages build the spec from the settings they share,
``PipelineSettings.objective_spec``:

.. list-table::
   :header-rows: 1

   * - Setting
     - Spec field
     - Default
   * - ``RUTTER_COV_WEIGHT``
     - ``coverage_gain``
     - 7.0
   * - ``RUTTER_COV_ALPHA``, read with ``RUTTER_COV_NORM``
     - ``cov_alpha``
     - 0.2
   * - ``RUTTER_LAM_CLEAR``
     - ``lambda_margin_clear``
     - 0.0
   * - ``RUTTER_TAU_CLEAR``
     - ``tau_clear_mm``
     - 0.8
   * - ``RUTTER_SMOOTH_REWARD``
     - ``smooth_clearance_reward``
     - off
   * - ``RUTTER_LAM_THREAD``
     - ``lambda_margin_thread``
     - 1.0
   * - ``RUTTER_SMOOTH_THREAD_REWARD``
     - ``smooth_thread_reward``
     - off

The clearance reward is off by default. It duplicates the clearance constraint,
and its gradient comes from the one closest surface sample, which flips between
near-tied samples and makes solves irreproducible.


Phase 1 Pool
------------

``rutter-phase1`` runs the full MRV pool through batched continuous optimization
and writes ``Phase1PoolPayload``.

The continuous variable layout is:

.. code-block:: text

   x = (arc_aps, (ml, sx, sy, off_R, off_A, depth) x probe)

Spin is represented by ``(sx, sy)`` during optimization; the objective includes
a unit-circle penalty so the pair behaves as a continuous spin angle without
angle wrap discontinuities. With ``RUTTER_SPIN_PARAM=angle`` the RProp passes
instead step each spin angle, held in the ``sx`` slot and wrapped into
``[-pi, pi)``, and the differentiated objective expands it back to the pair;
Phase 2 reads the same setting for its IPOPT solve. The pool stores the pair
either way.

Phase 1 has two continuous passes, of which only ``full`` runs by default
(``RUTTER_STAGE1=0``):

``reduced``
   Offsets and depth are pinned to zero and coverage is off. The goal is to
   obtain a physically plausible threaded/clear pose.

``full``
   Offsets and depth are free and coverage is on. The goal is to improve target
   coverage while preserving clearance.

The reduced pass is also the "limited" objective: it uses the same x-vector and
same compiled objective machinery as the full pass, but the bounds pin
``off_R``, ``off_A``, and ``depth`` to zero and the runtime coverage scale is
zero. In code this lets one JAX kernel serve both passes:

* Reduced/limited: ``lo == hi == 0`` for offsets/depth and coverage scale 0.
* Full: offsets/depth use their real bounds and coverage scale 1, so coverage
  enters at the spec's ``coverage_gain``.

The Phase-1 scalar objective is Phase 2's plus soft penalties:

.. code-block:: text

   minimize
       objective                      (see Objective)
       + threading_penalty
       + probe_probe_clearance_penalty
       + probe_fixture_clearance_penalty
       + arc_ap_and_intra_arc_ml_penalty
       + optional_brain_containment_penalty

The penalties are one-sided squared penalties: they are zero when a condition is
satisfied and grow smoothly when it is violated. The margin rewards in the
objective are saturating bonuses inside the feasible region; they encourage
slack without letting slack dominate coverage. The clearance model is dual-representation: probe bodies use
alpha-wrap SDFs and shank/headstage representations use analytic OBB/section
terms, with soft-min/top-k aggregation so the gradients are not controlled by a
single brittle closest point.

Phase 1 remains soft even when penalties have large weights. A candidate with a
small collision, a slightly bad threading value, or a cramped AP/ML separation
can still move through the batch. That is deliberate: the pool builder is a
throughput-oriented ranking stage, not the final feasibility judge.

Each stage is a single RProp pass over the probe surface samples Phase 2 reads
(``RUTTER_N_SURF``). Restarting the step sizes partway through a stage, or
reading fewer samples in its first steps, changes Phase 1's pool but not the
layouts Phase 2 delivers from it.

The current tuned defaults are:

* Sign-based iRprop-, chosen because ADAM's second moment can freeze after
  large collision-gradient spikes.
* ``RUTTER_WELL=thick``: soft SDF uses a solidified well body; final FCL still
  checks the true mesh.
* ``RUTTER_STAGE1=0``, ``RUTTER_STAGE2=500``: the full stage alone. Adding the
  reduced stage (500 steps) doubles Phase 1's time and delivers the same layouts
  after Phase 2; so does spending those steps in the full stage.
* ``RUTTER_FIXTURE_C2F=1``: the probe side of probe-well body clearance is read
  from the probe's coarse-to-fine samples, as probe pairs are, rather than its
  uniform surface samples. It delivers the same layouts with fewer fixture
  collisions; Phase 2 keeps the uniform samples.

Run directly:

.. code-block:: bash

   RUTTER_CONFIG=examples/837229-config.yml \
   RUTTER_HOLES=0283-300-04.holes.yml \
   RUTTER_OUT=scratch/837229_pool.json.gz \
   JAX_PLATFORMS=cuda uv run --python 3.13 rutter-phase1

Useful Phase-1 environment knobs:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Meaning
   * - ``RUTTER_MAX_ARCS``
     - ``3``
     - Maximum arcs in the MRV search.
   * - ``RUTTER_MAX_PROBES_PER_ARC``
     - ``4``
     - Per-arc cap used by the search.
   * - ``RUTTER_ONLY_NARCS``
     - ``0``
     - Restrict a run to one arc-count group; the overnight wrapper uses this
       for one GPU process per group.
   * - ``RUTTER_SEED_CACHE``
     - ``scratch/mrv_seeds_<config-stem>.json.gz``
     - Subject-specific enumerate/seed cache.
   * - ``RUTTER_OUT``
     - ``scratch/mrv_pool_results.json.gz``
     - Resumable pool output.
   * - ``RUTTER_LIMIT``
     - ``0``
     - Candidate cap for smoke tests.
   * - ``RUTTER_CHUNK``, ``RUTTER_RESTORE_CHUNK``
     - ``128``, ``128``
     - Batched JAX chunk sizes.
   * - ``RUTTER_COV_NORM``, ``RUTTER_COV_ALPHA``, ``RUTTER_COV_WEIGHT``
     - ``1``, ``0.2``, ``7.0``
     - Normalized, weighted coverage objective; with the rest of the
       objective's settings (see `Objective`_), Phase 2 reads the same values.
   * - ``RUTTER_MIN_CLEARANCE``
     - ``0.1``
     - Clearance in mm every stage requires (see `Feasibility`_).
   * - ``RUTTER_MIN_BODY_CLEARANCE``
     - unset
     - Clearance in mm from a probe's body to another body; unset,
       ``RUTTER_MIN_CLEARANCE``.
   * - ``RUTTER_MIN_BODY_FIXTURE_CLEARANCE``
     - unset
     - Clearance in mm from a probe's body to a fixture; unset,
       ``RUTTER_MIN_CLEARANCE``.
   * - ``RUTTER_MIN_COVERAGE_FRACTION``
     - ``0``
     - Each probe's coverage floor as a fraction of its threaded best (see
       `Coverage floor`_); 0 sets none.

Each Phase-1 record contains the discrete decision, explicit arc assignment,
full pose ``x``, reduced checkpoint ``x_reduced`` (the restored start when the
reduced stage is off), objective, ``min_clear`` (see `Feasibility`_), and
optional top-K FCL slack. Phase 2 rebuilds statics from the saved
assignment; it does not re-run enumeration or infer arc ordering from a
``frozenset``.


Phase 2 Handoff
---------------

``rutter-phase2`` loads a Phase-1 pool, selects top candidates by a chosen metric,
runs a constrained continuous polish, gates the results, and writes
``Phase2HandoffPayload``.

Default selection is ``RUTTER_SELECT_BY=learned`` (``pipeline/learned_rank.py``):
the objective's first ``RUTTER_RANK_HEAD`` (100) candidates, then the rest by a
gradient-boosted model's P(kept) × E[coverage percentile | kept]. Its features
mean the same thing on any subject: Phase-1 coverage and objective as
percentiles within the pool, the FCL, threading and clearance values at the
Phase-1 pose, slider geometry and per-probe offset and depth. They are measured
on every pool record before selection, on ``RUTTER_RANK_WORKERS`` (8) CPU
processes; 17,000 records take about four minutes on 16. The model is fitted at run
time on ``RUTTER_RANKER_TABLE``, a pool whose every record Phase 2 solved, written
by ``learned_rank.write_training_table``; the package ships none, and a cut past
the head without one is refused. A cut within the head needs no table and
measures nothing. The objective finds a
pool's easy best layouts first, and the model reaches the ones it buries, so
each prefix of the hybrid order holds about as many of a pool's best layouts as
the better of the two at that ``RUTTER_TOPK``. Past a few hundred, the hybrid
holds more.

``RUTTER_SELECT_BY=objective`` ranks by the objective, ascending because lower
is better; any other record field, such as ``min_clear``, sorts descending. ``min_clear``
is a poor ranker: Phase 1's penalty leaves converged candidates a few micrometres short
of the spec, so it orders them by that residue, and it ignores threading.

The default solver is ``RUTTER_SOLVER=ipopt`` using ``cyipopt.minimize_ipopt`` with a
limited-memory Hessian approximation. ``RUTTER_SOLVER=trust-constr`` keeps the scipy
path available. Phase 2 can run in a thread pool sharing one GPU context
(``RUTTER_POOL=thread``, default) or a process pool when needed.

Phase 2 uses the same Phase-1 x-vector and mostly the same JAX geometry kernels,
but it changes the mathematical contract. Feasibility terms become scipy
inequality constraints, each expressed as ``slack(x) >= 0``:

* Threading: ``threading_tolerance - g_thread`` for each real
  probe/section/shank tuple. Where a hole has walls, ``g_thread`` is the larger of
  the oval value and the wall distance scaled by ``2 / b``, so one tolerance means
  about the same distance for both. ``g_thread`` is the worst of the shank's four
  cross-section corner edges, whose half-widths come from the collision mesh;
  since ``g`` is convex over a section, the corners bound the whole blade.
  Ovals and walls are inset by ``RUTTER_THREADING_MARGIN_MM``, a buffer against
  the soft penalty settling just past ``g = 0`` (default 0.04 mm).
  ``RUTTER_THREAD_SHANK=centreline`` tests the centreline instead, whose inset
  (default 0.07 mm) must also cover the shank's half-width. Every stage — spin restore, the reduced stage, Phase 1 and Phase 2 —
  threads the same way; the hole-assignment gate stays centreline.
* Probe-probe clearance: soft dual-rep clearance, corrected for the soft
  minimum's shortfall, minus ``RUTTER_MIN_CLEARANCE`` for each pair/category, or
  ``RUTTER_MIN_BODY_CLEARANCE`` for body-body when set (see `Feasibility`_).
* Probe-fixture clearance: likewise, per fixture, probe and category, with
  ``RUTTER_MIN_BODY_FIXTURE_CLEARANCE`` for the body category.
* Brain containment, when present: negative signed distance margin at shank
  tips.
* Arc AP separation: ``abs_smooth(ap_i - ap_j)`` minus the spec's 16 deg.
* Same-arc ML separation: ``abs_smooth(ml_i - ml_j)`` minus the spec's 16 deg.
* Coverage floor, when set: each probe's coverage over its threaded best,
  minus ``RUTTER_MIN_COVERAGE_FRACTION`` (see `Coverage floor`_).

Phase 2 minimizes the `Objective`_ alone; the penalties Phase 1 adds to it are
these constraints.

This is the core Phase-1/Phase-2 distinction. Phase 1 asks, "which thousands of
discrete decisions can be moved into promising low-penalty basins quickly?"
Phase 2 asks, "can this smaller selected set satisfy the constraints while
retaining or improving coverage?"

Run directly:

.. code-block:: bash

   RUTTER_SOLVER=ipopt \
   RUTTER_CONFIG=examples/837229-config.yml \
   RUTTER_HOLES=0283-300-04.holes.yml \
   RUTTER_POSES=scratch/837229_pool.json.gz \
   RUTTER_OUT=scratch/837229_phase2_handoff.json \
   RUTTER_TOPK=200 \
   RUTTER_PLATFORM=gpu RUTTER_POOL=thread RUTTER_WORKERS=4 \
   JAX_PLATFORMS=cuda uv run --python 3.13 rutter-phase2

Phase 2 reports two feasibility axes:

* ``fcl``: final probe/fixture/implant FCL slack in mm.
* ``max_g_thread``: worst bore-threading constraint value.

Strict feasibility is ``fcl >= -1e-4`` and ``max_g_thread <= 0``. The handoff
``kept`` band is intentionally looser by default: ``RUTTER_FCL_TOL=0.2`` and
``RUTTER_G_TOL=0.2`` admit mildly fixable plans while keeping every result in the
``all`` list for inspection. Under a coverage floor a result is also kept only
when every probe sits within 0.01 of the floor (``COVERAGE_FLOOR_TOL``, in
fractions of the threaded best); each result records ``coverage_per_probe`` and
its worst ``coverage_floor_slack``.

IPOPT sees only the rows that can change. Padding and the pair category the
kernel never measures sit at the fill value of 1e3; left in, their
complementarity dwarfs every live row's, and the adaptive barrier update raises
μ by orders of magnitude from any start its first step cannot fully accept,
throwing the iterate far from the start. ``RUTTER_DROP_DEAD_ROWS`` (default on)
removes them. Clearance rows read distances no larger than 1.5 mm
(``CLEARANCE_CAP_MM``): a lookup outside a body's SDF grid returns the sentinel,
so an uncapped row jumps by 1e3 when its last sample leaves the grid, which
IPOPT reads as a violation. Rows that remain can still sit far from their bound,
a gain-100 clearance row at its cap at 130 against a threading row's ~0.5, and
IPOPT's default of starting every bound multiplier at 1 makes those rows the
starting complementarity; ``RUTTER_IP_BOUND_MULT_INIT`` defaults to
``mu-based``, which starts each at μ over its slack. ``RUTTER_IP_MU_MAX`` bounds
μ and defaults to IPOPT's own 1e5.

Trading Objective For Feasibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RUTTER_IP_ACC_COMPL_TOL`` (default ``1e-2``) bounds the complementarity leg of
IPOPT's acceptable-point test. Raising it to the same scale as ``ip_acc_tol``
lets a solve stop at an acceptable point much sooner, and that is a deliberate
trade rather than a free win, because the two axes behave differently along an
IPOPT trajectory:

* The objective descends monotonically and is still descending when a solve
  exits, so every iteration the looser tolerance cuts costs objective on every
  solve that would have converged anyway.
* Feasibility does not. A path that enters restoration leaves the feasible set
  and can return an infeasible iterate at the iteration cap having been feasible
  for much of the solve. Accepting earlier captures one of those iterates.

So loosen it when more plans matter more than each plan's quality, and leave it
alone when the reverse holds. The looser setting is not a substitute for raising
``RUTTER_P2_ITER``: it stops solves sooner, it does not make them converge. Do
not loosen ``ip_cvtol`` for the same purpose — its constraint vector is
mixed-unit, and ``1e-2`` there admits overlap that FCL then reports as a
collision.

The final ``ranked`` list is MMR-ranked: high post-Phase-2 coverage is balanced
against similarity to already selected hole assignments. This keeps the handoff
set useful to a human reviewer instead of returning many nearly identical
plans.


Emit Plans
----------

``rutter-emit`` is pure reconstruction. It loads the handoff, applies each saved
pose to a mesh-free planning state, reorders arcs/probes for rig readability,
and writes plan-only YAML files plus a decision tree and manifest.

A pose's offsets and depth are measured from the point the solver aimed at.
Under ``RUTTER_RETRO_DENSITY`` that is the mean of the target region's masked
cells rather than the config's target, so emit writes it as each probe's inline
target; emit a retro run without the variable and every probe lands off by the
gap between the two.

It also measures each plan, which is the one step that loads the meshes. The
objective's coverage is divided by a ceiling that depends on the probe kind, so
it cannot compare a quad base with a single shank on one region, nor two
configs that split the regions over different probes. ``manifest.json``
therefore carries, per plan, ``recorded``: for each region, the share of its
points within ``RUTTER_RECORDED_RADIUS_MM`` (0.2) of any probe's active sites.
The points are the region's retro cells when ``RUTTER_RETRO_DENSITY`` is on and
its voxels otherwise, and the manifest's ``measure`` says which. Beside it,
``recorded_mm3`` is the volume of each region's voxels within the same radius,
retro or not, which a series schedule's region minimum reads. A region is one
of a probe's ``regions``, or the probe's name without them, and is scored only
if the config has its ``structure:<region>`` asset; emit names the ones it
could not score. ``EmitSettings(recorded_radius_mm=None)`` skips the measure.

Run directly:

.. code-block:: bash

   RUTTER_CONFIG=examples/837229-config.yml \
   RUTTER_HOLES=0283-300-04.holes.yml \
   RUTTER_HANDOFF=scratch/837229_phase2_handoff.json \
   RUTTER_OUTDIR=scratch/837229_plans \
   JAX_PLATFORMS=cpu uv run --python 3.13 rutter-emit

Outputs:

.. code-block:: text

   scratch/837229_plans/
     manifest.json
     manifest.md
     tree.txt
     plans/
       plan-01-cov17.44-....plan.yml

Open a generated plan with the interactive planner:

.. code-block:: bash

   uv run rutter-plan examples/837229-config.yml \
     --plan scratch/837229_plans/plans/plan-01-cov17.44-....plan.yml


Key Modules
-----------

* ``optimization.pipeline.records``: TypedDict/dataclass boundaries for the
  stage payloads and callable bundles.
* ``optimization.pipeline.candidates``: visibility-atlas cache handling and MRV
  hole/arc enumeration.
* ``optimization.pipeline.phase1``: production Phase-1 driver: seed cache,
  spin restore, batched RProp, resume, and Phase-1 payload writing.
* ``optimization.pipeline.fixtures``: bounds, fixture/brain SDF construction,
  coverage data, and shared utility helpers.
* ``optimization.pipeline.phase2``: Phase-2 selection, solver dispatch,
  FCL/threading gate, and MMR ranking.
* ``optimization.pipeline.emit``: handoff-to-plan reconstruction and
  manifest/tree emission.
* ``optimization.pipeline.subject``: subject/runtime setup facade used by the
  pipeline.
* ``optimization.assignment.assignments``: lightweight discrete assignment
  carriers shared by the enumerator, static builders, and batched objectives.
* ``optimization.assignment.arc_placement``: bounded isotonic AP placement for
  separated arc seeds.
* ``optimization.assignment.allocation``: which regions share a probe and which
  kind each probe is, from the share a free line through a bore records. It
  runs before any solve and is NumPy only; ``optimization.pipeline.propose``
  feeds it a config's regions and probes, and ``rutter-series propose`` writes
  the result (see :doc:`series`).
* ``optimization.search.minimizers``: batched and chunked JAX objective
  builders. This is where per-candidate packed statics become reusable vmapped
  kernels.
* ``optimization.search.spin_restore``: the round-robin spin-basin search.
* ``optimization.objectives.statics``: per-candidate static geometry builder.
* ``optimization.objectives.feasibility`` and ``optimization.objectives.slacks``:
  the feasibility spec and one slack function per condition, shared by every
  stage.
* ``optimization.objectives.soft`` and ``optimization.objectives.constrained``:
  the differentiable objectives and constraints Phase 1 and Phase 2 use. The
  terms both phases compute identically — reading the variable vector, the
  comfort-range pull-back, the intra-arc ML separation and the normalized
  coverage — live once in ``optimization.objectives.terms``.
* ``optimization.validation.fcl``: ground-truth FCL validation.
* ``optimization.geometry``: hole geometry, rig kinematics, and the
  ``ProbeStaticInfo`` input carrier.
* ``optimization.clearance``: SDF builders and alpha-wrap envelopes, plus the
  JAX terms over them — ``poses``, ``smooth``, ``grids``, ``boxes``, ``pairs``
  (signed clearance per representation), ``aggregate`` (the category gains
  and the dual-representation combination) and ``sweep`` (the per-pair query
  both phases optimise, vmapped over pairs and fixtures).
* ``scripts/run_subject_overnight.sh``: recommended unattended production
  wrapper.
