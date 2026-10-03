Planning a Series
=================

A series is several insertion sessions on the same mice: each session records
one set of regions, and the sets share probes, a rig and an order. One file, the
series spec, states what differs between series. ``rutter-series`` turns it into
one planner config per subject, set and allocation, solves each, and picks the
plan for each session and the order of the sessions. Before any of that it can
propose the allocations themselves.

.. code-block:: bash

    rutter-series propose  dhc-bwcd.series.yml runs/dhc-bwcd   # a minute a set, no card
    rutter-series configs  dhc-bwcd.series.yml runs/dhc-bwcd
    rutter-series solve    dhc-bwcd.series.yml runs/dhc-bwcd   # hours, on the card
    rutter-series schedule dhc-bwcd.series.yml runs/dhc-bwcd
    rutter-series status   dhc-bwcd.series.yml runs/dhc-bwcd   # where each run stands

Each command is a function of ``aind_rutter.series``:
``propose.write_proposals``, ``configs.write_configs``, ``solve.solve``, and
``schedule.schedule_subject`` with ``write_schedule``. ``propose``, ``solve``
and ``schedule`` take ``--subject``, and ``propose`` and ``solve`` ``--set``,
to do part of a series.

The spec
--------

.. code-block:: yaml

    name: dhc-bwcd
    hemisphere: left

    subjects:
      template:
        base_config: dhc-base.yml       # relative to this file
        holes: /data/implants/0283-300-06/0283-300-06.holes.yml
        sets: all                       # or the names of the sets this mouse runs
        region_asset: {canonicalization_ref: identity-lps, material_ref: structure}
        region_target: {canonicalization_ref: identity-lps}

    inventory: {quadbase-dovetail: 4, "2.1": 8}

    solve: {max_arcs: [3, 4], topk: 2000, select_by: objective}

    propose: {max_group: 3, min_kept: 0.1, per_count: 1}

    schedule:
      costs: {arc_move: 1, slider_move: 1, spin: 1, mount: 10, arc_mount: 15}
      first: [0-standard, A-broad]    # the first sessions, in either order
      within: {D-thalamic: 3}         # among the first three sessions
      floors: [1.0, 0.95, 0.9]
      min_region_mm3: 0.05            # of every region, or the plan is dropped,
      min_region_fraction: 0.1        # or a tenth of a region too small for that

    regions:
      ACC:    {ccf: ACA}
      vCA1:   {ccf: CA1, keep: [ventral]}
      pDMS:   {ccf: CP, keep: [dorsal, medial, posterior]}
      SI_HDB: {ccf: [SI, NDB]}

    sets:
      D-thalamic:
        regions: [ORBl, MD, SMT, PF, PVT, VM, MOs, pDMS]
        score: centroid               # or retro
        solve: {max_arcs: [4]}        # only what this set states differently
        allocations:
          7p4q:
            - kind: quadbase-dovetail
              regions: [MOs, ORBl]
              points_RAS:
                MOs: [-1.8654, 2.5349, -1.786]
                ORBl: [-1.2525, 2.5768, -2.9882]
            - {kind: quadbase-dovetail, regions: [MD]}
            - {kind: "2.1", regions: [SMT]}
            - {kind: "2.1", regions: [PF]}
            - {kind: quadbase-dovetail, regions: [PVT]}
            - {kind: quadbase-dovetail, regions: [VM]}
            - {kind: "2.1", regions: [pDMS]}

``regions``
    A label and the voxels it stands for: one CCF acronym or several, and the
    halves of their extent to ``keep`` (see *CCF Regions* in
    :doc:`configuration`). A label that is itself a CCF acronym needs no
    entry.

``sets``
    The regions of one session, and the allocations of probes to them. Regions,
    sets and allocations hold for every subject. A set may state no allocation
    yet: ``propose`` reads only its regions, and there is nothing of it to
    solve or schedule until one is stated.

``allocations``
    Each is a list of probes. A probe records its ``regions``: the first anchors
    the pose and the rest are scored beside it. A region is aimed at its
    interior point unless ``points_RAS`` states another, which is how a probe
    shared by two regions is put on the line that passes through both. A probe
    is named after its regions, joined by ``_``, unless it states a ``name``.

``subjects``
    A subject brings its anatomy and its implant. ``base_config`` is a planner
    config with everything but the probes: assets, transforms, a probe asset for
    each kind in use, ``plan.arcs`` and the rig pitch. Any probes it has are
    replaced. ``region_asset`` and ``region_target`` hold what every region's
    structure asset and target need in this subject's config beyond the region
    itself, such as the annotation's ``src`` (``${paths.ccf_annotation_path}``
    unless stated), a ``transform`` and a material.

``inventory``
    How many probes of each kind can be mounted at once.

``score``
    What a set's coverage is taken over: ``centroid``, each region's target
    point, or ``retro``, the labelled cells inside each region. ``retro`` needs
    a ``retro-targets`` point asset in the subject's base config, in that
    subject's frame and uncut; the regions' ``keep`` does the trimming.

``solve``
    What decides a solve's result. ``max_arcs`` lists the arc caps to try in
    turn, a later one running only when the one before it gave no plans.
    ``topk`` is how many Phase-1 candidates Phase 2 refines, ``select_by`` how
    they are ranked (``learned`` needs a ``ranker_table``), and
    ``recorded_radius_mm`` the radius of the recorded share. A set's own
    ``solve`` restates only the fields it names.

``propose``
    How allocations are proposed; see `Proposing allocations`_. Every field
    has a default.

``schedule``
    How the plan for each session and the order of the sessions are chosen; see
    `Scheduling`_. Every field has the default shown above, except that
    ``first`` and ``within`` are empty.

What is refused
---------------

At load: an unknown field; a set region that is neither defined nor a CCF
acronym; a ``keep`` naming both of two opposite halves; an allocation that
leaves out a region of its set, records one twice or records one the set does
not list; more probes of a kind than the inventory holds; two probes with one
name; a subject listing a set that does not exist; arc caps that do not rise;
a ``first`` or ``within`` naming a set that does not exist; a floor outside
(0, 1]; a negative price.

When the configs are written: a base config with no arcs, or one that already
defines a region's structure asset; a set scored on ``retro`` whose base has no
cell cloud; and anything the planner's own config model
refuses, since every config is loaded back as it is written. A kind with no
``probe:<kind>`` asset is caught there.

What is written
---------------

.. code-block:: text

    <run>/series.lock.yml
    <run>/<subject>/configs/<set>/<allocation>.yml

Each config is its subject's base with a ``structure:<label>`` asset per region
of the set, a ``target:<label>`` at each one's interior point, and the
allocation's probes on the base's arcs in turn; the solve reassigns arcs. The
probes name their ``regions`` by label, so retro density and the recorded share
are taken over ``pDMS`` and not over all of ``CP``.

The lock file records the resolved spec, the package version and commit, the
hash of every config written, and the hash of every file the spec and the
configs name: base configs, bore files, meshes, volumes and transforms.

Proposing allocations
---------------------

Which regions share a probe, and which kind each probe is, decides more of what
a session records than anything the solver does afterwards. ``propose`` answers
it from the anatomy, the implant and the probes alone, for every probe count
the set's regions and the bores allow:

.. code-block:: text

    <run>/<subject>/proposed/<set>.yml           the proposals, in the spec's form
    <run>/<subject>/proposed/<set>.regions.yml   the config the regions were read from

A line through a bore, at angles the rig reaches, is scored as emit scores a
solved plan: the share of each region's points within ``recorded_radius_mm`` of
the active stretch of each shank on it. The points are the labelled cells when
the set's ``score`` is ``retro`` and the region's voxels otherwise, sampled
down to ``max_points`` a region. Lines are tried every ``angle_step_deg`` in
both arc angles, a row of shanks every ``spin_step_deg``, and the active
stretch every ``depth_step_mm`` down the line; every shank must thread the bore.

Up to ``max_group`` regions may share a probe when some line keeps, of every
one of them, at least ``min_kept`` of what that region's best line records on
the same kind of probe. A proposal splits the set's regions into such probes,
each on a bore of its own, with the kinds that record the most in all within
the inventory. The best ``per_count`` proposals at each probe count are
written, named ``6p``, then ``6p-2``.

.. code-block:: yaml

    alone:                # the most each region can get, a probe to itself
      MD: {quadbase-dovetail: 0.79, '2.1': 0.5}
    allocations:
      6p:
      - kind: quadbase-dovetail
        regions: [MD, VM]
        points_RAS: {MD: [-0.2284, -0.4216, -3.6113], VM: [-0.7562, -1.0009, -4.4702]}
      - {kind: '2.1', regions: [PL]}
    predicted:
      6p:
        mean: 0.4737
        regions: {MD: 0.3396, VM: 0.657, PL: 0.2541}
        bores: {MD_VM: 3, PL: 10}

To use a proposal, paste it under the set's ``allocations`` in the spec. It is
not read from the run directory: an allocation in the spec is a decision
someone made, and running the proposer again cannot change it. A shared probe
comes with ``points_RAS`` on the line found, the region nearest the middle of
the recorded stretch first.

``predicted`` is what a region can get, not what a plan will give. The line
knows nothing of fixtures or of the other probes, which are the solver's, so a
solved plan records less, and comparing a plan's ``recorded`` with it shows how
much the layout cost each region. The test is loose on purpose: a proposal that
cannot be laid out costs solver time, while a wrongly refused one costs a plan
nobody sees.

Solving
-------

``solve`` runs ``scripts/run_subject_overnight.sh`` once per config, in the
order the spec states them, each into its own directory:

.. code-block:: text

    <run>/<subject>/solve/<set>/<allocation>/
        pool.json.gz  phase2_handoff.json  plans/  manifest.json
        atlas.json.gz  mrv_seeds.json.gz   the geometry caches
        solve.arcs<N>.log                  the wrapper's output at each arc cap
        solve.json                         what was asked, and each attempt

The spec decides the variables that change a result: the config, the bore file,
the arc cap, ``RUTTER_TOPK``, ``RUTTER_SELECT_BY``, ``RUTTER_RETRO_DENSITY`` and
``RUTTER_RECORDED_RADIUS_MM``. Everything else comes from the environment
``solve`` is called in, which is where the machine's side belongs:
``RUTTER_WORKERS``, the chunk sizes, the GPU memory fraction. Any other
``RUTTER_`` variable set there passes through as well and is written to
``solve.json`` with the rest, so a result can be traced to it.

A rerun skips every run whose ``solve.json`` matches what the spec now asks:
the config and bore file by hash, the score and the ``solve`` fields. A run
made under different inputs is redone from nothing, caches and pool included.
A failed attempt ends that run and the series moves on; the next call repeats
it, and Phase 1 resumes from its pool.

Scheduling
----------

Between two sessions the rig is reworked: arcs slide or are swapped, probes
slide, turn, come off and go on. ``schedule`` picks one solved plan for each set
and the order of the sets so that the rework over the series is least.

A set's plans are pooled over every allocation solved for it, so whether a set
runs on six probes or seven is decided here. A plan is scored by its recorded
share averaged over the set's regions, which, unlike the solver's coverage,
compares across probe kinds and counts. For each of ``floors``, a plan
qualifies when it scores at least that share of its set's best plan, and one
schedule is made from the qualifying plans. A lower floor gives up recorded
share for less rework; the report shows each so the trade can be read off.

A mean can hide a region a plan misses. ``min_region_mm3`` drops, before any
floor, every plan reaching less than that volume of some region's voxels within
the recorded radius, and a set's best is then its best plan that remains. It is
a volume and not a share so that a large region is held to the same amount of
tissue as a small one; ``min_region_fraction`` then lowers the bar for a region
too small to give that volume to the fraction of it, so a thin region is not
held to more than it has. A set none of whose plans meets it is refused, naming
the plan that comes closest and how far short it falls. Both are 0, off, by
default.

The report lists, per session, the volume of each region reached and the share
of the set's plans that reach less. A low volume that beats most plans is a
region that is hard to reach, not a poor plan.

``costs`` prices each change, in units of sliding one arc: ``arc_move`` and
``slider_move`` for an arc or a probe kept and slid, ``spin`` for a kept probe
turned, ``mount`` for each probe put on or taken off, and ``arc_mount`` for an
arc put on or taken off, whose probes are charged as mounts as well. Probes of
one kind are interchangeable, and neither arcs nor the probes on an arc pass
each other. Depth and the in-plane offsets are manipulator travel and are not
charged, nor is building the first session's rig.

``first`` names the sets that take the first sessions, in either order among
themselves. ``within`` gives a set the latest session it may be, counted from
1. Both apply to the sets a subject runs.

The search is exact: every order the constraints allow and every qualifying
plan of each set. Of equally cheap schedules, the one recording most is taken.
Each subject is scheduled on its own.

.. code-block:: text

    <run>/<subject>/schedule.json     every floor's schedule: per session the set,
                                      allocation, plan, rig setup, changes and cost
    <run>/<subject>/schedule.md       the same as tables, with the command that
                                      opens each session in the planner
    <run>/<subject>/schedule/floor-<f>/<session>-<set>-<allocation>-plan-<n>.plan.yml

The files under ``schedule/`` are copies of the chosen plans, numbered by
session. They are what the report's ``rutter-plan`` commands load and save, so
a plan adjusted by hand is kept apart from the solved one; a copy that differs
from its solved plan is left as it is by a later ``schedule`` and named in the
report. The sessions are not one plan set for the planner's Plans tab, since
each has its own config and probes.

Only runs solved under what the spec now states are pooled. The others are
listed with their state, the same one ``status`` prints: ``no config``,
``pending``, ``stale``, ``failed`` or ``done``. A set with no solved plans is
refused, and so are plans with no recorded share or one that leaves out a
region of the set, whose mean would not be the set's.
