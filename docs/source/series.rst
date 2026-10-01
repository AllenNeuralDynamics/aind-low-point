Planning a Series
=================

A series is several insertion sessions on the same mice: each session records
one set of regions, and the sets share probes, a rig and an order. One file, the
series spec, states what differs between series. ``aind_rutter.series`` turns it
into one planner config per subject, set and allocation.

.. code-block:: python

    from aind_rutter.series.configs import write_configs
    from aind_rutter.series.spec import load_spec

    spec = load_spec("dhc-bwcd.series.yml")
    write_configs(spec, "runs/dhc-bwcd", spec_path="dhc-bwcd.series.yml")
    solve(spec, "runs/dhc-bwcd")        # from aind_rutter.series.solve

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
    The regions of one session, and one or more allocations of probes to them.
    Regions, sets and allocations hold for every subject.

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

What is refused
---------------

At load: an unknown field; a set region that is neither defined nor a CCF
acronym; a ``keep`` naming both of two opposite halves; an allocation that
leaves out a region of its set, records one twice or records one the set does
not list; more probes of a kind than the inventory holds; two probes with one
name; a subject listing a set that does not exist; arc caps that do not rise.

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
