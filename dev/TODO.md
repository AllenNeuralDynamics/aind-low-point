# Known defects and deferred work

Things found but not fixed, with enough to act on. A fix deletes its entry.

## Defects

### The bore file default is unreachable

`PipelineSettings.holes` defaults to `scratch/0283-300-04.holes.yml`, which is
gitignored, so a fresh clone cannot run the pipeline without being told where the
file is. Open question: is the bore file a fixed artefact of the implant design
(track it) or regenerated per run (derive it, or fail loudly)?

### The spin-component bounds disagree between two stages

`pipeline/fixtures.phase1_bounds` bounds each of `(sx, sy)` to ±1.1;
`objectives/packing` bounds them to ±1.5. Both comments call the figure "loose
around the unit circle", and `unit_circle_penalty` pulls the magnitude to 1
either way, so this changes how far a component may wander before the penalty
dominates rather than which poses are reachable. The remaining half of R7.

### `collide_one_to_many` is unreachable and uses the callback that drops pairs

`CollisionAdapter.collide_one_to_many` and `FCLBackend.collide_one_to_many` have
no callers. The backend one still goes through `fcl.defaultCollisionCallback`,
which stops accumulating once the global contact limit is hit — the reason
`collide_internal` was rewritten around a per-pair callback. Delete both, and
`_pairs_from_contacts` and `Contact` with them, or wire it up and fix the
callback.

## Deferred

### `mypy` reports five errors it never used to reach

Its `files` list named seven modules that the package moves had renamed, so
`mypy` checked one file and reported nothing. Repointed at the modules that
exist, it reports five, none of them a defect: a `**dict` expansion into a
`TypedDict`, two calls through a `dict[str, object]` value, a widened
`Path | None` that a model validator always fills, and a float dtype union.
Each wants a narrowed annotation rather than a code change.


### `pack_statics` is two implementations

`objectives/soft.py` and `objectives/reduced.py`. Deliberate: they pack
different layouts with different padding, and merging them now that
`objectives/layout.py` names both would mean a layout-generic packer — more
abstraction than two call sites earn. Revisit if a third appears.

### The optimizer does not clamp to rig limits

`ProbePose.from_planning_state` resolves angles through
`Kinematics.clamp_angles` (AP ±75°, ML ±42°); the two optimizer pose builders
take whatever they are handed. So a solver can produce a pose the app silently
moves. Enumeration bounds restrict the search, so this is latent rather than
live. `tests/test_pose_parity.py` pins the difference.

### Two result-changing values are read below the pipeline layer

`RUTTER_THREADING_MARGIN_MM` in `optimization/geometry/holes.py` and
`RUTTER_RETRO_DENSITY` in `optimization/pipeline/probe_setup.py`. Surfacing
them in settings means passing settings down into `geometry` and `objectives`,
which the package moves did not do; both are still read where they are used.
They are two of the three entries in `ENV_READ_BASELINE` that are not the JAX
platform and allocator variables. The third is the cache directory, spelled
`AIND_JAX_CACHE_DIR`, `JAX_CACHE_DIR` and `AIND_LOW_POINT_CACHE_DIR` in three
places and `RUTTER_` in none.
