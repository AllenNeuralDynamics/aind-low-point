# Known defects and deferred work

Things found but not fixed, with enough to act on. A fix deletes its entry.

## Defects

### The camera aims where the brain is not

`web/controller.py:1938` (`recenter_view`) frames on
`brain_spec.mesh.raw` — the asset's **pre-scene** mesh. A brain whose scene node
carries `transform: headframe_to_lps` is drawn tens of mm from there, so the
camera's focal point and its bounds refit both miss it. Startup calls this
through `apply_default_view`, so the first thing a user sees can be empty space.

`scripts/render_plan_figure.py:brain_bounds` has the same flaw.

**Fix:** `build.queries.brain_world_mesh(catalog, scene)`, which applies
the scene transform; it already backs the depth and over-insertion readouts.
Leave `fallback_to_raw` off — falling back to file coordinates is the failure
this replaces.

### The bore file default is unreachable

`PipelineSettings.holes` defaults to `scratch/0283-300-04.holes.yml`, which is
gitignored, so a fresh clone cannot run the pipeline without being told where the
file is. Open question: is the bore file a fixed artefact of the implant design
(track it) or regenerated per run (derive it, or fail loudly)?

## Deferred

### `mypy` reports five errors it never used to reach

Its `files` list named seven modules that the package moves had renamed, so
`mypy` checked one file and reported nothing. Repointed at the modules that
exist, it reports five, none of them a defect: a `**dict` expansion into a
`TypedDict`, two calls through a `dict[str, object]` value, a widened
`Path | None` that a model validator always fills, and a float dtype union.
Each wants a narrowed annotation rather than a code change.


### `_pack_statics` is two implementations

`objectives/phase1.py` and `objectives/reduced_jax.py`. Deliberate: they pack
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

`THREADING_MARGIN_MM` in `geometry.holes` and `RETRO_DENSITY` in
`pipeline.probe_setup`. Surfacing them in settings means passing settings down
into `geometry` and `objectives` — a layering change, so it belongs with the
package moves rather than with configuration.
