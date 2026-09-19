# aind_rutter architecture survey

This document maps the `aind_rutter` package as it stood on 2026-09-17 at commit
`a5420cc`. Its purpose is to prepare a renaming and refactoring pass: the code was
built MVP-first, and no one had reviewed its global structure. It records what the
package is, the defects the survey turned up, the structural problems behind them,
a proposed target architecture, and the decisions that proposal depends on.

The findings come from two sources. An AST pass over all 87 modules measured the
import graph, cross-module imports of private names, import-time side effects,
environment-variable reads and unreferenced definitions. Six read-only surveys
covered the frontends, the core model and runtime, the objectives, the
sdf/geometry/enumeration layer, the pipeline and `scripts/`, and the tests and
docs. Where this document says **verified**, the claim was re-checked against the
source after the surveys reported. Everything under "Proposed" is a proposal for
review, not a decision.

## Status

**Status: closed.** All seven steps of the proposed sequence have landed. The
survey below is a snapshot of commit `a5420cc`, so the module names and paths it
cites are those of the snapshot rather than of the tree today; the step records
that follow give the current spelling. What the survey found and did not fix is
in `dev/TODO.md`, which is where a reader should go next.

**Step 1** (commits `5a94195`–`14e1b81`) added the console-script,
runtime-build, trame-app and Phase-2 smoke tests, `tests/architecture/` with its
shrinking baselines, a synthetic subject at `tests/synthetic_subject.py`, and the
parity harness at `scripts/parity_phase2.py`.

**Step 2** fixed every verified defect but one. Each row of the table below
carries the commit that closed it. Three defects, D14 to D16, were found while
answering decision 6 and are recorded here rather than in the original survey's
numbering. D13 is half done: the Python floor, the CI matrix and the lock are
fixed, and what remains is watching a run go green, which needs a push to `main`
or a manually dispatched run.

**Step 3** deleted the dead code, once decisions 7 to 9 were answered: delete
the Jupyter/K3D frontend, delete unused capability rather than maintain it, and
nothing outside the repo imports this package. `src/` lost 2,809 lines and gained
507, and the module count went from 87 to 84. What went, and why:

| gone | commit |
|---|---|
| the Jupyter/K3D frontend, with `k3d` and `ipyevents` | `8b6d4ea` |
| per-candidate objectives, chunked spin restore, `probe_kinematics`, the non-batched pairwise clearance family, the abandoned headstage hull | `b4482aa` |
| `objectives/density.py` and the numpy capsule/oval primitives the jax kernels replaced | `f05212b` |
| `tricubic_sdf` and the `interp` parameter threaded through eleven signatures | `7aae1d5` |
| `CollisionOverlay`, `StoreSubscriber`, `as_transformable`, `Scene.by_tag`, `planning.Probe`, the legacy pivot callback, `coupled_axes` | `6065d48` |
| Phase 1's FCL top-K, which production already disabled | `f1dd474` |
| the `moment_restart` and `adam_const` minimizers, their comparison moved to `dev/POOL_RUN_CONFIGS.md` | `ea50fa6`, `ed06cfd` |
| `RUTTER_BODY_CLEARANCE=uniform` | `0f92a54` |

Two survey entries were **wrong**, and are kept rather than deleted. The Phase-2
process pool is not unused capability: it carries a VRAM preflight and MPS
support, is measured at about 2× the thread pool, and ran the 837772 re-run.
`NodeInstance.locked_axes` was not speculative dead code either — its consumer
existed and read around it. `runtime/build.py` computed which axes a calibrated
probe locks, the trame sliders grayed themselves from their own state variable,
and both slider handlers restated the condition inline. The four copies
disagreed: `build.py` locked on `calibrated` alone while the other three required
a calibration to be loaded, so a probe declaring `calibrated` with no calibration
file got grayed-out controls that were live. `locked_axes_for` is now the only
statement of it (`2d10499`).

**The config vocabulary rework** was not in the original sequence. It came out of
decision 8: the survey listed `OptionsModel` and "the unused `Capability` flags"
as dead code, and asking whether integration would solve a real problem turned up
that the flag vocabularies had lost coherence. Six mechanisms answered variants of
one question — `Kind`, `Role`, `Capability`, asset `tags`, `scene_tags`, and
`collision.group`/`mask`. `Role` was the broken one, asked both what a feature is
*for* and where its coordinates came *from*, which are independent: an annotation
centroid and a bore centre are both targets and only the first is water-localized.

The split now is **closed enums for classifications, tags for groupings**,
because a mistyped tag loads silently while a mistyped enum is refused:

| was | is | commit |
|---|---|---|
| `role` + `chem_shift_policy` + `chem_shift_apply_by_role` | `mr_signal: water\|fat\|none`, required whenever `imaging` is set | `5a3eee3` |
| `caps` (6 flags, 1 read) | `collidable: bool` | `3be090c` |
| `collision.group` / `collision.mask` label lists | a rule: both collidable, at least one `role: probe` | `3be090c` |
| `RECORDING_GEOMETRY` as the only source | a probe asset declares `recording`, the table as defaults | `8f690b4` |
| asset `tags`, written and never read | unioned onto the generated scene node | `e871665` |
| `OptionsModel` | gone | `e871665` |

Every migration was verified rather than asserted: behaviour computed from the
pre-migration configs and from the migrated ones agrees on every chemical-shift
decision and ppm, every collidability, and all 169 collision pairs.
`tests/config_semantics.json` pins that. `scripts/upgrade_config.py` brought an
older config forward, accepting the result only if none of it moved; it lives at
commit `e7e345c`, the last one whose models can read the fields it migrates.

The suite went 499 → 671 (step 2) → 730. Two findings under "Tests and
documentation" are fixed by step 1 and are marked where they appear. Everything
else stands as written.

**Step 4** gave every stage typed settings. Phase 1 read 21 values as module
globals at import, so its importable entry point could not be told anything;
`emit` read three more. Both now have `run(settings)` beside a `main()` that
builds the settings from the environment, as Phase 2 has had since it was
written, and `restore`, `enumeration` and `thick_well` take the settings rather
than reading for themselves.

The architecture baseline went from 54 recorded environment reads to 28. **Every
import-time read that remains is a jax platform or allocator variable** —
`JAX_PLATFORMS`, the `XLA_PYTHON_CLIENT_*` pair, and Phase 2's `PLATFORM`,
`POOL`, `THREADS` and `GPU_MEM_FRACTION` — which must be set before jax loads and
so cannot come from an object built inside `main`. The default subject path, found
in seven places by the survey, is now in one.

Two result-changing values are still read where they are used:
`THREADING_MARGIN_MM` in `geometry.holes` and `RETRO_DENSITY` in
`pipeline.probe_setup`. Both sit below the pipeline layer, so surfacing them
means passing settings down into `geometry` and `objectives` — a layering change
that belongs with steps 5 and 6 rather than with configuration.

Since nothing in the suite runs a Phase-1 pool, the conversion was checked by
resolution rather than outcome: `tests/phase1_env.py` records what each global
made of a given environment, read out of the module before the change, and the
settings object is held to it field by field.

One thing step 4 did not fix: **the `HOLES` default points into the gitignored
`scratch/`**, so a fresh clone cannot run the pipeline without being told where
the bore file is.

**Step 5** consolidated the duplicated domain math, each change behind the
Phase-2 parity harness where it touched the solver. Four of the eight rows in
"Domain math is implemented several times" had already been closed by steps 2 and
3; the rest:

| was | is |
|---|---|
| the pivot derived in four places, and a helper that looked the centre up from the built-in table whenever a caller omitted it | `pivot_from_shank_tips`, pure geometry, centre supplied by the caller |
| the per-probe variable count declared five times | one definition |
| the reduced stride written as a bare `3` at 21 sites | `objectives/layout.py`, which owns both layouts |
| `_weights_key` hand-written in three places | `objectives/cache_keys.weights_cache_key`, from `dataclasses.fields` |
| the named shank tip and the world brain mesh, once in the app and once in the export | `runtime.shanks.named_shank_tip_world`, `runtime.scene_geometry.brain_world_mesh` |

**Two of these were live defects rather than tidying.** `lambda_unit_circle` is
read inside both the Phase-1 and the reduced traced objectives and was in
neither's compile-cache key, so a kernel traced at one value was silently reused
at another — the same defect D2 fixed for Phase 2, never propagated. It is a
basin-selection knob, so the wrong value changes which candidates converge. And
`pivot_from_shank_tips` gave `probe_context` a different answer for an
unregistered kind than the four sites that resolved the geometry themselves.

The pose row closed as a test rather than a consolidation: the three builders
already agreed on the tip to 1e-6, which nothing had checked. They part company
only outside the rig's angular limits, where the app clamps through
`Kinematics.clamp_angles` and the two optimizer builders do not; feeding the
clamped angles to the optimizer reproduces the app exactly, and that is asserted.

`_pack_statics` stays as two copies. They pack genuinely different layouts, and
merging them now that the layouts are named would mean a layout-generic packer —
more abstraction than two sites justify.

`caps`, `collision`, `chem_shift_policy`, `chem_shift_apply_by_role` and the
`role` prefix inference are gone from the models (`ae32732`): the 113 local
configs carrying them were migrated and verified first, and a config that still
states one is refused by name. `2f91f9e` retired the unprefixed environment
names; every variable the pipeline reads is now `RUTTER_` plus the field's own
name, derived by `env_prefix` rather than listed per field.

**Step 6** moved and renamed, bottom-up, each commit green:

| commit | what moved |
|---|---|
| `3ba0539` | `domain/` — transforms, enums, catalog, scene, and `planning.py` split into `rig`, `plan` and `pose`. Probe-kind data moved in from `optimization/`, so the viewer no longer pulls in the optimizer. |
| `285a104` | `config.py` (2,511 lines) split into seven modules under `config/` |
| `cef55c4` | `runtime/` split into `build/` and `plan_io/`; the `build_runtime.py` shim deleted |
| `7f1c3d9` | `session/`, `render/`, `collision/`, `web/`, `ccf/` |
| `d6590a8` | the optimizer renamed for what each module does: `sdf/`→`clearance/`, `enumeration/`→`assignment/`, `objectives/phase1`→`soft`, `phase2`→`constrained`, plus `search/` and `validation/` |
| `c2c5a26` | decision 10: `ProbePlan` holds `offsets_LP` and `target_point_LPS` |

Two deviations from the proposed layout, both because the dependency direction
did not hold: the MRV enumerator stays in `pipeline/` (as `candidates.py`)
because it reads the atlas cache, builds the subject and takes settings; and
`optimization/problem/` was not created, because `variables` imports the
objective it lays out. The layer rule now covers every solver subpackage rather
than `objectives` alone.

`trame_controller.py` (2,131 lines), `clearance/kernels.py` (1,490),
`objectives/constrained.py` (1,007) and `objectives/soft.py` (911) moved whole.
Splitting them is the "Monoliths" finding below, not a move.

**Step 7** consolidated the documentation on decision 15, splitting it by who
reads it:

| commit | what moved |
|---|---|
| `f54f0aa` | the five short heading underlines and the one malformed table that step 6's renames left in the Sphinx build |
| `141f7d5` | `CONTRIBUTING.md`, which was still the generated template telling contributors to run `coverage`, `flake8`, `black` and `isort` |
| `c4cdbbf` | the concepts tour, the coordinate rule, the config model and the config vocabulary into `docs/source/`, rendered by myst-parser; `dev/PIPELINE.md` folded into the optimizer guide it pointed at |
| `490c1aa` | the module tree and the five layer rules into `docs/source/architecture.rst`, each rule naming the test that enforces it; `dev/MODULE_MAP.md` deleted as superseded |
| `9b1abb5` | five unbuilt designs into `dev/proposals/` and one superseded note into `dev/archive/`, with a `dev/README.md` saying what each lane is for |
| `5af2c63` | `CLAUDE.md` from 175 lines to 79: the invariants and the traps, with setup, style and the module tree removed to where they belong |

Fixing the documented commands turned up two defects of their own. `mypy` named
seven modules the moves had renamed, so it checked one file and passed
vacuously; repointed, it found two diagnostic keys written onto
`Phase2ResultRecord` and never declared (`b674758`). The `slow` pytest marker
was declared, deselected by default and used by no test, so the comment saying
it held back the heavy JAX tests was false.

**The monolith splits** followed step 7, each verified the way its risk
demanded: the trame controller against a dump of every attribute the class
resolves, the two solver files against the Phase-2 parity harness.

| commit | what split | from | to |
|---|---|---|---|
| `890a16d` | `web/controller.py` | 2,133 | 845, plus layout, readouts, materials, camera and keybindings |
| `15de049` | `clearance/kernels.py` | 1,490 | poses, smooth, grids, boxes, pairs, aggregate |
| `aabbe39` | `objectives/soft.py`, `constrained.py` | 911 and 1,007 | 835 and 857, plus sdf_inputs, rewards, slack_layout, evaluate |

The two objective modules stayed large on purpose. Each is now mostly one
closure factory — `_build_jit`, 386 lines in `soft` and 472 in `constrained` —
that captures the packed statics and returns the traced objective. Splitting
that means giving every closed-over array an explicit parameter, which changes
the traced code rather than moving it, and the parity harness is the only thing
that would catch a mistake.

**What is left.** D13's second half — nobody has yet watched a CI run go green
— and the seven reported defects R1 to R7, which are code readings with no
reproduction. Decisions 10 to 16 are answered below.

## The package today

`aind_rutter` has two halves that share a domain model:

- **The interactive planner** loads a YAML config, builds a scene, and lets a user
  place probes in a Trame web app.
- **The offline optimizer** searches for probe placements in three batch stages.

| area | lines | what it is |
|---|---|---|
| `core`, `common`, `orientation_codes`, `assets`, `scene`, `planning`, `commands` | ~1,300 | Domain model: transforms, enums, the asset catalog, the scene graph, plan state with rig limits and pose math, and plan-edit commands |
| `config.py` | 2,493 | Pydantic schema, bulk expansion, template merge, reference checks and YAML loading |
| `runtime/` | 2,810 | Assembles config into catalog, scene and plan state, plus plan import/export and several adapters for the optimizer; despite the name, it is not the live runtime |
| `state_change.py` | 136 | `PlanStore`, the live state frontends drive, plus a worker-thread helper |
| `app`, `trame_controller`, `rendering`, `pyvista_backend`, `collisions`, `fcl_backend`, `ccf_overlay`, `cli` | ~3,800 | Trame web frontend and its render/collision adapters |
| `controllers`, `k3d_backend` | 562 | Jupyter/K3D frontend; it still imports, but nothing constructs it |
| `optimization/sdf` | 3,208 | Probe envelopes, voxel SDF grids with disk caches, JAX clearance kernels, padded sweep tables |
| `optimization/geometry` | 1,047 | Hole and wall types, the probe-kind recording table, FCL BVH construction, a numpy pose formula |
| `optimization/enumeration` | 755 | Visibility atlas, arc placement and ML/spin seeding; the enumerator itself lives in `pipeline/enumeration.py` |
| `optimization/objectives` | 5,646 | JAX objective kernels for the reduced, soft (Phase 1) and constrained (Phase 2) problems, coverage, FCL validation |
| `optimization/pipeline` | 5,012 | Stage drivers `rutter-phase1`, `rutter-phase2` and `rutter-emit`, with shared assets, settings and file formats |

### Dependency direction

Most imports point from the pipeline down through objectives to sdf and geometry,
and from the frontends down to runtime and the domain model. The exceptions are the
layering violations listed under "Structural findings".

Two facts shape any restructuring:

- **Laziness hides package-level cycles.** `planning`, `runtime/build` and
  `trame_controller` import `optimization.geometry.recording` inside functions,
  while the optimizer imports `planning` at import time.
- **Importing a module has side effects.** 16 modules import jax at import time. 7
  pipeline modules call `os.environ.setdefault("JAX_PLATFORMS", ...)`, with
  different defaults.

## Defects found during the survey

These change behavior today, or will once someone reaches the path. D14–D16 came out of checking the Python floor for decision 6. They are
listed apart from the structural findings because each should be fixed on its own,
before the code around it moves.

| # | defect | evidence | effect | status |
|---|---|---|---|---|
| D1 | The JAX compile-cache directory depends on import order (fixed, `ac07108`) | `_setup_compile_cache` sets `scratch/jax_p2_cache` (`phase2_ipopt.py:105-118`). The next import in `_init` loads `objectives.reduced_jax`, which resets it to `/tmp/aind_rutter_jax_cache` at import (`reduced_jax.py:39-44`). | Production Phase 2 uses `/tmp`, not the intended directory. Removing a redundant `_init` call switched which cached kernels loaded, the likely cause of the unexplained 13/40 trajectory shift. Testable by pinning the directory. | verified (traced) |
| D2 | The Phase-2 compile-cache key omits data compiled into the kernel (fixed, `5cac789`) | `_JIT_CACHE` is keyed on shapes, a weights tuple, ceilings and dtype (`phase2.py:895-900`). `_build_jit` also bakes in coverage targets, fixture and brain grid values, and `coverage_n_samples`. `_weights_key` omits `lambda_unit_circle` (`phase2.py:237-265`). | A second `run()` in one process, with the same shapes but a different subject or well mode, reuses stale kernels | verified |
| D3 | `_init` resets `cov_data` but not the `cov_norm` cache (fixed, `5cac789`) | `phase2_ipopt.py:158` vs `:187` | A second `run()` reuses the previous subject's coverage ceilings | verified |
| D4 | Loading a plan never swaps a probe's mesh when its kind changes (fixed, `7d7043b`) | `apply_plan_model_to_state` dispatches `SetProbeKind` directly (`runtime/export.py:348`). Only `TrameController._on_probe_kind_change` swaps `node.asset_key`, and it returns early once the kinds match (`trame_controller.py:1234-1250`). | The rendered and collision mesh keep the old kind while readouts use the new one | verified code path; not reproduced |
| D5 | Rig export relabels arcs in the live session state (fixed, `55e00f2`) | `app.py:157` passes `store.state` to `export_plan_geometry`, which calls `reorder_plan_for_rig` (`runtime/export.py:186`). That function mutates the state in place (`:117-133`) without a dispatch. | Exporting silently renames arcs in the open session and notifies no subscriber | verified |
| D6 | Inline canonicalization transforms crash (fixed, `88ab16f`) | `resolve_transform_ref_cached` returns `TransformChain.composed_transform`, a `(R, t)` tuple (`runtime/transforms.py:83`, `core.py:89`); `canonicalize.py:51` then calls `.rotate_translate` on it | `AttributeError` when a config uses an inline transform | verified |
| D7 | `.npy` sources name a loader that does not exist (fixed, `fd096a6`) | `config.py:43` infers `numpy_points`; no loader registers that name | Any `.npy` source fails at build | verified |
| D8 | File-based calibrations always fail validation (fixed, `3d393ec`) | `config.py:1440-1442` forbids `reticle` with `file:`; `config.py:2003-2008` requires every calibration file's reticle to exist | Legacy `calibrations.files` entries with `file:` are unusable | verified |
| D9 | The app and the optimizer resolve targets in opposite priority (fixed, `1a71823`) | `planning.py:264-275` checks the inline RAS point first; `runtime/probe_context.py:36-58` checks `target_key` first | A plan setting both gets different targets in the app and the optimizer | verified |
| D10 | Phase 1 and Phase 2 parse shared settings differently (fixed, `a699886`) | `phase1_pool.py:144` tests `COV_NORM == "1"`, while `Phase2Settings` parses booleans. Phase 1 reads `CONFIG`; `Phase2Settings` prefers `RUTTER_CONFIG`. | `COV_NORM=true` normalizes Phase 2 only; an exported `RUTTER_CONFIG` sends the phases to different subjects | verified |
| D11 | The optimizer ignores a configured probe pivot (fixed, `3d7cb05`) | Nothing under `optimization/` reads `pivot_LPS`; the app honors it (`planning.py:405`) | Latent: every current config leaves it null | verified |
| D12 | The thread-pool Phase 2 ignores `settings.warmup` (fixed, `a9f8f60`) | `solve_candidates` warms unconditionally on the thread branch | `WARMUP=0` has no effect in the default pool mode | verified |
| D13 | CI has not run a test since at least June 2026 (half fixed, `0ae6d0d`) | CI tests 3.9 (`.github/workflows/ci-call.yml:24`); `pyproject.toml:9` requires `>=3.10`; CLAUDE.md says 3.13 is required. Every leg of run 33395088563 failed during setup: 3.9 on uv refusing an interpreter below `requires-python`, 3.13 on a stale `uv.lock` under `--locked`. | Three conflicting statements of the floor, and no leg of the matrix reaches pytest. The floor, the matrix and the lock are fixed; nobody has yet watched a run go green, which needs a push to `main` or a manually dispatched run. | verified (run log) |
| D14 | A fresh install cannot start the app (fixed, `0ae6d0d`) | `trame_controller.py:18` imports `pyvista.trame.ui`; pyvista's `trame/__init__.py` imports `trame_pyvista` unconditionally, and pyvista declares it only under its `jupyter` extra. The project depends on plain `pyvista>=0.46.5`. | `rutter-plan` fails to import on any resolution that picks pyvista 0.49; only the stale `uv.lock`, pinning 0.47.1, hides it | verified (3.11 and 3.14 installs) |
| D15 | The declared Python floor is impossible (fixed, `0ae6d0d`) | `pyproject.toml:9` requires `>=3.10`; `orientation_codes.py:3` imports `enum.StrEnum`, added in 3.11 | On 3.10 the package does not import; 25 test modules fail to collect | verified (measured) |
| D16 | `ruff check` does not run the configured rule set (fixed, `c8566ed`) | The installed ruff enables 415 rules with no config at all, and `[tool.ruff.lint]` uses `extend-select`, which adds to that default rather than replacing it | 497 findings in `src` under the documented lint command; with `select` in place of `extend-select`, zero | verified (measured) |
| R1 | The app's FCL manager is shared across threads without a lock | The worker thread mutates the manager (`collisions.py:287-301`) while the kind-change path uses it on the main thread. `FCLBackend.sync` omits group and mask (`fcl_backend.py:63-79`). | Possible race and mis-filtered new nodes | reported |
| R2 | The AP/ML readout skips angle clamping | `trame_controller.py:2106-2122` copies `planning.py:291-307` without `clamp_angles` | Sliders can show unclamped values | reported |
| R3 | Default opacities bypass the material override path | `trame_controller.py:2053-2063` sets actor opacity directly; any repaint restores the config value | Opacity resets on collision flips | reported |
| R4 | The CCF overlay ignores lateralized and descendant labels | `ccf_overlay.py:66` meshes `volume == label_id` | Region overlays disagree with config-declared CCF assets | reported |
| R5 | Two caches are keyed on object `id()` | `_SDF_JNP_CACHE` (`probe_static.py:74-97`), `_SDF_PACK_CACHE` (`phase1.py:812`) | Stale hits after garbage collection; unbounded growth | reported |
| R6 | The phases optimize against different fixture sets | Phase 1 soft objective: well only (`phase1_pool.py:275,346`). Phase 1 FCL: no implant (`:750`). Phase 2: all fixtures, and FCL includes the implant (`phase2_ipopt.py:140-147`). | May be intended; see decisions | verified code; intent open |
| R7 | Bound definitions disagree (AP half fixed, `a19047d`) | The coverage-ceiling AP bounds ignore head pitch while `phase1_bounds` shifts by it (`coverage.py:517` vs `phase1_geometry.py:55`). The sx/sy bounds are ±1.1 in one place (`phase1_geometry.py:59`) and ±1.5 in another (`batched_static.py:283`). | Inconsistent feasible regions between stages. The AP half was real but small: Gaussian coverage is rotationally symmetric about the target so its ceiling cannot move, and a KDE ceiling moves 0.39% on a slab tilted 75°, 0.003% at 60°, nothing below. The sx/sy half (±1.1 vs ±1.5) is open. | AP half fixed |

The weight fallbacks the objectives survey flagged are inert. `getattr(weights,
"lambda_unit_circle", 100.0)` never falls back, because both weight classes declare
the field with default 10. One consequence is real, though: `JointWeights` has no
`lambda_clearance_fixture`, so in the reduced objective that weight always equals
`lambda_clearance` (`batched_reduced.py:167`).

## Structural findings

### Configuration is scattered and partly hidden

The package reads 49 environment variables. Phase 2 has typed settings; Phase 1,
emit and enumeration read module-level globals at import. Lower layers read
variables that change results without appearing in any settings object:

| variable | read at |
|---|---|
| `THREADING_MARGIN_MM` | `geometry/holes.py:60`, per call |
| `RUTTER_BODY_CLEARANCE` | `sdf/build.py:44`, at import |
| `COVERAGE_WEIGHTS` | `runtime/probe_context.py:74` |
| `RETRO_DENSITY` | `pipeline/probe_setup.py:48` |
| `AIND_JAX_CACHE_DIR` | `optimization/jax_env.py`, on request (was `objectives/reduced_jax.py:39`, at import) |

Other problems in the same area:

- **The default subject paths are repeated in 7 places,** and the `HOLES` default
  points into the gitignored `scratch/`.
- **The pool kind is fixed at import.** `POOL`, `PLATFORM`, `THREADS` and
  `GPU_MEM_FRACTION` are read before jax loads, so the importable `run()` cannot
  choose thread or process.
- **"pool" means four things:** the Phase-1 output, the `POOL` executor variable,
  `multiprocessing.Pool`, and the driver's path variable.

### Import-time side effects

- **JAX platform.** Seven pipeline modules set a default `JAX_PLATFORMS`.
  `enumeration.py` and `phase1_build.py` default to `cpu`; `phase1_pool`,
  `restore` and `phase2_ipopt` default to `cuda`. The console entry points set the
  variable first, so the setdefaults are no-ops there. A Python caller gets
  whichever module it imported first.
- **Compile cache.** `objectives/reduced_jax.py` configures the process-wide JAX
  compile cache at import, which is the cause of D1.
- **Eager `__init__` imports.** `optimization/sdf/__init__.py` imports `kernels`,
  so `import aind_rutter.optimization.sdf.build` loads jax.
  `optimization/objectives/__init__.py` does the same through `phase1`, so the
  pure-numpy `density` module cannot load without jax.
- **Dead facade.** `optimization/__init__.py` is a 33-name lazy facade that
  nothing in `src`, `scripts` or `tests` imports.

### Domain math is implemented several times

| concept | copies |
|---|---|
| Probe pose from optimizer variables | JAX (`sdf/kernels.py:116-159`), numpy (`geometry/kinematics.py:39-78`, used only by the FCL gate), and the app (`planning.ProbePose.from_planning_state`). Parity tests cover rotation only (`tests/test_export_kinematics_parity.py`); nothing tests the JAX tip position. |
| Probe pivot | `runtime/build.py:242-274`, `enumeration/visibility_atlas.py:92-107`, `objectives/probe_static.py:117-128`, `objectives/batched_static.py:236-247` |
| Target resolution | `planning.py:264`, `runtime/probe_context.py:36`, `trame_controller.py:2021-2028`, with differing priority (D9) |
| Named shank tip, world brain mesh, depth ray | `runtime/export.py:188-250` and `trame_controller.py:711-778` |
| Variables per probe (6) | `objectives/variables.py:17`, `objectives/phase1.py:109`, `objectives/clearance_metrics.py:20`, `pipeline/restore.py:23`, `pipeline/phase2_diagnostics.py:20` |
| Reduced layout (3 per probe) | Hand-coded at about 20 sites in `batched_reduced`, `spin_restore`, `batched_static`, `phase1_pool` and `restore` |
| Statics packing | `_pack_statics` twice (`reduced_jax.py:461-580`, `phase1.py:815-897`); `_signature` three times |
| Config capability coercion | `_coerce_caps` six times in `config.py`, plus `runtime/build.py:144` |

### Layering violations

- **objectives → pipeline.** `objectives.phase2` and `objectives.spin_restore`
  import `pipeline.contracts` for return annotations, creating a package cycle.
- **App core → optimizer.** `planning.py:408`, `runtime/build.py:254` and
  `trame_controller.py:914` import the probe-kind recording table from
  `optimization.geometry.recording`.
- **Runtime builds unused optimizer data.** `runtime/build.py:284` builds a
  convex hull for every probe (`AssetSpec.headstage_hull`) that no code reads. It
  is the only reason the app imports `fcl`-dependent optimizer geometry at startup.
- **Runtime → interaction layer.** `runtime/export.py` and `runtime/plan_csv.py`
  import `PlanStore` and `commands`.
- **Optimizer adapter in runtime.** `runtime/probe_context.py` is an optimizer
  input adapter copied field-for-field into `ProbeStaticInfo`.
- **Frontend bypasses the store.** The frontend mutates scene nodes outside the
  store (`trame_controller.py:366, 668, 1231, 1257`). It also reaches into PyVista
  backend internals (`_actors`, `_highlighted`, `_flush_callback`) and imports
  runtime's private `_depth_along_probe_axis`.

### The public surface is wrong in both directions

- **69 private-name imports cross module boundaries.**
- **Runtime exports too much.** `runtime/__init__.py` lists 78 names in `__all__`,
  19 of them private; about 25 are used outside `runtime/`. `build_runtime.py`
  re-exports them again, and three tests, `app.py`, a script and the Sphinx docs
  import through that shim.
- **The objectives' working API is private.** Its public API is largely dead. The
  pipeline builds on `phase1._build_jit`, `_pack_statics`, `_signature`,
  `probe_static._build_probe_static`, `variables._poses` and
  `variables._apply_x_to_plan_state`.

### Dead code

Verified by reference search across `src`, `scripts` and `tests`:

- **`objectives/`, about 900 lines:**
  - `reduced_jax.make_jax_reduced_objective` and its private builders
  - `phase1.make_phase1_objective` and the layout converters
  - `spin_restore.make_batched_spin_restore_chunked`
  - `batched_static.initial_y_from_aa`
  - `variables.extract_spins`
  - `FCLValidator.is_feasible`
  - the `objectives/__init__` re-exports
- **`sdf/kernels.py`, about 360 lines:** the `pairwise_signed_clearance*` family
  and its jit wrappers. `tricubic_sdf` is unreachable because no caller passes
  `interp=`.
- **Geometry:** `geometry/probe_kinematics.py` (entirely), the headstage hull, and
  the numpy capsule API (`Capsule`, `shank_capsules_from_pose`,
  `pose_at_hole_best_fit`, `required_ap_deg`, used only by tests).
- **Pipeline:**
  - `phase1_build.make_batched_phase1_objective`, `make_adam`, and the
    unreachable cosine schedules
  - `phase1_geometry.build_fixture_collision_objs` and `final_feasibility_report`
  - `OptimizationRuntime.from_env`
  - the `pairwise` enumeration mode
- **Frontends:** `rendering.CollisionOverlay` and `CollisionOverlayStyle`, the
  unused `OverlayState` mutators, several collision helpers, and
  `state_change.StoreSubscriber`. The Jupyter frontend (`controllers.py`,
  `k3d_backend.py`) has no constructor anywhere; README, CLAUDE.md and the Sphinx
  architecture page still advertise it.
- **Core:** `config.select_from_resource`, `OptionsModel`, `planning.Probe`,
  `PoseResolver.get_pivot_for_asset`, `Kinematics.coupled_axes`, and
  `NodeInstance.locked_axes` (written, never read).

Two of the AST pass's dead-code candidates are live. `runtime/loaders._load_trimesh`
is registered as the `trimesh` loader. `config.expand_config` is documented public
API.

### Hidden and global state

- **Phase-2 worker state.** Workers share state through the module dict `_G`
  (`phase2_ipopt.py:102`). D2 and D3 follow from caching in it and in module-level
  compile caches.
- **Phase-1 module state.** Phase 1 reads 24 module constants inside its functions
  and keeps a mutable `_group_log_once`, and it has no `run(settings)`.
- **Frontend state.** `trame_controller` keeps an undeclared `_readout_state`,
  read sometimes through `getattr` fallbacks and sometimes directly.

### Names describe history rather than role

- **Pipeline:**
  - `phase1_pool` is named for its output.
  - `phase2_ipopt` is named for one of its two solvers.
  - `restore.py` holds no restore code.
  - `phase1_geometry` also serves Phase 2.
  - `pipeline/enumeration.py` shadows the `optimization/enumeration` package, and
    that package does not enumerate.
- **Objectives:**
  - `phase1` and `phase2` collide with the pipeline stage modules.
  - `reduced_jax` mostly holds the threading kernel.
  - `batched_reduced` and `batched_static` merge two independent axes into one
    name.
  - "static" names four different types.
- **sdf:** `ProbeSDF` also holds fixture and brain grids.
- **Core:**
  - `runtime/` is config assembly, not the runtime.
  - `export.py` also imports plans.
  - `core`, `common` and `state_change` say nothing about their contents.
  - "Kinematics" means rig limits in `planning` and pose formulas in the optimizer.
- **Leftovers of the package rename:** the environment variable
  `AIND_LOW_POINT_CACHE_DIR`, and docstrings citing flat modules that no longer
  exist.

### Monoliths

| file | lines | seams |
|---|---|---|
| `config.py` | 2,493 | Leaf models; template models; single and bulk specs; scene models; plan and rig models; root model and loading; a 445-line validator; the template merge engine; effective canonicalization |
| `trame_controller.py` | 2,140 | Layout and inline JS (~550 lines); bootstrap and state wiring (~380); intent-to-command (~320); derived geometry and readouts (~270); overlays and materials (~170); camera, picking and highlight (~150) |
| `sdf/kernels.py` | 1,751 | Pose; variable parameterization; smooth math; grid interpolation; box geometry; clearance categories; constraint gains and aggregation |
| `objectives/phase1.py` | 1,086 | Kernel builder, packer, SDF data types, rewards, dead wrappers |
| `objectives/phase2.py` | 1,021 | Objective, slack groups, Jacobians and Hessians, padded-row mask |

### Tests and documentation

- **36 of the 87 modules have no direct test import.** A coverage snapshot from
  2026-09-16 shows 0% for `trame_controller`, `phase1_pool`, `phase2_ipopt`,
  `emit`, `app`, `collisions` and `rendering`. `runtime/build.py` is at 16%. No
  test runs a stage driver, a frontend, or `build_runtime_from_config` end to end;
  D4 through D8 sit in that gap. *(Closed by step 1: each of those now has a
  smoke test over a synthetic subject.)*
- **The one structural test fails open.** `tests/test_optimization_pipeline_contracts.py`
  matches banned helpers by name and never asserts that the names it checks exist,
  so renaming a banned helper disables its rule without a failure. It also omits
  `thick_well.py`, which calls a banned helper. *(Step 1 added the existence
  assertions; the missing `thick_well.py` is a defect fix, still open.)*
- **A rename will break these tests:**
  - private-name imports in `test_hole_walls`, `test_live_constraint_rows`,
    `test_objectives_threading`, `test_phase2_diagnostics` and `test_reducers`
  - shim imports in `test_round_trip`, `test_reducers` and
    `test_export_rig_arc_consistency`
  - the old cache variable in `test_clearance_samples` and
    `test_fixture_surface_sampling`
  - environment variable names in `test_pipeline_settings`
- **The `slow` marker is dead config.** It is declared and deselected by default,
  but no test uses it.
- **Architecture prose lives in three overlapping places,** each stale differently:
  CLAUDE.md, `dev/MODULE_MAP.md` (predates `runtime/`, no optimization section),
  and `docs/source/architecture.rst`.
- **CLAUDE.md contains false statements:**
  - a test count of 406 (557 today) *(corrected)*
  - `dev/PIPELINE.md` described as a read-verified stage map (it is a 32-line
    pointer)
  - a citation of `dev/optimizer_plan.md` (does not exist)
- **Docs still describe pickle files.** `optimization.rst` and
  `pipeline/contracts.py` do.
- **`examples/` mixes three kinds of file:** 9 tracked inputs, 86 gitignored run
  outputs, and 4 untracked rig-export files that validate against neither config
  model. Every tracked config points at `/mnt/vast` or `/mnt/Data`, and no test
  loads any of them.

## Proposed target architecture

### Layer rules

Each rule is meant to be enforced by a test under `tests/architecture/`.
Package-level rules start from a baseline list of today's exceptions that can
only shrink.

1. `domain` imports nothing else from `aind_rutter`.
2. `config` imports only `domain` (enums and value types).
3. `build` and `plan_io` import `domain` and `config`.
4. `session` imports `domain`, `build` and `plan_io`; never a frontend or
   `optimization`.
5. `render`, `collision`, `web` and `jupyter` import `session` and below; never
   `optimization`.
6. Inside `optimization`, `problem` is jax-free. `objectives`, `clearance`,
   `kinematics`, `assignment` and `search` never import `pipeline`, and `pipeline`
   is the only package that reads settings.
7. Only settings modules, `jax_env` and the CLI shims read environment variables,
   and only CLI shims read them at import.
8. No module imports another module's private names.
9. No import cycles.
10. A named set of modules imports without loading jax, checked by subprocess.

### Package layout

```
aind_rutter/
  domain/
    transforms.py     ← core.py (AffineTransform, TransformChain, Transformable)
    enums.py          ← common.py + orientation_codes.py
    catalog.py        ← assets.py (+ Material)
    scene.py          ← scene.py
    rig.py            ← planning.py: limits, Kinematics, kinematic_violations, head pitch
    plan.py           ← planning.py: ProbePlan, PlanningState, resolve_probe_angles, resolve_target_lps
    pose.py           ← planning.py ProbePose/PoseResolver + runtime/shanks.py + export tip/depth helpers
                        (the one numpy pose; the JAX twin is parity-tested against it)
    probe_kinds.py    ← optimization/geometry/recording.py
    commands.py       ← commands.py
  config/             ← config.py split: models_common, models_catalog, models_scene,
                        models_plan, expand, templates, resolve, root (validation passes)
  build/              ← runtime/: assemble (build), loaders, reducers, canonicalize, chem_shift,
                        calibration (+ calibration_conversion), transforms, queries (scene_geometry)
  plan_io/            ← runtime/export.py split: roundtrip, replay, rig_export (non-mutating);
                        cli_csv (plan_csv)
  ccf/                ← ccf_ontology.py + CCF label utilities from runtime/loaders.py
  session/            store (PlanStore), worker (AsyncLatestWorker), factory (build_session),
                      readouts, intents (incl. probe-kind swap), warnings
  render/             protocol, adapter, overlays (rendering.py), pyvista (pyvista_backend.py)
  collision/          adapter, worker (collisions.py), fcl (fcl_backend.py)
  web/                cli, app, controller, layout, keyboard, camera, ccf_regions (ccf_overlay.py)
  jupyter/            widget (controllers.py), k3d_backend — only if kept
  optimization/
    problem/          jax-free: layout (full and reduced variable layouts, bounds), weights,
                      statics types, data types, constraint protocols, assignments
                      (← enumeration/contracts.py), plan_apply (← variables._apply_x_to_plan_state)
    kinematics/       JAX rotation, pose, pivot and spin parameterization (← sdf/kernels.py pose parts)
    geometry/         holes (+ HoleSection, cap_basis), fcl (make_fcl_bvh), probe_inputs (probes.py)
    clearance/        ← sdf/: cache, voxel_sdf (build), envelope, samples, interpolate, boxes,
                      smooth, pair_clearance, sweep (clearance_sweep)
    assignment/       ← enumeration/: atlas, visibility_atlas, arc_placement, seed_emission,
                      mrv (← pipeline/enumeration.py Enumerator)
    objectives/       terms/{threading, clearance, penalties}, coverage, packing,
                      reduced (batched_reduced), soft (phase1), constrained (phase2), metrics
    search/           spin_restore, minimizers (← pipeline/phase1_build.py)
    validation/fcl.py ← objectives/fcl_validator.py
    jax_env.py        the one place that sets the JAX platform and compile cache
    pipeline/
      cli.py          jax-free entry points: settings → jax_env → lazy stage import
      settings.py     Pipeline, Execution, Phase1, Phase2 and Emit settings
      payloads.py     file formats (unchanged)
      records.py      ← contracts.py
      subject.py      ← runtime_adapter.py + probe_setup.py + runtime/probe_context.py
      fixtures.py     ← phase1_geometry.py + thick_well.py
      phase1/         stage, seeding, restore, atlas_cache
      phase2/         stage, worker (_G → WorkerContext), selection, ranking, handoff, diagnostics
      emit/           stage, summaries
```

Deleted:
- `build_runtime.py` and `pipeline/restore.py`
- `geometry/probe_kinematics.py`
- `AssetSpec.headstage_hull`
- `optimization/__init__.py` facade content
- the dead code listed above, pending the keep/delete decisions below

### Tests, docs and examples

- **Tests** mirror the package layout. They gain `tests/architecture/` for the
  layer rules. Before any move, they also gain:
  - import and `--help` smoke tests for all five console scripts
  - a CPU `phase2.run` test on synthetic records
  - a headless `build_trame_app` test
  - a synthetic `build_runtime_from_config` test
  - the Phase-2 parity harness, tracked
- **Docs** consolidate into one `dev/ARCHITECTURE.md` that states the layer rules
  and names the test enforcing each. It absorbs `MODULE_MAP.md`, the module
  sections of `architecture.rst` and CLAUDE.md's tree. CLAUDE.md shrinks to
  invariants, commands and pointers. Historical and future-design notes move to
  `dev/archive/` and `dev/proposals/` with status headers.
- **Examples** keep one synthetic config that runs anywhere, validated by a test.
  Launch scripts move to `examples/launchers/`, and run outputs leave the tree.

## Proposed sequence

Each step is behavior-preserving except the defect fixes, and each ends green. Any
step touching a solver path also runs the Phase-2 parity harness against the
previous commit.

1. **Safety net.** Add the smoke tests above and the architecture tests with
   baseline exception lists, and track the parity harness. Nothing moves yet.
   *(Done — see Status.)*
2. **Fix the verified defects.** Start with D1 and D2: pinning the compile cache
   also makes runs reproducible, and D2 makes the importable `run()` safe. Then
   D3–D13, each with a regression test. *(Done, apart from D13's second half —
   see Status.)*
3. **Delete dead code,** after the keep/delete decisions. *(Done — see
   Status. The config vocabulary rework came out of it.)*
4. **Unify configuration.** Settings for every stage, `jax_env`, no import-time
   environment reads below the CLI. *(Done — see Status.)*
5. **Consolidate duplicated domain math** behind parity tests: pose, pivot,
   variable layout, target resolution. *(Done — see Status.)*
6. **Move and rename,** bottom-up with `git mv`: `domain` and `config`, then
   `build` and `plan_io`, then `session` and the frontends, then the optimization
   subpackages, then `pipeline`. Each move updates imports, docs and the baseline
   lists in the same commit.
7. **Consolidate the documentation.** *(Done — see Status.)*

## Decisions needed

### Before step 2 (defect fixes)

1. Is R6 intended? Phase 1 optimizes against the well only and runs FCL without
   the implant; Phase 2 uses every fixture and includes the implant.
2. Should rig export relabel arcs in the open session, or only in the exported copy (D5)?
3. May a loaded plan change a probe's kind (D4)? This decides whether the kind swap
   belongs in the session layer.
4. Should the optimizer honor a configured `pivot_LPS`, or should the field go
   (D11)?
5. For `.npy` sources (D7): register a `numpy_points` loader, or map the extension
   to an existing one?
6. What is the supported Python floor (D13)?

#### Answers

Recorded 2026-09-17, with the evidence each rests on. Step 2 proceeds on these.

1. **Split the difference: the soft objectives keep it, the FCL gate loses it.**
   Confirmed intended by the author — the implant is not needed in Phase 1
   because the threading terms already constrain the bore. The soft objective
   over the well alone (`phase1_pool.py:275,346`) stays, with the fixture set
   becoming an explicit setting rather than a hard-coded one-tuple.

   Phase 1's FCL check gains the implant, so that both phases compute the field
   named `fcl` the same way. Nothing gates on the Phase-1 value: it is written
   for the top `FCL_TOPK` by soft clearance, left NaN elsewhere, printed as a
   count, and every candidate enters the pool either way (`phase1_pool.py:745-770`).
   It is off entirely in production — `run_subject_overnight.sh:101` sets
   `FCL_TOPK=0` — so the fix changes a diagnostic column, and the pool's poses
   stay bitwise identical. Whether to keep that diagnostic at all is decision 8.

   One trap sits behind it: `select_by` names any pool field, so `SELECT_BY=fcl`
   sorts on NaN for every record outside the top-K, and `rank_order`'s missing-key
   sentinel does not apply to a key that is present and NaN. Make `rank_order`
   treat NaN as the worst value.

   **Superseded, 2026-09-17:** the check goes instead, which also answers its
   line in decision 8. Production has run with `FCL_TOPK=0` throughout, so no
   pool on disk carries a value, and a diagnostic nobody runs is not worth the
   fixture set it would need. Deleting it removes the per-candidate validator
   loop, the `FCL_TOPK` knob and the feasible-count line. It does not remove
   `PoolRecord.fcl`: that field is required by the pool schema, so removing it
   would make every existing pool unreadable. Make it optional, stop writing it,
   and leave old files loading as they do.

2. **Only in the exported copy.** `export_plan_geometry` is a reporter, and the
   relabel is cosmetic by its own docstring, so it has no business editing the
   session's domain state. It also edits without a dispatch, so `PlanStore`
   subscribers never hear about it: the UI keeps the old arc labels while the
   state holds new ones, and the next Save writes the relabelled arcs. Make
   `reorder_plan_for_rig` pure — take a state, return a reordered one — and let
   both callers use the return value. `emit.py:175` already deep-copies per plan
   and is unaffected. Relabelling the live session, if anyone wants it, is a
   separate dispatched command.

3. **Yes, and the swap belongs in the session layer.** `kind` is a required field
   of `ProbeDeclModel`, so every plan-only YAML carries one, and
   `planning_state_to_plan_model` writes whatever the session holds — a plan saved
   after a UI kind change carries the new kind and loads into a session that keeps
   the old mesh. `SetProbeKind`'s docstring assigns the mesh swap to the
   TrameController, and that assignment is the defect: `apply_plan_model_to_state`
   is shared by the app's plan load and `rutter-plan-csv`, and neither goes
   through the controller's dropdown handler. One operation updates the plan and
   the scene node's `asset_key` together. Exports and the CSV are unaffected —
   `export.py:220` resolves `probe:{plan.kind}` from the plan, not from the node —
   so the damage is confined to what is drawn and what collides.

4. **Honor it; the field stays.** The optimizer recomputes the pivot with the
   formula that built it — the shank tips' mean x and y with the kind's
   `active_center_mm` for z — in `probe_static.py:119` and
   `batched_static.py:238`, both copies of `runtime/build.py:266`. Reading
   `AssetSpec.pivot_LPS` instead makes the override work, gives the app and the
   optimizer one source, and deletes two copies of the same arithmetic, which step
   5 would otherwise have to consolidate anyway. The optimizer keeps its present
   computation as the fallback for a null pivot, which is what every config has
   today, so the change is bitwise identical on current data and the parity
   harness can prove it.

5. **Register `numpy_points`.** No loader can be mapped to instead: `csv_points`
   parses CSV through pandas and nothing else reads arrays. The name is already
   the documented default in `EXTENSION_DEFAULTS`, in two model docstrings and
   throughout `tests/config_factories.py`, so the contract exists and only the
   implementation is missing. Load with `allow_pickle=False` and require a float
   `(N, 3)`: a pickled `.npy` executes arbitrary code on load, which is the thing
   the JSON payload work removed from this pipeline.

6. **Floor 3.11, current 3.13, both tested.** The policy is to test the declared
   floor and a current release, which needs the declared floor to be true — and
   it is not. `requires-python = ">=3.10"` is contradicted by
   `orientation_codes.py:3`, which imports `enum.StrEnum`, added in 3.11: on 3.10
   the package does not import at all (25 collection errors, measured). On 3.11
   the suite reaches 550 passed against jax 0.10.2, and the failures that remain
   are D14 and a test assertion since fixed, not language incompatibilities.

   3.14 cannot be the upper leg yet. `python-fcl` does now ship cp314 wheels, so
   CLAUDE.md's stated reason is stale, but `scikit-image` (pulled in by
   `trimesh[recommend]`) and `mesh2sdf` do not, and both fail to build from
   source. Add 3.14 when those wheels land.

   So: `requires-python = ">=3.11"`, a CI matrix of `["3.11", "3.13"]` — quoted,
   because YAML reads an unquoted 3.10 as 3.1 — and a refreshed `uv.lock` so
   `--locked` passes. Carrying 3.11 means carrying two jax generations, which the
   Phase-2 solve is not bitwise identical across; if that is unwelcome, the
   alternative is a 3.13 floor and a single leg until 3.14 is installable.

### Before step 3 (dead code) — answered

**7. Delete the Jupyter/K3D frontend.** It had no constructor anywhere and no
test, so it could drift from the runtime unnoticed.

**8. Delete unused capability rather than maintain it** — with two exceptions
found by asking, per the question below, whether integration would solve a real
problem. The Phase-2 process pool and `NodeInstance.locked_axes` stay; see
Status. `OptionsModel` and the `Capability` flags led to the config vocabulary
rework rather than a straight deletion.

**9. Nothing outside this repo imports it**, so no move needs a compatibility
shim.

The list below is the question as it was asked.


7. Keep or delete the Jupyter/K3D frontend, along with its `k3d` and `ipyevents`
   dependencies and docs?
8. Keep or delete each unused capability:
   - the numpy capsule API
   - `make_phase1_objective` and the per-candidate reduced objective
   - the chunked spin restore
   - the density mixture/KDE
   - tricubic interpolation
   - `RUTTER_BODY_CLEARANCE=uniform`
   - the `moment_restart` and `adam_const` minimizers
   - Phase 1's FCL top-K (off in production)
   - the Phase-2 process-pool mode
   - legacy calibration files
   - `resources` / `from_resource` selectors
   - `OptionsModel`
   - the unused `Capability` flags
9. Do notebooks or scripts outside the repo import `build_runtime`, `rendering`,
   `collisions` or `state_change`? The answer decides whether moves need
   compatibility shims.

### Before steps 4–6 — answered

**10. Move `ProbePlan` to LPS.** Only two fields hold RAS —
`target_point_RAS` and `offsets_RA` — and both convert at first use. Narrow, but
it changes command signatures, so it goes with step 6.

**11. The optimizer stays separable.** Probe-kind data and pose math move to
`domain/`. Today `assets.py` and `runtime/build.py` import
`optimization/geometry/recording`, so launching the viewer pulls in the
optimizer; that is what the split ends.

**12. Descriptive stage names, and the proposed package layout stands.** The
console scripts keep their `rutter-phase1` / `rutter-phase2` spelling.

**13. Retire the unprefixed environment names**, once the configs are migrated.
`scripts/run_subject_overnight.sh` passes 38 of them and changes in the same
commit.

**14.** Resolved by the vocabulary rework: the collision rule is `collidable`
plus `role: probe`, and the group/mask labels are gone.

**15. Split by audience.** `CONTRIBUTING.md` takes setup, test and lint
commands, style and PR conventions; `docs/source` takes the stable reference
(`COORDINATES`, `CONFIG_MODEL`, `VOCABULARY`, `PIPELINE`); `dev/` keeps the
dated working notes and design records; `CLAUDE.md` keeps only what an agent
would otherwise get wrong, and shrinks.

**16. The subject configs stay tracked** in `examples/`.

The list below is the questions as they were asked.

### Before steps 4–6 (configuration, consolidation, renames)

10. Should `ProbePlan` store LPS internally, with RAS kept only in YAML and the UI?
    This changes command signatures.
11. Should the optimizer stay separable from the planner? The answer decides
    whether probe-kind data and pose math live in `domain` or in `optimization`.
12. Which vocabulary for the stages: "phase1/phase2", or descriptive names such as
    "soft/constrained"? Do the proposed package names stand: `build`, `session`,
    `clearance`, `assignment`, `search`?
13. Should all environment variables take a `RUTTER_` prefix, retiring the generic
    `POOL`, `N`, `OUT`, `LIMIT` and `MARGIN`? Should `AIND_LOW_POINT_CACHE_DIR`
    keep a deprecated alias?
14. Should the app's collision checking follow the optimizer's tag-based fixture
    set, or stay on configurable group and mask bits?
15. Should `docs/source` host the developer guide, or only user docs?
16. Should subject configs pointing at `/mnt/vast` stay tracked in `examples/`?
