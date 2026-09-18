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

The survey below is a snapshot of commit `a5420cc`. Step 1 of the proposed
sequence has since landed (commits `5a94195`–`14e1b81`): the console-script,
runtime-build, trame-app and Phase-2 smoke tests, `tests/architecture/` with its
shrinking baselines, a synthetic subject at `tests/synthetic_subject.py`, and the
parity harness at `scripts/parity_phase2.py`. The suite went from 499 to 557
tests. Two findings under "Tests and documentation" are fixed by that work and
are marked where they appear. Everything else stands as written. The six
decisions that gate step 2 are answered at the end; the rest are still open.

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
| D1 | The JAX compile-cache directory depends on import order | `_setup_compile_cache` sets `scratch/jax_p2_cache` (`phase2_ipopt.py:105-118`). The next import in `_init` loads `objectives.reduced_jax`, which resets it to `/tmp/aind_rutter_jax_cache` at import (`reduced_jax.py:39-44`). | Production Phase 2 uses `/tmp`, not the intended directory. Removing a redundant `_init` call switched which cached kernels loaded, the likely cause of the unexplained 13/40 trajectory shift. Testable by pinning the directory. | verified (traced) |
| D2 | The Phase-2 compile-cache key omits data compiled into the kernel | `_JIT_CACHE` is keyed on shapes, a weights tuple, ceilings and dtype (`phase2.py:895-900`). `_build_jit` also bakes in coverage targets, fixture and brain grid values, and `coverage_n_samples`. `_weights_key` omits `lambda_unit_circle` (`phase2.py:237-265`). | A second `run()` in one process, with the same shapes but a different subject or well mode, reuses stale kernels | verified |
| D3 | `_init` resets `cov_data` but not the `cov_norm` cache | `phase2_ipopt.py:158` vs `:187` | A second `run()` reuses the previous subject's coverage ceilings | verified |
| D4 | Loading a plan never swaps a probe's mesh when its kind changes | `apply_plan_model_to_state` dispatches `SetProbeKind` directly (`runtime/export.py:348`). Only `TrameController._on_probe_kind_change` swaps `node.asset_key`, and it returns early once the kinds match (`trame_controller.py:1234-1250`). | The rendered and collision mesh keep the old kind while readouts use the new one | verified code path; not reproduced |
| D5 | Rig export relabels arcs in the live session state | `app.py:157` passes `store.state` to `export_plan_geometry`, which calls `reorder_plan_for_rig` (`runtime/export.py:186`). That function mutates the state in place (`:117-133`) without a dispatch. | Exporting silently renames arcs in the open session and notifies no subscriber | verified |
| D6 | Inline canonicalization transforms crash | `resolve_transform_ref_cached` returns `TransformChain.composed_transform`, a `(R, t)` tuple (`runtime/transforms.py:83`, `core.py:89`); `canonicalize.py:51` then calls `.rotate_translate` on it | `AttributeError` when a config uses an inline transform | verified |
| D7 | `.npy` sources name a loader that does not exist | `config.py:43` infers `numpy_points`; no loader registers that name | Any `.npy` source fails at build | verified |
| D8 | File-based calibrations always fail validation | `config.py:1440-1442` forbids `reticle` with `file:`; `config.py:2003-2008` requires every calibration file's reticle to exist | Legacy `calibrations.files` entries with `file:` are unusable | verified |
| D9 | The app and the optimizer resolve targets in opposite priority | `planning.py:264-275` checks the inline RAS point first; `runtime/probe_context.py:36-58` checks `target_key` first | A plan setting both gets different targets in the app and the optimizer | verified |
| D10 | Phase 1 and Phase 2 parse shared settings differently | `phase1_pool.py:144` tests `COV_NORM == "1"`, while `Phase2Settings` parses booleans. Phase 1 reads `CONFIG`; `Phase2Settings` prefers `RUTTER_CONFIG`. | `COV_NORM=true` normalizes Phase 2 only; an exported `RUTTER_CONFIG` sends the phases to different subjects | verified |
| D11 | The optimizer ignores a configured probe pivot | Nothing under `optimization/` reads `pivot_LPS`; the app honors it (`planning.py:405`) | Latent: every current config leaves it null | verified |
| D12 | The thread-pool Phase 2 ignores `settings.warmup` | `solve_candidates` warms unconditionally on the thread branch | `WARMUP=0` has no effect in the default pool mode | verified |
| D13 | CI has not run a test since at least June 2026 | CI tests 3.9 (`.github/workflows/ci-call.yml:24`); `pyproject.toml:9` requires `>=3.10`; CLAUDE.md says 3.13 is required. Every leg of run 33395088563 failed during setup: 3.9 on uv refusing an interpreter below `requires-python`, 3.13 on a stale `uv.lock` under `--locked`. | Three conflicting statements of the floor, and no leg of the matrix reaches pytest | verified (run log) |
| D14 | A fresh install cannot start the app (fixed, `0ae6d0d`) | `trame_controller.py:18` imports `pyvista.trame.ui`; pyvista's `trame/__init__.py` imports `trame_pyvista` unconditionally, and pyvista declares it only under its `jupyter` extra. The project depends on plain `pyvista>=0.46.5`. | `rutter-plan` fails to import on any resolution that picks pyvista 0.49; only the stale `uv.lock`, pinning 0.47.1, hides it | verified (3.11 and 3.14 installs) |
| D15 | The declared Python floor is impossible (fixed, `0ae6d0d`) | `pyproject.toml:9` requires `>=3.10`; `orientation_codes.py:3` imports `enum.StrEnum`, added in 3.11 | On 3.10 the package does not import; 25 test modules fail to collect | verified (measured) |
| D16 | `ruff check` does not run the configured rule set (fixed, `c8566ed`) | The installed ruff enables 415 rules with no config at all, and `[tool.ruff.lint]` uses `extend-select`, which adds to that default rather than replacing it | 497 findings in `src` under the documented lint command; with `select` in place of `extend-select`, zero | verified (measured) |
| R1 | The app's FCL manager is shared across threads without a lock | The worker thread mutates the manager (`collisions.py:287-301`) while the kind-change path uses it on the main thread. `FCLBackend.sync` omits group and mask (`fcl_backend.py:63-79`). | Possible race and mis-filtered new nodes | reported |
| R2 | The AP/ML readout skips angle clamping | `trame_controller.py:2106-2122` copies `planning.py:291-307` without `clamp_angles` | Sliders can show unclamped values | reported |
| R3 | Default opacities bypass the material override path | `trame_controller.py:2053-2063` sets actor opacity directly; any repaint restores the config value | Opacity resets on collision flips | reported |
| R4 | The CCF overlay ignores lateralized and descendant labels | `ccf_overlay.py:66` meshes `volume == label_id` | Region overlays disagree with config-declared CCF assets | reported |
| R5 | Two caches are keyed on object `id()` | `_SDF_JNP_CACHE` (`probe_static.py:74-97`), `_SDF_PACK_CACHE` (`phase1.py:812`) | Stale hits after garbage collection; unbounded growth | reported |
| R6 | The phases optimize against different fixture sets | Phase 1 soft objective: well only (`phase1_pool.py:275,346`). Phase 1 FCL: no implant (`:750`). Phase 2: all fixtures, and FCL includes the implant (`phase2_ipopt.py:140-147`). | May be intended; see decisions | verified code; intent open |
| R7 | Bound definitions disagree | The coverage-ceiling AP bounds ignore head pitch while `phase1_bounds` shifts by it (`coverage.py:517` vs `phase1_geometry.py:55`). The sx/sy bounds are ±1.1 in one place (`phase1_geometry.py:59`) and ±1.5 in another (`batched_static.py:283`). | Inconsistent feasible regions between stages | reported |

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
| `AIND_JAX_CACHE_DIR` | `objectives/reduced_jax.py:39`, at import |

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
   D3–D13, each with a regression test.
3. **Delete dead code,** after the keep/delete decisions.
4. **Unify configuration.** Settings for every stage, `jax_env`, no import-time
   environment reads below the CLI.
5. **Consolidate duplicated domain math** behind parity tests: pose, pivot,
   variable layout, target resolution.
6. **Move and rename,** bottom-up with `git mv`: `domain` and `config`, then
   `build` and `plan_io`, then `session` and the frontends, then the optimization
   subpackages, then `pipeline`. Each move updates imports, docs and the baseline
   lists in the same commit.
7. **Consolidate the documentation.**

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

### Before step 3 (dead code)

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
