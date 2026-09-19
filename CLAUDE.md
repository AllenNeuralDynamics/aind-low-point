# CLAUDE.md — Rutter (`aind-rutter`)

Multi-probe insertion planning for AIND: an interactive planner and a
constrained-optimization solver over one runtime. The frontend is Trame +
PyVista (`web/app.py` + `web/controller.py`).

## Invariants

- **Internal canonical space is LPS millimeters.** RAS appears only at named
  user-facing boundaries — the config's `point_RAS` / `offsets_RA`, the
  sliders, the rig export — and converts at the planning boundary. `ProbePlan`
  holds `offsets_LP` and `target_point_LPS`. See `dev/COORDINATES.md`.
- **Develop on 3.13; the floor is 3.11.** Use `uv run --python 3.13 ...` for
  everything. CI tests the floor and 3.13. 3.11 is what `enum.StrEnum` needs;
  3.14 waits on cp314 wheels for `scikit-image` and `mesh2sdf`, whose source
  builds fail (`python-fcl` has them now).
- **Models are the source of truth.** When tests disagree with `config/`,
  fix the tests.
- **Classifications are enums, groupings are tags.** A config states
  `mr_signal`, `role`, `kind` and `collidable` as closed fields, so a
  misspelling is refused at load; `tags`/`scene_tags` are open and only draw a
  warning on a near miss. `mr_signal` decides chemical shift and is required on
  every asset and target whenever `imaging` is set. See `dev/VOCABULARY.md`.
  The fields these replaced — `caps`, `collision`, `chem_shift_policy`,
  `chem_shift_apply_by_role`, `options` — are refused by name; `e7e345c` is the
  last commit whose `scripts/upgrade_config.py` can migrate a config that has
  them.
- **Every pipeline stage is told, not configured by import order.**
  `phase1.run(Phase1Settings())`, `phase2.run(recs, Phase2Settings())`
  and `emit.run(EmitSettings())` take typed settings; constructor arguments
  outrank the environment. Every variable the pipeline reads is `RUTTER_` plus
  the field's name, upper-cased; the bare spellings are gone, because `CONFIG`,
  `OUT`, `LIMIT`, `N`, `WORKERS`, `POOL` and `PLATFORM` are set for unrelated
  reasons in ordinary shells. The only import-time environment reads left are
  the jax platform and allocator variables, which must precede the jax import.
- **What a config resolves to is pinned.** `tests/config_semantics.json` records
  every tracked config's per-asset chemical-shift decision and ppm and
  collidability, plus its scene nodes, fixture set and collision pairs.
  Regenerate with `uv run --python 3.13 python -m tests.config_semantics` only
  when a behaviour change is the point.

## Commands

```bash
ruff check                                # lint
ruff format                               # format
uv run --python 3.13 pytest -q            # tests (837 currently green)
uv sync --python 3.13                     # set up venv

# Phase-2 output parity against a baseline commit, on a subject written on the
# spot. Exits non-zero on any difference. Add --config/--holes/--poses for a
# real subject, --platform gpu to run on the card.
uv run --python 3.13 python scripts/parity_phase2.py --baseline HEAD~1
```

`tests/architecture/` holds the structural rules — import cycles, dependency
direction, private-name imports, environment reads, and which modules import
without jax. Each carries a baseline of today's exceptions that may only shrink:
fixing one means deleting its line, and a rule fails if a listed exception stops
violating it.

## Where things live

```
src/aind_rutter/
├── domain/                # the planning domain (imports no config, optimizer or UI)
│   ├── transforms.py      # AffineTransform, TransformChain, *Transformable
│   ├── enums.py           # Role, Kind, MRSignal, OrientationCode; KNOWN_SCENE_TAGS
│   ├── catalog.py         # AssetSpec, TargetSpec, AssetCatalog, Material
│   ├── scene.py           # NodeInstance, Scene
│   ├── rig.py             # JointRange, PoseLimits, Kinematics, AP/ML_LIMIT_DEG
│   ├── plan.py            # ProbePlan, PlanningState, resolve_target_LPS, locked axes
│   ├── pose.py            # ProbePose, PoseResolver, shank-tip detection
│   ├── probe_kinds.py     # RecordingGeometry per probe kind
│   └── commands.py        # Planning commands + apply_planning_command
├── config/                # the config DSL, split by what each model describes
│   ├── models_common.py   # sources, materials, transforms, imaging, paths
│   ├── models_catalog.py  # asset and target specs, single and bulk, templates
│   ├── models_scene.py    # SceneNodeModel, SceneModel
│   ├── models_plan.py     # arcs, probes, calibrations, head mount
│   ├── templates.py       # the template merge
│   ├── resolve.py         # effective canonicalization and transform
│   └── root.py            # ConfigModel and its cross-reference validation
├── build/                 # config → runtime (see below)
├── plan_io/               # reading a plan back out
│   ├── roundtrip.py       # planning_state_to_plan_model, save_plan_to_config
│   ├── replay.py          # apply_plan_model_to_state
│   ├── rig_export.py      # export_plan_geometry, reorder_plan_for_rig
│   └── cli_csv.py         # the rutter-plan-csv entry point
├── session/
│   ├── store.py           # PlanStore: one planning state, edited by command
│   └── worker.py          # AsyncLatestWorker
├── render/
│   ├── overlays.py        # OverlaySpec, OverlayState, OverlayResolver
│   ├── adapter.py         # RendererAdapter, RenderBackend protocol
│   └── pyvista.py         # PyVistaBackend + DebouncedFlush (trame)
├── collision/
│   ├── geometry.py        # FCL BVH and transform construction, with guards
│   ├── adapter.py         # CollisionAdapter, pair_bits (the pair rule)
│   ├── fcl.py             # FCLBackend (per-pair callback)
│   └── worker.py          # CollisionHandler (sync + async paths)
├── web/
│   ├── cli.py             # the rutter-plan entry point
│   ├── app.py             # build_trame_app() factory
│   ├── controller.py      # TrameController (Vuetify3 + PyVista)
│   └── ccf_regions.py     # CCFOverlayManager (lazy region meshes)
└── ccf/
    └── ontology.py        # Allen CCF structures + search
```

The top-level tree above is partial. Two big subpackages are not shown:

**`build/`** — turning a validated `ConfigModel` into a `RuntimeBundle`:
`assemble` (`build_runtime_from_config`), `loaders`, `reducers`, `canonicalize`,
`chem_shift`, `calibration` (the bank plus the NewScale frame), `queries`
(`head_pitch_deg_*`, fixture sets), `transforms`, `probe_context`. Its
`__init__` exports only `build_runtime_from_config` and `RuntimeBundle`;
everything else comes from its submodule.

**`optimization/`** — the placement-optimizer package. The solver
subpackages never import `pipeline`; the pipeline drives them.
- `assignment/` — which probe goes where: `visibility_atlas`, `atlas`,
  `arc_placement`, `seed_emission` (`emit_seed`), `assignments`
  (`ArcAssignment`/`HoleAssignment`)
- `geometry/` — `primitives` (`cap_basis`, `HoleSection`), `kinematics`
  (`pose_from_optimizer_vars`), `holes`, `headstages`, `probes`
- `clearance/` — `kernels` (`arc_angles_to_rotation`, `trilinear_sdf`),
  `voxel_sdf`, `envelope`, `samples`, `sweep`
- `objectives/` — `soft` (Phase 1), `constrained` (Phase 2), `threading`
  (`threading_g_matrix`), `reduced`, `packing`, `coverage`, `metrics`,
  `statics` (`JointWeights`), `variables`, `layout`, `cache_keys`
- `search/` — `spin_restore`, `minimizers` (the batched RProp/ADAM builders)
- `validation/` — `fcl`, the ground-truth check a candidate must pass
- `pipeline/` — the offline batch flow: `phase1`, `phase2`, `emit`, plus
  `candidates` (the enumerator and its atlas cache), `subject` (the runtime a
  stage optimizes over), `stage_setup`, `fixtures`, `thick_well`,
  `probe_setup`. Phase 2 is callable as `phase2.run(recs, settings)`; its
  inputs and outputs live in modules kept free of jax so they import without a
  GPU backend — `settings` (`Phase2Settings`, constructor over environment over
  defaults), `selection` (which candidates get solved), `handoff` (keep bands
  and provenance), `records` (the payload types), `payloads` (JSON files for
  the pool, handoff and Phase-1 caches; nothing reads pickles),
  `phase2_diagnostics`

Console entry points `rutter-phase1` / `rutter-phase2` / `rutter-emit` and the
`scripts/run_subject_overnight.sh` driver run the pipeline. **See
`dev/PIPELINE.md` for the read-verified stage map** (it's full of stale-docstring
traps — trust that doc, not the docstrings).

## Deep-dive docs

- `dev/CORE_CONCEPTS.md` — **start here.** Conceptual tour: catalog vs
  scene vs planning vs adapters, end-to-end slider-drag flow, common
  "why is it like this?" questions.
- `dev/MODULE_MAP.md` — per-module reference, layered architecture, data flow.
- `dev/COORDINATES.md` — LPS canonical rule, where conversions happen, frame
  composition, working in non-AIND template spaces.
- `dev/CONFIG_MODEL.md` — Pydantic model taxonomy, validation pipeline,
  template merge rules, plan-only YAML, gotchas.
- `dev/VOCABULARY.md` — which config fields are closed enums and which are open
  tags, and why; the fat/water physics behind `mr_signal`.
- `dev/PIPELINE.md` — **the placement-optimizer pipeline**, read-verified
  stage-by-stage (atlas → enumerate → spin restore → L-BFGS → ADAM rerank →
  Phase 2 → FCL → handoff), the legacy code, and the L-BFGS-vs-ADAM caveat.
  `dev/optimizer_plan.md` and `dev/spin_search_heuristics.md` are older design
  notes (some superseded — defer to PIPELINE.md on what's live).

End-user docs are in `docs/source/` (Sphinx). Don't bloat them with internals.
`docs/source/configuration.rst` is the guide someone writing a config for a new
subject reads; `tests/test_docs_configuration.py` loads its worked example, so it
cannot drift from the models.

## Code style

- 88-char line length, ruff-formatted.
- NumPy-style docstrings.
- Pydantic v2 with `extra="forbid"` on most models.
- `@dataclass(frozen=True, slots=True)` for immutable runtime data.
