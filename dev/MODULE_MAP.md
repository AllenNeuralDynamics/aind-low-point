# Module Map

Per-file reference for `src/aind_low_point/`. Layered top-down: enums → core
data → catalog/scene → planning state → adapters → frontends.

**New to the codebase?** Start with `dev/CORE_CONCEPTS.md` for the
mental model (catalog vs scene vs planning vs adapters, end-to-end
slider-drag walkthrough, the design questions you'll ask in week 1).
Come back here for per-file detail.

## Foundations

| File | Purpose |
|---|---|
| `__init__.py` | Empty package marker. |
| `common.py` | `Capability` (IntFlag), `Role` (str Enum), `Kind` (str Enum). |
| `orientation_codes.py` | `OrientationCode` StrEnum covering all 48 RAS-style codes. Used by canonicalization. |

## Core data primitives — `core.py`

Frozen dataclasses; transforms compose through `TransformChain`.

- `AffineTransform` — rotation + translation, lazy invert.
- `TransformChain` — tuple of `AffineTransform`s; `composed_transform` is `cached_property`.
- `MeshTransformable` / `PointsTransformable` — wrap geometry, expose `raw` and `transformed(R, t)`.
- `Transformed[W, RawT]` — pairs a wrapper with a `TransformChain`; `raw` is cached.
- `Material` — name, color (hex string), opacity, wireframe, visible, point_size.
- Type aliases: `Float3`, `Float3x3`, `FloatNx3`, `FloatAABB`, `Pair`.

## Asset catalog — `assets.py`

- `BaseSpec` — common fields: key, kind, role, default_material, metadata, tags,
  caps (Capability), collidable_group/mask, pivot_LPS.
- `AssetSpec(BaseSpec)` — adds `mesh: MeshTransformable | None` and
  `points: PointsTransformable | None`. Geometry is **post-load, in canonical
  LPS mm**.
- `TargetSpec(BaseSpec)` — adds `source_key`, `reducer`, `points`,
  `approach_vector`, `uncertainty_mm`. Defaults to `kind="points"`,
  `role=TARGET`, non-collidable.
- `AssetCatalog` — two dicts: `assets` and `targets`. `get_geometry(key)`
  resolves either.

## Scene — `scene.py`

- `NodeInstance` — `key`, `asset_key` (catalog FK), `transform: TransformChain`
  (base placement), tags, material override, locked axes, `extras` dict (e.g.
  `{"pose_source_probe": "PL"}`).
- `Scene` — `nodes: dict[str, NodeInstance]`.
- `resolve_base_pose(scene, id)` / `resolve_base_geometry(catalog, scene, id)`
  — apply only the static base transform (no probe pose).

## Planning domain — `planning.py`

The probe kinematics layer. Pure functions over `PlanningState`.

- `ProbePlan` — kind, arc_id, `bind_ap_to_arc`, ap_local, ml_local, spin,
  past_target_mm, offsets_RA, target_key / target_point_RAS, calibrated,
  `position_bearing_shank` (1-indexed; for multi-shank probes selects the
  shank whose tip is used for tip-RAS / brain-depth readouts and as the
  reference for `past_target_mm`).
- `JointRange` / `PoseLimits` — clamps for angles and translational envelope.
- `Kinematics` — `arc_angles: dict[str, float]`, limits, `coupled_axes`.
- `PlanningState` — kinematics, `probes: dict[str, ProbePlan]`, calibrations,
  `target_index: dict[str, Float3]` (LPS).
- `ProbePose` — resolved (ap, ml, spin, tip_LPS); `from_planning_state(ps, name)`
  resolves angles (calibration > arc > local) and tip (target + RA offset +
  past_target along probe axis), clamping to limits.
- `PoseResolver` — given scene + plan + pivot lookup, returns a node's full
  world TransformChain (`base ∘ dynamic`). Handles pivot wrap for probes
  rotated about a tip pivot.

## Commands — `commands.py`

Frozen dataclasses representing planning mutations. `PlanningCommand` is a
`Union` of all of them. `apply_planning_command(state, cmd) -> List[str]`
returns the list of probe names that need re-rendering.

Current commands: `SetProbeLocalAngles`, `SetProbeOffsetsRA`,
`NudgeProbeOffsetsRA`, `SetProbePastTarget`, `NudgeProbePastTarget`,
`SetProbeTarget`, `SetArcAngle`, `AssignProbeArc`, `BindProbeAPToArc`,
`SetProbeCalibrated`.

## Configuration — `config.py`

Pydantic v2 models (~2100 lines). See `dev/CONFIG_MODEL.md` for the structural
breakdown. The entry point is `ConfigModel.from_yaml(path)` (uses OmegaConf to
resolve `${...}` interpolation).

Key validation pipeline runs in `_xref_and_expand_templates` (model_validator
mode="after"): expand bulk specs → check template refs → apply templates →
cross-reference all keys → collect all errors before raising.

## Build runtime — `build_runtime.py`

Factory layer. `build_runtime_from_config(cfg) -> RuntimeBundle`.

- **Loaders** (registry): `trimesh` (force=mesh), `sitk_volume`, `csv_points`,
  `load_trimesh_lps`, `read_slicer_fcsv`. Register your own with
  `@register_loader("name")`.
- **Reducers** (registry): `mesh_centroid`, `mesh_center_mass`. Register with
  `@register_reducer("name")`.
- **Canonicalization**: `_apply_canonicalization_mesh` /
  `_apply_canonicalization_points` apply orientation flip + scale_to_mm +
  optional inline transform.
- **Chemical shift**: `ChemShiftContext` builds a per-ppm correction transform
  applied to MRI-derived geometry (anatomy role by default).
- **Round-trip**: `planning_state_to_plan_model` and `save_plan_to_config`
  serialize a live PlanningState back to a ConfigModel for YAML export.
  `apply_plan_model_to_state(plan, store)` is the reverse — dispatches per-
  arc + per-probe commands through the store so the renderer / collisions
  fan out as if the user edited each field. Used by the trame *Load plan*
  file picker.
- **Plan export**: `export_plan_geometry(state, catalog, *, source_config=...)`
  emits a thin per-probe geometric summary (kind, target, arc, angles,
  tip_RAS_mm, depth_from_brain_surface_mm) — for hand-off to physical
  execution. Different shape from `save_plan_to_config` (which round-trips
  the full ConfigModel); read-only, no loader.

`RuntimeBundle = (asset_catalog, scene, plan_state, label_index)`.

## State — `state_change.py`

- `PlanStore` — wraps `PlanningState`; `dispatch(cmd)` → mutate via
  `apply_planning_command` → notify subscribers synchronously with
  `(plan, changed_probe_ids)`.
- `AsyncLatestWorker` — Subscriber implementation that runs work off-thread.
  Calls `prepare(plan, ids)` on the main thread (reads PlanningState safely),
  posts to a worker thread which runs `work(request)`, then posts
  `deliver(result)` back to the main thread via `post_to_main`. Latest-only
  semantics: rapid updates collapse into a single in-flight request.
- `StoreSubscriber` — small RAII helper that subscribes on init and
  unsubscribes on `dispose()`.

## Rendering — `rendering.py`

Backend-agnostic adapter that pushes a 4×4 `model_matrix` to renderers.

- `RenderBackend` (Protocol) — create_mesh / update_mesh / create_points /
  update_points all take an optional `model_matrix`.
- `RendererAdapter` — `build(plan)` (full rebuild),
  `sync_nodes(plan, nodes, coll)` (subset), `repaint_materials(node_ids)`
  (material-only, no pose recompute — used by collision overlay).
- `RenderHandler` — Subscriber that updates probe nodes when their plans
  change.
- Overlay system: `OverlaySpec` (color, alpha, blend, priority, source) +
  `OverlayState` (multi-source per node) + `OverlayResolver` (folds overlays
  into final `ViewMaterial`).
- `on_collisions_changed_lambda` wires collision flips → repaint.

## Collisions — `collisions.py`

- `CollisionBackend` (Protocol) — rebuild / sync / `update_transforms` /
  remove / collide_internal / collide_one_to_many.
- `CollisionAdapter` — `rebuild(plan)`, `update_probe_transforms(plan, names)`
  (transform-only fast path), `collide_internal()`. `_spec_for_node` builds an
  `ObjSpec` with group/mask copied from the AssetSpec.
- `CollisionHandler` — subscriber-style `__call__(plan, ids)` for the
  synchronous path, plus factored `prepare` / `work` / `deliver` for use with
  `AsyncLatestWorker`.

## Backend implementations

- `k3d_backend.py` — `K3DBackend(plot)`. Sets `h.model_matrix` on K3D handles.
  Used by the Jupyter / ipywidgets controller.
- `pyvista_backend.py` — `PyVistaBackend(plotter)`. Sets `actor.user_matrix` on
  PyVista actors. Includes `DebouncedFlush` (asyncio `call_later`-based) used
  by trame to coalesce browser pushes.
- `fcl_backend.py` — `FCLBackend`. Per-pair custom collision callback (each
  pair gets a fresh `CollisionResult` with `num_max_contacts=1`) with
  group/mask filter. Replaces the previous `defaultCollisionCallback` which
  silently dropped pairs after hitting the global accumulator limit.

## Frontends

Two parallel UI implementations sharing all of the above:

### Jupyter — `controllers.py`

`ProbeWidgetController` — ipywidgets sliders + K3D plot for notebook use.
Has keyboard nav helpers, an arc-assign dropdown, and a target-snap dropdown.
Calls `store.dispatch(...)` directly.

### Trame web — `trame_controller.py` + `app.py`

- `TrameController` — Vuetify3 layout with PyVista 3D view via
  `pyvista.trame.ui.plotter_ui(... mode="client")`. Reactive state:
  probe / arc / target dropdowns, R/A offset / depth / AP-tilt / ML-tilt /
  spin sliders, optional Save YAML button, optional CCF region overlay.
- `build_trame_app(cfg, *, ccf_volume=None, save_path=None) -> Server` is the
  factory — wires runtime, store, render adapter, async collision worker,
  overlay state, optional CCF overlay manager, debounced flush, and returns a
  ready-to-`server.start()` trame server.

## CCF reference overlays

Independent of the catalog/scene; manage their own PyVista actors.

- `ccf_ontology.py` — `CCFOntology` loads bundled
  `data/allen_ccf_ontology.json` (~1300 structures); substring/autocomplete
  search by acronym and name.
- `ccf_overlay.py` — `CCFOverlayManager(plotter, volume_path)` lazy-extracts
  region meshes from a labelmap nrrd via marching cubes, decimates, and shows
  them as semi-transparent reference geometry. Bypasses
  AssetCatalog/RendererAdapter on purpose — these are reference overlays, not
  planned objects.

## Data flow at a glance

```
YAML (with OmegaConf ${} interpolation)
   │
   ▼  ConfigModel.from_yaml()
ConfigModel  (Pydantic models, validation, template expansion)
   │
   ▼  build_runtime_from_config()
RuntimeBundle
   │     ├── AssetCatalog (assets / targets — both in LPS mm)
   │     ├── Scene        (nodes with base transforms)
   │     └── PlanningState (kinematics / probes / calibrations / target_index)
   │
   ▼  PlanStore wraps PlanningState
PlanStore
   │
   │  store.dispatch(SetProbeLocalAngles(...))
   │     └── apply_planning_command() mutates state, returns changed probe ids
   │
   ▼  notify subscribers
   ├── RenderHandler  (sync, ~ms)
   │     └── RendererAdapter.sync_nodes()
   │           └── RenderBackend (K3D | PyVista) — pushes model_matrix
   │
   └── AsyncLatestWorker  (off-thread, latest-only)
         ├── prepare(plan, ids)   on main thread → (node_id, fcl.Transform)[]
         ├── work(transforms)     on worker     → CollisionState
         └── deliver(state, plan) on main thread → repaint_materials(flips)
```
