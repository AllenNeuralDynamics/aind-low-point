# Core Concepts

A conceptual tour of the codebase for someone new to it (or returning
after months away). For per-file detail see `dev/MODULE_MAP.md`; for
Pydantic field specifics see `dev/CONFIG_MODEL.md`. This doc is the
mental model that makes the other two readable.

## The single most important distinction

**Catalog vs Scene vs Planning vs Rendering — they are four separate
layers.** Getting this straight saves a lot of "wait, where does X live?"
confusion.

```
┌─ Catalog ──────────────────────┐  "what assets exist"
│  AssetSpec / TargetSpec        │   geometry + per-asset material +
│                                │   metadata. One per loaded mesh /
│                                │   point cloud. Lives in
│                                │   AssetCatalog.
└────────────────────────────────┘
              │ referenced by
              ▼
┌─ Scene ────────────────────────┐  "where are things placed"
│  NodeInstance                  │   asset_key → catalog. Has its own
│                                │   transform, tags, overrides.
│                                │   Many nodes can share one asset.
└────────────────────────────────┘
              │ referenced by
              ▼
┌─ Planning ─────────────────────┐  "what are we doing experimentally"
│  ProbePlan / Kinematics /      │   probe-positioning logic; arc
│  PlanningState                 │   angles, targets, depth. Has its
│                                │   own coordinate space and rules.
└────────────────────────────────┘
              │ subscribed to by
              ▼
┌─ Adapters ─────────────────────┐  "make it visible / detect overlap"
│  RendererAdapter (→PyVista/K3D)│   listen for planning changes,
│  CollisionAdapter (→FCL)       │   walk the scene, update the
│                                │   chosen backend.
└────────────────────────────────┘
```

Everything else (config, commands, controllers) is plumbing on top of
those four.

## The layers, one by one

### Catalog (`assets.py`, `core.py`)

A catalog entry is a **loaded asset** — a mesh or point cloud sitting in
memory in canonical-LPS-millimetres coordinates, with a default material
and metadata. It does **not** know where in the world it goes.

- `AssetSpec` — generic catalog entry; carries mesh **or** points geometry.
- `TargetSpec` — a points-only specialization for "where a probe should
  aim". Adds `source_key` (which asset it was derived from) and an
  optional `reducer` (how the N-point source was collapsed to one).
- `AssetCatalog` — `{key: AssetSpec}` + `{key: TargetSpec}`. Two flat
  dicts. Everything else references catalog entries by string key.

Keys are convention-driven: `brain`, `structure:ACA`, `target:L:MD`,
`probe:quadbase`, `probe:2.1`. The runtime cares only that they're
unique; some prefixes drive auto-inference at config-load time (e.g.
`structure:*` → `role: anatomy`).

### Scene (`scene.py`)

The scene is a collection of **placements**: "asset X goes here, with
that base transform, tagged so". Same asset → multiple nodes is fine
(though we rarely use that — each probe in the rig is one node).

- `NodeInstance` — one placement: `key` (unique id like `probe:MD`),
  `asset_key` (catalog FK), a base `transform`, a set of `tags`, an
  optional `material_override` (per-instance recolor / opacity tweak),
  and `extras` (open-ended dict; e.g. `pose_source_probe` for probes).
- `Scene` — dict of NodeInstances. Queryable by tag (`scene.by_tag(...)`).

The catalog is "the library"; the scene is "where the books are on the
shelves." The scene is also where probe nodes live — they reference the
`probe:<kind>` asset but get a dynamic pose layered on top of their base
transform at render time (see Planning, below).

### Planning (`planning.py`, `commands.py`)

The probe-positioning domain. **Independent of the scene** — planning
state never reads scene nodes directly, only catalog entries (to look
up probe-mesh shapes for pivot calculations).

- `ProbePlan` — declarative per-probe: kind, arc id, AP/ML/spin angles,
  `target_key` (or inline `target_point_RAS`), offsets, depth past
  target, calibration flag, `position_bearing_shank` (which shank's tip
  is the readout reference for multi-shank probes).
- `Kinematics` — `arc_angles: {arc_id: deg}` and rig limits.
- `PlanningState` — bundles `Kinematics`, `{name: ProbePlan}`,
  calibrations, and a `target_index: {key: LPS_point}` (precomputed
  from the catalog so probe-resolution doesn't need to re-read mesh
  centroids).
- `ProbePose` — *derived*: given a `PlanningState` and a probe name,
  resolve the angles (calibration > arc-bound > local) and compute the
  named-shank tip world position. Output is `(ap, ml, spin, tip_LPS)`.
- `PoseResolver` — given a scene node and a planning state, returns the
  full world transform = scene's base transform composed with the probe
  pose. This is what the renderer asks for each frame.

Mutations are **commands**: `SetProbeLocalAngles`, `SetProbePastTarget`,
`AssignProbeArc`, etc. They're frozen dataclasses; you build one and
dispatch it through the store.

### State management (`state_change.py`, `commands.py`)

Redux-ish unidirectional flow.

- `PlanStore` — wraps `PlanningState`; `dispatch(cmd)` mutates the
  state and notifies subscribers synchronously with the list of
  affected probe ids. That's it — no time-travel, no middleware.
- Subscribers are plain callables `(plan_state, changed_ids) -> None`.
  The renderer registers one, the collision system another, the
  trame controller a third (for readouts).
- `AsyncLatestWorker` — subscriber adapter for off-thread work
  (collision queries run here). Latest-only: rapid edits collapse
  to one in-flight job + a single follow-up.

The flow on every edit: **slider drag → controller → dispatch command
→ store applies + notifies → adapters react → frontend re-renders**.

### Adapters (`rendering.py`, `collisions.py`)

Adapters translate planning state into *backend operations* without
the planning layer knowing what backend exists.

- `RendererAdapter` — given a scene + planning state, computes each
  node's world transform and pushes it to a `RenderBackend`
  (`pyvista_backend.py` for trame, `k3d_backend.py` for Jupyter). Also
  handles material overrides and the **overlay system** — additional
  per-node tint / alpha contributions stacked by priority (collisions
  paint a red tint; selection or hover would paint others).
- `CollisionAdapter` — given a scene + planning state, builds
  collision geometry for nodes with the `COLLIDABLE` capability and
  asks a `CollisionBackend` (`fcl_backend.py`) for the current set of
  colliding pairs. Result flows back as a `CollisionState`.
- `RenderHandler` / `CollisionHandler` — the *subscribers*. They
  receive `(plan, changed_ids)` notifications, decide which nodes to
  touch, and call the adapter.

The split between adapter and handler exists so adapters can be tested
in isolation against fake backends, and so the async-collision worker
can drive the adapter without going through the subscriber path.

### Frontends (`controllers.py`, `trame_controller.py`, `app.py`)

Two parallel UIs, same runtime:

- `ProbeWidgetController` — ipywidgets + K3D, for Jupyter notebooks.
- `TrameController` — Vuetify3 components + PyVista (via VTK.js in the
  browser), for the web app. `app.py:build_trame_app(cfg)` is the
  factory that wires up the runtime, store, adapters, async worker,
  and controller into a ready-to-serve trame app.

Frontends own:
- The widgets and event handlers (slider drags, keyboard shortcuts).
- The readout strings (tip RAS, depth, kinematic status, collisions).
- The view-state that isn't planning state (camera, visibility toggles,
  speed mode, help dialog).

They translate user input into commands and dispatch them.

### Configuration (`config.py`, `runtime/build.py`)

The whole catalog + scene + planning state is built from a single YAML
via Pydantic models in `config.py`. The build pipeline:

1. **Parse** — `ConfigModel.from_yaml(path)` runs OmegaConf
   interpolation (`${paths.foo}`) then Pydantic validation.
2. **Expand** — bulk asset declarations (`keys: [...]`), atlas mesh
   packs (`acronyms: [...]`), derived targets (`derive_from: [...]`),
   and templates (`templates: [name]` or glob match) are all unrolled
   into individual `AssetSpec` / `TargetSpec` instances. See
   `dev/CONFIG_MODEL.md` for the merge mechanics.
3. **Cross-reference** — every `material_ref`, `transform.key`,
   `from_resource`, target ref in `ProbeDeclModel`, scene node `asset`
   — all checked against the catalog / registry. Errors collected,
   one `ValidationError` raised.
4. **Build runtime** — `build_runtime_from_config(cfg)` loads geometry
   (via loader registry), canonicalizes (orientation + scale + optional
   transform → LPS-mm), applies chemical-shift correction for MR
   imagery, builds the catalog + scene + planning state, returns a
   `RuntimeBundle`.

Once you have the bundle, the rest of the runtime never touches
`ConfigModel` again. Save-back (`save_plan_to_config`,
`planning_state_to_plan_model`, `export_plan_geometry`) goes the other
direction for export only.

## How a slider drag flows end-to-end

Pick up the *AP tilt* slider and drag it 1°. What happens:

```
1.  Vue → trame.state.ap_tilt = 13.0
       ↓
2.  trame state-change handler fires server-side
       ↓
3.  Controller dispatches SetProbeLocalAngles(name=..., ap_local=13.0)
       ↓
4.  PlanStore.dispatch:
        apply_planning_command mutates state.probes[name].ap_local
        returns ['<probe-name>'] (the affected ids)
        notifies subscribers synchronously
       ↓
5.  Sync subscribers:
        RenderHandler → RendererAdapter.on_store_change(plan, ['<name>'])
           - PoseResolver computes new world chain for that probe node
           - pyvista_backend updates the actor's user_matrix
        TrameController._on_plan_change_for_readouts(plan, ['<name>'])
           - Recompute tip-RAS / depth strings; push to trame state.
       ↓
6.  Async subscriber:
        AsyncLatestWorker captures the state snapshot, kicks
        CollisionHandler.prepare/work/deliver on a worker thread:
           - FCL rebuilds the probe's collision geometry
           - FCL collides; returns CollisionState
           - on_state_changed callback chain fires (overlay repaint,
             then our collision-readout refresh)
       ↓
7.  Trame ships scene/state updates over the WebSocket;
    VTK.js / Vue3 re-render the browser view.
```

The synchronous chain (1–5) handles "the probe moved"; the async chain
(6–7) handles "the probe collides with something" without blocking the
UI. Both fan out from the *same* `dispatch` call.

## Why this many layers?

Common questions, in order of how often they come up:

**"Why isn't the planning state just the scene?"**
Because planning is independent of *how* you visualize and *what* you
collide against. The same planning state drives the K3D Jupyter UI and
the trame web app with zero changes — only the frontend + adapter swap.

**"Why are catalogs and scenes separate?"**
Because one asset can have many placements (the probe mesh asset is
referenced by every probe node in the scene). Also: the scene is
something the user can mutate at runtime (showing/hiding fixtures,
overriding materials); the catalog is loaded-once-and-frozen.

**"Why commands instead of just calling `state.probes[name].ap = x`?"**
Two reasons. (a) The store's notification mechanism needs to know what
changed — commands return the affected ids, direct mutation wouldn't.
(b) Commands are serializable, so we can replay them, log them, or
generate them from non-UI sources (the optimizer dispatches commands
the same way the slider does).

**"Why do probes have a base transform AND a pose?"**
The scene node carries a *base* placement transform (where the rig's
chassis sits in world LPS — usually identity). The pose is the
*dynamic* transform driven by planning state. `PoseResolver` composes
them so the renderer doesn't need to know the difference. This way
non-probe nodes have a static transform and skip pose resolution
entirely.

**"Where's the dividing line between `tags` and `scene_tags`?"**
`tags` lives on the catalog (asset/target spec); `scene_tags` lives on
the scene (node instance). See `dev/CONFIG_MODEL.md`. Authoring tip:
`tags` is for code that wants to *find* things ("give me all CCF
regions"); `scene_tags` is for the user-facing UI ("show me probes").
