# Config Model

Field-level reference and gotcha list for `config.py`. The Pydantic model is
the source of truth — when this doc disagrees, fix the doc.

**New to the project?** Read `dev/CORE_CONCEPTS.md` first — it explains
what catalog / scene / planning / adapters are and how they fit
together. This doc assumes you already know that the YAML drives all
four.

Entry point: `ConfigModel.from_yaml(path)` — uses OmegaConf for `${...}`
interpolation, then `model_validate`. Plain `model_validate(dict)` is fine
for tests / programmatic config.

## Top-level shape

```yaml
version: 1
paths:           # PathsModel — free-form dict for ${paths.foo} interpolation
imaging:         # ImagingModel — magnet_frequency_MHz, chem_shift_*
materials:       # dict[str, MaterialModel] — named material registry
transforms:      # dict[str, TransformRecipeModel] — named transform registry
canonicalizations:  # dict[str, CanonicalizationDefModel]
asset_templates:    # dict[str, AssetTemplateModel]
target_templates:   # dict[str, TargetTemplateModel]
resources:       # list[ResourceModel] — load-once shared files (fcsv, etc.)
assets:          # list[AssetSpecUnion]  (Asset | BulkAsset)
targets:         # list[TargetSpecUnion] (Target | RangeTarget | DerivedTarget)
scene:           # SceneModel — list of SceneNodeModel
plan:            # PlanningModel — arcs, probes, calibrations, reticles
options:         # OptionsModel
```

## The four "kinds" of catalog entry

| Model | What it produces | Source |
|---|---|---|
| `AssetSpecModel` | One `AssetSpec` (mesh/points) | File via `src + loader`, or resource via `from_resource + selector` |
| `BulkAssetSpecModel` | List of `AssetSpec` | `keys: [...]`, optional `src` with `{name}` / `{key}` placeholders |
| `AtlasMeshPackSpecModel` | List of `AssetSpec` (meshes) | `atlas_dir + acronyms: [...]`, resolves CCF acronyms via the bundled ontology to `<id>.obj` files |
| `TargetSpecModel` | One `TargetSpec` (points) | File, derived from another asset, or inline points |
| `RangeTargetSpecModel` | List of `TargetSpec` | `key_pattern + range`, e.g. `target:hole:{n}` for `n in 1..16` |
| `DerivedTargetSpecModel` | List of `TargetSpec` | `derive_from: [asset_keys]`, applies same reducer to each |

All bulk/expansion variants route through `_passthrough_kwargs(self, exclude,
overrides)` — see the "Template merge" gotcha below.

### `AtlasMeshPackSpecModel` specifics

```yaml
- atlas_dir: ${paths.atlas_dir}     # directory of <id>.obj files
  acronyms: [VISp, MOs, CA1, BLA]   # CCF region acronyms (case-sensitive)
  key_prefix: atlas                 # → "atlas:VISp", "atlas:MOs", …
  file_extension: .obj              # default
  use_ccf_color: false              # default; set true to colour by CCF
  canonicalization_ref: atlas-template-lps
  material_ref: structure
```

- Acronym → CCF integer id resolves via `CCFOntology.from_bundled().find_by_acronym(...)`.
  The bundled ontology is cached as a module-level singleton so repeated
  expansions don't re-read the JSON.
- Acronym validity is checked in a `model_validator(mode="after")` — unknown
  acronyms produce a single `ValidationError` listing them all. Match is
  case-sensitive: `"VISp"` works, `"visp"` does not.
- `kind` and `loader` are forced to `mesh` / `trimesh` in `expand()` (an atlas
  pack is by definition a directory of meshes).
- `role` defaults to `Role.ANATOMY` (sensible for brain regions); user can
  override at the pack level. Other fields (templates, material, transform,
  scene_tags, caps, collision, …) behave the same as in `BulkAssetSpecModel`.
- `use_ccf_color: true` injects a per-region inline `material` with the CCF
  region's bundled `color_hex`. Other material fields (opacity, point_size,
  …) flow through from `material_ref` / pack-level `material` as usual. If
  the user explicitly sets `material.color` on the pack, that wins (explicit
  > implicit).

## Templates

`AssetTemplateModel` and `TargetTemplateModel` define reusable defaults. Specs
opt in via `templates: [name1, name2]` or via `_find_matching_templates(key)`
(name-based: `key.split(":")[0]` is matched against template keys).

Template application pipeline (in `_xref_and_expand_templates`):

1. **Bulk expansion** — bulk/range/derived models become individual
   `AssetSpecModel` / `TargetSpecModel` instances. Only fields explicitly set
   in the bulk YAML carry through (via `model_fields_set`); unset fields are
   left for the template merge.
2. **Template ref check** — `_check_template_ref` validates every named
   template exists before any merging happens (catches typos before they
   silently fall through to defaults).
3. **Template apply** — `apply_templates_generic` walks each spec, merges
   templates in declaration order (left-most wins for conflicts) using
   `merge_asset_template_model_dumps` / `merge_target_template_model_dumps`.
4. **Cross-reference** — every `material_ref`, `canonicalization_ref`,
   `transform.key`, `from_resource`, target ref in `ProbeDeclModel`, scene
   node `asset` — all checked against catalog/registry.
5. **All errors collected**, then a single `ValidationError` if any.

## Source disambiguation on AssetSpecModel

A single asset must use exactly one source:

- `src + loader` (file)
- `from_resource + selector` (extract from a previously-loaded resource)

Validated in `AssetSpecModel._check_required` (~line 600). Mixing or omitting
both raises.

## Source disambiguation on TargetSpecModel

Three valid forms, mutually exclusive:

- `src + loader` (file)
- `source_key + reducer` (derive from another asset; reducer reduces an
  N-point cloud to a single (3,) point)
- `point_RAS` inline (handled at the `ProbeDeclModel.target` level, not on
  `TargetSpecModel`; see `InlineTargetRefModel` below)

## ProbeDeclModel.target — the discriminated union

```yaml
plan:
  probes:
    PL:
      kind: "2.1"
      arc: b
      target:                           # CatalogTargetRefModel
        kind: catalog
        key: target:PL
      # OR
      target:                           # NodeTargetRefModel
        kind: node
        key: target:PL                  # node id in the scene
      # OR
      target:                           # InlineTargetRefModel
        kind: inline
        point_RAS: [1.5, -3.2, 4.0]
      # OR (coercion from a bare list/tuple → InlineTargetRefModel)
      target: [1.5, -3.2, 4.0]
```

The bare-list coercion happens in `ProbeDeclModel`'s pre-validator
(~line 1099). Inline targets skip catalog/node xref checks (they're
self-contained).

## CanonicalizationDefModel

See `dev/COORDINATES.md` for semantics. Validation specifics (in
`_check_canon_fields`):

- `source_space: FILE_NATIVE` requires a `transform` (no fallback orientation
  flip exists).
- For non-`FILE_NATIVE`, providing `transform` AND a meaningful orientation
  is an error — pick one. (Translation: don't combine an axis-flip with a
  rigid transform; fold them into one or the other.)
- `canonicalization_ref` and inline `canonicalization` are mutually exclusive.

## Selectors (for `from_resource`)

Resources can return structured payloads (a `dict[str, ndarray]` of named
points, a labelmap dict, a GLTF tree). A `Selector` extracts the piece this
spec wants:

- `NameSelector(name="bregma")` — look up by string key
- `IndexSelector(index=0)` — list/array index
- `PathSelector(path=["foo", "bar"])` — nested traversal
- `LabelSelector(label=42)` — labelmap entry

## Validation pipeline (the gnarly part)

`ConfigModel._xref_and_expand_templates` is the single model_validator that
does everything in `mode="after"`. Order matters:

1. Expand bulk/range/derived.
2. Build a flat list of all asset/target keys.
3. Check duplicates.
4. Check template references.
5. Apply templates.
6. Cross-reference: `material_ref`, `canonicalization_ref`, transform refs,
   resource refs, scene node `asset`, probe target refs, calibration probe
   names, reticle refs.
7. Spec-level validators: AssetSpecModel (src/loader vs resource/selector
   mutex), TargetSpecModel (single source mutex), CollidableRestriction
   (targets non-collidable by default).
8. Accumulate errors; raise once.

Errors are collected via the local `err(msg)` callback so the user sees all
problems at once, not one at a time.

## `tags` vs `scene_tags` (two fields, different scopes)

Both appear on almost every model and easy to confuse.

| Field | Lives on | Used for | When unset |
|---|---|---|---|
| `tags` | `AssetSpec` / `TargetSpec` (catalog) | Catalog-level metadata: `catalog.assets_with_tag(...)`, "is this a CCF region asset?". Does **not** reach the scene. | empty list |
| `scene_tags` | `NodeInstance` (scene) | Scene-level filters: `scene.by_tag(...)`, `VISIBILITY_GROUPS` (visibility toggles + opacity sliders), default-opacity overrides in the trame controller, collision-group selection. | empty list (no scene node unless `transform` is set) |

A node is auto-created from the asset when **either** `transform` is set
**or** `scene_tags` is non-empty (controlled by `auto_scene`, default
`True`). Set `auto_scene: false` to suppress.

### Well-known `scene_tags` values

These are what existing UI / runtime logic actively looks for. New values
are fine — they just won't trigger any behaviour unless someone wires them
up.

| Tag | Meaning / Behaviour |
|---|---|
| `static` | Doesn't move with probe state. Used for collision group inclusion. |
| `dynamic` | Repositioned on every probe state change (probes only). |
| `probe` | Identifies probe meshes. Matches the `("probes", "Probes", {"probe"}, set())` group in `VISIBILITY_GROUPS` → drives the *Probes* visibility switch + opacity slider on the Display tab. |
| `brain` | Drives the "Brain outline" visibility group; also what `recenter_view` finds when computing the camera focal point. |
| `structure` | CCF-region meshes; drives the "CCF regions" group. |
| `fixture` | Generic non-implant rig hardware (well, probe-guard, …). Drives the "Other fixtures" group; default opacity 0.6 via `_DEFAULT_OPACITY_BY_TAG`. |
| `implant` | The implant body. Drives the "Implant" group; default opacity 0.2. Note the implant typically carries **both** `fixture` and `implant`; the visibility-group exclusion column keeps the implant slider distinct from the "Other fixtures" slider. |
| `headframe` | Headframe mesh. Subject to fixture-group opacity defaults. |
| `target` | Visualised target points. |
| `hole` | Per-bore points on the implant (for hole extraction). |

### `ProbeDeclModel` defaults

```yaml
plan:
  probes:
    P1:
      kind: quadbase
      # defaults:
      # ap_local: null
      # bind_ap_to_arc: true
      # slider_ml: 0.0
      # spin: 0.0
      # past_target_mm: 0.0
      # offsets_RA: [0.0, 0.0]
      # position_bearing_shank: 1   # 1-indexed; multi-shank only — chooses
      #                              # the named shank for tip readouts +
      #                              # ``past_target_mm`` reference
      # calibrated: false
      # auto_scene: true
      # scene_tags: ["probe", "dynamic"]
      # chem_shift_policy: auto
```

`scene_tags` on probes is what causes `("probe", ...)` filtering elsewhere
to work; **don't override it** unless you know what you're doing —
omitting `dynamic` will make collision and rendering treat the probe as
static, omitting `probe` removes it from the visibility group and the
probe-set queries that drive selection / labels.

## Plan-only YAML (`Save plan` / `Load plan` buttons)

The trame `Save plan` button writes only the `plan:` block of the
`ConfigModel` — i.e. a serialized `PlanningModel`. The `Load plan` file
picker reads the same shape. This format is intentionally portable: it
contains no `assets`, no `targets`, no `transforms`. Any config that
shares the same probe roster (probe names + arc letters) can load any of
these YAMLs.

Top-level shape:

```yaml
arcs:
  a: 13.0
  b: -10.0
  c: -43.0
probes:
  P1:
    kind: quadbase
    arc: a
    slider_ml: -12.0
    spin: 141.0
    ap_local: null
    bind_ap_to_arc: true
    past_target_mm: 0.0675
    offsets_RA: [0.0, 0.0]
    position_bearing_shank: 1
    target: { kind: catalog, key: target:MD }
    # …
```

Loading dispatches one `SetArcAngle` per arc and a sequence of per-probe
commands (`SetProbeKind`, `AssignProbeArc`, `SetProbeLocalAngles`,
`SetProbeOffsetsRA`, `SetProbePastTarget`, `SetProbeTarget`,
`SetProbePositionBearingShank`, `SetProbeCalibrated`) through
`apply_plan_model_to_state(plan, store)` (`runtime/export.py`). Probes
not in the current state are skipped with a stdout warning.

The geometric **Export plan** button (`export_plan_geometry`) is a
different format: it emits a flat per-probe summary (`kind`, `target`,
`arc`, `angles_deg`, `tip_RAS_mm`, `depth_from_brain_surface_mm`) for
hand-off to physical execution. It's read-only; there's no loader for it.

## Known gotchas

- **Template merge "set" trap**: before the `_passthrough_kwargs` rewrite,
  `expand()` methods passed every field explicitly to child models. Pydantic
  marks those as set, so subsequent template merges can't fill in the
  defaults. Always use `model_fields_set` when forwarding from a parent
  model. The current `_passthrough_kwargs` helper in `config.py:627` handles
  this correctly.
- **`_merge_dict_shallow(None, dict)`**: must return the dict, not None.
  Old buggy form silently dropped overrides. Tests check this directly.
- **Reducer fields in target merge**: target `_merge_target_source_fields`
  must propagate `reducer` and `reducer_kwargs` through both the file-source
  and key-source branches (was historically dropped on one branch).
- **Scene nodes referencing targets**: post-refactor, `SceneNodeModel.asset`
  may name a target key, not just an asset key. The cross-reference accepts
  either. Useful when you want a derived/inline target visualized in the
  scene with its own placement transform.
- **`set(Kind.POINTS)`**: Kind is `str, Enum`, so `set(Kind.POINTS)` iterates
  the string `"points"` into single chars. Always use `{Kind.POINTS}` set
  literal.
- **`auto_scene` / `scene_tags`**: config-only fields. They drive automatic
  scene node generation but don't survive to the runtime spec. When
  round-tripping (`save_plan_to_config`), preserve them from the original
  `ProbeDeclModel`.

## Where to look when something goes wrong

| Symptom | Likely culprit | File location |
|---|---|---|
| "Unknown template" raised on valid YAML | `_check_template_ref` ordering | `config.py` ~line 1480 |
| Template values not appearing in expanded spec | `model_fields_set` not respected → use `_passthrough_kwargs` | `config.py:627` |
| Asset loaded but geometry missing in catalog | Loader registered with wrong arity / signature | `build_runtime.py` registry |
| Target at origin / "Missing target for key" warning | Target wasn't in `target_index`; check `_resolve_target_LPS_from_plan` fallback path | `planning.py:151` |
| Probe orientation off | Wrong `canonicalization_ref` for probe mesh (LSA vs ASR) | example config + `canonicalizations` block |
| Collisions silently missing pairs | This was the `defaultCollisionCallback` bug — fixed in `fcl_backend.py` | `fcl_backend.py:113` (per-pair callback) |
