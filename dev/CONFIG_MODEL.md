# Config Model

Field-level reference and gotcha list for `config.py`. The Pydantic model is
the source of truth — when this doc disagrees, fix the doc.

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
| `TargetSpecModel` | One `TargetSpec` (points) | File, derived from another asset, or inline points |
| `RangeTargetSpecModel` | List of `TargetSpec` | `key_pattern + range`, e.g. `target:hole:{n}` for `n in 1..16` |
| `DerivedTargetSpecModel` | List of `TargetSpec` | `derive_from: [asset_keys]`, applies same reducer to each |

`BulkAssetSpecModel.expand()`, `Range...expand()`, and `Derived...expand()` all
go through `_passthrough_kwargs(self, exclude, overrides)` — see the
"Template merge" gotcha below.

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
