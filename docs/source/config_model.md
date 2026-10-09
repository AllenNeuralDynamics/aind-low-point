# Config Model

Field-level reference and gotcha list for the `config/` package. The Pydantic model is
the source of truth — when this doc disagrees, fix the doc.

**New to the project?** Read [Core Concepts](concepts.md) first — it explains
what catalog / scene / planning / adapters are and how they fit
together. This doc assumes you already know that the YAML drives all
four.

Entry point: `ConfigModel.from_yaml(path)` — uses OmegaConf for `${...}`
interpolation, then `model_validate`. Plain `model_validate(dict)` is fine
for tests / programmatic config.

## Where `paths:` names are

A config reaches its input folders through `${paths.<name>}`, and the folders
differ by machine. `from_yaml` locates each name declared in `paths:` before it
resolves anything, taking the first of:

1. `--path NAME=DIR` on the command line (`rutter-plan`, `rutter-paths`).
2. `--paths FILE`, repeatable, then the files in `RUTTER_PATHS` (separated by
   `os.pathsep`). The pipeline reads `RUTTER_PATHS` through its settings.
3. `rutter.paths.yml` in the config's folder, then in each folder above it.
4. The machine's `paths.yml`, in `RUTTER_CONFIG_DIR` or else the platform's
   config folder (`~/.config/rutter` on Linux, `~/Library/Application
   Support/rutter` on macOS, `%LOCALAPPDATA%\rutter` on Windows).
5. For a pack value, the pack cache.
6. The value written in the config.

A paths file maps names to folders:

```yaml
template_dir: Z:/ephys/persist/data/MRI/Templates/AIND-MouseT1-v1.0
rutter_data: ../rutter          # relative to this file's folder
```

- A name derived from another in the config (`${.template_dir}/x.nii.gz`)
  follows wherever that name is located.
- Names a file sets that the config does not declare are ignored; a `--path`
  naming nothing declared is refused as a misspelling.
- A config may write `???` for a name only a paths file can supply; loading
  then fails listing every file searched.
- Drive-letter and UNC folders count as absolute on every platform, so one
  file means the same wherever it is read.

`rutter-paths config.yml` prints each name, its folder, what set it, every
file searched and the pack cache; values set by a paths file are also logged at
load.

### Packs

A `paths:` value may name a published, immutable folder instead of a location:

```yaml
paths:
  hardware: {pack: hardware/v1}
```

A paths file can point a pack at a local copy, under `packs:` (in one file a
name's own line beats its pack's line):

```yaml
packs:
  hardware/v1: /mnt/Data/rutter/hardware
```

Otherwise the name resolves to the pack's folder in the cache, and loading
fetches only what the config names: each file with its companions (same folder,
same stem: an `.obj`'s `.mtl`), and a named folder whole once the config is
validated and its bulk patterns expanded. Every download is checked against the
pack's manifest for size and sha256.

The store is `https://aind-scratch-data.s3.amazonaws.com/rutter/packs/`, laid
out as `<name>/<version>/manifest.json` and `<name>/<version>/files/...`. The
cache is in `RUTTER_CACHE` or the platform's cache folder, tagged with
`CACHEDIR.TAG`. Once a day the first process to use it removes versions unused
for 60 days, then the least recently used while it holds more than 10 GB, never
a version used in the last day. Only the machine's `paths.yml` may change the
store and limits:

```yaml
pack_store: s3://my-bucket/packs
cache: {max_gb: 50, max_age_days: 120}
```

The environment outranks it: `RUTTER_PACK_STORE`, `RUTTER_CACHE_MAX_GB`,
`RUTTER_CACHE_MAX_AGE_DAYS`. `--offline` (`rutter-plan`, `rutter-paths`; the
pipeline's `RUTTER_OFFLINE`) reads only what is cached and names what is
missing; `RUTTER_VERIFY_DOWNLOADS=0` skips the sha256 check.

| `rutter-data …` | |
|---|---|
| `pull hardware/v1` | Fetch the whole pack into the cache, e.g. before going offline. |
| `pull hardware/v1 --to DIR` | Copy it into a folder the cache never removes, and record it under `packs:` in the machine's `paths.yml`. |
| `publish DIR hardware/v2` | Upload a folder as a new version (files first, the manifest last); an existing version is refused. Paths macOS or Windows cannot hold are refused. S3 needs `uv run --with boto3` and credentials. |
| `verify hardware/v1 [DIR]` | Check a copy, or the cached files, against the manifest. |
| `cache`, `cache clear [PACK]` | What the cache holds against its limits; empty it, or one pack. |

A recipient who is online needs only the config. For one who is not,
`scripts/bundle_config.py CONFIG... --output DIR` (or `X.tar.zst`) copies the
configs as written, their plans, and the files they read under the pack or
`paths:` name each lives in, with a `rutter.paths.yml` at the top that points
those packs and names at the copies. A file named by an absolute path outside
every `paths:` name is listed and refuses the bundle, since it could only be
redirected by rewriting the config.

A relative location anywhere in a config read by `from_yaml` (an asset's
`src`, a transform's `path`, a calibration file) is taken against the config's
folder, not the working directory. A config built in code with
`model_validate` keeps the working directory.

The planner's **Save config** writes the config as it was written with only its
`plan:` block replaced (`replace_plan_block`), so comments, `paths:`, every
`${…}` reference and unexpanded templates survive, and the saved file runs on
any machine the original did.

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
plan:            # PlanningModel — arcs, probes, rig, calibrations, reticles
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
  scene_tags, collidable, mr_signal, …) behave the same as in
  `BulkAssetSpecModel`.
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

## `plan.rig` — the rig's mechanics

```yaml
plan:
  subject_from_rig: {axis_LPS: [1, 0, 0], angle_deg: 14}
  rig:
    kind: arc                     # the default, and the only kind so far
    name: dhc                     # optional; printed by each stage
    ap_deg: [-75, 46]             # arc AP, rig angles; default [-75, 75]
    ml_deg: [-42, 42]             # probe ML on its arc; the default
    min_arc_ap_sep_deg: 16        # least AP between two arcs; the default
    min_intra_arc_ml_sep_deg: 16  # least ML between probes on one arc; the default
```

Every field has a default, so a config that states no rig gets the arc rig
above with symmetric ranges. Both ranges are `[lo, hi]` in rig angles and need
not be symmetric; a range with `lo >= hi` is refused. Rig AP is subject AP plus
the head pitch from `subject_from_rig`, so with the block above every stage
searches subject AP in [-89, +32]. ML is the same in both frames.

The optimizer bounds, the enumeration windows and the coverage ceiling read the
ranges; the enumerator, the seed and both phases' constraints read the
separations, and the arc caps follow from both. A Phase-1 pool records the
separations in its feasibility spec and any range off the default under `rig`,
so Phase 2 refuses a pool built for another rig. The planner never clamps to
any of it, and reports a probe outside a range, or nearer than a separation,
in its kinematic readout and overlay. Loading a plan-only YAML keeps the open
config's rig.

## CanonicalizationDefModel

See [Coordinate Systems](coordinates.md) for semantics. Validation specifics
(in `config/root.py`):

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

## `tags` and `scene_tags`

`tags` describes the thing, `scene_tags` the placement, and the generated
`SceneNodeModel` carries the **union** of both. A node is generated when any of
`transform`, `scene_tags` or `tags` is set, under `auto_scene` (default `True`).

The tags something dispatches on are `domain.enums.KNOWN_SCENE_TAGS`; a config
tag within an edit or two of one of them draws a warning at load. [The config's
vocabulary](vocabulary.md) has what each does and why the chemical-shift and collision decisions are enum
fields rather than tags.

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
      # also_record: []             # further regions scored in coverage;
      #                              # ``target`` still anchors the pose
      # regions: []                 # structure per target, then per
      #                              # also_record entry (retro density masks)
      # past_target_mm: 0.0
      # offsets_RA: [0.0, 0.0]
      # position_bearing_shank: 1   # 1-indexed; multi-shank only — chooses
      #                              # the named shank for tip readouts +
      #                              # ``past_target_mm`` reference
      # calibrated: false
      # auto_scene: true
      # scene_tags: ["probe", "dynamic"]
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
`apply_plan_model_to_state(plan, store)` (`plan_io/replay.py`). Probes
not in the current state are skipped with a stdout warning.

The **Export rig poses** button (`export_plan_geometry`, written to
`<config>.poses.yml`) is a different format: it emits a flat per-probe
summary (`kind`, `target`, `arc`, `angles_deg`, `tip_RAS_mm`,
`depth_from_brain_surface_mm`) for hand-off to physical execution. It's
read-only; there's no loader for it. A loadable plan is `<config>.plan.yml`.

## Known gotchas

- **Template merge "set" trap**: before the `_passthrough_kwargs` rewrite,
  `expand()` methods passed every field explicitly to child models. Pydantic
  marks those as set, so subsequent template merges can't fill in the
  defaults. Always use `model_fields_set` when forwarding from a parent
  model. The `passthrough_kwargs` helper in `config/models_catalog.py` handles
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
| "Unknown template" raised on valid YAML | `_check_template_ref` ordering | `config/root.py` |
| Template values not appearing in expanded spec | `model_fields_set` not respected → use `passthrough_kwargs` | `config/models_catalog.py` |
| Asset loaded but geometry missing in catalog | Loader registered with wrong arity / signature | `build/loaders.py` registry |
| Target at origin / "Missing target for key" warning | Target wasn't in `target_index`; check the `resolve_target_LPS` fallback path | `domain/plan.py` |
| Probe orientation off | Wrong `canonicalization_ref` for probe mesh (LSA vs ASR) | example config + `canonicalizations` block |
| Collisions silently missing pairs | This was the `defaultCollisionCallback` bug | `collision/fcl.py` (per-pair callback) |
| An asset is never collided | `collidable` unset, or neither side of the pair is `role: probe` | `collision/adapter.py` (`pair_bits`) |
| Geometry off by a few mm along AP | wrong `mr_signal` — `water` is what gets corrected | `build/chem_shift.py` |
