# Coordinate Systems

## The rule

**Internal canonical space is LPS millimeters.** Every piece of geometry that
lives in `AssetCatalog`, `Scene`, `PlanningState.target_index`, or any
`Transform`-bound payload is in LPS mm.

The only places RAS shows up are user-facing config/UI boundaries (specifically
named: `point_RAS`, `offset_RAS`, `offsets_RA`). They get converted to LPS the
moment they cross into the runtime.

## Where conversions happen

### File → canonical (build time)

`build_runtime.py` `_load_geo()` runs every loaded asset/target through:

1. **Loader** (`trimesh`, `sitk_volume`, `csv_points`, …) — returns geometry
   in whatever coordinates the file uses.
2. **Canonicalization** (`_apply_canonicalization_mesh` / `_points`) — if a
   `CanonicalizationDefModel` applies:
   - convert orientation (`source_space` → `LPS`) via
     `aind_anatomical_utils.coordinate_systems.convert_coordinate_system`,
   - multiply by `scale_to_mm`,
   - apply optional `transform` (an `AffineTransform` resolved from the
     transforms registry).
3. **Chemical shift** (`ChemShiftContext.pt_transform_for_ppm`) — if the spec's
   role is in `apply_by_role` and the policy is on, shift vertices by the
   per-ppm correction.

After step 3 the geometry is stored in the spec's `mesh` / `points` field and
is **guaranteed in canonical LPS mm** (per the comment at `assets.py:62`).

### RAS → LPS (runtime)

Two specific conversions happen at the planning boundary, both in
`planning.py`:

- `ProbePlan.target_point_RAS` (inline RAS target) → LPS via
  `convert_coordinate_system(ras, "RAS", "LPS")` in
  `_resolve_target_LPS_from_plan`.
- `ProbePlan.offsets_RA` (the (R, A) tuple in mm) → LPS via the same converter
  on a `[R, A, 0]` vector inside `ProbePose.from_planning_state`.

After these, every coordinate inside the runtime is LPS mm.

### LPS → renderer (display time)

The renderer doesn't care about anatomical labels — it gets a 4×4
`model_matrix` from `RendererAdapter._upsert_node` (built by `_rt_to_matrix`
from a composed `(R, t)`). K3D and PyVista apply the matrix as a
GPU-side / actor-side transform; the underlying vertex buffers stay in their
canonical LPS layout.

## Frame composition

The full world transform for a scene node is

```
world(node) = base(node.transform)  ∘  dynamic(probe pose)
```

where:

- `base` is the `TransformChain` declared on the `NodeInstance` (e.g.
  `headframe_to_lps`),
- `dynamic` is the probe's `ProbePose.chain()` if the node is bound to a probe
  via `extras["pose_source_probe"]`, otherwise identity. If the asset has a
  `pivot_LPS`, the dynamic chain is wrapped `T_p ∘ dyn ∘ T_-p` so rotation
  occurs about the pivot rather than the origin.

Composition is implemented in `planning.PoseResolver.world_chain_for_node`.

## CanonicalizationDefModel anatomy

```yaml
canonicalizations:
  obj-wavefront:
    source_space: ASR    # OrientationCode — orientation flip vs LPS
    scale_to_mm: 1.0     # unit conversion (default 1.0)
    transform:           # optional AffineTransform (from transforms registry)
      key: headframe_to_lps
      inverted: false
```

- `source_space: LPS` and `scale_to_mm: 1.0` and no transform → identity
  (geometry is already canonical).
- `source_space: ASR` (typical OBJ Wavefront export) → axis permutation that
  takes ASR → LPS.
- `source_space: LSA` is used for Newscale probe meshes from `aind_mri_utils`
  (see `examples/786864-config.yml`'s `probe-mesh` entry).
- `source_space: FILE_NATIVE` is a sentinel for "the file's native space is
  arbitrary, you must provide a `transform`". Validated by
  `_check_canon_fields` in `config.py`.

## Adding a new source space

The `OrientationCode` enum (`orientation_codes.py`) already lists all 48
permutations. To add support for a fundamentally new frame (one not derivable
from a permutation, e.g. mirrored or rotated atlas):

1. Define an `AffineTransform` mapping that frame → LPS in a transforms
   registry entry (typically loaded from a SimpleITK `.h5` or `.mat`).
2. In your `CanonicalizationDefModel`, set
   `source_space: LPS, scale_to_mm: 1.0, transform: <ref>`.

The orientation step then no-ops, and your custom transform does the work.

## Working in a non-AIND template space

If you want the entire planning frame to be a different anatomical space
(e.g. an MRI template's LPS), there are two routes:

1. **Run natively in the template frame** — leave canonicalizations as
   `source_space: LPS, scale_to_mm: 1.0`. Probes, targets, and meshes all live
   in the template frame, and "LPS" inside the runtime means template-LPS.
   No transform plumbing needed.
2. **Mix with AIND-frame assets** — define a transform
   `template_to_aind_LPS` and reference it in a canonicalization that applies
   to template-frame geometry. AIND-frame geometry uses the existing
   canonicalizations. Both end up in a single LPS frame at runtime.

## Things that look like coordinate names but aren't

- `pivot_LPS` on `BaseSpec` — local-asset-space pivot used for tip rotation,
  not a world-frame coordinate.
- `Float3` etc. — type aliases. They're shape annotations, not frame tags.
- `caps & Capability.RENDERABLE` — capability bitflags, unrelated to coords.
