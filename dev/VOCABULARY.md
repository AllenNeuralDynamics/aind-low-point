# The config's vocabulary

A config makes two different kinds of statement, and they want different
machinery.

A **classification** has exactly one value from a closed domain — which
resonance localized this feature, what kind of geometry it is. A wrong answer
corrupts geometry, so these are **enum fields**, and a misspelling is refused at
load.

A **membership** is zero or more from an open set — which display group a node
joins, whether it is a fixture. A missing one degrades a view. These are
**tags**, and a subject config may introduce a tag no code reads.

The old vocabularies failed by mixing the two: `Role` was asked both what a
feature is *for* and where its coordinates came *from*, which are independent —
a target may be an annotation centroid (water) or a bore centre (fat). That
is what the per-asset chemical-shift overrides were working around.

## Where a feature's coordinates came from — `mr_signal`, not a tag

This one decides chemical-shift correction, so it is a **closed enum field**
rather than a tag: a misspelled tag loads silently and moves geometry, while
`mr_signal: wter` is refused at load.

| `mr_signal` | meaning |
|---|---|
| `water` | Localized from the water resonance — brain anatomy, segmented structures, and targets derived from the annotation volume. |
| `fat` | Localized from a vaseline fiducial in the same image — the headframe, the implant, and the bore centres that come with the implant fit. |
| `none` | Not localized from the image at all. Either CAD assumed rigid to the headframe (the cone, the well) or geometry the planner places (the probes). |

**A config with an `imaging` block must declare it on every asset and target**;
one without — an atlas-based plan — declares nothing, because there is no image
to be shifted in. Templates carry most of the declarations: `mr_signal: water`
on the `structure` template covers every structure it expands to.

The ppm is global. `ImagingModel.chem_shift_ppm_default` is what every config
sets; the per-asset `chem_shift_ppm` override exists and no config uses it.

**Why the correction applies to `mr-water`.** Fat and water resonate about
3.5 ppm apart, so the scanner, reconstructing at the water frequency, places
fat-derived signal `spacing × ppm × mag_freq / pixel_bandwidth` mm off along
the readout axis. The canonical frame here is the headframe's, and the
headframe is found from vaseline — so the fat-derived features define the
frame and the water-derived ones are translated into it.

**Why the cone and the well are `none` rather than `fat`.** Neither is
localized in the image. Their position is assumed from the headframe's, so they
arrive in the canonical frame already and there is nothing to correct. `fat`
records a measurement, not a frame.

Reversing the convention — making the water frame canonical — means moving the
correction to `fat`, which is why the fat set is named at all rather than left
implicit.

## What the thing is

| tag | acted on by |
|---|---|
| `probe` | the kinematic pivot (`runtime/build.py`), the collision pair filter, the trame probe lists |
| `well` | the thick-well SDF substitution (`pipeline/thick_well.py`) |
| `implant` | excluded from the optimizer's fixture set — probes thread *through* its bores |
| `fixture`, `cone`, `headframe` | included in the optimizer's fixture set |
| `bore` | a target that is a bore centre rather than an anatomical point |
| `brain`, `structure` | anatomy, for display grouping |

## What may happen to it

| tag | meaning |
|---|---|
| `collidable` | gets an FCL body. A pair is tested when both sides are collidable and at least one is a `probe`; fixtures do not collide with each other. |
| `static`, `dynamic` | whether the node's transform changes as a plan is edited |

## Asset templates

A probe *kind* asset (`probe:2.1`) is geometry the planner instances, not a
placement. It carries `auto_scene: false` so that tagging it does not generate
a scene node; the nodes that get posed come from `plan.probes`.

## Upgrading a config written before `mr_signal`

```bash
uv run --python 3.13 python scripts/upgrade_config_mr_signal.py \
    --dry-run examples/*_out.yml
```

`water` is derived from what the old `role` rule resolves to for that very
file, so the decision cannot move; the script re-reads the result and restores
the original if any key's answer changed. `fat` is assigned by key and is
documentation only — `fat` and `none` behave identically. The superseded
`chem_shift_policy` and `chem_shift_apply_by_role` are removed.

`ConfigModel.from_yaml(path, require_mr_signal=False)` loads an un-upgraded
config. Only the upgrade path passes it.

## Changing any of this

`tests/config_semantics.json` records what all five tracked configs resolve to
— per asset, the chemical-shift decision and ppm and the collidability; per
config, the scene nodes and the optimizer's fixture set. A diff there is a
behaviour change. Regenerate it only when the change is the point:

```bash
uv run --python 3.13 python -m tests.config_semantics
```
