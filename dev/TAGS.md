# Tags — the config's vocabulary

A tag says what an asset is, where its coordinates came from, or what the
runtime may do with it. Tags are the only such vocabulary: `Role`, the
`Capability` flags, the separate asset-level `tags` list, and the
`collision.group`/`collision.mask` labels each used to answer part of the same
question, and each has been folded into this one.

Tags are open. A subject config may introduce a tag no code reads; the tags
below are the ones something acts on.

## Where a feature's coordinates came from

These decide chemical-shift correction, and they are the only tags whose
misuse moves geometry silently.

| tag | meaning |
|---|---|
| `mr-water` | Localized from the water resonance — brain anatomy, segmented structures, and targets derived from the annotation volume. |
| `mr-fat` | Localized from a vaseline fiducial in the same image — the headframe, the implant, and the bore centres that come with the implant fit. |
| *(neither)* | Not localized from the image at all. Either CAD assumed rigid to the headframe (the cone, the well) or geometry the planner places (the probes). |

**Why the correction applies to `mr-water`.** Fat and water resonate about
3.5 ppm apart, so the scanner, reconstructing at the water frequency, places
fat-derived signal `spacing × ppm × mag_freq / pixel_bandwidth` mm off along
the readout axis. The canonical frame here is the headframe's, and the
headframe is found from vaseline — so the fat-derived features define the
frame and the water-derived ones are translated into it.

**Why the cone and the well are untagged.** Neither is localized in the image.
Their position is assumed from the headframe's, so they arrive in the
canonical frame already and there is nothing to correct. `mr-fat` records a
measurement, not a frame.

Reversing the convention — making the water frame canonical — means moving the
correction to `mr-fat`, which is why the fat set is tagged at all rather than
left implicit.

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

## Changing any of this

`tests/config_semantics.json` records what all five tracked configs resolve to
— per asset, the chemical-shift decision and ppm and the collidability; per
config, the scene nodes and the optimizer's fixture set. A diff there is a
behaviour change. Regenerate it only when the change is the point:

```bash
uv run --python 3.13 python -m tests.config_semantics
```
