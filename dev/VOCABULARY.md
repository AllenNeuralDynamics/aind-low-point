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

`collidable: true` on an asset gives it an FCL body — a bool, not a tag, so a
misspelling is refused. **Which pairs are then tested is a rule, not data:**
both sides collidable, and at least one with `role: probe`. Probes hit fixtures
and each other; fixtures do not hit each other.

That replaced per-asset `collision.group` / `collision.mask` label lists
compiled to bitmasks. Across every tracked config those labels only ever
expressed the two patterns the rule states.

| tag | meaning |
|---|---|
| `static`, `dynamic` | whether the node's transform changes as a plan is edited |

## Where a probe's electrodes are — `recording`

| form | meaning |
|---|---|
| omitted | resolve from the built-in table by kind; a kind that is not in it is refused at load |
| `recording: {active_ranges_mm: [[0.2, 3.065]], shank_pitch_mm: 0.25}` | this probe's active bank, one `(start, end)` in mm from the tip per shank |
| `recording: none` | no recording array: the probe targets with its tip, as a pipette does |

Before this, `RECORDING_GEOMETRY` in `optimization/geometry/recording.py` was
the only source, so a new holder variant needed a code change — and three of
its five entries were exactly that, the same NP 2.0 silicon under different
holders. An unregistered kind also resolved silently to tip-on-target, which is
the answer a pipette gets deliberately, so a mistyped kind was indistinguishable
from a legitimate array-less probe.

## Asset templates

A probe *kind* asset (`probe:2.1`) is geometry the planner instances, not a
placement. It carries `auto_scene: false` so that tagging it does not generate
a scene node; the nodes that get posed come from `plan.probes`.

## Upgrading a config to the current schema

```bash
uv run --python 3.13 python scripts/upgrade_config.py --list
uv run --python 3.13 python scripts/upgrade_config.py --dry-run examples/*_out.yml
```

Each migration rewrites the YAML text, so comments and `${...}` interpolations
survive. A run is accepted only if the config's behaviour is unchanged — the
same chemical-shift decision and ppm per key, the same collidability, the same
set of colliding pairs — and the original is restored otherwise.

**Run it before taking a version that deletes the superseded fields.** The
migrations read the old fields to derive the new ones, so a config can only be
upgraded while the code still understands it.

For `mr_signal`, `water` is derived from what the old `role` rule resolves to
for that very file, so the decision cannot move. `fat` is assigned by key and
is documentation only — `fat` and `none` behave identically. The superseded
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
