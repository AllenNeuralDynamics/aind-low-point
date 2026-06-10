# Target-Valid Visibility Atlas: Design

**Status:** design, not yet implemented.
**Replaces:** the chord-anchored `visibility_atlas.py` for boundary-feasible recovery.
**Supersedes:** the original `target_aligned_atlas_design.md` (corrected framing per agent feedback).
**Date:** 2026-05-18.

## Motivation

The current `visibility_atlas.py` pins probe pose by "centroid at target"
along the chord `target → top_sample`. Six pose DOFs collapse to two
(`top_sample × spin`), and configurations the optimizer's polish reaches
through `offset_R`, `offset_A`, or `past_target_mm` aren't representable.

Empirical failure: the 836656/T12 manual plan has `VM` at
`past_target = -1.5325 mm`. The atlas can't represent that pose at any
`top_sample`, so the manual basin gets clipped — even though the manual
is strictly feasible under the optimizer's actual constraints
(`g ≤ 1`, all pairwise clearances ≥ 0; min global clearance +0.113 mm).

The `oval_slack=0.2` knob masks some boundary clipping but not this one.

The fix is structural: switch from brute-force sampling over
`AP × ML × spin × top × offset × depth` to a geometry-driven
feasible-domain pipeline:

```
project hole-section ellipses from a target or near-target apex
  → intersect projected ellipses  (centerline feasibility region)
    → sample only feasible centerline directions
      → check spin only on feasible directions
        → attach depth / coverage / offset anchors
          → AP-binned anchors feed arc-first search
```

## Naming: target-valid, not target-aligned

Earlier framing said "the insertion line passes through `T`."  That's a
fine first implementation but too restrictive as an invariant.

The full optimizer can shift the shank-row centerline laterally:

```
centerline passes through  T + Δ   (small lateral offset)
active region still covers T       (within the section bounds)
```

So the long-term invariant is **target-valid**: the pose places the
recording bank usefully relative to `T`, not that the centerline
intersects `T` exactly.

- First implementation: apex = `T`. (Solves the `past_target` problem,
  which is along-line; doesn't yet solve lateral offset.)
- Data model: every anchor records `source_apex_offset` so future builds
  with a small ring of near-target apexes don't need an API change.

## Three pose-validity levels

The atlas should approximate **level 2**, not 1, so arc-first search can
reason about **level 3**.

1. **Bore feasibility.** "Does the probe physically pass through the
   hole?" Insufficient — a pose can thread but miss the target.
2. **Target-valid feasibility.** "Does the probe thread the hole *and*
   place the active recording region usefully on the target?" The right
   per-(probe, hole, AP) criterion.
3. **Joint rig feasibility.** "Can all probes do level 2 while sharing
   arc APs, separating MLs, satisfying inter-arc AP separation, and
   avoiding pairwise collision?" Handled by arc-first search.

## Stage A: centerline feasibility region

For probe target (or near-target apex) `T` and a hole with sections
`s_1, ..., s_N` (planes `P_i`, ellipses `E_i`):

A ray from `T` in direction `d` is a *centerline feasibility candidate*
if it crosses every `E_i`. Each `s_i` defines an elliptical view cone:

```
Cone_i = { d : line(T, d) ∩ P_i ∈ E_i }
```

Feasible directions are the intersection:

```
FeasibleDirs(T, hole) = ⋂_i Cone_i
```

Closed form: pick a parameterization plane `P_param`. Perspective-project
each `E_i` from apex `T` onto `P_param`. The result `E_i^param` is an
ellipse (the perspective projection of an ellipse from a point is a
conic; bounded → ellipse for the geometry we care about). Then:

```
CenterlineRegion = ⋂_i E_i^param        (convex, bounded by ellipse arcs)
```

### Sampling the region

- Smallest projected ellipse is the **rejection envelope** (always
  contains the intersection).
- Concentric-ring or low-discrepancy samples inside the envelope.
- Reject points outside any other `E_i^param`.
- Add jittered samples near the intersection boundary — manual-quality
  plans often sit near the threading edge.

Each accepted 2D sample `p` defines `d = (p − T) / ‖p − T‖`, which
converts to rig angles `(AP(d), ML(d))`.

**Be permissive.** False positives are cheap (Stage B drops them); false
negatives lose manual basins forever.

## Stage B: spin + multi-shank check

Stage A provides centerline directions. Stage B checks the
spin-and-shank constraint per direction.

For a K-shank probe at pose `(d, spin)`:

- Model the centerline as the **shank row centroid** through `T`.
- Shank `k`'s tip in probe-local frame: `tip_k = centroid + δ_k`.
- At pose, shank `k`'s line is parallel to `d`, offset by
  `R(d, spin) · δ_k`.
- For each `(shank k, section i)`: line-plane intersection, then
  ellipse-membership. Accept if all `K × S` tests pass.

`spin` rotates the K lateral offsets around `d`; the centerline
direction is independent of spin.

### Single-shank shortcut

`K = 1`: spin doesn't affect threading. Skip the spin axis entirely;
each accepted Stage A sample is an atlas anchor with `spin` unconstrained.

### Don't pre-shrink Stage A by shank half-width

Earlier draft proposed shrinking each `E_i^param` by the shank-row
half-width to prune Stage A. **Skip this in the first cut.** Worst-case
lateral offset is too conservative on anisotropic slots — some spins fit
where others don't, and a worst-case shrink loses those. Use permissive
Stage A + explicit Stage B shank checks. Only specialize if Stage B
acceptance rates are too low.

### Optional future: closed-form spin feasibility

For fixed `d`, each shank hit point as a function of spin is sinusoidal:

```
hit(spin) = centerline_hit + A cos(spin) + B sin(spin)
```

Ellipse membership is then a quadratic in `(cos spin, sin spin)`, so
each `(shank, section)` test defines feasible arcs on the unit circle.
The feasible spin set is the intersection of those arcs. Replaces the
spin grid with closed-form arc intersection. Not necessary for first
implementation.

## Depth and lateral offset: separate from threading

A line threads or doesn't thread — translating the active region
along a chosen shaft doesn't change the bore intersection. So:

- Stage A + B determine **threadable line geometry**.
- A separate "anchor decoration" step picks **where the recording bank
  sits along the line** (depth / `past_target_mm`).

For each accepted line, store one or more depth anchors:

- target at active center (`past_target = 0`)
- target slightly proximal / distal
- manual-like under-inserted / past-target offsets
- best-coverage depth (closed-form or quick line search)

Or store a feasible depth interval for target coverage.

For lateral offset: the first implementation uses apex `T` only. If the
manual diagnostic shows configurations off the centerline-through-T
slice, extend the apex set:

```
apex ∈ { T, T ± Δ_R, T ± Δ_A, ... }
```

Each near-target apex generates its own Stage A region. Every emitted
anchor records `source_apex_offset` so downstream code can distinguish.

## Atlas data model

```python
@dataclass(frozen=True, slots=True)
class PoseAnchor:
    probe_name: str
    hole_id: int

    # Rig pose
    ap_deg: float
    ml_deg: float
    spin_deg: float | None     # None for K=1

    # Translation
    off_R_mm: float
    off_A_mm: float
    depth_mm: float
    past_target_mm: float | None

    # Quality metrics
    threading_max_g: float     # min margin; ≤ 1 ⇒ feasible
    threading_margin: float    # 1 − threading_max_g
    coverage: float
    target_miss_mm: float
    offset_norm_mm: float
    robustness: float          # e.g. ML-perturbation slack

    # Provenance
    source_apex_offset_R_mm: float
    source_apex_offset_A_mm: float
    source_sample_id: int


@dataclass(frozen=True, slots=True)
class AtlasEntry:
    probe_name: str
    hole_id: int
    ap_bin: float
    anchors: tuple[PoseAnchor, ...]   # ≥ 1 anchor per (probe, hole, AP bin)
```

Multiple anchors per AP bin matter because the
maximum-coverage anchor is often a poor ML match for other probes on the
same arc — keep a lower-coverage but better-separated alternative.

## Cost comparison

| Stage | Current `visibility_atlas` | Target-valid atlas |
|---|---|---|
| Sampling domain | top oval (raw, 2D) | centerline feasibility region (2D, convex) |
| Sampling waste | many samples can't thread the bottom | every sample threads centerline by construction |
| Per-sample inner check | K × S tests at fixed spin grid | K × S tests at fixed spin grid (same kernel) |
| Pose DOF coverage | 2 (top, spin), offsets/depth pinned | 3+ via apex grid + depth decoration |
| Manual T12 VM recovery | misses (manual off the pinned slice) | hits (manual centerline lives in the projected-ellipse intersection) |
| Permissiveness | shrunk by `oval_slack` margin | permissive Stage A, explicit Stage B |

## Numerical / geometric cautions

- Keep all sections in Stage A initially — only drop intermediate sections
  after confirming they're redundant against top + bottom on real
  `holes.yml` data.
- Apex `T` near a section plane: perspective projection degenerates.
  Guard with a minimum distance / angle check.
- Use a permissive numerical tolerance on ellipse membership.
- Conic-form perspective projection is the right long-term implementation;
  sampling the source ellipse boundary and fitting a conic to the
  projected points is fine for a diagnostic prototype.
- Do not tighten thresholds just to shrink the pool. If the manual drops
  out, fix the atlas; don't filter harder.

## Integration with arc-first search

For a partition of probes into arc groups `{G_0, G_1, ...}`:

```
for each arc group G_j and AP bin a:
    LocalConfigs[G_j, a] = assignments of probes in G_j to
        distinct holes + anchors such that:
            every (probe, hole) has an atlas anchor at AP bin a
            intra-arc ML separation ≥ min_ml_sep_deg
            coverage / robustness above floor

    a is "supported" for G_j iff LocalConfigs is non-empty
```

Then pick `(a_0, a_1, ...)` with pairwise AP separation ≥
`min_arc_ap_sep_deg`, maximize aggregate score over the local-config
combinations.

**Arc APs are read off atlas support, not inferred from best-fit pose
clustering.** Best-fit AP clustering has been falsified by data
([[session_2026-05-14_optimizer_search_fix]]); atlas support is the
correct invariant.

## Manual-plan diagnostic gate

Before integration, every implementation pass must run a manual-membership
diagnostic on 836656/T12:

For each manual `(probe, hole)`:
- manual AP / ML / spin / depth
- nearest atlas anchor's AP / ML / spin / depth
- threading margin, coverage, target_miss at that anchor
- offset needed to reach manual exactly

Classify any failure as one of:

1. Centerline domain missing → Stage A region too small
2. Spin grid missed manual spin → coarser spin OK or use closed-form arcs
3. Multi-shank Stage B failed → re-evaluate shank model
4. Coverage / depth missing → extend depth anchors
5. Lateral offset needed → add near-target apex grid
6. AP/ML binning too coarse → finer bins near manual
7. Coverage threshold too strict → loosen the anchor accept threshold
8. Numerical tolerance issue → widen permissive ε

The atlas ships only when the manual basin is representable or the
failure is precisely diagnosed and explained.

## Things explicitly ruled out

Per prior diagnostics:

- AP × ML × spin brute-force atlas sampling
- H-only pre-cost ranking
- per-probe-first candidate generation
- Best-fit pairwise clearance ranking
- Best-fit AP clustering
- Bore-only feasibility with no target validity
- Counting a pose feasible just because it threads through the hole

Threading through the implant is necessary but not sufficient. The active
bank must also sit in a useful relationship to the target.

## Implementation order

1. New file: `src/aind_low_point/optimization/target_valid_atlas.py`.
   Side-by-side with `visibility_atlas.py`; don't modify the existing
   atlas yet.
2. Apex `T` only for the first diagnostic.
3. Implement perspective ellipse projection + intersection-region
   sampling on `P_param`.
4. Stage B: spin + K × S shank check, vmappable over (centerline, spin).
5. Emit `PoseAnchor` with the full field set, including provenance
   fields, even when first build leaves `source_apex_offset = 0`.
6. New diagnostic script: `scripts/diagnose_target_valid_atlas.py`.
   Mirrors `diagnose_local_arc_configs.py`; runs the manual-membership
   gate.
7. If manual clips, add the small near-target apex grid
   `{T, T ± Δ_R, T ± Δ_A}`.
8. Add depth / `past_target` anchors (or a feasible-interval anchor) so
   downstream polish gets multiple seeds per AP bin.
9. Wire atlas anchors into per-arc local-config generation.
10. Arc-first search picks partition + AP bins + holes + anchors using
    atlas support.
11. Selected candidates feed Phase 1 pool optimization and Phase 2 hard-constraint
    polish.

## Related

- `docs/source/optimization.rst` — maintained optimizer pipeline guide.
- [[manual_plan_files]] — manual plan loaded as
  `config + plan-overlay` pair; clearance check uses FCL BVH on full
  collision mesh.
- `src/aind_low_point/optimization/enumeration/visibility_atlas.py` — incumbent
  atlas, kept for comparison; do not edit.
- `src/aind_low_point/optimization/enumeration/atlas.py` — atlas carrier
  dataclasses shared by atlas builders and the pipeline.
