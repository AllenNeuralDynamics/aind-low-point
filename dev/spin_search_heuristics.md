# Spin-search heuristics for Stage 2/3 multi-basin recovery

**Status**: design notes, 2026-05-26. Captures domain-derived heuristics
for the spin multi-modality problem and a proposed codification. Not
yet implemented.

## Problem

`spin_deg = atan2(sy, sx)` parameterizes a 1-D angle with 2 DOF. Each
probe has multiple physically-distinct spin orientations (180° flips,
sometimes 90°/270° for non-quadbase). Gradient methods only find the
basin their warm-start lies in. Stage 2's L-BFGS-B picks one local
optimum per cand; for some cands this is FCL-infeasible while a
different spin basin would be feasible.

The 2026-05-26 unit-circle penalty fixed ~20% of failures by removing
the (sx, sy) magnitude wander. The residual ~30% failures are likely
true spin-basin issues that require explicitly choosing a different
basin.

## Domain heuristics (from manual planning experience)

**H1 — Threading constrains 4-shank spin to ~0° or ~180°.**
A quadbase probe has 4 shanks in a line. Each shank must enter its
corresponding section of the guide hole. The 4-section row has a
fixed orientation per hole (the "slot major axis"); the probe spin
must align with that axis. Two valid orientations exist: aligned
(θ_slot) or flipped (θ_slot + 180°). No other angle threads.

For single-shank probes the shank threads at any spin; only the body
geometry constrains the choice via H2/H3.

**H2 — Close pairs: optimal spin is geometric, not a heuristic.**
Each probe body has a known asymmetry vector in its local frame (e.g.
for quadbase the 4-shank row defines the "long axis"; for NP 2.x the
headstage flex/dovetail defines a primary direction). When two
probes are within body-body interaction distance, the spin that
maximizes their separation is **the one that rotates each probe's
long axis to be perpendicular to the gap direction**, so each probe
presents its NARROW PROFILE to the other.

This is computable per-cand:

```
gap_dir_world = (world_pose_B.position - world_pose_A.position) normalized
optimal_long_axis_A_world = any vector ⊥ gap_dir_world
optimal_spin_A = the spin angle that rotates A's local long-axis
                 vector into optimal_long_axis_A_world
```

The PER-PAIR optimal isn't a hand-tuned 180° preference — it's a
closed-form geometric calculation given:
  - the gap direction (from targets / world positions)
  - each probe's body-asymmetry vector in its canonical local frame
    (precomputable per probe type)
  - each probe's current (ap, ml) pose (which orients the local
    frame in world)

For symmetric bodies (no long axis), the result is degenerate and
spin doesn't matter for that pair.

The continuous optimal spin is then **quantized to the H1-allowable
set** (for quadbase: snap to whichever of {θ_slot, θ_slot + 180°} is
closer to the geometric optimum).

**H3 — Triangle: outsider rotates perpendicular to pair axis.**
For 3 probes in a tight cluster where (a, b) are a close pair and
c is within interaction range of both, c's spin should be ≈
perpendicular to the (a, b) axis. This minimizes c's profile in the
direction of both a and b.

**H4 — Far-field decoupling.**
Beyond some pairwise distance threshold, spin doesn't matter for the
pair's clearance. Don't enumerate spin alternatives for pairs that
are far-decoupled.

## Codification

### Stage 1: per-probe candidate generation

```python
def per_probe_spin_candidates(probe, hole, kind):
    """Discrete spin candidates for one probe. H1 enforced here."""
    slot_major = π/2 - hole.slot_theta_rad  # current warm-start
    if kind.startswith("quadbase"):
        # 4-shank: must align section row → exactly 2 options
        return [slot_major, slot_major + π]
    else:
        # 1-shank: spin free for threading; offer 4 orientations
        # so body asymmetry has options under H2/H3
        return [slot_major + k * π/2 for k in range(4)]
```

### Stage 2: structure-aware combination generation

```python
def build_spin_assignments(probes, statics, target_LPS,
                           D_close_mm=10.0, D_interact_mm=15.0,
                           D_far_mm=25.0, beam_B=64):
    # 1. Build coupling graph from target geometry
    coupling = [
        (i, j) for i, j in itertools.combinations(range(K), 2)
        if np.linalg.norm(target_LPS[i] - target_LPS[j]) < D_interact_mm
    ]

    # 2. Group into connected components — components are independent
    components = connected_components(coupling)

    # 3. For each component, beam-search per-probe spins using H2/H3
    component_solutions = []
    for comp in components:
        beam = [({}, 0.0)]  # (assignment, score)
        for probe_idx in order_by_degree(comp, coupling):
            new_beam = []
            for partial, partial_score in beam:
                for spin in per_probe_spin_candidates(...):
                    s = score_partial(
                        partial | {probe_idx: spin},
                        coupling, target_LPS,
                        D_close_mm, D_far_mm,
                    )
                    new_beam.append((partial | {probe_idx: spin}, s))
            beam = top_k(new_beam, beam_B)
        component_solutions.append(beam[:3])

    # 4. Product over components (independent → no coupling)
    return list(itertools.product(*component_solutions))


def score_partial(assignment, coupling_edges, target_LPS,
                  D_close_mm, D_far_mm,
                  body_asymmetry_local, ap_ml_per_probe):
    """Score uses the geometric optimal spin per pair, not a generic
    'prefer 180°' preference."""
    score = 0.0
    for i, j in coupling_edges:
        if i not in assignment or j not in assignment:
            continue
        d_target = np.linalg.norm(target_LPS[i] - target_LPS[j])
        if d_target > D_far_mm:
            continue                                  # H4: no contribution
        # H2 geometric: compute optimal spins per probe.
        gap_dir = normalize(target_LPS[j] - target_LPS[i])
        opt_spin_i = optimal_spin_perpendicular(
            body_asymmetry_local[i], ap_ml_per_probe[i], gap_dir)
        opt_spin_j = optimal_spin_perpendicular(
            body_asymmetry_local[j], ap_ml_per_probe[j], -gap_dir)
        # Reward closeness to the geometric optimum
        score += np.cos(assignment[i] - opt_spin_i)
        score += np.cos(assignment[j] - opt_spin_j)
    # H3: triangle term — for any (i, j, k) where all three are mutually
    # close, the third probe's geometric optimum is "perpendicular to
    # cluster centroid axis" — handle as a 3-body version of the same
    # rule (omitted here for brevity; same optimal_spin_perpendicular
    # idea with multiple gap directions averaged).
    return -score  # minimizing


def optimal_spin_perpendicular(local_asym_vec, ap_ml, gap_dir_world):
    """Spin that rotates probe's local long-axis to be perpendicular
    to the gap direction. Closed-form."""
    # 1. The local-frame asymmetry vector, transformed by (ap, ml),
    #    becomes a function of spin: long_axis_world(spin)
    # 2. We want long_axis_world(spin) ⊥ gap_dir_world
    # 3. Solve for spin analytically (the spin only affects rotation
    #    about the probe's local +z axis, so the equation is 1D in
    #    spin).
    ...
```

### Stage 3: full-DOF polish per kept assignment

Take the top-K spin assignments per Stage 2 cand. For each, build a
phase1_x with the chosen spins (other DOF inherited from the existing
augmented warm-start), run L-BFGS-B Phase 1 + trust-constr Phase 2
+ FCL validator. Keep the best per cand by lex_key.

## Scoping: when to apply

**Apply to top-K output of Stage 2, not the whole 8908-cand pool.**

The user's intuition: spin search is wasted on cands that are
infeasible for OTHER reasons (e.g. probe-probe distance too small to
ever clear, threading geometry hopeless, etc.). Past some infeasibility
threshold there's no point investigating spin alternatives.

Proposed gating:
- Run Stage 2 polish + offset-augmentation + violation-eval as today
- Take cands with `violation_fn < CUTOFF` (CUTOFF ≈ 200-500, capturing
  the band where some cands are FCL-feasible per current chain)
- For each, apply the spin-heuristic generator
- Polish + validator on each generated assignment

Per cand: ~10-30 polishes (cluster-dependent) vs the naive 2^K ≈ 128.

For 836656/T12 with ~500 cands at violation_fn < 200 and ~20 polishes
per cand at ~5s each = 50,000s / 8 workers = ~1.7h. Tractable.

## Comparison to alternatives

| Approach | Per-cand cost | Coverage | Effort |
|---|---|---|---|
| Spin penalty alone (today) | +0% | H1 partial, H2/H3 not addressed | landed |
| Per-probe 2-flip enumeration | 2^K = 128 polishes | H1 only | medium |
| CMA-ES on (sx, sy) | ~10s per cand | All H1-H4 stochastically | medium |
| **Heuristic beam search** | 10-30 polishes (cluster-dependent) | H1+H2+H3+H4 by construction | high |

The heuristic approach is most expensive in implementation but lowest
in compute and highest in interpretability — failures point at which
heuristic group is incomplete.

## Risks / things to calibrate

1. **Distance thresholds** (`D_close`, `D_interact`, `D_far`) need
   sweeping on a few cands. Should come from looking at when
   body-body distance starts correlating with spin sensitivity in
   polish outcome.

2. **The "facing" preference (H2: Δspin ≈ 180°) might be wrong some
   of the time** for specific headstage geometries. Worth empirically
   verifying on diverse cands before hard-coding the cos shape.

3. **H3 assumes the headstage profile is wider than it is tall.** If
   not strictly true, "perpendicular" isn't the right direction.
   Could parameterize as "rotate to align with the principal axis of
   the cluster's bounding ellipse."

4. **Connected-components decomposition** assumes coupling is mostly
   local. If a fully-connected 5+ probe cluster appears, beam search
   alone won't beat enumeration in that component — would need
   iterated local search inside the cluster.

5. **Triangle term coupling weight (0.3)** is a guess. Tune by sweeping
   on cands with known triangle structures.

## Validation plan

Pick 5–10 cands the spin-only-penalty couldn't fix (e.g. 1044, 5275,
4217, 6235, 1290 from the post-penalty failure set), apply the
heuristic generator to each, run the chain on each generated assignment,
check FCL feasibility.

- If any reach FCL feasibility via a heuristic-generated assignment →
  approach validated; codify and scale to top-K.
- If none do → heuristics incomplete, need different approach (CMA-ES
  on (sx, sy) or full-DOF MLSL).

## Open questions

- Should H2/H3 use body-body distance (more accurate) or target-LPS
  distance (cheaper, doesn't require pose computation)? Target distance
  is a reasonable proxy since probes typically point AT their targets,
  but pose-after-Stage-2 might be more informative.
- Should the candidate set for single-shank probes be continuous (free
  parameter optimized within the polish) instead of discrete (4
  quadrants)? Continuous loses the "basin enumeration" property.
- How does this interact with the existing offset-polish augmentation?
  The augmentation polishes offsets at the polished spin; if we
  generate a new spin, should we re-do the offset polish from there?
  Probably yes — the offsets that were optimal for the old spin won't
  be optimal for a flipped spin.
