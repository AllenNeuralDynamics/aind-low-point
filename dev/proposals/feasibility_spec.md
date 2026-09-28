# One feasibility definition for every stage (proposal, 2026-09-27)

Every stage that tests a plan should ask the same question. Today five code
paths answer it, with three thresholds, two clearance queries and two sets of
gains, and nothing ties them together, so experiments keep reading a Phase-1
"clear" as a Phase-2 "feasible" when they are different tests.

Decided by the user: clearance is 0.1 mm between real surfaces, read typically
(soft minimum plus its measured shortfall) in both phases. Phase 1 keeps
checking the well alone, since the headframe and cone cost VRAM and block
almost nothing (below). Duplicated code stays only where JAX performance
requires it, and each such copy says so at the site.

## What each stage tests today

| Stage | Code | Pair query | Clearance test | Gains (pair / fixture) | Fixtures |
|---|---|---|---|---|---|
| Spin restore | `search/spin_restore.py` | old: all body samples, shank corners, SAT | soft ≥ **0.0** (`JointWeights`) | all 1 | well |
| Reduced, full stages | `objectives/soft.py` | swept: coarse-to-fine body, dense box, SAT | soft ≥ **0.1** (`Phase1Weights` default) | (1,1,1,100) / (1,100) | well |
| Pool `min_clear` score, production's Phase-2 ranking | `objectives/metrics.py` | old, unrolled | hard min, reported | — | well |
| Phase 2 | `objectives/constrained.py` | swept | soft + bias ≥ **0.2** (`Phase2Settings.minclear`) | (1,1,g,g) / (1,g), g = `obb_gain` | headframe, cone, well |
| Handoff | `pipeline/handoff.py` | FCL, true meshes | ≥ −1e-4 strict, ≥ −0.2 kept | — | all, with implant |

Also:
- `objectives/reduced.py` builds a third objective with the old query; production calls only its `extract_arrays`.
- `objectives/threading.py` holds a fourth per-candidate objective that nothing calls.
- Threading tolerance, arc and ML separation, brain margin and shaft length are declared in `JointWeights`, `Phase1Weights` and `Phase2Weights`. They agree only because they were copied.
- `Phase2Weights.min_clearance_mm` defaults to 0.0, so any caller that builds it without the setting tests contact, not 0.1.
- Bias, as used above: `PROBE_PAIR_SOFTMIN_BIAS_MM` (0.10, 0.09, 0, 0) and `FIXTURE_PAIR_SOFTMIN_BIAS_MM` (0.05, 0.06), measured at β 20 with top-k 16/8/8 (X76, X78).

## Design

1. **`FeasibilitySpec`**, a frozen dataclass in a jax-free module
   (`objectives/feasibility.py`), so settings and the handoff can import it.
   - Fields:
     - `min_clearance_mm`
     - `softmin_beta` and the three top-ks
     - `threading_tolerance` and `shaft_length_mm`
     - `min_arc_ap_sep_deg` and `min_intra_arc_ml_sep_deg`
     - `brain_margin_mm`
   - The bias constants move beside it, tagged with the β and top-k they were measured at. A spec with the correction refuses any other β or top-k, so changing either forces a re-measure.
   - `JointWeights`, `Phase1Weights` and `Phase2Weights` drop those fields, hold a required `spec`, and keep only their own λ's.
   - `weights_cache_key` recurses into it.

2. **One slack function per condition.** Each returns mm or degrees, and ≥ 0 means feasible:
   - `pair_clearance_slack`: soft + bias − min_clearance
   - `fixture_clearance_slack`
   - `threading_slack`: tol − g
   - `brain_slack`
   - `arc_separation_slack`
   - `ml_separation_slack`

   Phase 1 penalizes `(gain · relu(−slack))²` and Phase 2 constrains `gain · slack ≥ 0`, so Phase 1's zero-penalty set is Phase 2's feasible set by construction. The gains stay solver scaling, from one function in `aggregate.py` that takes Phase 2's `obb_gain`.

3. **One pair query.** Factor the per-pair body of `swept_pair_clearances` into `pair_clearance(Ra, ta, Rb, tb, ka, kb, tables, ...)`, and route every caller through it:
   - The sweep is its vmap.
   - The spin restore calls it with the swept probe's trial pose.
   - `min_clear` becomes the swept query, vmapped, not the unrolled one.

   The query functions lose their β and top-k defaults, so a caller cannot silently fall back on a value the spec does not hold.

4. **One setting, pinned in the pool.**
   - `PipelineSettings.min_clearance` (`RUTTER_MIN_CLEARANCE`) replaces `Phase2Settings.minclear`, and `RUTTER_MINCLEAR` is refused by name.
   - Phase 1 writes the spec into the pool header beside `well`, and Phase 2 refuses a pool whose spec differs.
   - The handoff records the spec.
   - The pool's score becomes the spec's worst slack. Phase 1's "feasible" is then Phase 2's, evaluated at Phase 1's pose.

5. **Named exceptions.** Each carries a comment at its site giving the reason:
   - **Phase 1 checks the well alone** (VRAM). The spec names the fixtures, and Phase 1 declares its subset.
     - In X67, among 4096 Phase-1 poses, the cone was the only collision keeping 4 from FCL-clear, the well 3, and the headframe none. Every headframe contact sat beside a well contact.
     - Split by what the start collided with, Phase 2 ended strict from 75–100% of starts that hit only a fixture (few per pool), against 31–60% of starts that hit only another probe (X80–X86 solves; not yet an X entry).
     - Optionally, the pool's final score reads all fixtures, so its flag matches Phase 2's even though the stages do not.
   - **The restore evaluates only the pairs touching the probe it sweeps** (K − 1 of them, not all). The argmin is identical. If the swept query proves too slow there (to be measured), the restore keeps a cheaper query and says so. It only picks a spin basin, and the pool score re-reads every candidate with the full query.
   - **Phase 1's coarse passes read fewer samples** (X57). The final score is at full fidelity.
   - **Phase 1's straight-through SAT gradient** changes the gradient, not the value, so no test moves.

6. **Deleted.**
   - `threading.py`'s dead objective: `_signature`, `_build_jit`, `pack_statics` and the cache.
   - `reduced.py`'s objective. `extract_arrays` moves to `packing.py`, and `test_hole_walls`' wall check runs through the restore's loss.
   - `dual_rep_pair_clearance` and `pairwise_signed_clearance_dual(_world)` once nothing calls them.
   - Every stage-local copy of a threshold.

7. **Tests.**
   - On one small problem, Phase 1's per-(pair, category) violations and Phase 2's negative rows name the same entries, on the well rows both check.
   - Phase 1's penalty is zero exactly when every spec slack is ≥ 0.
   - Each weights class refuses construction without a spec.
   - `test_jit_cache_keys` covers the spec's fields.

8. **Vocabulary**, in code, docs and experiment notes:
   - **feasible**: every spec slack ≥ 0 under the model.
   - **collision-free**: FCL ≥ −1e-4 on the true meshes, implant included. The −1e-4 becomes a named constant that the handoff and the scratch scorers import rather than restate.
   - **strict**: collision-free and threading g ≤ 0.
   - **kept**: the handoff's band, flagged as a band.

## Order of work

1. **Structure, bitwise.**
   - Items 1–3 and 6, with today's values carried as three explicit specs in the drivers: restore 0.0, Phase 1 0.1 without bias, Phase 2 0.2 with bias. The discrepancy is then visible in one place.
   - The spin restore and `min_clear` keep their old query in this step. Their numbers move only in step 2.
   - Checks: the parity harness for Phase 2, a Phase-1 trace parity on one chunk, and the suite.
2. **One spec.**
   - Collapse to a single spec (0.1, typical correction) and switch the restore and `min_clear` to the swept query and gains. Then item 4.
   - Every stage's numbers move, and production's ranking field changes meaning.
   - Measure the restore's cost on the swept query and the all-fixture pool score's time and memory. Then re-run the control and the re-roll floor end to end, since X80–X87 are not comparable across this change.
3. **Fixture geometry, separately.** The STEP distance fields (sign by fwn; α-wrap only for open meshes) and the well model are their own arm, not part of this refactor.
   - The phases already share the well model through `RUTTER_WELL`.
   - The thick fill can only lower the field, so it rejects plans rather than passing collisions.
   - X66 and X68 found that an exact wall distance regressed Phase 1 (FCL-clear 197 to 155), because its gradient points out through the outer half of the wall. The solid mode is in `scratch/bisect-wt/solid`.

## Not settled here

- Which Phase-1 field ranks Phase 2's input. R1 (X87) is a build, and `min_clear` changes meaning in step 2.
- The empty `body_shank_obb` category. Removing it changes Phase 2's row count, so it is a parity question of its own.
- `comfortable_ap_deg` and `comfortable_ml_deg` are soft bounds, not feasibility, but are declared three times as well.
