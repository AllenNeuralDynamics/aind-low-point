# Placement-optimizer pipeline

**Status:** current operational summary. The detailed developer guide is
`docs/source/optimization.rst`.

The live pipeline is now:

```text
CONFIG + HOLES
  -> alp-phase1  (MRV enumerate, spin restore, batched RProp/coarse-fine pool)
  -> alp-phase2  (IPOPT/trust-constr polish, FCL/threading gate, MMR ranking)
  -> alp-emit    (plan-only YAMLs, tree.txt, manifest.md)
```

For the operational runbook, strategy, algorithm notes, environment knobs, and
module map, read `docs/source/optimization.rst`.

For the tuned Phase-1 presets, read `dev/POOL_RUN_CONFIGS.md`.

Additional optimizer notes:

- `dev/spin_basin_experiments.md` - why the round-robin spin restore remains
  the production spin-basin finder.
- `dev/target_valid_atlas_design.md` - target-valid atlas design notes.
- `dev/PIPELINE_PLAN.md` - current pipeline hardening notes.

Vocabulary for the current code:

- Use **Phase 1 pool** for `alp-phase1`.
- Use **Phase 2 handoff** for `alp-phase2`.
- Use **emit** for `alp-emit`.
- Avoid bare "Stage 2" and "Stage 3" labels; use the concrete phase names above.
