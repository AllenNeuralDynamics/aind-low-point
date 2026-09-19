# dev/

Working notes. The reference documentation is `docs/source/` — the concepts
tour, the architecture guide, the coordinate rule, the config model, the config
vocabulary and the optimizer guide all live there and are kept current. This
directory holds what is dated instead: measurements, findings, designs and
defects. When a note here disagrees with `docs/source/`, the built docs are
right.

Every note opens with a **Status** line giving what it is and when it was
written. Convert relative dates when you write one.

| | |
|---|---|
| `TODO.md` | known defects and deferred work, each with enough to act on. A fix deletes its entry. |
| `POOL_RUN_CONFIGS.md` | tuned Phase-1 presets, with the measurements behind them. |
| `PHASE2_CONDITIONING.md` | measured Phase-2 solver conditioning, IPOPT behaviour and open defects. |
| `spin_basin_experiments.md` | why the round-robin spin restore is still the production spin-basin finder. |
| `ARCHITECTURE_SURVEY.md` | the 2026-09 survey and the seven-step sequence that came out of it. |
| `proposals/` | designs that are not built. Read the Status line before acting on one. |
| `archive/` | superseded. Kept for the reasoning, not for the conclusions. |
