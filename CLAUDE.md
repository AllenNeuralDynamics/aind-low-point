# CLAUDE.md — Rutter (`aind-rutter`)

Multi-probe insertion planning for AIND: an interactive planner and a
constrained-optimization solver over one runtime. The frontend is Trame +
PyVista (`web/app.py` + `web/controller.py`, with the page, the readouts, the
materials and the camera as mixins beside it).

Setup, the checks and the commit and PR conventions are in `CONTRIBUTING.md`;
the architecture, the coordinate rule, the config model and the optimizer are
in `docs/source/`. This file holds only what those would not stop you getting
wrong.

## Invariants

- **Internal canonical space is LPS millimeters.** RAS appears only at named
  user-facing boundaries — the config's `point_RAS` / `offsets_RA`, the
  sliders, the rig export — and converts at the planning boundary. `ProbePlan`
  holds `offsets_LP` and `target_point_LPS`. See `docs/source/coordinates.md`.
- **Develop on 3.13; the floor is 3.11.** Use `uv run --python 3.13 ...` for
  everything, and sync with `--all-extras` or 18 test modules lose JAX and fail
  to collect. CI tests the floor and 3.13. 3.11 is what `enum.StrEnum` needs;
  3.14 waits on cp314 wheels for `scikit-image` and `mesh2sdf`, whose source
  builds fail (`python-fcl` has them now).
- **Models are the source of truth.** When tests disagree with `config/`,
  fix the tests.
- **Classifications are enums, groupings are tags.** A config states
  `mr_signal`, `role`, `kind` and `collidable` as closed fields, so a
  misspelling is refused at load; `tags`/`scene_tags` are open and only draw a
  warning on a near miss. `mr_signal` decides chemical shift and is required on
  every asset and target whenever `imaging` is set. See
  `docs/source/vocabulary.md`. The fields these replaced — `caps`, `collision`,
  `chem_shift_policy`, `chem_shift_apply_by_role`, `options` — are refused by
  name; `e7e345c` is the last commit whose `scripts/upgrade_config.py` can
  migrate a config that has them.
- **Every pipeline stage is told, not configured by import order.**
  `phase1.run(Phase1Settings())`, `phase2.run(recs, Phase2Settings())`
  and `emit.run(EmitSettings())` take typed settings; constructor arguments
  outrank the environment. Every variable the pipeline reads is `RUTTER_` plus
  the field's name, upper-cased; the bare spellings are gone, because `CONFIG`,
  `OUT`, `LIMIT`, `N`, `WORKERS`, `POOL` and `PLATFORM` are set for unrelated
  reasons in ordinary shells. The only import-time environment reads left are
  the jax platform and allocator variables, which must precede the jax import.
- **What a config resolves to is pinned.** `tests/config_semantics.json` records
  every tracked config's per-asset chemical-shift decision and ppm and
  collidability, plus its scene nodes, fixture set and collision pairs.
  Regenerate with `uv run --python 3.13 python -m tests.config_semantics` only
  when a behaviour change is the point.

## Traps

- **Optimizer docstrings drift.** Several name a stage or a module that was
  renamed around them. `docs/source/optimization.rst` is read-verified against
  the code; trust it over a docstring.
- **Run the parity harness for any change to a solver path.** It compares
  Phase-2 output against a baseline commit on a subject written on the spot and
  exits non-zero on any difference. Add `--config`/`--holes`/`--poses` for a
  real subject, `--platform gpu` to run on the card.

  ```bash
  uv run --python 3.13 python scripts/parity_phase2.py --baseline HEAD~1
  ```

- **`tests/architecture/` fails in both directions.** Each rule carries a
  baseline of today's exceptions that may only shrink, so a rule also fails
  when a listed exception stops violating it.
- **One-off tests go in `/tmp`, prototypes in `scratch/`.** `scripts/` is for
  tracked production drivers.

## Where things live

`docs/source/architecture.rst` carries the module tree and the five layer
rules. Two things it is worth knowing before you open it:

- `domain/` imports no config, no optimizer and no UI. That is what lets the
  viewer run without the optimizer stack, and it is enforced.
- `optimization/` is the solver plus `pipeline/`, the offline batch flow that
  drives it. The solver subpackages never import `pipeline`.

`dev/` holds dated working notes, `dev/proposals/` designs nobody has built,
and `dev/archive/` superseded reasoning. `dev/TODO.md` is the defect list.
