# CLAUDE.md — aind-low-point

Probe placement planning ("Pinpoint in Python") for AIND. Two frontends share a
common runtime: K3D + ipywidgets (Jupyter, `controllers.py`) and Trame +
PyVista (web app, `app.py` + `trame_controller.py`).

## Invariants

- **Internal canonical space is LPS millimeters.** RAS appears only at named
  user-facing boundaries (`point_RAS`, `offset_RAS`, `offsets_RA`) and is
  converted at the planning boundary. See `dev/COORDINATES.md`.
- **Python 3.13 required.** `python-fcl` has no 3.14 wheel. Use
  `uv run --python 3.13 ...` for everything.
- **Models are the source of truth.** When tests disagree with `config.py`,
  fix the tests.

## Commands

```bash
ruff check                                # lint
ruff format                               # format
uv run --python 3.13 pytest -q            # tests (406 currently green)
uv sync --python 3.13                     # set up venv
```

## Where things live

```
src/aind_low_point/
├── core.py                # AffineTransform, TransformChain, Material, *Transformable
├── common.py              # Capability (IntFlag), Role, Kind enums
├── orientation_codes.py   # OrientationCode (48 RAS-style codes)
├── assets.py              # AssetSpec, TargetSpec, AssetCatalog
├── scene.py               # NodeInstance, Scene
├── planning.py            # ProbePlan, Kinematics, PlanningState, ProbePose, PoseResolver
├── commands.py            # Planning commands + apply_planning_command
├── config.py              # All Pydantic models, validation, template expansion
├── build_runtime.py       # Thin re-export shim → runtime/ (see below)
├── state_change.py        # PlanStore, AsyncLatestWorker
├── rendering.py           # RendererAdapter, RenderBackend protocol, overlays
├── collisions.py          # CollisionAdapter, CollisionHandler (sync + async paths)
├── k3d_backend.py         # K3DBackend (Jupyter)
├── pyvista_backend.py     # PyVistaBackend + DebouncedFlush (trame)
├── fcl_backend.py         # FCLBackend (per-pair callback, group/mask filter)
├── controllers.py         # ProbeWidgetController (K3D + ipywidgets)
├── trame_controller.py    # TrameController (Vuetify3 + PyVista)
├── app.py                 # build_trame_app() factory
├── ccf_ontology.py        # Allen CCF structures + search
└── ccf_overlay.py         # CCFOverlayManager (lazy region meshes in PyVista)
```

The top-level tree above is partial. Two big subpackages are not shown:

**`runtime/`** — config→runtime build and the planning/rig boundary:
`build` (loaders, `build_runtime_from_config`, `save_plan_to_config`),
`loaders`, `reducers`, `export` (rig-facing pose emit), `canonicalize`,
`chem_shift`, `calibration`, `scene_geometry` (`head_pitch_deg_*`),
`probe_context`, `shanks`, `transforms`. (`build_runtime.py` at the top level is
just a re-export shim.)

**`optimization/`** — the placement-optimizer package, reorganized
flat→subpackages (the old flat `optimization/*.py` module names are gone):
- `enumeration/` — `visibility_atlas`, `atlas`, `arc_placement`,
  `seed_emission` (`emit_seed`), `contracts` (`ArcAssignment`/`HoleAssignment`)
- `geometry/` — `primitives` (`cap_basis`, `HoleSection`), `kinematics`
  (`pose_from_optimizer_vars`), `probe_kinematics`, `holes`, `recording`,
  `headstages`, `probes`
- `objectives/` — `reduced_jax` (`threading_g_matrix`), `phase1`, `phase2`,
  `fcl_validator`, `coverage`, `density`, `batched_reduced`, `batched_static`,
  `spin_restore`, `probe_static` (`JointWeights`), `variables`,
  `clearance_metrics`
- `sdf/` — `kernels` (`arc_angles_to_rotation`, `trilinear_sdf`), `build`,
  `envelope`, `clearance_sweep`
- `pipeline/` — the offline batch flow: Phase-1 `phase1_pool`, Phase-2
  `phase2_ipopt`, `emit`, plus `enumeration`/`phase1_build`/`phase1_geometry`/
  `restore`/`thick_well`/`probe_setup`/`runtime_adapter`/`contracts`

Console entry points `alp-phase1` / `alp-phase2` / `alp-emit` and the
`scripts/run_subject_overnight.sh` driver run the pipeline. **See
`dev/PIPELINE.md` for the read-verified stage map** (it's full of stale-docstring
traps — trust that doc, not the docstrings).

## Deep-dive docs

- `dev/CORE_CONCEPTS.md` — **start here.** Conceptual tour: catalog vs
  scene vs planning vs adapters, end-to-end slider-drag flow, common
  "why is it like this?" questions.
- `dev/MODULE_MAP.md` — per-module reference, layered architecture, data flow.
- `dev/COORDINATES.md` — LPS canonical rule, where conversions happen, frame
  composition, working in non-AIND template spaces.
- `dev/CONFIG_MODEL.md` — Pydantic model taxonomy, validation pipeline,
  template merge rules, tags vs scene_tags, plan-only YAML, gotchas.
- `dev/PIPELINE.md` — **the placement-optimizer pipeline**, read-verified
  stage-by-stage (atlas → enumerate → spin restore → L-BFGS → ADAM rerank →
  Phase 2 → FCL → handoff), the legacy code, and the L-BFGS-vs-ADAM caveat.
  `dev/optimizer_plan.md` and `dev/spin_search_heuristics.md` are older design
  notes (some superseded — defer to PIPELINE.md on what's live).

End-user docs are in `docs/source/` (Sphinx). Don't bloat them with internals.

## Code style

- 88-char line length, ruff-formatted.
- NumPy-style docstrings.
- Pydantic v2 with `extra="forbid"` on most models.
- `@dataclass(frozen=True, slots=True)` for immutable runtime data.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
