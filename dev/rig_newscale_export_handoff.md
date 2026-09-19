# Handoff: rig-facing NewScale export (tab + session file + CLI)

**Status:** designed, not implemented. This doc is detailed enough for another
agent to implement without re-deriving the design. Author: design session
2026-06-22.

## 1. Problem & goal

Rig operators currently turn an Rutter plan into NewScale (manipulator) µm
coordinates by hand-editing a Jupyter notebook:
`~/Documents/Code/aind-mri-targeting/notebooks/calibration_notebook.py`.

That notebook conflates three different kinds of thing:

- **Mechanical / already-known** — calibration fit, the bregma→NewScale
  transform, reticle offsets. (Reticle offsets are *hardcoded* in the notebook,
  `reticle_offsets = {"H": np.array([0.076, 0.062, 0.311])}`, even though they
  also live in the plan config's `calibrations.reticles`. Pure DRY violation.)
- **Genuine rig-day decisions** — which calibration files are current (per-session
  physical measurements, NOT in the plan), and the **physical-manipulator-serial
  → planned-target mapping** (+ overshoot). These are legitimately interactive
  and legitimately not in the plan.
- **Escape hatch** — convert an arbitrary bregma RAS (mm) coordinate to NewScale
  µm for an ad-hoc pipette/target. Valuable; must survive.

**Goal:** replace the notebook's hokey parts with (a) a **durable rig-session
file** capturing only the session-specific decisions, (b) a **dedicated "Rig"
tab** in the existing trame app that loads/edits/saves that file with a live
NewScale preview, and (c) an **`rutter-newscale` CLI** that consumes the identical
file headlessly. All three sit over one shared conversion core.

### Persona note (why a separate tab, not folding into Pose)
Planners use the trame app's 3D/Pose flow; rig operators just need
"plan + today's calibration → NewScale µm". They are different people. A
dedicated tab keeps the planning UI uncluttered and gives operators a focused
surface. **The rig already runs the trame app**, so a tab there is the cohesive
home (plan + calibrations + coordinate machinery are already loaded). The
notebook stays untouched until the tab is accepted — no bridge burned.

## 2. Design decisions (confirmed with user)

1. **Separate "Rig" tab** in the trame app (not a panel inside Pose, not a
   standalone GUI, not CLI-primary). Rig already runs trame.
2. **One durable rig-session YAML** is the source of truth for the
   session-specific half (cal file paths, reticle id, serial→target+overshoot,
   ad-hoc rows). The tab is a **load → edit → save editor** over this file with
   live preview; it is NOT a thing with hidden state that "optionally exports".
3. **`assign` is keyed by plan target name** (the stable side, fixed by the
   plan), value carries the physical `serial` + `overshoot_um`. You decide
   "which manipulator lands at this target", not the reverse.
4. **Auto-seed:** loading a plan without a session pre-fills `assign` with every
   plan target and blank serials, so first-time operators get a skeleton instead
   of a blank file.
5. **Overshoot defaults to 0.** Rutter's emitted tip (`tip_RAS_mm`) already bakes
   in the planned depth-past-target (`distance_past_target` is exported as 0; the
   tip *is* the landing point — see `runtime/plan_csv.py` and the
   `rutter-plan-csv` work). So the rig `overshoot_um` is *purely additional* manual
   overshoot, applied on the probe Z axis. Do NOT re-add planned depth here.
6. **CLI is a free byproduct**, not the operator's primary surface. Same core,
   same session file → reproducible.
7. **Tab and CLI consume the identical file** → fully interchangeable; the
   session file + plan + cal files deterministically reproduce the NewScale CSV
   (audit trail; archive with the surgery record; carry yesterday's session as a
   template and just swap serials).

## 3. What already exists (reuse — do NOT reinvent)

### 3a. The conversion core — `src/aind_rutter/calibration_conversion.py`
Already done, fully tested elsewhere. Two functions, accept `(3,)` or `(N,3)`:

```python
newscale_to_lps(xyz_newscale, cal: AffineTransform) -> ndarray   # NewScale → subject LPS
lps_to_newscale(xyz_lps,      cal: AffineTransform) -> ndarray   # subject LPS → NewScale
```

Internally: NewScale↔bregma-RAS is the per-probe calibration `(R,t)` via
`aind_mri_utils.reticle_calibrations.transform_{bregma_to_probe,probe_to_bregma}`;
bregma-RAS↔subject-LPS is a sign flip on x and y (`_bregma_RAS_to_subject_LPS`,
bregma is at LPS origin). **Frame trap:** these take **subject LPS**, but the
plan export gives **bregma RAS** (`tip_RAS_mm`) and the escape hatch takes
bregma RAS. Flip RAS→LPS first (negate x,y) — or just call
`_bregma_RAS_to_subject_LPS` (it's symmetric) — before `lps_to_newscale`.
Equivalently you could call `transform_bregma_to_probe(R,t)` directly on the
bregma-RAS point; pick one path and keep it consistent. Prefer routing through
the existing module so there's a single conversion implementation.

### 3b. Calibration loading — `src/aind_rutter/build/calibration.py`
- `_load_calibration_bank(cal_file: CalibrationSourceModel, reticles) -> {code: (R,t)}`
  — wraps `fit_rotation_params_from_{manual_calibration,parallax}`.
- `_merge_stacked_sources(sources, reticles) -> {code: (R,t)}` — loads an ordered
  list of `CalibrationSourceModel` and merges with **last-source-wins**.
- `_get_calibration_rt(CalibrationsModel, reticles) -> {probe_name: AffineTransform}`
  — the full config path.

The session-bank loader (step 2 below) should reuse `_merge_stacked_sources` to
go from session cal paths → `{serial: AffineTransform}`. Build
`CalibrationSourceModel`s from the session's `manual`/`parallax` lists (a manual
`.xlsx` sets `file=`; a parallax dir sets `directory=` + `reticle=`). NOTE the
model invariant (`config.py:1431` `_xor_and_require_reticle`): a source is
EITHER `file` (no reticle) OR `directory` (reticle required).

### 3c. Calibration config models — `src/aind_rutter/config.py`
- `CalibrationReticleModel` (`:1402`): `name`, `offset_RAS: list[float]` (len 3),
  `rotation_z: float`. **Reticle offsets come from here**, never hardcode.
- `CalibrationSourceModel` (`:1411`): `file` XOR `directory`(+`reticle`).
- `CalibrationsModel` (`:1451`): stacked mode (`sources` + `probe_to_code`) or
  legacy mode (`files` + `probe_to_ref`). The config's `calibrations.reticles`
  dict is where reticle offsets live for the session loader.

### 3d. The plan geometry payload — `export_plan_geometry`
`src/aind_rutter/plan_io/rig_export.py`.
Per-probe dict already carries everything the rig needs:
`kind`, `target.position_RAS_mm` (anatomical centroid, no offset),
`arc`, `angles_rig_deg`, `angles_subject_deg`, `offsets_RA_mm`,
`past_target_mm`, **`tip_RAS_mm`** (world RAS of the position-bearing shank tip,
offsets+depth+angles applied), `position_bearing_shank`,
`depth_from_brain_surface_mm`. Top-level: `head_pitch_about_L_deg`,
`arc_angles_subject_deg`, `arc_angles_rig_deg`. **Use `tip_RAS_mm` as the target
point** (this is the landing point; consistent with the `rutter-plan-csv` CSV's
`target_pt_*`).

### 3e. The trame tab scaffold — `src/aind_rutter/web/controller.py`
- Tab bar at `:1432`: `VTabs(v_model=("ctrl_tab",))` with
  `VTab(text="Pose"/"Display"/"Files", value=...)`, and a
  `VTabsWindow(v_model=("ctrl_tab",))` with one `VTabsWindowItem(value=...)` per
  tab (`:1440-1453`). **Add a `VTab(text="Rig", value="rig")` and a matching
  `VTabsWindowItem(value="rig")` that calls `self._build_rig_tab()`.**
- `state.ctrl_tab` default is set at `:286`.
- Existing per-probe NewScale UI to mirror for idiom (state keys, `@state.change`
  handlers, `VCard`/`VCardText`/`VSelect`/`VTextField` density="compact"):
  `_build_newscale_section` (`:1518`), `_compute_newscale_readout` (`:976`),
  `_apply_newscale_to_probe` (`:895`). Note these are tied to the *plan's*
  calibrations (`store.state.calibrations`) and the planning probe identity —
  the rig tab deliberately uses a **separate session bank**, decoupled from the
  plan's calibration identity.
- State init block starts ~`:259` inside the `build_app` `_init`/on-ready path;
  new `rig_*` state keys go there alongside the existing
  `probe_newscale_*` keys (`:335`).

### 3f. The notebook this replaces (reference for exact rig semantics)
`~/Documents/Code/aind-mri-targeting/notebooks/calibration_notebook.py`. Key
behaviours to preserve: overshoot is applied on probe Z (`overshoot/1000` mm)
*after* `transform_bregma_to_probe`; NewScale coords are rounded to half-µm
(`np.round(2000*probe_target)/2`); per-probe fit error is reported
(`errs_by_probe` from `debug_parallax_and_manual_calibrations`). The notebook's
target-CSV ingestion already accepts our `rutter-plan-csv` columns (`structure`,
`target_pt_{R,A,S}`) — renames `structure→point`, `target_pt_*→ML/AP/DV (mm)`.

## 4. Rig-session file schema

```yaml
# rig-session-837229.yml
plan: plan-10-...plan.yml          # path to the *.plan.yml these assignments target
calibrations:
  manual:   [calibration_info_348_..._.xlsx]   # 0+ manual files (xlsx)
  parallax: []                                 # 0+ parallax dirs (each needs a reticle)
reticle: H                          # id into config calibrations.reticles
assign:                             # keyed by PLAN TARGET name (stable side)
  BLA: {serial: 46110, overshoot_um: 0}
  PL:  {serial: 46100, overshoot_um: 200}
adhoc:                              # escape hatch: arbitrary bregma RAS mm
  pipette: {serial: 46123, bregma_mm: [1, 1, -1]}
```

`serial` is the physical manipulator id used as the calibration probe code
(stringify it for bank lookup; banks key on `str(code)` — see
`calibration.py:39`). `overshoot_um` defaults to 0.

## 5. Implementation steps

Build **1–3 + 5 first** (UI-independent, fully unit-testable headlessly),
confirm green, then wire the tab (4). The build/export/conversion path is
**jax-free** — do not import anything that drags in jax (optimizer extras aren't
installed in the rig/test venv; optimizer test modules already fail collection
for unrelated reasons).

### Step 1 — session model (`config.py`)
Add Pydantic v2 models (`extra="forbid"`, match the repo style):
- `RigAssignEntry`: `serial: str` (coerce int→str), `overshoot_um: float = 0.0`.
- `RigAdhocEntry`: `serial: str`, `bregma_mm: list[float]` (len 3).
- `RigCalibrationsModel`: `manual: list[str] = []`, `parallax: list[str] = []`.
- `RigSessionModel`: `plan: str`, `calibrations: RigCalibrationsModel`,
  `reticle: str`, `assign: dict[str, RigAssignEntry] = {}`,
  `adhoc: dict[str, RigAdhocEntry] = {}`. Add `from_yaml(path)` / `to_yaml(path)`
  (mirror `ConfigModel.from_yaml`; dump with `sort_keys=False`).

### Step 2 — shared core (new `src/aind_rutter/plan_io/newscale.py`)
```python
def load_session_bank(
    session: RigSessionModel,
    reticles: dict[str, CalibrationReticleModel],
) -> tuple[dict[str, AffineTransform], dict[str, float]]:
    """serial → (R,t) AffineTransform, plus serial → fit error (µm).

    Builds CalibrationSourceModels from session.calibrations.{manual,parallax}
    (manual→file=, parallax→directory=+reticle=session.reticle), then reuses
    calibration._merge_stacked_sources(...). Errors: thread through
    debug_parallax_and_manual_calibrations' errs_by_probe (or recompute via the
    fit residual) so the tab/CLI can surface per-probe error like the notebook.
    """


@dataclass(frozen=True, slots=True)
class NewscaleRow:
    target: str
    serial: str
    bregma_RAS_mm: tuple[float, float, float]  # the point we aimed at (tip or adhoc)
    newscale_um: tuple[float, float, float]  # rounded to half-µm
    overshoot_um: float
    fit_error_um: float | None


def session_to_newscale_rows(
    session: RigSessionModel,
    plan_payload: dict,  # export_plan_geometry(...) output
    bank: dict[str, AffineTransform],
    errors: dict[str, float] | None = None,
) -> list[NewscaleRow]:
    """For each assign entry: target tip_RAS_mm from plan_payload['probes'][target],
    RAS→subject-LPS, lps_to_newscale(.., bank[serial]), add overshoot_um on probe
    Z (overshoot is in NewScale/probe frame µm — apply after the transform, like
    the notebook: probe_target + [0,0,overshoot/1000] in mm then ×1000), round to
    half-µm. adhoc rows: same but bregma_mm instead of the plan tip. Raise a clear
    error if a serial is missing from the bank or a target is missing from the plan.
    """
```
Keep it pure (numpy only) and import-light. This is the single source of truth
both the tab and CLI call.

### Step 3 — CLI (`src/aind_rutter/plan_io/newscale_cli.py`, entry `rutter-newscale`)
`rutter-newscale rig-session.yml [--config examples/837229-config.yml] [--out X.csv]`:
1. `RigSessionModel.from_yaml(session)`.
2. `cfg = ConfigModel.from_yaml(config)`; `bundle = build_runtime_from_config(cfg)`.
3. Apply the plan: load `session.plan` YAML → `PlanningModel(**raw)` →
   `apply_plan_model_to_state(plan_model, PlanStore(bundle.plan_state))`
   (mirror `runtime/plan_csv.py:plan_to_rows`).
4. `payload = export_plan_geometry(store.state, bundle.asset_catalog, scene=bundle.scene)`.
5. `bank, errs = load_session_bank(session, cfg.plan_reticles_or_calibrations_reticles)`
   (resolve the reticles dict from the config's calibrations model).
6. `rows = session_to_newscale_rows(session, payload, bank, errs)`.
7. Write CSV (columns mirror the notebook output: `Target, Serial, ML (mm),
   AP (mm), DV (mm), X (µm), Y (µm), Z (µm), Overshoot (µm), Error (µm)`) and
   print the per-probe error report.
Register in `pyproject.toml [project.scripts]` next to `rutter-plan-csv`. Default
`--config` to `examples/837229-config.yml` for parity with `rutter-plan-csv`.

### Step 4 — Rig tab (`trame_controller.py`)
- Add `VTab(text="Rig", value="rig")` (`:1437` block) + a
  `VTabsWindowItem(value="rig")` (`:1452` block) calling `self._build_rig_tab()`.
- `_build_rig_tab()`:
  - **Session file** row: `VTextField` (`rig_session_file`) + "Load session" /
    "Save session" `VBtn`s. Load → `RigSessionModel.from_yaml` → populate state.
    Save → assemble `RigSessionModel` from state → `to_yaml`.
  - **Calibration files**: text fields/lists for manual + parallax + a reticle
    `VSelect` (items from the config's reticles). A "Load calibration" button
    runs `load_session_bank` and stores the bank + errors on the controller.
  - **Assignment table** (`VDataTable` or a `v_for` over `rig_rows`): one row per
    plan target — target name (read-only), serial `VTextField`/`VSelect`,
    `overshoot_um` `VTextField`, and live `X/Y/Z µm` + `Error µm` readouts.
    Recompute `rig_rows` via `session_to_newscale_rows` on any change.
  - **Auto-seed:** when a plan is loaded but `assign` is empty, seed one row per
    `payload['probes']` key with blank serial / overshoot 0.
  - **Escape hatch**: a small card — target label + bregma ML/AP/DV `VTextField`s
    + serial select → live NewScale readout; an "Add to session" button appends
    to `adhoc`.
  - **Export**: "Export CSV" `VBtn` (same CSV as the CLI). Wire an `on_*`
    callback through `build_trame_app` like `on_export_plan` (`app.py:148`) so
    the output path is injectable, or write next to the session file.
- New `state` keys near `:335`: `rig_session_file`, `rig_manual_files`,
  `rig_parallax_dirs`, `rig_reticle`, `rig_rows` (list of dicts for the table),
  `rig_adhoc_*` inputs, `rig_status`. Add `@state.change` handlers mirroring the
  existing newscale handlers.
- The bank lives on the controller instance (like `ccf_overlay`), not in
  serialized trame state (AffineTransforms aren't trivially JSON state).

### Step 5 — tests (`tests/`)
- `test_newscale_session.py`:
  - `RigSessionModel` YAML round-trip; int serial coerces to str.
  - `session_to_newscale_rows` on a **synthetic** plan payload (hand-built dict
    with `probes: {BLA: {tip_RAS_mm: [...], ...}}`) + a fake bank
    (`AffineTransform(rotation=np.eye(3), translation=...)`); assert the NewScale
    output equals a hand-computed `lps_to_newscale` (remember the RAS→LPS flip)
    and that `overshoot_um` adds on Z and rounds to half-µm.
  - adhoc path; missing-serial and missing-target raise clear errors.
  - auto-seed fills one row per plan target.
  - All **jax-free**. Build a `RigSessionModel` directly; no need to run the
    optimizer or load real cal files (mock the bank).
- CLI smoke test: write a tmp session pointing at a tmp plan + `examples/837229-
  config.yml`, run `main([...])`, assert the CSV has the expected header and one
  row per assign entry. (May need a real-ish plan fixture; reuse an existing
  `examples/*plan*.yml` or a small hand-written one.)

## 6. Verification
- `ruff check && ruff format`
- `uv run --python 3.13 pytest -q tests/test_newscale_session.py` then full suite.
- Build 837229 + run `rutter-newscale` on a hand-written session; eyeball that the
  µm are sane and match the notebook for the same target+calibration.
- Launch the trame app, open the Rig tab, load a plan, auto-seed, assign a
  serial, confirm live NewScale + error; save/reload the session file.

## 7. Gotchas / invariants
- **Frame:** `lps_to_newscale` wants **subject LPS**; the plan tip and escape
  hatch are **bregma RAS**. Negate x,y first (or route through
  `calibration_conversion._bregma_RAS_to_subject_LPS`). Get this wrong and
  ML/AP flip sign.
- **Overshoot is additional, not planned depth.** Rutter already baked planned
  depth into `tip_RAS_mm` (CSV `distance_past_target=0`). Default `overshoot_um=0`.
- **Reticle offsets from config**, never hardcode (the notebook's
  `{"H": [0.076,0.062,0.311]}` is the anti-pattern being removed).
- **Serial = calibration probe code**, stringified; banks key on `str(code)`.
- **Last-source-wins** when merging manual+parallax (manual after parallax to
  prioritize manual, matching the notebook's `probes_to_ignore_manual` intent).
- **jax-free** throughout (rig/test venv lacks optimizer extras).
- **rig vs subject AP sign** (memory `rig_ap_sign_convention`): not directly used
  here (NewScale conversion is via calibration `(R,t)`, not arc angles), but if
  you ever surface arc/rig angles in this tab, `rig_ap = subject_ap +
  head_pitch_about_L`.
- **Don't touch the notebook** until the tab is accepted.

## 8. Files touched (summary)
- `src/aind_rutter/config.py` — `RigSessionModel` + sub-models.
- `src/aind_rutter/plan_io/newscale.py` — NEW, shared core.
- `src/aind_rutter/plan_io/newscale_cli.py` — NEW, `rutter-newscale`.
- `src/aind_rutter/web/controller.py` — Rig tab (`_build_rig_tab`, state,
  handlers, tab-bar entries at `:1437`/`:1452`).
- `src/aind_rutter/web/app.py` — optional `on_export_newscale` wiring
  (mirror `on_export_plan`, `:148`).
- `pyproject.toml` — `rutter-newscale` entry point.
- `tests/test_newscale_session.py` — NEW.
- Reuse (no change): `calibration_conversion.py`, `runtime/calibration.py`,
  `plan_io/rig_export.py`, `plan_io/cli_csv.py` (pattern reference).
```
