# Rutter

![CI](https://github.com/AllenNeuralDynamics/aind-rutter/actions/workflows/ci.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/aind-rutter)](https://pypi.org/project/aind-rutter/)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Multi-probe insertion planning for *in vivo* neurophysiology: an interactive
planner and a constrained-optimization solver over one shared runtime.

> A **rutter** is a mariner's handbook of sailing directions, landmarks and
> instrument settings, compiled before a voyage.

Given a subject's anatomy, a chronic implant, and the geometry and reach of the
rig, Rutter returns complete sets of insertions that are collision-free against
the actual hardware meshes, thread their implant bores, and stay inside the
brain — ranked by how much of each target structure they reach and by how much
they differ from one another, so what you get is a real choice rather than one
answer.

## Two ways in

**Plan by hand.** A browser app (trame + PyVista) for placing probes
directly, with live collision feedback and CCF region overlays.

**Solve.** An offline pipeline enumerates candidate assignments of probes to
arcs and bores, scores every one of them with a batched relaxation on the GPU,
then solves the most promising exactly under hard constraints and verifies the
survivors against exact mesh collision.

Both lanes read and write the same plan-only YAML, so a solved plan opens in
the app and a hand-built plan can be exported to the rig.

## Install

Python 3.13 is what development happens on; 3.11 is the supported floor. 3.14
waits on `scikit-image` and `mesh2sdf` wheels.

The planner installs anywhere, with nothing to build:

```bash
pip install aind-rutter        # or: uv add aind-rutter
```

The solver is a separate matter — see **Platforms** below.

```bash
uv sync --python 3.13          # a checkout, planner and solver both
```

## Platforms

The two halves have different requirements, because the solver wants a GPU and
the planner does not.

| | Planner (`rutter-plan`) | Solver (`rutter-phase1/2/emit`) |
|---|---|---|
| **Linux** | yes | yes — the supported configuration |
| **macOS** | yes | CPU only, for development; too slow for real runs |
| **Windows** | yes | not natively; use WSL2 |
| **Windows + WSL2** | yes | yes, identical to Linux |

Two things decide this, and neither is about Rutter. JAX publishes its CUDA
plugin for Linux only — `jax-cuda12-plugin` has no Windows or macOS wheels at
any version — and the optimizer is built around having a card: it preflights
VRAM, pools workers over MPS, and stores collision grids in bf16. On CPU it
runs and is not worth running. WSL2 is JAX's own answer for GPU on Windows, and
it gives you an ordinary Linux userspace, so everything below applies unchanged.

Two extras sit behind the solver. `optimization` carries JAX and the mesh
tooling and installs from wheels. `ipopt` carries Phase 2's production solver
and is separate because cyipopt publishes no wheels at all, on any platform: it
builds against a native IPOPT that has to be on the system first.

```bash
sudo apt install coinor-libipopt-dev   # Debian/Ubuntu, including WSL2
uv sync --python 3.13 --extra ipopt
```

Without it Phase 2 still runs, on scipy's `trust-constr`. That is the fallback
rather than the intent: on a stalled-candidate comparison, IPOPT's
limited-memory mode recovered 7 of 11 where `trust-constr` recovered 4.

A config asset may also be a STEP file, read and tessellated at load. That
needs the `step` extra (OpenCascade, from wheels), which the planner and solver
otherwise do without.

## Setting up a machine

A config names its input folders in its `paths:` block, and the shared ones —
hardware meshes, implant bores, the MRI template — as published **packs**:

```yaml
paths:
  hardware: {pack: hardware/v1}
```

With nothing else set up, a pack is fetched on first use from
`s3://aind-scratch-data/rutter/packs/` (public, no credentials) into a local
cache, only the files the config reads. A machine file is needed only to
point a pack or a name at a copy already on disk, such as a shared drive.

The machine file is `paths.yml` in rutter's config folder:

| Platform | Machine file | Pack cache |
|---|---|---|
| Linux, WSL2 | `~/.config/rutter/paths.yml` (or `$XDG_CONFIG_HOME/rutter`) | `~/.cache/rutter` |
| macOS | `~/Library/Application Support/rutter/paths.yml` | `~/Library/Caches/rutter` |
| Windows | `%LOCALAPPDATA%\rutter\paths.yml` | `%LOCALAPPDATA%\rutter\Cache` |

`RUTTER_CONFIG_DIR` replaces the config folder and `RUTTER_CACHE` the cache.
Create the folder and write the file by hand:

```yaml
# Local copies of published packs; configs name the packs, not these folders.
packs:
  hardware/v1: /mnt/Data/rutter
  mouse-t1/v1.0: /mnt/Data/MRI/templates/AIND-MouseT1-v1.0

# Any other paths: name a config declares, by name.
retro_dir: Z:/ephys/persist/data/MRI/retro   # drive letters work everywhere
scratch: ../scratch                           # relative to this file's folder
```

or let `rutter-data pull hardware/v1 --to DIR` copy a pack to `DIR` and add
the `packs:` line itself. Check a copy you made another way with
`rutter-data verify hardware/v1 DIR`.

`rutter-paths config.yml` shows where every name resolved, what set it, and
each paths file it looked for, the machine file included, so run it to confirm
the setup. A `rutter.paths.yml` beside a config (or in a folder above it), or
one named by `RUTTER_PATHS`, outranks the machine file for the configs it
covers; the full order, the cache limits and the offline mode are in
`docs/source/config_model.md`.

## Commands

| Command | Does |
|---|---|
| `rutter-plan config.yml` | Open the interactive planner. `--plan` loads a plan at startup; `--port` moves it off 8080. |
| `rutter-bores implant.step out.yml --diagram implant.diagram.yml` | Fit the threading bores (`RUTTER_HOLES`) to an implant STEP; needs the `step` extra. |
| `rutter-phase1` | Enumerate and score candidate configurations. |
| `rutter-phase2` | Solve the top candidates exactly, then gate on collision and threading. |
| `rutter-emit` | Write the ranked survivors as plan-only YAML. |
| `rutter-series propose\|configs\|solve\|schedule\|status spec.yml run/` | Plan a series of sessions from one spec: proposed allocations of probes to regions, its configs, their solves, and the plan per session and session order with the least rig rework. |
| `rutter-plan-csv` | Convert a plan into a target CSV for the rig. |
| `rutter-calibrate config.yml SOURCE... --reticle-from cal.xlsx --match-plan plan.yml --out out.yml` | Add calibration sources (parallax directories, manual workbooks) to a config, last winning: the reticle, each probe's code, and `calibrated` on the probes it maps. |
| `rutter-paths [config.yml]` | Show where each of a config's `paths:` names resolves on this machine, what set it, and every paths file read. |
| `rutter-data pull\|publish\|verify\|cache` | Fetch, publish and check packs, the immutable folders of shared data a config can name, and see or clear the cache. |

## Documentation

`docs/source/` holds both guides — the user guide and the developer
reference — and `dev/` holds dated working notes and known defects. Start at
`docs/source/concepts.md` for a tour of how the pieces fit, and read
`CONTRIBUTING.md` before changing anything.

Build the docs with:

```bash
uv sync --python 3.13 --all-extras --group docs
uv run --python 3.13 sphinx-build -b html docs/source docs/build/html
```
