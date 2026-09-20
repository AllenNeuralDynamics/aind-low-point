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

## Commands

| Command | Does |
|---|---|
| `rutter-plan config.yml` | Open the interactive planner. `--plan` loads a plan at startup. |
| `rutter-phase1` | Enumerate and score candidate configurations. |
| `rutter-phase2` | Solve the top candidates exactly, then gate on collision and threading. |
| `rutter-emit` | Write the ranked survivors as plan-only YAML. |
| `rutter-plan-csv` | Convert a plan into a target CSV for the rig. |

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
