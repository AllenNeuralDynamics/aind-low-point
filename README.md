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

Development happens on Python 3.13; 3.11 is the supported floor. 3.14 waits on
`scikit-image` and `mesh2sdf` wheels.

```bash
uv sync --python 3.13
```

That includes the solver. To install the planner alone — without JAX, IPOPT and
the mesh tooling — take the package without its `optimization` extra.

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
