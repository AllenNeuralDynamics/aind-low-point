# Contributing

## Setup

Development happens on Python 3.13; 3.11 is the supported floor, and CI tests
both. Everything runs through [uv](https://docs.astral.sh/uv/) — nothing is
installed into a global environment.

```bash
uv sync --python 3.13
```

The `dev` group pulls in the `optimization` extra, so a plain sync gives a
working test run. It stays an extra for *installs*, because someone who only
wants the planner should not have to take a gigabyte of solver with it.

The `ipopt` extra is deliberately not in that group. cyipopt publishes no
wheels, so every install builds from source against a native IPOPT, which a CI
runner does not have. Add it if you have IPOPT — `sudo apt install
coinor-libipopt-dev`, then `uv sync --extra ipopt` — and the Phase-2 tests will
then cover the production solver as well as trust-constr. Without it they cover
trust-constr alone.

Develop the solver on Linux. JAX ships its CUDA plugin for Linux only, so
macOS is CPU and Windows is WSL2; the README's **Platforms** table has the
detail. Working on the planner alone needs none of this.

## Checks

```bash
uv run --python 3.13 pytest -q        # the suite
ruff check                            # lint
ruff format                           # format, 88 columns
mypy                                  # types, on the eight modules it lists
codespell                             # spelling
interrogate                           # docstring coverage, floor 30%
uv run --python 3.13 sphinx-build -q -E -b html docs/source docs/build/html
```

CI runs the first five of these on 3.11 and 3.13, plus a wheel build that
walk-imports every submodule. It installs with a plain `uv sync --locked`, so
anything the suite needs has to be in a default dependency group rather than
behind an extra.

`ruff`, `mypy`, `codespell` and `interrogate` are expected on the PATH rather
than in the project environment, so they take no `uv run`.

Build the docs before merging anything that touches `docs/source`. Sphinx
reports a malformed table or a short heading underline as a warning and exits
zero, so a broken page does not fail any other check.

Four checks fail in ways worth recognising:

- **`tests/architecture/`** holds the structural rules — import cycles,
  dependency direction, private-name imports, environment reads, and which
  modules import without JAX. Each carries a baseline of today's exceptions that
  may only shrink. Fixing a violation means deleting its line; a rule also fails
  when a listed exception stops violating it, so the lists cannot rot.
- **`tests/config_semantics.json`** pins what every tracked config resolves to:
  per-asset chemical-shift decision and ppm, collidability, scene nodes, fixture
  set and collision pairs. Regenerate it with
  `uv run --python 3.13 python -m tests.config_semantics` only when changing that
  behaviour is the point of the commit.
- **`scripts/parity_phase2.py`** compares Phase-2 output against a baseline
  commit on a subject written on the spot, and exits non-zero on any difference.
  Run it for any change to a solver path:
  `uv run --python 3.13 python scripts/parity_phase2.py --baseline HEAD~1`.
- **Models are the source of truth.** When a test disagrees with
  `src/aind_rutter/config/`, the test is what changes.

## Style

- 88-column lines, `ruff format`.
- NumPy-style docstrings.
- Pydantic v2, `extra="forbid"` on most models.
- `@dataclass(frozen=True, slots=True)` for immutable runtime data.
- Comments say what the code cannot: why this way, what breaks otherwise. No
  history, no ticket or run identifiers, no measurements from the analysis that
  prompted the change — that belongs in the commit message, where it stays
  attached to the diff.

## Commits

Conventional commits, [Angular
flavour](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit):

```text
<type>(<scope>): <short summary>
```

`type` is one of `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`,
`test` or `chore`; `scope` names the affected package and is optional. Write the
summary in the imperative, and use the body for what changed and why.

Commitizen bumps the version from these types when CI passes on `main`:
`fix` gives a patch, `feat` a minor. `major_version_zero` is set, so a
`BREAKING CHANGE:` footer bumps the minor as well while the version stays below
1.0 — mark it anyway, because it is what the changelog reads.

## Pull requests

Internal contributors branch; external contributors fork. Either way the branch
has to be green on `ruff check`, the suite and the architecture tests before
review.

## Documentation

Four places, split by who reads them:

| Where | Holds |
|---|---|
| `docs/source/` | the user guide and the stable developer reference, built by Sphinx |
| `dev/` | dated working notes, design records and known defects |
| `CLAUDE.md` | what a coding agent would otherwise get wrong |
| `CONTRIBUTING.md` | this: setup, checks, style and conventions |

`docs/source/configuration.rst` is what someone writing a config for a new
subject reads, and `tests/test_docs_configuration.py` loads its worked example,
so it cannot drift from the models. Anything dated, superseded or specific to
one run belongs in `dev/`, not in the built docs.
