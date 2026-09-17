"""Import graph and import-time side effects of ``aind_rutter``, read from the AST.

Parsing rather than importing keeps the rules cheap and side-effect free: several
pipeline modules set JAX environment variables and allocate a GPU context at
import, so a rule that imported them would both be slow and change the process it
runs in.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = "aind_rutter"


@dataclass(frozen=True)
class Import:
    """One ``import`` statement reaching another module of the package."""

    importer: str
    target: str
    names: tuple[str, ...]
    at_import_time: bool
    line: int


@dataclass(frozen=True)
class EnvRead:
    """One read or write of ``os.environ`` in the package."""

    module: str
    name: str
    op: str
    at_import_time: bool
    line: int


@dataclass
class ModuleFacts:
    """Per-module AST facts."""

    name: str
    path: Path
    imports: list[Import] = field(default_factory=list)
    env: list[EnvRead] = field(default_factory=list)
    jax_at_import: list[int] = field(default_factory=list)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class _Walker(ast.NodeVisitor):
    """Collect imports, environment access and import-time jax loads."""

    def __init__(self, module: str, path: Path, modules: set[str]) -> None:
        self.facts = ModuleFacts(module, path)
        self.module = module
        self.is_package = path.name == "__init__.py"
        self.modules = modules
        self.depth = 0  # >0 inside a function body, i.e. not run at import

    def _scoped(self, node: ast.AST) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_FunctionDef = _scoped
    visit_AsyncFunctionDef = _scoped
    visit_Lambda = _scoped

    def _resolve(self, node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            return node.module
        base = self.module.split(".")
        keep = len(base) - node.level + (1 if self.is_package else 0)
        prefix = ".".join(base[:keep])
        return f"{prefix}.{node.module}" if node.module else prefix

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root == "jax" and self.depth == 0:
                self.facts.jax_at_import.append(node.lineno)
            if root == PACKAGE:
                self._record(alias.name, (), node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target = self._resolve(node)
        if not target:
            self.generic_visit(node)
            return
        if target.split(".")[0] == "jax" and self.depth == 0:
            self.facts.jax_at_import.append(node.lineno)
        if target.split(".")[0] == PACKAGE:
            names = []
            for alias in node.names:
                # `from pkg import submodule` imports a module, not a name.
                if f"{target}.{alias.name}" in self.modules:
                    self._record(f"{target}.{alias.name}", (), node.lineno)
                else:
                    names.append(alias.name)
            if names:
                self._record(target, tuple(names), node.lineno)
        self.generic_visit(node)

    def _record(self, target: str, names: tuple[str, ...], line: int) -> None:
        self.facts.imports.append(
            Import(self.module, target, names, self.depth == 0, line)
        )

    @staticmethod
    def _env_name(arg: ast.expr) -> str:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        return "<dynamic>"

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and node.args:
            value = func.value
            is_environ = isinstance(value, ast.Attribute) and value.attr == "environ"
            if is_environ and func.attr in ("get", "setdefault", "pop"):
                self._record_env(node.args[0], func.attr, node.lineno)
            elif func.attr == "getenv":
                self._record_env(node.args[0], "getenv", node.lineno)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr == "environ":
            op = "set" if isinstance(node.ctx, ast.Store) else "get"
            self._record_env(node.slice, op, node.lineno)
        self.generic_visit(node)

    def _record_env(self, arg: ast.expr, op: str, line: int) -> None:
        self.facts.env.append(
            EnvRead(self.module, self._env_name(arg), op, self.depth == 0, line)
        )


@dataclass(frozen=True)
class Graph:
    """The package's modules and the edges between them."""

    modules: dict[str, ModuleFacts]

    @property
    def imports(self) -> Iterator[Import]:
        for facts in self.modules.values():
            yield from facts.imports

    @property
    def env_reads(self) -> Iterator[EnvRead]:
        for facts in self.modules.values():
            yield from facts.env

    def owner(self, target: str) -> str | None:
        """The longest known module that an import target resolves to."""
        parts = target.split(".")
        for k in range(len(parts), 0, -1):
            candidate = ".".join(parts[:k])
            if candidate in self.modules:
                return candidate
        return None

    def edges(self, *, import_time_only: bool = False) -> dict[str, set[str]]:
        """Module-to-module edges, each pointing at the imported module."""
        out: dict[str, set[str]] = {m: set() for m in self.modules}
        for imp in self.imports:
            if import_time_only and not imp.at_import_time:
                continue
            target = self.owner(imp.target)
            if target is not None and target != imp.importer:
                out[imp.importer].add(target)
        return out

    def cycles(self, *, import_time_only: bool = False) -> list[tuple[str, ...]]:
        """Strongly connected components of more than one module, sorted."""
        return _sccs(self.edges(import_time_only=import_time_only))


def _sccs(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    """Tarjan's algorithm, iterative so deep graphs cannot blow the stack."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[tuple[str, ...]] = []
    counter = 0

    for root, children in graph.items():
        if root in index:
            continue
        work: list[tuple[str, Iterator[str]]] = [(root, iter(sorted(children)))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph.get(child, ())))))
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    component = []
                    while True:
                        popped = stack.pop()
                        on_stack.discard(popped)
                        component.append(popped)
                        if popped == node:
                            break
                    if len(component) > 1:
                        out.append(tuple(sorted(component)))
    return sorted(out)


@cache
def module_graph() -> Graph:
    """Parse every module of the package once per session."""
    paths = sorted(
        p for p in (SRC / PACKAGE).rglob("*.py") if "__pycache__" not in p.parts
    )
    names = {_module_name(p): p for p in paths}
    modules: dict[str, ModuleFacts] = {}
    for name, path in names.items():
        walker = _Walker(name, path, set(names))
        walker.visit(ast.parse(path.read_text(), filename=str(path)))
        modules[name] = walker.facts
    return Graph(modules)


@cache
def console_scripts() -> dict[str, str]:
    """The ``name -> module:attr`` entry points declared in pyproject."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return dict(pyproject["project"].get("scripts", {}))


def import_in_subprocess(module: str, *, env: dict[str, str] | None = None) -> str:
    """Import ``module`` in a clean interpreter and report the loaded packages.

    Returns the newline-separated top-level names in ``sys.modules`` afterwards,
    so a caller can assert on what the import pulled in.
    """
    import os
    import subprocess

    code = (
        "import sys, importlib;"
        f"importlib.import_module({module!r});"
        "print('\\n'.join(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    child_env = {**os.environ, **(env or {})}
    # The parent's conftest pins JAX to CPU; unset it so a rule about which
    # modules load jax measures the module, not the test harness.
    child_env.pop("JAX_PLATFORMS", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        cwd=ROOT,
        timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(f"importing {module} failed:\n{result.stderr}")
    return result.stdout
