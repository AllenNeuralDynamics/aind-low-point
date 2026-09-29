"""No matmul in a module that imports jax.

On an Ampere or newer GPU a float32 matmul runs at TF32 by default, which the
solver's derivatives do not survive, and every product in traced code is over
xyz, where the tensor cores buy nothing. Traced code spells them
``clearance.vec3.dot3``/``matmul3``; numpy code in the same module spells them
``np.matmul``/``np.dot``, because ``@`` does not say which library it runs in.
CI has no GPU, so nothing else would notice one coming back.
"""

from __future__ import annotations

import ast

from tests.architecture.graph import PACKAGE, SRC

_MATMUL_NAMES = {
    "dot",
    "dot_general",
    "batch_matmul",
    "einsum",
    "inner",
    "matmul",
    "multi_dot",
    "tensordot",
    "vdot",
}
_JAX_NAMESPACES = {
    "jnp",
    "jax.numpy",
    "jnp.linalg",
    "jax.numpy.linalg",
    "lax",
    "jax.lax",
}


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _imports_jax(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            a.name.split(".")[0] == "jax" for a in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "jax"
        ):
            return True
    return False


def _matmuls(tree: ast.Module) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp | ast.AugAssign) and isinstance(
            node.op, ast.MatMult
        ):
            found.append((node.lineno, "@"))
        elif isinstance(node, ast.Attribute) and node.attr in _MATMUL_NAMES:
            base = _dotted(node.value)
            if base in _JAX_NAMESPACES:
                found.append((node.lineno, f"{base}.{node.attr}"))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("jax"):
            found += [
                (node.lineno, a.name) for a in node.names if a.name in _MATMUL_NAMES
            ]
    return found


def test_no_matmul_where_jax_is_imported() -> None:
    violations = []
    for path in sorted((SRC / PACKAGE).rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        if not _imports_jax(tree):
            continue
        rel = path.relative_to(SRC)
        violations += [f"{rel}:{line} {what}" for line, what in _matmuls(tree)]
    assert not violations, "matmul in a jax module:\n" + "\n".join(violations)
