"""Products over xyz, written as multiply-adds.

Traced code never uses a matmul (``@``, ``jnp.dot``, ``einsum``…) for these:
on an Ampere or newer GPU, XLA runs a float32 matmul on the tensor cores at
JAX's default precision, TF32, whose 10-bit mantissa moves a point 10 mm from
its pivot by several micrometres and corrupts the derivatives through it. A
3-term sum gains nothing from the tensor cores; multiply-adds run in float32 on
every platform and fuse with the kernel around them.
``tests/architecture/test_traced_products.py`` keeps matmuls out.
"""

from __future__ import annotations

from jax import Array


def dot3(a: Array, b: Array) -> Array:
    """``sum(a * b, axis=-1)`` over a last axis of length 3; other axes broadcast.

    ``dot3(R, v)`` is ``R @ v``.
    """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def matmul3(a: Array, b: Array) -> Array:
    """``a @ b`` for ``a (..., m, 3)`` and ``b (..., 3, n)``; batch axes broadcast."""
    return (
        a[..., :, 0:1] * b[..., 0:1, :]
        + a[..., :, 1:2] * b[..., 1:2, :]
        + a[..., :, 2:3] * b[..., 2:3, :]
    )
