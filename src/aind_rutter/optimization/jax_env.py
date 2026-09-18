"""The one place that configures JAX's persistent compile cache.

Compiling the Phase-1 and Phase-2 graphs costs tens of seconds, and a spawned
worker cannot see the parent's in-memory cache, so every stage wants the on-disk
one: the first process to compile a signature writes it and the rest load it.

Configuring it is a process-global side effect, which is why it happens here and
on request. Two modules used to do it at import with different directories, so
which one a run got depended on import order — and the import that ran last won,
silently sending Phase 2 somewhere other than where it had asked for.

Importing this module loads neither jax nor a GPU backend; only
:func:`configure_compile_cache` does.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the cache has always effectively gone. Outside the working tree, so a
# clone is not carrying compiled kernels around, and cleared by a reboot.
DEFAULT_CACHE_DIR = Path("/tmp/aind_rutter_jax_cache")

# The names read below, in precedence order. Kept as a constant for the tests
# and the docs; the reads themselves spell the names out, so a survey of what
# reads the environment can see them.
CACHE_DIR_ENV = ("AIND_JAX_CACHE_DIR", "JAX_CACHE_DIR")


def compile_cache_dir(path: str | Path | None = None) -> Path:
    """Where the compile cache belongs, resolved without touching jax."""
    if path is not None:
        return Path(path)
    value = os.environ.get("AIND_JAX_CACHE_DIR") or os.environ.get("JAX_CACHE_DIR")
    return Path(value) if value else DEFAULT_CACHE_DIR


def configure_compile_cache(path: str | Path | None = None) -> Path | None:
    """Point JAX's persistent compile cache at ``path``; return where it went.

    Returns ``None`` when the cache could not be enabled. A missing cache costs
    compile time and nothing else, so a failure here is logged rather than
    raised.
    """
    import jax

    target = compile_cache_dir(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(target))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except Exception:
        logger.warning("JAX compile cache disabled: %s unusable", target, exc_info=True)
        return None
    return target
