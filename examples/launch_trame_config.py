"""Launch the trame + PyVista probe planner from a YAML config file.

Superseded by the ``alp-plan`` console script
(``aind_low_point.cli:main``). This thin shim is kept so existing
``python examples/launch_trame_config.py ...`` invocations keep working.

Usage:
    uv run --python 3.13 python examples/launch_trame_config.py \\
        examples/836656-config-T12.yml

Prefer the installed entry point:
    uv run --python 3.13 alp-plan examples/836656-config-T12.yml
"""

from __future__ import annotations

from aind_low_point.cli import main

if __name__ == "__main__":
    main()
