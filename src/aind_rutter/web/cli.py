"""Command-line entry point for the trame + PyVista probe planner.

Installed as the ``rutter-plan`` console script. Loads a full ``ConfigModel``
YAML, builds the trame app, and starts the server. Pass ``--plan`` to also
auto-load a plan-only ``PlanningModel`` YAML at startup.

Examples
--------
Open a config::

    rutter-plan examples/836656-config-T12.yml

Open a config with a handoff plan applied at startup::

    rutter-plan examples/836656-config-T12.yml \\
        --plan scratch/handoff/plans/plan-01-cov17.44-....yml

Step through every plan emit wrote for it::

    rutter-plan examples/836656-config-T12.yml --plans scratch/handoff

Reopen a session from the rig poses it exported, over the plan it started
from::

    rutter-plan examples/836656-config-T12.yml --poses rutter.poses.yml

Serve on another port, where 8080 is taken::

    rutter-plan examples/836656-config-T12.yml --port 8090
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from aind_rutter.config import ConfigModel
from aind_rutter.config.locations import Locator, parse_overrides
from aind_rutter.web.app import build_trame_app


@dataclass
class Args:
    """Command-line arguments for the ``rutter-plan`` launcher."""

    config: tyro.conf.Positional[Path]
    """Path to the full ``ConfigModel`` YAML config file."""

    ccf_volume: Path | None = None
    """Path to a warped CCF segmentation volume (.nrrd)."""

    plan: Path | None = None
    """Plan-only YAML (the PlanningModel block, no asset lists) applied at
    startup when it exists (default: ``<config>.plan.yml``). The Files tab
    saves and loads through the browser."""

    poses: Path | None = None
    """A rig-poses export (the Files tab's "poses"), laid over the startup
    plan to restore each probe's pose. A probe read through its calibration
    keeps the plan's AP and ML, which the calibration replaced."""

    plans: Path | None = None
    """An emit output directory (or its ``plans/`` folder) to step through
    in the Plans tab. Its plans must share this config's probes."""

    paths: Annotated[tuple[Path, ...], tyro.conf.UseAppendAction] = ()
    """Paths file locating the config's ``paths:`` names, read ahead of
    ``RUTTER_PATHS`` and the files found by search; repeatable, first wins."""

    path: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """``NAME=DIR`` for one ``paths:`` name, ahead of every paths file;
    repeatable."""

    offline: bool = False
    """Fetch nothing: packs are read from the cache, and a missing file fails
    naming it. ``RUTTER_OFFLINE=1`` does the same."""

    port: int = 8080
    """Port the planner serves on; 0 takes any free one."""


def main() -> None:
    """Parse arguments, build the trame app, and start the server."""
    args = tyro.cli(Args)

    config_path = args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # When the user passes --plan explicitly, honour it; otherwise default
    # to <config>.plan.yml. Either way only auto-apply at startup if the
    # file actually exists, so the (possibly absent) default never errors.
    plan_path = args.plan
    if plan_path is None:
        plan_path = config_path.with_suffix(".plan.yml")
    if args.poses is not None and not args.poses.exists():
        print(f"Poses file not found: {args.poses}", file=sys.stderr)
        sys.exit(1)
    apply_plan_on_start = plan_path.exists() or args.poses is not None

    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    # Say where each input folder came from when a paths file set it.
    logging.getLogger("aind_rutter.config.locations").setLevel(logging.INFO)

    try:
        locator = Locator.from_environment(
            args.paths, parse_overrides(args.path), offline=args.offline
        )
        cfg = ConfigModel.from_yaml(config_path, locator)
    except LookupError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    server = build_trame_app(
        cfg,
        ccf_volume=args.ccf_volume,
        plan_path=plan_path,
        source_config_path=config_path,
        apply_plan_on_start=apply_plan_on_start,
        plans=args.plans,
        poses_path=args.poses,
    )

    print(f"Loaded config: {config_path}")
    # The plan-apply outcome is reported by build_trame_app's on_load_plan
    # (it prints "Loaded plan from …" on success, or a clear skip/validation
    # message otherwise) — don't claim success here before it has run.
    print("Starting trame server...")
    server.start(port=args.port)


if __name__ == "__main__":
    main()
