"""Command-line entry point for the trame + PyVista probe planner.

Installed as the ``alp-plan`` console script. Loads a full ``ConfigModel``
YAML, builds the trame app, and starts the server. Pass ``--plan`` to also
auto-load a plan-only ``PlanningModel`` YAML at startup.

Examples
--------
Open a config::

    alp-plan examples/836656-config-T12.yml

Open a config with a handoff plan applied at startup::

    alp-plan examples/836656-config-T12.yml \\
        --plan scratch/handoff/plans/plan-01-cov17.44-....yml
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from aind_low_point.app import build_trame_app
from aind_low_point.config import ConfigModel


@dataclass
class Args:
    """Command-line arguments for the ``alp-plan`` launcher."""

    config: tyro.conf.Positional[Path]
    """Path to the full ``ConfigModel`` YAML config file."""

    ccf_volume: Path | None = None
    """Path to a warped CCF segmentation volume (.nrrd)."""

    save: Path | None = None
    """Path for the 'Save' button — full updated config YAML
    (default: ``<config>_out.yml``)."""

    export_plan: Path | None = None
    """Path for the 'Export plan' button — slim per-probe geometric
    summary, NOT a full config (default: ``<config>_plan.yml``)."""

    plan: Path | None = None
    """Path for the 'Save plan' / 'Load plan' buttons — plan-only YAML
    (just the PlanningModel block, no asset lists). Default:
    ``<config>.plan.yml``. Used for both save and load, so you can
    round-trip into the same file. When this file exists, it is also
    applied to the planning state at startup."""


def main() -> None:
    """Parse arguments, build the trame app, and start the server."""
    args = tyro.cli(Args)

    config_path = args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    save_path = args.save
    if save_path is None:
        save_path = config_path.with_stem(config_path.stem + "_out")

    export_plan_path = args.export_plan
    if export_plan_path is None:
        export_plan_path = config_path.with_stem(config_path.stem + "_plan")

    # When the user passes --plan explicitly, honour it; otherwise default
    # to <config>.plan.yml. Either way only auto-apply at startup if the
    # file actually exists, so the (possibly absent) default never errors.
    plan_path = args.plan
    if plan_path is None:
        plan_path = config_path.with_suffix(".plan.yml")
    apply_plan_on_start = plan_path.exists()

    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    cfg = ConfigModel.from_yaml(config_path)
    server = build_trame_app(
        cfg,
        ccf_volume=args.ccf_volume,
        save_path=save_path,
        export_plan_path=export_plan_path,
        plan_path=plan_path,
        source_config_path=config_path,
        apply_plan_on_start=apply_plan_on_start,
    )

    print(f"Loaded config: {config_path}")
    # The plan-apply outcome is reported by build_trame_app's on_load_plan
    # (it prints "Loaded plan from …" on success, or a clear skip/validation
    # message otherwise) — don't claim success here before it has run.
    print("Starting trame server...")
    server.start()


if __name__ == "__main__":
    main()
