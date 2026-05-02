"""Launch the trame + PyVista probe planner from a YAML config file.

Usage:
    uv run --python 3.13 python examples/launch_trame_config.py \\
        examples/786864-config.yml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aind_low_point.app import build_trame_app
from aind_low_point.config import ConfigModel


def main():
    parser = argparse.ArgumentParser(description="Trame probe planner")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--ccf-volume",
        type=Path,
        default=None,
        help="Path to warped CCF segmentation volume (.nrrd)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Path to save updated YAML config (default: <config>_out.yml)",
    )
    args = parser.parse_args()

    config_path: Path = args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    save_path = args.save
    if save_path is None:
        save_path = config_path.with_stem(config_path.stem + "_out")

    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    cfg = ConfigModel.from_yaml(config_path)
    server = build_trame_app(
        cfg, ccf_volume=args.ccf_volume, save_path=save_path
    )

    print(f"Loaded config: {config_path}")
    print("Starting trame server...")
    server.start()


if __name__ == "__main__":
    main()
