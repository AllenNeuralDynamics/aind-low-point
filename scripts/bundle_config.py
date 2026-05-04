"""Bundle one or more config files with all referenced assets into a
portable tar.zst.

Walks every (validated) ConfigModel to find every referenced file path,
copies them into a staging directory keyed off the original absolute
paths, rewrites each config so that all path references go through a
single ``${paths.bundle}`` indirection, and tars/compresses the result.

When multiple configs are passed, the file set is the **union**
(deduplicated) so the bundle stays small even with shared assets.
Each rewritten config is written into the bundle root with a filename
derived from the source.

Receiving end:

    tar -xf <bundle>.tar.zst -C /some/dir
    # edit one of the configs in <bundle>/, set paths.bundle to the
    # extracted dir, OR launch with the BUNDLE_DIR env var
    BUNDLE_DIR=/some/dir/<bundle> uv run --python 3.13 \\
        python examples/launch_trame_config.py /some/dir/<bundle>/config.yml

Usage:
    # single config
    uv run --python 3.13 python scripts/bundle_config.py \\
        examples/786864-config.yml \\
        --output 786864-bundle.tar.zst

    # multiple configs, unioned asset set
    uv run --python 3.13 python scripts/bundle_config.py \\
        examples/build5-template-config.yml examples/786864-config.yml \\
        --output build5-asset-pack.tar.zst
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from aind_low_point.config import ConfigModel


def collect_existing_file_paths(obj, out: set[Path]) -> None:
    """Walk a (possibly nested) container and collect every value that
    resolves to an existing file on disk. Accepts strings and Path objects."""
    if isinstance(obj, dict):
        for v in obj.values():
            collect_existing_file_paths(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_existing_file_paths(v, out)
    elif isinstance(obj, (str, Path)):
        s = str(obj)
        if s.startswith("/"):
            p = Path(s)
            if p.is_file():
                out.add(p.resolve())


def deep_rewrite_absolute_paths(obj, replacement_prefix: str):
    """Replace every value that looks like an absolute path with
    ``replacement_prefix + <full_original_path>``. Mirrors the original
    filesystem layout under the replacement prefix, so any chain of
    interpolations in the source YAML continues to work after rewriting
    (we don't need to find a common ancestor)."""
    if isinstance(obj, dict):
        return {k: deep_rewrite_absolute_paths(v, replacement_prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_rewrite_absolute_paths(v, replacement_prefix) for v in obj]
    if isinstance(obj, str) and obj.startswith("/"):
        # Strip leading '/' so we don't double up on the replacement prefix.
        return f"{replacement_prefix}{obj}"
    return obj


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "config",
        type=Path,
        nargs="+",
        help="One or more YAML config files; asset sets are unioned.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output tar.zst path",
    )
    p.add_argument(
        "--bundle-name",
        default=None,
        help=(
            "Name of the directory inside the tarball "
            "(default: derived from output filename)"
        ),
    )
    p.add_argument(
        "--zstd-level",
        type=int,
        default=19,
        help="zstd compression level 1-22 (default 19)",
    )
    args = p.parse_args()

    for cp in args.config:
        if not cp.is_file():
            print(f"Config not found: {cp}", file=sys.stderr)
            return 1

    bundle_name = args.bundle_name
    if bundle_name is None:
        stem = args.output.name
        for suffix in (".tar.zst", ".tar.zstd", ".tzst", ".tar"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        bundle_name = stem

    # 1. Validate every config so bulk / range / atlas-pack specs are
    #    expanded into concrete per-asset srcs, then union the file sets.
    files: set[Path] = set()
    missing: set[str] = set()
    cfg_dumps: dict[Path, dict] = {}
    for cp in args.config:
        print(f"Loading {cp}")
        cfg = ConfigModel.from_yaml(cp)
        dump = cfg.model_dump(mode="python")
        cfg_dumps[cp] = dump
        per_cfg: set[Path] = set()
        collect_existing_file_paths(dump, per_cfg)
        files |= per_cfg
        # also note any concrete absolute paths that don't exist
        walk: list = [dump]
        while walk:
            node = walk.pop()
            if isinstance(node, dict):
                walk.extend(node.values())
            elif isinstance(node, (list, tuple)):
                walk.extend(node)
            elif isinstance(node, (str, Path)):
                s = str(node)
                if s.startswith("/") and not Path(s).exists():
                    missing.add(s)
        print(f"  {len(per_cfg)} files referenced")

    if not files:
        print("No existing file references found.", file=sys.stderr)
        return 2

    if missing:
        print("WARNING: some referenced paths do not exist on this machine:")
        for m in sorted(missing):
            print(f"  {m}")
        print("They will be skipped from the bundle.")

    sorted_files = sorted(files)
    print(f"Union: {len(sorted_files)} files across {len(args.config)} config(s)")

    # 2. Stage the bundle in a temp directory. Mirror each file's full
    #    original path under ``<bundle>/data/`` so rewriting absolute
    #    paths is just a string-prefix concat — no common-ancestor logic
    #    needed (which can't span entries shallower than the deepest
    #    bundled path, like ``base_path: /mnt/vast/scratch``).
    with tempfile.TemporaryDirectory(prefix="bundle-") as tmp:
        tmpdir = Path(tmp)
        bundle_root = tmpdir / bundle_name
        bundle_data = bundle_root / "data"
        bundle_data.mkdir(parents=True)

        total_bytes = 0
        for src in sorted_files:
            # src is absolute; strip leading "/" and re-anchor under data/.
            rel = Path(*src.parts[1:])
            dst = bundle_data / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Use copyfile (data only) + explicit chmod so the copy is
            # readable by anyone who extracts the bundle. shutil.copy2
            # preserves the source perms verbatim but drops ACLs, which
            # leaves files mode 0000 when the source relied on an ACL
            # (common on shared scratch filesystems).
            shutil.copyfile(src, dst)
            os.chmod(dst, 0o644)
            total_bytes += dst.stat().st_size
        print(f"Copied {len(sorted_files)} files, {total_bytes / 1e6:.1f} MB raw")

        # 3. Rewrite each source config: every absolute path becomes
        #    ``${paths.bundle}/data<original_absolute_path>``.
        replacement = "${paths.bundle}/data"
        config_filenames: list[str] = []
        for cp in args.config:
            with cp.open() as f:
                raw = yaml.safe_load(f)
            rewritten = deep_rewrite_absolute_paths(raw, replacement)
            existing_paths = rewritten.get("paths", {}) or {}
            new_paths = {
                "bundle": "${oc.env:BUNDLE_DIR,/EDIT_ME/path/to/extracted/bundle}",
                **existing_paths,
            }
            rewritten["paths"] = new_paths
            out_name = cp.name
            with (bundle_root / out_name).open("w") as f:
                yaml.safe_dump(
                    rewritten, f, default_flow_style=False, sort_keys=False
                )
            config_filenames.append(out_name)

        # 4. README so the recipient knows what to do.
        configs_block = "\n".join(f"├── {name}" for name in config_filenames)
        run_block = "\n".join(
            f"BUNDLE_DIR=$(pwd)/{bundle_name} \\\n"
            f"    uv run --python 3.13 python examples/launch_trame_config.py "
            f"{bundle_name}/{name}"
            for name in config_filenames
        )
        (bundle_root / "README.md").write_text(
            f"""# {bundle_name}

Self-contained aind-low-point planning asset pack.

## Layout

```
{bundle_name}/
├── README.md
{configs_block}
└── data/             (all referenced assets, original absolute paths preserved)
```

## Run any of the included configs

```bash
# After: tar -xf {bundle_name}.tar.zst -C /some/dir
cd /some/dir

{run_block}
```

Or edit one of the configs in `{bundle_name}/` and replace the
`paths.bundle` line with the absolute extracted path.
"""
        )

        # 5. tar with zstd compression.
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["ZSTD_CLEVEL"] = str(args.zstd_level)
        cmd = ["tar", "--zstd", "-cf", str(out), bundle_name]
        print(f"Running: {' '.join(cmd)} (cwd={tmpdir})")
        subprocess.run(cmd, cwd=tmpdir, env=env, check=True)

    out_size = args.output.stat().st_size
    ratio = out_size / total_bytes if total_bytes else 0
    print(
        f"Wrote {args.output}: {out_size / 1e6:.1f} MB compressed "
        f"(ratio {ratio:.2f} of {total_bytes / 1e6:.1f} MB raw)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
