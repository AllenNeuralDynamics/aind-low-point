"""Bundle configs with the files they read into one folder that runs anywhere.

Each config is copied as written, with its ``<config>.plan.yml`` if present.
Every file a config reads is copied under the ``paths:`` name or pack it lives
in, and a ``rutter.paths.yml`` at the top points those names and packs at the
copies, so nothing inside is rewritten: the configs find it by searching
upward. A file a config names by an absolute path outside every ``paths:``
name cannot be redirected that way; it is listed, and the bundle refused
unless ``--allow-outside``.

    uv run --python 3.13 python scripts/bundle_config.py \\
        examples/build5-template-config.yml examples/786864-config.yml \\
        --output build5-bundle            # a folder
        --output build5-bundle.tar.zst    # or one archive of it

Packs are copied only as far as the configs read them; a recipient who is
online needs no bundle at all, since packs come from the store.
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

from aind_rutter.config import ConfigModel
from aind_rutter.config.locations import (
    PATHS_FILE_NAME,
    Loaded,
    Locator,
    load_located,
)

ARCHIVE_SUFFIX = ".tar.zst"


def read_files(config: Path, loaded: Loaded, locator: Locator) -> set[Path]:
    """Every existing file the config reads, with its companions (same folder
    and stem, as an ``.obj``'s ``.mtl``), bulk patterns expanded."""
    expanded = ConfigModel.from_yaml(config, locator).model_dump(mode="json")
    found: set[Path] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key != "paths":
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and os.path.isabs(node):
            path = Path(node)
            if path.is_file():
                found.add(path)
            elif path.is_dir():
                found.update(p for p in path.rglob("*") if p.is_file())

    walk(loaded.data)
    walk(expanded)
    for path in list(found):
        stem = path.name.split(".", 1)[0]
        found.update(
            p
            for p in path.parent.glob(f"{stem}.*")
            if p.is_file() and p.name.split(".", 1)[0] == stem
        )
    return found


def roots(loaded: Loaded) -> list[tuple[Path, str, str]]:
    """``(folder, paths-file key, bundle folder)`` for each pack and each
    ``paths:`` name written as a folder of its own, longest first. A derived
    name (``${.base}/x``) follows its base and needs no line."""
    resolved = loaded.data.get("paths") or {}
    out = []
    for name, loc in loaded.locations.items():
        if loc.pack is not None:
            key, dest = str(loc.pack), f"packs/{loc.pack.name}/{loc.pack.version}"
        elif "${" not in loc.value and os.path.isabs(str(resolved[name])):
            key, dest = name, f"data/{name}"
        else:
            continue
        out.append((Path(os.path.abspath(str(resolved[name]))), key, dest))
    return sorted(out, key=lambda r: -len(r[0].parts))


def bundle(
    configs: list[Path], out: Path, locator: Locator, *, allow_outside: bool
) -> int:
    sources: dict[str, Path] = {}  # paths-file key -> the folder copied
    lines: dict[str, str] = {}
    copies: dict[Path, Path] = {}
    outside: dict[Path, list[str]] = {}
    for config in configs:
        loaded = load_located(config, locator)
        config_roots = roots(loaded)
        for folder, key, dest in config_roots:
            # One paths file serves every config, so a name means one folder.
            if sources.setdefault(key, folder) != folder:
                raise SystemExit(
                    f"{config}: {key} is {folder}, but {sources[key]} in an "
                    "earlier config; bundle them separately"
                )
            lines[key] = dest
        for path in read_files(config, loaded, locator):
            for folder, _, dest in config_roots:
                if path == folder or folder in path.parents:
                    copies[path] = Path(dest) / path.relative_to(folder)
                    break
            else:
                outside.setdefault(path, []).append(config.name)
    if outside:
        print("Named by an absolute path outside every paths: name:", file=sys.stderr)
        for path, names in sorted(outside.items()):
            print(f"  {path}  ({', '.join(names)})", file=sys.stderr)
        if not allow_outside:
            print(
                "Give each a paths: name, or pass --allow-outside to bundle "
                "without them.",
                file=sys.stderr,
            )
            return 1

    out.mkdir(parents=True)
    for config in configs:
        shutil.copy2(config, out / config.name)
        plan = config.with_suffix(".plan.yml")
        if plan.is_file():
            shutil.copy2(plan, out / plan.name)
    for src, rel in sorted(copies.items()):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / rel)
    # A pack key is name/version; a paths: name never holds a slash.
    packs = {k: v for k, v in lines.items() if "/" in k}
    paths_file: dict[str, object] = {
        k: v for k, v in sorted(lines.items()) if k not in packs
    }
    if packs:
        paths_file["packs"] = dict(sorted(packs.items()))
    (out / PATHS_FILE_NAME).write_text(
        "# Folders relative to this file; every config here finds it.\n"
        + yaml.safe_dump(paths_file, sort_keys=False)
    )
    run = f"uv run --python 3.13 rutter-plan {configs[0].name}"
    (out / "README.md").write_text(
        "# Rutter bundle\n\n"
        "Configs as written, the files they read, and `rutter.paths.yml`,\n"
        "which points the configs' `paths:` names and packs at those files.\n"
        "With aind-rutter installed (see its README), from this folder:\n\n"
        f"```bash\n{run}\n```\n"
    )
    total = sum(p.stat().st_size for p in copies)
    print(f"{len(configs)} config(s), {len(copies)} files, {total / 1e6:.1f} MB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help=f"a folder, or *{ARCHIVE_SUFFIX}"
    )
    parser.add_argument("--allow-outside", action="store_true")
    args = parser.parse_args()
    locator = Locator.from_environment()
    configs = [c.resolve() for c in args.configs]
    if len({c.name for c in configs}) != len(configs):
        raise SystemExit("two configs share a file name")
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    if not args.output.name.endswith(ARCHIVE_SUFFIX):
        return bundle(configs, args.output, locator, allow_outside=args.allow_outside)
    name = args.output.name.removesuffix(ARCHIVE_SUFFIX)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / name
        code = bundle(configs, folder, locator, allow_outside=args.allow_outside)
        if code == 0:
            out = args.output.resolve()
            subprocess.run(
                ["tar", "--zstd", "-cf", str(out), name], cwd=tmp, check=True
            )
            print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return code


if __name__ == "__main__":
    sys.exit(main())
