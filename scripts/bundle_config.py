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


def _resolve_paths_block(raw_paths: dict) -> dict[str, str]:
    """Resolve OmegaConf interpolation in just the ``paths`` block."""
    from omegaconf import OmegaConf

    oc = OmegaConf.create({"paths": raw_paths})
    return OmegaConf.to_container(oc, resolve=True)["paths"]


def _named_roots_from_paths(
    raw_paths: dict,
) -> list[tuple[str, Path]]:
    """Return ``(key, abs_path)`` for every ``paths.*`` entry whose
    resolved value is an absolute path on disk. Sorted by path length
    descending so longest-prefix matching falls out naturally when
    multiple roots overlap (e.g. ``atlas_dir`` is a child of
    ``template_dir``)."""
    resolved = _resolve_paths_block(raw_paths)
    roots: list[tuple[str, Path]] = []
    for key, val in resolved.items():
        if isinstance(val, str) and val.startswith("/"):
            roots.append((key, Path(val)))
    roots.sort(key=lambda kv: len(str(kv[1])), reverse=True)
    return roots


def _bundle_path_for(
    abs_path: Path,
    named_roots: list[tuple[str, Path]],
    namespace: str | None,
    misc_used: dict[str, int],
) -> str:
    """Compute a clean ``data/...`` bundle path for *abs_path*.

    Files that fall under a named ``paths.*`` root are mirrored relative
    to that root under ``data/<key>/`` (and ``data/<namespace>/<key>/``
    if a namespace is set). Files outside every root land in
    ``data/_misc/`` with a counter suffix on collisions — no machine
    paths leak into the bundle.
    """
    prefix = "data" if namespace is None else f"data/{namespace}"
    for key, root_path in named_roots:
        try:
            rel = abs_path.relative_to(root_path)
        except ValueError:
            continue
        return f"{prefix}/{key}/{rel}"
    base = abs_path.name
    counter = misc_used.get(base, 0)
    if counter:
        stem, dot, ext = base.rpartition(".")
        base = f"{stem}_{counter}.{ext}" if dot else f"{base}_{counter}"
    misc_used[abs_path.name] = counter + 1
    return f"{prefix}/_misc/{base}"


def _rewrite_paths_block(
    raw_paths: dict,
    namespace: str | None,
) -> dict:
    """Rewrite a ``paths.*`` block so each absolute-path entry points at
    its bundle subdirectory and ``bundle`` is injected at the top.

    Interpolation chains whose resolved value happens to be an absolute
    path (e.g. ``atlas_dir: ${.template_dir}/...``) are also flattened
    to a direct ``${paths.bundle}/data/<key>`` literal — we don't try
    to preserve chain semantics because the bundle layout already gives
    each named root its own clean directory.
    """
    resolved = _resolve_paths_block(raw_paths)
    prefix = "data" if namespace is None else f"data/{namespace}"
    new_paths: dict = {
        "bundle": "${oc.env:BUNDLE_DIR,/EDIT_ME/path/to/extracted/bundle}",
    }
    for key, val in raw_paths.items():
        rval = resolved.get(key)
        if isinstance(rval, str) and rval.startswith("/"):
            new_paths[key] = f"${{paths.bundle}}/{prefix}/{key}"
        else:
            new_paths[key] = val
    return new_paths


def _rewrite_inline_paths(
    obj,
    file_to_bundle_rel: dict[Path, str],
):
    """Walk *obj* and replace any leaf string that's an absolute path of
    a bundled file with ``${paths.bundle}/<bundle_rel>``.

    The common case (every file is referenced via ``${paths.foo}/...``
    interpolation) needs no rewriting here because the ``paths.*`` block
    rewrite already handles it. This is the fall-through for inline
    absolute paths — they get sent to ``data/_misc/`` and the YAML's
    pointer is updated to match.
    """
    if isinstance(obj, dict):
        return {k: _rewrite_inline_paths(v, file_to_bundle_rel) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_inline_paths(v, file_to_bundle_rel) for v in obj]
    if isinstance(obj, str) and obj.startswith("/"):
        rel = file_to_bundle_rel.get(Path(obj).resolve())
        if rel is not None:
            return f"${{paths.bundle}}/{rel}"
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

    # 2. For each config, build its named-root list (paths.* entries
    #    whose resolved values are absolute paths) and pick a namespace
    #    only when there are multiple configs sharing the bundle.
    cfg_to_roots: dict[Path, list[tuple[str, Path]]] = {}
    cfg_to_raw: dict[Path, dict] = {}
    cfg_to_namespace: dict[Path, str | None] = {}
    multi = len(args.config) > 1
    for cp in args.config:
        with cp.open() as f:
            raw = yaml.safe_load(f)
        cfg_to_raw[cp] = raw
        cfg_to_roots[cp] = _named_roots_from_paths(raw.get("paths", {}) or {})
        cfg_to_namespace[cp] = cp.stem if multi else None

    # 3. Decide bundle-relative target for every file. Use the longest
    #    matching named root from any config; fall back to ``_misc``.
    file_to_bundle_rel: dict[Path, str] = {}
    misc_used: dict[str, int] = {}
    # Aggregate roots across all configs (longest path wins). For multi
    # config setups we also need to know which namespace the root came
    # from; track that alongside.
    flat_roots: list[tuple[str, Path, str | None]] = []
    for cp, roots in cfg_to_roots.items():
        ns = cfg_to_namespace[cp]
        for key, root_path in roots:
            flat_roots.append((key, root_path, ns))
    flat_roots.sort(key=lambda t: len(str(t[1])), reverse=True)

    for abs_path in sorted_files:
        matched = False
        for key, root_path, ns in flat_roots:
            try:
                rel = abs_path.relative_to(root_path)
            except ValueError:
                continue
            prefix = "data" if ns is None else f"data/{ns}"
            file_to_bundle_rel[abs_path] = f"{prefix}/{key}/{rel}"
            matched = True
            break
        if not matched:
            base = abs_path.name
            counter = misc_used.get(base, 0)
            slug = base
            if counter:
                stem, dot, ext = base.rpartition(".")
                slug = f"{stem}_{counter}.{ext}" if dot else f"{base}_{counter}"
            misc_used[base] = counter + 1
            file_to_bundle_rel[abs_path] = f"data/_misc/{slug}"

    # 4. Stage in temp dir, copy files into their clean bundle paths.
    with tempfile.TemporaryDirectory(prefix="bundle-") as tmp:
        tmpdir = Path(tmp)
        bundle_root = tmpdir / bundle_name
        bundle_root.mkdir()

        total_bytes = 0
        for abs_path, bundle_rel in sorted(file_to_bundle_rel.items()):
            dst = bundle_root / bundle_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # copyfile + chmod 0o644 so the copy is readable on the
            # other side. shutil.copy2 preserves the source perms which
            # break when the source uses ACLs (common on scratch FSs).
            shutil.copyfile(abs_path, dst)
            os.chmod(dst, 0o644)
            total_bytes += dst.stat().st_size
        print(f"Copied {len(sorted_files)} files, {total_bytes / 1e6:.1f} MB raw")

        # 5. Rewrite each source config so every absolute path goes
        #    through ``${paths.bundle}/data/<key>/`` rather than mirroring
        #    the original filesystem layout.
        config_filenames: list[str] = []
        for cp in args.config:
            raw = cfg_to_raw[cp]
            ns = cfg_to_namespace[cp]
            raw["paths"] = _rewrite_paths_block(raw.get("paths", {}) or {}, ns)
            for k in list(raw.keys()):
                if k != "paths":
                    raw[k] = _rewrite_inline_paths(raw[k], file_to_bundle_rel)
            out_name = cp.name
            with (bundle_root / out_name).open("w") as f:
                yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)
            config_filenames.append(out_name)

        # 4. README so the recipient knows what to do.
        configs_block = "\n".join(f"├── {name}" for name in config_filenames)
        primary_cfg = config_filenames[0]
        run_examples = "\n".join(
            f"BUNDLE_DIR=$(pwd)/{bundle_name} \\\n"
            f"    uv run --python 3.13 python examples/launch_trame_config.py "
            f"{bundle_name}/{name}"
            for name in config_filenames
        )
        (bundle_root / "README.md").write_text(
            f"""# {bundle_name}

A self-contained aind-low-point planning asset pack: every mesh, mask, and
transform referenced by the bundled config(s) is included under `data/`,
with paths rewritten to a single `${{paths.bundle}}` indirection so the
config travels with the data.

## What's inside

```
{bundle_name}/
├── README.md         (this file)
{configs_block}
└── data/             (every referenced asset; original absolute paths preserved)
```

## Prerequisites

1. **Python 3.13** — `python-fcl` does not ship 3.14 wheels. The launcher uses
   `uv run --python 3.13 …`, which will fetch 3.13 if it isn't installed.
2. **uv** — <https://docs.astral.sh/uv/>. Install with `curl -LsSf
   https://astral.sh/uv/install.sh | sh` or your distro's package manager.
3. **aind-low-point** — clone the repo and install in editable mode:
   ```bash
   git clone git@github.com:AllenNeuralDynamics/aind-low-point.git
   cd aind-low-point
   uv sync --python 3.13
   ```

## Get started

```bash
# 1. Extract the pack anywhere
tar -xf {bundle_name}.tar.zst -C ~/data

# 2. Launch the trame web app from your aind-low-point checkout. Set
#    BUNDLE_DIR (the configs default to it via OmegaConf) or edit
#    paths.bundle in the YAML to the absolute extracted path.
cd ~/aind-low-point   # or wherever you cloned the repo
{run_examples}
```

The launcher prints a URL (default <http://localhost:8080/index.html>);
open it in any browser. Stop the server with Ctrl-C.

## Quick UI tour

- **Probe / Arc / Probe type / Target** drop-downs — pick which probe you're
  editing, which arc it's bound to, which mesh to render it as, and which
  catalog target it aims at.
- **R / A / Depth** sliders — translate the probe entry point in mm; depth
  pushes the tip past the target along the probe axis.
- **AP tilt / ML tilt / Spin** sliders — angles in degrees. AP is shared
  across probes on the same arc when bound; ML and spin are per-probe.
- **Set target** — apply the dropdown selection; resets depth to zero.
- **Save YAML** (if launched with `--save`) — writes the current plan to
  the path you passed.
- **CCF region search / opacity** (if launched with `--ccf-volume <path>`)
  — toggle individual brain regions on/off in the 3D view.

Probes that aim at a CCF-derived target are coloured to match the Allen
region colour automatically; probe meshes can be swapped on the fly via
the Probe type dropdown.

## Editing the config

Each config has a `paths.bundle` line at the very top. By default it
reads `${{oc.env:BUNDLE_DIR,/EDIT_ME/path/to/extracted/bundle}}`, so
you can either:

1. set `BUNDLE_DIR` in the launching shell (recommended), or
2. replace the placeholder with the absolute extracted path.

Everything else under `paths:` is interpolated relative to that single
indirection.

## Optional: regenerate this pack

```bash
uv run --python 3.13 python scripts/bundle_config.py \\
    examples/{primary_cfg} \\
    --output {bundle_name}.tar.zst
```

(Pass multiple configs to union their asset sets in a single bundle.)
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
