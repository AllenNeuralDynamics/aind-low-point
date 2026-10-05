"""The ``rutter-data`` command: publish, pull and check packs, and see the cache.

rutter-data pull    hardware/v1 [--to DIR]
rutter-data publish DIR hardware/v2 [--store URL]
rutter-data verify  hardware/v1 [DIR]
rutter-data cache   [clear [PACK]]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from aind_rutter.config.locations import Locator, record_pack_folder
from aind_rutter.packs import store as store_io
from aind_rutter.packs.cache import PackCache
from aind_rutter.packs.manifest import (
    Manifest,
    PackRef,
    build_manifest,
    file_sha256,
    pack_files,
    portability_problems,
)


def _cache(*, offline: bool = False) -> tuple[PackCache, Path | None]:
    locator = Locator.from_environment(offline=offline)
    return locator.pack_cache(), locator.machine_file


def _pull(args: argparse.Namespace) -> int:
    pack = PackRef.parse(args.pack)
    cache, machine_file = _cache()
    if args.to is None:
        fetched = cache.open(pack).ensure_all()
        print(f"{pack}: {len(fetched)} file(s) fetched into {cache.version_dir(pack)}")
        return 0
    folder = args.to.expanduser().resolve()
    manifest = store_io.read_manifest(cache.store, pack)
    for i, (path, entry) in enumerate(sorted(manifest.files.items()), 1):
        target = folder / path
        if target.is_file() and target.stat().st_size == entry.size:
            continue
        print(f"[{i}/{len(manifest.files)}] {path}")
        store_io.download(
            store_io.file_url(cache.store, pack, f"files/{path}"),
            target,
            entry,
            check_sha=cache.settings.verify_downloads,
        )
    if machine_file is None:
        print(f"{pack} copied to {folder}")
        return 0
    record_pack_folder(machine_file, pack, folder)
    print(f"{pack} copied to {folder} and recorded in {machine_file}")
    return 0


def _publish(args: argparse.Namespace) -> int:
    pack = PackRef.parse(args.pack)
    store = store_io.store_url(args.store or _cache()[0].settings.pack_store)
    folder = args.folder.resolve()
    errors, warnings = portability_problems([p for p, _ in pack_files(folder)])
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1
    if store_io.exists(store_io.file_url(store, pack, "manifest.json")):
        print(
            f"{pack} is already published in {store}; versions never change, "
            "so publish a new version",
            file=sys.stderr,
        )
        return 1
    manifest = build_manifest(folder, pack)
    print(
        f"{pack}: {len(manifest.files)} files, {manifest.total_bytes() / 1e6:.1f} MB "
        f"to {store}"
    )
    store_io.publish(folder, manifest, args.store or store)
    print(f"published {pack}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    pack = PackRef.parse(args.pack)
    cache, _ = _cache(offline=args.offline)
    if args.folder is None:
        cached = cache.open(pack)
        manifest, folder, whole = cached.manifest, cached.folder, False
    else:
        manifest = store_io.read_manifest(cache.store, pack)
        folder, whole = args.folder.resolve(), True
    bad = verify_folder(manifest, folder, whole=whole)
    for line in bad:
        print(line)
    checked = "every file" if whole else "the files the cache holds"
    verdict = f"{len(bad)} problem(s)" if bad else "OK"
    print(f"{pack}: {checked} in {folder}: {verdict}")
    return 1 if bad else 0


def verify_folder(manifest: Manifest, folder: Path, *, whole: bool) -> list[str]:
    """Problems with ``folder`` against ``manifest``: missing files (when it
    should hold the whole pack), wrong sizes or contents, and extra files."""
    out = []
    present = dict(pack_files(folder))
    for path, entry in sorted(manifest.files.items()):
        local = present.pop(path, None)
        if local is None:
            if whole:
                out.append(f"missing  {path}")
        elif local.stat().st_size != entry.size:
            out.append(f"size     {path}: {local.stat().st_size} != {entry.size}")
        elif file_sha256(local) != entry.sha256:
            out.append(f"content  {path}: sha256 differs")
    out += [f"extra    {p}" for p in sorted(present)]
    return out


def _cache_cmd(args: argparse.Namespace) -> int:
    cache, _ = _cache()
    if args.action == "clear":
        removed = cache.clear(args.pack)
        freed = sum(v.size for v in removed) / 2**30
        print(f"removed {len(removed)} version(s), {freed:.2f} GiB")
        return 0
    versions = cache.versions() if cache.root.is_dir() else []
    s = cache.settings
    total = sum(v.size for v in versions)
    print(
        f"{cache.top}: {total / 2**30:.2f} of {s.cache_max_gb:g} GiB; versions "
        f"unused for {s.cache_max_age_days:g} days are removed"
    )
    now = time.time()
    for v in sorted(versions, key=lambda v: -v.last_used):
        days = (now - v.last_used) / 86400
        print(f"  {v.label:<32} {v.size / 1e6:9.1f} MB  used {days:.0f} day(s) ago")
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="rutter-data", description="Publish, pull and check packs."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pull = commands.add_parser(
        "pull", help="fetch a whole pack: into the cache, or into a folder that stays"
    )
    pull.add_argument("pack", help="name/version, e.g. hardware/v1")
    pull.add_argument(
        "--to",
        type=Path,
        help="copy into this folder and record it in the machine's paths file",
    )
    pull.set_defaults(run=_pull)

    publish = commands.add_parser(
        "publish", help="write a folder as a new pack version"
    )
    publish.add_argument("folder", type=Path)
    publish.add_argument("pack", help="name/version; refused if it already exists")
    publish.add_argument(
        "--store",
        help="s3://bucket/prefix, an S3 URL or file:// (default: the pack store)",
    )
    publish.set_defaults(run=_publish)

    verify = commands.add_parser(
        "verify", help="check a folder or the cache against a pack's manifest"
    )
    verify.add_argument("pack")
    verify.add_argument(
        "folder", type=Path, nargs="?", help="a full copy (default: the cache)"
    )
    verify.add_argument(
        "--offline",
        action="store_true",
        help="check the cache without reading the manifest from the store",
    )
    verify.set_defaults(run=_verify)

    cache = commands.add_parser(
        "cache", help="what the cache holds; `clear [PACK]` empties it"
    )
    cache.add_argument("action", nargs="?", choices=["clear"])
    cache.add_argument("pack", nargs="?", help="a pack name or name/version")
    cache.set_defaults(run=_cache_cmd)

    args = parser.parse_args()
    try:
        sys.exit(args.run(args))
    except (OSError, ValueError) as e:
        print(f"rutter-data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
