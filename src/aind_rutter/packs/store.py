"""Reading from a pack store over HTTPS (or ``file://``), and publishing to one."""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
import shutil
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from aind_rutter.packs.manifest import FileEntry, Manifest, PackRef

log = logging.getLogger(__name__)

DEFAULT_STORE = "https://aind-scratch-data.s3.amazonaws.com/rutter/packs/"
TIMEOUT_S = 60
RETRIES = 4
FREE_SPACE_MARGIN = 100 * 2**20

sleep: Callable[[float], None] = time.sleep  # replaced in tests


class NotInStore(FileNotFoundError):
    """The store answered that it has no such object."""


def file_url(store: str, pack: PackRef, path: str) -> str:
    """``<store>/<name>/<version>/<path>``, each segment quoted; ``path`` is
    ``manifest.json`` or ``files/<pack path>``."""
    base = store if store.endswith("/") else store + "/"
    quoted = "/".join(urllib.parse.quote(s) for s in (pack.name, pack.version))
    return base + quoted + "/" + urllib.parse.quote(path)


def _ssl_context() -> ssl.SSLContext:
    # A python.org build on macOS has no certificate store of its own.
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _open(url: str):
    """The response for ``url``, retried with backoff on a dropped connection
    or a server error; a 404 (or a missing file) raises ``NotInStore``."""
    delay = 1.0
    for attempt in range(1, RETRIES + 1):
        try:
            if url.startswith("file:"):
                return urllib.request.urlopen(url, timeout=TIMEOUT_S)
            return urllib.request.urlopen(
                url, timeout=TIMEOUT_S, context=_ssl_context()
            )
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                # An open S3 bucket answers 403 for a missing key.
                raise NotInStore(url) from e
            if e.code < 500 or attempt == RETRIES:
                raise
        except urllib.error.URLError as e:
            if isinstance(e.reason, FileNotFoundError):
                raise NotInStore(url) from e
            if attempt == RETRIES:
                raise
        except (ConnectionError, TimeoutError, http.client.HTTPException):
            if attempt == RETRIES:
                raise
        log.info("retrying %s (attempt %d of %d)", url, attempt + 1, RETRIES)
        sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


def read_text(url: str) -> str:
    with _open(url) as r:
        return r.read().decode("utf-8")


def exists(url: str) -> bool:
    try:
        with _open(url):
            return True
    except NotInStore:
        return False


def read_manifest(store: str, pack: PackRef) -> Manifest:
    try:
        text = read_text(file_url(store, pack, "manifest.json"))
    except NotInStore:
        raise NotInStore(f"{pack} is not published in {store}") from None
    return Manifest.from_json(text, pack)


def download(url: str, target: Path, entry: FileEntry, *, check_sha: bool) -> None:
    """Write ``url`` to ``target`` by way of a temporary name, so a failed or
    concurrent download never leaves a partial file under the real name."""
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free
    if free < entry.size + FREE_SPACE_MARGIN:
        raise OSError(
            f"{target.parent}: {free / 2**30:.1f} GiB free, too little for "
            f"{entry.size / 2**30:.2f} GiB. Free space or run `rutter-data "
            "cache clear`."
        )
    part = target.with_name(f".{target.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.part")
    digest = hashlib.sha256()
    size = 0
    try:
        with _open(url) as r, part.open("wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != entry.size:
            raise OSError(f"{url}: received {size} bytes of {entry.size}")
        if check_sha and digest.hexdigest() != entry.sha256:
            raise OSError(f"{url}: content does not match the manifest's sha256")
        place(part, target, entry.size)
    finally:
        part.unlink(missing_ok=True)


def place(part: Path, target: Path, size: int) -> None:
    """Rename into place. On Windows a rename fails while another process (a
    second worker, antivirus, the indexer) holds the target open; retried for
    about a second, and a target already there at the right size is success."""
    for attempt in range(6):
        try:
            os.replace(part, target)
            return
        except PermissionError:
            if target.exists() and target.stat().st_size == size:
                return
            if attempt == 5:
                raise
            sleep(0.2)


def publish(folder: Path, manifest: Manifest, store: str) -> None:
    """Upload every file, then the manifest, so a version exists only once
    complete. ``file://`` stores are copied to; S3 needs ``boto3``."""
    files = [(f"files/{p}", folder / p) for p in sorted(manifest.files)]
    if store.startswith("file:"):
        root = Path(urllib.request.url2pathname(urllib.parse.urlparse(store).path))
        base = root / manifest.pack.name / manifest.pack.version
        for key, src in files:
            (base / key).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, base / key)
        (base / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        return
    bucket, prefix = s3_location(store)
    try:
        import boto3
    except ImportError:
        raise SystemExit(
            "publishing to S3 needs boto3: uv run --with boto3 rutter-data publish ..."
        ) from None
    client = boto3.client("s3")
    key_base = f"{prefix}{manifest.pack.name}/{manifest.pack.version}/"
    for i, (key, src) in enumerate(files, 1):
        log.info("[%d/%d] %s", i, len(files), key)
        client.upload_file(str(src), bucket, key_base + key)
    client.put_object(
        Bucket=bucket,
        Key=key_base + "manifest.json",
        Body=manifest.to_json().encode(),
        ContentType="application/json",
    )


def s3_location(store: str) -> tuple[str, str]:
    """``(bucket, prefix)`` for an ``s3://`` or virtual-hosted S3 HTTPS store."""
    parsed = urllib.parse.urlparse(store)
    prefix = parsed.path.lstrip("/")
    prefix = prefix if not prefix or prefix.endswith("/") else prefix + "/"
    if parsed.scheme == "s3":
        return parsed.netloc, prefix
    host = parsed.netloc.split(".")
    if parsed.scheme == "https" and len(host) >= 4 and host[1].startswith("s3"):
        return host[0], prefix
    raise ValueError(f"{store}: publish writes to file:// or an S3 store")


def store_url(store: str) -> str:
    """A store as the URL it is read from: ``s3://bucket/prefix`` becomes its
    public HTTPS address, a plain folder becomes ``file://``."""
    if store.startswith("s3://"):
        bucket, prefix = s3_location(store)
        return f"https://{bucket}.s3.amazonaws.com/{prefix}"
    if "://" not in store:
        return Path(store).expanduser().resolve().as_uri() + "/"
    return store if store.endswith("/") else store + "/"
