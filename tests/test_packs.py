"""Packs, the store and the cache. Imports only ``aind_rutter.packs`` so the
macOS and Windows jobs run it without the rest of the stack."""

from __future__ import annotations

import io
import os
import time
import urllib.error
from pathlib import Path

import pytest

from aind_rutter.packs import store as store_io
from aind_rutter.packs.cache import (
    DAY_S,
    LAYOUT,
    CachedVersion,
    PackCache,
    PackSettings,
    select_removals,
)
from aind_rutter.packs.manifest import (
    FileEntry,
    Manifest,
    PackRef,
    build_manifest,
    check_pack_path,
    file_sha256,
    pack_files,
    portability_problems,
)

PACK = PackRef("hardware", "v1")
NOW = 1_800_000_000.0


@pytest.fixture(autouse=True)
def _no_pack_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "RUTTER_PACK_STORE",
        "RUTTER_CACHE",
        "RUTTER_CACHE_MAX_GB",
        "RUTTER_CACHE_MAX_AGE_DAYS",
        "RUTTER_OFFLINE",
        "RUTTER_VERIFY_DOWNLOADS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(store_io, "sleep", lambda s: None)


def _write(folder: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(data)
    return folder


FILES = {
    "cone/cone.obj": b"v 0 0 0\n",
    "cone/cone.mtl": b"newmtl a\n",
    "cone/other.obj": b"v 1 1 1\n",
    "well/well.step": b"ISO-10303-21;\n",
}


@pytest.fixture
def store(tmp_path: Path) -> str:
    """A ``file://`` store holding ``hardware/v1``."""
    src = _write(tmp_path / "src", FILES)
    url = store_io.store_url(str(tmp_path / "store"))
    store_io.publish(src, build_manifest(src, PACK), url)
    return url


def _cache(tmp_path: Path, store: str, **kw: object) -> PackCache:
    return PackCache(
        PackSettings(pack_store=store, cache=tmp_path / "cache", **kw)  # type: ignore[arg-type]
    )


# ---- names and manifests ----


@pytest.mark.parametrize("text", ["hardware/v1", "mouse-t1/v1.0", " a/b_2 "])
def test_a_pack_reference_parses(text: str) -> None:
    ref = PackRef.parse(text)
    assert str(ref) == text.strip()


@pytest.mark.parametrize("text", ["Hardware/v1", "hardware", "a/b/c", "/v1", "a/.b"])
def test_a_malformed_pack_reference_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="not a pack"):
        PackRef.parse(text)


def test_a_manifest_round_trips(tmp_path: Path) -> None:
    src = _write(tmp_path / "src", {**FILES, ".DS_Store": b"", "__MACOSX/x": b""})
    manifest = build_manifest(src, PACK)
    assert sorted(manifest.files) == sorted(FILES)
    assert manifest.files["cone/cone.obj"] == FileEntry(
        len(FILES["cone/cone.obj"]), file_sha256(src / "cone/cone.obj")
    )
    assert Manifest.from_json(manifest.to_json(), PACK) == manifest
    assert manifest.total_bytes() == sum(len(v) for v in FILES.values())
    assert manifest.under("cone") == [
        "cone/cone.mtl",
        "cone/cone.obj",
        "cone/other.obj",
    ]
    assert manifest.companions("cone/cone.obj") == ["cone/cone.mtl"]
    assert manifest.is_folder("well") and not manifest.is_folder("we")


def test_a_manifest_for_another_pack_or_layout_is_refused(tmp_path: Path) -> None:
    text = build_manifest(_write(tmp_path, FILES), PACK).to_json()
    with pytest.raises(ValueError, match="manifest is for"):
        Manifest.from_json(text, PackRef("hardware", "v2"))
    with pytest.raises(ValueError, match="newer rutter"):
        Manifest.from_json(
            text.replace('"manifest_version": 1', '"manifest_version": 2'), PACK
        )


@pytest.mark.parametrize(
    "path", ["", "/etc/passwd", "../x", "a/../../x", "a//b", "C:/x", "a\\b", "./a"]
)
def test_a_manifest_path_outside_the_pack_is_refused(path: str) -> None:
    with pytest.raises(ValueError, match="relative POSIX path"):
        check_pack_path(path)


def test_portability_problems() -> None:
    errors, warnings = portability_problems(
        [
            "a/Mesh.obj",
            "a/mesh.obj",
            "b/x?.obj",
            "c/aux.txt",
            "d/trailing.",
            "e/" + "x" * 160,
        ]
    )
    assert len(errors) == 4
    assert any("differ only in case" in e for e in errors)
    assert any("'?'" in e for e in errors)
    assert any("reserved" in e for e in errors)
    assert any("trailing" in e for e in errors)
    assert len(warnings) == 1


def test_an_empty_folder_is_not_a_pack(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no files"):
        build_manifest(tmp_path, PACK)


# ---- the store ----


def test_publishing_writes_files_then_the_manifest(tmp_path: Path, store: str) -> None:
    assert store_io.exists(store_io.file_url(store, PACK, "manifest.json"))
    assert sorted(store_io.read_manifest(store, PACK).files) == sorted(FILES)
    with pytest.raises(store_io.NotInStore, match="not published"):
        store_io.read_manifest(store, PackRef("hardware", "v9"))


def test_store_urls() -> None:
    assert store_io.store_url("s3://bucket/rutter/packs") == (
        "https://bucket.s3.amazonaws.com/rutter/packs/"
    )
    assert store_io.s3_location(store_io.DEFAULT_STORE) == (
        "aind-scratch-data",
        "rutter/packs/",
    )
    assert store_io.file_url("https://h/p/", PACK, "files/a b.obj").endswith(
        "/p/hardware/v1/files/a%20b.obj"
    )
    with pytest.raises(ValueError, match="publish writes"):
        store_io.s3_location("https://example.org/packs/")


def _entry(data: bytes) -> FileEntry:
    import hashlib

    return FileEntry(len(data), hashlib.sha256(data).hexdigest())


def test_a_download_lands_whole_or_not_at_all(tmp_path: Path, store: str) -> None:
    url = store_io.file_url(store, PACK, "files/cone/cone.obj")
    good = _entry(FILES["cone/cone.obj"])
    target = tmp_path / "out" / "cone.obj"

    with pytest.raises(OSError, match="received"):
        store_io.download(
            url, target, FileEntry(good.size + 1, good.sha256), check_sha=True
        )
    with pytest.raises(OSError, match="sha256"):
        store_io.download(url, target, FileEntry(good.size, "0" * 64), check_sha=True)
    assert list(target.parent.iterdir()) == []

    store_io.download(url, target, FileEntry(good.size, "0" * 64), check_sha=False)
    assert target.read_bytes() == FILES["cone/cone.obj"]


def test_a_dropped_connection_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def flaky(url: str, **kw: object) -> io.BytesIO:
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.URLError(ConnectionResetError())
        return io.BytesIO(b"ok")

    monkeypatch.setattr(store_io.urllib.request, "urlopen", flaky)
    assert store_io.read_text("https://h/x") == "ok"
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("code", "raised"),
    [
        (404, store_io.NotInStore),
        (403, store_io.NotInStore),
        (400, urllib.error.HTTPError),
    ],
)
def test_a_client_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, code: int, raised: type[Exception]
) -> None:
    calls = []

    def refuse(url: str, **kw: object) -> None:
        calls.append(url)
        raise urllib.error.HTTPError(url, code, "no", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(store_io.urllib.request, "urlopen", refuse)
    with pytest.raises(raised):
        store_io.read_text("https://h/x")
    assert len(calls) == 1


def test_a_server_error_gives_up_after_the_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fail(url: str, **kw: object) -> None:
        calls.append(url)
        raise urllib.error.HTTPError(url, 503, "busy", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(store_io.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.HTTPError):
        store_io.read_text("https://h/x")
    assert len(calls) == store_io.RETRIES


# ---- the cache ----


def test_opening_fetches_only_the_manifest(tmp_path: Path, store: str) -> None:
    cache = _cache(tmp_path, store)
    cached = cache.open(PACK)
    assert (cache.top / "CACHEDIR.TAG").is_file()
    assert cached.folder == cache.version_dir(PACK) / "files"
    assert list(cached.folder.iterdir()) == []
    assert "stores" in cached.folder.parts  # not the default store


def test_a_referenced_file_comes_with_its_companions(
    tmp_path: Path, store: str
) -> None:
    cached = _cache(tmp_path, store).open(PACK)
    cached.ensure_referenced(
        [cached.folder / "cone/cone.obj", "/elsewhere/x"], folders=False
    )
    assert sorted(p for p, _ in pack_files(cached.folder)) == [
        "cone/cone.mtl",
        "cone/cone.obj",
    ]


def test_a_referenced_folder_waits_for_the_second_pass(
    tmp_path: Path, store: str
) -> None:
    cached = _cache(tmp_path, store).open(PACK)
    cached.ensure_referenced([str(cached.folder / "cone")], folders=False)
    assert (cached.folder / "cone").is_dir()
    assert list(pack_files(cached.folder)) == []
    cached.ensure_referenced([str(cached.folder / "cone")], folders=True)
    assert len(list(pack_files(cached.folder))) == 3


def test_offline_uses_what_is_cached_and_says_what_is_not(
    tmp_path: Path, store: str
) -> None:
    online = _cache(tmp_path, store).open(PACK)
    online.ensure(["cone/cone.obj"])
    offline = _cache(tmp_path, store, offline=True).open(PACK)
    assert offline.ensure(["cone/cone.obj"]) == []
    with pytest.raises(FileNotFoundError, match="rutter-data pull hardware/v1"):
        offline.ensure(["well/well.step"])
    with pytest.raises(FileNotFoundError, match="RUTTER_OFFLINE"):
        _cache(tmp_path, store, offline=True).open(PackRef("hardware", "v2"))


def _version(label: str, size: int, days_ago: float) -> CachedVersion:
    return CachedVersion(Path(label), label, size, NOW - days_ago * DAY_S)


def test_versions_unused_past_the_age_limit_go() -> None:
    old, recent = _version("a/v1", 1, 61), _version("a/v2", 1, 59)
    assert select_removals(
        [old, recent], now=NOW, max_bytes=10, max_age_s=60 * DAY_S
    ) == [old]


def test_over_the_size_limit_the_least_recently_used_go_first() -> None:
    a, b, c = _version("a/v1", 4, 10), _version("b/v1", 4, 5), _version("c/v1", 4, 2)
    assert select_removals([c, a, b], now=NOW, max_bytes=8, max_age_s=60 * DAY_S) == [a]
    assert select_removals([c, a, b], now=NOW, max_bytes=3, max_age_s=60 * DAY_S) == [
        a,
        b,
        c,
    ]


def test_nothing_used_in_the_last_day_is_removed() -> None:
    busy = [_version("a/v1", 8, 0.5), _version("b/v1", 8, 0.1)]
    assert select_removals(busy, now=NOW, max_bytes=1, max_age_s=0) == []


def _age(vdir: Path, days: float) -> None:
    t = time.time() - days * DAY_S
    os.utime(vdir / ".used", (t, t))


def test_a_sweep_removes_stale_versions_and_leftovers(
    tmp_path: Path, store: str
) -> None:
    cache = _cache(tmp_path, store)
    cached = cache.open(PACK)
    cached.ensure_all()
    _age(cached.vdir, 61)
    old_layout = cache.top / "v0"
    old_layout.mkdir()
    stale_part = cached.folder / ".x.part"
    stale_part.write_bytes(b"")
    os.utime(stale_part, (time.time() - 2 * DAY_S,) * 2)

    removed = cache.sweep()
    assert [v.label.split(" ")[0] for v in removed] == ["hardware/v1"]
    assert not cached.vdir.exists()
    assert not old_layout.exists()
    assert not (cache.root / "trash").exists() or not any(
        (cache.root / "trash").iterdir()
    )


def test_an_untagged_folder_keeps_other_layouts(tmp_path: Path, store: str) -> None:
    cache = _cache(tmp_path, store)
    (cache.top / "v2").mkdir(parents=True)
    cache.sweep()
    assert (cache.top / "v2").is_dir()


def test_one_sweep_a_day_however_many_processes_ask(tmp_path: Path, store: str) -> None:
    cache = _cache(tmp_path, store)
    cache.root.mkdir(parents=True)
    (cache.root / ".sweep-2020-01-01").touch()
    calls = []
    cache.sweep = lambda now=None: calls.append(now) or []  # type: ignore[method-assign]
    for _ in range(3):
        cache.sweep_daily(NOW)
    cache.sweep_daily(NOW + DAY_S)
    assert calls == [NOW, NOW + DAY_S]
    assert [p.name for p in cache.root.glob(".sweep-*")] == [
        f".sweep-{time.strftime('%Y-%m-%d', time.localtime(NOW + DAY_S))}"
    ]


def test_clear_removes_one_version_one_pack_or_everything(
    tmp_path: Path, store: str
) -> None:
    src = _write(tmp_path / "src2", {"x.obj": b"x"})
    for version in ("v2", "v3"):
        store_io.publish(src, build_manifest(src, PackRef("hardware", version)), store)
    cache = _cache(tmp_path, store)
    for version in ("v1", "v2", "v3"):
        cache.open(PackRef("hardware", version))

    assert len(cache.clear("hardware/v2")) == 1
    assert len(cache.versions()) == 2
    assert len(cache.clear("hardware")) == 2
    assert cache.versions() == []


def test_the_environment_outranks_the_machine_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine = {"pack_store": "file:///m/", "cache": {"max_gb": 2, "max_age_days": 7}}
    settings = PackSettings.with_machine_file(machine)
    assert (
        settings.pack_store,
        settings.cache_max_gb,
        settings.cache_max_age_days,
    ) == (
        "file:///m/",
        2,
        7,
    )
    monkeypatch.setenv("RUTTER_CACHE_MAX_GB", "5")
    settings = PackSettings.with_machine_file(machine)
    assert (settings.cache_max_gb, settings.cache_max_age_days) == (5, 7)
    assert PackSettings.with_machine_file({}).cache_max_gb == 5


def test_the_cache_layout_is_versioned(tmp_path: Path, store: str) -> None:
    cache = _cache(tmp_path, store)
    assert cache.root == cache.top / LAYOUT
