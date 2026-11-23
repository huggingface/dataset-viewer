import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import anyio
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from duckdb import OutOfMemoryException
from libcommon.parquet_utils import ParquetFileMetadataItem
from pytest import TempPathFactory

from libapi import duckdb
from libapi.duckdb import check_available_disk_space
from libapi.exceptions import ResponseNotReadyError

# TODO(QL): test duckdb indexing


@pytest.mark.parametrize("subpath", [None, "does_not_exist"])
def test_check_available_disk_space(tmp_path_factory: TempPathFactory, subpath: Optional[str]) -> None:
    path = tmp_path_factory.mktemp("test_check_available_disk_space")
    if subpath:
        path = path / subpath
    check_available_disk_space(path=path, required_space=1)


@pytest.mark.anyio
async def test_index_is_built_once_and_never_read_before_complete(tmp_path: Path) -> None:
    parquet_metadata_directory = tmp_path / "parquet-metadata"
    (parquet_metadata_directory / "ds/default/train").mkdir(parents=True)
    pq.write_table(pa.table({"a": [1, 2, 3]}), parquet_metadata_directory / "ds/default/train/0000.parquet")  # type: ignore
    split_parquet_files: list[ParquetFileMetadataItem] = [
        {
            "dataset": "ds",
            "config": "default",
            "split": "train",
            "url": "https://huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "filename": "0000.parquet",
            "size": 1,
            "num_rows": 3,
            "parquet_metadata_subpath": "ds/default/train/0000.parquet",
        }
    ]
    builds = 0

    def build_index_file(_cache_folder: Path, index_folder: Path, *args: Any) -> None:
        nonlocal builds
        _dataset, _revision, _config, _split, repo_file_location, *_ = args
        builds += 1
        index_file = Path(f"{index_folder}/{repo_file_location}")
        tmp_index_file = Path(f"{index_file}.tmp")
        tmp_index_file.write_bytes(b"partial")
        time.sleep(0.6)
        tmp_index_file.write_bytes(b"complete")
        tmp_index_file.replace(index_file)

    async def request() -> Optional[bytes]:
        try:
            index_file_location, _ = await duckdb.get_index_file_location_and_build_if_missing(
                duckdb_index_file_directory=tmp_path / "duckdb-index",
                dataset="ds",
                revision="revision",
                config="default",
                split="train",
                hf_token=None,
                max_split_size_bytes=10**9,
                extensions_directory=None,
                parquet_metadata_directory=parquet_metadata_directory,
                split_parquet_files=split_parquet_files,
                features={"a": {"dtype": "int64", "_type": "Value"}},
            )
        except ResponseNotReadyError:
            return None
        return Path(index_file_location).read_bytes()

    async def requests_every_100ms() -> list[Optional[bytes]]:
        tasks = []
        for _ in range(10):
            tasks.append(asyncio.create_task(request()))
            await asyncio.sleep(0.1)
        return await asyncio.gather(*tasks)

    # requests give up waiting after 0.2s, while the build goes on for 0.6s
    with (
        patch.object(duckdb, "build_index_file", build_index_file),
        patch.object(duckdb, "INDEX_BUILD_WAIT_SECONDS", 0.2),
    ):
        contents = await requests_every_100ms()

    assert builds == 1
    assert set(contents) == {None, b"complete"}


@pytest.mark.anyio
async def test_index_builds_do_not_take_the_default_threads(tmp_path: Path) -> None:
    release = threading.Event()
    limiter = anyio.CapacityLimiter(2)
    with (
        patch.object(duckdb, "build_index_file", lambda *_: release.wait() and False),
        patch.object(duckdb, "_index_build_limiter", limiter),
        patch.object(duckdb, "INDEX_BUILD_STALE_SECONDS", float("inf")),
    ):
        # more builds than the 40 threads of the default anyio limiter
        builds = [
            asyncio.create_task(duckdb._build_index_file_once(str(tmp_path / f"{i}.duckdb"), None)) for i in range(50)
        ]
        await asyncio.sleep(0.2)  # let the builds take their threads
        with anyio.move_on_after(1) as scope:
            await anyio.to_thread.run_sync(lambda: None)
        borrowed = limiter.borrowed_tokens
        release.set()
        await asyncio.gather(*builds)

    assert not scope.cancelled_caught  # a thread was free for the auth check and the queries
    assert borrowed == 2


@pytest.mark.anyio
async def test_stale_index_builds_are_skipped(tmp_path: Path) -> None:
    built: list[str] = []
    fresh, stale = str(tmp_path / "fresh.duckdb"), str(tmp_path / "stale.duckdb")
    # the last request time is shared by the uvicorn workers through the mtime of this file
    Path(fresh + ".requested").touch()
    Path(stale + ".requested").touch()
    old = time.time() - duckdb.INDEX_BUILD_STALE_SECONDS - 1
    os.utime(stale + ".requested", (old, old))

    def build_index_file(name: str) -> bool:
        built.append(name)
        return True

    with patch.object(duckdb, "build_index_file", build_index_file):
        await duckdb._build_index_file_once(fresh, None, "fresh")
        await duckdb._build_index_file_once(stale, None, "stale")

    assert built == ["fresh"]


@pytest.mark.anyio
async def test_shared_index_cache(tmp_path: Path) -> None:
    built: list[str] = []
    local, shared = tmp_path / "local", tmp_path / "shared"

    def build_index_file(name: str) -> bool:
        built.append(name)
        Path(local / name / "index.duckdb").write_bytes(f"index of {name}".encode())
        return True

    async def get_index(name: str) -> bytes:
        index = local / name / "index.duckdb"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.with_suffix(".duckdb.requested").touch()
        await duckdb._build_index_file_once(str(index), str(shared / name / "index.duckdb"), name)
        return index.read_bytes()

    # another pod already built "a": it is copied from the shared cache, not built
    (shared / "a").mkdir(parents=True)
    (shared / "a" / "index.duckdb").write_bytes(b"index of a")
    with patch.object(duckdb, "build_index_file", build_index_file):
        assert await get_index("a") == b"index of a"
        # "b" is built here, then copied to the shared cache for the other pods, in the background
        assert await get_index("b") == b"index of b"
        await asyncio.gather(*duckdb._all_tasks)

    assert built == ["b"]
    assert (shared / "b" / "index.duckdb").read_bytes() == b"index of b"
    assert not list(shared.rglob("*.tmp"))


@pytest.mark.anyio
async def test_incomplete_index_is_not_shared(tmp_path: Path) -> None:
    index, shared = tmp_path / "local" / "index.duckdb", tmp_path / "shared" / "index.duckdb"
    index.parent.mkdir(parents=True)
    index.with_suffix(".duckdb.requested").touch()

    def build_index_file(_: str) -> bool:
        index.write_bytes(b"index without the transformed columns")
        return False

    with patch.object(duckdb, "build_index_file", build_index_file):
        await duckdb._build_index_file_once(str(index), str(shared), "ds")
        await asyncio.gather(*duckdb._all_tasks)

    assert index.is_file()
    assert not shared.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("error,builds", [(OutOfMemoryException("Out of Memory Error"), 1), (RuntimeError("Hub"), 2)])
async def test_failed_index_builds_are_not_retried_right_away(tmp_path: Path, error: Exception, builds: int) -> None:
    parquet_metadata_directory = tmp_path / "parquet-metadata"
    (parquet_metadata_directory / "ds/default/train").mkdir(parents=True)
    pq.write_table(pa.table({"a": [1, 2, 3]}), parquet_metadata_directory / "ds/default/train/0000.parquet")  # type: ignore
    split_parquet_files: list[ParquetFileMetadataItem] = [
        {
            "dataset": "ds",
            "config": "default",
            "split": "train",
            "url": "https://huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "filename": "0000.parquet",
            "size": 1,
            "num_rows": 3,
            "parquet_metadata_subpath": "ds/default/train/0000.parquet",
        }
    ]
    started = 0

    def build_index_file(*_: Any) -> bool:
        nonlocal started
        started += 1
        raise error

    async def request() -> None:
        with pytest.raises(type(error)):
            await duckdb.get_index_file_location_and_build_if_missing(
                duckdb_index_file_directory=tmp_path / "duckdb-index",
                dataset="ds",
                revision="revision",
                config="default",
                split="train",
                hf_token=None,
                max_split_size_bytes=10**9,
                extensions_directory=None,
                parquet_metadata_directory=parquet_metadata_directory,
                split_parquet_files=split_parquet_files,
                features={"a": {"dtype": "int64", "_type": "Value"}},
            )
        await asyncio.sleep(0)  # let the build task end

    with patch.object(duckdb, "build_index_file", build_index_file):
        await request()
        await request()
        assert started == builds
        with patch.object(duckdb, "FAILED_INDEX_BUILD_RETRY_SECONDS", 0):
            await request()

    assert started == builds + 1


@pytest.mark.anyio
async def test_served_shared_indexes_are_marked_as_used(tmp_path: Path) -> None:
    local, shared = tmp_path / "local", tmp_path / "shared"
    parquet_metadata_directory = tmp_path / "parquet-metadata"
    (parquet_metadata_directory / "ds/default/train").mkdir(parents=True)
    pq.write_table(pa.table({"a": [1]}), parquet_metadata_directory / "ds/default/train/0000.parquet")  # type: ignore
    split_parquet_files: list[ParquetFileMetadataItem] = [
        {
            "dataset": "ds",
            "config": "default",
            "split": "train",
            "url": "https://huggingface.co/datasets/ds/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "filename": "0000.parquet",
            "size": 1,
            "num_rows": 1,
            "parquet_metadata_subpath": "ds/default/train/0000.parquet",
        }
    ]

    def build_index_file(_cache_folder: Path, index_folder: Path, *args: Any) -> bool:
        _dataset, _revision, _config, _split, repo_file_location, *_ = args
        Path(f"{index_folder}/{repo_file_location}").write_bytes(b"index")
        return True

    async def request() -> None:
        await duckdb.get_index_file_location_and_build_if_missing(
            duckdb_index_file_directory=local,
            dataset="ds",
            revision="revision",
            config="default",
            split="train",
            hf_token=None,
            max_split_size_bytes=10**9,
            extensions_directory=None,
            parquet_metadata_directory=parquet_metadata_directory,
            split_parquet_files=split_parquet_files,
            features={"a": {"dtype": "int64", "_type": "Value"}},
            shared_index_directory=shared,
        )
        await asyncio.gather(*duckdb._all_tasks)

    with patch.object(duckdb, "build_index_file", build_index_file):
        await request()  # built here, then uploaded to the shared cache
        [shared_index] = shared.rglob("index.duckdb")
        marker = Path(str(shared_index) + duckdb.SHARED_INDEX_USED_SUFFIX)
        # the first marker races with the upload, in the background: serve the shared index again
        duckdb._shared_index_marked_at.clear()
        await request()
        first_mark = marker.read_text()
        await request()  # marked less than SHARED_INDEX_USED_INTERVAL_SECONDS ago: not rewritten
        assert marker.read_text() == first_mark


@pytest.mark.anyio
async def test_markers_do_not_delay_the_copies_from_the_shared_cache(tmp_path: Path) -> None:
    release = threading.Event()
    with patch.object(duckdb, "_mark_shared_index_used", lambda _: release.wait()):
        # more markers than their limiter takes, all blocked
        markers = [
            asyncio.create_task(
                anyio.to_thread.run_sync(
                    duckdb._mark_shared_index_used, str(i), limiter=duckdb._shared_index_marker_limiter
                )
            )
            for i in range(10)
        ]
        await asyncio.sleep(0.2)
        with anyio.move_on_after(1) as scope:
            await anyio.to_thread.run_sync(lambda: None, limiter=duckdb._shared_cache_limiter)
        release.set()
        await asyncio.gather(*markers)

    assert not scope.cancelled_caught  # a copy got a slot while the markers waited
