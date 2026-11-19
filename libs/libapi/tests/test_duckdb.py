import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import anyio
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
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
        patch.object(duckdb, "build_index_file", lambda *_: release.wait()),
        patch.object(duckdb, "_index_build_limiter", limiter),
    ):
        # more builds than the 40 threads of the default anyio limiter
        builds = [asyncio.create_task(duckdb._build_index_file_once(str(tmp_path / f"{i}.duckdb"))) for i in range(50)]
        await asyncio.sleep(0.2)  # let the builds take their threads
        with anyio.move_on_after(1) as scope:
            await anyio.to_thread.run_sync(lambda: None)
        borrowed = limiter.borrowed_tokens
        release.set()
        await asyncio.gather(*builds)

    assert not scope.cancelled_caught  # a thread was free for the auth check and the queries
    assert borrowed == 2
