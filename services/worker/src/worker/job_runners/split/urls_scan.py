# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import asyncio
import logging
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, TypedDict

import aiohttp
import pyarrow as pa
from aiolimiter import AsyncLimiter
from datasets import Features, Value
from fsspec import AbstractFileSystem  # type: ignore
from hffs.fs import HfFileSystem
from libcommon.constants import (
    PARQUET_REVISION,
    PROCESSING_STEP_SPLIT_URLS_SCAN_VERSION,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response
from libcommon.storage import StrPath
from pyarrow.parquet import ParquetFile
from tqdm.contrib.concurrent import thread_map

from worker.config import AppConfig, UrlsScanConfig
from worker.job_runner import CompleteJobResult, ConfigNotFoundError, JobRunnerError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner

UrlsScanJobRunnerErrorCode = Literal[
    "TooManyColumnsError",
    "PreviousStepStatusError",
    "PreviousStepFormatError",
    "ParquetResponseEmptyError",
    "FileSystemError",
]


class UrlsScanJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: UrlsScanJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class TooManyColumnsError(UrlsScanJobRunnerError):
    """Raised when the dataset exceeded the max number of columns."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooManyColumnsError", cause, True)


class PreviousStepStatusError(UrlsScanJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(UrlsScanJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class ParquetResponseEmptyError(UrlsScanJobRunnerError):
    """Raised when no parquet files were found for split."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ParquetResponseEmptyError", cause, False)


class FileSystemError(UrlsScanJobRunnerError):
    """Raised when an error happen reading from File System."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "FileSystemError", cause, False)


class UrlsScanResponse(TypedDict):
    urls_columns: List[str]
    opt_in_urls: List[str]
    opt_out_urls: List[str]
    num_scanned_rows: int
    total_num_rows: int


class OptInUrl(TypedDict):
    url: str
    row_idx: int
    column_name: str


class OptOutUrl(TypedDict):
    url: str
    row_idx: int
    column_name: str


def get_parquet_fs(dataset: str, hf_token: Optional[str]) -> AbstractFileSystem:
    """Get the parquet filesystem for a dataset.
    The parquet files are stored in a separate branch of the dataset repository (see PARQUET_REVISION)
    Args:
        dataset (str): The dataset name.
        hf_token (Optional[str]): The token to access the filesystem.
    Returns:
        HfFileSystem: The parquet filesystem.
    """
    return HfFileSystem(dataset, repo_type="dataset", revision=PARQUET_REVISION, token=hf_token)


async def check_spawning(image_urls: List[str], session, semaphore, limiter) -> dict:
    url = f"https://opts-api.spawningaiapi.com/api/v2/query/urls"
    if not image_urls:
        return {"urls": []}
    elif len(image_urls) == 1:
        image_urls = image_urls + [""]  # the API requires >1 urls
    async with semaphore:
        async with limiter:
            async with session.post(url=url, data="\n".join(image_urls)) as resp:
                spawning_response = await resp.json()
                return spawning_response


async def opt_in_out_task(image_urls: List[str], session, semaphore, limiter) -> tuple:
    spawning_response = await check_spawning(image_urls, session, semaphore, limiter)
    opt_in_urls_indices = [i for i in range(len(image_urls)) if spawning_response["urls"][i]["optIn"]]
    opt_out_urls_indices = [i for i in range(len(image_urls)) if spawning_response["urls"][i]["optOut"]]
    return opt_in_urls_indices, opt_out_urls_indices


async def scan_urls(
    urls: List[str],
    batch_size: int,
    spawning_token: str,
    max_concurrent_requests_number: int,
    max_requests_per_second: int,
) -> Tuple[List[int], List[int]]:
    offsets = []
    tasks = []
    semaphore = asyncio.Semaphore(value=max_concurrent_requests_number)
    limiter = AsyncLimiter(max_requests_per_second, time_period=1)

    headers = {"Authorization": f"API {spawning_token}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        for offset in range(0, len(urls), batch_size):
            offsets.append(offset)
            tasks.append(
                asyncio.create_task(opt_in_out_task(urls[offset : offset + batch_size], session, semaphore, limiter))
            )
        await asyncio.wait(tasks)

    opt_in_urls_indices = []
    opt_out_urls_indices = []
    for offset, task in zip(offsets, tasks):
        batch_opt_in_urls_indices, batch_opt_out_urls_indices = task.result()
        for batch_opt_in_urls_idx in batch_opt_in_urls_indices:
            opt_in_urls_indices.append(offset + batch_opt_in_urls_idx)
        for batch_opt_out_urls_idx in batch_opt_out_urls_indices:
            opt_out_urls_indices.append(offset + batch_opt_out_urls_idx)

    return opt_in_urls_indices, opt_out_urls_indices


def compute_first_rows_response(
    dataset: str,
    config: str,
    split: str,
    hf_token: Optional[str],
    rows_max_number: int,
    columns_max_number: int,
) -> UrlsScanResponse:
    logging.info(f"get first-rows for dataset={dataset} config={config} split={split}")

    # first ensure the tuple (dataset, config, split) exists on the Hub
    try:
        upstream_response = get_response(kind="config-parquet", dataset=dataset, config=config)
        if upstream_response["http_status"] != HTTPStatus.OK:
            raise PreviousStepStatusError(
                f"Previous step gave an error: {upstream_response['http_status']}. This job should not have been"
                " created."
            )

        parquet_files_content = upstream_response["content"]["parquet_files"]
        sources = sorted(
            f"{config}/{parquet_file['filename']}"
            for parquet_file in parquet_files_content
            if parquet_file["split"] == split and parquet_file["config"] == config
        )
        if not sources:
            raise ParquetResponseEmptyError("No parquet files found.")
    except DoesNotExist:
        raise ConfigNotFoundError(f"The config '{config}' does not exist for the dataset.'")
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.") from e

    logging.debug(f"Found {len(sources)} parquet files for {dataset=}, {config=}, {split=}: {sources}")

    fs = get_parquet_fs(dataset=dataset, hf_token=hf_token)
    desc = f"{dataset}/{config}/{split}"
    try:
        parquet_files: List[ParquetFile] = thread_map(
            partial(ParquetFile, filesystem=fs), sources, desc=desc, unit="pq", disable=True
        )
    except Exception as e:
        raise FileSystemError(f"Could not read the parquet files: {e}") from e
    total_num_rows = sum(parquet_file.metadata.num_rows for parquet_file in parquet_files)

    # get the features
    features = Features.from_arrow_schema(parquet_files[0].schema.to_arrow_schema())

    if features and len(features) > columns_max_number:
        raise TooManyColumnsError(
            f"The number of columns ({len(features)}) exceeds the maximum supported number of columns"
            f" ({columns_max_number}). This is a current limitation of the datasets viewer. You can reduce the number"
            " of columns if you want the viewer to work."
        )

    # get the first rows to look for URLs columns
    string_columns = [column for column in features if features[column] == Value("string")]
    first_rows = parquet_files[0].read_row_group(i=0, columns=string_columns).slice(0, 100).to_pydict()
    urls_columns = []
    for string_column in string_columns:
        urls_count = sum(1 for string in first_rows[string_column] if string.startswith("https://") or string.startswith("http://"))
        if urls_count and urls_count / len(first_rows[string_column]) > .5:
            urls_columns.append(string_column)

    if not urls_columns:
        return UrlsScanResponse(
            urls_columns=[], opt_in_urls=[], opt_out_urls=[], num_scanned_rows=0, total_num_rows=total_num_rows
        )

    # get the rows
    num_scanned_rows = 0
    last_row_group_id = 0
    row_group_readers = []
    for parquet_file in parquet_files:
        for group_id in range(parquet_file.metadata.num_row_groups):
            last_row_group_id = group_id
            row_group_readers.append(partial(parquet_file.read_row_group, i=group_id, columns=urls_columns))
            if num_scanned_rows + parquet_file.metadata.row_group(group_id).num_rows >= rows_max_number:
                num_scanned_rows = rows_max_number
                break
            else:
                num_scanned_rows += parquet_file.metadata.row_group(group_id).num_rows
        else:
            continue
        break

    if len(row_group_readers) == 0:
        raise ParquetResponseEmptyError("No parquet files found.")

    # get the urls
    pa_table = pa.concat_tables([row_group_readers[i]() for i in range(0, last_row_group_id + 1)]).slice(
        0, num_scanned_rows
    )
    urls = [url for urls_array in pa_table for url in urls_array.to_pylist()]
    column_names = list(pa_table.column_names)

    # scan the urls
    opt_in_urls_indices, opt_out_urls_indices = asyncio.run(scan_urls(urls))
    opt_in_urls = [
        OptInUrl(url=url, row_idx=url_idx % num_scanned_rows, column_name=column_names[url_idx // num_scanned_rows])
        for url_idx, url in enumerate(opt_in_urls_indices)
    ]
    opt_out_urls = [
        OptOutUrl(url=url, row_idx=url_idx % num_scanned_rows, column_name=column_names[url_idx // num_scanned_rows])
        for url_idx, url in enumerate(opt_out_urls_indices)
    ]

    # return scan result
    return UrlsScanResponse(
        urls_columns=urls_columns,
        opt_in_urls=opt_in_urls,
        opt_out_urls=opt_out_urls,
        num_scanned_rows=num_scanned_rows,
        total_num_rows=total_num_rows,
    )


class UrlsScanJobRunner(DatasetsBasedJobRunner):
    urls_scan_config: UrlsScanConfig

    @staticmethod
    def get_job_type() -> str:
        return "urls-scan"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_URLS_SCAN_VERSION

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        urls_scan_config: UrlsScanConfig,
        hf_datasets_cache: Path,
        assets_directory: StrPath,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.urls_scan_config = urls_scan_config
        self.assets_directory = assets_directory
        self.assets_base_url = app_config.assets.base_url

    def compute(self) -> CompleteJobResult:
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return CompleteJobResult(
            compute_first_rows_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                hf_token=self.common_config.hf_token,
                rows_max_number=self.urls_scan_config.rows_max_number,
                columns_max_number=self.urls_scan_config.columns_max_number,
            )
        )

    def get_new_splits(self, _: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by compute."""
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return {SplitFullName(dataset=self.dataset, config=self.config, split=self.split)}
