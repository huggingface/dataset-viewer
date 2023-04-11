# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import asyncio
import itertools
import logging
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, TypedDict, Union

import aiohttp
import pyarrow as pa
from aiolimiter import AsyncLimiter
from datasets import (
    Dataset,
    DownloadConfig,
    IterableDataset,
    get_dataset_config_info,
    load_dataset,
)
from libcommon.constants import PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response
from libcommon.storage import StrPath

from worker.config import AppConfig, OptinOutUrlsScanConfig
from worker.job_runner import CompleteJobResult, ConfigNotFoundError, JobRunnerError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner
from worker.job_runners.split.first_rows_from_streaming import (
    Row,
    SplitFirstRowsResponse,
    retry,
)

OptinOutUrlsScanJobRunnerErrorCode = Literal[
    "InfoError",
    "TooManyColumnsError",
    "PreviousStepStatusError",
    "PreviousStepFormatError",
    "StreamingRowsError",
    "NormalRowsError",
    "MissingSpawningTokenError",
]


class OptinOutUrlsScanJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: OptinOutUrlsScanJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class InfoError(OptinOutUrlsScanJobRunnerError):
    """Raised when the info could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "InfoError", cause, True)


class TooManyColumnsError(OptinOutUrlsScanJobRunnerError):
    """Raised when the dataset exceeded the max number of columns."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooManyColumnsError", cause, True)


class PreviousStepStatusError(OptinOutUrlsScanJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(OptinOutUrlsScanJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class StreamingRowsError(OptinOutUrlsScanJobRunnerError):
    """Raised when the rows could not be fetched in streaming mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "StreamingRowsError", cause, True)


class NormalRowsError(OptinOutUrlsScanJobRunnerError):
    """Raised when the rows could not be fetched in normal mode."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "NormalRowsError", cause, True)


class MissingSpawningTokenError(OptinOutUrlsScanJobRunnerError):
    """Raised when the spawning.ai token is not set."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "MissingSpawningTokenError", cause, False)


class OptinOutUrlsScanResponse(TypedDict):
    urls_columns: List[str]
    opt_in_urls: List[str]
    opt_out_urls: List[str]
    num_scanned_rows: int


class OptInUrl(TypedDict):
    url: str
    row_idx: int
    column_name: str


class OptOutUrl(TypedDict):
    url: str
    row_idx: int
    column_name: str


@retry
def get_rows(
    dataset: str,
    config: str,
    split: str,
    streaming: bool,
    rows_max_number: int,
    use_auth_token: Union[bool, str, None] = False,
    column_names: Optional[List[str]] = None,
) -> List[Row]:
    download_config = DownloadConfig(delete_extracted=True)
    ds = load_dataset(
        dataset,
        name=config,
        split=split,
        streaming=streaming,
        use_auth_token=use_auth_token,
        download_config=download_config,
    )
    if streaming:
        if not isinstance(ds, IterableDataset):
            raise TypeError("load_dataset should return an IterableDataset in streaming mode")
    elif not isinstance(ds, Dataset):
        raise TypeError("load_dataset should return a Dataset in normal mode")
    if column_names:
        ds = ds.select_columns(column_names)
    rows_plus_one = list(itertools.islice(ds, rows_max_number + 1))
    # ^^ to be able to detect if a split has exactly ROWS_MAX_NUMBER rows
    if len(rows_plus_one) <= rows_max_number:
        logging.debug(f"all the rows in the split have been fetched ({len(rows_plus_one)})")
    else:
        logging.debug(f"the rows in the split have been truncated ({rows_max_number} rows)")
    return rows_plus_one[:rows_max_number]


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


async def opt_in_out_scan_urls(
    urls: List[str],
    urls_number_per_batch: int,
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
        for offset in range(0, len(urls), urls_number_per_batch):
            offsets.append(offset)
            tasks.append(
                asyncio.create_task(
                    opt_in_out_task(urls[offset : offset + urls_number_per_batch], session, semaphore, limiter)
                )
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


def compute_opt_in_out_urls_scan_response(
    dataset: str,
    config: str,
    split: str,
    hf_token: Optional[str],
    rows_max_number: int,
    columns_max_number: int,
    urls_number_per_batch: int,
    spawning_token: Optional[str],
    max_concurrent_requests_number: int,
    max_requests_per_second: int,
) -> OptinOutUrlsScanResponse:
    logging.info(f"get opt-in-out-urls-scan for dataset={dataset} config={config} split={split}")

    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False
    if not spawning_token:
        raise MissingSpawningTokenError("OPT_IN_OUT_URLS_SCAN_SPAWNING_TOKEN is not set")

    # get the first rows from previous job
    try:
        upstream_response = get_response(
            kind="split-first-rows-from-streaming", dataset=dataset, config=config, split=split
        )
        upstream_response_content = SplitFirstRowsResponse(
            dataset=dataset,
            config=config,
            split=split,
            features=upstream_response["content"]["features"],
            rows=upstream_response["content"]["rows"],
        )
        features = upstream_response_content["features"]
        first_rows = upstream_response_content["rows"]
    except DoesNotExist as e:
        raise ConfigNotFoundError(f"The config '{config}' does not exist for the dataset.'", e) from e
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    if upstream_response["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(
            f"Previous step gave an error: {upstream_response['http_status']}. This job should not have been created."
        )

    if features and len(features) > columns_max_number:
        raise TooManyColumnsError(
            f"The number of columns ({len(features)}) exceeds the maximum supported number of columns"
            f" ({columns_max_number}). This is a current limitation of the datasets viewer. You can reduce the number"
            " of columns if you want the viewer to work."
        )

    # get the info
    try:
        info = get_dataset_config_info(
            path=dataset,
            config_name=config,
            use_auth_token=use_auth_token,
        )
    except Exception as err:
        raise InfoError(
            f"The info cannot be fetched for the config '{config}' of the dataset.",
            cause=err,
        ) from err

    # look for URLs columns using the first rows
    string_type_dict = {"dtype": "string", "_type": "Value"}
    string_columns = [column for column in features if features[column] == string_type_dict]
    urls_columns = []
    for string_column in string_columns:
        urls_count = sum(
            1
            for row in first_rows
            if isinstance(row["row"].get(string_column), str)
            and row["row"][string_column].startswith("https://")
            or row["row"][string_column].startswith("http://")
        )
        if urls_count and urls_count / len(first_rows) > 0.5:
            urls_columns.append(string_column)

    if not urls_columns:
        return OptinOutUrlsScanResponse(
            has_urls_columns=False,
            urls_columns=[],
            opt_in_urls=[],
            opt_out_urls=[],
            opt_in_urls_indices=[],
            opt_out_urls_indices=[],
            num_scanned_rows=0,
        )

    # get the rows
    try:
        rows = get_rows(
            dataset=dataset,
            config=config,
            split=split,
            streaming=True,
            rows_max_number=rows_max_number,
            use_auth_token=use_auth_token,
            column_names=urls_columns,
        )
    except Exception as err:
        MAX_SIZE_FALLBACK = 100_000_000
        if info.size_in_bytes is None or info.size_in_bytes > MAX_SIZE_FALLBACK:
            raise StreamingRowsError(
                "Cannot load the dataset split (in streaming mode) to extract the first rows.",
                cause=err,
            ) from err
        try:
            rows = get_rows(
                dataset=dataset,
                config=config,
                split=split,
                streaming=False,
                rows_max_number=rows_max_number,
                use_auth_token=use_auth_token,
            )
        except Exception as err:
            raise NormalRowsError(
                "Cannot load the dataset split (in normal download mode) to extract the first rows.",
                cause=err,
            ) from err

    # get the urls
    num_scanned_rows = len(rows)
    urls = [row[urls_column] for row in rows for urls_column in urls_columns]

    # scan the urls
    opt_in_urls_indices, opt_out_urls_indices = asyncio.run(
        opt_in_out_scan_urls(
            urls,
            urls_number_per_batch=urls_number_per_batch,
            spawning_token=spawning_token,
            max_concurrent_requests_number=max_concurrent_requests_number,
            max_requests_per_second=max_requests_per_second,
        )
    )
    opt_in_urls = [
        OptInUrl(url=url, row_idx=url_idx // len(urls_columns), column_name=urls_columns[url_idx % len(urls_columns)])
        for url_idx, url in enumerate(opt_in_urls_indices)
    ]
    opt_out_urls = [
        OptOutUrl(url=url, row_idx=url_idx // len(urls_columns), column_name=urls_columns[url_idx % len(urls_columns)])
        for url_idx, url in enumerate(opt_out_urls_indices)
    ]

    # return scan result
    return OptinOutUrlsScanResponse(
        has_urls_columns=True,
        urls_columns=urls_columns,
        opt_in_urls=opt_in_urls,
        opt_out_urls=opt_out_urls,
        opt_in_urls_indices=opt_in_urls_indices,
        opt_out_urls_indices=opt_out_urls_indices,
        num_scanned_rows=num_scanned_rows,
    )


class OptinOutUrlsScanJobRunner(DatasetsBasedJobRunner):
    urls_scan_config: OptinOutUrlsScanConfig

    @staticmethod
    def get_job_type() -> str:
        return "opt-in-out-urls-scan"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        urls_scan_config: OptinOutUrlsScanConfig,
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
            compute_opt_in_out_urls_scan_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                hf_token=self.common_config.hf_token,
                rows_max_number=self.urls_scan_config.rows_max_number,
                columns_max_number=self.urls_scan_config.columns_max_number,
                urls_number_per_batch=self.urls_scan_config.urls_number_per_batch,
                spawning_token=self.urls_scan_config.spawning_token,
                max_concurrent_requests_number=self.urls_scan_config.max_concurrent_requests_number,
                max_requests_per_second=self.urls_scan_config.max_requests_per_second,
            )
        )

    def get_new_splits(self, _: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by compute."""
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return {SplitFullName(dataset=self.dataset, config=self.config, split=self.split)}
