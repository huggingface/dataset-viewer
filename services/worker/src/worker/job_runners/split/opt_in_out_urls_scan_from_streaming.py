# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from asyncio import Semaphore, create_task, run, wait
from http import HTTPStatus
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, Union

from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from datasets import get_dataset_config_info
from libcommon.constants import PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo
from libcommon.simple_cache import SplitFullName

from worker.config import AppConfig, OptInOutUrlsScanConfig
from worker.job_runner import (
    CompleteJobResult,
    JobRunnerError,
    get_previous_step_or_raise,
)
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner
from worker.utils import (
    OptInOutUrlsScanResponse,
    OptUrl,
    SplitFirstRowsResponse,
    get_rows_or_raise,
)

SplitOptInOutUrlsScanJobRunnerErrorCode = Literal[
    "InfoError",
    "TooManyColumnsError",
    "PreviousStepStatusError",
    "PreviousStepFormatError",
    "MissingSpawningTokenError",
    "ExternalServerError",
]


class SplitOptInOutUrlsScanJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: SplitOptInOutUrlsScanJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class InfoError(SplitOptInOutUrlsScanJobRunnerError):
    """Raised when the info could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "InfoError", cause, True)


class TooManyColumnsError(SplitOptInOutUrlsScanJobRunnerError):
    """Raised when the dataset exceeded the max number of columns."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "TooManyColumnsError", cause, True)


class PreviousStepStatusError(SplitOptInOutUrlsScanJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(SplitOptInOutUrlsScanJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class MissingSpawningTokenError(SplitOptInOutUrlsScanJobRunnerError):
    """Raised when the spawning.ai token is not set."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "MissingSpawningTokenError", cause, False)


class ExternalServerError(SplitOptInOutUrlsScanJobRunnerError):
    """Raised when the spawning.ai server is not responding."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ExternalServerError", cause, False)


async def check_spawning(
    image_urls: List[str], session: ClientSession, semaphore: Semaphore, limiter: AsyncLimiter, spawning_url: str
) -> Any:
    if not image_urls:
        return {"urls": []}
    elif len(image_urls) == 1:
        image_urls = image_urls + [""]  # the API requires >1 urls
    async with semaphore:
        async with limiter:
            async with session.post(url=spawning_url, data="\n".join(image_urls)) as resp:
                spawning_response = await resp.json()
                return spawning_response


async def opt_in_out_task(
    image_urls: List[str], session: ClientSession, semaphore: Semaphore, limiter: AsyncLimiter, spawning_url: str
) -> Tuple[List[Any], List[Any]]:
    try:
        spawning_response = await check_spawning(image_urls, session, semaphore, limiter, spawning_url)
    except Exception:
        raise ExternalServerError(message=f"Error when trying to connect to {spawning_url}")
    if "urls" not in spawning_response:
        raise ExternalServerError(message=f"Error when trying to connect to {spawning_url}: '{spawning_response}'")
    opt_in_urls_indices = [i for i in range(len(image_urls)) if spawning_response["urls"][i]["optIn"]]
    opt_out_urls_indices = [i for i in range(len(image_urls)) if spawning_response["urls"][i]["optOut"]]
    return opt_in_urls_indices, opt_out_urls_indices


async def opt_in_out_scan_urls(
    urls: List[str],
    urls_number_per_batch: int,
    spawning_token: str,
    max_concurrent_requests_number: int,
    max_requests_per_second: int,
    spawning_url: str,
) -> Tuple[List[int], List[int]]:
    offsets = []
    tasks = []
    semaphore = Semaphore(value=max_concurrent_requests_number)
    limiter = AsyncLimiter(max_requests_per_second, time_period=1)

    headers = {"Authorization": f"API {spawning_token}"}
    async with ClientSession(headers=headers) as session:
        for offset in range(0, len(urls), urls_number_per_batch):
            offsets.append(offset)
            limit = offset + urls_number_per_batch
            tasks.append(
                create_task(opt_in_out_task(urls[offset:limit], session, semaphore, limiter, spawning_url))
            )  # noqa: E203
        await wait(tasks)

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
    spawning_url: str,
) -> OptInOutUrlsScanResponse:
    logging.info(f"get opt-in-out-urls-scan for dataset={dataset} config={config} split={split}")

    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False
    if not spawning_token:
        raise MissingSpawningTokenError("OPT_IN_OUT_URLS_SCAN_SPAWNING_TOKEN is not set")

    # get the first rows from previous job
    upstream_response = get_previous_step_or_raise(
        kinds=["split-first-rows-from-streaming"], dataset=dataset, config=config, split=split
    )
    try:
        first_rows_response = upstream_response.response
        upstream_response_content = SplitFirstRowsResponse(
            dataset=dataset,
            config=config,
            split=split,
            features=first_rows_response["content"]["features"],
            rows=first_rows_response["content"]["rows"],
        )

        features = upstream_response_content["features"]
        first_rows = upstream_response_content["rows"]
    except KeyError as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

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
    string_columns = [feature["name"] for feature in features if feature["type"] == string_type_dict]
    urls_columns = []
    for string_column in string_columns:
        urls_count = sum(
            1
            for row in first_rows
            if isinstance(row["row"].get(string_column), str)
            and (row["row"][string_column].startswith("https://") or row["row"][string_column].startswith("http://"))
        )
        if urls_count and urls_count / len(first_rows) > 0.5:
            urls_columns.append(string_column)

    if not urls_columns:
        return OptInOutUrlsScanResponse(
            urls_columns=[],
            opt_in_urls=[],
            opt_out_urls=[],
            num_opt_in_urls=0,
            num_opt_out_urls=0,
            num_urls=0,
            num_scanned_rows=0,
            has_urls_columns=False,
        )

    if len(urls_columns) > columns_max_number:
        raise TooManyColumnsError(
            f"The number of columns ({len(urls_columns)}) exceeds the maximum supported number of columns to scan"
            f" ({columns_max_number})."
        )

    # get the rows
    rows = get_rows_or_raise(
        dataset=dataset,
        config=config,
        split=split,
        info=info,
        rows_max_number=rows_max_number,
        use_auth_token=use_auth_token,
        column_names=urls_columns,
    )

    # get the urls
    num_scanned_rows = len(rows)
    urls = [row[urls_column] for row in rows for urls_column in urls_columns]

    # scan the urls
    opt_in_urls_indices, opt_out_urls_indices = run(
        opt_in_out_scan_urls(
            urls,
            urls_number_per_batch=urls_number_per_batch,
            spawning_token=spawning_token,
            max_concurrent_requests_number=max_concurrent_requests_number,
            max_requests_per_second=max_requests_per_second,
            spawning_url=spawning_url,
        )
    )

    opt_in_urls = [
        OptUrl(
            url=urls[url_idx],
            row_idx=url_idx // len(urls_columns),
            column_name=urls_columns[url_idx % len(urls_columns)],
        )
        for url_idx in opt_in_urls_indices
    ]
    opt_out_urls = [
        OptUrl(
            url=urls[url_idx],
            row_idx=url_idx // len(urls_columns),
            column_name=urls_columns[url_idx % len(urls_columns)],
        )
        for url_idx in opt_out_urls_indices
    ]

    # return scan result
    return OptInOutUrlsScanResponse(
        urls_columns=urls_columns,
        opt_in_urls=opt_in_urls,
        opt_out_urls=opt_out_urls,
        num_opt_in_urls=len(opt_in_urls),
        num_opt_out_urls=len(opt_out_urls),
        num_urls=len(urls),
        num_scanned_rows=num_scanned_rows,
        has_urls_columns=True,
    )


class SplitOptInOutUrlsScanJobRunner(DatasetsBasedJobRunner):
    urls_scan_config: OptInOutUrlsScanConfig

    @staticmethod
    def get_job_type() -> str:
        return "split-opt-in-out-urls-scan"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        hf_datasets_cache: Path,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.urls_scan_config = app_config.urls_scan

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
                spawning_url=self.urls_scan_config.spawning_url,
            )
        )

    def get_new_splits(self, _: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by compute."""
        if self.config is None or self.split is None:
            raise ValueError("config and split are required")
        return {SplitFullName(dataset=self.dataset, config=self.config, split=self.split)}
