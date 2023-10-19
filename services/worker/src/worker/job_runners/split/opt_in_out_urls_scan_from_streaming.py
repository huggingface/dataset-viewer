# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from asyncio import Semaphore, create_task, run, wait
from pathlib import Path
from typing import Any, Optional

from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from datasets import get_dataset_config_info
from libcommon.constants import PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_SCAN_VERSION
from libcommon.exceptions import (
    ExternalServerError,
    InfoError,
    MissingSpawningTokenError,
    PreviousStepFormatError,
    TooManyColumnsError,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.utils import JobInfo

from worker.config import AppConfig, OptInOutUrlsScanConfig
from worker.dtos import CompleteJobResult, OptInOutUrlsScanResponse, OptUrl
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithDatasetsCache
from worker.utils import disable_dataset_scripts_support, get_rows_or_raise


async def check_spawning(
    image_urls: list[str], session: ClientSession, semaphore: Semaphore, limiter: AsyncLimiter, spawning_url: str
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
    image_urls: list[str], session: ClientSession, semaphore: Semaphore, limiter: AsyncLimiter, spawning_url: str
) -> tuple[list[Any], list[Any]]:
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
    urls: list[str],
    urls_number_per_batch: int,
    spawning_token: str,
    max_concurrent_requests_number: int,
    max_requests_per_second: int,
    spawning_url: str,
) -> tuple[list[int], list[int]]:
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
    dataset_scripts_allow_list: list[str],
) -> OptInOutUrlsScanResponse:
    """
    Get the response of split-opt-in-out-urls-scan cache for a specific split of a dataset from huggingface.co.
    The response is not used directly in the API but it is an input for config-opt-in-out-urls-scan processing step.
    Note that only image URLs are scanned, see image_url_columns.py for details about the detection heuristic.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
        hf_token (`str` or `None`):
            An authentication token (See https://huggingface.co/settings/token)
        rows_max_number (`int`):
            The maximum number of rows of the response.
        columns_max_number (`int`):
            The maximum number of supported columns.
        urls_number_per_batch (`int`):
            The number of batch URLs to be sent to spawning service.
        spawning_token (`str` or `None`):
            An authentication token to use spawning service (See https://api.spawning.ai/spawning-api)
        max_concurrent_requests_number (`int`):
            The maximum number of requests to be processed concurrently.
        max_requests_per_second (`int`):
            The maximum number of requests to be processed by second.
        spawning_url (`str`):
            Spawgning API URL
        dataset_scripts_allow_list (`list[str]`):
            List of datasets for which we support dataset scripts.
            Unix shell-style wildcards also work in the dataset name for namespaced datasets,
            for example `some_namespace/*` to refer to all the datasets in the `some_namespace` namespace.
            The keyword `{{ALL_DATASETS_WITH_NO_NAMESPACE}}` refers to all the datasets without namespace.

    Returns:
        [`OptInOutUrlsScanResponse`]
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
          If the content of the previous step has not the expected format
        - [`libcommon.exceptions.InfoError`]
          If the config info could not be obtained using the datasets library.
        - [`libcommon.exceptions.TooManyColumnsError`]
          If the number of columns (features) exceeds the maximum supported number of columns.
        - [`libcommon.exceptions.StreamingRowsError`]
          If the split rows could not be obtained using the datasets library in streaming mode.
        - [`libcommon.exceptions.NormalRowsError`]
          If the split rows could not be obtained using the datasets library in normal mode.
        - [`libcommon.exceptions.DatasetWithScriptNotSupportedError`]
            If the dataset has a dataset script and is not in the allow list.
    """
    logging.info(f"get opt-in-out-urls-scan for dataset={dataset} config={config} split={split}")

    if not spawning_token:
        raise MissingSpawningTokenError("OPT_IN_OUT_URLS_SCAN_SPAWNING_TOKEN is not set")

    # get image url columns from previous job
    upstream_response = get_previous_step_or_raise(
        kinds=["split-image-url-columns"],
        dataset=dataset,
        config=config,
        split=split,
    )
    try:
        image_url_columns_response = upstream_response.response
        image_url_columns = image_url_columns_response["content"]["columns"]
    except KeyError as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    # get the info
    try:
        info = get_dataset_config_info(
            path=dataset,
            config_name=config,
            token=hf_token,
        )
    except Exception as err:
        raise InfoError(
            f"The info cannot be fetched for the config '{config}' of the dataset.",
            cause=err,
        ) from err

    if not image_url_columns:
        return OptInOutUrlsScanResponse(
            urls_columns=[],
            opt_in_urls=[],
            opt_out_urls=[],
            num_opt_in_urls=0,
            num_opt_out_urls=0,
            num_urls=0,
            num_scanned_rows=0,
            has_urls_columns=False,
            full_scan=None,
        )

    if len(image_url_columns) > columns_max_number:
        raise TooManyColumnsError(
            f"The number of columns ({len(image_url_columns)}) exceeds the maximum supported number of columns to scan"
            f" ({columns_max_number})."
        )

    # get the rows
    with disable_dataset_scripts_support(dataset_scripts_allow_list):
        rows_content = get_rows_or_raise(
            dataset=dataset,
            config=config,
            split=split,
            info=info,
            rows_max_number=rows_max_number,
            token=hf_token,
            column_names=image_url_columns,
        )
    rows = rows_content["rows"]

    # get the urls
    num_scanned_rows = len(rows)
    urls = [row[urls_column] for row in rows for urls_column in image_url_columns]

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
            row_idx=url_idx // len(image_url_columns),
            column_name=image_url_columns[url_idx % len(image_url_columns)],
        )
        for url_idx in opt_in_urls_indices
    ]
    opt_out_urls = [
        OptUrl(
            url=urls[url_idx],
            row_idx=url_idx // len(image_url_columns),
            column_name=image_url_columns[url_idx % len(image_url_columns)],
        )
        for url_idx in opt_out_urls_indices
    ]

    # return scan result
    return OptInOutUrlsScanResponse(
        urls_columns=image_url_columns,
        opt_in_urls=opt_in_urls,
        opt_out_urls=opt_out_urls,
        num_opt_in_urls=len(opt_in_urls),
        num_opt_out_urls=len(opt_out_urls),
        num_urls=len(urls),
        num_scanned_rows=num_scanned_rows,
        has_urls_columns=True,
        full_scan=rows_content["all_fetched"],
    )


class SplitOptInOutUrlsScanJobRunner(SplitJobRunnerWithDatasetsCache):
    urls_scan_config: OptInOutUrlsScanConfig

    @staticmethod
    def get_job_type() -> str:
        return "split-opt-in-out-urls-scan"

    # ^ TODO: Change step name referring to image URLs scan specifically.

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
        return CompleteJobResult(
            compute_opt_in_out_urls_scan_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                hf_token=self.app_config.common.hf_token,
                rows_max_number=self.urls_scan_config.rows_max_number,
                columns_max_number=self.urls_scan_config.columns_max_number,
                urls_number_per_batch=self.urls_scan_config.urls_number_per_batch,
                spawning_token=self.urls_scan_config.spawning_token,
                max_concurrent_requests_number=self.urls_scan_config.max_concurrent_requests_number,
                max_requests_per_second=self.urls_scan_config.max_requests_per_second,
                spawning_url=self.urls_scan_config.spawning_url,
                dataset_scripts_allow_list=self.app_config.common.dataset_scripts_allow_list,
            )
        )
