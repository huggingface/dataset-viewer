# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Tuple

from libcommon.constants import PROCESSING_STEP_DATASET_OPT_IN_OUT_URLS_COUNT_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    DoesNotExist,
    get_previous_step_or_raise,
    get_response,
)

from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner
from worker.utils import JobResult, OptInOutUrlsCountResponse


def compute_opt_in_out_urls_count_response(dataset: str) -> Tuple[OptInOutUrlsCountResponse, float]:
    logging.info(f"get opt-in-out-urls-count for dataset={dataset}")

    config_names_response = get_previous_step_or_raise(kinds=["dataset-config-names"], dataset=dataset)
    content = config_names_response.response["content"]
    if "config_names" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'config_names'.")

    urls_columns = []
    num_opt_in_urls = 0
    num_opt_out_urls = 0
    num_urls = 0
    num_scanned_rows = 0
    full_scan_count = 0
    try:
        total = 0
        pending = 0
        for config_item in content["config_names"]:
            config = config_item["config"]
            total += 1
            try:
                response = get_response(kind="config-opt-in-out-urls-count", dataset=dataset, config=config)
            except DoesNotExist:
                logging.debug("No response found in previous step for this dataset: 'config-opt-in-out-urls-count'.")
                pending += 1
                continue
            if response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {response['http_status']}.")
                continue
            else:
                if response["progress"] and response["progress"] < 1.0:
                    logging.debug(f"Previous step is still in progress: {response['progress']}.")
                    pending += 1
                    continue
            split_opt_in_out_content = response["content"]
            urls_columns.extend(split_opt_in_out_content["urls_columns"])
            num_opt_in_urls += split_opt_in_out_content["num_opt_in_urls"]
            num_opt_out_urls += split_opt_in_out_content["num_opt_out_urls"]
            num_urls += split_opt_in_out_content["num_urls"]
            num_scanned_rows += split_opt_in_out_content["num_scanned_rows"]
            full_scan_count += 1 if split_opt_in_out_content["full_scan"] else 0
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    unique_urls_columns = sorted(list(set(urls_columns)))
    has_urls_columns = len(unique_urls_columns) > 0
    progress = (total - pending) / total if total else 1.0
    full_scan = full_scan_count == total

    return (
        OptInOutUrlsCountResponse(
            urls_columns=unique_urls_columns,
            has_urls_columns=has_urls_columns,
            num_opt_in_urls=num_opt_in_urls,
            num_opt_out_urls=num_opt_out_urls,
            num_scanned_rows=num_scanned_rows,
            num_urls=num_urls,
            full_scan=full_scan,
        ),
        progress,
    )


class DatasetOptInOutUrlsCountJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-opt-in-out-urls-count"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_OPT_IN_OUT_URLS_COUNT_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_opt_in_out_urls_count_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
