# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Tuple

from libcommon.constants import PROCESSING_STEP_CONFIG_OPT_IN_OUT_URLS_COUNT_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    get_previous_step_or_raise,
    get_response,
)
from mongoengine.errors import DoesNotExist

from worker.job_runners.config.config_job_runner import ConfigJobRunner
from worker.utils import JobResult, OptInOutUrlsCountResponse


def compute_opt_in_out_urls_scan_response(dataset: str, config: str) -> Tuple[OptInOutUrlsCountResponse, float]:
    logging.info(f"get config-opt-in-out-urls-count for dataset={dataset} config={config}")

    split_names_response = get_previous_step_or_raise(
        kinds=["config-split-names-from-streaming", "config-split-names-from-info"],
        dataset=dataset,
        config=config,
    )
    content = split_names_response.response["content"]
    if "splits" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'splits'.")

    urls_columns = []
    num_opt_in_urls = 0
    num_opt_out_urls = 0
    num_urls = 0
    num_scanned_rows = 0
    full_scan_count = 0
    try:
        total = 0
        pending = 0
        for split_item in content["splits"]:
            split = split_item["split"]
            total += 1
            try:
                response = get_response(
                    kind="split-opt-in-out-urls-count", dataset=dataset, config=config, split=split
                )
            except DoesNotExist:
                logging.debug("No response found in previous step for this dataset: 'split-opt-in-out-urls-count'.")
                pending += 1
                continue
            if response["http_status"] != HTTPStatus.OK:
                logging.debug(f"Previous step gave an error: {response['http_status']}.")
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


class ConfigOptInOutUrlsCountJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-opt-in-out-urls-count"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_OPT_IN_OUT_URLS_COUNT_VERSION

    def compute(self) -> JobResult:
        response_content, progress = compute_opt_in_out_urls_scan_response(dataset=self.dataset, config=self.config)
        return JobResult(response_content, progress=progress)
