# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus

from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import (
    CacheEntryDoesNotExistError,
    get_response,
)

from worker.dtos import JobResult, OptInOutUrlsCountResponse
from worker.job_runners.config.config_job_runner import ConfigJobRunner
from worker.utils import get_split_names


def compute_opt_in_out_urls_count_response(dataset: str, config: str) -> tuple[OptInOutUrlsCountResponse, float]:
    logging.info(f"compute 'config-opt-in-out-urls-count' for {dataset=} {config=}")

    urls_columns = []
    num_opt_in_urls = 0
    num_opt_out_urls = 0
    num_urls = 0
    num_scanned_rows = 0
    full_scan_count = 0
    splits = get_split_names(dataset=dataset, config=config)
    try:
        total = 0
        pending = 0
        for split in splits:
            total += 1
            try:
                response = get_response(
                    kind="split-opt-in-out-urls-count", dataset=dataset, config=config, split=split
                )
            except CacheEntryDoesNotExistError:
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

    def compute(self) -> JobResult:
        response_content, progress = compute_opt_in_out_urls_count_response(dataset=self.dataset, config=self.config)
        return JobResult(response_content, progress=progress)
