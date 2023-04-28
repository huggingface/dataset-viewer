# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, Literal, Mapping, Optional, Tuple

from libcommon.constants import PROCESSING_STEP_CONFIG_OPT_IN_OUT_URLS_SCAN_VERSION
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.job_runner import (
    JobResult,
    JobRunner,
    JobRunnerError,
    ParameterMissingError,
    get_previous_step_or_raise,
)
from worker.utils import OptInOutUrlsScanResponse

ConfigOptInOutUrlsScanJobRunnerErrorCode = Literal["PreviousStepFormatError"]


class ConfigOptInOutUrlsScanJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ConfigOptInOutUrlsScanJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepFormatError(ConfigOptInOutUrlsScanJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


def compute_opt_in_out_urls_scan_response(dataset: str, config: str) -> Tuple[OptInOutUrlsScanResponse, float]:
    logging.info(f"get config-opt-in-out-urls-scan for dataset={dataset} config={config}")

    split_names_response = get_previous_step_or_raise(
        kinds=["/split-names-from-streaming"], dataset=dataset, config=config
    )
    content = split_names_response.response["content"]
    if "splits" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'splits'.")

    urls_columns = []
    num_opt_in_urls = 0
    num_opt_out_urls = 0
    num_urls = 0
    num_scanned_rows = 0
    try:
        total = 0
        pending = 0
        for split_item in content["splits"]:
            split = split_item["split"]
            total += 1
            try:
                response = get_response(kind="split-opt-in-out-urls-scan", dataset=dataset, config=config, split=split)
            except DoesNotExist:
                logging.debug("No response found in previous step for this dataset: 'split-opt-in-out-urls-scan'.")
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
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    unique_urls_columns = list(set(urls_columns))
    has_urls_columns = len(unique_urls_columns) > 0
    progress = (total - pending) / total if total else 1.0

    return (
        OptInOutUrlsScanResponse(
            urls_columns=unique_urls_columns,
            has_urls_columns=has_urls_columns,
            num_opt_in_urls=num_opt_in_urls,
            num_opt_out_urls=num_opt_out_urls,
            num_scanned_rows=num_scanned_rows,
            num_urls=num_urls,
        ),
        progress,
    )


class ConfigOptInOutUrlsScanJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-opt-in-out-urls-scan"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_OPT_IN_OUT_URLS_SCAN_VERSION

    def compute(self) -> JobResult:
        if self.dataset is None:
            raise ParameterMissingError("'dataset' parameter is required")
        if self.config is None:
            raise ParameterMissingError("'config' parameter is required")
        response_content, progress = compute_opt_in_out_urls_scan_response(dataset=self.dataset, config=self.config)
        return JobResult(response_content, progress=progress)

    def get_new_splits(self, _: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {SplitFullName(dataset=self.dataset, config=self.config, split=None)}
