# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_COUNT_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise

from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import CompleteJobResult, OptInOutUrlsCountResponse


def compute_opt_in_out_urls_count_response(
    dataset: str,
    config: str,
    split: str,
) -> OptInOutUrlsCountResponse:
    logging.info(f"get opt-in-out-urls-count for dataset={dataset} config={config} split={split}")

    opt_in_out_urls_scan = get_previous_step_or_raise(
        kinds=["split-opt-in-out-urls-scan"], dataset=dataset, config=config, split=split
    )

    try:
        content = opt_in_out_urls_scan.response["content"]
        opt_in_out_urls_count = OptInOutUrlsCountResponse(
            has_urls_columns=content["has_urls_columns"],
            num_opt_in_urls=content["num_opt_in_urls"],
            num_opt_out_urls=content["num_opt_out_urls"],
            num_scanned_rows=content["num_scanned_rows"],
            num_urls=content["num_urls"],
            urls_columns=content["urls_columns"],
            full_scan=content["full_scan"],
        )
    except KeyError as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return opt_in_out_urls_count


class SplitOptInOutUrlsCountJobRunner(SplitJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "split-opt-in-out-urls-count"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_OPT_IN_OUT_URLS_COUNT_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_opt_in_out_urls_count_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
            )
        )
