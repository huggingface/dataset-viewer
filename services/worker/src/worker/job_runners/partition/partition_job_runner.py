# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.exceptions import ParameterMissingError
from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_runners.split.split_job_runner import SplitJobRunner


class PartitionJobRunner(SplitJobRunner):
    partition_start: int
    partition_end: int

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config, processing_step=processing_step)
        if job_info["params"]["partition_start"] is None or job_info["params"]["partition_end"] is None:
            raise ParameterMissingError("'partition_start' and 'partition_end' parameters are required")
        self.partition_start = job_info["params"]["partition_start"]
        self.partition_end = job_info["params"]["partition_end"]
