# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.exceptions import ParameterMissingError
from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import PARTITIONS_SEPARATOR, partition_values_from_string


class PartitionJobRunner(SplitJobRunner):
    partition: str
    partition_start: int
    partition_end: int

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config, processing_step=processing_step)
        if job_info["params"]["partition"] is None:
            raise ParameterMissingError("'partition' parameter is required")
        self.partition = job_info["params"]["partition"]
        (partition_start, partition_end) = partition_values_from_string(self.partition)
        if partition_start is None or partition_end is None:
            raise ValueError(
                "'partition' parameter has a wrong format, expected is"
                f" partition_start{PARTITIONS_SEPARATOR}partition_end"
            )
        self.partition_start = partition_start
        self.partition_end = partition_end
