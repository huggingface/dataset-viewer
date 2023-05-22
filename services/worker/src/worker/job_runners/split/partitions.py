# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import PROCESSING_STEP_SPLIT_PARTITIONS_VERSION
from libcommon.exceptions import PreviousStepFormatError, SplitNotFoundError
from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo

from worker.config import AppConfig, PartitionConfig
from worker.job_runners.split.split_job_runner import SplitJobRunner
from worker.utils import (
    CompleteJobResult,
    PartitionsReponse,
    get_previous_step_or_raise,
    partition_to_string,
)


def compute_partitions(
    dataset: str,
    config: str,
    split: str,
    chunck_size: int,
) -> PartitionsReponse:
    logging.info(f"get partitions for dataset={dataset} config={config} split={split}")

    # get the sizes from previous job
    size_upstream_response = get_previous_step_or_raise(kinds=["config-size"], dataset=dataset, config=config)
    try:
        size_response = size_upstream_response.response
        splits_sizes = size_response["content"]["size"]["splits"]
    except KeyError as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    # generating partitions
    split_size = next((split_item for split_item in splits_sizes if split_item["split"] == split), None)
    if split_size is None:
        raise SplitNotFoundError(f"The split '{split}' does not exist for the config '{config}' of the dataset.")

    num_rows = split_size["num_rows"]
    partitions = [
        partition_to_string(
            partition_start=offset,
            partition_end=offset + chunck_size - 1 if offset + chunck_size - 1 < num_rows else num_rows - 1,
        )
        for offset in range(0, num_rows, chunck_size)
    ]

    return PartitionsReponse(partitions=partitions, num_rows=num_rows)


class SplitPartitionsJobRunner(SplitJobRunner):
    partition_config: PartitionConfig

    @staticmethod
    def get_job_type() -> str:
        return "split-partitions"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_PARTITIONS_VERSION

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
        self.partition_config = app_config.partition

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_partitions(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                chunck_size=self.partition_config.chunk_size,
            )
        )
