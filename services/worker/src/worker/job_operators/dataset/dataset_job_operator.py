# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path

from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo
from worker.job_operators._datasets_based_job_operator import (
    DatasetsBasedJobOperator,
)

from worker.config import AppConfig
from worker.job_operator import JobOperator
from worker.job_runner import JobRunner, ParameterMissingError


class DatasetJobOperator(JobOperator):
    dataset: str

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config, processing_step=processing_step)
        if job_info["dataset"] is None:
            raise ParameterMissingError("'dataset' parameter is required")
        self.dataset = job_info["dataset"]


class DatasetCachedJobRunner(DatasetsBasedJobOperator, DatasetJobOperator):
    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        hf_datasets_cache: Path,
    ) -> None:
        DatasetsBasedJobOperator.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        DatasetJobOperator.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
