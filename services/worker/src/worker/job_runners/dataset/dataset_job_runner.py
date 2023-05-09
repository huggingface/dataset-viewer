# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path

from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo

from worker.common_exceptions import ParameterMissingError
from worker.config import AppConfig
from worker.job_runner import JobRunner
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner


class DatasetJobRunner(JobRunner):
    dataset: str

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config, processing_step=processing_step)
        if job_info["params"]["dataset"] is None:
            raise ParameterMissingError("'dataset' parameter is required")
        self.dataset = job_info["params"]["dataset"]


class DatasetCachedJobRunner(DatasetsBasedJobRunner, DatasetJobRunner):
    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        hf_datasets_cache: Path,
    ) -> None:
        DatasetsBasedJobRunner.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        DatasetJobRunner.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
