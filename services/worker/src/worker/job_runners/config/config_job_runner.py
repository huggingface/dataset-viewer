# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path

from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo

from worker.config import AppConfig
from worker.job_runner import JobRunner, ParameterMissingError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner


class ConfigRunner(JobRunner):
    config: str

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config, processing_step=processing_step)
        if self._config is None:
            raise ParameterMissingError("'config' parameter is required")
        self.config = self._config


class ConfigCacheRunner(DatasetsBasedJobRunner, ConfigRunner):
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
        ConfigRunner.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
