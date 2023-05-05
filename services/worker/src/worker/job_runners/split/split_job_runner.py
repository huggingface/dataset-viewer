# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path

from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo

from worker.config import AppConfig
from worker.job_runner import ParameterMissingError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner
from worker.job_runners.config.config_job_runner import ConfigJobRunner


class SplitJobRunner(ConfigJobRunner):
    split: str

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config, processing_step=processing_step)
        if self._split is None:
            raise ParameterMissingError("'split' parameter is required")
        self.split = self._split


class SplitCachedJobRunner(DatasetsBasedJobRunner, SplitJobRunner):
    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        hf_datasets_cache: Path,
    ) -> None:
        DatasetsBasedJobRunner.__init__(
            self,
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        SplitJobRunner.__init__(
            self,
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
