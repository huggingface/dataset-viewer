# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path

from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_operators._datasets_based_job_operator import DatasetsBasedJobOperator
from worker.job_operators.dataset.dataset_job_operator import DatasetJobOperator
from worker.job_runner import ParameterMissingError


class ConfigJobOperator(DatasetJobOperator):
    config: str

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config, processing_step=processing_step)
        if job_info["params"]["config"] is None:
            raise ParameterMissingError("'config' parameter is required")
        self.config = job_info["params"]["config"]


class ConfigCachedJobOperator(DatasetsBasedJobOperator, ConfigJobOperator):
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
        ConfigJobOperator.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
        )
