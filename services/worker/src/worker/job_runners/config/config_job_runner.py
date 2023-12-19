# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path

from libcommon.exceptions import ParameterMissingError
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_runners._job_runner_with_datasets_cache import (
    JobRunnerWithDatasetsCache,
)
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner
from worker.utils import check_config_exists


class ConfigJobRunner(DatasetJobRunner):
    config: str

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config)
        if job_info["params"]["config"] is None:
            raise ParameterMissingError("'config' parameter is required")
        self.config = job_info["params"]["config"]

    def validate(self) -> None:
        check_config_exists(dataset=self.dataset, config=self.config)


class ConfigJobRunnerWithDatasetsCache(JobRunnerWithDatasetsCache, ConfigJobRunner):
    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        hf_datasets_cache: Path,
    ) -> None:
        JobRunnerWithDatasetsCache.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
            hf_datasets_cache=hf_datasets_cache,
        )
        ConfigJobRunner.__init__(
            self=self,
            job_info=job_info,
            app_config=app_config,
        )
