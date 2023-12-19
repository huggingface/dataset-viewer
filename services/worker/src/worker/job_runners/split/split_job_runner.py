# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path

from libcommon.exceptions import ParameterMissingError
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.job_runners._job_runner_with_cache import JobRunnerWithCache
from worker.job_runners._job_runner_with_datasets_cache import (
    JobRunnerWithDatasetsCache,
)
from worker.job_runners.config.config_job_runner import ConfigJobRunner
from worker.utils import check_split_exists


class SplitJobRunner(ConfigJobRunner):
    split: str

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config)
        if job_info["params"]["split"] is None:
            raise ParameterMissingError("'split' parameter is required")
        self.split = job_info["params"]["split"]

    def validate(self) -> None:
        check_split_exists(dataset=self.dataset, config=self.config, split=self.split)


class SplitJobRunnerWithDatasetsCache(JobRunnerWithDatasetsCache, SplitJobRunner):
    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        hf_datasets_cache: Path,
    ) -> None:
        JobRunnerWithDatasetsCache.__init__(
            self,
            job_info=job_info,
            app_config=app_config,
            hf_datasets_cache=hf_datasets_cache,
        )
        SplitJobRunner.__init__(
            self,
            job_info=job_info,
            app_config=app_config,
        )


class SplitJobRunnerWithCache(JobRunnerWithCache, SplitJobRunner):
    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        cache_directory: Path,
    ) -> None:
        JobRunnerWithCache.__init__(
            self,
            job_info=job_info,
            app_config=app_config,
            cache_directory=cache_directory,
        )
        SplitJobRunner.__init__(
            self,
            job_info=job_info,
            app_config=app_config,
        )
