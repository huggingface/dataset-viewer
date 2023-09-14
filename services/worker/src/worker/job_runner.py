# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from abc import ABC, abstractmethod
from typing import Optional

from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.dtos import JobResult, JobRunnerInfo


class JobRunner(ABC):
    job_info: JobInfo
    app_config: AppConfig
    processing_step: ProcessingStep

    @staticmethod
    @abstractmethod
    def get_job_type() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_job_runner_version() -> int:
        pass

    @staticmethod
    def get_parallel_job_runner() -> Optional[JobRunnerInfo]:  # In the future it could be a list
        return None

    def __init__(self, job_info: JobInfo, app_config: AppConfig, processing_step: ProcessingStep) -> None:
        self.job_info = job_info
        self.app_config = app_config
        self.processing_step = processing_step

    def pre_compute(self) -> None:
        """Hook method called before the compute method."""
        pass

    @abstractmethod
    def compute(self) -> JobResult:
        pass

    def post_compute(self) -> None:
        """Hook method called after the compute method."""
        pass

    def validate(self) -> None:
        """
        Validate that this job should be run.
        It should raise an error if e.g. the config/split of the dataset to process doesn't exist.
        """
        pass
