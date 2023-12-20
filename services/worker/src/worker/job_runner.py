# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from abc import ABC, abstractmethod

from libcommon.constants import PARALLEL_STEPS_LISTS
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.dtos import JobResult


def get_parallel_step_names(job_type: str) -> list[str]:
    for parallel_step_names in PARALLEL_STEPS_LISTS:
        if job_type in parallel_step_names:
            return [step_name for step_name in parallel_step_names if step_name != job_type]
    return []


class JobRunner(ABC):
    job_info: JobInfo
    app_config: AppConfig
    parallel_step_names: list[str] = []

    @staticmethod
    @abstractmethod
    def get_job_type() -> str:
        pass

    def __init__(self, job_info: JobInfo, app_config: AppConfig) -> None:
        self.job_info = job_info
        self.app_config = app_config
        self.parallel_step_names = get_parallel_step_names(self.get_job_type())

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
