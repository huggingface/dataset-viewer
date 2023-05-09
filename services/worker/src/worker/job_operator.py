# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from abc import ABC, abstractmethod
from typing import Optional

from libcommon.processing_graph import ProcessingStep
from libcommon.utils import JobInfo

from worker.config import AppConfig
from worker.utils import JobResult, OperatorInfo


class JobOperator(ABC):
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
    def get_parallel_operator() -> Optional[OperatorInfo]:  # In the future it could be a list of parallel operators
        return None

    def __init__(self, job_info: JobInfo, app_config: AppConfig, processing_step: ProcessingStep) -> None:
        self.job_info = job_info
        self.app_config = app_config
        self.processing_step = processing_step
        self.job_type = job_info["type"]
        self.job_id = job_info["job_id"]
        self.force = job_info["force"]
        self.priority = job_info["priority"]

    def pre_compute(self) -> None:
        """Hook method called before the compute method."""
        pass

    @abstractmethod
    def compute(self) -> JobResult:
        pass

    def post_compute(self) -> None:
        """Hook method called after the compute method."""
        pass
