# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Optional
from libcommon.exceptions import (
    CustomError,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo
from libcommon.simple_cache import (
    DoesNotExist,
    get_response_without_content_info,
)
from worker.config import AppConfig
from worker.utils import JobResult


class JobOperatorError(CustomError):
    """Base class for job runner exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: str,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class ResponseAlreadyComputedError(JobOperatorError):
    """Raised when response has been already computed by another operator."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ResponseAlreadyComputedError", cause, True)


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

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
    ) -> None:
        self.job_info = job_info
        self.job_type = job_info["type"]
        self.job_id = job_info["job_id"]
        self.dataset = job_info["dataset"]
        self.force = job_info["force"]
        self.app_config = app_config
        self.priority = job_info["priority"]
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

    def raise_if_parallel_response_exists(self, parallel_cache_kind: str, parallel_job_version: int) -> None:
        try:
            existing_response = get_response_without_content_info(
                kind=parallel_cache_kind,
                dataset=self.dataset,
                job_info=self.job_info,
            )
            dataset_git_revision = self.get_dataset_git_revision()
            if (
                existing_response["http_status"] == HTTPStatus.OK
                and existing_response["job_runner_version"] == parallel_job_version
                and existing_response["progress"] == 1.0  # completed response
                and dataset_git_revision is not None
                and existing_response["dataset_git_revision"] == dataset_git_revision
            ):
                raise ResponseAlreadyComputedError(
                    f"Response has already been computed and stored in cache kind: {parallel_cache_kind}. Compute will"
                    " be skipped."
                )
        except DoesNotExist:
            logging.debug(f"no cache found for {parallel_cache_kind}.")
