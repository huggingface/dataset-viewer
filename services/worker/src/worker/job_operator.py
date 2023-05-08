# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import List, Optional
from libcommon.exceptions import (
    CustomError,
)
from libcommon.utils import orjson_dumps
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo
from libcommon.simple_cache import (
    CacheEntryWithDetails,
    BestResponse,
    DoesNotExist,
    get_best_response,
    get_response_without_content_params,
)
from worker.config import AppConfig
from worker.utils import JobResult
from libcommon.exceptions import (
    CustomError,
    ErrorResponseWithCause,
    ErrorResponseWithoutCause,
)

def get_previous_step_or_raise(
    kinds: List[str], dataset: str, config: Optional[str] = None, split: Optional[str] = None
) -> BestResponse:
    """Get the previous step from the cache, or raise an exception if it failed."""
    best_response = get_best_response(kinds=kinds, dataset=dataset, config=config, split=split)
    if best_response.response["http_status"] != HTTPStatus.OK:
        raise PreviousStepError.from_response(
            response=best_response.response,
            kind=best_response.kind,
            dataset=dataset,
            config=config,
            split=split,
        )
    return best_response


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


class PreviousStepError(JobOperatorError):
    """Raised when the previous step failed. It contains the contents of the error response,
    and the details contain extra information about the previous step.
    """

    error_with_cause: ErrorResponseWithCause
    error_without_cause: ErrorResponseWithoutCause

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: str,
        cause: Optional[BaseException],
        disclose_cause: bool,
        error_with_cause: ErrorResponseWithCause,
        error_without_cause: ErrorResponseWithoutCause,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )
        self.error_with_cause = error_with_cause
        self.error_without_cause = error_without_cause

    @staticmethod
    def from_response(
        response: CacheEntryWithDetails,
        kind: str,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> "PreviousStepError":
        if response.get("http_status") == HTTPStatus.OK:
            raise ValueError("Cannot create a PreviousStepError, the response should contain an error")

        message = response["content"]["error"] if "error" in response["content"] else "Unknown error"
        status_code = response["http_status"]
        error_code = response["error_code"] or "PreviousStepError"
        cause = None  # No way to create the same exception
        disclose_cause = orjson_dumps(response["details"]) == orjson_dumps(response["content"])
        error_without_cause: ErrorResponseWithoutCause = {"error": message}
        error_with_cause: ErrorResponseWithCause = {
            "error": message,
            # Add lines in the traceback to give some info about the previous step error (a bit hacky)
            "cause_traceback": [
                "The previous step failed, the error is copied to this step:",
                f"  {kind=} {dataset=} {config=} {split=}",
                "---",
            ],
        }
        if "cause_exception" in response["details"] and isinstance(response["details"]["cause_exception"], str):
            error_with_cause["cause_exception"] = response["details"]["cause_exception"]
        if "cause_message" in response["details"] and isinstance(response["details"]["cause_message"], str):
            error_with_cause["cause_message"] = response["details"]["cause_message"]
        if (
            "cause_traceback" in response["details"]
            and isinstance(response["details"]["cause_traceback"], list)
            and all(isinstance(line, str) for line in response["details"]["cause_traceback"])
        ):
            error_with_cause["cause_traceback"].extend(response["details"]["cause_traceback"])
        return PreviousStepError(
            message=message,
            status_code=status_code,
            code=error_code,
            cause=cause,
            disclose_cause=disclose_cause,
            error_without_cause=error_without_cause,
            error_with_cause=error_with_cause,
        )

    def as_response_with_cause(self) -> ErrorResponseWithCause:
        return self.error_with_cause

    def as_response_without_cause(self) -> ErrorResponseWithoutCause:
        return self.error_without_cause


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
            existing_response = get_response_without_content_params(
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
