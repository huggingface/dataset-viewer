# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional

from libcommon.config import CommonConfig
from libcommon.dataset import DatasetNotFoundError, get_dataset_git_revision
from libcommon.exceptions import (
    CustomError,
    ErrorResponseWithCause,
    ErrorResponseWithoutCause,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo, Priority, Queue, Status
from libcommon.simple_cache import (
    BestResponse,
    CacheEntryWithDetails,
    DoesNotExist,
    SplitFullName,
    get_best_response,
    get_response,
    get_response_without_content,
    upsert_response,
)
from libcommon.utils import orjson_dumps

from worker.config import WorkerConfig

GeneralJobRunnerErrorCode = Literal[
    "ParameterMissingError",
    "NoGitRevisionError",
    "SplitNotFoundError",
    "UnexpectedError",
    "TooBigContentError",
    "JobRunnerCrashedError",
    "JobRunnerExceededMaximumDurationError",
    "ResponseAlreadyComputedError",
]

# List of error codes that should trigger a retry.
ERROR_CODES_TO_RETRY: list[str] = ["ClientConnectionError"]


@dataclass
class JobResult:
    content: Mapping[str, Any]
    progress: float

    def __post_init__(self) -> None:
        if self.progress < 0.0 or self.progress > 1.0:
            raise ValueError(f"Progress should be between 0 and 1, but got {self.progress}")


@dataclass
class CompleteJobResult(JobResult):
    content: Mapping[str, Any]
    progress: float = field(init=False, default=1.0)


class JobRunnerError(CustomError):
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


class GeneralJobRunnerError(JobRunnerError):
    """General class for job runner exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: GeneralJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class SplitNotFoundError(GeneralJobRunnerError):
    """Raised when the split does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="SplitNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class ParameterMissingError(GeneralJobRunnerError):
    """Raised when request is missing some parameter."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.BAD_REQUEST,
            code="ParameterMissingError",
            cause=cause,
            disclose_cause=False,
        )


class NoGitRevisionError(GeneralJobRunnerError):
    """Raised when the git revision returned by huggingface_hub is None."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="NoGitRevisionError",
            cause=cause,
            disclose_cause=False,
        )


class TooBigContentError(GeneralJobRunnerError):
    """Raised when content size in bytes is bigger than the supported value."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="TooBigContentError",
            cause=cause,
            disclose_cause=False,
        )


class UnexpectedError(GeneralJobRunnerError):
    """Raised when the job runner raised an unexpected error."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="UnexpectedError",
            cause=cause,
            disclose_cause=False,
        )
        logging.error(message, exc_info=cause)


class JobRunnerCrashedError(GeneralJobRunnerError):
    """Raised when the job runner crashed and the job became a zombie."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobRunnerCrashedError",
            cause=cause,
            disclose_cause=False,
        )


class JobRunnerExceededMaximumDurationError(GeneralJobRunnerError):
    """Raised when the job runner was killed because the job exceeded the maximum duration."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            code="JobRunnerExceededMaximumDurationError",
            cause=cause,
            disclose_cause=False,
        )


class ResponseAlreadyComputedError(GeneralJobRunnerError):
    """Raised when response has been already computed by another job runner."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="ResponseAlreadyComputedError",
            cause=cause,
            disclose_cause=True,
        )


class PreviousStepError(JobRunnerError):
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


class JobRunner(ABC):
    """
    Base class for job runners. A job runner is a class that processes a job, for a specific processing step.

    It cannot be instantiated directly, but must be subclassed.

    Args:
        job_info (:obj:`JobInfo`):
            The job to process. It contains the job_id, the job type, the dataset, the config, the split
            the force flag, and the priority level.
        common_config (:obj:`CommonConfig`):
            The common config.
        processing_step (:obj:`ProcessingStep`):
            The processing step to process.
    """

    job_id: str
    dataset: str
    config: Optional[str] = None
    split: Optional[str] = None
    force: bool
    priority: Priority
    worker_config: WorkerConfig
    common_config: CommonConfig
    processing_step: ProcessingStep
    _dataset_git_revision: Optional[str] = None

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
        common_config: CommonConfig,
        worker_config: WorkerConfig,
        processing_step: ProcessingStep,
    ) -> None:
        self.job_type = job_info["type"]
        self.job_id = job_info["job_id"]
        self.dataset = job_info["dataset"]
        self.config = job_info["config"]
        self.split = job_info["split"]
        self.force = job_info["force"]
        self.priority = job_info["priority"]
        self.common_config = common_config
        self.worker_config = worker_config
        self.processing_step = processing_step
        self.setup()

    def setup(self) -> None:
        job_type = self.get_job_type()
        if self.processing_step.job_type != job_type:
            raise ValueError(
                f"The processing step's job type is {self.processing_step.job_type}, but the job runner only processes"
                f" {job_type}"
            )
        if self.job_type != job_type:
            raise ValueError(
                f"The submitted job type is {self.job_type}, but the job runner only processes {job_type}"
            )

    def __str__(self) -> str:
        return (
            f"JobRunner(job_id={self.job_id} dataset={self.dataset} config={self.config}"
            + f" split={self.split} force={self.force})"
        )

    def log(self, level: int, msg: str) -> None:
        logging.log(level=level, msg=f"[{self.processing_step.job_type}] {msg}")

    def debug(self, msg: str) -> None:
        self.log(level=logging.DEBUG, msg=msg)

    def info(self, msg: str) -> None:
        self.log(level=logging.INFO, msg=msg)

    def warning(self, msg: str) -> None:
        self.log(level=logging.WARNING, msg=msg)

    def exception(self, msg: str) -> None:
        self.log(level=logging.ERROR, msg=msg)

    def critical(self, msg: str) -> None:
        self.log(level=logging.CRITICAL, msg=msg)

    def run(self) -> Literal[Status.SUCCESS, Status.ERROR, Status.SKIPPED]:
        try:
            self.info(f"compute {self}")
            result: Literal[Status.SUCCESS, Status.ERROR, Status.SKIPPED] = (
                Status.SKIPPED if self.should_skip_job() else Status.SUCCESS if self.process() else Status.ERROR
            )
        except Exception:
            self.exception(f"error while computing {self}")
            result = Status.ERROR
        self.create_children_jobs()
        return result

    def get_dataset_git_revision(self) -> Optional[str]:
        """Get the git revision of the dataset repository."""
        if self._dataset_git_revision is None:
            self._dataset_git_revision = get_dataset_git_revision(
                dataset=self.dataset, hf_endpoint=self.common_config.hf_endpoint, hf_token=self.common_config.hf_token
            )
        return self._dataset_git_revision

    # TODO: set the git revision as part of the job_info -> no need to get info from the Hub
    # if None: run the job
    def should_skip_job(self) -> bool:
        """Return True if the job should be skipped, False otherwise.

        The job must be skipped if:
        - force is False
        - and a cache entry exists for the dataset
        - and we can get the git commit and it's not None
        - and the cached entry has been created with the same git commit of the dataset repository
        - and the cached entry has been created with the same major version of the job runner
        - and the cached entry, if an error, is not among the list of errors that should trigger a retry
        - and the cached entry is complete (has a progress of 1.)

        Returns:
            :obj:`bool`: True if the job should be skipped, False otherwise.
        """
        if self.force:
            return False
        try:
            cached_response = get_response_without_content(
                kind=self.processing_step.cache_kind, dataset=self.dataset, config=self.config, split=self.split
            )
        except DoesNotExist:
            # no entry in the cache
            return False
        if cached_response["error_code"] in ERROR_CODES_TO_RETRY:
            # the cache entry result was a temporary error - we process it
            return False
        if (
            cached_response["job_runner_version"] is None
            or self.get_job_runner_version() > cached_response["job_runner_version"]
        ):
            return False
        if cached_response["progress"] is not None and cached_response["progress"] < 1.0:
            # this job is still waiting for more inputs to be complete - we should not skip it.
            # this can happen with fan-in jobs
            return False
        try:
            dataset_git_revision = self.get_dataset_git_revision()
        except Exception:
            # an exception occurred while getting the git revision from the Hub - the job will fail anyway, but we
            # process it to store the error in the cache
            return False
        return dataset_git_revision is not None and cached_response["dataset_git_revision"] == dataset_git_revision
        # skip if the git revision has not changed

    def process(
        self,
    ) -> bool:
        dataset_git_revision = None
        try:
            dataset_git_revision = self.get_dataset_git_revision()
            if dataset_git_revision is None:
                self.debug(f"the dataset={self.dataset} has no git revision, don't update the cache")
                raise NoGitRevisionError(f"Could not get git revision for dataset {self.dataset}")
            try:
                self.pre_compute()
                job_result = self.compute()
                content = job_result.content

                # Validate content size
                if len(orjson_dumps(content)) > self.worker_config.content_max_bytes:
                    raise TooBigContentError(
                        "The computed response content exceeds the supported size in bytes"
                        f" ({self.worker_config.content_max_bytes})."
                    )
            finally:
                # ensure the post_compute hook is called even if the compute raises an exception
                self.post_compute()
            upsert_response(
                kind=self.processing_step.cache_kind,
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                content=content,
                http_status=HTTPStatus.OK,
                job_runner_version=self.get_job_runner_version(),
                dataset_git_revision=dataset_git_revision,
                progress=job_result.progress,
            )
            self.debug(f"dataset={self.dataset} config={self.config} split={self.split} is valid, cache updated")
            return True
        except DatasetNotFoundError:
            # To avoid filling the cache, we don't save this error. Otherwise, DoS is possible.
            self.debug(f"the dataset={self.dataset} could not be found, don't update the cache")
            return False
        except Exception as err:
            e = err if isinstance(err, CustomError) else UnexpectedError(str(err), err)
            upsert_response(
                kind=self.processing_step.cache_kind,
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                content=dict(e.as_response()),
                http_status=e.status_code,
                error_code=e.code,
                details=dict(e.as_response_with_cause()),
                job_runner_version=self.get_job_runner_version(),
                dataset_git_revision=dataset_git_revision,
            )
            self.debug(
                f"response for dataset={self.dataset} config={self.config} split={self.split} had an error, cache"
                " updated"
            )
            return False

    def pre_compute(self) -> None:
        """Hook method called before the compute method."""
        pass

    @abstractmethod
    def compute(self) -> JobResult:
        pass

    # should be overridden if the job has children jobs of type "split"
    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute.

        Can be empty.

        Args:
            content (:obj:`Mapping[str, Any]`): the content created by the compute.
        Returns:
            :obj:`set[SplitFullName]`: the set of new splits full names.
        """
        return set()

    def create_children_jobs(self) -> None:
        """Create children jobs for the current job."""
        if len(self.processing_step.children) <= 0:
            return
        try:
            response_in_cache = get_response(
                kind=self.processing_step.cache_kind, dataset=self.dataset, config=self.config, split=self.split
            )
        except Exception:
            # if the response is not in the cache, we don't create the children jobs
            return
        if response_in_cache["http_status"] == HTTPStatus.OK:
            new_split_full_names_for_split: set[SplitFullName] = self.get_new_splits(response_in_cache["content"])
            new_split_full_names_for_config: set[SplitFullName] = {
                SplitFullName(dataset=s.dataset, config=s.config, split=None) for s in new_split_full_names_for_split
            }
        elif self.processing_step.input_type == "split":
            new_split_full_names_for_split = {
                SplitFullName(dataset=self.dataset, config=self.config, split=self.split)
            }
            new_split_full_names_for_config = {SplitFullName(dataset=self.dataset, config=self.config, split=None)}
        elif self.processing_step.input_type == "config":
            new_split_full_names_for_split = set()
            new_split_full_names_for_config = {SplitFullName(dataset=self.dataset, config=self.config, split=None)}

        else:
            new_split_full_names_for_split = set()
            new_split_full_names_for_config = set()
        new_split_full_names_for_dataset = {SplitFullName(dataset=self.dataset, config=None, split=None)}

        for processing_step in self.processing_step.children:
            new_split_full_names = (
                new_split_full_names_for_split
                if processing_step.input_type == "split"
                else new_split_full_names_for_config
                if processing_step.input_type == "config"
                else new_split_full_names_for_dataset
            )
            # compute the responses for the new splits
            queue = Queue()
            for split_full_name in new_split_full_names:
                # we force the refresh of the children step responses if the current step refresh was forced
                queue.upsert_job(
                    job_type=processing_step.job_type,
                    dataset=split_full_name.dataset,
                    config=split_full_name.config,
                    split=split_full_name.split,
                    force=self.force,
                    priority=self.priority,
                )
            logging.debug(
                f"{len(new_split_full_names)} jobs"
                f"of type {processing_step.job_type} added to queue for dataset={self.dataset}"
            )

    def post_compute(self) -> None:
        """Hook method called after the compute method."""
        pass

    def set_crashed(self, message: str, cause: Optional[BaseException] = None) -> None:
        error = JobRunnerCrashedError(message=message, cause=cause)
        upsert_response(
            kind=self.processing_step.cache_kind,
            dataset=self.dataset,
            config=self.config,
            split=self.split,
            content=dict(error.as_response()),
            http_status=error.status_code,
            error_code=error.code,
            details=dict(error.as_response_with_cause()),
            job_runner_version=self.get_job_runner_version(),
            dataset_git_revision=self.get_dataset_git_revision(),
        )
        logging.debug(
            "response for"
            f" dataset={self.dataset} config={self.config} split={self.split} had an error (crashed),"
            " cache updated"
        )

    def set_exceeded_maximum_duration(self, message: str, cause: Optional[BaseException] = None) -> None:
        error = JobRunnerExceededMaximumDurationError(message=message, cause=cause)
        upsert_response(
            kind=self.processing_step.cache_kind,
            dataset=self.dataset,
            config=self.config,
            split=self.split,
            content=dict(error.as_response()),
            http_status=error.status_code,
            error_code=error.code,
            details=dict(error.as_response_with_cause()),
            job_runner_version=self.get_job_runner_version(),
            dataset_git_revision=self.get_dataset_git_revision(),
        )
        logging.debug(
            f"response for dataset={self.dataset} config={self.config} split={self.split} had an error (exceeded"
            " maximum duration), cache updated"
        )

    def raise_if_parallel_response_exists(self, parallel_cache_kind: str, parallel_job_version: int) -> None:
        try:
            existing_response = get_response_without_content(
                kind=parallel_cache_kind, dataset=self.dataset, config=self.config, split=self.split
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
