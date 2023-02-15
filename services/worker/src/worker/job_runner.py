# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, Literal, Mapping, Optional

from libcommon.config import CommonConfig
from libcommon.dataset import DatasetNotFoundError, get_dataset_git_revision
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo, Priority, Queue, Status
from libcommon.simple_cache import (
    DoesNotExist,
    SplitFullName,
    delete_response,
    get_response,
    get_response_without_content,
    get_split_full_names_for_dataset_and_kind,
    upsert_response,
)
from libcommon.utils import orjson_dumps
from packaging import version

from worker.config import WorkerConfig

GeneralJobRunnerErrorCode = Literal[
    "ConfigNotFoundError",
    "NoGitRevisionError",
    "SplitNotFoundError",
    "UnexpectedError",
    "TooBigContentError",
]

# List of error codes that should trigger a retry.
ERROR_CODES_TO_RETRY: list[str] = ["ClientConnectionError"]


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


class ConfigNotFoundError(GeneralJobRunnerError):
    """Raised when the config does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="ConfigNotFoundError",
            cause=cause,
            disclose_cause=False,
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

    @staticmethod
    @abstractmethod
    def get_job_type() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_version() -> str:
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
        logging.log(level=level, msg=f"[{self.processing_step.endpoint}] {msg}")

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

    def compare_major_version(self, other_version: str) -> int:
        """
        Compare the major version of job runner's self version and the other version's.

        Args:
            other_version (:obj:`str`): the other semantic version

        Returns:
            :obj:`int`: the difference between the major version of both versions.
            0 if they are equal. Negative if job runner's major version is lower than other_version, positive
              otherwise.
        Raises:
            :obj:`ValueError`: if job runner's version or other_version is not a valid semantic version.
        """
        try:
            return version.parse(self.get_version()).major - version.parse(other_version).major
        except Exception as err:
            raise RuntimeError(f"Could not get major versions: {err}") from err

    def get_dataset_git_revision(self) -> Optional[str]:
        """Get the git revision of the dataset repository."""
        return get_dataset_git_revision(
            dataset=self.dataset, hf_endpoint=self.common_config.hf_endpoint, hf_token=self.common_config.hf_token
        )

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
            cached_response["worker_version"] is None
            or self.compare_major_version(cached_response["worker_version"]) != 0
        ):
            # no job runner version in the cache, or the job runner has been updated - we process the job to update
            # the cache
            # note: the collection field is named "worker_version" for historical reasons, it might be renamed
            #   "job_runner_version" in the future.
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
                content = self.compute()

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
                worker_version=self.get_version(),
                dataset_git_revision=dataset_git_revision,
            )
            self.debug(f"dataset={self.dataset} config={self.config} split={self.split} is valid, cache updated")
            return True
        except (
            DatasetNotFoundError,
            ConfigNotFoundError,
            SplitNotFoundError,
        ):
            # To avoid filling the cache, we don't save these errors. Otherwise, DoS is possible.
            self.debug(
                f"the dataset={self.dataset}, config {self.config} or split {self.split} could not be found, don't"
                " update the cache"
            )
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
                worker_version=self.get_version(),
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
    def compute(self) -> Mapping[str, Any]:
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
        if response_in_cache["http_status"] != HTTPStatus.OK:
            # if the response is not valid, we don't create the children jobs
            return
        new_split_full_names_for_split: set[SplitFullName] = self.get_new_splits(response_in_cache["content"])
        new_split_full_names_for_config: set[SplitFullName] = {
            SplitFullName(dataset=s.dataset, config=s.config, split=None) for s in new_split_full_names_for_split
        }
        new_split_full_names_for_dataset: set[SplitFullName] = {
            SplitFullName(dataset=s.dataset, config=None, split=None) for s in new_split_full_names_for_config
        }  # should be self.dataset
        for processing_step in self.processing_step.children:
            new_split_full_names = (
                new_split_full_names_for_split
                if processing_step.input_type == "split"
                else new_split_full_names_for_config
                if processing_step.input_type == "config"
                else new_split_full_names_for_dataset
            )
            # remove obsolete responses from the cache
            split_full_names_in_cache = get_split_full_names_for_dataset_and_kind(
                dataset=self.dataset, kind=processing_step.cache_kind
            )
            split_full_names_to_delete = split_full_names_in_cache.difference(new_split_full_names)
            for split_full_name in split_full_names_to_delete:
                delete_response(
                    kind=processing_step.cache_kind,
                    dataset=split_full_name.dataset,
                    config=split_full_name.config,
                    split=split_full_name.split,
                )
            logging.debug(
                f"{len(split_full_names_to_delete)} obsolete responses"
                f"of kind {processing_step.cache_kind} deleted from cache for dataset={self.dataset}"
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
