# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, Literal, Mapping, Optional

from packaging import version

from libcommon.config import CommonConfig
from libcommon.dataset import DatasetNotFoundError, get_dataset_git_revision
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import JobInfo, Status
from libcommon.simple_cache import get_response_without_content, upsert_response


def parse_version(string_version: str) -> version.Version:
    parsed_version = version.parse(string_version)
    if isinstance(parsed_version, version.LegacyVersion):
        raise ValueError(f"LegacyVersion is not supported: {parsed_version}")
    return parsed_version


WorkerErrorCode = Literal[
    "ConfigNotFoundError",
    "NoGitRevisionError",
    "SplitNotFoundError",
    "UnexpectedError",
]


class WorkerError(CustomError):
    """Base class for worker exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: WorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=str(code), cause=cause, disclose_cause=disclose_cause
        )


class ConfigNotFoundError(WorkerError):
    """Raised when the config does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="ConfigNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class SplitNotFoundError(WorkerError):
    """Raised when the split does not exist."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="SplitNotFoundError",
            cause=cause,
            disclose_cause=False,
        )


class NoGitRevisionError(WorkerError):
    """Raised when the git revision returned by huggingface_hub is None."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.NOT_FOUND,
            code="NoGitRevisionError",
            cause=cause,
            disclose_cause=False,
        )


class UnexpectedError(WorkerError):
    """Raised when the response for the split has not been found."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="UnexpectedError",
            cause=cause,
            disclose_cause=False,
        )


class Worker(ABC):
    """
    Base class for workers. A worker is a class that processes a job, for a specific processing step.

    It cannot be instantiated directly, but must be subclassed.

    Args:
        job_info (:obj:`JobInfo`):
            The job to process. It contains the job_id, the job type, the dataset, the config, the split
            and the force flag.
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
        processing_step: ProcessingStep,
    ) -> None:
        self.job_type = job_info["type"]
        self.job_id = job_info["job_id"]
        self.dataset = job_info["dataset"]
        self.config = job_info["config"]
        self.split = job_info["split"]
        self.force = job_info["force"]
        self.common_config = common_config
        self.processing_step = processing_step
        self.setup()

    def setup(self) -> None:
        worker_job_type = self.get_job_type()
        if self.processing_step.job_type != worker_job_type:
            raise ValueError(
                f"The processing step's job type is {self.processing_step.job_type}, but the worker only processes"
                f" {worker_job_type}"
            )
        if self.job_type != worker_job_type:
            raise ValueError(
                f"The submitted job type is {self.job_type}, but the worker only processes {worker_job_type}"
            )

    def __str__(self):
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

    def critical(self, msg: str) -> None:
        self.log(level=logging.CRITICAL, msg=msg)

    def exception(self, msg: str) -> None:
        self.log(level=logging.ERROR, msg=msg)

    def run(self) -> Literal[Status.SUCCESS, Status.ERROR, Status.SKIPPED]:
        try:
            self.info(f"compute {self}")
            if self.should_skip_job():
                return Status.SKIPPED
            elif self.process():
                return Status.SUCCESS
            else:
                return Status.ERROR
        except Exception:
            self.exception(f"error while computing {self}")
            return Status.ERROR

    def compare_major_version(self, other_version: str) -> int:
        """
        Compare the major version of worker's self version and the other version's.

        Args:
            other_version (:obj:`str`): the other semantic version

        Returns:
            :obj:`int`: the difference between the major version of both versions.
            0 if they are equal. Negative if worker's major version is lower than other_version, positive otherwise.
        Raises:
            :obj:`ValueError`: if worker's version or other_version is not a valid semantic version.
        """
        try:
            return parse_version(self.get_version()).major - parse_version(other_version).major
        except Exception as err:
            raise RuntimeError(f"Could not get major versions: {err}") from err

    def get_dataset_git_revision(self) -> Optional[str]:
        """Get the git revision of the dataset repository."""
        return get_dataset_git_revision(
            dataset=self.dataset, hf_endpoint=self.common_config.hf_endpoint, hf_token=self.common_config.hf_token
        )

    def should_skip_job(self) -> bool:
        """Return True if the job should be skipped, False otherwise.

        The job must be skipped if:
        - force is False
        - and a cache entry exists for the dataset
        - and the result was successful
        - and it has been created with the same major version of the worker
        - and it has been created with the exact same git commit of the dataset repository

        Returns:
            :obj:`bool`: True if the job should be skipped, False otherwise.
        """
        if self.force:
            return False
        try:
            cached_response = get_response_without_content(
                kind=self.processing_step.cache_kind, dataset=self.dataset, config=self.config, split=self.split
            )
            dataset_git_revision = self.get_dataset_git_revision()
            return (
                # TODO: use "error_code" to decide if the job should be skipped (ex: retry if temporary error)
                cached_response["http_status"] == HTTPStatus.OK
                and cached_response["worker_version"] is not None
                and self.compare_major_version(cached_response["worker_version"]) == 0
                and cached_response["dataset_git_revision"] is not None
                and cached_response["dataset_git_revision"] == dataset_git_revision
            )
        except Exception:
            return False

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

    def post_compute(self) -> None:
        """Hook method called after the compute method."""
        pass


class WorkerFactory(ABC):
    """
    Base class for worker factories. A worker factory is a class that creates a worker.

    It cannot be instantiated directly, but must be subclassed.
    """

    def create_worker(self, job_info: JobInfo) -> Worker:
        return self._create_worker(job_info=job_info)

    @abstractmethod
    def _create_worker(self, job_info: JobInfo) -> Worker:
        pass
