# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import random
import time
from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, Literal, Mapping, Optional

from packaging import version
from psutil import cpu_count, getloadavg, swap_memory, virtual_memory

from libcommon.config import CommonConfig, QueueConfig, WorkerConfig
from libcommon.dataset import DatasetNotFoundError, get_dataset_git_revision
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import EmptyQueueError, Queue, Status
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
    processing_step: ProcessingStep
    queue: Queue
    common_config: CommonConfig
    queue_config: QueueConfig
    worker_config: WorkerConfig
    version: str

    def __init__(
        self,
        processing_step: ProcessingStep,
        common_config: CommonConfig,
        queue_config: QueueConfig,
        worker_config: WorkerConfig,
        version: str,
    ) -> None:
        self.processing_step = processing_step
        self.common_config = common_config
        self.queue_config = queue_config
        self.worker_config = worker_config
        self.version = version
        self.setup()

    def setup(self) -> None:
        self.queue = Queue(
            type=self.processing_step.job_type, max_jobs_per_namespace=self.queue_config.max_jobs_per_namespace
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

    def has_memory(self) -> bool:
        if self.worker_config.max_memory_pct <= 0:
            return True
        virtual_memory_used: int = virtual_memory().used  # type: ignore
        virtual_memory_total: int = virtual_memory().total  # type: ignore
        percent = (swap_memory().used + virtual_memory_used) / (swap_memory().total + virtual_memory_total)
        ok = percent < self.worker_config.max_memory_pct
        if not ok:
            self.info(
                f"memory usage (RAM + SWAP) is too high: {percent:.0f}% - max is {self.worker_config.max_memory_pct}%"
            )
        return ok

    def has_cpu(self) -> bool:
        if self.worker_config.max_load_pct <= 0:
            return True
        load_pct = max(getloadavg()[:2]) / cpu_count() * 100
        # ^ only current load and 5m load. 15m load is not relevant to decide to launch a new job
        ok = load_pct < self.worker_config.max_load_pct
        if not ok:
            self.info(f"cpu load is too high: {load_pct:.0f}% - max is {self.worker_config.max_load_pct}%")
        return ok

    def sleep(self) -> None:
        jitter = 0.75 + random.random() / 2  # nosec
        # ^ between 0.75 and 1.25
        duration = self.worker_config.sleep_seconds * jitter
        self.debug(f"sleep during {duration:.2f} seconds")
        time.sleep(duration)

    def loop(self) -> None:
        try:
            while True:
                if self.has_memory() and self.has_cpu() and self.process_next_job():
                    # loop immediately to try another job
                    # see https://github.com/huggingface/datasets-server/issues/265
                    continue
                self.sleep()
        except BaseException as e:
            self.critical(f"quit due to an uncaught error while processing the job: {e}")
            raise

    def process_next_job(self) -> bool:
        self.debug("try to process a job")

        try:
            started_job_info = self.queue.start_job()
            job_id = started_job_info["job_id"]
            dataset = started_job_info["dataset"]
            config = started_job_info["config"]
            split = started_job_info["split"]
            force = started_job_info["force"]
            parameters_for_log = "dataset={dataset}" + ("" if split is None else f"config={config} split={split}")
            self.debug(f"job assigned: {job_id} for {parameters_for_log}")
        except EmptyQueueError:
            self.debug("no job in the queue")
            return False

        finished_status: Literal[Status.SUCCESS, Status.ERROR, Status.SKIPPED]
        try:
            self.info(f"compute {parameters_for_log}")
            if self.should_skip_job(dataset=dataset, config=config, split=split, force=force):
                finished_status = Status.SKIPPED
            else:
                self.process(dataset=dataset, config=config, split=split, force=force)
                finished_status = Status.SUCCESS
        except Exception:
            self.exception(f"error while computing {parameters_for_log}")
            finished_status = Status.ERROR
        finally:
            self.queue.finish_job(job_id=job_id, finished_status=finished_status)
            self.debug(f"job finished with {finished_status.value}: {job_id} for {parameters_for_log}")
        return True

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
            return parse_version(self.version).major - parse_version(other_version).major
        except Exception as err:
            raise RuntimeError(f"Could not get major versions: {err}") from err

    def should_skip_job(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = False
    ) -> bool:
        """Return True if the job should be skipped, False otherwise.

        The job must be skipped if:
        - force is False
        - and a cache entry exists for the dataset
        - and the result was successful
        - and it has been created with the same major version of the worker
        - and it has been created with the exact same git commit of the dataset repository

        Args:
            dataset (:obj:`str`): The name of the dataset.
            config (:obj:`str`, `optional`): The name of the configuration.
            split (:obj:`str`, `optional`): The name of the split.
            force (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to force the job to be run.

        Returns:
            :obj:`bool`: True if the job should be skipped, False otherwise.
        """
        if force or config is None or split is None:
            return False
        try:
            cached_response = get_response_without_content(
                kind=self.processing_step.cache_kind, dataset=dataset, config=config, split=split
            )
            dataset_git_revision = get_dataset_git_revision(
                dataset=dataset, hf_endpoint=self.common_config.hf_endpoint, hf_token=self.common_config.hf_token
            )
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
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        dataset_git_revision = None
        try:
            dataset_git_revision = get_dataset_git_revision(
                dataset=dataset, hf_endpoint=self.common_config.hf_endpoint, hf_token=self.common_config.hf_token
            )
            if dataset_git_revision is None:
                self.debug(f"the dataset={dataset} has no git revision, don't update the cache")
                raise NoGitRevisionError(f"Could not get git revision for dataset {dataset}")
            content = self.compute(dataset=dataset, config=config, split=split, force=force)
            upsert_response(
                kind=self.processing_step.cache_kind,
                dataset=dataset,
                config=config,
                split=split,
                content=content,
                http_status=HTTPStatus.OK,
                worker_version=self.version,
                dataset_git_revision=dataset_git_revision,
            )
            self.debug(f"dataset={dataset} config={config} split={split} is valid, cache updated")
            return True
        except (
            DatasetNotFoundError,
            ConfigNotFoundError,
            SplitNotFoundError,
        ):
            # To avoid filling the cache, we don't save these errors. Otherwise, DoS is possible.
            self.debug(
                f"the dataset={dataset}, config {config} or split {split} could not be found, don't update the cache"
            )
            return False
        except Exception as err:
            e = err if isinstance(err, CustomError) else UnexpectedError(str(err), err)
            upsert_response(
                kind=self.processing_step.cache_kind,
                dataset=dataset,
                config=config,
                split=split,
                content=dict(e.as_response()),
                http_status=e.status_code,
                error_code=e.code,
                details=dict(e.as_response_with_cause()),
                worker_version=self.version,
                dataset_git_revision=dataset_git_revision,
            )
            self.debug(f"response for dataset={dataset} config={config} split={split} had an error, cache updated")
            return False

    @abstractmethod
    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        force: bool = False,
    ) -> Mapping[str, Any]:
        pass
