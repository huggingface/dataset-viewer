# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Literal, Optional

from libcommon.config import CommonConfig
from libcommon.dataset import DatasetNotFoundError, get_dataset_git_revision
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue
from libcommon.simple_cache import (
    DoesNotExist,
    SplitFullName,
    get_response_params,
    get_response_without_content_params,
    upsert_response_params,
)
from libcommon.utils import JobInfo, JobParams, Priority, Status, orjson_dumps

from worker.common_exceptions import (
    JobManagerCrashedError,
    JobManagerExceededMaximumDurationError,
    NoGitRevisionError,
    ResponseAlreadyComputedError,
    TooBigContentError,
    UnexpectedError,
)
from worker.config import AppConfig, WorkerConfig
from worker.job_runner import JobRunner

# List of error codes that should trigger a retry.
ERROR_CODES_TO_RETRY: list[str] = ["ClientConnectionError"]


class JobManager:
    """
    A job manager is a class that handles a job runner compute, for a specific processing step.

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
    job_params: JobParams
    force: bool
    priority: Priority
    worker_config: WorkerConfig
    common_config: CommonConfig
    processing_step: ProcessingStep
    _dataset_git_revision: Optional[str] = None
    job_runner: JobRunner

    def __init__(self, job_info: JobInfo, app_config: AppConfig, job_runner: JobRunner) -> None:
        self.job_info = job_info
        self.job_type = job_info["type"]
        self.job_id = job_info["job_id"]
        self.force = job_info["force"]
        self.priority = job_info["priority"]
        self.job_params = job_info["params"]
        self.dataset = self.job_params["dataset"]
        self.common_config = app_config.common
        self.worker_config = app_config.worker
        self.job_runner = job_runner
        self.processing_step = self.job_runner.processing_step
        self.setup()

    def setup(self) -> None:
        job_type = self.job_runner.get_job_type()
        if self.processing_step.job_type != job_type:
            raise ValueError(
                f"The processing step's job type is {self.processing_step.job_type}, but the job manager only"
                f" processes {job_type}"
            )
        if self.job_type != job_type:
            raise ValueError(
                f"The submitted job type is {self.job_type}, but the job manager only processes {job_type}"
            )

    def __str__(self) -> str:
        return f"JobManager(job_id={self.job_id} dataset={self.dataset} job_info={self.job_info}"

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
            cached_response = get_response_without_content_params(
                kind=self.processing_step.cache_kind,
                job_params=self.job_info["params"],
            )
        except DoesNotExist:
            # no entry in the cache
            return False
        if cached_response["error_code"] in ERROR_CODES_TO_RETRY:
            # the cache entry result was a temporary error - we process it
            return False
        if (
            cached_response["job_runner_version"] is None
            or self.job_runner.get_job_runner_version() > cached_response["job_runner_version"]
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

    def raise_if_parallel_response_exists(self, parallel_cache_kind: str, parallel_job_version: int) -> None:
        try:
            existing_response = get_response_without_content_params(
                kind=parallel_cache_kind,
                job_params=self.job_info["params"],
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
                self.job_runner.pre_compute()
                parallel_job_runner = self.job_runner.get_parallel_job_runner()
                if parallel_job_runner:
                    self.raise_if_parallel_response_exists(
                        parallel_cache_kind=parallel_job_runner["job_type"],
                        parallel_job_version=parallel_job_runner["job_runner_version"],
                    )

                job_result = self.job_runner.compute()
                content = job_result.content

                # Validate content size
                if len(orjson_dumps(content)) > self.worker_config.content_max_bytes:
                    raise TooBigContentError(
                        "The computed response content exceeds the supported size in bytes"
                        f" ({self.worker_config.content_max_bytes})."
                    )
            finally:
                # ensure the post_compute hook is called even if the compute raises an exception
                self.job_runner.post_compute()
            upsert_response_params(
                kind=self.processing_step.cache_kind,
                dataset=self.dataset,
                job_params=self.job_info["params"],
                content=content,
                http_status=HTTPStatus.OK,
                job_runner_version=self.job_runner.get_job_runner_version(),
                dataset_git_revision=dataset_git_revision,
                progress=job_result.progress,
            )
            self.debug(f"dataset={self.dataset} job_info={self.job_info} is valid, cache updated")
            return True
        except DatasetNotFoundError:
            # To avoid filling the cache, we don't save this error. Otherwise, DoS is possible.
            self.debug(f"the dataset={self.dataset} could not be found, don't update the cache")
            return False
        except Exception as err:
            e = err if isinstance(err, CustomError) else UnexpectedError(str(err), err)
            upsert_response_params(
                kind=self.processing_step.cache_kind,
                dataset=self.dataset,
                job_params=self.job_info["params"],
                content=dict(e.as_response()),
                http_status=e.status_code,
                error_code=e.code,
                details=dict(e.as_response_with_cause()),
                job_runner_version=self.job_runner.get_job_runner_version(),
                dataset_git_revision=dataset_git_revision,
            )
            self.debug(f"response for dataset={self.dataset} job_info={self.job_info} had an error, cache updated")
            return False

    def create_children_jobs(self) -> None:
        """Create children jobs for the current job."""
        if len(self.processing_step.children) <= 0:
            return
        try:
            response_in_cache = get_response_params(
                kind=self.processing_step.cache_kind, dataset=self.dataset, job_params=self.job_info["params"]
            )
        except Exception:
            # if the response is not in the cache, we don't create the children jobs
            return
        if response_in_cache["http_status"] == HTTPStatus.OK:
            new_split_full_names_for_split: set[SplitFullName] = self.job_runner.get_new_splits(
                response_in_cache["content"]
            )
            new_split_full_names_for_config: set[SplitFullName] = {
                SplitFullName(dataset=s.dataset, config=s.config, split=None) for s in new_split_full_names_for_split
            }
        # TODO (Andrea): Change the way it works without to depend on specific fields,
        # maybe operator can return the list of children
        elif self.processing_step.input_type == "split":
            new_split_full_names_for_split = {
                SplitFullName(
                    dataset=self.dataset,
                    config=self.job_info["params"]["config"],
                    split=self.job_info["params"]["split"],
                )
            }
            new_split_full_names_for_config = {
                SplitFullName(dataset=self.dataset, config=self.job_info["params"]["config"], split=None)
            }
        elif self.processing_step.input_type == "config":
            new_split_full_names_for_split = set()
            new_split_full_names_for_config = {
                SplitFullName(dataset=self.dataset, config=self.job_info["params"]["config"], split=None)
            }

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

    def set_crashed(self, message: str, cause: Optional[BaseException] = None) -> None:
        error = JobManagerCrashedError(message=message, cause=cause)
        upsert_response_params(
            kind=self.processing_step.cache_kind,
            dataset=self.dataset,
            job_params=self.job_info["params"],
            content=dict(error.as_response()),
            http_status=error.status_code,
            error_code=error.code,
            details=dict(error.as_response_with_cause()),
            job_runner_version=self.job_runner.get_job_runner_version(),
            dataset_git_revision=self.get_dataset_git_revision(),
        )
        logging.debug(
            f"response for dataset={self.dataset} job_info={self.job_info} had an error (crashed), cache updated"
        )

    def set_exceeded_maximum_duration(self, message: str, cause: Optional[BaseException] = None) -> None:
        error = JobManagerExceededMaximumDurationError(message=message, cause=cause)
        upsert_response_params(
            kind=self.processing_step.cache_kind,
            dataset=self.dataset,
            job_params=self.job_info["params"],
            content=dict(error.as_response()),
            http_status=error.status_code,
            error_code=error.code,
            details=dict(error.as_response_with_cause()),
            job_runner_version=self.job_runner.get_job_runner_version(),
            dataset_git_revision=self.get_dataset_git_revision(),
        )
        logging.debug(
            f"response for dataset={self.dataset} job_info={self.job_info} had an error (exceeded"
            " maximum duration), cache updated"
        )
