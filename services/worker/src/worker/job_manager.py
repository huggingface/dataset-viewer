# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, Mapping, Optional, TypedDict

from libcommon.config import CommonConfig
from libcommon.exceptions import (
    CustomError,
    DatasetNotFoundError,
    JobManagerCrashedError,
    JobManagerExceededMaximumDurationError,
    ResponseAlreadyComputedError,
    TooBigContentError,
    UnexpectedError,
)
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Queue
from libcommon.simple_cache import (
    CachedArtifactError,
    DoesNotExist,
    get_response_without_content_params,
    upsert_response_params,
)
from libcommon.state import DatasetState
from libcommon.utils import JobInfo, JobParams, Priority, orjson_dumps

from worker.config import AppConfig, WorkerConfig
from worker.job_runner import JobRunner

# List of error codes that should trigger a retry.
ERROR_CODES_TO_RETRY: list[str] = ["ClientConnectionError"]


class JobOutput(TypedDict):
    content: Mapping[str, Any]
    http_status: HTTPStatus
    error_code: Optional[str]
    details: Optional[Mapping[str, Any]]
    progress: Optional[float]


class JobResult(TypedDict):
    is_success: bool
    output: Optional[JobOutput]


class JobManager:
    """
    A job manager is a class that handles a job runner compute, for a specific processing step.

    Args:
        job_info (:obj:`JobInfo`):
            The job to process. It contains the job_id, the job type, the dataset, the revision, the config,
            the split and the priority level.
        common_config (:obj:`CommonConfig`):
            The common config.
        processing_step (:obj:`ProcessingStep`):
            The processing step to process.
    """

    job_id: str
    job_params: JobParams
    priority: Priority
    worker_config: WorkerConfig
    common_config: CommonConfig
    processing_step: ProcessingStep
    processing_graph: ProcessingGraph
    job_runner: JobRunner

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        job_runner: JobRunner,
        processing_graph: ProcessingGraph,
    ) -> None:
        self.job_info = job_info
        self.job_type = job_info["type"]
        self.job_id = job_info["job_id"]
        self.priority = job_info["priority"]
        self.job_params = job_info["params"]
        self.common_config = app_config.common
        self.worker_config = app_config.worker
        self.job_runner = job_runner
        self.processing_graph = processing_graph
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
        return f"JobManager(job_id={self.job_id} dataset={self.job_params['dataset']} job_info={self.job_info}"

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

    def run_job(self) -> JobResult:
        try:
            job_result: JobResult = self.process()
        except Exception:
            job_result = {
                "is_success": False,
                "output": None,
            }
        result_str = "SUCCESS" if job_result["is_success"] else "ERROR"
        self.debug(f"job output with {result_str} - {self}")
        return job_result

    def finish(self, job_result: JobResult) -> None:
        # check if the job is still in started status
        # if not, it means that the job was cancelled, and we don't want to update the cache
        job_was_valid = Queue().finish_job(
            job_id=self.job_id,
            is_success=job_result["is_success"],
        )
        if job_was_valid and job_result["output"]:
            self.set_cache(job_result["output"])
            logging.debug("the job output has been written to the cache.")
            self.backfill()
            logging.debug("the dataset has been backfilled.")
        else:
            logging.debug("the job output has not been written to the cache, and the dataset has not been backfilled.")

    def raise_if_parallel_response_exists(self, parallel_cache_kind: str, parallel_job_version: int) -> None:
        try:
            existing_response = get_response_without_content_params(
                kind=parallel_cache_kind,
                job_params=self.job_params,
            )

            if (
                existing_response["http_status"] == HTTPStatus.OK
                and existing_response["job_runner_version"] == parallel_job_version
                and existing_response["progress"] == 1.0  # completed response
                and existing_response["dataset_git_revision"] == self.job_params["revision"]
            ):
                raise ResponseAlreadyComputedError(
                    f"Response has already been computed and stored in cache kind: {parallel_cache_kind}. Compute will"
                    " be skipped."
                )
        except DoesNotExist:
            logging.debug(f"no cache found for {parallel_cache_kind}.")

    def process(
        self,
    ) -> JobResult:
        self.info(f"compute {self}")
        try:
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
            self.debug(
                f"dataset={self.job_params['dataset']} revision={self.job_params['revision']} job_info={self.job_info}"
                " is valid"
            )
            return {
                "is_success": True,
                "output": {
                    "content": content,
                    "http_status": HTTPStatus.OK,
                    "error_code": None,
                    "details": None,
                    "progress": job_result.progress,
                },
            }
        except DatasetNotFoundError:
            # To avoid filling the cache, we don't save this error. Otherwise, DoS is possible.
            self.debug(f"the dataset={self.job_params['dataset']} could not be found, don't update the cache")
            return {"is_success": False, "output": None}
        except CachedArtifactError as err:
            # A previous step (cached artifact required by the job runner) is an error. We copy the cached entry,
            # so that users can see the underlying error (they are not interested in the internals of the graph).
            # We add an entry to details: "copied_from_artifact", with its identification details, to have a chance
            # to debug if needed.
            self.debug(f"response for job_info={self.job_info} had an error from a previous step")
            return {
                "is_success": False,
                "output": {
                    "content": err.cache_entry_with_details["content"],
                    "http_status": err.cache_entry_with_details["http_status"],
                    "error_code": err.cache_entry_with_details["error_code"],
                    "details": err.enhanced_details,
                    "progress": None,
                },
            }
        except Exception as err:
            e = err if isinstance(err, CustomError) else UnexpectedError(str(err), err)
            self.debug(f"response for job_info={self.job_info} had an error")
            return {
                "is_success": False,
                "output": {
                    "content": dict(e.as_response()),
                    "http_status": e.status_code,
                    "error_code": e.code,
                    "details": dict(e.as_response_with_cause()),
                    "progress": None,
                },
            }

    def backfill(self) -> None:
        """Evaluate the state of the dataset and backfill the cache if necessary."""
        DatasetState(
            dataset=self.job_params["dataset"],
            revision=self.job_params["revision"],
            processing_graph=self.processing_graph,
            error_codes_to_retry=ERROR_CODES_TO_RETRY,
            priority=self.priority,
        ).backfill()

    def set_cache(self, output: JobOutput) -> None:
        upsert_response_params(
            # inputs
            kind=self.processing_step.cache_kind,
            job_params=self.job_params,
            job_runner_version=self.job_runner.get_job_runner_version(),
            # output
            content=output["content"],
            http_status=output["http_status"],
            error_code=output["error_code"],
            details=output["details"],
            progress=output["progress"],
        )

    def set_crashed(self, message: str, cause: Optional[BaseException] = None) -> None:
        self.debug(
            "response for"
            f" dataset={self.job_params['dataset']} revision={self.job_params['revision']} job_info={self.job_info}"
            " had an error (crashed)"
        )
        error = JobManagerCrashedError(message=message, cause=cause)
        self.finish(
            job_result={
                "is_success": False,
                "output": {
                    "content": dict(error.as_response()),
                    "http_status": error.status_code,
                    "error_code": error.code,
                    "details": dict(error.as_response_with_cause()),
                    "progress": None,
                },
            }
        )

    def set_exceeded_maximum_duration(self, message: str, cause: Optional[BaseException] = None) -> None:
        self.debug(
            "response for"
            f" dataset={self.job_params['dataset']} revision={self.job_params['revision']} job_info={self.job_info}"
            " had an error (exceeded maximum duration)"
        )
        error = JobManagerExceededMaximumDurationError(message=message, cause=cause)
        self.finish(
            job_result={
                "is_success": False,
                "output": {
                    "content": dict(error.as_response()),
                    "http_status": error.status_code,
                    "error_code": error.code,
                    "details": dict(error.as_response_with_cause()),
                    "progress": None,
                },
            }
        )
