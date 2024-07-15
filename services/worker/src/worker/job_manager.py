# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libcommon.config import CommonConfig
from libcommon.dtos import JobInfo, JobParams, JobResult, Priority
from libcommon.exceptions import (
    CustomError,
    DatasetNotFoundError,
    DatasetScriptError,
    JobManagerCrashedError,
    JobManagerExceededMaximumDurationError,
    PreviousStepStillProcessingError,
    TooBigContentError,
    UnexpectedError,
)
from libcommon.orchestrator import finish_job
from libcommon.processing_graph import processing_graph
from libcommon.simple_cache import (
    CachedArtifactError,
    CachedArtifactNotFoundError,
)
from libcommon.utils import get_duration_or_none, orjson_dumps

from worker.config import AppConfig, WorkerConfig
from worker.job_runner import JobRunner
from worker.utils import is_dataset_script_error


class JobManager:
    """
    A job manager is a class that handles a job runner compute, for a specific processing step.

    Args:
        job_info (`JobInfo`):
            The job to process. It contains the job_id, the job type, the dataset, the revision, the config,
            the split and the priority level.
        app_config (`AppConfig`):
            The app config.
        job_runner (`JobRunner`):
            The job runner to use.
    """

    job_id: str
    job_params: JobParams
    priority: Priority
    worker_config: WorkerConfig
    common_config: CommonConfig
    job_runner: JobRunner
    job_runner_version: int

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        job_runner: JobRunner,
    ) -> None:
        self.job_info = job_info
        self.job_type = job_info["type"]
        self.job_id = job_info["job_id"]
        self.priority = job_info["priority"]
        self.job_params = job_info["params"]
        self.common_config = app_config.common
        self.worker_config = app_config.worker
        self.job_runner = job_runner
        self.job_runner_version = processing_graph.get_processing_step_by_job_type(self.job_type).job_runner_version
        self.setup()

    def setup(self) -> None:
        job_type = self.job_runner.get_job_type()
        if self.job_type != job_type:
            raise ValueError(
                f"The submitted job type is {self.job_type}, but the job manager only processes {job_type}"
            )

    def __str__(self) -> str:
        return f"JobManager(job_id={self.job_id} dataset={self.job_params['dataset']} job_info={self.job_info}"

    def log(self, level: int, msg: str) -> None:
        logging.log(level=level, msg=f"[{self.job_type}] {msg}")

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
            self.job_runner.validate()
            job_result: JobResult = self.process()
        except Exception:
            job_result = {
                "job_info": self.job_info,
                "job_runner_version": self.job_runner_version,
                "is_success": False,
                "output": None,
                "duration": get_duration_or_none(self.job_info["started_at"]),
            }
        result_str = "SUCCESS" if job_result["is_success"] else "ERROR"
        self.debug(f"job output with {result_str} - {self}")
        return job_result

    def finish(self, job_result: JobResult) -> None:
        finish_job(job_result=job_result)

    def process(
        self,
    ) -> JobResult:
        self.info(f"compute {self}")
        started_at = self.job_info["started_at"]
        try:
            try:
                self.job_runner.pre_compute()
                job_result = self.job_runner.compute()
                content = job_result.content

                # Validate content size
                if len(orjson_dumps(content)) > self.worker_config.content_max_bytes:
                    raise TooBigContentError(
                        "The computed response content exceeds the supported size in bytes"
                        f" ({self.worker_config.content_max_bytes})."
                    )
            except CachedArtifactNotFoundError as err:
                raise PreviousStepStillProcessingError(
                    message="The previous steps are still being processed", cause=err
                )
            finally:
                # ensure the post_compute hook is called even if the compute raises an exception
                self.job_runner.post_compute()
            self.debug(
                f"dataset={self.job_params['dataset']} revision={self.job_params['revision']} job_info={self.job_info}"
                " is valid"
            )
            return {
                "job_info": self.job_info,
                "job_runner_version": self.job_runner_version,
                "is_success": True,
                "output": {
                    "content": content,
                    "http_status": HTTPStatus.OK,
                    "error_code": None,
                    "details": None,
                    "progress": job_result.progress,
                },
                "duration": get_duration_or_none(started_at),
            }
        except DatasetNotFoundError:
            # To avoid filling the cache, we don't save this error. Otherwise, DoS is possible.
            self.debug(f"the dataset={self.job_params['dataset']} could not be found, don't update the cache")
            return {
                "job_info": self.job_info,
                "job_runner_version": self.job_runner_version,
                "is_success": False,
                "output": None,
                "duration": get_duration_or_none(started_at),
            }
        except CachedArtifactError as err:
            # A previous step (cached artifact required by the job runner) is an error. We copy the cached entry,
            # so that users can see the underlying error (they are not interested in the internals of the graph).
            # We add an entry to details: "copied_from_artifact", with its identification details, to have a chance
            # to debug if needed.
            self.debug(f"response for job_info={self.job_info} had an error from a previous step")
            return {
                "job_info": self.job_info,
                "job_runner_version": self.job_runner_version,
                "is_success": False,
                "output": {
                    "content": err.cache_entry_with_details["content"],
                    "http_status": err.cache_entry_with_details["http_status"],
                    "error_code": err.cache_entry_with_details["error_code"],
                    "details": err.enhanced_details,
                    "progress": None,
                },
                "duration": get_duration_or_none(started_at),
            }
        except Exception as err:
            e = (
                err
                if isinstance(err, CustomError)
                else DatasetScriptError(str(err), err)
                if is_dataset_script_error()
                else UnexpectedError(str(err), err)
            )
            self.debug(f"response for job_info={self.job_info} had an error")
            return {
                "job_info": self.job_info,
                "job_runner_version": self.job_runner_version,
                "is_success": False,
                "output": {
                    "content": dict(e.as_response()),
                    "http_status": e.status_code,
                    "error_code": e.code,
                    "details": dict(e.as_response_with_cause()),
                    "progress": None,
                },
                "duration": get_duration_or_none(started_at),
            }

    def set_crashed(self, message: str, cause: Optional[BaseException] = None) -> None:
        self.info(
            "response for"
            f" dataset={self.job_params['dataset']} revision={self.job_params['revision']} job_info={self.job_info}"
            " had an error (crashed)"
        )
        error = JobManagerCrashedError(message=message, cause=cause)
        self.finish(
            job_result={
                "job_info": self.job_info,
                "job_runner_version": self.job_runner_version,
                "is_success": False,
                "output": {
                    "content": dict(error.as_response()),
                    "http_status": error.status_code,
                    "error_code": error.code,
                    "details": dict(error.as_response_with_cause()),
                    "progress": None,
                },
                "duration": get_duration_or_none(self.job_info["started_at"]),
            }
        )

    def set_exceeded_maximum_duration(self, message: str, cause: Optional[BaseException] = None) -> None:
        self.info(
            "response for"
            f" dataset={self.job_params['dataset']} revision={self.job_params['revision']} job_info={self.job_info}"
            " had an error (exceeded maximum duration)"
        )
        error = JobManagerExceededMaximumDurationError(message=message, cause=cause)
        self.finish(
            job_result={
                "job_info": self.job_info,
                "job_runner_version": self.job_runner_version,
                "is_success": False,
                "output": {
                    "content": dict(error.as_response()),
                    "http_status": error.status_code,
                    "error_code": error.code,
                    "details": dict(error.as_response_with_cause()),
                    "progress": None,
                },
                "duration": get_duration_or_none(self.job_info["started_at"]),
            }
        )
