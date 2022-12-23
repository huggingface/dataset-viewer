# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass
from http import HTTPStatus

from libcommon.exceptions import CustomError
from libcommon.simple_cache import DoesNotExist, get_response
from libcommon.worker import JobInfo, Worker
from typing import List, Literal, Optional

from parquet_based.config import AppConfig, ParquetBasedConfig


ParquetBasedWorkerErrorCode = Literal[
    "ErroneousCachedParquetFilesError", "MalformedCachedParquetFilesError", "NoCachedParquetFilesError"
]


class ParquetBasedWorkerError(CustomError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ParquetBasedWorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(message, status_code, str(code), cause, disclose_cause)


class NoCachedParquetFilesError(ParquetBasedWorkerError):
    """Raised when no cached response for the parquet files could be found."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "NoCachedParquetFilesError", cause, True)


class ErroneousCachedParquetFilesError(ParquetBasedWorkerError):
    """Raised when the cached response for the parquet files is erroneous."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.NOT_FOUND, "ErroneousCachedParquetFilesError", cause, True)


class MalformedCachedParquetFilesError(ParquetBasedWorkerError):
    """Raised when the cached response for the parquet files is malformed."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "MalformedCachedParquetFilesError", cause, False)


@dataclass
class ParquetFile:
    dataset: str
    config: str
    split: str
    url: str
    filename: str
    size: int


class ParquetBasedWorker(Worker):
    """Base class for workers that use parquet."""

    parquet_based_config: ParquetBasedConfig
    parquet_files: List[ParquetFile] = []

    def __init__(self, job_info: JobInfo, app_config: AppConfig) -> None:
        job_type = job_info["type"]
        try:
            processing_step = app_config.processing_graph.graph.get_step_by_job_type(job_type)
        except ValueError as e:
            raise ValueError(
                f"Unsupported job type: '{job_type}'. The job types declared in the processing graph are:"
                f" {[step.job_type for step in app_config.processing_graph.graph.steps.values()]}"
            ) from e
        super().__init__(job_info=job_info, common_config=app_config.common, processing_step=processing_step)
        self.parquet_based_config = app_config.parquet_based

    def load_parquet_files(self) -> None:
        try:
            response = get_response(kind="/parquet", dataset=self.dataset, config=self.config, split=self.split)
        except DoesNotExist as e:
            raise NoCachedParquetFilesError(
                f"Parquet files for dataset '{self.dataset}', config '{self.config}', split '{self.split}' were not"
                " found in the cache."
            ) from e
        if response["http_status"] != HTTPStatus.OK:
            raise ErroneousCachedParquetFilesError(
                f"An error was raised when computing the parquet files for the dataset '{self.dataset}', config"
                f" '{self.config}', split '{self.split}'. Cannot compute the size of the dataset."
            )
        try:
            self.parquet_files = [
                ParquetFile(
                    dataset=parquet_file["dataset"],
                    config=parquet_file["config"],
                    split=parquet_file["split"],
                    url=parquet_file["url"],
                    filename=parquet_file["filename"],
                    size=parquet_file["size"],
                )
                for parquet_file in response["content"]["parquet_files"]
            ]
        except Exception as e:
            raise MalformedCachedParquetFilesError(
                f"The cached response for the parquet files for the dataset '{self.dataset}', config '{self.config}',"
                f" split '{self.split}' is malformed."
            ) from e

    def pre_compute(self) -> None:
        self.load_parquet_files()

    def post_compute(self) -> None:
        pass
