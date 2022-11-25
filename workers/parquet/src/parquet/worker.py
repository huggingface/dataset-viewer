# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import importlib.metadata
import logging
from http import HTTPStatus
from typing import Optional

from libcommon.queue import Queue
from libcommon.simple_cache import get_response_without_content, upsert_response
from libcommon.worker import Worker

from parquet.config import WorkerConfig
from parquet.response import compute_parquet_response, get_dataset_git_revision
from parquet.utils import (
    CacheKind,
    DatasetNotFoundError,
    JobType,
    UnexpectedError,
    WorkerCustomError,
)


class ParquetWorker(Worker):
    config: WorkerConfig

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(queue_config=worker_config.queue, version=importlib.metadata.version(__package__))
        self._queue = Queue(
            type=JobType.PARQUET.value, max_jobs_per_namespace=worker_config.queue.max_jobs_per_namespace
        )
        self.config = worker_config

    @property
    def queue(self):
        return self._queue

    def should_skip_job(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = None
    ) -> bool:
        """Return True if the job should be skipped, False otherwise.

        The job must be skipped if:
        - a cache entry exists for the dataset
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
        if force:
            return False
        try:
            cached_response = get_response_without_content(kind=CacheKind.SPLITS.value, dataset=dataset)
            dataset_git_revision = get_dataset_git_revision(
                dataset=dataset, hf_endpoint=self.config.common.hf_endpoint, hf_token=self.config.parquet.hf_token
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

    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        dataset_git_revision = None
        try:
            dataset_git_revision = get_dataset_git_revision(
                dataset=dataset, hf_endpoint=self.config.common.hf_endpoint, hf_token=self.config.parquet.hf_token
            )
        except DatasetNotFoundError:
            logging.debug(f"the dataset={dataset} could not be found, don't update the cache")
            return False
        if dataset_git_revision is None:
            logging.debug(f"the dataset={dataset} has no git revision, don't update the cache")
            return False

        try:
            parquet_response_result = compute_parquet_response(
                dataset=dataset,
                hf_endpoint=self.config.common.hf_endpoint,
                hf_token=self.config.parquet.hf_token,
                source_revision=self.config.parquet.source_revision,
                target_revision=self.config.parquet.target_revision,
                commit_message=self.config.parquet.commit_message,
                url_template=self.config.parquet.url_template,
                supported_datasets=self.config.parquet.supported_datasets,
            )
            content = parquet_response_result["parquet_response"]
            if parquet_response_result["dataset_git_revision"] != dataset_git_revision:
                raise UnexpectedError("The dataset git revision has changed during the job")
            upsert_response(
                kind=CacheKind.PARQUET.value,
                dataset=dataset,
                content=dict(content),
                http_status=HTTPStatus.OK,
                worker_version=self.version,
                dataset_git_revision=dataset_git_revision,
            )
            logging.debug(f"dataset={dataset} is valid, cache updated")
            return True
        except DatasetNotFoundError:
            logging.debug(f"the dataset={dataset} could not be found, don't update the cache")
            return False
        except WorkerCustomError as err:
            upsert_response(
                kind=CacheKind.PARQUET.value,
                dataset=dataset,
                content=dict(err.as_response()),
                http_status=err.status_code,
                error_code=err.code,
                details=dict(err.as_response_with_cause()),
                worker_version=self.version,
                dataset_git_revision=dataset_git_revision,
            )
            logging.debug(f"parquet response for dataset={dataset} had an error, cache updated")
            return False
        except Exception as err:
            e = UnexpectedError(str(err), err)
            upsert_response(
                kind=CacheKind.PARQUET.value,
                dataset=dataset,
                content=dict(e.as_response()),
                http_status=e.status_code,
                error_code=e.code,
                details=dict(e.as_response_with_cause()),
                worker_version=self.version,
                dataset_git_revision=dataset_git_revision,
            )
            logging.debug(f"parquet response for dataset={dataset} had a server error, cache updated")
            return False
