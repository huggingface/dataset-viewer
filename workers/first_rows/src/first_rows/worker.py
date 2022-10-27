# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import importlib.metadata
import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import get_first_rows_response, upsert_first_rows_response
from libqueue.worker import Worker

from first_rows.config import WorkerConfig
from first_rows.response import compute_first_rows_response, get_dataset_git_revision
from first_rows.utils import (
    ConfigNotFoundError,
    DatasetNotFoundError,
    Queues,
    SplitNotFoundError,
    UnexpectedError,
    WorkerCustomError,
)


class FirstRowsWorker(Worker):
    config: WorkerConfig

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(queue_config=worker_config.queue, version=importlib.metadata.version(__package__))
        self._queues = Queues(max_jobs_per_namespace=worker_config.queue.max_jobs_per_namespace)
        self.config = worker_config

    @property
    def queue(self):
        return self._queues.first_rows

    def should_skip_job(self, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> bool:
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

        Returns:
            :obj:`bool`: True if the job should be skipped, False otherwise.
        """
        if config is None or split is None:
            return False
        try:
            cache_entry = get_first_rows_response(dataset_name=dataset, config_name=config, split_name=split)
            dataset_git_revision = get_dataset_git_revision(
                dataset=dataset, hf_endpoint=self.config.common.hf_endpoint, hf_token=self.config.common.hf_token
            )
            return (
                cache_entry["http_status"] == HTTPStatus.OK
                and cache_entry["worker_version"] is not None
                and self.compare_major_version(cache_entry["worker_version"]) == 0
                and cache_entry["dataset_git_revision"] is not None
                and cache_entry["dataset_git_revision"] == dataset_git_revision
            )
        except Exception:
            return False

    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> bool:
        if config is None or split is None:
            raise ValueError("config and split are required")
        try:
            result = compute_first_rows_response(
                dataset=dataset,
                config=config,
                split=split,
                assets_base_url=self.config.common.assets_base_url,
                hf_endpoint=self.config.common.hf_endpoint,
                hf_token=self.config.common.hf_token,
                min_cell_bytes=self.config.first_rows.min_cell_bytes,
                max_size_fallback=self.config.first_rows.fallback_max_dataset_size,
                rows_max_bytes=self.config.first_rows.max_bytes,
                rows_max_number=self.config.first_rows.max_number,
                rows_min_number=self.config.first_rows.min_number,
                assets_directory=self.config.cache.assets_directory,
            )
            upsert_first_rows_response(
                dataset_name=dataset,
                config_name=config,
                split_name=split,
                response=dict(result["first_rows_response"]),
                http_status=HTTPStatus.OK,
                worker_version=self.version,
                dataset_git_revision=result["dataset_git_revision"],
            )
            logging.debug(f"dataset={dataset} config={config} split={split} is valid, cache updated")
            return True
        except (DatasetNotFoundError, ConfigNotFoundError, SplitNotFoundError):
            logging.debug(
                f"the dataset={dataset}, config {config} or split {split} could not be found, don't update the cache"
            )
            return False
        except WorkerCustomError as err:
            upsert_first_rows_response(
                dataset_name=dataset,
                config_name=config,
                split_name=split,
                response=dict(err.as_response()),
                http_status=err.status_code,
                error_code=err.code,
                details=dict(err.as_response_with_cause()),
            )
            logging.debug(
                f"first-rows response for dataset={dataset} config={config} split={split} had an error, cache updated"
            )
            return False
        except Exception as err:
            e = UnexpectedError(str(err), err)
            upsert_first_rows_response(
                dataset_name=dataset,
                config_name=config,
                split_name=split,
                response=dict(e.as_response()),
                http_status=e.status_code,
                error_code=e.code,
                details=dict(e.as_response_with_cause()),
            )
            logging.debug(
                f"first-rows response for dataset={dataset} config={config} split={split} had a server"
                " error, cache updated"
            )
            return False
