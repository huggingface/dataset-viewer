# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import importlib.metadata
import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import (
    delete_first_rows_responses,
    get_dataset_first_rows_response_splits,
    get_splits_response,
    upsert_splits_response,
)
from libqueue.worker import Worker

from splits.config import WorkerConfig
from splits.response import compute_splits_response, get_dataset_git_revision
from splits.utils import (
    DatasetNotFoundError,
    Queues,
    UnexpectedError,
    WorkerCustomError,
)


class SplitsWorker(Worker):
    config: WorkerConfig

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(queue_config=worker_config.queue, version=importlib.metadata.version(__package__))
        self._queues = Queues(max_jobs_per_namespace=worker_config.queue.max_jobs_per_namespace)
        self.config = worker_config

    @property
    def queue(self):
        return self._queues.splits

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
        if force:
            return False
        try:
            cache_entry = get_splits_response(dataset)
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
        try:
            splits_response_result = compute_splits_response(
                dataset=dataset, hf_endpoint=self.config.common.hf_endpoint, hf_token=self.config.common.hf_token
            )
            response = splits_response_result["splits_response"]
            upsert_splits_response(
                dataset_name=dataset,
                response=dict(response),
                http_status=HTTPStatus.OK,
                worker_version=self.version,
                dataset_git_revision=splits_response_result["dataset_git_revision"],
            )
            logging.debug(f"dataset={dataset} is valid, cache updated")

            splits_in_cache = get_dataset_first_rows_response_splits(dataset_name=dataset)
            new_splits = [(s["dataset"], s["config"], s["split"]) for s in response["splits"]]
            splits_to_delete = [s for s in splits_in_cache if s not in new_splits]
            for d, c, s in splits_to_delete:
                delete_first_rows_responses(dataset_name=d, config_name=c, split_name=s)
            logging.debug(
                f"{len(splits_to_delete)} 'first-rows' responses deleted from the cache for obsolete splits of"
                f" dataset={dataset}"
            )
            for d, c, s in new_splits:
                self._queues.first_rows.add_job(dataset=d, config=c, split=s)
            logging.debug(f"{len(new_splits)} 'first-rows' jobs added for the splits of dataset={dataset}")
            return True
        except DatasetNotFoundError:
            logging.debug(f"the dataset={dataset} could not be found, don't update the cache")
            return False
        except WorkerCustomError as err:
            upsert_splits_response(
                dataset_name=dataset,
                response=dict(err.as_response()),
                http_status=err.status_code,
                error_code=err.code,
                details=dict(err.as_response_with_cause()),
            )
            logging.debug(f"splits response for dataset={dataset} had an error, cache updated")
            return False
        except Exception as err:
            e = UnexpectedError(str(err), err)
            upsert_splits_response(
                dataset_name=dataset,
                response=dict(e.as_response()),
                http_status=e.status_code,
                error_code=e.code,
                details=dict(e.as_response_with_cause()),
            )
            logging.debug(f"splits response for dataset={dataset} had a server error, cache updated")
            return False
