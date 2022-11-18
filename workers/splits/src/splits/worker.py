# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import importlib.metadata
import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import (
    delete_response,
    get_dataset_response_ids,
    get_response_without_content,
    upsert_response,
)
from libqueue.worker import Worker

from splits.config import WorkerConfig
from splits.response import compute_splits_response, get_dataset_git_revision
from splits.utils import (
    CacheKind,
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
            cached_response = get_response_without_content(kind=CacheKind.SPLITS.value, dataset=dataset)
            dataset_git_revision = get_dataset_git_revision(
                dataset=dataset, hf_endpoint=self.config.common.hf_endpoint, hf_token=self.config.common.hf_token
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
        try:
            splits_response_result = compute_splits_response(
                dataset=dataset, hf_endpoint=self.config.common.hf_endpoint, hf_token=self.config.common.hf_token
            )
            content = splits_response_result["splits_response"]
            upsert_response(
                kind=CacheKind.SPLITS.value,
                dataset=dataset,
                content=dict(content),
                http_status=HTTPStatus.OK,
                worker_version=self.version,
                dataset_git_revision=splits_response_result["dataset_git_revision"],
            )
            logging.debug(f"dataset={dataset} is valid, cache updated")

            new_splits = [(s["dataset"], s["config"], s["split"]) for s in content["splits"]]
            # remove obsolete first-rows responses from the cache
            first_rows_responses_in_cache = [
                (s["dataset"], s["config"], s["split"])
                for s in get_dataset_response_ids(dataset=dataset)
                if s["kind"] == CacheKind.FIRST_ROWS.value
            ]
            first_rows_responses_to_delete = [s for s in first_rows_responses_in_cache if s not in new_splits]
            for d, c, s in first_rows_responses_to_delete:
                delete_response(kind=CacheKind.FIRST_ROWS.value, dataset=d, config=c, split=s)
            logging.debug(
                f"{len(first_rows_responses_to_delete)} 'first-rows' responses deleted from the cache for obsolete"
                f" splits of dataset={dataset}"
            )
            # compute the 'first-rows' responses for the new splits
            for d, c, s in new_splits:
                # we force the refresh of the /first_rows responses if the /splits refresh was forced
                self._queues.first_rows.add_job(dataset=d, config=c, split=s, force=force)
            logging.debug(f"{len(new_splits)} 'first-rows' jobs added for the splits of dataset={dataset}")
            return True
        except DatasetNotFoundError:
            logging.debug(f"the dataset={dataset} could not be found, don't update the cache")
            return False
        except WorkerCustomError as err:
            upsert_response(
                kind=CacheKind.SPLITS.value,
                dataset=dataset,
                content=dict(err.as_response()),
                http_status=err.status_code,
                error_code=err.code,
                details=dict(err.as_response_with_cause()),
            )
            logging.debug(f"splits response for dataset={dataset} had an error, cache updated")
            return False
        except Exception as err:
            e = UnexpectedError(str(err), err)
            upsert_response(
                kind=CacheKind.SPLITS.value,
                dataset=dataset,
                content=dict(e.as_response()),
                http_status=e.status_code,
                error_code=e.code,
                details=dict(e.as_response_with_cause()),
            )
            logging.debug(f"splits response for dataset={dataset} had a server error, cache updated")
            return False
