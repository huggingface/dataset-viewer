# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import (
    delete_first_rows_responses,
    get_dataset_first_rows_response_splits,
    upsert_splits_response,
)
from libqueue.worker import Worker

from splits.config import WorkerConfig
from splits.response import get_splits_response
from splits.utils import (
    DatasetNotFoundError,
    Queues,
    UnexpectedError,
    WorkerCustomError,
)


class SplitsWorker(Worker):
    config: WorkerConfig

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(queue_config=worker_config.queue)
        self._queues = Queues(max_jobs_per_dataset=worker_config.queue.max_jobs_per_dataset)
        self.config = worker_config

    @property
    def queue(self):
        return self._queues.splits

    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> bool:
        try:
            response = get_splits_response(
                dataset=dataset, hf_endpoint=self.config.common.hf_endpoint, hf_token=self.config.common.hf_token
            )
            upsert_splits_response(dataset_name=dataset, response=dict(response), http_status=HTTPStatus.OK)
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
